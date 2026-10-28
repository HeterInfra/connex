# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Connex PP on upstream vLLM (connex.adapters.vllm.pp): the logic that does
not need GPUs or a running engine. End-to-end runs are in
benchmarks/pp_upstream (see docs/results.md, section 13)."""

import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from connex.adapters.vllm.pp.link import _peer_id, _Ring  # noqa: E402


def test_ring_first_fit_and_out_of_order_free():
    r = _Ring(4096)
    a = r.alloc(1000)          # rounded up to 1024
    b = r.alloc(1000)
    c = r.alloc(1000)
    assert (a, b, c) == (0, 1024, 2048)
    r.free(b)                  # a hole in the middle is not reused before wrap
    d = r.alloc(1000)
    assert d == 3072
    r.free(a)
    assert r.alloc(1024) == 0  # wraps to the freed head
    with pytest.raises(TimeoutError):
        r.alloc(2048, timeout=0.05)


def test_ring_blocks_until_free():
    r = _Ring(1024)
    off = r.alloc(1024)
    threading.Timer(0.05, r.free, args=(off,)).start()
    t0 = time.monotonic()
    assert r.alloc(512, timeout=2) == 0
    assert time.monotonic() - t0 >= 0.04


def test_ring_rejects_oversized_message():
    with pytest.raises(ValueError, match="CONNEX_PP_RING_MB"):
        _Ring(1024).alloc(4096)


def test_peer_id_from_url():
    assert _peer_id("10.0.0.2:47021#cxpp|1|0|3") == ("cxpp", 1, 0, 3)
    assert _peer_id("bad") == ()


def _push_pair():
    """Two connex-rs engines in this process (sender, receiver), or skip."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (stream wait/write on the flags)")
    connex_rs = pytest.importorskip("connex_rs")
    if not hasattr(connex_rs.Engine, "put_chain"):
        pytest.skip("connex-rs without put_chain")
    from connex.adapters.vllm.pp.link import make_engine

    try:
        return make_engine("cxtest", 0, 0, 0), make_engine("cxtest", 1, 0, 0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no connex-rs engine here: {e}")


def test_push_channel_delivers_in_order_and_reuses_slots():
    from connex.adapters.vllm.pp.rdmachan import PushRx, PushTx

    a, b = _push_pair()
    dev = torch.device("cuda", torch.cuda.current_device())
    rx = PushRx(b, b.local_url, dev, num_slots=2, slot_bytes=1 << 16)
    tx = PushTx(a, dev, rx.info)
    src = torch.empty(4 << 16, dtype=torch.uint8, device=dev)
    assert a.register_memory(src.data_ptr(), src.numel()) == 0
    got = []
    for v in range(6):   # 3x around a 2-slot ring: the gate must hold the sender
        off = (v % 4) << 16
        src[off:off + 4096].fill_(v + 1)
        tx.push(v, src.data_ptr() + off, 4096, gate=v - rx.n + 1 if v >= rx.n else None)
        buf = rx.wait(v)
        got.append(buf[:4096].clone())
        rx.done(v)
    torch.cuda.synchronize()
    assert [int(t.float().mean().item()) for t in got] == [1, 2, 3, 4, 5, 6]


def test_push_channel_host_ring_and_cpu_release():
    from connex.adapters.vllm.pp.rdmachan import PushRx, PushTx

    a, b = _push_pair()
    dev = torch.device("cuda", torch.cuda.current_device())
    rx = PushRx(b, b.local_url, dev, num_slots=8, slot_bytes=4096, host=True)
    tx = PushTx(a, dev, rx.info, staging=True, host_staging=True)
    tokens = torch.arange(16, dtype=torch.int64, device=dev)
    tx.push_staged(40, [tokens.view(torch.uint8)])
    out = torch.zeros_like(tokens)
    out.view(torch.uint8).copy_(rx.wait(40)[:128], non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(out, tokens)
    # a payload that came the slow way, delivered from the CPU
    payload = (tokens + 1).cpu().numpy().tobytes()   # before the wait (a copy to the
    side = torch.cuda.Stream(dev)                       # host would wait for it)
    out2 = torch.zeros_like(tokens)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        out2.view(torch.uint8).copy_(rx.wait(41)[:128], non_blocking=True)
    rx.fill(41, payload)
    side.synchronize()
    assert torch.equal(out2, tokens + 1)
    # a sender that is gone: release lets the stream go
    with torch.cuda.stream(side):
        rx.wait(1000)
    rx.release_all()
    side.synchronize()


vllm = pytest.importorskip("vllm")


def test_tp_endpoints_route_entries():
    from connex.adapters.vllm.pp.group import ConnexPPGroup

    g = SimpleNamespace(external=True, tp_size=2, tp_rank=1, rank_in_group=1,
                        _route=["h0:1#m|0|0|0;h0:2#m|0|1|0", "h1:1#m|1|0|0;h1:2#m|1|1|0"])
    per_rank = ConnexPPGroup._per_rank.__get__(g)
    rank_url = ConnexPPGroup._rank_url.__get__(g)
    assert per_rank(g._route[0])
    assert rank_url(g._route[0], True) == "h0:2#m|0|1|0"
    assert rank_url(g._route[0]) == "h0:1#m|0|0|0"      # sampled tokens: TP rank 0
    g._route = ["h0:1#m|0|0|0", g._route[1]]            # peer with one endpoint
    assert not per_rank(g._route[0])
    g._route = [g._route[0], "h1:1#m|1|0|0"]            # we have one endpoint
    assert not per_rank("h0:1#m|0|0|0;h0:2#m|0|1|0")


def _executor_stub(replicas, route, replica_id=0):
    from connex.adapters.vllm.pp.executor import ConnexPPExecutor

    ex = SimpleNamespace(replicas=replicas, route=route, replica_id=replica_id)
    ex._choose_route = ConnexPPExecutor._choose_route.__get__(ex)
    return ex


def test_route_keeps_live_members_and_prefers_own_replica():
    from connex.adapters.vllm.pp.executor import Replica

    s1 = [Replica(1, 0, "a", "d0", alive=True), Replica(1, 1, "b", "d1", alive=True)]
    s2 = [Replica(2, 0, "c", "d2", alive=True), Replica(2, 1, "e", "d3", alive=True)]
    ex = _executor_stub([s1, s2], [None, None], replica_id=1)
    assert ex._choose_route()
    assert [r.idx for r in ex.route] == [1, 1]
    # a dead member is replaced, a live one is kept (sticky)
    s1[1].alive = False
    assert ex._choose_route()
    assert [r.idx for r in ex.route] == [0, 1]
    s1[1].alive = True
    assert ex._choose_route()
    assert [r.idx for r in ex.route] == [0, 1]


def test_route_skips_busy_and_reports_missing_stage():
    from connex.adapters.vllm.pp.executor import Replica

    s1 = [Replica(1, 0, "a", "d0", alive=True, busy=True), Replica(1, 1, "b", "d1")]
    ex = _executor_stub([s1], [None])
    assert not ex._choose_route()
    assert ex.route == [None]
    s1[1].alive = True
    assert ex._choose_route()
    assert ex.route[0] is s1[1]


def test_lost_step_reports_lost_kv_once():
    """After a failover the first lost output tells the scheduler that the
    replaced stage's KV is gone; later lost outputs only carry no tokens."""
    from connex.adapters.vllm.pp.executor import ConnexPPExecutor, Pending

    sent = []
    ex = SimpleNamespace(lock=threading.RLock(), lost_pending=True, epoch=3,
                         local_data_url="u0", route=[SimpleNamespace(data_url="u1")],
                         ctl=SimpleNamespace(send=lambda *a, **k: sent.append(a)))
    ex._route_urls = ConnexPPExecutor._route_urls.__get__(ex)
    ex._attach_lost = ConnexPPExecutor._attach_lost.__get__(ex)
    lost = ConnexPPExecutor._lost_output.__get__(ex)
    out1 = lost(Pending(1, 2, ["a", "b"]))
    assert out1.kv_connector_output.lost_all_kv
    assert out1.sampled_token_ids == [[], []]
    assert ex.epoch == 4 and len(sent) == 2   # abort of the old epoch, both endpoints
    out2 = lost(Pending(2, 2, ["a"]))
    assert out2.kv_connector_output is None
