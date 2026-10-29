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


def test_push_channel_small_gpu_message_and_counter_wrap():
    """#1: a 2 KB push from GPU memory crashed in UCX (put_short); it is
    padded to the zero-copy threshold now. #9: counters wrap past 2^32 (and
    pass 0x7FFFFFFF) on a 2-slot ring with the gate holding the sender."""
    from connex.adapters.vllm.pp.rdmachan import ZCOPY_MIN, PushRx, PushTx

    a, b = _push_pair()
    dev = torch.device("cuda", torch.cuda.current_device())
    rx = PushRx(b, b.local_url, dev, num_slots=2, slot_bytes=2048)
    assert rx.slot_bytes == ZCOPY_MIN
    tx = PushTx(a, dev, rx.info)
    src = torch.empty(4 * ZCOPY_MIN, dtype=torch.uint8, device=dev)
    assert a.register_memory(src.data_ptr(), src.numel()) == 0
    got = []
    for base in (0x7FFFFFFC, 0xFFFFFFFC):
        # the state after messages base-2 and base-1 (the streams are idle)
        torch.cuda.synchronize()
        rx.arr[:rx.n] = [(base - 2 + 1) & 0xFFFFFFFF, (base - 1 + 1) & 0xFFFFFFFF]
        rx.arr[rx.n:] = rx.arr[:rx.n]
        tx.bell.numpy().view("uint32")[0] = base & 0xFFFFFFFF
        rx.last = base - 1
        for v in range(base, base + 6):
            vv = v   # unbounded here; flags carry (v + 1) mod 2^32
            off = (v % 4) * ZCOPY_MIN
            src[off:off + 2048].fill_(v % 251)
            tx.push(vv, src.data_ptr() + off, 2048, gate=vv - rx.n + 1)
            buf = rx.wait(vv)
            got.append((v % 251, buf[:2048].clone()))
            rx.done(vv)
    torch.cuda.synchronize()
    assert all(int(t.float().mean().item()) == want for want, t in got)


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
                         ctl=SimpleNamespace(engine=SimpleNamespace(
                             send_msg=lambda u, raw: sent.append((u, pickle.loads(raw))))))
    ex._route_urls = ConnexPPExecutor._route_urls.__get__(ex)
    ex._attach_lost = ConnexPPExecutor._attach_lost.__get__(ex)
    ex._send_abort = ConnexPPExecutor._send_abort.__get__(ex)
    lost = ConnexPPExecutor._lost_output.__get__(ex)
    out1 = lost(Pending(1, 2, ["a", "b"]))
    assert out1.kv_connector_output.lost_all_kv
    assert out1.sampled_token_ids == [[], []]
    t0 = time.monotonic()
    while len(sent) < 2 and time.monotonic() - t0 < 2:   # sent off-thread
        time.sleep(0.01)
    # abort of the old epoch, both endpoints
    assert ex.epoch == 4 and sorted(sent) == [("u0", (3, 4)), ("u1", (3, 4))]
    out2 = lost(Pending(2, 2, ["a"]))
    assert out2.kv_connector_output is None


# --------------------------------------------------------------------------
# Regression tests of the review fixes (CPU; the GPU ones skip without one).
# --------------------------------------------------------------------------

import pickle  # noqa: E402
import queue  # noqa: E402
from collections import deque  # noqa: E402

from connex.adapters.vllm.pp import link as L  # noqa: E402


class _Engine:
    def __init__(self):
        self.sent = []

    def send_msg(self, peer, data):
        self.sent.append((peer, pickle.loads(data)))


def _link():
    """A StageLink without connex-rs, GPU or threads."""
    lk = object.__new__(L.StageLink)
    lk.engine = _Engine()
    lk.url = "10.0.0.1:1#cxpp|1|0|0"
    lk.inbox = L._Inbox()
    lk.push_fallback = {}
    lk.min_epoch = 0
    lk.stats = {"sent": 0, "recv": 0, "bytes_sent": 0, "bytes_recv": 0, "read_s": 0.0,
                "chain_err": 0, "fallback_fill": 0}
    lk._push_rx, lk._rx_chans, lk._tx_chans = {}, {}, {}
    lk._chan_lock = threading.RLock()
    lk._retired_cids = deque(maxlen=256)
    lk._graveyard = None
    lk._announced = {}
    lk._tx = queue.Queue()
    lk._reap = queue.Queue()
    lk._closed = False
    lk._fatal = False
    lk.fatals = []
    lk.on_fatal = lk.fatals.append
    lk.abort_callbacks = []
    lk._post_lock, lk._out_lock = threading.Lock(), threading.Lock()
    lk._posted, lk._ready, lk._outstanding = {}, {}, {}
    lk._waiter = None
    lk._prev_off = None
    lk._rx_done = None
    lk.send_ring = L._Ring(1 << 20)
    lk.ipc_slot_bytes = 0
    return lk


class _Slot:
    """A push channel's receive side, recording CPU fills."""

    def __init__(self):
        self.filled = []
        self.retired = False

    def fill(self, v, payload):
        self.filled.append((v, payload))


def test_sampled_bytes_before_or_after_the_receive_registers_are_filled():
    """#4: BYTES that arrive before the receive registered its fallback used
    to land in the inbox and the slot was never filled (the stream waited
    forever)."""
    lk = _link()
    key = (0, 20, 0)
    lk._handle(("cxpp", 2, 0, 0), pickle.dumps((L.BYTES, key, "u", b"tok")))   # early
    ch = _Slot()
    lk.expect_push(key, ch, 80)
    assert ch.filled == [(80, b"tok")] and key not in lk.inbox.items
    key2 = (0, 21, 0)
    lk.expect_push(key2, ch, 84)                                                # late
    lk._handle(("cxpp", 2, 0, 0), pickle.dumps((L.BYTES, key2, "u", b"tok2")))
    assert ch.filled[-1] == (84, b"tok2") and not lk.inbox.items and not lk.push_fallback


def test_racing_bytes_and_receive_always_fill(monkeypatch):
    """#4 under contention: whichever side comes first, the slot is filled
    exactly once and nothing stays in the inbox."""
    lk = _link()
    ch = _Slot()
    keys = [(0, i, 0) for i in range(300)]

    def rx():
        for k in keys:
            lk._handle(("p",), pickle.dumps((L.BYTES, k, "u", b"x")))

    t = threading.Thread(target=rx)
    t.start()
    for i, k in enumerate(keys):
        lk.expect_push(k, ch, i)
    t.join()
    assert sorted(v for v, _ in ch.filled) == list(range(300))
    assert not lk.inbox.items and not lk.push_fallback


class _T:
    """A connex-rs transfer stand-in."""

    def __init__(self, done=True, err=None):
        self._done, self.err = done, err

    def done(self):
        return self._done

    def wait(self, timeout_ms=None):
        if self.err is not None:
            raise self.err
        return True


class _Tx:
    def __init__(self, cid=7):
        self.cid, self.dead, self.first_done, self.created = cid, False, None, time.monotonic()
        self.n = 64

    def staged_bytes(self, v, n):
        return b"S" * n


def test_failed_hidden_chain_tells_the_receiver_which_stops():
    """#2: a chain error that is not a cancel (post/flush/rkey failure,
    timeout) marks the channel dead and sends CHANFAIL; the receiver of a
    live epoch stops (fail-over) instead of waiting forever or using
    garbage; for an aborted epoch nothing happens."""
    tx_link, rx_link = _link(), _link()
    tx = _Tx()
    r = L._Reap(_T(err=RuntimeError("UCX error -3: no connection")), None, tx, "peer-url",
                (2, 10, -1), "h", 5)
    tx_link._chain_failed(r, RuntimeError("UCX error -3"))
    assert tx.dead and tx_link.stats["chain_err"] == 1
    (peer, msg), = tx_link.engine.sent
    assert peer == "peer-url" and msg[0] == L.CHANFAIL and msg[1:4] == (7, 5, (2, 10, -1))
    rx_link._handle(("cxpp", 0, 0, 0), pickle.dumps(msg))
    assert rx_link.fatals and "failed" in rx_link.fatals[0]
    rx_link2 = _link()
    rx_link2.min_epoch = 3                   # that epoch was aborted meanwhile
    rx_link2._handle(("cxpp", 0, 0, 0), pickle.dumps(msg))
    assert not rx_link2.fatals


def test_failed_sampled_chain_is_resent_the_slow_way():
    lk = _link()
    tx = _Tx()
    lk._chain_failed(L._Reap(_T(), None, tx, "u", (1, 4, 0), "s", 16, 8),
                     TimeoutError("timeout"))
    ev, url, hdr = lk._tx.get_nowait()
    assert tx.dead and url == "u" and hdr[0] == L.BYTES and hdr[1] == (1, 4, 0)
    assert hdr[3] == b"S" * 8


def test_reaper_frees_in_completion_order_not_fifo():
    """#3: one stuck chain must not hold every later ring free."""
    lk = _link()
    stuck, ok = _T(done=False), _T()
    o1, o2 = lk.send_ring.alloc(4096), lk.send_ring.alloc(4096)
    th = threading.Thread(target=lk._reap_loop, daemon=True)
    th.start()
    lk._reap.put(L._Reap(stuck, o1, _Tx(), "u", (0, 1, -1), "h", 0))
    lk._reap.put(L._Reap(ok, o2, _Tx(), "u", (0, 2, -1), "h", 1))
    t0 = time.monotonic()
    while o2 in lk.send_ring.live and time.monotonic() - t0 < 2:
        time.sleep(0.005)
    assert o2 not in lk.send_ring.live and o1 in lk.send_ring.live
    stuck._done = True
    while o1 in lk.send_ring.live and time.monotonic() - t0 < 2:
        time.sleep(0.005)
    assert o1 not in lk.send_ring.live
    lk._closed = True
    lk._reap.put(None)


def test_cancelled_chain_is_not_a_failure():
    lk = _link()
    th = threading.Thread(target=lk._reap_loop, daemon=True)
    th.start()
    tx = _Tx()
    lk._reap.put(L._Reap(_T(err=RuntimeError("canceled")), None, tx, "u", (0, 1, -1), "h", 0))
    time.sleep(0.05)
    assert not tx.dead and not lk.engine.sent
    lk._closed = True
    lk._reap.put(None)


class _Chan:
    def __init__(self, cid=None, key=None):
        self.cid, self.key, self.retired, self.retire_calls = cid, key, False, 0
        self.from_key = None

    def retire(self):
        self.retired = True
        self.retire_calls += 1
        return []

    def close(self):
        pass


def test_tx_channels_to_peers_off_the_route_are_retired():
    """#3: channels to peers that left the route (and are no spares) are
    retired: their chains are cancelled, their ring space freed."""
    lk = _link()
    a, b = ("m", 1, 0, 0), ("m", 1, 0, 1)
    ca, cb = _Chan(), _Chan()
    lk._tx_chans = {(a, ("p", "h")): ca, (b, ("p", "h")): cb}
    lk.retain_tx({a})
    assert cb.retired and not ca.retired and list(lk._tx_chans) == [(a, ("p", "h"))]


def test_push_on_unknown_channel_raises_for_a_live_epoch():
    """#10: used to return None (the caller then ran on zeros)."""
    lk = _link()
    lk.device = torch.device("cpu")
    hdr = L._Header([], 16, None, ("push", 99, 0), ("cxpp", 0, 0, 0))
    lk.inbox.items[(1, 5, -1)] = hdr
    import connex.adapters.vllm.pp.link as link_mod

    orig = link_mod.torch.cuda.current_stream
    link_mod.torch.cuda.current_stream = lambda dev=None: None
    try:
        with pytest.raises(RuntimeError, match="unknown channel"):
            lk.recv_tensors((1, 5, -1))
        lk.inbox.items[(1, 6, -1)] = hdr
        lk.min_epoch = 2                      # aborted meanwhile: None, no error
        lk.inbox.items[(1, 6, -1)] = hdr
        assert lk.recv_tensors((1, 6, -1)) is None
    finally:
        link_mod.torch.cuda.current_stream = orig


def test_failed_read_stops_instead_of_releasing_the_stream():
    """#10: a read with rc != 0 used to fill the slot (stream ran on
    garbage); now it is fatal for a live epoch, a release for an aborted one."""
    lk = _link()
    lk.recv_buf = SimpleNamespace(data_ptr=lambda: 0)
    filled = []
    lk._waiter = SimpleNamespace(fill=lambda slot, p: filled.append(slot))
    lk.engine.transfer_sync_read = lambda *a: -5
    lk.engine.send_msg = lambda *a: None
    hdr = (L.READY, (1, 3, -1), "src", 0, 9, 100)
    lk._read_into(("p",), hdr, (0, 4))
    assert lk.fatals and not filled
    lk2 = _link()
    lk2.recv_buf, lk2._waiter = lk.recv_buf, lk._waiter
    lk2.engine.transfer_sync_read = lambda *a: -5
    lk2.engine.send_msg = lambda *a: None
    lk2.min_epoch = 2
    lk2._read_into(("p",), hdr, (0, 4))
    assert not lk2.fatals and filled == [4]


def test_small_messages_read_at_least_the_zero_copy_threshold():
    """#1, read path: a GPU RMA below UCX_ZCOPY_THRESH segfaults in
    uct_rc_mlx5_ep_put_short / bcopy; reads are padded to ZCOPY_MIN."""
    lk = _link()
    lk.recv_buf = SimpleNamespace(data_ptr=lambda: 0)
    lk._waiter = SimpleNamespace(fill=lambda slot, p: None)
    lens = []
    lk.engine.transfer_sync_read = lambda url, dst, src, n, t: lens.append(n) or 0
    lk.engine.send_msg = lambda *a: None
    lk._read_into(("p",), (L.READY, (0, 1, -1), "s", 0, 1, 2048), (0, 0))
    assert lens == [L.ZCOPY_MIN] and L.ZCOPY_MIN >= 4096


def test_push_slots_are_at_least_the_zero_copy_threshold():
    from connex.adapters.vllm.pp.rdmachan import ZCOPY_MIN, PushTx, slot_size

    assert slot_size(16) == ZCOPY_MIN == 4096 and slot_size(5000) == 5120
    with pytest.raises(ValueError, match="slots of 2048"):
        PushTx(None, None, (1, "u", 0, 2048, 4, 0))


def test_cyclic_flag_order_and_release_values():
    """#9: values are counts mod 2^32 compared cyclically (as the GPU's
    CU_STREAM_WAIT_VALUE_GEQ); 0x7FFFFFFF is an ordinary value and a count
    may wrap; a release is far ahead of every awaited count."""
    from connex.adapters.vllm.pp.ipcchan import AHEAD, MASK, reached

    assert reached(5, 5) and reached(6, 5) and not reached(4, 5)
    assert reached(0x80000000, 0x7FFFFFFF) and reached(0, 0xFFFFFFFF)   # wraps
    assert not reached(0xFFFFFFFF, 0)
    hi = 0xFFFFFF00
    rel = (hi + AHEAD) & MASK
    assert all(reached(rel, w & MASK) for w in range(hi - 100, hi + 1))


def test_push_rx_fill_and_release_below_use_cyclic_values():
    """#9 / aborted epochs: a CPU fill never lowers a flag; waits of older
    epochs are released, newer ones not."""
    import numpy as np

    from connex.adapters.vllm.pp.rdmachan import PushRx

    rx = object.__new__(PushRx)
    rx.n, rx.slot_bytes = 4, 4096
    rx.arr = np.zeros(8, dtype=np.uint32)
    rx.ring = torch.zeros(4 * 4096, dtype=torch.uint8)
    rx.lock = threading.Lock()
    rx.pending = {0xFFFFFFFE: 1, 7: 2}
    rx.arr[2] = 0xFFFFFFFB    # the slot's previous message, 0xFFFFFFFA
    rx.arr[3] = 4             # message 3 of slot 3
    rx.release_below(2)
    assert rx.arr[0xFFFFFFFE % 4] == 0xFFFFFFFF and rx.arr[7 % 4] == 4
    rx.arr[1] = 10
    rx.fill(1, None)          # 2 < 10: already past, stays
    assert rx.arr[1] == 10
    rx.fill(9, b"ab")
    assert rx.arr[1] == 10 and bytes(rx.ring[4096:4098].tolist()) == b"ab"


def test_graveyard_reasserts_a_release_until_streams_pass():
    """#7: a flag write the peer queued before the release (a stream write
    or a NIC put) can lower the flag again; the retired channel re-asserts
    its release on every drain until it is closed."""
    from connex.adapters.vllm.pp.ipcchan import Graveyard

    class _Ev:
        def __init__(self):
            self.ok = False

        def query(self):
            return self.ok

    ev = _Ev()

    class _C:
        n = 0
        closed = False

        def retire(self):
            return [ev]

        def reassert(self):
            self.n += 1

        def close(self):
            self.closed = True

    g, c = Graveyard(), _C()
    g.add(c)
    g.drain()
    g.drain()
    assert c.n == 2 and not c.closed
    ev.ok = True
    g.drain()
    assert c.closed


def test_push_tx_cancel_words_per_epoch_not_the_doorbell():
    """#7: cancelling writes a CPU-only word per epoch; a stream write still
    queued for the doorbell cannot undo it."""
    import numpy as np

    from connex.adapters.vllm.pp.rdmachan import PushTx

    tx = object.__new__(PushTx)
    tx.cancel_np = np.zeros(64, dtype=np.int32)
    tx.cancel = SimpleNamespace(data_ptr=lambda: 1000)
    tx.word_epoch = [-1] * 64
    tx.dead = False
    assert tx._word(3) == 1000 + 12 and tx._word(5) == 1000 + 20
    tx.cancel_below(5)
    assert tx.cancel_np[3] == 1 and tx.cancel_np[5] == 0
    assert tx._word(3 + 64) == 1000 + 12 and tx.cancel_np[3] == 0   # reused for a new epoch
    tx.streams = {}
    tx.retire()
    assert tx.dead and (tx.cancel_np == 1).all()


def test_staged_slot_waits_for_the_chain_still_reading_it():
    """#8: the host staging slot of message v is reused by v+n; its copy
    must not start while the chain of v still reads it."""
    from connex.adapters.vllm.pp.rdmachan import PushTx

    tx = object.__new__(PushTx)
    tx.n, tx.slot_bytes, tx.timeout_ms, tx.host_staging = 2, 4096, 1000, True
    tx.staging = torch.zeros(2 * 4096, dtype=torch.uint8)
    waited = []

    class _Prev:
        def done(self):
            return False

        def wait(self, t):
            waited.append(t)

    tx.slot_t = [_Prev(), None]
    pushed = []
    tx.push = lambda v, ptr, n, gate=None, epoch=0: pushed.append((v, n, gate, epoch)) or "t"
    tx.push_staged(2, [torch.ones(8, dtype=torch.uint8)], gate=1, epoch=4)
    assert waited == [1000] and pushed == [(2, 8, 1, 4)] and tx.slot_t[0] == "t"


def test_sampled_push_is_gated_on_the_slot_being_free():
    """#8: sampled-token pushes now carry a gate (the receiver freed the
    slot's previous message), like hidden states."""
    lk = _link()
    calls = []

    class _STx(_Tx):
        def push_staged(self, v, chunks, gate=None, epoch=0):
            calls.append((v, gate, epoch))
            return _T()

    tx = _STx()
    assert lk.push_sampled(tx, "u", (3, 40, 0), [torch.ones(4, dtype=torch.uint8)])
    assert calls == [(160, 160 - 64 + 1, 3)]
    assert lk.push_sampled(tx, "u", (3, 2, 1), [torch.ones(4, dtype=torch.uint8)])
    assert calls[-1] == (9, None, 3)


def test_retired_push_channels_and_announcements_are_pruned():
    """#11: _push_rx and _announced no longer grow without bound."""
    lk = _link()
    peer = ("m", 1, 0, 0)
    old = _Chan(cid=1, key=(peer, ("p", "h")))
    cur = _Chan(cid=2, key=(peer, ("p", "h")))
    lk._push_rx = {1: old, 2: cur}
    lk._rx_chans = {(peer, ("p", "h")): cur}
    lk._announced = {(peer, "n1"): 1.0, (("m", 1, 0, 1), "n2"): 1.0}
    lk.close_chans(lambda k: k[0] == peer)
    assert not lk._push_rx and not lk._rx_chans and old.retired and cur.retired
    assert 1 in lk._retired_cids and 2 in lk._retired_cids


def test_inbox_items_nobody_took_are_pruned():
    lk = _link()
    lk.inbox.items = {(2, 5, 0): b"a", (2, 9000, 0): b"b", (2, -1, -2): b"c"}
    lk.push_fallback = {(2, 6, 0): 1}
    assert lk.prune_inbox(2, 100) == 1
    assert set(lk.inbox.items) == {(2, 9000, 0), (2, -1, -2)} and not lk.push_fallback


def test_abort_releases_channel_waits_and_cancels_old_pushes():
    """#2/#3: a hung (but alive) sender: on the epoch change the streams
    waiting for its pushes of older epochs go, and our own pushes of older
    epochs stop holding their queue."""
    lk = _link()
    released, cancelled = [], []
    rx = SimpleNamespace(release_below=lambda e: released.append(e) or 1)
    tx = SimpleNamespace(cancel_below=lambda e: cancelled.append(e))
    lk._push_rx = {1: rx}
    lk._tx_chans = {("p", ("p", "h")): tx}
    lk.abort(4)
    assert released == [4] and cancelled == [4] and lk.min_epoch == 4


def test_ports_never_collide():
    """#5: data port = 47021 + 100*stage + 2*replica + 100*tp_rank made stage k
    TP rank 1 equal stage k+1 TP rank 0."""
    from connex.adapters.vllm.pp import ports

    seen = {}
    for st in range(1, 8):
        for rep in range(8):
            c = ports.control_port(st, rep)
            assert c not in seen, (st, rep, seen.get(c))
            seen[c] = (st, rep, "ctl")
            for tp in range(8):
                p = ports.data_port(st, rep, tp)
                assert p not in seen, ((st, rep, tp), seen.get(p))
                seen[p] = (st, rep, tp)
                assert ports.rank_port(ports.data_port(st, rep), tp) == p
    with pytest.raises(ValueError):
        ports.data_port(1, 0, 15)
    with pytest.raises(ValueError):
        ports.control_port(1, 62)
    with pytest.raises(ValueError):
        ports.control_port(19, 0)


def test_watchdog_fails_over_a_stuck_stage_that_answers_pings():
    """#2: the watchdog used to fail over only a stage without a pong; a
    stage with a hung GPU still answers PING."""
    from connex.adapters.vllm.pp.executor import Pending, Replica, pick_stuck

    a = Replica(1, 0, "a", "d", alive=True, pong=10.0, done_seq=41)
    b = Replica(2, 0, "b", "d", alive=True, pong=10.0, done_seq=39)
    late = [Pending(40, 1, []), Pending(42, 1, [])]
    v, why = pick_stuck([a, b], late, t_ping=9.0)
    assert v is b and "answers" in why
    b.pong = 5.0                       # no pong: that one first
    assert pick_stuck([a, b], late, 9.0)[0] is b
    a.pong, b.pong, b.done_seq = 5.0, 10.0, 50
    assert pick_stuck([a, b], late, 9.0) == (a, "step timeout, no pong")
    a.pong, a.done_seq = 10.0, 50      # everybody past it: the last stage
    assert pick_stuck([a, b], late, 9.0)[0] is b
    assert pick_stuck([a, b], [], 9.0) == (None, "")


def test_no_token_window_spans_all_streams():
    """Failover metric: the largest gap across all streams after the kill
    (the per-request stall missed requests that started after it)."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pp_upstream",
                        "failover_load.py")
    spec = importlib.util.spec_from_file_location("failover_load", path)
    fl = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(fl)
    except ImportError:
        pytest.skip("aiohttp missing")
    out = [{"times": [9.5, 9.9, 10.05]}, {"times": [12.6, 12.7]}]
    assert fl.no_token_window(out, t_kill=10.0) == 2.55


def test_same_host_sampled_channel_timeout_is_a_failure():
    """#6: when the channel message did not come in time the step read
    garbage and ch.i fell one behind; now it raises (unless aborted)."""
    from connex.adapters.vllm.pp.group import ConnexPPGroup

    lk = SimpleNamespace(_rx_chans={}, min_epoch=1, take=lambda key, timeout=None: None)
    g = SimpleNamespace(link=lk)
    pop = ConnexPPGroup._pop_sampled.__get__(g)
    with pytest.raises(RuntimeError, match="no sampled-token channel"):
        pop("10.0.0.1:2#cxpp|1|0|0", 1, torch.zeros(4))
    lk.min_epoch = 2                       # epoch aborted: just return
    pop("10.0.0.1:2#cxpp|1|0|0", 1, torch.zeros(4))
