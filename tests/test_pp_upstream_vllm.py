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


vllm = pytest.importorskip("vllm")


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
