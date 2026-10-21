# SPDX-License-Identifier: Apache-2.0
"""Churn fixes for CONNEX_PP_INFLIGHT=K>1 (see docs/design/pp_inflight.md):
unique DEALER identities, no 0 ms recv spin, UNHEALTHY = suspect only,
return sockets of surviving stage-0 replicas kept on deregistration."""

import threading
import time

import pytest
import zmq

from test_connexte_protocol import FakeRouter, cx
from test_pipelining import _k, _reply


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)


class _BlockingRouter(FakeRouter):
    """poll() waits like a real socket when nothing is queued."""

    def poll(self, timeout, flags=None):
        if not self.msgs:
            time.sleep(timeout / 1000.0)
        return 1 if self.msgs else 0


class _Ev:
    def __init__(self, stage, replica):
        self.payload = {"lid": {"stage_idx": stage, "replica_idx": replica}}


def _routing(c):
    c._sticky_peer = {}
    c._deregistered_replicas = set()
    c._suspect_replicas = set()
    c._peer_lock = threading.Lock()
    c._peer_dealer_cache = {}
    c._last_failed_peer = {}
    c.forward_socket = c.return_socket = None
    return c


# ---- ZMQ identity collision (the lost re-sent step) ---------------------------

def _router_receives_after_recreate(identity_fn):
    ctx = zmq.Context()
    r = ctx.socket(zmq.ROUTER)
    port = r.bind_to_random_port("tcp://127.0.0.1")
    try:
        d = ctx.socket(zmq.DEALER)
        d.setsockopt(zmq.LINGER, 1000)
        d.setsockopt(zmq.IDENTITY, identity_fn())
        d.connect(f"tcp://127.0.0.1:{port}")
        d.send(b"a")
        assert r.poll(2000) and r.recv_multipart()[1] == b"a"
        for _ in range(50):                 # in-flight backlog, like K>1 sends
            d.send(b"x" * 1_000_000)
        d.close()                           # still draining (LINGER)
        d2 = ctx.socket(zmq.DEALER)
        d2.setsockopt(zmq.LINGER, 0)
        d2.setsockopt(zmq.IDENTITY, identity_fn())
        d2.connect(f"tcp://127.0.0.1:{port}")
        while r.poll(300):
            r.recv_multipart()
        d2.send(b"retry")
        got = r.poll(1000) and r.recv_multipart()[1] == b"retry"
        d2.close()
        return bool(got)
    finally:
        r.close(0)
        ctx.term()


def test_fixed_identity_recreated_dealer_is_silently_dropped():
    # documents the root cause: same identity while the old pipe drains
    assert not _router_receives_after_recreate(lambda: b"rk-return-3")


def test_unique_identity_recreated_dealer_is_delivered():
    assert _router_receives_after_recreate(lambda: cx._dealer_identity("rk-return-3"))


def test_dealer_identity_unique():
    ids = {cx._dealer_identity("rk-1") for _ in range(100)}
    assert len(ids) == 100 and all(i.startswith(b"rk-1-") for i in ids)


# ---- recv_step: no 0 ms spin, one timeout per attempt -------------------------

def test_recv_step_timeout_does_not_spin_or_reconnect_first_time():
    c, _ = _k(0, 4)
    _routing(c)
    inval = []
    c._invalidate_stage_connection = lambda s: inval.append(s)
    calls = []
    orig = c.recv_tensor_dict

    def counting():
        calls.append(1)
        return orig()
    c.recv_tensor_dict = counting
    a = c.new_step_id()
    c.router_socket = _BlockingRouter([])
    t = time.monotonic()
    assert c.recv_step(a, timeout_s=0.05) is None
    assert time.monotonic() - t < 1.0
    assert len(calls) <= 2                      # not a burst of 1 ms polls
    assert inval == []                          # healthy peer: keep the socket
    assert c.recv_step(a, timeout_s=0.02) is None
    assert inval == [1]                         # 2nd timeout of the same step: reconnect
    c.abandon_step(a)
    assert a not in c._step_timeouts


def test_recv_step_still_matches_after_timeouts():
    c, _ = _k(0, 2)
    _routing(c)
    c._invalidate_stage_connection = lambda s: None
    a, b = c.new_step_id(), c.new_step_id()
    c.router_socket = FakeRouter([])
    assert c.recv_step(a, timeout_s=0.02) is None
    c.router_socket.msgs += [_reply(b, 2), _reply(a, 1)]
    assert c.recv_step(a, timeout_s=1)["mb"] == 1
    assert c.recv_step(b, timeout_s=1)["mb"] == 2


# ---- UNHEALTHY is a suspicion, confirmed by the data path ---------------------

def test_unhealthy_of_unrelated_replica_changes_nothing():
    c, _ = _k(0, 4)
    _routing(c)
    c._sticky_peer = {1: 0}
    for st in (0, 1, 2, 3):
        c._on_endpoint_unhealthy(_Ev(st, 1))
    assert c._suspect_replicas == set() and c._deregistered_replicas == set()
    assert c._sticky_peer == {1: 0} and c.downstream_healthy()


def test_unhealthy_of_our_peer_is_suspect_then_confirmed_by_timeout():
    c, _ = _k(0, 4)
    _routing(c)
    c._sticky_peer = {1: 0}
    c._on_endpoint_unhealthy(_Ev(1, 0))
    assert c._suspect_replicas == {(1, 0)} and c._sticky_peer == {1: 0}
    assert not c.downstream_healthy()
    a = c.new_step_id()
    c.router_socket = FakeRouter([])
    assert c.recv_step(a, timeout_s=0.02) is None   # data path confirms
    assert (1, 0) in c._deregistered_replicas and 1 not in c._sticky_peer
    assert c._suspect_replicas == set()


def test_registered_clears_suspect():
    c, _ = _k(0, 4)
    _routing(c)
    c._sticky_peer = {1: 0}
    c._on_endpoint_unhealthy(_Ev(1, 0))
    ev = _Ev(1, 0)
    ev.payload["address_changed"] = False
    c._on_endpoint_registered(ev)
    assert c.downstream_healthy()


# ---- last stage keeps the return socket of the surviving stage 0 --------------

class _Sock:
    closed = False

    def close(self, *a):
        self.closed = True


def test_stage0_deregistration_keeps_return_sockets():
    c, _ = _k(3, 4)
    _routing(c)
    keep = _Sock()
    c._peer_dealer_cache["return_tcp://10.0.0.8:25500"] = (keep, "tcp://10.0.0.8:25500")
    fallback = c.return_socket = _Sock()
    c._on_endpoint_deregistered(_Ev(0, 1))
    assert not keep.closed
    assert "return_tcp://10.0.0.8:25500" in c._peer_dealer_cache
    assert fallback.closed and c.return_socket is None


def test_stage0_reply_pull_failure_is_retried_not_fatal(monkeypatch):
    """The last stage announced a reply and died before stage 0 pulled the
    payload: the pull fails with "Mooncake transfer failed". Stage 0 must
    treat it like a timeout (return None -> idempotent re-send / failover)
    instead of raising and killing the engine."""
    import struct
    from test_connexte_protocol import _comm

    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    url = b"10.0.0.1:50000"
    tail = cx._encode_g2(0, cx._encode_layout([]), b"", b"tcp://s0:1", b"")
    sig = (b"CONNEXTE" + struct.pack("!QQI", 0x1000, 64, len(url)) + url
           + struct.pack("!Q", 7) + tail)
    c, _ = _comm(0, n=2, msgs=[sig])
    c._inflight = 1
    c._tls = threading.local()
    c._pending_corr = 7
    invalidated = []
    c._invalidate_stage_connection = lambda s: invalidated.append(s)

    def boom(*a, **k):
        raise RuntimeError("Mooncake transfer failed with code -1")

    for name in ("_transfer_sync", "_recv_gdr_payload", "_gdr_dst"):
        monkeypatch.setattr(c, name, boom, raising=False)
    assert c.recv_tensor_dict() is None
    assert invalidated == [1]
