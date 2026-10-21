# SPDX-License-Identifier: Apache-2.0
"""Fast handover (CONNEXTE_FAST_HANDOVER): a suspected peer is confirmed by a
TCP probe, and a confirmed death fails every in-flight step of the chain at
once instead of one recv timeout per step (insights/reprefill-on-failover.md).

Runs through the vLLM shim; the logic lives in connex.pp.endpoint, so module
flags are patched there."""

import socket
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

import connex.pp.endpoint as core
from test_churn_fixes import _BlockingRouter, _Ev, _routing
from test_connexte_protocol import FakeRouter, cx
from test_pipelining import _k, _reply


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    monkeypatch.setattr(core, "_REMOTE_FAILOVER_DELAY_S", 0.0)


def _listener():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    return s, f"tcp://127.0.0.1:{s.getsockname()[1]}"


def _dead_addr():
    s, addr = _listener()
    s.close()
    return addr


class _Epc:
    def __init__(self, table):
        self.table = table  # (stage, replica) -> addr

    def get_pipeline_peers(self):
        return [SimpleNamespace(lid=SimpleNamespace(stage_idx=s, replica_idx=r), addresses=[a])
                for (s, r), a in self.table.items()]


def _stage0(k, table=None):
    c, _ = _k(0, k)
    _routing(c)
    c._init_handover()
    c._sticky_peer = {1: 0}
    c._epc_client = _Epc(table or {})
    c._route_dirty = set()
    return c


# ---- probe ----------------------------------------------------------------------

def test_probe_alive_dead_unknown():
    s, addr = _listener()
    try:
        assert cx._probe_tcp(addr) is True
    finally:
        s.close()
    t = time.monotonic()
    assert cx._probe_tcp(_dead_addr()) is False
    assert time.monotonic() - t < 0.5               # refused: no timeout wait
    assert cx._probe_tcp(None) is None and cx._probe_tcp("tcp://*:5") is None


# ---- confirmation needs the probe, not just a missed lease -----------------------

def test_suspect_but_alive_peer_is_kept():
    s, addr = _listener()
    try:
        c = _stage0(4, {(1, 0): addr})
        c._suspect_replicas.add((1, 0))
        assert not c.confirm_peer_failure()          # slow, not dead
        assert c._sticky_peer == {1: 0} and c._chain_epoch == 0
    finally:
        s.close()


def test_timeout_plus_refused_probe_fails_over_without_lease():
    c = _stage0(4, {(1, 0): _dead_addr()})
    assert c.confirm_peer_failure()                  # no UNHEALTHY needed
    assert (1, 0) in c._deregistered_replicas and 1 not in c._sticky_peer
    assert c._chain_epoch == 1


def test_unhealthy_of_dead_peer_fails_over_via_background_probe():
    c = _stage0(4, {(1, 0): _dead_addr()})
    c._on_endpoint_unhealthy(_Ev(1, 0))
    for _ in range(100):
        if c._chain_epoch:
            break
        time.sleep(0.01)
    assert c._chain_epoch == 1 and 1 not in c._sticky_peer


def test_unhealthy_of_live_peer_changes_nothing():
    s, addr = _listener()
    try:
        c = _stage0(4, {(1, 0): addr})
        c._on_endpoint_unhealthy(_Ev(1, 0))
        time.sleep(0.2)
        assert c._chain_epoch == 0 and c._sticky_peer == {1: 0}
    finally:
        s.close()


# ---- all in-flight steps fail at once --------------------------------------------

def test_multi_slot_failover_fails_all_inflight_steps_at_once():
    c = _stage0(4, {(1, 0): _dead_addr()})
    c._invalidate_stage_connection = lambda s: None
    steps = [c.new_step_id() for _ in range(4)]
    for a in steps:
        c._mark_sent(a)
    c.router_socket = _BlockingRouter([])
    c.on_peer_dead(1, 0)
    assert all(c.step_stale(a) for a in steps)
    t = time.monotonic()
    for a in steps:                                  # was: 4 x timeout
        assert c.recv_step(a, timeout_s=5.0) is None
    assert time.monotonic() - t < 0.5
    for a in steps:                                  # re-sent on the new chain
        c._mark_sent(a)
        assert not c.step_stale(a)
    c.router_socket.msgs += [_reply(a, i) for i, a in enumerate(steps)][::-1]
    assert [c.recv_step(a, timeout_s=1)["mb"] for a in steps] == [0, 1, 2, 3]


def test_waiting_recv_step_is_woken_by_failover():
    c = _stage0(4)
    c._invalidate_stage_connection = lambda s: None
    a = c.new_step_id()
    c._mark_sent(a)
    c.router_socket = _BlockingRouter([])
    threading.Timer(0.1, lambda: c.on_peer_dead(1, 0)).start()
    t = time.monotonic()
    assert c.recv_step(a, timeout_s=5.0) is None
    assert time.monotonic() - t < 1.0


def test_k1_recv_is_woken_by_failover():
    c = _stage0(1)
    c._invalidate_stage_connection = lambda s: None
    c._mark_sent()
    c.router_socket = _BlockingRouter([])
    c._recent_step_s = []
    threading.Timer(0.1, lambda: c.on_peer_dead(1, 0)).start()
    t = time.monotonic()
    assert c.recv_tensor_dict() is None              # base timeout is seconds
    assert time.monotonic() - t < 1.0


def test_late_reply_after_failover_still_matches_resent_step():
    c = _stage0(2)
    a, b = c.new_step_id(), c.new_step_id()
    c._mark_sent(a), c._mark_sent(b)
    c.on_peer_dead(1, 0)
    c._mark_sent(a), c._mark_sent(b)                 # both re-sent
    c.router_socket = FakeRouter([_reply(b, 2), _reply(a, 1), _reply(a, 9)])
    assert c.recv_step(a, timeout_s=1)["mb"] == 1
    assert c.recv_step(b, timeout_s=1)["mb"] == 2    # duplicate of a dropped


# ---- death farther down the chain (stage 2/3) ------------------------------------

def test_remote_death_resends_only_overdue_steps():
    c = _stage0(4, {(2, 1): _dead_addr()})
    old = c.new_step_id()
    c._mark_sent(old)
    c._sent_at[old] = (0, 0, time.monotonic() - 10)  # sent long ago: overdue
    fresh = c.new_step_id()
    c._mark_sent(fresh)
    c._probe_and_act(2, 1)
    assert c._remote_epoch == 1 and c._chain_epoch == 0
    assert c._sticky_peer == {1: 0}                  # our own peer is untouched
    assert c.step_stale(old) and not c.step_stale(fresh)


def test_remote_unhealthy_but_alive_changes_nothing():
    s, addr = _listener()
    try:
        c = _stage0(4, {(3, 1): addr})
        c._probe_and_act(3, 1)
        assert c._remote_epoch == 0
    finally:
        s.close()


def test_middle_stage_does_not_probe_beyond_its_peer():
    c, _ = _k(1, 4)
    _routing(c)
    c._init_handover()
    c._sticky_peer = {2: 0}
    c._epc_client = _Epc({(3, 0): _dead_addr()})
    c._on_endpoint_unhealthy(_Ev(3, 0))
    time.sleep(0.1)
    assert c._remote_epoch == 0 and not c._probing


def test_disabled_falls_back_to_lease(monkeypatch):
    monkeypatch.setattr(core, "_FAST_HANDOVER", False)
    c = _stage0(4, {(1, 0): _dead_addr()})
    assert not c.confirm_peer_failure()              # no probe, not suspect
    c._suspect_replicas.add((1, 0))
    assert c.confirm_peer_failure()


# ---- K=1: send and recv run on different threads in the fork ---------------------

def test_k1_duplicate_reply_never_taken_for_next_step():
    from test_pipelining import BASE
    c = _stage0(1)
    c._pending_corr = BASE + 5
    c.router_socket = FakeRouter([_reply(BASE + 5, 5)])
    assert c.recv_tensor_dict()["mb"] == 5
    # recv of the next step starts before the send thread set _pending_corr;
    # a late duplicate of step 5 (it was re-sent) must not be accepted
    assert c._pending_corr is None
    c.router_socket = FakeRouter([_reply(BASE + 5, 5), _reply(BASE + 6, 6)])
    assert c.recv_tensor_dict()["mb"] == 6


class _StolenRouter(FakeRouter):
    """poll() sees a message, but another thread reads it first."""

    def __init__(self, c, msg):
        super().__init__([])
        self.c, self.msg, self.stolen = c, msg, False

    def poll(self, timeout, flags=None):
        return 1

    def recv_multipart(self, flags=0):
        if not self.stolen:
            self.stolen = True
            self.c._stash.append([b"id", self.msg])   # what _drain_acks does
            raise zmq.Again()
        return super().recv_multipart(flags)


def test_k1_message_stolen_by_drain_acks_is_not_a_timeout():
    from test_pipelining import BASE
    c = _stage0(1)
    c._pending_corr = BASE + 1
    c.router_socket = _StolenRouter(c, _reply(BASE + 1, 1))
    assert c.recv_tensor_dict()["mb"] == 1


def test_k1_failover_before_recv_start_does_not_abort():
    c = _stage0(1)
    c._mark_sent()
    c.on_peer_dead(1, 0)          # failover, then the (new-chain) recv starts
    from test_pipelining import BASE
    c._pending_corr = BASE + 1
    c.router_socket = FakeRouter([_reply(BASE + 1, 1)])
    assert c.recv_tensor_dict()["mb"] == 1
