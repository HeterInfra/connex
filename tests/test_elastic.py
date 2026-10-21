# SPDX-License-Identifier: Apache-2.0
"""Membership events, leases and ElasticPeerSet under join/leave/failure."""

import socket
import threading
import time

import pytest

from connex.directory import DirectoryClient, MembershipEvent
from connex.directory_server import DirectoryServer
from connex.elastic import ElasticPeerSet
from connex.ucx_transport import TransportMode, UCXTransport


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _until(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def server():
    srv = DirectoryServer(f"127.0.0.1:{_free_port()}", f"127.0.0.1:{_free_port()}",
                          lease_s=0.5)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def make(server):
    made = []

    def _make(stage, replica=0, addrs=None):
        c = DirectoryClient(server.bind_addr, "m", stage_idx=stage,
                            replica_idx=replica, local_addresses=addrs)
        c.start()
        made.append(c)
        return c

    yield _make
    for c in made:
        c.stop()


def _zmq(port=None):
    t = UCXTransport("127.0.0.1", port or _free_port(), mode=TransportMode.ZEROMQ,
                     device="cpu", buffer_size=1024, num_buffers=1)
    t.start_sync()
    return t


def test_push_events_join_and_leave(make):
    watcher = make(0)
    seen = []
    got = threading.Event()

    def on_event(ev, info):
        seen.append((ev, info.lid))
        got.set()

    watcher.subscribe_events(on_event)
    assert watcher.start_event_listener()
    r = make(1, addrs=["127.0.0.1:7000"])
    assert r.register()
    assert _until(lambda: (MembershipEvent.JOIN, ("m", 1, 0, 0)) in seen)
    assert r.deregister()
    assert _until(lambda: (MembershipEvent.LEAVE, ("m", 1, 0, 0)) in seen)


def test_lease_expiry_evicts_silent_member(server, make):
    r = make(1, addrs=["127.0.0.1:7000"])
    assert r.register()
    r.start_heartbeat(0.1)
    time.sleep(0.8)  # heartbeats keep it alive past one lease
    assert ("m", 1, 0, 0) in server.members()
    r._hb_stop.set()  # simulate a crash: heartbeats stop
    assert _until(lambda: ("m", 1, 0, 0) not in server.members(), timeout=2)


def test_report_failure_evicts_and_live_member_rejoins(server, make):
    r = make(1, addrs=["127.0.0.1:7000"])
    assert r.register()
    r.start_heartbeat(0.1)
    reporter = make(0)
    assert reporter.report_failure(("m", 1, 0, 0))
    # A falsely reported member is alive and re-registers on its next heartbeat.
    assert _until(lambda: ("m", 1, 0, 0) in server.members(), timeout=2)


def test_peer_set_routes_only_connected_members(server, make):
    rx1, rx2 = _zmq(), _zmq()
    r1 = make(1, 1, [f"127.0.0.1:{rx1.local_port}"])
    assert r1.register()
    tx = _zmq()
    owner = make(0)
    peers = ElasticPeerSet(owner, tx, stage_idx=1, drain_grace_s=0.1)
    try:
        peers.start()
        assert peers.wait_ready(1, timeout=3)
        assert peers.peers() == [("m", 1, 0, 1)]

        # Scale out: becomes routable without any send blocking on connect.
        r2 = make(1, 2, [f"127.0.0.1:{rx2.local_port}"])
        assert r2.register()
        assert peers.wait_ready(2, timeout=3)

        picks = {peers.pick() for _ in range(4)}
        assert picks == {("m", 1, 0, 1), ("m", 1, 0, 2)}
        for lid in picks:
            peers.send(lid, b"x")
        assert rx1.recv_sync(timeout_ms=2000) == b"x"
        assert rx2.recv_sync(timeout_ms=2000) == b"x"

        # Graceful scale in: removed from routing on the LEAVE event.
        assert r1.deregister()
        assert _until(lambda: peers.peers() == [("m", 1, 0, 2)])

        # Failure: removed locally and evicted from the directory.
        peers.mark_failed(("m", 1, 0, 2))
        assert peers.peers() == []
        assert ("m", 1, 0, 2) not in server.members()
    finally:
        peers.stop()
        for t in (tx, rx1, rx2):
            t.stop_sync()


def test_peer_set_falls_back_to_polling(server, make):
    rx = _zmq()
    owner = make(0)
    owner.start_event_listener = lambda *a, **k: False  # directory without events
    tx = _zmq()
    peers = ElasticPeerSet(owner, tx, stage_idx=1, poll_s=0.05)
    try:
        peers.start()
        r = make(1, 0, [f"127.0.0.1:{rx.local_port}"])
        assert r.register()
        assert peers.wait_ready(1, timeout=3)
        assert r.deregister()
        assert _until(lambda: peers.peers() == [])
    finally:
        peers.stop()
        tx.stop_sync()
        rx.stop_sync()
