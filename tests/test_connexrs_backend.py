# SPDX-License-Identifier: Apache-2.0
"""connex-rs data-plane backend: the real Rust engine (connex_rs.Engine)
over UCX on localhost (TCP), no GPU, no cluster, no fakes on the data path.

* the TransferEngine surface the endpoint uses (managed buffers, registered
  memory, one-sided reads), signals and peer events;
* real ConnexTEEndpoints with connex-rs engines: steady state, and failover
  driven by the transport's peer-failure event alone (no lease, no probe);
* same-identity rejoin: a restarted stage under the same id and URL is read
  from again without anything else restarting.

Skipped when connex_rs is not installed (pip install connex[rs]).
"""

import os
import threading
import time

import pytest
import torch

connex_rs = pytest.importorskip("connex_rs")

from connex.pp import Stage0Link, transports  # noqa: E402
from connex.pp.routing import Lid, Route  # noqa: E402
from test_pp_core import (  # noqa: E402
    DownstreamEngine, Pump, Stage0Engine, _port, _prefill_then_decode, _worker,
)



@pytest.fixture(autouse=True)
def _ucx_env(monkeypatch):
    # UCX reads its environment when an engine starts; keep it local to
    # these tests (other tests use ucx-py with their own transports).
    monkeypatch.setenv("UCX_TLS", os.environ.get("CONNEX_RS_TEST_TLS", "tcp"))
    monkeypatch.setenv("UCX_TCP_CM_REUSEADDR", "y")
    monkeypatch.setenv("UCX_WARN_UNUSED_ENV_VARS", "n")


def _engine(stage, replica=0, port=0, **kw):
    return connex_rs.Engine("t", stage, 0, replica, listen_ip="127.0.0.1", port=port, **kw)


def _event(eng, kind, timeout_s=5.0):
    t = time.monotonic()
    while time.monotonic() - t < timeout_s:
        ev = eng.wait_event(100)
        if ev and ev["type"] == kind:
            return ev
    return None


# ---- TransferEngine surface ---------------------------------------------------------

def test_transfer_engine_surface():
    a, b = _engine(0), _engine(1)
    try:
        n = 1 << 20
        src = a.allocate_managed_buffer(n)
        dst = b.allocate_managed_buffer(n)
        data = os.urandom(n)
        a.write_bytes_to_buffer(src, data, n)
        assert b.transfer_sync_read(a.local_url, dst, src, n) == 0
        assert b.read_bytes_from_buffer(dst, n) == data
        # a sub-range at an offset, into a torch tensor registered by pointer
        t = torch.zeros(4096, dtype=torch.uint8)
        assert b.register_memory(t.data_ptr(), t.numel()) == 0
        assert b.transfer_sync_read(a.local_url, t.data_ptr(), src + 100, 4096) == 0
        assert bytes(t.numpy()) == data[100:4196]
        assert b.unregister_memory(t.data_ptr()) == 0
        # unknown remote region: an error code, not a dead connection
        assert b.transfer_sync_read(a.local_url, dst, 0x1000, 64) < 0
        assert b.transfer_sync_read(a.local_url, dst, src, 64) == 0
        # one-sided write + signal: data is in place when the message arrives
        b.write_bytes_to_buffer(dst, b"x" * 64, 64)
        b.put_signal(a.local_url, dst, src, 64, b"sig").wait(5000)
        peer, msg = a.recv_msg(5000)
        assert peer == ("t", 1, 0, 0) and msg == b"sig"
        assert a.read_bytes_from_buffer(src, 64) == b"x" * 64
        a.free_managed_buffer(src, n)
        b.free_managed_buffer(dst, n)
        assert transports.verify_engine(a) is None
        st = b.stats()
        assert st["inflight"] == 0 and st["regions"] == 0
    finally:
        a.close()
        b.close()


def test_blocking_calls_release_the_gil():
    """A read that waits (rkey round trip, peer progress) must not stall
    other Python threads."""
    a, b = _engine(0), _engine(1)
    try:
        n = 16 << 20
        src, dst = a.allocate_managed_buffer(n), b.allocate_managed_buffer(n)
        ticks = []
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                ticks.append(time.monotonic())
                time.sleep(0.001)
        th = threading.Thread(target=spin)
        th.start()
        t0 = time.monotonic()
        for _ in range(20):
            assert b.transfer_sync_read(a.local_url, dst, src, n) == 0
        took = time.monotonic() - t0
        stop.set()
        th.join()
        assert len([x for x in ticks if x >= t0]) > took / 0.001 * 0.3
    finally:
        a.close()
        b.close()


def test_peer_failure_event_and_same_identity_rejoin():
    b = _engine(1)
    a = _engine(0)
    port, url = a.port, a.local_url
    try:
        src = a.allocate_managed_buffer(4096)
        dst = b.allocate_managed_buffer(4096)
        a.write_bytes_to_buffer(src, b"1" * 4096, 4096)
        assert b.transfer_sync_read(url, dst, src, 4096) == 0
        t0 = time.monotonic()
        a.close()                                  # the process "dies"
        ev = _event(b, "failed")
        assert ev is not None and ev["peer"] == ("t", 0, 0, 0)
        assert time.monotonic() - t0 < 1.0
        assert b.transfer_sync_read(url, dst, src, 4096, timeout_ms=1000) < 0
        a = _engine(0, port=port)                  # same id, same URL
        assert a.local_url == url
        src = a.allocate_managed_buffer(4096)
        a.write_bytes_to_buffer(src, b"2" * 4096, 4096)
        for _ in range(50):
            if b.transfer_sync_read(url, dst, src, 4096) == 0:
                break
            time.sleep(0.05)
        assert b.read_bytes_from_buffer(dst, 4096) == b"2" * 4096
    finally:
        a.close()
        b.close()


def test_backend_from_env(monkeypatch):
    monkeypatch.setenv("CONNEX_PP_BACKEND", "connex_rs")
    assert transports.backend_from_env() == "connex-rs"
    monkeypatch.setenv("CONNEX_PP_BACKEND", "zmq")
    assert transports.backend_from_env() == "zmq"
    monkeypatch.setenv("CONNEX_PP_BACKEND", "bogus")
    with pytest.raises(ValueError):
        transports.backend_from_env()
    # auto: connex-rs only with an RDMA device, else Mooncake, else ZeroMQ
    monkeypatch.delenv("CONNEX_PP_BACKEND")
    monkeypatch.setattr(transports, "rdma_present", lambda: True)
    assert transports.backend_from_env() == "connex-rs"
    monkeypatch.setattr(transports, "rdma_present", lambda: False)
    monkeypatch.setattr(transports, "MOONCAKE_AVAILABLE", True)
    assert transports.backend_from_env() == "mooncake"
    monkeypatch.setattr(transports, "MOONCAKE_AVAILABLE", False)
    assert transports.backend_from_env() == "zmq"


# ---- real endpoints over connex-rs -------------------------------------------------

@pytest.fixture
def crs(monkeypatch):
    """mk(stage, n, replica, k, routes) -> (endpoint, membership, engine);
    the route carries the connex-rs URL as second address."""
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 400)
    engines = []

    def mk(stage, n, replica=0, k=1, routes=None):
        eng = _engine(stage, replica)
        engines.append(eng)
        from connex.pp import StaticMembership
        mem = StaticMembership(stage, replica)
        if routes is not None:
            mem.routes = routes
        ep = core.ConnexTEEndpoint(stage, n, _port(), "127.0.0.1:1", device="cpu",
                                   membership=mem, transfer_engine=eng,
                                   local_url=eng.local_url, inflight=k)
        _made.append(ep)
        return ep, mem, eng

    _made = []
    yield mk
    for ep in _made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass
    for e in engines:
        e.close()


@pytest.mark.parametrize("k", [1, 4])
def test_real_endpoints_steady_state_connexrs(crs, k):
    routes = []
    eps = []
    for s in range(3):
        ep, _, eng = crs(s, 3, k=k, routes=routes)
        routes.append(Route(Lid(s, 0), [ep.router_endpoint, eng.local_url]))
        eps.append(ep)
    assert eps[0]._mooncake_available
    e1, e2 = DownstreamEngine(), DownstreamEngine()
    s0 = Stage0Link(eps[0], Stage0Engine(), recv_timeout_s=2.0)
    with Pump([_worker(eps[1], e1), _worker(eps[2], e2)]):
        replies, _ = _prefill_then_decode(s0, ["a", "b"], decode=6)
    assert all(torch.equal(r["h"], torch.full((2,), 2.0)) for r in replies)
    assert e2.ledger.kv_len == {"a": 10, "b": 10}


def test_failover_by_transport_event_alone(crs):
    """K=4: stage-1 replica a dies with steps in flight. No lease event and
    no data-path timeout: the connex-rs peer-failure event alone fails the
    chain over, and the stale steps complete on replica b quickly."""
    routes = []
    s0ep, _, _ = crs(0, 3, k=4, routes=routes)
    s1a, _, e1a_eng = crs(1, 3, replica=0, k=4, routes=routes)
    s1b, _, _ = crs(1, 3, replica=1, k=4, routes=routes)
    s2, _, _ = crs(2, 3, k=4, routes=routes)
    for ep, rep in ((s0ep, 0), (s1a, 0), (s1b, 1), (s2, 0)):
        routes.append(Route(Lid(ep.stage_idx, rep), [ep.router_endpoint, ep._local_url]))
    s0 = Stage0Link(s0ep, Stage0Engine(), recv_timeout_s=10.0)
    pump_a = Pump([_worker(s1a, DownstreamEngine("1a"))])
    with Pump([_worker(s1b, DownstreamEngine("1b")), _worker(s2, DownstreamEngine())]), pump_a:
        _prefill_then_decode(s0, ["w"], decode=1)
        assert s0ep._sticky_peer[1] == 0
        deadline = time.monotonic() + 5
        while (1, 0) not in s0ep._crs_alive and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (1, 0) in s0ep._crs_alive          # linked by the watcher
        pump_a.stop.set()
        pump_a.threads[0].join(5)
        ids = [s0.submit({"h": torch.full((1,), float(i)), "nsched": [4]}, [f"r{i}"], [0])
               for i in range(4)]
        t = time.monotonic()
        e1a_eng.close()                            # stage-1 replica a "dies"
        s1a.close()
        outs = [s0.wait(c) for c in ids]
        took = time.monotonic() - t
    assert [o["h"].item() for o in outs] == [2.0, 3.0, 4.0, 5.0]
    assert s0ep._sticky_peer[1] == 1 and s0ep._chain_epoch == 1
    assert (1, 0) in s0ep._crs_dead
    assert took < 2.0, took


def test_transport_confirmed_remote_death_resends_k1_step_sooner(monkeypatch):
    """Stage 0, K=1: a death two stages down reported by the transport makes
    the in-flight step stale after ~2 step times (not the 0.5 s floor that a
    probe-confirmed death keeps); K>1 keeps the floor (a duplicate of a step
    still in progress downstream could be computed twice there)."""
    import logging
    import connex.pp.endpoint as core
    monkeypatch.setattr(core, "_TRANSPORT_FAILOVER_DELAY_S", 0.0)
    monkeypatch.setattr(core, "_REMOTE_FAILOVER_DELAY_S", 0.0)

    def mk(k):
        ep = core.ConnexTEEndpoint.__new__(core.ConnexTEEndpoint)
        ep.stage_idx, ep.num_stages, ep.log = 0, 4, logging.getLogger("t")
        ep.is_first = True
        ep._init_handover()
        ep._inflight, ep._recent_step_s = k, [0.05] * 8
        return ep

    for source, k, stale_at_0_15 in (("transport", 1, True), ("probe", 1, False),
                                      ("transport", 4, False)):
        ep = mk(k)
        rec = (ep._chain_epoch, ep._remote_epoch, time.monotonic() - 0.15)
        ep.on_peer_dead(2, 1, source=source)
        assert ep._send_is_stale(rec) is stale_at_0_15, (source, k)
        fresh = (ep._chain_epoch, ep._remote_epoch, time.monotonic())
        assert not ep._send_is_stale(fresh)
