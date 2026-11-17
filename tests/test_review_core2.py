# SPDX-License-Identifier: Apache-2.0
"""Regression tests for review fixes in front_router, epc_client and
elastic."""

import socket
import time


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


def _orchestrator():
    from connex.front_router import (
        EpochManager, InMemoryPersistence, LogicalEndpointID, MigrationOrchestrator,
        RouteManager,
    )
    p = InMemoryPersistence()
    rm = RouteManager(p)
    lid = LogicalEndpointID("m", 0)
    rm.add_route(lid, ["a:1"], [], 0)
    return MigrationOrchestrator(rm, EpochManager(p), _Bus()), rm, lid, p


def test_aborted_migration_cannot_be_cut_over():
    orch, rm, lid, _ = _orchestrator()
    mig = orch.prepare_migration(lid, "b:2")
    assert orch.abort_migration(mig.migration_id, "test")
    assert not orch.execute_cutover(mig.migration_id)
    assert rm.get_route(lid).addresses == ["a:1"]


def test_cutover_is_persisted_and_not_replayed():
    orch, rm, lid, p = _orchestrator()
    mig = orch.prepare_migration(lid, "b:2")
    assert orch.execute_cutover(mig.migration_id)
    assert p.get(f"routes/{lid.to_key()}")["addresses"] == ["b:2"]
    assert not orch.execute_cutover(mig.migration_id)
    assert not orch.abort_migration(mig.migration_id, "late")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_front_router_stop_joins_api_thread():
    from connex.front_router import FrontRouter, FrontRouterConfig
    r = FrontRouter(FrontRouterConfig(
        api_address=f"tcp://127.0.0.1:{_free_port()}",
        event_pub_address=f"tcp://127.0.0.1:{_free_port()}"))
    r.start()
    t = r._api_thread
    r.stop()
    assert not t.is_alive()
    # publishing after stop must be a no-op, not a crash
    from connex.front_router import Event, EventType
    r.event_bus.publish(Event(EventType.EPOCH_BUMPED, 0, {}))


def test_close_connection_does_not_match_key_prefix():
    from connex.epc_client import DataPlaneManager, EPCV2ClientConfig, LogicalEndpointID
    dp = DataPlaneManager(EPCV2ClientConfig())
    a, b = LogicalEndpointID("m", 1, 0, 1), LogicalEndpointID("m", 1, 0, 10)
    dp.get_or_create_connection(a, "x:1")
    dp.get_or_create_connection(b, "y:1")
    dp.close_connection(a)
    assert list(dp._connections) == [f"{b.to_key()}@y:1"]


class _Transport:
    def __init__(self):
        self.closed = []

    def close_endpoint(self, ep):
        self.closed.append(ep)


class _Dir:
    model_name = "m"


def test_elastic_draining_endpoint_closed_after_stop():
    from connex.elastic import ElasticPeerSet
    tr = _Transport()
    ps = ElasticPeerSet(_Dir(), tr, stage_idx=1, drain_grace_s=0.05)
    ps._ready[("m", 1, 0, 0)] = ("ep0", "h:1")
    ps._wanted[("m", 1, 0, 0)] = "h:1"
    ps._remove(("m", 1, 0, 0))  # starts draining ep0
    ps.stop()
    time.sleep(0.2)
    assert tr.closed == ["ep0"]
    ps._want(("m", 1, 0, 1), "h:2")  # late event after stop: no exception
