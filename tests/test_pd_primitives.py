# SPDX-License-Identifier: Apache-2.0
"""PD (prefill-decode) endpoint primitives in the versioned route table.

These are the building blocks a PD connector would use; there is no engine
integration in this repository yet (see README, Scope and status).
"""

from connex.mobility_control import (
    ConnectionCapability,
    ConnectionState,
    VersionedRouteTable,
)
from connex.protocol import LogicalEndpointID, PDRole


def _table():
    return VersionedRouteTable(LogicalEndpointID("m", 0, 0, 0, pd_role=PDRole.PREFILL))


def _connect(route):
    route.connection_state = ConnectionState.CONNECTED
    return route


def test_role_lookup_returns_only_connected_active_endpoints_of_that_role():
    t = _table()
    cap = ConnectionCapability(supports_rdma=True)
    d0 = _connect(t.register_pd_endpoint("m", PDRole.DECODE, 0, ["10.0.0.1:1"], cap))
    t.register_pd_endpoint("m", PDRole.DECODE, 1, ["10.0.0.2:1"], cap)  # not connected yet
    _connect(t.register_pd_endpoint("m", PDRole.PREFILL, 0, ["10.0.0.3:1"], cap))

    assert [r.primary_address for r in t.get_decode_endpoints()] == ["10.0.0.1:1"]
    assert [r.primary_address for r in t.get_prefill_endpoints()] == ["10.0.0.3:1"]
    assert d0.lid.is_decode and d0.lid.is_pd_mode

    t.mark_route_draining(d0.lid, d0.epoch)  # leaving: no new KV goes there
    assert t.get_decode_endpoints() == []


def test_decode_endpoint_migration_keeps_identity_and_prefers_newer_epoch():
    t = _table()
    cap = ConnectionCapability()
    old = _connect(t.register_pd_endpoint("m", PDRole.DECODE, 0, ["10.0.0.1:1"], cap))
    new = _connect(t.add_route(old.lid, ["10.0.0.9:1"], cap, epoch=t.increment_epoch()))
    assert new.lid == old.lid and new.epoch > old.epoch
    assert t.get_route(old.lid).primary_address == "10.0.0.9:1"
    assert [r.primary_address for r in t.get_decode_endpoints()] == ["10.0.0.9:1"]
