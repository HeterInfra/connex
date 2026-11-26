# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Register replies at scale: cached encoded route tables, snapshot modes.

Old clients (no ``snapshot`` field) must get the same reply bytes as before;
new clients ask for a compact table or none. The router keeps per-route JSON
fragments and a per-model compact table, which must follow every change.
"""

import json
import socket
import time

import pytest

from connex.epc_client import EPCV2Client, EPCV2ClientConfig
from connex.front_router import (
    EncodedJSON,
    FrontRouter,
    FrontRouterConfig,
    LogicalEndpointID,
    RouteEntry,
    RouteState,
    TransportCapability,
)

MODEL = "m"


class _CapturePub:
    def __init__(self):
        self.sent = []

    def send_multipart(self, parts):
        self.sent.append(parts)


def _fr():
    fr = FrontRouter(FrontRouterConfig())
    fr.event_bus._pub_socket = _CapturePub()
    fr.event_bus._running = True
    return fr


def _lid(model=MODEL, stage=0, replica=0, shard=0):
    return {"model_name": model, "stage_idx": stage, "shard_idx": shard, "replica_idx": replica}


def _reg(fr, snapshot=None, addr="tcp://h:1", group_size=None, **kw):
    params = {"lid": _lid(**kw), "addresses": [addr], "capabilities": [0, 2]}
    if snapshot is not None:
        params["snapshot"] = snapshot
    if group_size is not None:
        params["group_size"] = group_size
    return fr._handle_request({"method": "register", "params": params})


def _old_reply_bytes(fr, model, reply):
    """The reply the router built before encoded tables existed."""
    ref = {
        "success": True,
        "epoch": reply["epoch"],
        "is_reconnect": reply["is_reconnect"],
        "address_changed": reply["address_changed"],
        "route_table": [r.to_dict() for r in fr.route_manager.get_routes_for_model(model)],
        "seq": reply["seq"],
    }
    return json.dumps(ref).encode("utf8")


def _expected_compact(fr, model=MODEL):
    return [r.to_compact_dict() for r in fr.route_manager.get_routes_for_model(model)]


def _cached_compact(fr, model=MODEL):
    return json.loads(fr.route_manager.model_routes_json(model, compact=True).text)


# ------------------------------------------------------------ old clients

def test_old_client_reply_is_byte_identical():
    fr = _fr()
    _reg(fr, replica=0)
    _reg(fr, model="other")
    _reg(fr, stage=1, replica=0, shard=0, group_size=2)    # metadata in full form
    _reg(fr, stage=1, replica=0, shard=1, group_size=2)
    _reg(fr, model="mé", replica=0)                   # non-ASCII model name
    # a route loaded from persistence can carry an int timestamp
    fr.route_manager.get_route(LogicalEndpointID(MODEL, 0, 0, 0)).last_heartbeat = 1700000000
    for kw in ({"replica": 1}, {"replica": 1, "addr": "tcp://h:2"}, {"model": "mé"}):
        r = _reg(fr, **kw)
        assert "snapshot" not in r and "router_id" not in r
        wire = fr._encode_reply(r)
        assert wire == _old_reply_bytes(fr, kw.get("model", MODEL), r)
        # in-process callers still see a list of dicts
        assert r["route_table"] == json.loads(wire)["route_table"]
        assert "last_heartbeat" in r["route_table"][0]


def test_route_json_fragments_follow_changes():
    r = RouteEntry(LogicalEndpointID(MODEL, 0, 0, 0), ["a:1"], [TransportCapability.ZEROMQ], 1)

    def check():
        assert r.to_json() == json.dumps(r.to_dict())
        assert r.to_compact_json() == json.dumps(r.to_compact_dict())

    check()
    r.last_heartbeat = time.time() + 5
    check()
    r.addresses = ["b:2", "c:3"]
    check()
    r.capabilities = [TransportCapability.UCX_RDMA]
    check()
    r.epoch = 7
    check()
    r.state = RouteState.UNHEALTHY
    check()
    r.metadata = {"group_size": 4}
    check()
    assert json.loads(r.to_compact_json())["group_size"] == 4
    r.registered_at = 12.5
    check()
    r.last_heartbeat = float("inf")    # json.dumps writes Infinity, not inf
    check()


# ------------------------------------------------------------ new clients

def test_compact_and_none_replies():
    fr = _fr()
    _reg(fr, replica=0)
    _reg(fr, model="other")
    r = _reg(fr, snapshot="compact", replica=1)
    wire = json.loads(fr._encode_reply(r))
    assert wire["snapshot"] == "compact" and wire["router_id"] == fr.router_id
    assert wire["route_table"] == _expected_compact(fr)
    assert sorted(x["lid"]["replica_idx"] for x in wire["route_table"]) == [0, 1]
    assert "last_heartbeat" not in wire["route_table"][0]
    assert wire["seq"] == fr.event_bus.seq - 1    # read before its own REGISTERED

    r = _reg(fr, snapshot="none", replica=2)
    wire = json.loads(fr._encode_reply(r))
    assert "route_table" not in wire and wire["snapshot"] == "none"
    assert wire["success"] and wire["seq"] == fr.event_bus.seq - 1

    r = _reg(fr, snapshot="bogus", replica=3)        # unknown -> full table
    assert r["snapshot"] == "full" and "last_heartbeat" in r["route_table"][0]


def test_get_routes_encoded_matches_dicts():
    fr = _fr()
    for i in range(3):
        _reg(fr, stage=i % 2, replica=i)
    _reg(fr, model="other")
    rm = fr.route_manager
    for params, routes in (
        ({"model_name": MODEL}, rm.get_routes_for_model(MODEL)),
        ({}, rm.get_all_routes()),
        ({"model_name": MODEL, "stage_idx": 1}, rm.get_routes_for_stage(MODEL, 1)),
    ):
        for compact in (False, True):
            r = fr._handle_request({"method": "get_routes",
                                    "params": dict(params, compact=compact)})
            got = json.loads(fr._encode_reply(r))["routes"]
            want = [x.to_compact_dict() if compact else x.to_dict() for x in routes]
            assert got == want


# --------------------------------------------------------- invalidation

def _age(rm, route, ts):
    # oldest heartbeat goes to the front of the expiry order
    route.last_heartbeat = ts
    rm._hb_order[route.lid.to_key()] = ts
    rm._hb_order.move_to_end(route.lid.to_key(), last=False)


def test_compact_table_cache_invalidation():
    fr = _fr()
    rm = fr.route_manager
    _reg(fr, replica=0)
    _reg(fr, replica=1)
    first = rm.model_routes_json(MODEL, compact=True)
    assert json.loads(first.text) == _expected_compact(fr)

    # re-registration with nothing changed reuses the cached table
    _reg(fr, snapshot="compact", replica=1)
    assert rm.model_routes_json(MODEL, compact=True) is first
    # heartbeats do not touch the compact form
    fr._handle_request({"method": "heartbeat", "params": {"lid": _lid(replica=0)}})
    assert rm.model_routes_json(MODEL, compact=True) is first

    def changed():
        nonlocal first
        cur = rm.model_routes_json(MODEL, compact=True)
        assert cur is not first
        assert json.loads(cur.text) == _expected_compact(fr)
        first = cur

    _reg(fr, replica=2)                                   # new route
    changed()
    _reg(fr, replica=2, addr="tcp://h:9")                 # address change
    changed()
    fr._handle_request({"method": "update_route", "params": {
        "lid": _lid(replica=2), "addresses": ["tcp://h:10"]}})
    changed()
    _reg(fr, replica=1, group_size=2)                     # metadata (group_size)
    changed()
    fr._handle_request({"method": "deregister", "params": {"lid": _lid(replica=0)}})
    changed()

    # expiry: UNHEALTHY first, then removal after the grace period
    route = rm.get_route(LogicalEndpointID(MODEL, 0, 0, 2))
    old = time.time() - 100
    _age(rm, route, old)
    unhealthy, removed = rm.collect_expired(timeout_s=10, grace_s=1000)
    assert unhealthy and not removed
    changed()
    assert any(x["state"] == int(RouteState.UNHEALTHY) for x in json.loads(first.text))
    # heartbeat recovery back to ACTIVE
    fr._handle_request({"method": "heartbeat", "params": {"lid": _lid(replica=2)}})
    changed()
    _age(rm, route, old)
    rm.collect_expired(timeout_s=10, grace_s=1000)
    changed()
    _, removed = rm.collect_expired(timeout_s=10, grace_s=0)
    assert removed
    changed()

    # other models are cached separately and unaffected
    _reg(fr, model="other")
    assert rm.model_routes_json(MODEL, compact=True) is first
    # removing the last route leaves no stale entry
    for r in rm.get_routes_for_model(MODEL):
        rm.remove_route(r.lid)
    assert rm.model_routes_json(MODEL, compact=True) == []
    assert MODEL not in rm._compact_json


def test_full_reply_carries_live_heartbeat_times():
    fr = _fr()
    _reg(fr, replica=0)
    _reg(fr, replica=1)
    r1 = _reg(fr, replica=1)
    fr.route_manager.get_route(LogicalEndpointID(MODEL, 0, 0, 0)).last_heartbeat = 123.0
    r2 = _reg(fr, replica=1)
    assert fr._encode_reply(r2) == _old_reply_bytes(fr, MODEL, r2)
    hb = {x["lid"]["replica_idx"]: x["last_heartbeat"] for x in r2["route_table"]}
    assert hb[0] == 123.0 and r1 is not r2


def test_encoded_json_behaves_like_its_value():
    e = EncodedJSON('[{"a": 1}, {"b": 2}]')
    assert len(e) == 2 and e[0] == {"a": 1} and list(e) == [{"a": 1}, {"b": 2}]
    assert e == [{"a": 1}, {"b": 2}] and {"b": 2} in e
    assert FrontRouter._encode_reply({"x": e, "y": 1}) == b'{"x": [{"a": 1}, {"b": 2}], "y": 1}'
    assert FrontRouter._encode_reply({"y": [1]}) == b'{"y": [1]}'


# ---------------------------------------------------- end to end (ZMQ)

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _router(api, ev):
    r = FrontRouter(FrontRouterConfig(api_address=f"tcp://127.0.0.1:{api}",
                                      event_pub_address=f"tcp://127.0.0.1:{ev}"))
    r.start()
    return r


def _client(api, ev, stage, replica=0, **kw):
    return EPCV2Client(EPCV2ClientConfig(
        front_router_api=f"tcp://127.0.0.1:{api}",
        front_router_events=f"tcp://127.0.0.1:{ev}",
        model_name=MODEL, stage_idx=stage, replica_idx=replica,
        local_addresses=[f"10.0.{stage}.{replica}:1"],
        capabilities=[TransportCapability.ZEROMQ],
        heartbeat_interval_s=0.1, **kw))


@pytest.fixture
def ports():
    return _free_port(), _free_port()


def _keys(routes):
    return sorted(r.lid.to_key() for r in routes)


@pytest.mark.parametrize("mode", ["compact", "full", "none"])
def test_e2e_snapshot_modes(ports, mode):
    api, ev = ports
    router = _router(api, ev)
    clients = []
    try:
        for i in range(3):
            c = _client(api, ev, stage=i, register_snapshot=mode)
            assert c.start()
            clients.append(c)
            time.sleep(0.05)
        want = _keys(router.route_manager.get_routes_for_model(MODEL))
        assert len(want) == 3
        for c in clients:
            assert _wait(lambda c=c: _keys(c.route_table.get_all()) == want), mode
            assert _wait(c.cache_in_sync)
            assert _wait(lambda c=c: c._last_seq == router.event_bus.seq)
        # stage lookups and pipeline peers work in every mode
        assert [r.lid.stage_idx for r in clients[0].get_stage_peers(2)] == [2]
        assert clients[1].get_next_stage().lid.stage_idx == 2
        # a later change arrives as an event and is applied incrementally
        before = clients[0].stats["full_refreshes"]
        c = _client(api, ev, stage=3, register_snapshot=mode)
        assert c.start()
        clients.append(c)
        assert _wait(lambda: len(clients[0].route_table.get_all()) == 4)
        assert _wait(lambda: clients[0]._last_seq == router.event_bus.seq)
        assert clients[0].stats["full_refreshes"] == before
    finally:
        for c in clients:
            c.stop()
        router.stop()


def test_e2e_new_client_with_router_that_ignores_snapshot(ports, monkeypatch):
    """Older routers do not know the field and send the full table."""
    api, ev = ports
    orig = FrontRouter._handle_register

    def old_router(self, params):
        params = {k: v for k, v in params.items() if k != "snapshot"}
        reply = orig(self, params)
        reply["route_table"] = list(reply["route_table"])   # plain old form
        return reply

    monkeypatch.setattr(FrontRouter, "_handle_register", old_router)
    router = _router(api, ev)
    a = _client(api, ev, stage=0, register_snapshot="none")
    b = _client(api, ev, stage=1)
    try:
        assert a.start() and b.start()
        assert _wait(lambda: len(b.route_table.get_all()) == 2)
        assert _wait(lambda: len(a.route_table.get_all()) == 2)
        assert _wait(lambda: b._last_seq == router.event_bus.seq)
    finally:
        a.stop()
        b.stop()
        router.stop()


def test_e2e_router_restart_resyncs_seq(ports):
    """A new router restarts seq at 0: the client must not keep its old cache."""
    api, ev = ports
    router = _router(api, ev)
    peers = [_client(api, ev, stage=s) for s in (1, 2, 3)]
    me = _client(api, ev, stage=0)
    try:
        for p in peers:
            assert p.start()
        assert me.start()
        for p in peers:      # generate events so the old router's seq is well ahead
            p.update_my_address([f"10.9.{p.lid.stage_idx}.0:2"])
        assert _wait(lambda: me._last_seq == router.event_bus.seq)
        old_seq = me._last_seq
        assert old_seq >= 6 and len(me.route_table.get_all()) == 4
        for p in peers:
            p.stop()
        router.stop()

        router = _router(api, ev)             # empty, seq 0, new router_id
        assert _wait(lambda: router.route_manager.get_route(me.lid) is not None, 10)
        # the cache now matches the new router (only me), not the old one
        assert _wait(lambda: _keys(me.route_table.get_all()) == [me.lid.to_key()], 5)
        assert _wait(lambda: me._last_seq == router.event_bus.seq) and me._last_seq < old_seq
        q = _client(api, ev, stage=5)
        assert q.start()
        try:
            assert _wait(lambda: len(me.route_table.get_all()) == 2)
        finally:
            q.stop()
    finally:
        me.stop()
        router.stop()
