"""Control-plane robustness tests for EPCV2Client <-> FrontRouter.

Covers: thread-safe/self-healing REQ channels, dedicated heartbeat channel,
incremental route-cache maintenance from events (with seq gap detection),
stage-filtered queries, and failure-detection timing with small timeouts.
All routers run in-process on free ports.
"""

import json
import socket
import threading
import time

import pytest

from connex.epc_client import (
    APITimeoutError,
    EPCV2Client,
    EPCV2ClientConfig,
    Event as ClientEvent,
    EventType as CEventType,
    RouteState as CRouteState,
)
from connex.front_router import (
    Event,
    EventType,
    FrontRouter,
    FrontRouterConfig,
    LogicalEndpointID,
    RouteManager,
    RouteState,
    TransportCapability,
)

MODEL = "m"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=3.0, step=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(step)
    return cond()


@pytest.fixture
def router():
    api, ev = _free_port(), _free_port()
    r = FrontRouter(FrontRouterConfig(
        api_address=f"tcp://127.0.0.1:{api}",
        event_pub_address=f"tcp://127.0.0.1:{ev}",
    ))
    r.api_ep = f"tcp://127.0.0.1:{api}"
    r.ev_ep = f"tcp://127.0.0.1:{ev}"
    yield r
    r.stop()


def _client(router, stage, replica=0, **kw):
    cfg = EPCV2ClientConfig(
        front_router_api=router.api_ep,
        front_router_events=router.ev_ep,
        model_name=MODEL,
        stage_idx=stage,
        replica_idx=replica,
        local_addresses=[f"10.0.{stage}.{replica}:1"],
        capabilities=[TransportCapability.ZEROMQ],
        **kw,
    )
    return EPCV2Client(cfg)


# ---------------------------------------------------------------------------
# Wire compatibility
# ---------------------------------------------------------------------------

def test_event_wire_compat():
    e = Event(type=EventType.ENDPOINT_REGISTERED, epoch=3, payload={"lid": {}})
    d = e.to_dict()
    assert "seq" not in d  # unpublished events keep the old shape
    e.seq = 7
    d = json.loads(e.serialize())
    assert set(d) == {"type", "epoch", "payload", "timestamp", "source_lid", "seq"}
    # Old-format JSON (no seq) still parses on the client.
    old = json.dumps({"type": 9, "epoch": 1, "payload": {}}).encode()
    ce = ClientEvent.deserialize(old)
    assert ce.seq is None and ce.type == CEventType.ROUTE_UPDATED
    assert ClientEvent.deserialize(e.serialize()).seq == 7


# ---------------------------------------------------------------------------
# RouteManager indexing / O(expired) expiry
# ---------------------------------------------------------------------------

def test_route_manager_stage_index_and_expiry():
    rm = RouteManager()
    for st in range(3):
        for rep in range(4):
            rm.add_route(LogicalEndpointID(MODEL, st, 0, rep), [f"a{st}{rep}"], [], 0)
    assert [r.lid.replica_idx for r in rm.get_routes_for_stage(MODEL, 1)] == [0, 1, 2, 3]
    assert rm.get_routes_for_stage(MODEL, 9) == []

    # Nothing expires.
    assert rm.collect_expired(10.0, 5.0) == ([], [])
    time.sleep(0.25)
    # Everything heartbeats except stage 2 replica 3.
    stale = LogicalEndpointID(MODEL, 2, 0, 3)
    for r in rm.get_all_routes():
        if r.lid != stale:
            rm.heartbeat(r.lid)
    unhealthy, removed = rm.collect_expired(0.2, 10.0)
    assert [r.lid for r in unhealthy] == [stale] and removed == []
    assert rm.get_route(stale).state == RouteState.UNHEALTHY
    # Heartbeat brings it back (reported as recovered).
    assert rm.heartbeat(stale) == (True, True)
    assert rm.get_route(stale).state == RouteState.ACTIVE
    # Past timeout + grace -> removed, and stage index is cleaned.
    rm.collect_expired(10.0, 5.0, now=time.time() + 10.5)
    _, removed = rm.collect_expired(10.0, 5.0, now=time.time() + 16)
    assert len(removed) == 12
    assert rm.get_all_routes() == [] and rm.get_routes_for_stage(MODEL, 0) == []


def test_expiry_respects_out_of_band_heartbeat():
    rm = RouteManager()
    lid = LogicalEndpointID(MODEL, 0)
    route = rm.add_route(lid, ["a"], [], 0)
    later = time.time() + 100
    route.last_heartbeat = later  # set without touch()
    unhealthy, _ = rm.collect_expired(10.0, 5.0, now=later + 1)
    assert unhealthy == []
    assert rm.get_route(lid).state == RouteState.ACTIVE


# ---------------------------------------------------------------------------
# Thread safety and timeout recovery
# ---------------------------------------------------------------------------

def test_concurrent_heartbeat_and_refresh(router):
    router.start()
    c = _client(router, 0, heartbeat_interval_s=0.02)
    peers = [_client(router, 1, r) for r in range(3)]
    assert c.start()
    try:
        for p in peers:
            assert p.start()
        errors = []
        hb0 = router._stats["heartbeats"]

        def worker(i):
            try:
                for _ in range(30):
                    if i % 3 == 0:
                        assert c.refresh_routes()
                    elif i % 3 == 1:
                        c.get_stage_peers(1, refresh="always")
                    else:
                        assert c._send_heartbeat()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(9)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors, errors
        assert c._api.resets == 0 and c._hb.resets == 0
        assert router._stats["heartbeats"] - hb0 >= 90
        assert len(c.get_stage_peers(1, refresh="never")) == 3
    finally:
        for p in peers:
            p.stop()
        c.stop()


def test_timeout_recovers_socket(router):
    router.start()
    c = _client(router, 0, api_timeout_ms=200, heartbeat_interval_s=0.05)
    assert c.start()
    try:
        orig = router._handle_request
        slow = {"n": 1}

        def handle(req):
            if req.get("method") == "get_routes" and slow["n"] > 0:
                slow["n"] -= 1
                time.sleep(0.5)
            return orig(req)

        router._handle_request = handle
        with pytest.raises(APITimeoutError):
            c._call_api("get_routes", {"model_name": MODEL})
        assert c._api.resets == 1
        # The old code would now raise EFSM forever; the channel must recover.
        time.sleep(0.4)  # let the router drain the slow request
        for _ in range(5):
            assert c._call_api("get_epoch", {})["epoch"] == 0
        assert c.refresh_routes()
        assert c._send_heartbeat()
    finally:
        c.stop()


def test_heartbeat_not_blocked_by_api_channel(router):
    router.start()
    c = _client(router, 0, heartbeat_interval_s=0.05)
    assert c.start()
    try:
        # Simulate a long API call in another thread by holding its lock.
        with c._api._lock:
            hb0 = router._stats["heartbeats"]
            time.sleep(0.5)
            assert router._stats["heartbeats"] - hb0 >= 5
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# Incremental cache maintenance
# ---------------------------------------------------------------------------

def test_incremental_cache_updates(router):
    router.start()
    c = _client(router, 0)
    assert c.start()
    others = []
    try:
        time.sleep(0.2)  # SUB slow-joiner
        base_full = c.stats["full_refreshes"]
        for r in range(3):
            o = _client(router, 1, r)
            assert o.start()
            others.append(o)
        assert _wait(lambda: len(c.get_stage_peers(1, "never")) == 3)
        assert c.cache_in_sync()
        assert c.stats["full_refreshes"] == base_full  # no full refresh
        assert c.stats["incremental_applied"] >= 3

        # Address change -> ROUTE_UPDATED carries the route.
        assert others[1].update_my_address(["10.9.9.9:1"])
        assert _wait(lambda: c.get_stage_peers(1, "never")[1].addresses == ["10.9.9.9:1"])

        # Deregistration removes it.
        others[2].stop()
        others.pop()
        assert _wait(lambda: len(c.get_stage_peers(1, "never")) == 2)

        # 'auto' while in sync never hits the router.
        q0 = c.stats["stage_queries"]
        c.get_stage_peers(1)
        assert c.stats["stage_queries"] == q0
        assert c.stats["full_refreshes"] == base_full
        # Cache equals router truth.
        truth = sorted(r.lid.to_key() for r in router.route_manager.get_routes_for_model(MODEL))
        assert sorted(r.lid.to_key() for r in c.get_pipeline_peers()) == truth
    finally:
        for o in others:
            o.stop()
        c.stop()


def test_seq_gap_triggers_full_refresh(router):
    router.start()
    c = _client(router, 0)
    assert c.start()
    try:
        time.sleep(0.1)
        # Mutate router state behind the client's back, then deliver an event
        # whose seq skips ahead: the client must resync with a full refresh.
        router.route_manager.add_route(LogicalEndpointID(MODEL, 2, 0, 5), ["x"], [], 0)
        last = c._last_seq
        full0 = c.stats["full_refreshes"]
        ev = ClientEvent(type=CEventType.EPOCH_BUMPED, epoch=0, payload={}, seq=last + 5)
        c._handle_event(ev)
        assert c.stats["full_refreshes"] == full0 + 1
        assert c.stats["seq_gaps"] >= 1
        assert [r.lid.replica_idx for r in c.get_stage_peers(2, "never")] == [5]
    finally:
        c.stop()


def test_lag_detected_via_heartbeat_seq(router):
    router.start()
    c = _client(router, 0, heartbeat_interval_s=0.05)
    assert c.start()
    try:
        time.sleep(0.1)
        # Pretend we missed events: rewind processed seq below router seq.
        router.event_bus.publish(Event(type=EventType.EPOCH_BUMPED, epoch=0, payload={}))
        with c._sync_lock:
            c._last_seq = -100
        full0 = c.stats["full_refreshes"]
        assert _wait(lambda: c.stats["full_refreshes"] > full0, timeout=3.0)
        assert _wait(c.cache_in_sync)
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# Filtered queries
# ---------------------------------------------------------------------------

def test_filtered_stage_query(router):
    for st in range(4):
        for rep in range(5):
            router.route_manager.add_route(
                LogicalEndpointID(MODEL, st, 0, rep), [f"h{st}{rep}"], [1], 0)
    router.route_manager.add_route(LogicalEndpointID("other", 1), ["o"], [], 0)
    h = router._handle_request
    full = h({"method": "get_routes", "params": {"model_name": MODEL}})
    assert len(full["routes"]) == 20 and "seq" in full
    st = h({"method": "get_stage_routes", "params": {"model_name": MODEL, "stage_idx": 2}})
    assert [r["lid"]["replica_idx"] for r in st["routes"]] == list(range(5))
    assert all(r["lid"]["stage_idx"] == 2 for r in st["routes"])
    assert "last_heartbeat" not in st["routes"][0]  # compact
    multi = h({"method": "get_routes",
               "params": {"model_name": MODEL, "stage_idx": [0, 3]}})
    assert {r["lid"]["stage_idx"] for r in multi["routes"]} == {0, 3}
    assert "metadata" in multi["routes"][0]  # get_routes stays full by default
    assert "error" in h({"method": "get_stage_routes", "params": {"model_name": MODEL}})

    router.start()
    c = _client(router, 1, 9)
    assert c.start()
    try:
        peers = c.get_stage_peers(2, refresh="always")
        assert [p.lid.replica_idx for p in peers] == list(range(5))
        assert peers[0].capabilities == [TransportCapability.UCX_TCP]
    finally:
        c.stop()


def test_stage_query_falls_back_on_old_router(router):
    router.start()
    orig = router._handle_request

    def old_router(req):
        if req.get("method") == "get_stage_routes":
            return {"error": f"Unknown method: {req['method']}"}
        return orig(req)

    router._handle_request = old_router
    c = _client(router, 0)
    assert c.start()
    try:
        assert [p.lid.stage_idx for p in c.get_stage_peers(0, "always")] == [0]
        assert c._stage_query_supported is False
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# Failure detection timing
# ---------------------------------------------------------------------------

def test_detection_timing_small_timeouts(router):
    mm = router.member_manager
    mm.HEARTBEAT_TIMEOUT_S = 0.3
    mm.check_interval_s = 0.05
    mm.GRACE_PERIOD_S = 0.3
    router.start()
    live = _client(router, 0, heartbeat_interval_s=0.05)
    watcher = _client(router, 1, heartbeat_interval_s=0.05)
    assert live.start() and watcher.start()
    got = {}
    watcher.on_event(CEventType.ENDPOINT_UNHEALTHY,
                     lambda e: got.setdefault("unhealthy", time.time()))
    watcher.on_event(CEventType.ENDPOINT_DEREGISTERED,
                     lambda e: got.setdefault("removed", time.time()))
    watcher.on_event(CEventType.ENDPOINT_REGISTERED,
                     lambda e: got.setdefault("recovered", e.payload.get("recovered")))
    try:
        time.sleep(0.2)
        dead = LogicalEndpointID(MODEL, 2)
        router._handle_request({"method": "register", "params": {
            "lid": dead.to_dict(), "addresses": ["d"]}})
        t0 = time.time()
        assert _wait(lambda: "unhealthy" in got, timeout=2.0)
        detect = got["unhealthy"] - t0
        assert 0.25 <= detect < 0.6, detect
        # Healthy heartbeating clients are never flagged.
        for lid in (live.lid, watcher.lid):
            assert router.route_manager.get_route(
                LogicalEndpointID(lid.model_name, lid.stage_idx, 0, lid.replica_idx)
            ).state == RouteState.ACTIVE
        assert watcher.get_stage_peers(2, "never")[0].state == CRouteState.UNHEALTHY
        # Removal after grace is announced and reflected in the cache.
        assert _wait(lambda: "removed" in got, timeout=2.0)
        assert _wait(lambda: watcher.get_stage_peers(2, "never") == [])
    finally:
        live.stop()
        watcher.stop()


def test_recovery_after_unhealthy_is_announced(router):
    mm = router.member_manager
    mm.HEARTBEAT_TIMEOUT_S = 0.2
    mm.check_interval_s = 0.05
    router.start()
    watcher = _client(router, 1, heartbeat_interval_s=0.05)
    assert watcher.start()
    rec = []
    watcher.on_event(CEventType.ENDPOINT_REGISTERED,
                     lambda e: rec.append(e.payload) if e.payload.get("recovered") else None)
    try:
        time.sleep(0.1)
        lid = LogicalEndpointID(MODEL, 0, 0, 3)
        router._handle_request({"method": "register", "params": {
            "lid": lid.to_dict(), "addresses": ["z"]}})
        assert _wait(lambda: router.route_manager.get_route(lid).state == RouteState.UNHEALTHY)
        assert _wait(lambda: watcher.get_stage_peers(0, "never")
                     and watcher.get_stage_peers(0, "never")[0].state == CRouteState.UNHEALTHY)
        router._handle_request({"method": "heartbeat", "params": {"lid": lid.to_dict()}})
        assert _wait(lambda: rec)
        assert rec[0]["address_changed"] is False and rec[0]["is_reconnect"] is True
        assert _wait(lambda: watcher.get_stage_peers(0, "never")[0].state == CRouteState.ACTIVE)
    finally:
        watcher.stop()


def test_env_config(monkeypatch):
    monkeypatch.setenv("CONNEX_HEARTBEAT_INTERVAL_S", "1")
    monkeypatch.setenv("CONNEX_HEARTBEAT_TIMEOUT_S", "3")
    assert EPCV2ClientConfig().heartbeat_interval_s == 1.0
    assert EPCV2ClientConfig(heartbeat_interval_s=7).heartbeat_interval_s == 7
    from connex.front_router import MemberManager
    mm = MemberManager(RouteManager())
    assert mm.HEARTBEAT_TIMEOUT_S == 3.0 and mm.check_interval_s == 0.75
    monkeypatch.delenv("CONNEX_HEARTBEAT_TIMEOUT_S")
    mm = MemberManager(RouteManager())
    assert mm.HEARTBEAT_TIMEOUT_S == 30.0 and mm.check_interval_s == 5.0
