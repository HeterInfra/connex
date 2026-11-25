# SPDX-License-Identifier: Apache-2.0
"""Regression tests for control-plane and serving fixes (CPU only)."""
import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web

from connex.serving import fake_backend as fb
from connex.serving.proxy import Pool, Proxy


def run(coro):
    return asyncio.run(coro)


async def _serve(app):
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def _proxy(urls, **kw):
    proxy = Proxy(Pool(urls), **kw)
    runner, url = await _serve(proxy.app())
    return proxy, runner, url


# -- proxy: client disconnect stops generation -----------------------------------

def test_client_disconnect_cancels_backend_request():
    async def go():
        be = fb.FakeBackend(tpot_ms=50)
        bu = await be.start()
        proxy, runner, url = await _proxy([bu], probe_interval=0, idle_timeout=5)
        proxy.disconnect_poll_s = 0.05
        ep = proxy.pool.eps[bu]
        try:
            body = {"model": "fake", "prompt": "w5 w9", "max_tokens": 100}   # ~5 s
            async with aiohttp.ClientSession(trust_env=False) as s:
                with pytest.raises(asyncio.TimeoutError):
                    async with s.post(url + "/v1/completions", json=body,
                                      timeout=aiohttp.ClientTimeout(total=0.3)) as r:
                        await r.read()
            t0 = time.monotonic()
            while ep.inflight and time.monotonic() - t0 < 3.0:
                await asyncio.sleep(0.02)
            return ep.inflight, time.monotonic() - t0
        finally:
            await runner.cleanup()
            await be.stop()

    inflight, dt = run(go())
    assert inflight == 0 and dt < 1.0, (inflight, dt)


# -- proxy: last-resort replica is usable ---------------------------------------

def test_all_blacklisted_last_resort_serves_request():
    async def go():
        # a first token that is not already buffered when the wait starts
        bes = [fb.FakeBackend(ttft_ms=100) for _ in range(2)]
        urls = [await b.start() for b in bes]
        proxy, runner, url = await _proxy(urls, probe_interval=0)
        try:
            for u in urls:
                proxy.pool.mark_bad(proxy.pool.eps[u], "test")
            body = {"model": "fake", "prompt": "w5 w9", "max_tokens": 4, "stream": True}
            async with aiohttp.ClientSession(trust_env=False) as s:
                async with s.post(url + "/v1/completions", json=body) as r:
                    return r.status, await r.read()
        finally:
            await runner.cleanup()
            for b in bes:
                await b.stop()

    status, data = run(go())
    assert status == 200, data
    assert b"[DONE]" in data and b"error" not in data


# -- proxy: one blackholed replica does not delay detection of the others -------

def test_blackholed_replica_does_not_delay_other_probes():
    async def go():
        conns = []

        async def hold(reader, writer):   # accept, never answer
            conns.append(writer)

        hole = await asyncio.start_server(hold, "127.0.0.1", 0)
        hu = f"http://127.0.0.1:{hole.sockets[0].getsockname()[1]}"
        bes = [fb.FakeBackend() for _ in range(2)]
        urls = [await b.start() for b in bes]
        proxy, runner, url = await _proxy([hu] + urls, probe_interval=0.1)
        try:
            await asyncio.sleep(0.5)          # probe rounds are under way
            await bes[1].stop()
            t0 = time.monotonic()
            ep = proxy.pool.eps[urls[1]]
            while ep.healthy(time.monotonic()) and time.monotonic() - t0 < 6.0:
                await asyncio.sleep(0.02)
            return time.monotonic() - t0
        finally:
            await runner.cleanup()
            await bes[0].stop()
            for w in conns:
                w.close()
            hole.close()

    dt = run(go())
    assert dt < 1.5, dt


# -- proxy: non-streaming n > 1 keeps choices apart -----------------------------

async def _n2_backend():
    async def comp(req):
        b = await req.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)

        async def send(obj):
            await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        for k in range(3):
            for i in range(2):
                await send({"id": "cmpl-1", "object": "text_completion", "created": 1,
                            "model": "fake", "choices": [{"index": i, "text": f" c{i}t{k}",
                                                          "logprobs": None,
                                                          "finish_reason": "length" if k == 2 else None}]})
        await send({"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "fake",
                    "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 6,
                                             "total_tokens": 8}})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def models(req):
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_post("/v1/completions", comp)
    app.router.add_get("/v1/models", models)
    return await _serve(app)


def test_non_streaming_n2_aggregates_per_choice():
    async def go():
        brunner, bu = await _n2_backend()
        proxy, runner, url = await _proxy([bu], probe_interval=0)
        try:
            body = {"model": "fake", "prompt": "w5 w9", "max_tokens": 3, "n": 2}
            async with aiohttp.ClientSession(trust_env=False) as s:
                async with s.post(url + "/v1/completions", json=body) as r:
                    return r.status, await r.json()
        finally:
            await runner.cleanup()
            await brunner.cleanup()

    status, out = run(go())
    assert status == 200, out
    ch = out["choices"]
    assert [c["index"] for c in ch] == [0, 1]
    assert ch[0]["text"] == " c0t0 c0t1 c0t2" and ch[1]["text"] == " c1t0 c1t1 c1t2"
    assert ch[0]["finish_reason"] == ch[1]["finish_reason"] == "length"


# -- PD router: unhealthy events and probing of members that left the pool -----

def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _pd_router(pre_pool, dec_pool, **kw):
    from connex.serving.pd_router import PDRouter

    router = PDRouter(pre_pool, dec_pool, idle_timeout=5.0, max_attempts=4, **kw)
    runner, url = await _serve(router.app())
    return router, runner, url


async def _wait_healthy(pools, n, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if all(len(p.healthy_urls()) >= k for p, k in zip(pools, n)):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("members not probed")


def test_pd_unhealthy_event_cuts_hung_prefill():
    import zmq
    import zmq.asyncio

    from connex.pd.membership import PD_STAGE
    from connex.protocol import PDRole
    from connex.serving.pd_router import ElasticPool

    async def go():
        kv = {}
        hung = fb.FakePDBackend("prefill", kv)
        hung.fail = ("stall", 0)          # prefill hangs; /v1/models still answers
        ok = fb.FakePDBackend("prefill", kv)
        dec = fb.FakePDBackend("decode", kv)
        ok_u, hung_u, dec_u = [await b.start() for b in (ok, hung, dec)]
        port = _free_port()
        ctx = zmq.asyncio.Context.instance()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://127.0.0.1:{port}")
        pre, dpool = ElasticPool([ok_u, hung_u]), ElasticPool([dec_u])
        router, runner, url = await _pd_router(pre, dpool, probe_interval=0.05,
                                               front_router_events=f"tcp://127.0.0.1:{port}")
        router._members[(PDRole.PREFILL, 7)] = hung_u
        events = asyncio.ensure_future(router._failure_events_loop())
        try:
            await _wait_healthy([pre, dpool], [2, 1])
            await asyncio.sleep(0.3)       # SUB connected
            pre.rr = 0                     # next pick: the hung prefill

            async def announce():
                while not hung.requests:
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.2)
                ev = {"type": 2, "epoch": 0, "payload": {"lid": {
                    "model_name": "m", "stage_idx": PD_STAGE[PDRole.PREFILL], "shard_idx": 0,
                    "replica_idx": 7}}}
                await pub.send_multipart([b"connex.endpoint_unhealthy", json.dumps(ev).encode()])

            t = asyncio.ensure_future(announce())
            body = {"model": "fake", "prompt": "w5 w9 w13", "max_tokens": 4}
            t0 = time.monotonic()
            async with aiohttp.ClientSession(trust_env=False) as s:
                async with s.post(url + "/v1/completions", json=body,
                                  timeout=aiohttp.ClientTimeout(total=5)) as r:
                    st, out = r.status, await r.json()
            await t
            return st, out, time.monotonic() - t0, len(hung.requests), ok.prefills
        finally:
            events.cancel()
            await runner.cleanup()
            pub.close(0)
            for b in (ok, hung, dec):
                await b.stop()

    st, out, dt, n_hung, n_ok = run(go())
    assert st == 200, out
    assert n_hung == 1 and n_ok == 1
    assert dt < 2.0, dt


class _StuckablePD(fb.FakePDBackend):
    stuck = False

    async def models(self, req):
        if self.stuck:
            await asyncio.sleep(3600)
        return await super().models(req)


def test_pd_member_that_left_pool_is_still_probed_while_busy():
    from connex.serving.pd_router import ElasticPool

    async def go():
        kv = {}
        hung = _StuckablePD("prefill", kv)
        hung.fail = ("stall", 0)
        ok = fb.FakePDBackend("prefill", kv)
        dec = fb.FakePDBackend("decode", kv)
        ok_u, hung_u, dec_u = [await b.start() for b in (ok, hung, dec)]
        pre, dpool = ElasticPool([]), ElasticPool([dec_u])
        pre.set_dynamic([ok_u, hung_u])
        router, runner, url = await _pd_router(pre, dpool, probe_interval=0.05)
        try:
            await _wait_healthy([pre, dpool], [2, 1])
            pre.rr = 0

            async def leave_and_hang():
                while not hung.requests:
                    await asyncio.sleep(0.02)
                # the member leaves the pool (e.g. unhealthy at the front-router),
                # then its host stops answering altogether
                pre.set_dynamic([ok_u])
                await asyncio.sleep(0.2)
                hung.stuck = True

            t = asyncio.ensure_future(leave_and_hang())
            body = {"model": "fake", "prompt": "w5 w9 w13", "max_tokens": 4}
            async with aiohttp.ClientSession(trust_env=False) as s:
                async with s.post(url + "/v1/completions", json=body,
                                  timeout=aiohttp.ClientTimeout(total=12)) as r:
                    st, out = r.status, await r.json()
            await t
            return st, out, ok.prefills
        finally:
            await runner.cleanup()
            for b in (ok, hung, dec):
                await b.stop()

    st, out, n_ok = run(go())
    assert st == 200, out
    assert n_ok == 1


# -- front-router ------------------------------------------------------------------

class _CapturePub:
    def __init__(self):
        self.sent = []

    def send_multipart(self, parts):
        self.sent.append((parts[0].decode(), json.loads(parts[1])))

    def close(self, linger=0):
        pass


def _fr():
    from connex.front_router import FrontRouter, FrontRouterConfig

    fr = FrontRouter(FrontRouterConfig())
    pub = _CapturePub()
    fr.event_bus._pub_socket = pub
    fr.event_bus._running = True
    return fr, pub


def _lid(model="m", stage=0, replica=0):
    return {"model_name": model, "stage_idx": stage, "shard_idx": 0, "replica_idx": replica}


def _age(rm, lid, seconds):
    ts = time.time() - seconds
    rm.get_route(lid).last_heartbeat = ts
    rm._touch_locked(lid.to_key(), ts)


def _register(fr, **kw):
    return fr._handle_request({"method": "register", "params": {
        "lid": _lid(**kw), "addresses": ["tcp://h:1"], "capabilities": [0]}})


def test_model_scoped_routes_do_not_scan_other_models():
    from connex.front_router import LogicalEndpointID, RouteManager, TransportCapability

    rm = RouteManager()
    for i in range(50000):
        rm.add_route(LogicalEndpointID("big", i % 8, 0, i), ["a"], [TransportCapability(0)], 0)
    rm.add_route(LogicalEndpointID("small", 0, 0, 0), ["a"], [TransportCapability(0)], 0)
    t0 = time.perf_counter()
    for _ in range(50):
        out = rm.get_routes_for_model("small")
    per_call = (time.perf_counter() - t0) / 50
    assert [r.lid.model_name for r in out] == ["small"]
    assert per_call < 1e-3, per_call       # a scan of 50k routes takes several ms
    # the index follows removals and expiry
    rm.remove_route(LogicalEndpointID("small", 0, 0, 0))
    assert rm.get_routes_for_model("small") == [] and "small" not in rm._by_model
    assert len(rm.get_routes_for_model("big")) == 50000


def test_register_reply_still_carries_model_route_table():
    fr, _ = _fr()
    _register(fr, replica=0)
    _register(fr, model="other")
    r = _register(fr, replica=1)
    assert r["success"] and sorted(x["lid"]["replica_idx"] for x in r["route_table"]) == [0, 1]
    assert all(x["lid"]["model_name"] == "m" for x in r["route_table"])
    assert "last_heartbeat" in r["route_table"][0]    # full form, as before


def test_late_unhealthy_callback_does_not_follow_recovery():
    from connex.front_router import LogicalEndpointID

    fr, pub = _fr()
    _register(fr)
    lid = LogicalEndpointID.from_dict(_lid())
    rm = fr.route_manager
    _age(rm, lid, 10)
    unhealthy, _ = rm.collect_expired(3.0, 10.0)      # health thread marks it ...
    assert [r.lid for r in unhealthy] == [lid]
    # ... the API thread serves its heartbeat (recovery, REGISTERED) ...
    fr._handle_request({"method": "heartbeat", "params": {"lid": _lid()}})
    # ... and only then does the health thread run its callback
    fr._on_endpoint_unhealthy(lid)
    topics = [t for t, _ in pub.sent]
    assert topics[-1] == "connex.endpoint_registered", topics
    assert "connex.endpoint_unhealthy" not in topics


def test_late_expiry_callback_does_not_follow_reregistration():
    from connex.front_router import LogicalEndpointID, RouteState

    fr, pub = _fr()
    _register(fr)
    lid = LogicalEndpointID.from_dict(_lid())
    rm = fr.route_manager
    _age(rm, lid, 100)
    rm.get_route(lid).state = RouteState.UNHEALTHY
    _, removed = rm.collect_expired(3.0, 10.0)
    assert [r.lid for r in removed] == [lid]
    _register(fr)                                    # endpoint comes back
    fr._on_endpoint_expired(lid)                     # late callback
    topics = [t for t, _ in pub.sent]
    assert topics[-1] == "connex.endpoint_registered", topics
    assert rm.get_route(lid) is not None


def test_stale_migrating_route_expires():
    from connex.front_router import LogicalEndpointID, RouteState

    fr, _ = _fr()
    _register(fr)
    lid = LogicalEndpointID.from_dict(_lid())
    rm = fr.route_manager
    rm.update_route_state(lid, RouteState.MIGRATING)
    _age(rm, lid, 5)
    unhealthy, _ = rm.collect_expired(3.0, 10.0)
    assert [r.lid for r in unhealthy] == [lid]
    _age(rm, lid, 100)
    _, removed = rm.collect_expired(3.0, 10.0)
    assert [r.lid for r in removed] == [lid]


def test_migration_never_cut_over_is_aborted_and_trimmed():
    from connex.front_router import LogicalEndpointID, RouteState

    fr, pub = _fr()
    _register(fr)
    lid = LogicalEndpointID.from_dict(_lid())
    mo = fr.migration_orchestrator
    m = mo.prepare_migration(lid, "tcp://new:1")
    assert fr.route_manager.get_route(lid).state == RouteState.MIGRATING
    now = time.time()
    assert mo.reap(now + 1) == []
    assert mo.reap(now + mo.SHADOW_TIMEOUT_S + mo.CUTOVER_TIMEOUT_S + 1) == [m.migration_id]
    assert fr.route_manager.get_route(lid).state == RouteState.ACTIVE
    assert pub.sent[-1][0] == "connex.migration_failed"
    assert m.migration_id in mo._pending
    mo.reap(time.time() + mo.FINISHED_TTL_S + 1)
    assert mo._pending == {}


# -- EPC client: re-registration backoff --------------------------------------------

def test_reregistration_backs_off_with_api_timeout():
    from connex.epc_client import EPCV2Client, EPCV2ClientConfig

    c = EPCV2Client(EPCV2ClientConfig(heartbeat_interval_s=0.1, api_timeout_ms=4321))
    c._running = True
    calls = []

    def call_api(method, params, channel=None, timeout_ms=None):
        calls.append((method, timeout_ms))
        if method == "heartbeat":
            return {"success": False, "epoch": 0}
        raise RuntimeError("API error: timeout")

    c._call_api = call_api
    for _ in range(5):
        assert c._send_heartbeat() is False
    regs = [t for m, t in calls if m == "register"]
    assert len(regs) == 1, calls
    assert regs[0] == 4321


# -- directory client cache ----------------------------------------------------------

def test_directory_lookup_cache_survives_unrelated_events():
    from connex.directory import DirectoryClient
    from connex.directory_server import DirectoryServer

    srv = DirectoryServer(f"127.0.0.1:{_free_port()}")
    srv.start()
    made = []

    def client(stage, addrs=None):
        c = DirectoryClient(srv.bind_addr, "m", stage_idx=stage, local_addresses=addrs)
        c.start()
        made.append(c)
        return c

    def wait_epoch(c):
        t0 = time.monotonic()
        while c.get_current_epoch() < srv.epoch and time.monotonic() - t0 < 3:
            time.sleep(0.02)
        assert c.get_current_epoch() >= srv.epoch

    try:
        s0, s1, s2 = client(0, ["127.0.0.1:6000"]), client(1), client(2, ["127.0.0.1:6002"])
        assert s1.start_event_listener()
        assert s0.register()
        wait_epoch(s1)
        calls = []
        orig = s1._control_call

        def counted(msg, timeout_ms=5000):
            calls.append(msg.get("type"))
            return orig(msg, timeout_ms)

        s1._control_call = counted
        assert s1.lookup_stage(0).primary_address == "127.0.0.1:6000"
        assert s2.register()                      # unrelated membership change
        wait_epoch(s1)
        for _ in range(5):
            assert s1.lookup_stage(0).primary_address == "127.0.0.1:6000"
        assert calls.count("lookup") == 1, calls
        # a change of the looked-up LID itself is still seen
        assert s0.register(["127.0.0.1:7000"])
        wait_epoch(s1)
        assert s1.lookup_stage(0).primary_address == "127.0.0.1:7000"
    finally:
        for c in made:
            c.stop()
        srv.stop()


# -- elastic ZmqTransport ---------------------------------------------------------------

def test_zmq_transport_dead_peer_does_not_stall_other_sends():
    import threading

    from connex.elastic import ZmqTransport

    tx, rx = ZmqTransport(), ZmqTransport()
    try:
        live = tx.connect_sync("127.0.0.1", rx.local_port)
        dead = tx.connect_sync("127.0.0.1", _free_port())   # nobody listening
        stop = threading.Event()

        def flood():   # fills the dead peer's queue; then each send blocks SNDTIMEO
            while not stop.is_set():
                try:
                    tx.send_sync(dead, b"x")
                except Exception:
                    pass

        th = threading.Thread(target=flood, daemon=True)
        th.start()
        time.sleep(0.5)
        worst = 0.0
        for _ in range(5):
            t0 = time.monotonic()
            tx.send_sync(live, b"y")
            worst = max(worst, time.monotonic() - t0)
            assert rx.recv_sync(2000) == b"y"
        stop.set()
        th.join(5)
        return_worst = worst
    finally:
        tx.stop_sync()
        rx.stop_sync()
    assert return_worst < 0.5, return_worst


# -- session proxy admin endpoints and re-prefill timeout ------------------------------

def test_session_admin_requires_loopback_or_token():
    from types import SimpleNamespace

    from connex.agents.session import SessionProxy

    p = SessionProxy(Pool([]), probe_interval=0)

    class Req(SimpleNamespace):
        async def json(self):
            return {"target": "http://x", "url": "http://x"}

    req = lambda remote, auth="": Req(remote=remote, headers={"Authorization": auth},
                                      match_info={"sid": "s"})

    async def call(h, r):
        try:
            return (await h(r)).status
        except web.HTTPException as e:
            return e.status

    async def go():
        out = [await call(p.handle_migrate, req("10.0.0.5")),
               await call(p.handle_drain, req("10.0.0.5"))]
        p.admin_token = "s3cret"
        out.append(await call(p.handle_migrate, req("10.0.0.5", "Bearer s3cret")))
        out.append(await call(p.handle_migrate, req("127.0.0.1")))
        return out

    st = run(go())
    assert st[:2] == [403, 403], st
    assert st[2] != 403 and st[3] != 403, st     # 400: unknown session, but allowed


def test_reprefill_mover_times_out_on_hung_target():
    from connex.agents.session import ReprefillMover, Session, SessionProxy

    async def go():
        async def stuck(req):
            await asyncio.sleep(3600)

        app = web.Application()
        app.router.add_post("/v1/completions", stuck)
        brunner, bu = await _serve(app)
        p = SessionProxy(Pool([bu]), probe_interval=0)
        await p._startup()
        s = Session("a")
        s.kind = "completions"
        s.context = {"model": "fake", "prompt": "w5"}
        mover = ReprefillMover()
        mover.timeout_s = 0.3
        t0 = time.monotonic()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(mover.move(p, s, None, bu), 5)
            return time.monotonic() - t0
        finally:
            await p._cleanup()
            await brunner.cleanup()

    assert run(go()) < 2.0
