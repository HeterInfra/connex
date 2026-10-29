# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""PD router: prefill then decode, over elastic prefill and decode pools.

An OpenAI-API front end for prefill-decode disaggregation with the Connex
KV connector (``connex.adapters.vllm.pd_connector``; the SGLang backend uses
SGLang's own bootstrap flow). For each request it

1. sends the prompt to a prefill instance with ``max_tokens=1`` and
   ``kv_transfer_params={"do_remote_decode": true, "transfer_id": ...}``,
   and takes the ``kv_transfer_params`` of the response;
2. streams the request from a decode instance with those params; the decode
   instance pulls the prompt KV from the prefill instance.

Pools are elastic. Members come from ``--prefill``/``--decode`` URLs, files
re-read every second, and/or the Connex front-router (``--front-router``,
instances registered by the connector with their PD role). A new member
takes traffic once a probe of ``/v1/models`` succeeds; a member that leaves
(deregisters, is removed from a file, or ``POST /pd/drain``) takes no new
requests and finishes the ones it has.

Failures, on top of the resume proxy (``connex.serving.proxy``):

* **prefill fails** (connect error, HTTP 5xx, dies while prefilling): the
  prefill is retried on another prefill instance.
* **decode fails before its first token** while the decode instance is alive
  (e.g. its KV pull failed because the prefill died during the transfer, with
  ``kv_load_failure_policy="fail"``): the request is prefilled again on
  another prefill instance and sent to a decode instance.
* **decode dies mid-generation** after k tokens: the request resumes on
  another decode instance with the prompt plus the k generated token ids and
  ``max_tokens - k``. If the prefill instance still holds the request's KV
  (it does until the decode releases it, or ``retain_s`` passes) the new
  decode instance pulls the prompt KV again and computes only the k tokens;
  otherwise (prefill gone, hold expired, or text-based chat resume, whose
  re-templated prefix need not tokenise identically) it is prefilled again.

    python -m connex.serving.pd_router --port 8000 \\
        --prefill http://p0:8100,http://p1:8101 --decode http://d0:8200 \\
        --front-router tcp://router:15555 --model meta-llama/Meta-Llama-3-8B
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid

import aiohttp
from aiohttp import web

from connex.serving.proxy import (
    RETRYABLE,
    Pool,
    Proxy,
    Resumer,
    StreamBroken,
    UpstreamFailed,
)

log = logging.getLogger("connex.pd_router")

UNPROBED = 1e18


class ElasticPool(Pool):
    """Pool whose members are the union of static URLs and a dynamic source;
    members join unprobed (no traffic until a probe succeeds)."""

    def __init__(self, urls=(), endpoints_file=None, blacklist_s=5.0, probe_new=True):
        self.static = [u.strip().rstrip("/") for u in urls if u.strip()]
        self.dynamic: list[str] = []
        self.drained: set[str] = set()
        self.probe_new = probe_new
        super().__init__(self.static, endpoints_file, blacklist_s)

    def set_urls(self, urls):
        urls = [u.strip().rstrip("/") for u in urls if u.strip()]
        for u in urls:
            if u not in self.eps:
                ep = self._new(u)
                self.eps[u] = ep
        new_order = [u for u in dict.fromkeys(urls) if u not in self.drained]
        if new_order != self.order:
            gone = set(self.order) - set(new_order)
            added = set(new_order) - set(self.order)
            if gone or added:
                log.info("pool change: +%s -%s", sorted(added), sorted(gone))
        self.order = new_order

    def _new(self, url):
        from connex.serving.proxy import Endpoint

        ep = Endpoint(url)
        if self.probe_new:
            ep.bad_until = UNPROBED   # joins once a probe succeeds; waiters are not woken
        return ep

    def set_dynamic(self, urls):
        self.dynamic = list(urls)
        self.refresh()

    def refresh(self):
        file_urls = []
        if self.file:
            try:
                file_urls = [x.strip() for x in open(self.file) if x.strip()]
            except OSError:
                pass
        self.set_urls(self.static + file_urls + self.dynamic)

    def reload(self):
        self.refresh()

    def drain(self, url):
        url = url.rstrip("/")
        self.drained.add(url)
        self.order = [u for u in self.order if u != url]

    def undrain(self, url):
        self.drained.discard(url.rstrip("/"))
        self.refresh()

    def healthy_urls(self):
        now = time.monotonic()
        return [u for u in self.order if self.eps[u].healthy(now)]


class PrefillFailed(UpstreamFailed):
    pass


class PDResumer(Resumer):
    def __init__(self, proxy, kind, body):
        super().__init__(proxy, kind, body)
        self.kv = None            # kv_transfer_params from the prefill instance
        self.kv_ep = None         # prefill endpoint that holds them
        self.kv_tokens = 0        # completion tokens emitted when the KV was produced
        self.prefills = 0
        self.avoid_prefill = None # prefill whose KV hand-off just failed
        self.spare_endpoint = False
        self.prefill_task = None  # SGLang: the concurrent prefill request
        self.abort_first = None

    async def build(self, ep):
        b = await super().build(ep)
        if self.p.engine == "sglang":
            await self.p.sglang_dispatch(self, b)
            return b
        if self.kv is not None:
            reuse = self.kv_ep is not None and self.kv_ep.healthy(time.monotonic())
            # a resumed request may reuse the prompt KV only if its new prompt
            # extends the old one token for token (token-id resume)
            if self.k != self.kv_tokens and not (self.use_ids and not self.chat):
                reuse = False
            if not reuse:
                self.kv = None
            elif self.k > 0:
                self.p.stats["kv_repulls"] += 1   # resumed: decode pulls the held KV again
        if self.kv is None:
            self.kv, self.kv_ep = await self.p.prefill(self, b)
            self.kv_tokens = self.k
        b["kv_transfer_params"] = dict(self.kv)
        return b


class PDRouter(Proxy):
    resumer_cls = PDResumer

    def __init__(self, prefill: ElasticPool, decode: ElasticPool, front_router=None,
                 model=None, membership_interval=0.5, prefill_timeout=None,
                 engine="vllm", bootstrap=None, bootstrap_port=8998, front_router_events=None,
                 **kw):
        super().__init__(decode, **kw)
        if engine not in ("vllm", "sglang"):
            raise ValueError(f"engine must be vllm or sglang, got {engine!r}")
        self.engine = engine
        self.bootstrap = dict(bootstrap or {})   # prefill URL -> "host:port"
        self.bootstrap_port = bootstrap_port
        self.prefill_pool = prefill
        self.front_router = front_router
        self.model = model
        self.membership_interval = membership_interval
        self.prefill_timeout = prefill_timeout
        if front_router and not front_router_events:
            front_router_events = front_router.rsplit(":", 1)[0] + ":15560"
        self.front_router_events = front_router_events
        import os

        self.admin_token = os.getenv("CONNEX_PD_ADMIN_TOKEN") or None
        self._members: dict[tuple, str] = {}     # (role, replica) -> url, from fetch_pools
        self.stats.update({"prefills": 0, "prefill_retries": 0, "kv_reprefills": 0,
                           "kv_repulls": 0, "membership_errors": 0, "failure_events": 0})

    # -- lifecycle --------------------------------------------------------------
    def app(self):
        app = super().app()
        app.router.add_get("/pd/pools", self.handle_pools)
        app.router.add_post("/pd/drain", self.handle_drain)
        app.router.add_post("/pd/undrain", self.handle_undrain)
        return app

    async def _startup(self, app=None):
        await super()._startup(app)
        if self.prefill_pool.file:
            self._bg.append(asyncio.create_task(self._reload_prefill_loop()))
        if self.front_router:
            self._bg.append(asyncio.create_task(self._membership_loop()))
            if self.front_router_events:
                self._bg.append(asyncio.create_task(self._failure_events_loop()))

    async def _reload_prefill_loop(self):
        while True:
            await asyncio.sleep(1.0)
            self.prefill_pool.reload()

    async def _membership_loop(self):
        from connex.pd.membership import fetch_pools
        from connex.protocol import PDRole

        loop = asyncio.get_running_loop()
        while True:
            pools = await loop.run_in_executor(None, fetch_pools, self.front_router, self.model)
            if pools is None:
                self.stats["membership_errors"] += 1
            else:
                for m in pools[PDRole.PREFILL]:
                    if m.kv_addr:
                        self.bootstrap[m.url] = m.kv_addr
                for role in (PDRole.PREFILL, PDRole.DECODE):
                    for m in pools[role]:
                        self._members[(role, m.replica)] = m.url
                self.prefill_pool.set_dynamic([m.url for m in pools[PDRole.PREFILL]])
                self.pool.set_dynamic([m.url for m in pools[PDRole.DECODE]])
            await asyncio.sleep(self.membership_interval)

    async def _failure_events_loop(self):
        """Front-router failure events -> mark the instance dead at once.

        A failed group (a TP rank's lease lapsed, or a peer's data plane saw a
        rank die) is announced as deregistered with a failure reason. The
        instance's in-flight streams and prefills are cut and resume on
        another instance right away, instead of waiting for the engine to
        exit or the idle timeout. A graceful leave (no reason, or
        ``group_left``) is not a failure: its requests finish."""
        import zmq
        import zmq.asyncio

        from connex.pd.membership import STAGE_ROLE

        ctx = zmq.asyncio.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.SUBSCRIBE, b"connex.endpoint_deregistered")
        sub.connect(self.front_router_events)
        try:
            while True:
                try:
                    _, data = await sub.recv_multipart()
                    ev = json.loads(data)
                except (ValueError, zmq.ZMQError):
                    continue
                payload = ev.get("payload") or {}
                reason = payload.get("reason")
                if not reason or reason in ("group_left", "expired"):
                    continue
                lid = payload.get("lid") or {}
                role = STAGE_ROLE.get(lid.get("stage_idx"))
                url = self._members.get((role, lid.get("replica_idx")))
                if url is None:
                    continue
                pool = self.prefill_pool if role == "prefill" else self.pool
                ep = pool.eps.get(url)
                if ep is not None and ep.healthy(time.monotonic()):
                    self.stats["failure_events"] += 1
                    pool.mark_bad(ep, f"front-router: {reason}")
        finally:
            sub.close(0)

    async def _probe_loop(self):
        while True:
            eps = [e for p in (self.pool, self.prefill_pool)
                   for e in list(p.eps.values()) if e.url in p.order]
            await asyncio.gather(*(self._probe_any(e) for e in eps), return_exceptions=True)
            await asyncio.sleep(self.probe_interval)

    async def _probe_any(self, ep):
        pool = self.prefill_pool if ep.url in self.prefill_pool.eps else self.pool
        if ep.bad_until >= UNPROBED:     # new member: one success is enough
            if await self._alive(ep):
                log.info("member %s is up", ep.url)
                pool.mark_good(ep)
            return
        if await self._alive(ep) or await self._alive(ep):
            pool.mark_good(ep)
        else:
            pool.mark_bad(ep, "probe failed twice")

    # -- prefill ------------------------------------------------------------------
    async def prefill(self, st, body):
        """Run the prefill of ``body`` on some prefill instance; returns
        (kv_transfer_params, endpoint). Retries on other instances."""
        path = "/v1/chat/completions" if st.chat else "/v1/completions"
        req = {k: v for k, v in body.items()
               if k not in ("stream_options", "logprobs", "top_logprobs",
                            "return_tokens_as_token_ids", "min_tokens", "kv_transfer_params")}
        req["stream"] = False
        req["max_tokens"] = 1
        if "max_completion_tokens" in req:
            req["max_completion_tokens"] = 1
        tried = [st.avoid_prefill.url] if st.avoid_prefill is not None else []
        st.avoid_prefill = None
        last = "no prefill instances"
        pool = self.prefill_pool
        for attempt in range(max(self.max_attempts, len(pool.order) + 1)):
            ep = pool.pick(exclude=tried) or pool.pick()
            if ep is None:
                break
            if attempt:
                self.stats["prefill_retries"] += 1
            tid = uuid.uuid4().hex
            req["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False,
                                         "transfer_id": tid}
            ep.inflight += 1
            st.prefills += 1
            try:
                post = self.session.post(ep.url + path, json=req,
                                         timeout=aiohttp.ClientTimeout(total=self.prefill_timeout))
                rtask = asyncio.ensure_future(self._post_json(post))
                # cut the wait short if the instance is declared dead meanwhile
                # (not if it already was: then it is our last resort)
                watch = {rtask}
                if ep.healthy(time.monotonic()):
                    watch.add(asyncio.ensure_future(ep.dead.wait()))
                try:
                    done, _ = await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in watch - {rtask}:
                        t.cancel()
                if rtask not in done:
                    rtask.cancel()
                    raise StreamBroken("prefill instance died")
                status, data = rtask.result()
                if status != 200:
                    raise StreamBroken(f"prefill HTTP {status}: {str(data)[:200]}")
                kv = (data or {}).get("kv_transfer_params")
                if not kv:
                    raise StreamBroken("prefill response without kv_transfer_params "
                                       "(is the instance running with kv_role=kv_producer?)")
                self.stats["prefills"] += 1
                if st.prompt_tokens is None:
                    st.prompt_tokens = ((data.get("usage") or {}).get("prompt_tokens"))
                return kv, ep
            except (StreamBroken, ValueError, *RETRYABLE) as e:
                last = f"{type(e).__name__}: {e}"
                pool.mark_bad(ep, last)
                tried.append(ep.url)
                log.warning("prefill on %s failed: %s", ep.url, last)
            finally:
                ep.inflight -= 1
        raise PrefillFailed(f"prefill failed: {last}", st)

    async def _post_json(self, post):
        async with post as r:
            try:
                return r.status, await r.json(content_type=None)
            except ValueError:
                return r.status, None

    # -- SGLang: prefill and decode run concurrently, paired by a bootstrap room --
    def _bootstrap_of(self, url):
        addr = self.bootstrap.get(url)
        if addr:
            h, p = addr.rsplit(":", 1)
            return h, int(p)
        from urllib.parse import urlparse

        return urlparse(url).hostname, self.bootstrap_port

    async def sglang_dispatch(self, st, b):
        pool = self.prefill_pool
        avoid = [st.avoid_prefill.url] if st.avoid_prefill is not None else []
        st.avoid_prefill = None
        pep = pool.pick(exclude=avoid) or pool.pick()
        if pep is None:
            raise PrefillFailed("no prefill instances", st)
        host, port = self._bootstrap_of(pep.url)
        b.update(bootstrap_host=host, bootstrap_port=port,
                 bootstrap_room=uuid.uuid4().int & ((1 << 63) - 1))
        path = "/v1/chat/completions" if st.chat else "/v1/completions"
        pb = {k: v for k, v in b.items() if k != "stream_options"}
        pb["stream"] = False
        st.kv_ep = pep
        st.prefills += 1
        self.stats["prefills"] += 1

        async def run():
            pep.inflight += 1
            try:
                async with self.session.post(
                        pep.url + path, json=pb,
                        timeout=aiohttp.ClientTimeout(total=self.prefill_timeout)) as r:
                    await r.read()
                    if r.status != 200:
                        return f"prefill HTTP {r.status}"
                return None
            except (*RETRYABLE,) as e:
                pool.mark_bad(pep, f"{type(e).__name__}: {e}")
                return f"prefill {type(e).__name__}"
            finally:
                pep.inflight -= 1

        st.prefill_task = asyncio.ensure_future(run())

        async def abort():
            err = await asyncio.shield(st.prefill_task)
            if err is None:
                await asyncio.Event().wait()   # prefill fine: never abort
            return err

        st.abort_first = abort

    # -- decode -------------------------------------------------------------------
    async def _attempt(self, st, ep, path):
        k0 = st.k
        try:
            async for c in super()._attempt(st, ep, path):
                yield c
        except (StreamBroken, ValueError, KeyError, *RETRYABLE):
            pt = st.prefill_task
            if self.engine == "sglang":
                if pt is not None and pt.done() and not pt.cancelled() and pt.result():
                    # the paired prefill failed: not the decode instance's fault
                    st.avoid_prefill = st.kv_ep
                    st.spare_endpoint = st.k == k0
                    self.stats["prefill_retries"] += 1
                elif st.k > k0:
                    self.stats["kv_reprefills"] += 1
                raise
            if st.k == k0 and st.kv is not None and await self._alive(ep):
                # decode alive but nothing generated: the KV hand-off failed
                # (prefill died mid-transfer, hold expired). Prefill again.
                st.avoid_prefill = st.kv_ep
                st.kv = None
                st.spare_endpoint = True
                self.stats["kv_reprefills"] += 1
                log.warning("decode %s failed before its first token; re-prefilling", ep.url)
            raise

    # -- admin --------------------------------------------------------------------
    async def handle_pools(self, request):
        now = time.monotonic()

        def desc(p):
            return [{"url": u, "healthy": p.eps[u].healthy(now),
                     "probed": p.eps[u].bad_until < UNPROBED, "inflight": p.eps[u].inflight}
                    for u in p.order] + [{"url": u, "drained": True,
                                          "inflight": p.eps[u].inflight}
                                         for u in p.drained if u in p.eps]

        return web.json_response({"prefill": desc(self.prefill_pool), "decode": desc(self.pool)})

    def _admin_allowed(self, request) -> bool:
        """Pool changes are admin operations: allowed from loopback, or with
        ``Authorization: Bearer <token>`` when ``CONNEX_PD_ADMIN_TOKEN`` (or
        ``--admin-token``) is set. Without that a router bound to 0.0.0.0
        would let any client drain every instance."""
        import hmac

        tok = self.admin_token
        if tok:
            got = request.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), f"Bearer {tok}".encode()):
                return True
        remote = (request.remote or "").split("%", 1)[0]
        return remote in ("127.0.0.1", "::1", "localhost") or remote.startswith("::ffff:127.")

    async def _drain_target(self, request):
        if not self._admin_allowed(request):
            raise web.HTTPForbidden(text="pool admin needs loopback or CONNEX_PD_ADMIN_TOKEN")
        body = await request.json()
        return body.get("url", ""), body.get("role")

    async def handle_drain(self, request):
        url, role = await self._drain_target(request)
        for name, p in (("prefill", self.prefill_pool), ("decode", self.pool)):
            if role in (None, name):
                p.drain(url)
        return web.json_response({"ok": True})

    async def handle_undrain(self, request):
        url, role = await self._drain_target(request)
        for name, p in (("prefill", self.prefill_pool), ("decode", self.pool)):
            if role in (None, name):
                p.undrain(url)
        return web.json_response({"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prefill", default="", help="comma-separated prefill base URLs")
    ap.add_argument("--decode", default="", help="comma-separated decode base URLs")
    ap.add_argument("--prefill-file", default=None, help="one URL per line, re-read every 1 s")
    ap.add_argument("--decode-file", default=None, help="one URL per line, re-read every 1 s")
    ap.add_argument("--front-router", default=None,
                    help="tcp://host:port of the Connex front-router (PD pool membership)")
    ap.add_argument("--model", default=None, help="model name the instances register under")
    ap.add_argument("--front-router-events", default=None,
                    help="tcp://host:port of the front-router's event feed (default: the "
                         "front-router host, port 15560); failure events cut streams at once")
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm",
                    help="vllm: prefill then decode with kv_transfer_params; sglang: prefill "
                         "and decode concurrently, paired by a bootstrap room")
    ap.add_argument("--bootstrap", default="",
                    help="SGLang: comma-separated prefill_url=host:bootstrap_port")
    ap.add_argument("--bootstrap-port", type=int, default=8998,
                    help="SGLang: bootstrap port of prefill instances not listed in --bootstrap")
    ap.add_argument("--membership-interval", type=float, default=0.5)
    ap.add_argument("--idle-timeout", type=float, default=5.0)
    ap.add_argument("--first-token-timeout", type=float, default=None)
    ap.add_argument("--prefill-timeout", type=float, default=None)
    ap.add_argument("--connect-timeout", type=float, default=2.0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--blacklist-s", type=float, default=5.0)
    ap.add_argument("--probe-interval", type=float, default=0.5)
    ap.add_argument("--admin-token", default=None,
                    help="bearer token for POST /pd/drain and /pd/undrain from non-loopback "
                         "clients (default: CONNEX_PD_ADMIN_TOKEN; unset = loopback only)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    if a.front_router and not a.model:
        ap.error("--front-router needs --model")
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    pre = ElasticPool([u for u in a.prefill.split(",") if u], a.prefill_file, a.blacklist_s)
    dec = ElasticPool([u for u in a.decode.split(",") if u], a.decode_file, a.blacklist_s)
    router = PDRouter(pre, dec, front_router=a.front_router, model=a.model,
                      membership_interval=a.membership_interval, prefill_timeout=a.prefill_timeout,
                      engine=a.engine, bootstrap_port=a.bootstrap_port,
                      front_router_events=a.front_router_events,
                      bootstrap=dict(x.rsplit("=", 1) for x in a.bootstrap.split(",") if x),
                      idle_timeout=a.idle_timeout, connect_timeout=a.connect_timeout,
                      max_attempts=a.max_attempts, probe_interval=a.probe_interval,
                      first_token_timeout=a.first_token_timeout)
    if a.admin_token:
        router.admin_token = a.admin_token
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    web.run_app(router.app(), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
