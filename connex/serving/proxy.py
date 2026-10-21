# SPDX-License-Identifier: Apache-2.0
"""Thin, optional data-path proxy in front of the stage-0 OpenAI API servers.

Connex has no request router on the data path: clients pick a stage-0 replica
themselves. When that replica dies, its in-flight requests are lost. This
proxy hides that from the client:

* **Replica choice.** Endpoints come from ``--endpoints`` and/or an endpoints
  file (re-read every second, same format as ``trace_load.py``). A background
  prober GETs ``/v1/models`` on a fresh connection every ``--probe-interval``
  seconds; a failed probe, connect error or broken stream blacklists the
  replica for ``--blacklist-s``. The least-loaded healthy replica wins.
* **Failure before the first token** (connect refused, reset, HTTP 5xx, or the
  replica getting blacklisted while we wait for prefill): transparently retry
  on another replica. No idle timeout runs before the first token, so a slow
  prefill never trips it; instead the wait is cut short only when the replica
  itself is seen dead (probe or another request failing on it).
* **Failure after k > 0 tokens** (reset, EOF without ``[DONE]``, or no chunk
  for ``--idle-timeout`` seconds): resume on another replica by re-sending the
  prompt plus the k tokens already emitted, with ``max_tokens`` (and
  ``min_tokens``) reduced by k, and splice the new stream onto the old one.
  Completions resume from token ids (``prompt = prompt_ids + generated_ids``,
  no re-tokenisation; ids come from ``logprobs`` + ``return_tokens_as_token_ids``
  which the proxy injects and strips again). Chat resumes from text through
  vLLM's ``continue_final_message``. The client sees one stream: same ``id``,
  continuous text, and a usage block with the original prompt_tokens and the
  total completion_tokens.

Every backend call is made with ``stream=true`` (non-streaming client requests
are aggregated by the proxy), so one code path covers both.

Greedy decoding (temperature 0) resumes to exactly the uninterrupted output.
With sampling the result is still a valid sample but not the same RNG stream.
Requests with ``n > 1``, ``best_of`` or a batch of prompts are only retried
before their first token, never resumed.

    python -m connex.serving.proxy --port 5199 \\
        --endpoints http://stage0-a:5100,http://stage0-b:5101
"""
import argparse
import asyncio
import json
import logging
import time

import aiohttp
from aiohttp import web

try:  # optional, ~5-10x faster per chunk
    import orjson

    _loads = orjson.loads

    def _dumps(obj):
        return orjson.dumps(obj)
except ImportError:  # pragma: no cover
    _loads = json.loads

    def _dumps(obj):
        return json.dumps(obj, separators=(",", ":")).encode()

log = logging.getLogger("connex.proxy")

RETRYABLE = (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError)


class StreamBroken(Exception):
    pass


class Endpoint:
    __slots__ = ("url", "bad_until", "inflight", "dead")

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.bad_until = 0.0
        self.inflight = 0
        self.dead = asyncio.Event()   # set while blacklisted: wakes waiters

    def healthy(self, now):
        return now >= self.bad_until


class Pool:
    def __init__(self, urls=(), endpoints_file=None, blacklist_s=5.0):
        self.eps = {}
        self.order = []
        self.file = endpoints_file
        self.blacklist_s = blacklist_s
        self.rr = 0
        self.set_urls(urls)

    def set_urls(self, urls):
        urls = [u.strip().rstrip("/") for u in urls if u.strip()]
        for u in urls:
            if u not in self.eps:
                self.eps[u] = Endpoint(u)
        self.order = urls

    def reload(self):
        if not self.file:
            return
        try:
            urls = [l.strip() for l in open(self.file) if l.strip()]
        except OSError:
            return
        if urls:
            self.set_urls(urls)

    def mark_bad(self, ep, why=""):
        if ep.healthy(time.monotonic()):
            log.warning("blacklisting %s for %.1fs: %s", ep.url, self.blacklist_s, why)
        ep.bad_until = time.monotonic() + self.blacklist_s
        ep.dead.set()

    def mark_good(self, ep):
        if ep.bad_until:
            ep.bad_until = 0.0
            ep.dead = asyncio.Event()

    def pick(self, exclude=()):
        now = time.monotonic()
        cands = [self.eps[u] for u in self.order if u not in exclude]
        if not cands:
            return None
        healthy = [e for e in cands if e.healthy(now)]
        if not healthy:   # last resort: the one whose blacklist expires first
            return min(cands, key=lambda e: e.bad_until)
        self.rr += 1
        n = len(healthy)
        return min((healthy[(self.rr + i) % n] for i in range(n)), key=lambda e: e.inflight)


def _ids_from_logprobs(lp, chat):
    if not lp:
        return None
    toks = [c.get("token") for c in lp.get("content") or []] if chat else lp.get("tokens")
    if not toks:
        return None
    out = []
    for t in toks:
        if not isinstance(t, str) or not t.startswith("token_id:"):
            return None
        out.append(int(t[9:]))
    return out


class Resumer:
    """State of one client request across backend attempts."""

    def __init__(self, proxy, kind, body):
        self.p = proxy
        self.kind = kind                       # "completions" | "chat"
        self.chat = kind == "chat"
        self.body = body
        so = body.get("stream_options") or {}
        self.client_usage = bool(so.get("include_usage"))
        self.client_cont_usage = bool(so.get("continuous_usage_stats"))
        prompt = body.get("prompt")
        self.resumable = (body.get("n") in (None, 1) and body.get("best_of") in (None, 1)
                          and (self.chat or isinstance(prompt, str) or
                               (isinstance(prompt, list) and prompt and isinstance(prompt[0], int))))
        lp = body.get("logprobs")
        self.client_lp = lp not in (None, False)
        # token-id resume: completions only (chat cannot take token ids as input)
        self.use_ids = (self.resumable and not self.chat and proxy.token_ids
                        and (not self.client_lp or body.get("return_tokens_as_token_ids")))
        self.inject_lp = self.use_ids and not self.client_lp
        self.ids = []            # emitted completion token ids (if use_ids)
        self.text = []           # emitted text pieces
        self.k = 0               # emitted completion tokens
        self.prompt_tokens = None
        self.prompt_ids = prompt if isinstance(prompt, list) else None
        self.rid = None
        self.created = None
        self.model = body.get("model")
        self.finish = None
        self.stop_reason = None
        self.attempts = 0
        self.sent_role = False

    # -- request construction -------------------------------------------------
    async def build(self, ep):
        b = dict(self.body)
        b["stream"] = True
        b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        if self.inject_lp:
            b["logprobs"] = True if self.chat else 0
            b["return_tokens_as_token_ids"] = True
        if self.k == 0:
            return b
        for key in ("max_tokens", "max_completion_tokens"):
            if b.get(key) is not None:
                b[key] = b[key] - self.k
        if b.get("min_tokens"):
            b["min_tokens"] = max(0, b["min_tokens"] - self.k)
        gen_text = "".join(self.text)
        if self.chat:
            msgs = [dict(m) for m in b["messages"]]
            if b.get("continue_final_message") and msgs and msgs[-1].get("role") == "assistant":
                msgs[-1]["content"] = (msgs[-1].get("content") or "") + gen_text
            else:
                msgs.append({"role": "assistant", "content": gen_text})
            b["messages"] = msgs
            b["add_generation_prompt"] = False
            b["continue_final_message"] = True
            return b
        b["echo"] = False
        if self.use_ids and len(self.ids) == self.k:
            if self.prompt_ids is None:
                self.prompt_ids = await self.p.tokenize(ep, self.model, self.body["prompt"])
            if self.prompt_ids is not None:
                b["prompt"] = list(self.prompt_ids) + self.ids
                return b
        if isinstance(b["prompt"], str):
            b["prompt"] = b["prompt"] + gen_text
            return b
        raise StreamBroken("cannot resume: token-id prompt without generated ids")

    def can_resume(self):
        return self.k == 0 or self.resumable

    def remaining(self):
        m = self.body.get("max_tokens", self.body.get("max_completion_tokens"))
        return None if m is None else m - self.k

    # -- chunk processing -------------------------------------------------------
    def on_chunk(self, c, base_k):
        """Update state from one backend chunk; return the chunk to forward or None."""
        usage = c.get("usage")
        choices = c.get("choices") or []
        if usage and base_k == 0 and self.prompt_tokens is None:
            self.prompt_tokens = usage.get("prompt_tokens")
        if not choices:          # trailing usage-only chunk: we emit our own
            if usage and usage.get("completion_tokens") is not None:
                self.k = max(self.k, base_k + usage["completion_tokens"])
            return None
        if self.rid is None:
            self.rid, self.created = c.get("id"), c.get("created")
            self.model = c.get("model", self.model)
        ch = choices[0]
        if len(choices) > 1 or ch.get("index", 0) != 0:
            self.resumable = False
        if self.chat:
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ""
            if not piece and ch.get("finish_reason") is None and "role" in delta:
                if self.sent_role:
                    return None  # a retried/resumed stream's role preamble
                self.sent_role = True
        else:
            piece = ch.get("text") or ""
        ids = ch.get("token_ids") or (_ids_from_logprobs(ch.get("logprobs"), self.chat)
                                      if self.use_ids else None)
        if usage and usage.get("completion_tokens") is not None:
            self.k = base_k + usage["completion_tokens"]
        elif ids:
            self.k += len(ids)
        elif piece:
            self.k += 1
        if self.use_ids:
            if ids:
                self.ids.extend(ids)
            if len(self.ids) != self.k:
                self.use_ids = False   # lost track: fall back to text resume
        self.text.append(piece)
        if ch.get("finish_reason") is not None:
            self.finish = ch["finish_reason"]
            self.stop_reason = ch.get("stop_reason")
        # rewrite for the client
        c["id"], c["created"], c["model"] = self.rid, self.created, self.model
        if self.inject_lp:
            ch["logprobs"] = None
        if "token_ids" in ch and not self.body.get("return_token_ids"):
            del ch["token_ids"]
        if self.client_cont_usage:
            c["usage"] = self.usage()
        elif "usage" in c:
            c["usage"] = None
        return c

    def usage(self):
        pt = self.prompt_tokens
        if pt is None and self.prompt_ids is not None:
            pt = len(self.prompt_ids)
        return {"prompt_tokens": pt, "completion_tokens": self.k,
                "total_tokens": (pt or 0) + self.k}


class Proxy:
    def __init__(self, pool, idle_timeout=2.5, connect_timeout=2.0, max_attempts=4,
                 probe_interval=1.0, token_ids=True, first_token_timeout=None):
        self.pool = pool
        self.idle_timeout = idle_timeout
        self.first_token_timeout = first_token_timeout
        self.connect_timeout = connect_timeout
        self.max_attempts = max_attempts
        self.probe_interval = probe_interval
        self.token_ids = token_ids
        self.session = None
        self.probe_session = None
        self.stats = {"requests": 0, "retries": 0, "resumes": 0, "failed": 0}
        self._bg = []

    # -- lifecycle --------------------------------------------------------------
    def app(self):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/proxy/stats", self.handle_stats)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app=None):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.connect_timeout),
            auto_decompress=False, trust_env=False)
        self.probe_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=True), trust_env=False)
        if self.pool.file:
            self.pool.reload()
            self._bg.append(asyncio.create_task(self._reload_loop()))
        if self.probe_interval:
            self._bg.append(asyncio.create_task(self._probe_loop()))

    async def _cleanup(self, app=None):
        for t in self._bg:
            t.cancel()
        await self.session.close()
        await self.probe_session.close()

    async def _reload_loop(self):
        while True:
            await asyncio.sleep(1.0)
            self.pool.reload()

    async def _probe_loop(self):
        while True:
            await asyncio.sleep(self.probe_interval)
            await asyncio.gather(*(self._probe(e) for e in list(self.pool.eps.values())
                                   if e.url in self.pool.order), return_exceptions=True)

    async def _alive(self, ep, timeout=2.0):
        try:
            async with self.probe_session.get(ep.url + "/v1/models",
                                              timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
        except RETRYABLE:
            return False

    async def _probe(self, ep):
        # two consecutive failures before blacklisting: a busy-but-alive API
        # server must not get its pending prefills cut short
        if await self._alive(ep) or await self._alive(ep):
            self.pool.mark_good(ep)
        else:
            self.pool.mark_bad(ep, "probe failed twice")

    async def tokenize(self, ep, model, prompt):
        try:
            async with self.session.post(ep.url + "/tokenize", json={"model": model, "prompt": prompt},
                                         timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    return (await r.json())["tokens"]
        except RETRYABLE:
            pass
        return None

    # -- core -------------------------------------------------------------------
    async def _attempt(self, st, ep, path):
        """Stream one backend attempt; yield client chunks. Raises StreamBroken
        (or a ClientError) if the stream ends without [DONE]."""
        req = await st.build(ep)
        base_k = st.k
        ep.inflight += 1
        try:
            async with self.session.post(ep.url + path, json=req) as r:
                if r.status != 200:
                    payload = await r.read()
                    if r.status >= 500 or r.status in (404, 408, 429) or base_k > 0:
                        raise StreamBroken(f"HTTP {r.status}")
                    raise HTTPPassthrough(r.status, payload, r.content_type)
                content = r.content
                got_first = st.k > base_k
                # Idle detection without a per-chunk wait_for (which costs a Task
                # per chunk): one watchdog per stream closes the response when no
                # line arrived for idle_timeout; the pending readline then fails.
                loop = asyncio.get_running_loop()
                last = [loop.time()]
                fired = [False]
                watchdog = None
                try:
                    while True:
                        if got_first or base_k > 0:
                            if watchdog is None and self.idle_timeout:
                                watchdog = asyncio.ensure_future(self._watchdog(r, last, fired))
                            try:
                                line = await content.readline()
                            except RETRYABLE:
                                if fired[0]:
                                    raise asyncio.TimeoutError()
                                raise
                            if fired[0]:
                                raise asyncio.TimeoutError()
                            last[0] = loop.time()
                        else:
                            line = await self._wait_first(content, ep)
                        if not line:
                            raise StreamBroken("EOF before [DONE]")
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            return
                        out = st.on_chunk(_loads(data), base_k)
                        if st.k > base_k:
                            got_first = True
                        if out is not None:
                            yield out
                finally:
                    if watchdog is not None:
                        watchdog.cancel()
        finally:
            ep.inflight -= 1

    async def _watchdog(self, r, last, fired):
        loop = asyncio.get_running_loop()
        while True:
            wait = last[0] + self.idle_timeout - loop.time()
            if wait <= 0:
                fired[0] = True
                r.close()
                return
            await asyncio.sleep(max(wait, 0.01))

    async def _wait_first(self, content, ep):
        """Wait for the next line before the first token, without a fixed idle
        timeout: give up only if the replica is declared dead meanwhile."""
        read = asyncio.ensure_future(content.readline())
        dead = asyncio.ensure_future(ep.dead.wait())
        try:
            done, _ = await asyncio.wait({read, dead}, timeout=self.first_token_timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if read in done:
                return read.result()
            raise StreamBroken("replica died before first token" if done else "first-token timeout")
        finally:
            dead.cancel()
            if not read.done():
                read.cancel()

    async def generate(self, kind, body):
        """Async generator of client chunks for one request, with retry/resume."""
        st = Resumer(self, kind, body)
        path = "/v1/chat/completions" if st.chat else "/v1/completions"
        self.stats["requests"] += 1
        tried = []
        last_err = "no endpoints"
        while st.attempts < self.max_attempts:
            if st.remaining() is not None and st.remaining() <= 0 and st.k > 0:
                st.finish = st.finish or "length"
                break
            ep = self.pool.pick(exclude=tried) or self.pool.pick()
            if ep is None:
                break
            st.attempts += 1
            if st.attempts > 1:
                self.stats["resumes" if st.k else "retries"] += 1
            k_before = st.k
            try:
                async for c in self._attempt(st, ep, path):
                    yield c
                yield st
                return
            except HTTPPassthrough:
                raise
            except (StreamBroken, ValueError, KeyError, *RETRYABLE) as e:
                last_err = f"{type(e).__name__}: {e}"
                if isinstance(e, asyncio.TimeoutError):
                    last_err = f"idle > {self.idle_timeout}s"
                if isinstance(e, asyncio.TimeoutError) and await self._alive(ep):
                    log.info("%s idle but alive: resuming elsewhere without blacklisting", ep.url)
                else:
                    self.pool.mark_bad(ep, last_err)
                tried.append(ep.url)
                if len(tried) >= len(self.pool.order):
                    tried = tried[-1:]   # allow going back to a replica that recovered
                log.warning("attempt %d on %s failed after %d tokens (%d this attempt): %s",
                            st.attempts, ep.url, st.k, st.k - k_before, last_err)
                if not st.can_resume():
                    break
        self.stats["failed"] += 1
        raise UpstreamFailed(last_err, st)

    # -- HTTP handlers ----------------------------------------------------------
    async def handle_completions(self, request):
        return await self._handle(request, "completions")

    async def handle_chat(self, request):
        return await self._handle(request, "chat")

    async def _handle(self, request, kind):
        body = await request.json()
        gen = self.generate(kind, body)
        if body.get("stream"):
            return await self._stream(request, gen)
        return await self._aggregate(gen, kind, body)

    async def _stream(self, request, gen):
        resp = None
        try:
            async for c in gen:
                if isinstance(c, Resumer):
                    if c.client_usage:
                        await resp.write(_sse({"id": c.rid, "object": _obj(c.chat), "created": c.created,
                                               "model": c.model, "choices": [], "usage": c.usage()}))
                    await resp.write(b"data: [DONE]\n\n")
                    break
                if resp is None:
                    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
                    await resp.prepare(request)
                await resp.write(_sse(c))
        except HTTPPassthrough as e:
            if resp is None:
                return web.Response(status=e.status, body=e.payload, content_type=e.ctype)
            raise
        except UpstreamFailed as e:
            if resp is None:
                return web.json_response({"error": {"message": f"all replicas failed: {e}",
                                                    "type": "upstream_failed"}}, status=502)
            # mid-stream and out of options: end the stream visibly short
            await resp.write(_sse({"error": {"message": f"stream lost after {e.st.k} tokens: {e}",
                                             "type": "upstream_failed"}}))
            await resp.write(b"data: [DONE]\n\n")
        if resp is None:   # zero chunks but success: empty completion
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _aggregate(self, gen, kind, body):
        text, lps, st = [], [], None
        try:
            async for c in gen:
                if isinstance(c, Resumer):
                    st = c
                    break
                ch = c["choices"][0]
                text.append((ch.get("delta") or {}).get("content") or "" if kind == "chat"
                            else ch.get("text") or "")
                if ch.get("logprobs"):
                    lps.append(ch["logprobs"])
        except HTTPPassthrough as e:
            return web.Response(status=e.status, body=e.payload, content_type=e.ctype)
        except UpstreamFailed as e:
            return web.json_response({"error": {"message": f"all replicas failed: {e}",
                                                "type": "upstream_failed"}}, status=502)
        content = "".join(text)
        if kind == "chat":
            choice = {"index": 0, "message": {"role": "assistant", "content": content},
                      "logprobs": _merge_lp(lps, True), "finish_reason": st.finish,
                      "stop_reason": st.stop_reason}
        else:
            choice = {"index": 0, "text": content, "logprobs": _merge_lp(lps, False),
                      "finish_reason": st.finish, "stop_reason": st.stop_reason}
        return web.json_response({"id": st.rid, "object": "chat.completion" if kind == "chat"
                                  else "text_completion", "created": st.created, "model": st.model,
                                  "choices": [choice], "usage": st.usage()})

    async def handle_models(self, request):
        for _ in range(len(self.pool.order) or 1):
            ep = self.pool.pick()
            if ep is None:
                break
            try:
                async with self.session.get(ep.url + "/v1/models",
                                            timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return web.Response(status=r.status, body=await r.read(),
                                        content_type=r.content_type)
            except RETRYABLE as e:
                self.pool.mark_bad(ep, f"models: {e}")
        return web.json_response({"error": "no healthy replica"}, status=502)

    async def handle_health(self, request):
        now = time.monotonic()
        ok = any(self.pool.eps[u].healthy(now) for u in self.pool.order)
        return web.json_response({"healthy": ok}, status=200 if ok else 503)

    async def handle_stats(self, request):
        now = time.monotonic()
        return web.json_response({**self.stats, "endpoints": [
            {"url": u, "healthy": self.pool.eps[u].healthy(now), "inflight": self.pool.eps[u].inflight}
            for u in self.pool.order]})


class HTTPPassthrough(Exception):
    def __init__(self, status, payload, ctype):
        super().__init__(f"HTTP {status}")
        self.status, self.payload, self.ctype = status, payload, ctype or "application/json"


class UpstreamFailed(Exception):
    def __init__(self, msg, st):
        super().__init__(msg)
        self.st = st


def _obj(chat):
    return "chat.completion.chunk" if chat else "text_completion"


def _sse(obj):
    return b"data: " + _dumps(obj) + b"\n\n"


def _merge_lp(lps, chat):
    if not lps:
        return None
    if chat:
        return {"content": [x for lp in lps for x in lp.get("content") or []]}
    out = {"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []}
    for lp in lps:
        for k in out:
            out[k].extend(lp.get(k) or [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5199)
    ap.add_argument("--endpoints", default="", help="comma-separated stage-0 base URLs")
    ap.add_argument("--endpoints-file", default=None, help="one URL per line, re-read every 1 s")
    ap.add_argument("--idle-timeout", type=float, default=2.5,
                    help="max gap between chunks once the first token arrived (s)")
    ap.add_argument("--first-token-timeout", type=float, default=None,
                    help="optional hard cap on time to first token (default: none; a dead "
                         "replica is detected by the prober instead)")
    ap.add_argument("--connect-timeout", type=float, default=2.0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--blacklist-s", type=float, default=5.0)
    ap.add_argument("--probe-interval", type=float, default=1.0, help="0 disables probing")
    ap.add_argument("--no-token-ids", action="store_true",
                    help="resume completions from text instead of token ids")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    pool = Pool([u for u in a.endpoints.split(",") if u], a.endpoints_file, a.blacklist_s)
    proxy = Proxy(pool, a.idle_timeout, a.connect_timeout, a.max_attempts, a.probe_interval,
                  not a.no_token_ids, a.first_token_timeout)
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    web.run_app(proxy.app(), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
