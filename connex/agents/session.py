# SPDX-License-Identifier: Apache-2.0
"""Session-aware routing and live session migration (see package doc)."""
import argparse
import logging
import time

from aiohttp import web

from ..serving.proxy import Pool, Proxy

log = logging.getLogger("connex.agents")

HEADER = "X-Connex-Session"


class Session:
    __slots__ = ("sid", "url", "epoch", "inflight", "context", "kind", "model",
                 "last_used", "moves", "migrating")

    def __init__(self, sid):
        self.sid = sid
        self.url = None          # placement
        self.epoch = 0
        self.inflight = set()    # Resumers
        self.context = None      # last full context (request + reply), for re-prefill
        self.kind = None
        self.model = None
        self.last_used = time.monotonic()
        self.moves = []
        self.migrating = None    # asyncio.Event while a migration prepares the target

    @property
    def lid(self):
        return f"session/{self.sid}"

    def info(self):
        return {"lid": self.lid, "placement": self.url, "epoch": self.epoch,
                "inflight": len(self.inflight), "moves": self.moves[-5:]}


class KVMover:
    """Makes ``dst`` hold the prefix state of ``session``. Returns a dict
    (at least ``{"mode": ...}``) or raises; ``SessionProxy`` falls back to
    the next mover."""

    name = "kv"

    async def move(self, proxy, session, src, dst):
        raise NotImplementedError


class ReprefillMover(KVMover):
    """Re-prefill the session's conversation on ``dst`` (one generated
    token), so ``dst``'s prefix cache holds it before traffic moves. Works
    on any engine with automatic prefix caching (vLLM, SGLang)."""

    name = "reprefill"

    async def move(self, proxy, session, src, dst):
        live = [st for st in session.inflight if st.k > 0 and st.can_resume()]
        if live:     # warm what the cut-over request will resend: prompt + output so far
            st = live[-1]
            ctx = await st.build(proxy.pool.eps[dst])
        else:
            ctx = session.context
        if ctx is None:
            return {"mode": self.name, "tokens": 0}
        body = dict(ctx)
        body.update(stream=False, max_tokens=1, temperature=0.0)
        body.pop("max_completion_tokens", None)
        for k in ("stream_options", "logprobs", "return_tokens_as_token_ids", "min_tokens"):
            body.pop(k, None)
        path = "/v1/chat/completions" if session.kind == "chat" else "/v1/completions"
        async with proxy.session.post(dst.rstrip("/") + path, json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"re-prefill on {dst}: HTTP {r.status}")
            out = await r.json()
        return {"mode": self.name, "tokens": (out.get("usage") or {}).get("prompt_tokens")}


class SessionProxy(Proxy):
    def __init__(self, pool, movers=None, session_ttl_s=3600.0, **kw):
        super().__init__(pool, **kw)
        self.sessions = {}
        self.movers = list(movers) if movers else [ReprefillMover()]
        self.ttl = session_ttl_s
        self._next_sweep = time.monotonic() + (session_ttl_s or 0)
        self.stats.update(sessions=0, migrations=0, failovers=0)

    # -- routing ----------------------------------------------------------------
    def app(self):
        app = super().app()
        app.router.add_get("/sessions", self.handle_sessions)
        app.router.add_post("/sessions/{sid}/migrate", self.handle_migrate)
        app.router.add_post("/sessions/drain", self.handle_drain)
        return app

    async def _handle(self, request, kind):
        body = await request.json()
        sid = body.pop("session_id", None) or request.headers.get(HEADER)
        if sid is not None:
            body["_connex_session"] = str(sid)
        gen = self.generate(kind, body)
        if body.get("stream"):
            return await self._stream(request, gen)
        return await self._aggregate(gen, kind, body)

    def get_session(self, sid):
        s = self.sessions.get(sid)
        if s is None:
            self.expire_sessions()
            s = self.sessions[sid] = Session(sid)
            self.stats["sessions"] += 1
        return s

    def expire_sessions(self, now=None):
        """Forget sessions idle (no request in flight) for longer than the
        TTL, so their stored contexts do not accumulate forever. Runs at most
        once per TTL/10 when a new session is created."""
        now = time.monotonic() if now is None else now
        if not self.ttl or now < self._next_sweep:
            return 0
        self._next_sweep = now + self.ttl / 10
        dead = [sid for sid, s in self.sessions.items()
                if not s.inflight and now - s.last_used > self.ttl]
        for sid in dead:
            del self.sessions[sid]
        return len(dead)

    def begin(self, st):
        sid = st.body.pop("_connex_session", None)
        st.sid = sid
        if sid is None:
            return
        s = self.get_session(sid)
        s.inflight.add(st)
        s.kind, s.model, s.last_used = st.kind, st.model, time.monotonic()

    def end(self, st):
        sid = getattr(st, "sid", None)
        s = self.sessions.get(sid) if sid else None
        if s is None:
            return
        s.inflight.discard(st)
        s.last_used = time.monotonic()
        if st.finish is not None:      # completed: remember the full context
            text = "".join(st.text)
            b = {k: v for k, v in st.body.items() if k not in ("stream", "stream_options")}
            if st.chat:
                b["messages"] = list(b["messages"]) + [{"role": "assistant", "content": text}]
                b["add_generation_prompt"] = False
                b["continue_final_message"] = False
            elif st.use_ids and st.prompt_ids is not None and len(st.ids) == st.k:
                b["prompt"] = list(st.prompt_ids) + list(st.ids)
            elif isinstance(b.get("prompt"), str):
                b["prompt"] = b["prompt"] + text
            s.context = b

    def pick(self, st, tried):
        sid = getattr(st, "sid", None)
        if sid is None:
            return super().pick(st, tried)
        s = self.get_session(sid)
        now = time.monotonic()
        ep = self.pool.eps.get(s.url) if s.url else None
        if ep is not None and ep.url not in tried and ep.healthy(now) and ep.url in self.pool.order:
            return ep
        ep = self.pool.pick(exclude=tried) or self.pool.pick()
        if ep is None:
            return None
        if s.url is not None and s.url != ep.url:
            # placement lost (dead / removed): fail over, re-prefill there
            self.stats["failovers"] += 1
            s.epoch += 1
            s.moves.append({"epoch": s.epoch, "from": s.url, "to": ep.url, "mode": "failover"})
            log.warning("session %s: %s unavailable, failing over to %s (epoch %d)",
                        sid, s.url, ep.url, s.epoch)
        s.url = ep.url
        return ep

    # -- migration --------------------------------------------------------------
    async def migrate(self, sid, target, mode="auto"):
        """Move session ``sid`` to ``target``. ``mode``: "auto" (movers in
        order, re-prefill last), a mover name, or "none" (cut over only; the
        next turn re-prefills on demand)."""
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        target = target.rstrip("/")
        if target not in self.pool.eps:
            raise ValueError(f"unknown endpoint {target}")
        src = s.url
        t0 = time.perf_counter()
        info = {"mode": "none"}
        if src != target and mode != "none":
            movers = self.movers if mode == "auto" else [m for m in self.movers if m.name == mode]
            if not movers:
                raise ValueError(f"no mover {mode!r}")
            for m in movers:
                try:
                    info = await m.move(self, s, src, target)
                    break
                except Exception as e:  # fall back to the next mover
                    log.warning("session %s: mover %s failed: %s", sid, m.name, e)
                    info = {"mode": "none", "error": repr(e)}
        t_prep = time.perf_counter() - t0
        s.epoch += 1
        s.url = target
        moved = sum(self.move(st) for st in list(s.inflight)) if src != target else 0
        rec = {"epoch": s.epoch, "from": src, "to": target, "prepare_s": t_prep,
               "inflight_moved": moved, **info}
        s.moves.append(rec)
        self.stats["migrations"] += 1
        return rec

    async def drain(self, url, targets=None, mode="auto"):
        """Move every session placed on ``url`` elsewhere (scale-down)."""
        url = url.rstrip("/")
        out = []
        for s in [s for s in self.sessions.values() if s.url == url]:
            ep = self.pool.pick(exclude=[url])
            if ep is None or ep.url == url:
                raise RuntimeError("no other endpoint to drain to")
            out.append(await self.migrate(s.sid, ep.url, mode))
        return out

    # -- admin HTTP -------------------------------------------------------------
    async def handle_sessions(self, request):
        return web.json_response({sid: s.info() for sid, s in self.sessions.items()})

    async def handle_migrate(self, request):
        b = await request.json()
        try:
            rec = await self.migrate(request.match_info["sid"], b["target"], b.get("mode", "auto"))
        except (KeyError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(rec)

    async def handle_drain(self, request):
        b = await request.json()
        return web.json_response(await self.drain(b["url"], mode=b.get("mode", "auto")))


def main():
    ap = argparse.ArgumentParser(description="Connex session-aware proxy")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5299)
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--endpoints-file", default=None)
    ap.add_argument("--idle-timeout", type=float, default=30.0)
    ap.add_argument("--probe-interval", type=float, default=1.0)
    ap.add_argument("--no-token-ids", action="store_true",
                    help="resume from text (engines without token-id logprobs, e.g. SGLang)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    pool = Pool([u for u in a.endpoints.split(",") if u], a.endpoints_file)
    proxy = SessionProxy(pool, idle_timeout=a.idle_timeout, probe_interval=a.probe_interval,
                         token_ids=not a.no_token_ids)
    web.run_app(proxy.app(), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
