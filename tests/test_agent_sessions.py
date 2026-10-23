# SPDX-License-Identifier: Apache-2.0
"""connex.agents: sticky session routing, live migration of an in-flight
generation, re-prefill warm-up of the target, failover, drain (CPU, fakes)."""
import asyncio
import json

import aiohttp
from aiohttp import web

from connex.agents import KVMover, SessionProxy
from connex.serving import fake_backend as fb
from connex.serving.proxy import Pool

PROMPT = "w5 w9 w13 w21"


class Rig:
    def __init__(self, n=3, tpot_ms=0, movers=None):
        self.backends = [fb.FakeBackend(tpot_ms=tpot_ms) for _ in range(n)]
        self.movers = movers

    async def __aenter__(self):
        self.urls = [await b.start() for b in self.backends]
        self.proxy = SessionProxy(Pool(self.urls), movers=self.movers, idle_timeout=2.0,
                                  probe_interval=0)
        self.runner = web.AppRunner(self.proxy.app(), shutdown_timeout=0.1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.client = aiohttp.ClientSession(trust_env=False)
        return self

    async def __aexit__(self, *a):
        await self.client.close()
        await self.runner.cleanup()
        for b in self.backends:
            await b.stop()

    def owner(self, req_count_before):
        return [i for i, b in enumerate(self.backends)
                if len(b.requests) > req_count_before[i]]

    async def turn(self, sid, prompt, n=12, header=False):
        body = {"model": "fake", "prompt": prompt, "max_tokens": n, "temperature": 0,
                "ignore_eos": True, "stream": True}
        hdr = {}
        if header:
            hdr["X-Connex-Session"] = sid
        else:
            body["session_id"] = sid
        text = []
        async with self.client.post(self.url + "/v1/completions", json=body, headers=hdr) as r:
            assert r.status == 200
            async for line in r.content:
                if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                    c = json.loads(line[6:])
                    if c.get("choices"):
                        text.append(c["choices"][0]["text"])
        return "".join(text)


def counts(rig):
    return [len(b.requests) for b in rig.backends]


def test_sticky_routing_and_no_session_field_leak():
    async def go():
        async with Rig() as rig:
            c0 = counts(rig)
            for k in range(4):
                await rig.turn("s1", PROMPT, header=bool(k % 2))
            used = rig.owner(c0)
            assert len(used) == 1                         # all turns on one instance
            for b in rig.backends:
                assert all("session_id" not in r and "_connex_session" not in r
                           for r in b.requests)
            assert rig.proxy.sessions["s1"].url == rig.urls[used[0]]
    asyncio.run(go())


def test_migrate_inflight_generation_greedy_identical():
    async def go():
        async with Rig(tpot_ms=20) as rig:
            _, ref = fb.reference(PROMPT, 40)
            await rig.turn("s", "w1 w2")                  # place the session
            src = rig.proxy.sessions["s"].url
            dst = next(u for u in rig.urls if u != src)
            task = asyncio.create_task(rig.turn("s", PROMPT, n=40))
            await asyncio.sleep(0.3)                      # mid-generation
            rec = await rig.proxy.migrate("s", dst)
            out = await task
            assert out == ref                             # client sees one stream
            assert rec["inflight_moved"] == 1 and rec["mode"] == "reprefill"
            assert rig.proxy.sessions["s"].url == dst and rig.proxy.sessions["s"].epoch == 1
            d = rig.backends[rig.urls.index(dst)].requests
            assert d[0]["max_tokens"] == 1                # warm-up (re-prefill) first
            assert d[1]["prompt"][:len(fb.tokenize(PROMPT))] == fb.tokenize(PROMPT)
            assert 0 < len(d[1]["prompt"]) - len(fb.tokenize(PROMPT)) < 40  # resumed
            assert rig.proxy.stats["failed"] == 0 and rig.proxy.stats["moves"] == 1
            c0 = counts(rig)
            await rig.turn("s", PROMPT)                   # next turn lands on dst
            assert rig.owner(c0) == [rig.urls.index(dst)]
    asyncio.run(go())


def test_failover_when_placement_dies():
    async def go():
        async with Rig() as rig:
            await rig.turn("s", PROMPT)
            src = rig.proxy.sessions["s"].url
            i = rig.urls.index(src)
            rig.backends[i].fail, rig.backends[i].fail_count = ("http500", 0), None
            _, ref = fb.reference(PROMPT, 12)
            assert await rig.turn("s", PROMPT) == ref
            s = rig.proxy.sessions["s"]
            assert s.url != src and s.epoch == 1 and s.moves[-1]["mode"] == "failover"
    asyncio.run(go())


def test_mover_fallback_and_drain():
    class Broken(KVMover):
        name = "kv"

        async def move(self, proxy, session, src, dst):
            raise RuntimeError("no kv path")

    async def go():
        from connex.agents import ReprefillMover
        async with Rig(movers=[Broken(), ReprefillMover()]) as rig:
            for sid in ("a", "b", "c", "d"):
                await rig.turn(sid, PROMPT)
            victim = rig.proxy.sessions["a"].url
            recs = await rig.proxy.drain(victim)
            assert recs and all(r["mode"] == "reprefill" for r in recs)
            assert all(s.url != victim for s in rig.proxy.sessions.values())
            async with rig.client.get(rig.url + "/sessions") as r:
                info = await r.json()
            assert info["a"]["lid"] == "session/a"
    asyncio.run(go())
