# SPDX-License-Identifier: Apache-2.0
"""PD router against fake prefill/decode instances: dispatch, failover,
elastic pools (CPU only)."""
import asyncio
import json
import time

import aiohttp
from aiohttp import web

from connex.serving import fake_backend as fb
from connex.serving.pd_router import ElasticPool, PDRouter

PROMPT = "w5 w9 w13 w21"
MAXT = 12


def run(coro):
    return asyncio.run(coro)


class Rig:
    def __init__(self, n_prefill=2, n_decode=2, tpot_ms=0, probe=0.05, idle=0.5):
        self.kv = {}
        self.prefill = [fb.FakePDBackend("prefill", self.kv) for _ in range(n_prefill)]
        self.decode = [fb.FakePDBackend("decode", self.kv, tpot_ms=tpot_ms) for _ in range(n_decode)]
        self.probe, self.idle = probe, idle

    async def __aenter__(self):
        pu = [await b.start() for b in self.prefill]
        du = [await b.start() for b in self.decode]
        self.pre_pool, self.dec_pool = ElasticPool(pu), ElasticPool(du)
        self.router = PDRouter(self.pre_pool, self.dec_pool, idle_timeout=self.idle,
                               probe_interval=self.probe, max_attempts=4)
        self.runner = web.AppRunner(self.router.app(), shutdown_timeout=0.1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.client = aiohttp.ClientSession(trust_env=False)
        await self.wait_probed()
        return self

    async def wait_probed(self, timeout=3.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if len(self.pre_pool.healthy_urls()) == len(self.pre_pool.order) and \
                    len(self.dec_pool.healthy_urls()) == len(self.dec_pool.order):
                return
            await asyncio.sleep(0.02)

    async def __aexit__(self, *a):
        await self.client.close()
        await self.runner.cleanup()
        for b in self.prefill + self.decode:
            await b.stop()

    async def stream(self, body):
        chunks = []
        async with self.client.post(self.url + "/v1/completions", json={**body, "stream": True}) as r:
            if r.status != 200:
                return r.status, await r.json()
            async for line in r.content:
                if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                    chunks.append(json.loads(line[6:]))
        return 200, chunks


def body(prompt=PROMPT, n=MAXT, **kw):
    return {"model": "fake", "prompt": prompt, "max_tokens": n, "temperature": 0,
            "stream_options": {"include_usage": True}, **kw}


def text_of(chunks):
    return "".join(c["choices"][0]["text"] for c in chunks if c.get("choices"))


def check(chunks, prompt=PROMPT, n=MAXT):
    assert not any("error" in c for c in chunks), chunks
    assert text_of(chunks) == fb.reference(prompt, n)[1]
    u = [c for c in chunks if not c.get("choices") and c.get("usage")][-1]["usage"]
    assert u["completion_tokens"] == n and u["prompt_tokens"] == len(fb.tokenize(prompt))
    assert len({c["id"] for c in chunks}) == 1


def test_prefill_then_decode_with_kv_params():
    async def go():
        async with Rig(1, 1) as r:
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            (p,), (d,) = r.prefill, r.decode
            assert p.prefills == 1 and d.pulls == 1
            preq = p.requests[-1]
            assert preq["max_tokens"] == 1 and preq["stream"] is False
            assert preq["kv_transfer_params"]["do_remote_decode"] is True
            dreq = d.requests[-1]
            assert dreq["kv_transfer_params"]["do_remote_prefill"] is True
            assert dreq["max_tokens"] == MAXT
    run(go())


def test_non_streaming_and_chat():
    async def go():
        async with Rig(1, 1) as r:
            async with r.client.post(r.url + "/v1/completions", json=body()) as resp:
                out = await resp.json()
            assert out["choices"][0]["text"] == fb.reference(PROMPT, MAXT)[1]
            msgs = [{"role": "user", "content": "w3 w4"}]
            async with r.client.post(r.url + "/v1/chat/completions",
                                     json={"model": "fake", "messages": msgs, "max_tokens": 5}) as resp:
                out = await resp.json()
            seq, ref = fb.chat_ids(msgs), []
            for _ in range(5):
                seq.append(fb.next_token(seq))
                ref.append(seq[-1])
            assert out["choices"][0]["message"]["content"] == "".join(fb.detok(t) for t in ref)
    run(go())


def test_prefill_failure_retried_on_another_prefill():
    async def go():
        async with Rig(2, 1) as r:
            for p in r.prefill:
                p.fail, p.fail_count = ("http500", 0), 1
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            assert r.router.stats["prefill_retries"] >= 1
    run(go())


def test_prefill_instance_killed_before_prefill():
    async def go():
        async with Rig(2, 1, probe=0) as r:
            await r.prefill[0].stop()
            await r.prefill[1].stop()
            r.prefill[1] = fb.FakePDBackend("prefill", r.kv)
            url = await r.prefill[1].start()
            r.pre_pool.set_dynamic([url])
            r.pre_pool.mark_good(r.pre_pool.eps[url])
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
    run(go())


def test_decode_dies_midstream_repulls_kv_from_prefill():
    async def go():
        async with Rig(1, 2) as r:
            for d in r.decode:
                d.fail, d.fail_count = ("reset", 5), 1
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            assert r.router.stats["resumes"] == 2       # each decode died once
            assert r.router.stats["kv_repulls"] == 2
            assert r.prefill[0].prefills == 1          # no second prefill: KV re-pulled
            assert sum(d.pulls for d in r.decode) == 3
            resumed = max((q for d in r.decode for q in d.requests), key=lambda q: len(q["prompt"]))
            assert resumed["max_tokens"] == MAXT - 10
            assert resumed["kv_transfer_params"]["remote_num_tokens"] == len(fb.tokenize(PROMPT))
    run(go())


def test_decode_dies_and_prefill_gone_reprefills_extended_prompt():
    async def go():
        async with Rig(2, 2, tpot_ms=20, idle=2.0) as r:
            r.decode[0].fail, r.decode[0].fail_count = ("reset", 5), 1
            r.dec_pool.drain(r.decode[1].url)     # first attempt goes to decode 0
            task = asyncio.create_task(r.stream(body()))
            while not r.decode[0].requests:
                await asyncio.sleep(0.005)
            r.dec_pool.undrain(r.decode[1].url)
            owner = next(p for p in r.prefill if p.prefills)
            await owner.stop()                    # prefill gone while decode runs
            r.pre_pool.mark_bad(r.pre_pool.eps[owner.url], "test kill")
            st, chunks = await task
            assert st == 200
            check(chunks)
            assert sum(p.prefills for p in r.prefill) == 2
            last = r.decode[1].requests[-1]
            # extended prompt (prompt + 5 generated ids) prefilled again
            assert last["kv_transfer_params"]["remote_num_tokens"] == len(fb.tokenize(PROMPT)) + 5
    run(go())


def test_prefill_dies_during_transfer_reprefills_elsewhere():
    """Prefill answered, then died before the decode pulled the KV: the decode
    instance fails the request (load failure) and the router prefills again
    on another prefill instance, without blacklisting the decode instance."""
    async def go():
        async with Rig(2, 1) as r:
            orig = fb.FakePDBackend._generate
            killed = []

            async def gen(self, req, b, ids, chat):
                if self.role == "decode" and not killed:
                    owner = r.kv[b["kv_transfer_params"]["transfer_id"]]["owner"]
                    killed.append(owner)
                    await owner.stop()         # dies mid-transfer
                return await orig(self, req, b, ids, chat)

            fb.FakePDBackend._generate = gen
            try:
                st, chunks = await r.stream(body())
            finally:
                fb.FakePDBackend._generate = orig
            assert st == 200
            check(chunks)
            assert r.router.stats["kv_reprefills"] == 1
            assert sum(p.prefills for p in r.prefill) == 2
            assert r.dec_pool.healthy_urls() == [r.decode[0].url]
    run(go())


def test_scale_up_and_down_under_load_without_failures():
    async def go():
        async with Rig(1, 1, tpot_ms=2) as r:
            extra_p = fb.FakePDBackend("prefill", r.kv)
            extra_d = fb.FakePDBackend("decode", r.kv, tpot_ms=2)

            async def one(i):
                prompt = f"w{i} w{i + 3} w7"
                st, chunks = await r.stream(body(prompt, 10))
                assert st == 200
                check(chunks, prompt, 10)

            tasks = [asyncio.create_task(one(i)) for i in range(20)]
            await asyncio.sleep(0.02)
            # scale up both pools
            r.pre_pool.set_dynamic([await extra_p.start()])
            r.dec_pool.set_dynamic([await extra_d.start()])
            tasks += [asyncio.create_task(one(i)) for i in range(20, 40)]
            await r.wait_probed()
            assert len(r.dec_pool.healthy_urls()) == 2
            tasks += [asyncio.create_task(one(i)) for i in range(40, 60)]
            # scale down: the original decode and prefill leave (drain), in-flight finish
            r.dec_pool.drain(r.decode[0].url)
            r.pre_pool.drain(r.prefill[0].url)
            n_before = len(r.decode[0].requests)
            tasks += [asyncio.create_task(one(i)) for i in range(60, 80)]
            await asyncio.gather(*tasks)
            assert len(r.decode[0].requests) == n_before     # no new work after drain
            assert extra_d.pulls > 0 and extra_p.prefills > 0
            assert r.router.stats["failed"] == 0
            await extra_p.stop(), await extra_d.stop()
    run(go())


def test_new_member_takes_no_traffic_until_probed():
    async def go():
        async with Rig(1, 1, probe=0) as r:
            pending = "http://127.0.0.1:9"   # nothing listens
            r.dec_pool.set_dynamic([pending])
            assert pending in r.dec_pool.order and pending not in r.dec_pool.healthy_urls()
            for _ in range(5):
                st, chunks = await r.stream(body())
                assert st == 200
                check(chunks)
    run(go())


def test_kill_decode_under_load_goodput():
    async def go():
        async with Rig(2, 3, tpot_ms=3) as r:
            async def one(i):
                prompt = f"w{i} w{i + 1}"
                st, chunks = await r.stream(body(prompt, 20))
                assert st == 200
                check(chunks, prompt, 20)

            tasks = [asyncio.create_task(one(i)) for i in range(30)]
            await asyncio.sleep(0.03)
            await r.decode[0].stop()        # dies mid-stream for its requests
            await asyncio.gather(*tasks)
            assert r.router.stats["failed"] == 0
    run(go())


def test_pools_endpoint_and_drain_api():
    async def go():
        async with Rig(1, 2) as r:
            async with r.client.post(r.url + "/pd/drain",
                                     json={"url": r.decode[0].url, "role": "decode"}) as resp:
                assert resp.status == 200
            async with r.client.get(r.url + "/pd/pools") as resp:
                pools = await resp.json()
            assert [e["url"] for e in pools["decode"] if not e.get("drained")] == [r.decode[1].url]
            st, chunks = await r.stream(body())
            check(chunks)
            assert r.decode[0].requests == []
    run(go())


def test_membership_from_front_router():
    from connex.front_router import FrontRouter, FrontRouterConfig
    from connex.pd.membership import PDMember, fetch_pools
    from connex.protocol import PDRole

    fr = FrontRouter(FrontRouterConfig(api_address="tcp://127.0.0.1:25655",
                                       event_pub_address="tcp://127.0.0.1:25660"))
    fr.start()
    members = []
    try:
        api = "tcp://127.0.0.1:25655"
        for role, i in ((PDRole.PREFILL, 0), (PDRole.DECODE, 0), (PDRole.DECODE, 1)):
            m = PDMember(api, "m", role, i, f"http://h{role[0]}{i}:80",
                         kv_addr="h:17100" if role == PDRole.PREFILL else None,
                         events="tcp://127.0.0.1:25660")
            assert m.start()
            members.append(m)
        pools = fetch_pools(api, "m")
        assert [x.url for x in pools[PDRole.PREFILL]] == ["http://hp0:80"]
        assert pools[PDRole.PREFILL][0].kv_addr == "h:17100"
        assert [x.url for x in pools[PDRole.DECODE]] == ["http://hd0:80", "http://hd1:80"]
        members.pop().stop()                      # graceful leave (scale down)
        assert [x.url for x in fetch_pools(api, "m")[PDRole.DECODE]] == ["http://hd0:80"]
        assert fetch_pools("tcp://127.0.0.1:1", "m", timeout_ms=200) is None
    finally:
        for m in members:
            m.stop()
        fr.stop()


class SGLRig(Rig):
    def __init__(self, n_prefill=2, n_decode=2, tpot_ms=0, probe=0.05, idle=0.5):
        self.kv = {}
        self.prefill = [fb.FakeSGLangPDBackend("prefill", self.kv) for _ in range(n_prefill)]
        self.decode = [fb.FakeSGLangPDBackend("decode", self.kv, tpot_ms=tpot_ms)
                       for _ in range(n_decode)]
        self.probe, self.idle = probe, idle

    async def __aenter__(self):
        await super().__aenter__()
        self.router.engine = "sglang"
        return self


def test_sglang_mode_pairs_prefill_and_decode_by_room():
    async def go():
        async with SGLRig(1, 1) as r:
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            pq, dq = r.prefill[0].requests[-1], r.decode[0].requests[-1]
            assert pq["bootstrap_room"] == dq["bootstrap_room"]
            assert dq["bootstrap_port"] == 8998 and pq["stream"] is False
    run(go())


def test_sglang_mode_prefill_failure_retried_without_blaming_decode():
    async def go():
        async with SGLRig(2, 1) as r:
            for p in r.prefill:
                p.fail, p.fail_count = ("http500", 0), 1
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            assert r.router.stats["prefill_retries"] >= 1
            assert r.dec_pool.healthy_urls() == [r.decode[0].url]
    run(go())


def test_sglang_mode_decode_dies_midstream_reprefills():
    async def go():
        async with SGLRig(1, 2) as r:
            for d in r.decode:
                d.fail, d.fail_count = ("reset", 4), 1
            st, chunks = await r.stream(body())
            assert st == 200
            check(chunks)
            assert r.prefill[0].prefills == 3      # every resume is a new prefill
    run(go())
