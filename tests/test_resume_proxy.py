# SPDX-License-Identifier: Apache-2.0
"""Request retry/resume proxy against fake vLLM backends that die mid-stream."""
import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web

from connex.serving import fake_backend as fb
from connex.serving.proxy import Pool, Proxy

PROMPT = "w5 w9 w13 w21"
MAXT = 12


def chat_reference(messages, n):
    seq = fb.chat_ids(messages)
    out = []
    for _ in range(n):
        t = fb.next_token(seq)
        seq.append(t)
        out.append(t)
    return len(fb.chat_ids(messages)), "".join(fb.detok(t) for t in out)


def run(coro):
    return asyncio.run(coro)


class Rig:
    def __init__(self, n=2, idle=0.3, probe=0, token_ids=True, ttft_ms=0, max_attempts=4):
        self.backends = [fb.FakeBackend(ttft_ms=ttft_ms) for _ in range(n)]
        self.kw = dict(idle_timeout=idle, probe_interval=probe, token_ids=token_ids,
                       max_attempts=max_attempts)

    async def __aenter__(self):
        urls = [await b.start() for b in self.backends]
        self.proxy = Proxy(Pool(urls), **self.kw)
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

    async def stream(self, path, body):
        body = {**body, "stream": True}
        chunks = []
        async with self.client.post(self.url + path, json=body) as r:
            if r.status != 200:
                return r.status, await r.json()
            async for line in r.content:
                if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                    chunks.append(json.loads(line[6:]))
        return 200, chunks

    async def post(self, path, body):
        async with self.client.post(self.url + path, json=body) as r:
            return r.status, await r.json()


def comp_body(**kw):
    return {"model": "fake", "prompt": PROMPT, "max_tokens": MAXT, "temperature": 0,
            "ignore_eos": True, "stream_options": {"include_usage": True}, **kw}


def text_of(chunks, chat=False):
    if chat:
        return "".join((c["choices"][0].get("delta") or {}).get("content") or ""
                       for c in chunks if c.get("choices"))
    return "".join(c["choices"][0]["text"] for c in chunks if c.get("choices"))


def final_usage(chunks):
    us = [c for c in chunks if not c.get("choices") and c.get("usage")]
    assert len(us) == 1
    return us[0]["usage"]


def check_completion(chunks):
    ids, ref = fb.reference(PROMPT, MAXT)
    assert text_of(chunks) == ref
    u = final_usage(chunks)
    assert u == {"prompt_tokens": len(fb.tokenize(PROMPT)), "completion_tokens": MAXT,
                 "total_tokens": len(fb.tokenize(PROMPT)) + MAXT}
    assert len({c["id"] for c in chunks}) == 1          # one logical stream
    assert all(c["choices"][0]["logprobs"] is None for c in chunks if c.get("choices"))
    assert all("token_ids" not in c["choices"][0] for c in chunks if c.get("choices"))
    fins = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")]
    assert fins[-1] == "length" and all(f is None for f in fins[:-1])


def test_steady_state_passthrough():
    async def go():
        async with Rig() as rig:
            for _ in range(4):
                st, chunks = await rig.stream("/v1/completions", comp_body())
                assert st == 200
                check_completion(chunks)
            assert rig.proxy.stats["retries"] == rig.proxy.stats["resumes"] == 0
            # injected fields: ids via logprobs, usage stats; client never sees them
            req = rig.backends[0].requests[0]
            assert req["return_tokens_as_token_ids"] and req["logprobs"] == 0
    run(go())


def test_no_usage_unless_requested():
    async def go():
        async with Rig() as rig:
            body = comp_body()
            body.pop("stream_options")
            st, chunks = await rig.stream("/v1/completions", body)
            assert all(c.get("choices") for c in chunks)
            assert all(c.get("usage") is None for c in chunks)
            body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            rig.backends[0].fail = rig.backends[1].fail = ("reset", 4)
            rig.backends[0].fail_count = rig.backends[1].fail_count = 1
            st, chunks = await rig.stream("/v1/completions", body)
            ks = [c["usage"]["completion_tokens"] for c in chunks if c.get("choices")]
            assert ks == list(range(1, MAXT + 1))     # monotone across the splice
    run(go())


@pytest.mark.parametrize("fail", [("reset", 0), ("http500", 0)])
def test_zero_token_failure_retried(fail):
    async def go():
        async with Rig() as rig:
            rig.backends[0].fail = fail                    # replica 0 always broken
            for _ in range(3):
                st, chunks = await rig.stream("/v1/completions", comp_body())
                assert st == 200
                check_completion(chunks)
            assert rig.proxy.stats["resumes"] == 0
    run(go())


def test_connect_refused_retried():
    async def go():
        async with Rig() as rig:
            await rig.backends[0].stop()                   # port closed
            for _ in range(3):
                st, chunks = await rig.stream("/v1/completions", comp_body())
                check_completion(chunks)
    run(go())


@pytest.mark.parametrize("k", [1, 5, MAXT - 1])
@pytest.mark.parametrize("token_ids", [True, False])
def test_midstream_reset_resumed(k, token_ids):
    async def go():
        async with Rig(token_ids=token_ids) as rig:
            for b in rig.backends:
                b.fail, b.fail_count = ("reset", k), 1
            st, chunks = await rig.stream("/v1/completions", comp_body())
            check_completion(chunks)
            assert rig.proxy.stats["resumes"] >= 1
            resumed = [r for b in rig.backends for r in b.requests if r["max_tokens"] != MAXT]
            assert resumed and resumed[0]["max_tokens"] == MAXT - k
            if token_ids:
                ids, _ = fb.reference(PROMPT, MAXT)
                assert resumed[0]["prompt"] == fb.tokenize(PROMPT) + ids[:k]
            else:
                assert isinstance(resumed[0]["prompt"], str)
    run(go())


def test_two_failures_at_different_points():
    async def go():
        async with Rig(n=3) as rig:
            for b, k in zip(rig.backends, (3, 7, 10)):
                b.fail, b.fail_count = ("reset", k), 1
            st, chunks = await rig.stream("/v1/completions", comp_body())
            check_completion(chunks)
            assert rig.proxy.stats["resumes"] >= 2
    run(go())


def test_stall_midstream_idle_timeout():
    async def go():
        async with Rig(idle=0.3) as rig:
            for b in rig.backends:
                b.fail, b.fail_count = ("stall", 4), 1
            t0 = time.time()
            st, chunks = await rig.stream("/v1/completions", comp_body())
            check_completion(chunks)
            assert time.time() - t0 < 2.0
    run(go())


def test_slow_prefill_not_cut():
    async def go():
        async with Rig(idle=0.2, ttft_ms=800) as rig:
            st, chunks = await rig.stream("/v1/completions", comp_body())
            check_completion(chunks)
            assert rig.proxy.stats["retries"] == rig.proxy.stats["resumes"] == 0
    run(go())


def test_dead_before_first_token_detected_by_blacklist():
    async def go():
        async with Rig(idle=0.2) as rig:
            rig.backends[0].fail = rig.backends[1].fail = ("stall", 0)
            rig.backends[0].fail_count = rig.backends[1].fail_count = 1
            task = asyncio.create_task(rig.stream("/v1/completions", comp_body()))
            await asyncio.sleep(0.3)
            busy = [e for e in rig.proxy.pool.eps.values() if e.inflight]
            assert len(busy) == 1
            for b in rig.backends:
                b.fail = None                                  # the other replica is fine
            rig.proxy.pool.mark_bad(busy[0], "test: replica declared dead")
            st, chunks = await asyncio.wait_for(task, 3)
            check_completion(chunks)
            assert rig.proxy.stats["retries"] == 1
    run(go())


def test_prober_blacklists_closed_port():
    async def go():
        async with Rig(probe=0.1) as rig:
            await rig.backends[1].stop()
            await asyncio.sleep(0.4)
            eps = list(rig.proxy.pool.eps.values())
            now = time.monotonic()
            assert eps[0].healthy(now) and not eps[1].healthy(now)
            st, chunks = await rig.stream("/v1/completions", comp_body())
            check_completion(chunks)
            assert rig.proxy.stats["retries"] == 0           # never sent to the dead one
    run(go())


def test_non_streaming_resumed():
    async def go():
        async with Rig() as rig:
            for b in rig.backends:
                b.fail, b.fail_count = ("reset", 6), 1
            st, out = await rig.post("/v1/completions", {**comp_body(), "stream": False})
            assert st == 200
            assert out["choices"][0]["text"] == fb.reference(PROMPT, MAXT)[1]
            assert out["choices"][0]["finish_reason"] == "length"
            assert out["usage"]["completion_tokens"] == MAXT
            assert out["usage"]["prompt_tokens"] == len(fb.tokenize(PROMPT))
    run(go())


@pytest.mark.parametrize("stream", [True, False])
def test_chat_resumed(stream):
    msgs = [{"role": "user", "content": "w7 w8 w9"}]
    body = {"model": "fake", "messages": msgs, "max_tokens": MAXT, "temperature": 0,
            "stream_options": {"include_usage": True}}
    n_prompt, ref = chat_reference(msgs, MAXT)

    async def go():
        async with Rig() as rig:
            for b in rig.backends:
                b.fail, b.fail_count = ("reset", 5), 1
            if stream:
                st, chunks = await rig.stream("/v1/chat/completions", body)
                assert text_of(chunks, chat=True) == ref
                roles = [c for c in chunks if c.get("choices") and "role" in c["choices"][0]["delta"]]
                assert len(roles) == 1
                u = final_usage(chunks)
            else:
                st, out = await rig.post("/v1/chat/completions", {**body, "stream": False})
                assert out["choices"][0]["message"]["content"] == ref
                u = out["usage"]
            assert u["prompt_tokens"] == n_prompt and u["completion_tokens"] == MAXT
            resumed = [r for b in rig.backends for r in b.requests if r.get("continue_final_message")]
            assert resumed and resumed[0]["max_tokens"] == MAXT - 5
            assert resumed[0]["add_generation_prompt"] is False
    run(go())


def test_all_replicas_dead():
    async def go():
        async with Rig(max_attempts=3) as rig:
            for b in rig.backends:
                b.fail = ("reset", 0)
            st, out = await rig.stream("/v1/completions", comp_body())
            assert st == 502
            # replica 0 always dies after 4 tokens, replica 1 is gone: 3 attempts
            # (r0: 4 tokens, r1: refused, r0: 4 more) then the stream ends short
            rig.backends[0].fail = ("reset", 4)
            await rig.backends[1].stop()
            st, chunks = await rig.stream("/v1/completions", comp_body())
            assert st == 200 and "error" in chunks[-1]
            assert text_of(chunks[:-1]) == fb.reference(PROMPT, 8)[1]
    run(go())


def test_client_error_passthrough():
    async def go():
        async with Rig() as rig:
            st, out = await rig.stream("/v1/completions", comp_body(max_tokens=-1))
            assert st == 400 and out["error"] == "max_tokens must be positive"
            assert rig.proxy.stats["retries"] == 0
            assert sum(len(b.requests) for b in rig.backends) == 1
    run(go())


def test_batch_prompt_not_resumed():
    async def go():
        async with Rig() as rig:
            for b in rig.backends:
                b.fail, b.fail_count = ("reset", 3), 1
            body = comp_body(n=2)
            st, chunks = await rig.stream("/v1/completions", body)
            assert "error" in chunks[-1]                # cut, but visibly
            assert rig.proxy.stats["resumes"] == 0
    run(go())


def test_logprobs_zero_is_passed_through():
    """logprobs=0 (token ids only, no alternatives) is a logprobs request:
    0 == False must not make the proxy strip the client's logprobs."""
    async def go():
        async with Rig(1) as r:
            st, chunks = await r.stream("/v1/completions",
                                        comp_body(logprobs=0, return_tokens_as_token_ids=True))
            assert st == 200
            lps = [c["choices"][0]["logprobs"] for c in chunks if c.get("choices")]
            assert lps and all(lp and lp["tokens"][0].startswith("token_id:") for lp in lps)
    run(go())
