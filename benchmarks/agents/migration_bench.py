# SPDX-License-Identifier: Apache-2.0
"""Agent session migration on real engines: next-turn TTFT and correctness.

Two (or more) OpenAI-compatible engines with automatic prefix caching
(vLLM or SGLang). For each context length L, a session runs one long turn
on its placement (context of ~L tokens, built from a fixed text), then:

* sticky: the next turn on the same instance (prefix cache hit)
* cold: migrate with mode "none" (cut over only); the next turn re-prefills
  on the target on demand
* reprefill: migrate with the re-prefill mover (the target prefills the
  context before traffic moves); report the migration time and the next
  turn's TTFT on the target
* inflight: migrate in the middle of a long greedy generation; the client
  stream must equal the unmigrated output

Every next turn is greedy and must produce the same text in all variants.

    python migration_bench.py --engines http://h:8201,http://h:8202 \\
        --model NAME --tokenizer PATH --lengths 8192,32768 --out mig.jsonl
"""
import argparse
import asyncio
import json
import time

import aiohttp
from aiohttp import web

from connex.agents import SessionProxy
from connex.serving.proxy import Pool

FILLER = ("The archive lists shipments by date, origin and weight. Each entry notes the "
          "carrier, the inspection result and any delay with its cause. ")


def context(tok, n, salt):
    """~n tokens; ``salt`` first, so no two sessions share a cached prefix."""
    head = f"Session {salt}. Archive copy {abs(hash(salt)) % 10**9}.\n"
    base = tok(FILLER).input_ids
    ids = (base * (n // len(base) + 1))[:n - len(tok(head).input_ids)]
    return head + tok.decode(ids)


class Client:
    def __init__(self, url):
        self.url = url
        self.s = aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=1800))

    async def turn(self, sid, model, prompt, max_tokens, on_token=None):
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, "session_id": sid, "ignore_eos": True}
        t0 = time.perf_counter()
        ttft, text, last, gap = None, [], None, 0.0
        async with self.s.post(self.url + "/v1/completions", json=body) as r:
            assert r.status == 200, await r.text()
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                c = json.loads(line[6:])
                if c.get("choices") and c["choices"][0].get("text"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        gap = max(gap, now - last)
                    last = now
                    text.append(c["choices"][0]["text"])
                    if on_token:
                        await on_token(len(text))
        return {"ttft_s": ttft, "total_s": time.perf_counter() - t0, "text": "".join(text),
                "max_gap_s": gap}

    async def close(self):
        await self.s.close()


async def run(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    urls = [u.rstrip("/") for u in a.engines.split(",")]
    proxy = SessionProxy(Pool(urls), idle_timeout=600, probe_interval=0,
                         token_ids=not a.text_resume)
    runner = web.AppRunner(proxy.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    purl = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    cl = Client(purl)
    out = []
    try:
        for L in [int(x) for x in a.lengths.split(",")]:
            q1 = "\n\nQuestion: summarise the archive in one sentence.\nAnswer:"
            q2 = "\n\nQuestion: which fields does each entry note?\nAnswer:"
            for variant in ("sticky", "cold", "reprefill"):
                for rep in range(a.reps):
                    sid = f"{variant}-{L}-{rep}"
                    ctx = context(tok, L, f"{a.salt}-{sid}")
                    # turn 1: places the session and fills its prefix cache
                    first = await cl.turn(sid, a.model, ctx + q1, a.turn_tokens)
                    s = proxy.sessions[sid]
                    src = s.url
                    dst = next(u for u in urls if u != src)
                    hist = ctx + q1 + first["text"]
                    rec = {"len": L, "variant": variant, "rep": rep, "turn1_ttft_s": first["ttft_s"]}
                    if variant != "sticky":
                        m = await proxy.migrate(sid, dst, mode="none" if variant == "cold"
                                                else "reprefill")
                        rec.update(migrate_s=m["prepare_s"], mode=m["mode"])
                    nxt = await cl.turn(sid, a.model, hist + q2, a.turn_tokens)
                    rec.update(next_ttft_s=nxt["ttft_s"], next_total_s=nxt["total_s"],
                               placement_changed=proxy.sessions[sid].url != src,
                               text=nxt["text"])
                    out.append(rec)
                    print(json.dumps({k: v for k, v in rec.items() if k != "text"}), flush=True)
            texts = {r["text"] for r in out if r["len"] == L}
            print(json.dumps({"len": L, "next_turn_identical": len(texts) == 1}), flush=True)
            # in-flight migration of a long greedy generation
            ctx = context(tok, L, f"{a.salt}-inflight-{L}")
            ref = await cl.turn(f"ref-{L}", a.model, ctx + q2, a.inflight_tokens)
            sid = f"inflight-{L}"
            # same instance as the reference, so the target starts cold
            proxy.get_session(sid).url = proxy.sessions[f"ref-{L}"].url
            await cl.turn(sid, a.model, ctx + q1, 1)
            dst = next(u for u in urls if u != proxy.sessions[sid].url)
            moved = {}

            async def on_tok(n):     # migrate concurrently; the client keeps reading
                if n == a.inflight_tokens // 4 and not moved:
                    moved["task"] = asyncio.create_task(
                        proxy.migrate(sid, dst, mode=a.inflight_mode))
            res = await cl.turn(sid, a.model, ctx + q2, a.inflight_tokens, on_token=on_tok)
            moved["rec"] = await moved["task"]
            rec = {"len": L, "variant": "inflight", "identical": res["text"] == ref["text"],
                   "migrate_s": moved["rec"]["prepare_s"], "inflight_moved":
                   moved["rec"]["inflight_moved"], "mode": moved["rec"]["mode"],
                   "total_s": res["total_s"], "ref_total_s": ref["total_s"],
                   "max_gap_s": res["max_gap_s"], "ref_max_gap_s": ref["max_gap_s"]}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        await cl.close()
        await runner.cleanup()
    if a.out:
        with open(a.out, "a") as f:
            for r in out:
                f.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--turn-tokens", type=int, default=32)
    ap.add_argument("--inflight-tokens", type=int, default=256)
    ap.add_argument("--inflight-mode", default="reprefill")
    ap.add_argument("--text-resume", action="store_true",
                    help="resume from text (engines without token-id logprobs, e.g. SGLang)")
    ap.add_argument("--salt", default=str(int(time.time())))
    ap.add_argument("--out", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
