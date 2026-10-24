#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load generator and checker for prefill-decode disaggregation.

Prompts are deterministic lists of token ids (seeded), sent as
``/v1/completions`` with greedy decoding and ``ignore_eos``, so every run of
the same (seed, index) must produce the same tokens on any correct deployment.

    # reference outputs from a non-disaggregated instance
    pd_bench.py ref  --url http://127.0.0.1:8300 --n 64 --out ref.json
    # correctness: the PD deployment must reproduce them exactly
    pd_bench.py check --url http://127.0.0.1:8000 --ref ref.json
    # load: open loop (--qps) or closed loop (--concurrency); with --ref the
    # outputs are checked too (goodput = complete and correct)
    pd_bench.py load --url http://127.0.0.1:8000 --qps 4 --duration 60 --out run.json
    # teacher-forced check of a run (needs --keep-ids): every generated token
    # must be the argmax, within --tol nats, given its own prefix, scored on a
    # non-disaggregated instance
    pd_bench.py score --url http://127.0.0.1:8300 --file run.json

Exact match against a reference run is a strict check that also fails on
benign near-ties (batching changes floating-point results; two runs of the
same non-disaggregated instance at different concurrency differ too). The
score check is the primary one: a decode that used wrong or missing KV
produces tokens far from the argmax.

Output JSON: per-request ttft/tpot/e2e/ok/correct and a summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


TOPICS = ["the history of the Roman empire", "how photosynthesis works", "the rules of chess",
          "the life cycle of stars", "how compilers work", "the causes of the French revolution",
          "the water cycle", "how vaccines train the immune system", "the architecture of a CPU",
          "the migration of birds", "the invention of the printing press",
          "how neural networks learn", "plate tectonics", "the economics of inflation",
          "the structure of DNA", "the geography of South America"]


def prompts(n, prompt_len, seed, vocab=(1000, 30000), text=False):
    if text:   # natural language: fewer near-ties than random token ids
        return [f"Write a detailed, multi-paragraph essay about {TOPICS[i % len(TOPICS)]}"
                f" (essay {i}, seed {seed}):\n\n" for i in range(n)]
    out = []
    for i in range(n):
        r = random.Random(seed * 1_000_003 + i)
        out.append([r.randint(*vocab) for _ in range(prompt_len)])
    return out


async def one(session, url, model, prompt, max_tokens, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "logprobs": 0, "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = None
    ids, times = [], []
    err = None
    try:
        async with session.post(url + "/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:200]}"
            else:
                async for line in r.content:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    c = json.loads(line[6:])
                    if "error" in c:
                        err = str(c["error"])[:200]
                        break
                    for ch in c.get("choices") or []:
                        toks = ch.get("token_ids") or [
                            int(t.split(":", 1)[1]) if t.startswith("token_id:") else t
                            for t in (ch.get("logprobs") or {}).get("tokens") or []]
                        now = time.perf_counter()
                        for t in toks:
                            ids.append(t)
                            times.append(now)
                        if toks and first is None:
                            first = now
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()
    ok = err is None and len(ids) == max_tokens
    return {"ok": ok, "err": err, "n": len(ids), "ids": ids,
            "ttft": (first - t0) if first else None, "e2e": t1 - t0,
            "tpot": ((times[-1] - first) / (len(ids) - 1)) if first and len(ids) > 1 else None,
            # longest stall between tokens: the visible cost of a failover
            "max_gap": max((b - a for a, b in zip(times, times[1:])), default=None),
            "start": t0}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(res, wall, max_tokens):
    ok = [r for r in res if r["ok"]]
    s = {"requests": len(res), "ok": len(ok), "failed": len(res) - len(ok),
         "goodput": len([r for r in ok if r.get("correct", True)]) / max(1, len(res)),
         "wall_s": wall, "throughput_tok_s": sum(r["n"] for r in res) / wall if wall else None,
         "req_s": len(ok) / wall if wall else None}
    for k in ("ttft", "tpot", "e2e", "max_gap"):
        v = [r[k] for r in ok]
        s[k] = {"mean": statistics.mean([x for x in v if x is not None]) if any(v) else None,
                "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99), "max": pct(v, 100)}
    if any("correct" in r for r in res):
        s["incorrect"] = len([r for r in ok if not r.get("correct", True)])
    errs = {}
    for r in res:
        if r["err"]:
            errs[r["err"][:80]] = errs.get(r["err"][:80], 0) + 1
    s["errors"] = errs
    return s


async def run_load(a):
    ps = prompts(a.n, a.prompt_len, a.seed, text=a.text)
    ref = json.load(open(a.ref))["outputs"] if a.ref else None
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, trust_env=False) as s:
        res = [None] * len(ps)
        t0 = time.perf_counter()

        async def go(i):
            res[i] = await one(s, a.url, a.model, ps[i], a.max_tokens, a.timeout)
            res[i]["i"] = i

        if a.concurrency:
            q = asyncio.Queue()
            for i in range(len(ps)):
                q.put_nowait(i)

            async def worker():
                while not q.empty():
                    i = q.get_nowait()
                    if a.duration and time.perf_counter() - t0 > a.duration:
                        return
                    await go(i)

            await asyncio.gather(*(worker() for _ in range(a.concurrency)))
        else:
            rng = random.Random(a.seed)
            tasks = []
            for i in range(len(ps)):
                if a.duration and time.perf_counter() - t0 > a.duration:
                    break
                tasks.append(asyncio.create_task(go(i)))
                await asyncio.sleep(rng.expovariate(a.qps))
            await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    res = [r for r in res if r is not None]
    if ref:
        for r in res:
            r["correct"] = r["ok"] and r["ids"] == ref[r["i"]][:a.max_tokens]
    return res, wall


async def run_ref(a):
    ps = prompts(a.n, a.prompt_len, a.seed, text=a.text)
    async with aiohttp.ClientSession(trust_env=False) as s:
        sem = asyncio.Semaphore(a.concurrency or 8)

        async def go(p):
            async with sem:
                return await one(s, a.url, a.model, p, a.max_tokens, a.timeout)

        res = await asyncio.gather(*(go(p) for p in ps))
    bad = [i for i, r in enumerate(res) if not r["ok"]]
    if bad:
        raise SystemExit(f"reference run failed for {len(bad)} prompts: {res[bad[0]]['err']}")
    return [r["ids"] for r in res]


async def run_score(a):
    d = json.load(open(a.file))
    args = d["args"]
    ps = prompts(args["n"], args["prompt_len"], args["seed"], text=args.get("text", False))
    tot = bad = bad_reqs = 0
    worst, per = 0.0, []
    sem = asyncio.Semaphore(8)
    async with aiohttp.ClientSession(trust_env=False) as s:
        async def one_score(r):
            gen = r.get("ids") or []
            if isinstance(ps[r["i"]], str):
                async with s.post(a.url + "/tokenize",
                                  json={"model": a.model, "prompt": ps[r["i"]]}) as resp:
                    ps[r["i"]] = (await resp.json())["tokens"]
            body = {"model": a.model, "prompt": ps[r["i"]] + gen, "max_tokens": 1,
                    "temperature": 0, "logprobs": 1, "echo": True,
                    "return_tokens_as_token_ids": True}
            async with sem:
                async with s.post(a.url + "/v1/completions", json=body) as resp:
                    return r, gen, (await resp.json())["choices"][0]["logprobs"]

        out = await asyncio.gather(*(one_score(r) for r in d["requests"] if r.get("ok")))
    for r, gen, lp in out:
        n0 = len(ps[r["i"]])
        nb, first = 0, None
        for k in range(len(gen)):
            j = n0 + k
            gap = max(lp["top_logprobs"][j].values()) - lp["token_logprobs"][j]
            worst = max(worst, gap)
            if gap > a.tol:
                nb += 1
                first = k if first is None else first
        tot += len(gen)
        bad += nb
        bad_reqs += nb > 0
        if nb:
            per.append((r["i"], nb, first))
    return {"file": a.file, "requests": len(out), "tokens": tot, "non_argmax_tokens": bad,
            "requests_with_non_argmax": bad_reqs, "worst_gap": round(worst, 3), "tol": a.tol,
            "detail": per[:20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["ref", "check", "load", "score"])
    ap.add_argument("--file", default=None, help="score: a run saved with --keep-ids")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="natural-language prompts")
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-ids", action="store_true")
    a = ap.parse_args()
    if a.mode == "score":
        r = asyncio.run(run_score(a))
        print(json.dumps(r))
        if a.out:
            json.dump(r, open(a.out, "w"))
        return
    if a.mode == "ref":
        outs = asyncio.run(run_ref(a))
        json.dump({"args": vars(a), "outputs": outs}, open(a.out or "ref.json", "w"))
        print(f"wrote {len(outs)} reference outputs")
        return
    if a.mode == "check":
        a.concurrency = a.concurrency or 8
    res, wall = asyncio.run(run_load(a))
    s = summary(res, wall, a.max_tokens)
    print(json.dumps(s, indent=1))
    if a.out:
        if not a.keep_ids:
            for r in res:
                r.pop("ids", None)
        json.dump({"args": vars(a), "summary": s, "requests": res}, open(a.out, "w"))
    if a.mode == "check" and (s["failed"] or s.get("incorrect")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
