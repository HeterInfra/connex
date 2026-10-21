# SPDX-License-Identifier: Apache-2.0
"""Open-loop trace replay against one or more stage-0 API endpoints.

Replays request traces (TIMESTAMP,ContextTokens,GeneratedTokens)
with the original inter-arrival times. A request succeeds if it streams all
GeneratedTokens tokens within --timeout seconds (the goodput
definition: no client-visible timeout or dropped stream).

    python trace_load.py --trace traces/gamma_qps4.csv \\
        --endpoints http://<stage0-host>:5100 --model cte-model --out run.json

Endpoints may change during a run: they are re-read from --endpoints-file
(one URL per line) every second if given, which the churn driver updates.
Generate a synthetic trace with gen_trace.py.
"""

import argparse
import asyncio
import csv
import json
import random
import time
from datetime import datetime

import aiohttp

WORDS = ("time person year way day thing man world life hand part child eye woman "
         "place work week case point government company number group problem fact").split()


def load_trace(path, limit):
    rows = list(csv.DictReader(open(path)))[:limit or None]
    t0 = datetime.fromisoformat(rows[0]["TIMESTAMP"])
    out = []
    for r in rows:
        off = (datetime.fromisoformat(r["TIMESTAMP"]) - t0).total_seconds()
        out.append((off, int(r["ContextTokens"]), int(r["GeneratedTokens"])))
    # Timestamps have 1 s resolution: spread requests within the same second.
    by_sec = {}
    for off, c, g in out:
        by_sec.setdefault(off, []).append((c, g))
    spread = []
    for sec, reqs in sorted(by_sec.items()):
        for k, (c, g) in enumerate(reqs):
            spread.append((sec + k / len(reqs), c, g))
    return spread


def prompt_of(n_tokens, rng):
    # ~1 token per common English word for Llama-3's tokenizer
    return " ".join(rng.choice(WORDS) for _ in range(max(1, n_tokens)))


async def one(session, url, model, prompt, gen, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0,
            "ignore_eos": True, "stream": True}
    t0 = time.time()
    ttft, n = None, 0
    try:
        async with session.post(url + "/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                return dict(start=t0, end=time.time(), ok=False, err=f"HTTP {r.status}", ep=url)
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
        end = time.time()
        ok = n >= gen
        return dict(start=t0, end=end, ok=ok, ttft=ttft, tokens=n, ep=url,
                    tpot=(end - t0 - ttft) / max(n - 1, 1) if ttft else None,
                    err="" if ok else f"short stream {n}/{gen}")
    except Exception as e:
        return dict(start=t0, end=time.time(), ok=False, ep=url, tokens=n,
                    err=type(e).__name__ + ":" + str(e)[:60])


async def run(args):
    trace = load_trace(args.trace, args.limit)
    if args.repeat > 1:  # replay back-to-back to cover longer churn experiments
        span = trace[-1][0] + 1.0
        trace = [(off + k * span, c, g) for k in range(args.repeat) for off, c, g in trace]
    rng = random.Random(0)
    endpoints = args.endpoints.split(",")
    results = []

    async def refresh_endpoints():
        nonlocal endpoints
        while True:
            await asyncio.sleep(1)
            try:
                eps = [l.strip() for l in open(args.endpoints_file) if l.strip()]
                if eps:
                    endpoints = eps
            except OSError:
                pass

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        if args.endpoints_file:
            refresher = asyncio.create_task(refresh_endpoints())
        t0 = time.time()
        tasks = []
        for i, (off, ctx, gen) in enumerate(trace):
            delay = t0 + off - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            url = endpoints[i % len(endpoints)]
            tasks.append(asyncio.create_task(
                one(session, url, args.model, prompt_of(ctx, rng), gen, args.timeout)))
        results = await asyncio.gather(*tasks)
        if args.endpoints_file:
            refresher.cancel()
    return t0, results


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None


def summarize(t0, res):
    ok = [r for r in res if r["ok"]]
    lat = [r["end"] - r["start"] for r in ok]
    return {
        "requests": len(res), "goodput": round(len(ok) / max(len(res), 1), 4),
        "e2e_mean_s": round(sum(lat) / len(lat), 3) if lat else None,
        "e2e_p50_s": pct(lat, 50), "e2e_p99_s": pct(lat, 99),
        "ttft_mean_ms": round(1000 * sum(r["ttft"] for r in ok) / len(ok), 1) if ok else None,
        "tpot_mean_ms": round(1000 * sum(r["tpot"] for r in ok if r["tpot"]) / len(ok), 2) if ok else None,
        "errors": sorted({r["err"] for r in res if not r["ok"]})[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--endpoints", required=True, help="comma-separated stage-0 base URLs")
    ap.add_argument("--endpoints-file", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0, res = asyncio.run(run(args))
    s = summarize(t0, res)
    for r in res:
        r["rel_start"] = round(r["start"] - t0, 3)
    json.dump({"summary": s, "config": vars(args), "requests": res}, open(args.out, "w"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
