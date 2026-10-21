# SPDX-License-Identifier: Apache-2.0
"""Steady-state overhead of connex.serving.proxy, measured on CPU with the fake backend.

Starts a fake vLLM server (chunks carry their send time) and the proxy as
separate processes, then streams the same requests directly and through the
proxy, alternating. All streams start together (closed loop, equal length),
so the first chunks of each pass hit the request-arrival burst; steady_p99
excludes each stream's first 4 chunks. Per-chunk latency = client receive time - backend send
time; the difference between the two paths is the proxy's added latency.

    python benchmarks/proxy_overhead.py --requests 200 --concurrency 16 --tokens 64 --tpot-ms 10
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def wait_up(url):
    async with aiohttp.ClientSession(trust_env=False) as s:
        for _ in range(100):
            try:
                async with s.get(url + "/v1/models") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"{url} not up")


async def load(url, n, conc, tokens):
    lat, ttft, steady = [], [], []
    sem = asyncio.Semaphore(conc)
    body = {"model": "fake", "prompt": "w5 w9 w13", "max_tokens": tokens, "temperature": 0,
            "stream": True}

    async def one(s):
        async with sem:
            t0 = time.time()
            first = True
            i = 0
            async with s.post(url + "/v1/completions", json=body) as r:
                async for line in r.content:
                    now = time.time()
                    if line.startswith(b"data: {"):
                        c = json.loads(line[6:])
                        if "_t" in c:
                            lat.append(now - c["_t"])
                            if i >= 4:   # past the request-arrival burst
                                steady.append(now - c["_t"])
                            i += 1
                            if first:
                                ttft.append(now - t0)
                                first = False

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), trust_env=False) as s:
        await asyncio.gather(*(one(s) for _ in range(n)))
    return lat, ttft, steady


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tpot-ms", type=float, default=10)
    ap.add_argument("--backend-port", type=int, default=15911)
    ap.add_argument("--proxy-port", type=int, default=15912)
    a = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": ROOT}
    be = f"http://127.0.0.1:{a.backend_port}"
    px = f"http://127.0.0.1:{a.proxy_port}"
    procs = [
        subprocess.Popen([sys.executable, "-m", "connex.serving.fake_backend", "--port",
                          str(a.backend_port), "--tpot-ms", str(a.tpot_ms), "--stamp"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, "-m", "connex.serving.proxy", "--host", "127.0.0.1",
                          "--port", str(a.proxy_port), "--endpoints", be, "--log-level", "WARNING"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        asyncio.run(wait_up(be))
        asyncio.run(wait_up(px))
        res = {"direct": ([], [], []), "proxy": ([], [], [])}
        asyncio.run(load(px, a.requests // 4 or 1, a.concurrency, a.tokens))   # warmup
        for name in ("direct", "proxy", "direct", "proxy"):
            for acc, xs in zip(res[name], asyncio.run(load(be if name == "direct" else px,
                                                           a.requests // 2, a.concurrency, a.tokens))):
                acc.extend(xs)
        out = {}
        for name, (lat, ttft, steady) in res.items():
            out[name] = {"chunks": len(lat), "p50_ms": round(pct(lat, 50) * 1e3, 3),
                         "p99_ms": round(pct(lat, 99) * 1e3, 3),
                         "steady_p99_ms": round(pct(steady, 99) * 1e3, 3),
                         "ttft_p50_ms": round(pct(ttft, 50) * 1e3, 3)}
        for k in ("p50_ms", "p99_ms", "steady_p99_ms", "ttft_p50_ms"):
            out[f"added_{k}"] = round(out["proxy"][k] - out["direct"][k], 3)
        print(json.dumps({"config": vars(a), **out}))
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
