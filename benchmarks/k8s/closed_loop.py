# SPDX-License-Identifier: Apache-2.0
"""Closed-loop throughput: N clients, fixed prompt/output length, D seconds.
    python closed_loop.py --model cte-model --clients 64 --max-tokens 64 --duration 60"""
import argparse, asyncio, json, os, time
import aiohttp

async def main(a):
    t_end = time.time() + a.duration
    toks, reqs, lat = 0, 0, []
    async def client(i):
        nonlocal toks, reqs
        async with aiohttp.ClientSession() as s:
            while time.time() < t_end:
                body = {"model": a.model, "prompt": f"Client {i}: tell me about topic {reqs}.",
                        "max_tokens": a.max_tokens, "temperature": 0, "ignore_eos": True}
                t = time.time()
                try:
                    async with s.post(a.url + "/v1/completions", json=body,
                                      timeout=aiohttp.ClientTimeout(total=120)) as r:
                        d = await r.json()
                    toks += d["usage"]["completion_tokens"]; reqs += 1; lat.append(time.time() - t)
                except Exception:
                    pass
    t0 = time.time()
    await asyncio.gather(*[client(i) for i in range(a.clients)])
    el = time.time() - t0; lat.sort()
    print(json.dumps({"clients": a.clients, "tok_s": round(toks / el, 1), "req_s": round(reqs / el, 2),
                      "lat_p50": round(lat[len(lat)//2], 2) if lat else None}))

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("CONNEX_API_URL", "http://127.0.0.1:5100"),
                help="stage-0 OpenAI API base URL (env CONNEX_API_URL)")
ap.add_argument("--model", required=True)
ap.add_argument("--clients", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--duration", type=float, default=60)
asyncio.run(main(ap.parse_args()))
