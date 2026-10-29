# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Kill a downstream stage replica under load and measure the handover.

C closed-loop clients stream greedy completions (ignore_eos, fixed length)
from stage 0. KILL_AFTER seconds in, a shell command kills a replica. For
every request in flight at the kill, the handover is the time from the kill
to its first token after the kill; the stall is its largest gap between two
tokens around the kill. ``no_token_window_s`` is the service-level gap: the
longest time after the kill during which no stream got a token (requests
that start after the kill and wait for their first token count too, unlike
the per-request stall). Requests that fail are counted (they should be 0).

    python failover_load.py --url http://127.0.0.1:8200 --clients 32 \\
        --duration 40 --kill "kill -9 -- -<pgid>" --kill-after 15 --out kill.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time

import aiohttp

TOPICS = ["the history of the Roman empire", "how photosynthesis works", "the rules of chess",
          "the life cycle of stars", "how compilers work", "the water cycle",
          "how vaccines train the immune system", "the architecture of a CPU",
          "the migration of birds", "plate tectonics", "the structure of DNA",
          "the theory of relativity", "how the internet routes packets"]


async def client(s, a, cid, stop, out):
    k = 0
    while time.time() < stop:
        prompt = (f"Write a detailed essay about {TOPICS[(cid + k) % len(TOPICS)]} "
                  f"(client {cid}, request {k}):\n\n")
        k += 1
        body = {"model": a.model, "prompt": prompt, "max_tokens": a.max_tokens,
                "temperature": 0, "ignore_eos": True, "stream": True}
        t0 = time.time()
        times = []
        ok = False
        err = None
        try:
            async with s.post(a.url + "/v1/completions", json=body,
                              timeout=aiohttp.ClientTimeout(total=600)) as r:
                if r.status != 200:
                    err = f"HTTP {r.status}"
                else:
                    async for line in r.content:
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        if line == b"data: [DONE]":
                            ok = True
                            break
                        d = json.loads(line[5:])
                        if d.get("choices") and d["choices"][0].get("text") is not None:
                            times.append(time.time())
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
        out.append({"cid": cid, "t0": t0, "t_end": time.time(), "ok": ok and len(times) > 0,
                    "n": len(times), "times": times, "err": err})


async def main_async(a):
    start = time.time()
    stop = start + a.duration
    out: list = []
    async with aiohttp.ClientSession(trust_env=False) as s:
        tasks = [asyncio.create_task(client(s, a, c, stop, out)) for c in range(a.clients)]
        t_kill = None
        probes: list = []
        if a.kill:
            await asyncio.sleep(a.kill_after)
            t_kill = time.time()
            rc = subprocess.run(a.kill, shell=True).returncode
            print(f"kill rc={rc} at +{t_kill - start:.1f}s", flush=True)
            tasks.append(asyncio.create_task(probe(s, a, stop, probes)))
        await asyncio.gather(*tasks)
    return start, t_kill, out, probes


async def probe(s, a, stop, probes):
    """Is the service answering? A 1-token completion every 0.25 s."""
    while time.time() < stop + 120:
        t = time.time()
        ok = False
        try:
            async with s.post(a.url + "/v1/completions",
                              json={"model": a.model, "prompt": "Hi", "max_tokens": 1},
                              timeout=aiohttp.ClientTimeout(total=5)) as r:
                ok = r.status == 200
        except Exception:  # noqa: BLE001
            pass
        probes.append((t, ok))
        if ok and t > stop:
            return
        await asyncio.sleep(0.25)


def no_token_window(out, t_kill, horizon=60.0):
    """Longest interval, from the last token before the kill to 60 s after
    it, with no token on any stream."""
    ts = sorted(t for r in out for t in r["times"])
    before = [t for t in ts if t <= t_kill]
    after = [t for t in ts if t_kill < t <= t_kill + horizon]
    seq = before[-1:] + after
    if len(seq) < 2:
        return None
    return round(max(b - x for x, b in zip(seq, seq[1:])), 3)


def summarize(start, t_kill, out, a, probes=()):
    done = [r for r in out if r["ok"]]
    failed = [r for r in out if not r["ok"]]
    toks = sum(r["n"] for r in done)
    # steady state: requests that finished before the kill (or all, no kill)
    pre = [r for r in done if t_kill is None or r["t_end"] < t_kill]
    tpot = [(r["times"][-1] - r["times"][0]) / (r["n"] - 1) for r in pre if r["n"] > 1]
    ttft = [r["times"][0] - r["t0"] for r in pre if r["times"]]
    res = {"clients": a.clients, "max_tokens": a.max_tokens, "duration": a.duration,
           "requests_ok": len(done), "requests_failed": len(failed),
           "errors": sorted({str(r["err"]) for r in failed})[:5],
           "output_tok_per_s": round(toks / (max(r["t_end"] for r in out) - start), 1),
           "steady_tpot_ms_median": round(statistics.median(tpot) * 1e3, 2) if tpot else None,
           "steady_ttft_ms_median": round(statistics.median(ttft) * 1e3, 1) if ttft else None}
    if t_kill is not None:
        inflight = [r for r in out if r["t0"] < t_kill < r["t_end"]]
        hand, stall = [], []
        for r in inflight:
            after = [t for t in r["times"] if t > t_kill]
            if after:
                hand.append(after[0] - t_kill)
            ts = [r["t0"]] + r["times"]
            gaps = [b - x for x, b in zip(ts, ts[1:]) if b > t_kill and x < t_kill + 60]
            if gaps:
                stall.append(max(gaps))
        q = lambda v, p: round(sorted(v)[min(len(v) - 1, int(p * len(v)))], 3) if v else None  # noqa: E731
        res["no_token_window_s"] = no_token_window(out, t_kill)
        down = [t for t, ok in probes if not ok]
        up_after = [t for t, ok in probes if ok and down and t > down[0]]
        window = [r for r in out if r["t0"] >= t_kill - 0 and r["t0"] < t_kill + a.window]
        res.update({"tokens_lost": sum(r["n"] for r in failed),
                    "service_down_s": round(up_after[0] - t_kill, 2) if up_after else (
                        None if down else 0.0),
                    "goodput_window": round(sum(r["ok"] for r in inflight + window)
                                            / max(1, len(inflight + window)), 3),
                    "tokens_window": sum(r["n"] for r in out if r["ok"] and r["t_end"] > t_kill
                                         and r["t_end"] <= t_kill + a.window)})
        res.update({"kill_at_s": round(t_kill - start, 2), "inflight_at_kill": len(inflight),
                    "inflight_failed": sum(not r["ok"] for r in inflight),
                    "handover_s_median": q(hand, 0.5), "handover_s_max": q(hand, 1.0),
                    "stall_s_median": q(stall, 0.5), "stall_s_max": q(stall, 1.0)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--clients", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--duration", type=float, default=40)
    ap.add_argument("--kill", default="")
    ap.add_argument("--kill-after", type=float, default=15)
    ap.add_argument("--window", type=float, default=60,
                    help="seconds after the kill for the goodput window")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    start, t_kill, out, probes = asyncio.run(main_async(a))
    res = summarize(start, t_kill, out, a, probes)
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": res, "requests": out}, f)


if __name__ == "__main__":
    main()
