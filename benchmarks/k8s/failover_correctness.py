# SPDX-License-Identifier: Apache-2.0
"""Greedy-output correctness across a downstream failover.

Sends N fixed long-generation greedy requests at once (temperature 0,
ignore_eos, max_tokens M) to one stage-0 endpoint, optionally runs a shell
command (e.g. scale a downstream stage replica to 0) KILL_AFTER seconds in,
and records every completion's tokens. `compare` reports, per prompt, whether
two runs are identical and where they first diverge. Because batching makes
greedy decoding diverge at near-ties even without churn, `score` is the
primary check: it teacher-forces every completion through a (stable) pipeline
(prompt + generated ids, echo, logprobs=1) and counts generated tokens that
are not the argmax given their own prefix by more than --tol nats. A request
decoded without its earlier KV produces such tokens; a correct one does not.

    python failover_correctness.py run --url http://<stage0-replica1>:5101 \\
        --out ref1.json
    python failover_correctness.py run --url http://<stage0-replica1>:5101 \\
        --kill "kubectl -n $NAMESPACE scale deploy cte-s2-r1 --replicas=0" \\
        --kill-after 6 --out kill.json
    python failover_correctness.py compare ref1.json kill.json
    python failover_correctness.py score --url http://<stage0-replica0>:5100 kill.json
"""
import argparse
import asyncio
import json
import subprocess
import time

import aiohttp

TOPICS = ["the history of the Roman empire", "how photosynthesis works",
          "the rules of chess", "the life cycle of stars", "how compilers work",
          "the causes of the French revolution", "the water cycle",
          "how vaccines train the immune system", "the architecture of a CPU",
          "the migration of birds", "the invention of the printing press",
          "how neural networks learn", "plate tectonics", "the economics of inflation",
          "the structure of DNA", "the geography of South America",
          "the theory of relativity", "how the internet routes packets",
          "the culture of ancient Egypt", "the chemistry of cooking",
          "the physics of black holes", "how bees communicate",
          "the development of jazz", "the design of suspension bridges"]


def prompts(n):
    return [f"Write a detailed, multi-paragraph essay about {TOPICS[i % len(TOPICS)]}"
            f" (essay {i}):\n\n" for i in range(n)]


async def one(s, url, model, i, p, max_tokens):
    body = {"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "logprobs": 1, "return_tokens_as_token_ids": True}
    t = time.time()
    try:
        async with s.post(url + "/v1/completions", json=body,
                          timeout=aiohttp.ClientTimeout(total=600)) as r:
            d = await r.json()
        c = d["choices"][0]
        return {"i": i, "ok": True, "text": c["text"], "tokens": c["logprobs"]["tokens"],
                "finish_reason": c.get("finish_reason"), "t": round(time.time() - t, 2)}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "error": repr(e)[:300], "t": round(time.time() - t, 2)}


async def run(a):
    ps = prompts(a.n)
    async with aiohttp.ClientSession(trust_env=False) as s:
        tasks = [asyncio.create_task(one(s, a.url, a.model, i, p, a.max_tokens))
                 for i, p in enumerate(ps)]
        kill = None
        if a.kill:
            await asyncio.sleep(a.kill_after)
            kill = {"cmd": a.kill, "t": a.kill_after,
                    "rc": subprocess.run(a.kill, shell=True).returncode}
        res = await asyncio.gather(*tasks)
    json.dump({"url": a.url, "n": a.n, "max_tokens": a.max_tokens, "kill": kill,
               "results": res}, open(a.out, "w"), indent=1)
    ok = sum(r["ok"] for r in res)
    print(json.dumps({"out": a.out, "ok": ok, "n": a.n,
                      "finish": sorted({str(r.get("finish_reason")) for r in res}),
                      "max_t": max(r["t"] for r in res)}))


def first_div(x, y):
    for k, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return k
    return None if len(x) == len(y) else min(len(x), len(y))


def _post(url, path, body):
    import urllib.request
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(op.open(req, timeout=300))


def score(a):
    d = json.load(open(a.file))
    ps = prompts(d["n"])
    tot = bad = bad_reqs = 0
    worst, per = 0.0, []
    for r in d["results"]:
        if not r.get("ok"):
            per.append((r["i"], "failed"))
            continue
        pid = _post(a.url, "/tokenize", {"model": a.model, "prompt": ps[r["i"]]})["tokens"]
        gen = [int(t.split(":")[1]) for t in r["tokens"]]
        e = _post(a.url, "/v1/completions", {
            "model": a.model, "prompt": pid + gen, "max_tokens": 1, "temperature": 0,
            "logprobs": 1, "echo": True, "return_tokens_as_token_ids": True})
        lp = e["choices"][0]["logprobs"]
        nb, first = 0, None
        for k in range(len(gen)):
            j = len(pid) + k
            top = max(lp["top_logprobs"][j].values())
            gap = top - lp["token_logprobs"][j]
            worst = max(worst, gap)
            if gap > a.tol:
                nb += 1
                first = k if first is None else first
        tot += len(gen)
        bad += nb
        bad_reqs += nb > 0
        if nb:
            per.append((r["i"], nb, first))
    print(json.dumps({"file": a.file, "tokens": tot, "non_argmax_tokens": bad,
                      "requests_with_non_argmax": bad_reqs, "worst_gap": round(worst, 3),
                      "tol": a.tol, "detail": per}))


def compare(a):
    A = {r["i"]: r for r in json.load(open(a.a))["results"]}
    B = {r["i"]: r for r in json.load(open(a.b))["results"]}
    same, divs, failed = 0, [], 0
    for i in sorted(A):
        ra, rb = A[i], B.get(i)
        if not (ra.get("ok") and rb and rb.get("ok")):
            failed += 1
            continue
        d = first_div(ra["tokens"], rb["tokens"])
        if d is None:
            same += 1
        else:
            divs.append((i, d))
    print(json.dumps({"a": a.a, "b": a.b, "identical": same, "diverged": len(divs),
                      "failed": failed, "first_divergence": divs}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", default="cte-model")
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--kill", default="")
    r.add_argument("--kill-after", type=float, default=6)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    sc = sub.add_parser("score")
    sc.add_argument("file")
    sc.add_argument("--url", required=True)
    sc.add_argument("--model", default="cte-model")
    sc.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    elif a.cmd == "score":
        score(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
