#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-replica pipeline experiments with tensor parallelism per stage.

Starts a deployment with ``local_pp.Deployment`` (docker, one container per
stage), drives load with ``benchmarks/pd/pd_bench.py`` against replica 0's
stage 0 (and, for ``kill``, the resume proxy over both chains is not used:
stage 0 itself fails over downstream stages), optionally kills one TP rank
of a downstream stage mid-run, and scores every generated token
teacher-forced on a non-disaggregated vLLM instance started afterwards
(``--score-python``, ``--score-gpus``).

    # one chain, two stages of TP 2: correctness and steady state
    tp_pp_experiments.py correct --chains "0+1,2+3" ...
    tp_pp_experiments.py perf --chains "0+1,2+3" --qps 1,2 ...
    # two chains sharing GPUs (stage 0s on GPUs 0,1; stage 1s on 2,3), kill
    # TP rank 1 of chain 0's stage 1 at 30 s, restart it at 75 s
    tp_pp_experiments.py kill --chains "0+1,2+3;0+1,2+3" --victim 0:1 --rank 1 ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pd"))
from local_pp import Deployment, add_args, wait_http  # noqa: E402

BENCH = os.path.join(os.path.dirname(HERE), "pd", "pd_bench.py")


def bench(out, url, *args):
    r = subprocess.run([sys.executable, BENCH, *args, "--url", url, "--out", out],
                       capture_output=True, text=True, env={**os.environ, "NO_PROXY": "*"})
    if r.returncode not in (0, 1):
        raise RuntimeError(f"pd_bench failed: {r.stderr[-2000:]}")
    return json.load(open(out))


def score(a, files):
    if not a.score_gpus:
        return {}
    from local_pd import Deployment as PD

    d = PD("vllm", "connex", a.score_model or a.model, os.path.join(a.out, "score-logs"),
           a.score_python, gpu_mem=0.6, max_model_len=4096, base_port=8300)
    inst = d.add("single", a.score_gpus)
    out = {}
    try:
        d.wait([inst])
        for f in files:
            sf = f[:-5] + ".score.json"
            subprocess.run([sys.executable, BENCH, "score", "--url", inst.url, "--file", f,
                            "--out", sf, "--model", "model"], check=True, capture_output=True,
                           env={**os.environ, "NO_PROXY": "*"})
            out[os.path.basename(f)] = json.load(open(sf))
    finally:
        d.down()
    return out


def load_args(a, keep=True):
    return ["--model", "model", "--prompt-len", str(a.prompt_len), "--max-tokens", str(a.max_tokens)] + \
        (["--keep-ids"] if keep else [])


def cmd_correct(a, d):
    url = d.stage(0, 0).url
    f = os.path.join(a.out, "correct.json")
    r = bench(f, url, "load", "--text", "--n", str(a.n), "--concurrency", "8",
              *load_args(a))
    return {"summary": r["summary"], "files": [f]}


def cmd_perf(a, d):
    url = d.stage(0, 0).url
    bench(os.path.join(a.out, "warmup.json"), url, "load", "--n", "16", "--concurrency", "4",
          "--seed", "99", "--model", "model", "--prompt-len", str(a.prompt_len),
          "--max-tokens", "16")
    res = {}
    for q in [x for x in a.qps.split(",") if x]:
        f = os.path.join(a.out, f"qps{q}.json")
        r = bench(f, url, "load", "--qps", q, "--duration", str(a.duration),
                  "--n", str(int(float(q) * a.duration) + 8), "--seed", "1",
                  *load_args(a, keep=False))
        res[f"qps{q}"] = r["summary"]
    if a.clients:
        f = os.path.join(a.out, f"closed{a.clients}.json")
        r = bench(f, url, "load", "--concurrency", str(a.clients), "--n", str(4 * a.clients),
                  "--seed", "2", *load_args(a, keep=False))
        res[f"closed{a.clients}"] = r["summary"]
    return {"perf": res, "files": []}


def cmd_kill(a, d):
    rep, st = (int(x) for x in a.victim.split(":"))
    victim = d.stage(rep, st)
    url = d.stage(0, 0).url
    tl = {}
    t_start = time.time()

    def chaos():
        time.sleep(a.at)
        tl["pids"] = victim.pids()
        t = victim.kill_rank(a.rank)
        tl["kill"] = t
        if t is None:
            tl["error"] = f"rank {a.rank} not found in {victim.name}: {tl['pids']}"
            return
        while time.time() - t < 60:
            if not victim.alive():
                tl["group_exit"] = time.time() - t
                break
            time.sleep(0.05)
        if a.restart_at:
            time.sleep(max(0.0, t_start + a.restart_at - time.time()))
            t2 = time.time()
            victim.start()
            try:
                victim.ready()
                tl["restart_ready"] = time.time() - t2
            except Exception as e:  # noqa: BLE001
                tl["restart_error"] = str(e)

    th = threading.Thread(target=chaos, daemon=True)
    th.start()
    f = os.path.join(a.out, "kill.json")
    r = bench(f, url, "load", "--qps", str(a.kill_qps), "--n", str(a.n), "--seed", "3",
              *load_args(a))
    th.join(timeout=1200)
    if tl.get("restart_ready"):
        # the restarted stage serves again: drain chain 1 traffic onto it by
        # killing nothing; just check a follow-up batch completes
        f2 = os.path.join(a.out, "after.json")
        r2 = bench(f2, url, "load", "--n", "16", "--concurrency", "4", "--seed", "4",
                   *load_args(a))
        tl["after_restart"] = r2["summary"]
    logs = victim.logs(400) if victim.alive() else ""
    s0 = d.stage(0, 0).logs(2000)
    tl["stage0_failover_lines"] = [x for x in s0.splitlines()
                                   if "fail" in x.lower() and "connex" in x.lower()][-20:]
    tl["victim_tail"] = logs.splitlines()[-5:]
    return {"summary": r["summary"], "timeline": tl, "files": [f]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["correct", "perf", "kill"])
    add_args(ap)
    ap.add_argument("--score-python", default="")
    ap.add_argument("--score-gpus", default="")
    ap.add_argument("--score-model", default="", help="model path for the scoring instance")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--qps", default="1,2")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--victim", default="0:1", help="replica:stage")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--at", type=float, default=30)
    ap.add_argument("--restart-at", type=float, default=0)
    ap.add_argument("--kill-qps", type=float, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.logdir = os.path.join(a.out, "logs")
    d = Deployment(a)
    t0 = time.time()
    try:
        d.up()
        for s in d.stages:
            if s.stage == 0:
                wait_http(s.url, 60)
        res = {"correct": cmd_correct, "perf": cmd_perf, "kill": cmd_kill}[a.mode](a, d)
    finally:
        d.down()
    res["args"] = vars(a)
    res["wall_s"] = time.time() - t0
    files = res.pop("files", [])
    if files:
        res["score"] = score(a, files)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
