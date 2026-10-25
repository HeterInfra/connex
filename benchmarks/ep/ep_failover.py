# SPDX-License-Identifier: Apache-2.0
"""Kill one EP rank of a serving EP group under load; measure recovery.

Two vLLM EP groups (each tensor/expert parallel over two GPUs) serve behind
the resume proxy (``connex.serving.proxy``); ``EPGroupSupervisor`` watches
them. At ``--kill-at`` seconds one worker process (EP rank 1) of group A is
SIGKILLed. vLLM stops the whole group (the ranks share collectives every
step); the proxy resumes A's in-flight requests on group B from the tokens
already produced, the supervisor drops A from the routing set, restarts it
and puts it back once it serves. Output JSON: per-request results
(``max_gap`` = longest stall between two tokens), summary, supervisor
events, and kill timestamp.

    ep_failover.py --serve ./serve.sh --gpus-a 2,3 --gpus-b 0,1 --out fo.json

``--serve`` is a script ``serve.sh GPUS PORT LOG [args]`` that execs
``vllm serve`` (see docs/ep.md). For SGLang groups pass
``--worker-needle sglang::scheduler --kill-needle scheduler_TP1`` (SGLang
also stops the whole server when one scheduler process dies).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "pd"))
import pd_bench  # noqa: E402

from connex.ep.groups import EPGroup, EPGroupSupervisor  # noqa: E402


def launch(serve, gpus, port, logf, extra):
    return subprocess.Popen([serve, gpus, str(port), logf] + extra, start_new_session=True)


def wait_up(url, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with op.open(url + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{url} did not come up")


def workers(pgid, needle):
    out = subprocess.run(["ps", "-eo", "pid,pgid,args"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines()[1:]:
        f = line.split(None, 2)
        if len(f) == 3 and int(f[1]) == pgid and needle in f[2]:
            pids.append(int(f[0]))
    return pids


def kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--gpus-a", default="2,3")
    ap.add_argument("--gpus-b", default="0,1")
    ap.add_argument("--port-a", type=int, default=8301)
    ap.add_argument("--port-b", type=int, default=8302)
    ap.add_argument("--proxy-port", type=int, default=5199)
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--extra", default="", help="extra vllm args (one string)")
    ap.add_argument("--kill-at", type=float, default=20.0)
    ap.add_argument("--kill-needle", default="Worker_TP1",
                    help="substring of the process title of the rank to kill")
    ap.add_argument("--worker-needle", default="VLLM::Worker",
                    help="substring of the process titles of all ranks (process watch)")
    ap.add_argument("--qps", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--detect", choices=["health", "process"], default="process",
                    help="health: /health probes only; process: also watch the rank "
                         "processes (a dead rank is seen within --probe-s)")
    ap.add_argument("--probe-s", type=float, default=0.1)
    ap.add_argument("--reload-interval", type=float, default=0.2,
                    help="how often the proxy re-reads the endpoints file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = a.extra.split() if a.extra else []
    ua, ub = f"http://127.0.0.1:{a.port_a}", f"http://127.0.0.1:{a.port_b}"
    la, lb = os.path.join(a.logdir, "fo-a.log"), os.path.join(a.logdir, "fo-b.log")
    procs = {"a": launch(a.serve, a.gpus_a, a.port_a, la, extra),
             "b": launch(a.serve, a.gpus_b, a.port_b, lb, extra)}
    proxy = None
    try:
        start_s = {"a": wait_up(ua), "b": wait_up(ub)}
        print("groups up", start_s, flush=True)
        epf = os.path.join(a.logdir, "fo-endpoints")
        n_restart = [0]

        def restart_a():
            n_restart[0] += 1
            kill_group(procs["a"])
            time.sleep(2)          # let the GPU memory go
            procs["a"] = launch(a.serve, a.gpus_a, a.port_a, la + f".{n_restart[0]}", extra)
            return procs["a"]

        def pids(k):
            if a.detect != "process":
                return None
            return lambda: workers(procs[k].pid, a.worker_needle)

        sup = EPGroupSupervisor([EPGroup("ep-group/a", ua, restart_a, procs["a"], pids("a")),
                                 EPGroup("ep-group/b", ub, None, procs["b"], pids("b"))], epf,
                                probe_s=a.probe_s, fail_after=2)
        proxy = subprocess.Popen([sys.executable, "-m", "connex.serving.proxy", "--port",
                                  str(a.proxy_port), "--endpoints-file", epf,
                                  "--probe-interval", "0.5", "--reload-interval",
                                  str(a.reload_interval)], start_new_session=True,
                                 stdout=open(os.path.join(a.logdir, "fo-proxy.log"), "w"),
                                 stderr=subprocess.STDOUT)
        wait_up(f"http://127.0.0.1:{a.proxy_port}", 60)
        sup.start()
        kill = {}

        def killer():
            time.sleep(a.kill_at)
            pids = workers(procs["a"].pid, a.kill_needle)
            kill["t"] = time.time()
            kill["pids"] = pids
            for p in pids:
                os.kill(p, signal.SIGKILL)
            print(f"killed {pids} ({a.kill_needle}) of group a", flush=True)

        threading.Thread(target=killer, daemon=True).start()
        la_args = argparse.Namespace(n=a.n, prompt_len=0, seed=7, text=True, ref=None,
                                     url=f"http://127.0.0.1:{a.proxy_port}", model="olmoe",
                                     max_tokens=a.max_tokens, timeout=600, concurrency=0,
                                     qps=a.qps, duration=a.duration)
        t_load = time.time()
        res, wall = asyncio.run(pd_bench.run_load(la_args))
        s = pd_bench.summary(res, wall, a.max_tokens)
        # requests in flight at the kill: started before and finished after it
        kt = kill.get("t")
        off = time.time() - time.perf_counter()
        inflight = [r for r in res if kt and r["start"] + off <= kt < r["start"] + off + r["e2e"]]
        sup.stop()
        out = {"args": dict(vars(a), prompt_len=0, seed=7, text=True), "start_s": start_s, "kill": kill, "t_load": t_load,
               "summary": s, "events": sup.events, "requests": res,
               "inflight_at_kill": len(inflight),
               "inflight_ok": sum(r["ok"] for r in inflight)}
        json.dump(out, open(a.out, "w"))
        down = next((e for e in sup.events if e["kind"] == "down"), None)
        rejoin = next((e for e in sup.events if e["kind"] == "rejoin"), None)
        print(json.dumps({"requests": s["requests"], "ok": s["ok"], "failed": s["failed"],
                          "inflight_at_kill": len(inflight),
                          "inflight_ok": sum(r["ok"] for r in inflight),
                          "max_gap_max": s["max_gap"]["max"], "max_gap_p99": s["max_gap"]["p99"],
                          "detect_s": down and kt and down["t"] - kt,
                          "rebuild_s": rejoin and rejoin.get("rebuild_s"),
                          "kill_to_rejoin_s": rejoin and kt and rejoin["t"] - kt}), flush=True)
    finally:
        if proxy:
            kill_group(proxy)
        for p in procs.values():
            kill_group(p)


if __name__ == "__main__":
    main()
