#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel PD experiments on one host (vLLM or SGLang).

Every run starts a deployment with ``local_pd.Deployment`` (GPU specs with
``+`` are TP groups, e.g. ``1+2``), runs ``pd_bench.py`` against the PD
router, optionally kills one TP rank of an instance and restarts it, and
scores every generated token teacher-forced on a non-disaggregated instance
(started on ``--score-gpus`` after the deployment is torn down).

    # correctness, prefill TP 2 -> decode TP 1
    tp_experiments.py correct --model M --prefill 1+2 --decode 3 --out runs/p2d1
    # steady state
    tp_experiments.py perf --model M --prefill 0+1 --decode 2+3 --qps 1,2,4 --out runs/p2d2
    # group failure: kill rank 1 of decode instance 1 at 30 s, restart it at 60 s
    tp_experiments.py kill --model M --prefill 0 --decode 1+2,3 --front-router \\
        --victim decode1 --rank 1 --at 30 --restart-at 60 --out runs/kill
    # kill a whole instance (--rank -1), here across hosts (--hosts, see local_pd.py)
    tp_experiments.py kill --hosts hosts.json --model M --prefill A:2 --decode B:2,C:2 \\
        --victim decode1 --rank -1 --at 30 --out runs/dkill
    # pool scaling: add decode B:3 at 20 s, drain decode1 at 50 s, stop it at 80 s
    tp_experiments.py scale --hosts hosts.json --model M --prefill A:2 --decode B:2 \\
        --add decode=B:3 --at 20 --drain-at 50 --stop-at 80 --out runs/scale

Results (JSON per step plus ``result.json``) go to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from local_pd import Deployment, RemoteInstance, load_hosts, wait_http  # noqa: E402

BENCH = os.path.join(HERE, "pd_bench.py")


def bench(out, url, *args):
    cmd = [sys.executable, BENCH, *args, "--url", url, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "NO_PROXY": "*"})
    if r.returncode not in (0, 1):
        raise RuntimeError(f"pd_bench {args[0]} failed: {r.stderr[-2000:]}")
    return json.load(open(out))


def deploy(a, logdir):
    d = Deployment(a.engine, a.connector, a.model, logdir, a.python, gpu_mem=a.gpu_mem,
                   max_model_len=a.max_model_len, front_router=a.front_router,
                   extra_args=a.extra.split(), retain_s=a.retain_s,
                   hosts=load_hosts(a.hosts))
    try:
        if a.front_router:
            d.start_front_router()
            time.sleep(1)
        for role, specs in (("prefill", a.prefill), ("decode", a.decode)):
            for g in specs.split(","):
                inst = d.add(role, g)
                d.wait([inst])      # one at a time: vLLM memory profiling, NCCL ports
        if a.mode == "scale":   # pools from files, edited during the run
            for role in ("prefill", "decode"):
                d.set_pool_file(role, [i.url for i in d.instances if i.role == role])
        d.start_router(static=not a.front_router and a.mode != "scale")
        wait_http(d.router_url, 60)
        if a.front_router or a.mode == "scale":   # every member listed and probed
            _wait_pools(d, len(a.prefill.split(",")), len(a.decode.split(",")))
    except BaseException:
        d.down()            # no stray instances or routers after a failed start
        raise
    return d


def _pools(d):
    import urllib.request

    with urllib.request.urlopen(d.router_url + "/pd/pools", timeout=2) as r:
        return json.load(r)


def _wait_pools(d, n_pre, n_dec, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            p = _pools(d)
            if sum(e["healthy"] for e in p["prefill"]) >= n_pre and \
                    sum(e["healthy"] for e in p["decode"]) >= n_dec:
                return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise TimeoutError(f"pools not complete: {_pools(d)}")


def score(a, files, logdir):
    """Score runs on a non-disaggregated instance (TP of --score-gpus)."""
    if not a.score_gpus:
        return {}
    eng = a.score_engine or a.engine
    d = Deployment(eng, a.connector, a.model, logdir, a.score_python or a.python,
                   gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, base_port=8300,
                   extra_args=a.extra.split() if eng == a.engine else [],
                   hosts=load_hosts(a.hosts))
    inst = d.add("single", a.score_gpus)
    out = {}
    try:
        d.wait([inst])
        for f in files:
            sf = f[:-5] + ".score.json"
            subprocess.run([sys.executable, BENCH, "score", "--url", inst.url, "--file", f,
                            "--out", sf], check=True, capture_output=True,
                           env={**os.environ, "NO_PROXY": "*"})
            out[os.path.basename(f)] = json.load(open(sf))
    finally:
        d.down()
    return out


def cmd_correct(a, d):
    f = os.path.join(a.out, "correct.json")
    r = bench(f, d.router_url, "load", "--text", "--n", str(a.n), "--max-tokens",
              str(a.max_tokens), "--concurrency", "8", "--keep-ids")
    return {"summary": r["summary"], "files": [f]}


def cmd_perf(a, d):
    files, res = [], {}
    # warm-up (CUDA graphs, connex-rs links)
    bench(os.path.join(a.out, "warmup.json"), d.router_url, "load", "--n", "16",
          "--prompt-len", str(a.prompt_len), "--max-tokens", "16", "--concurrency", "4",
          "--seed", "99")
    for q in [x for x in a.qps.split(",") if x]:
        f = os.path.join(a.out, f"qps{q}.json")
        r = bench(f, d.router_url, "load", "--qps", q, "--duration", str(a.duration),
                  "--n", str(int(float(q) * a.duration) + 8), "--prompt-len", str(a.prompt_len),
                  "--max-tokens", str(a.max_tokens), "--seed", "1")
        res[f"qps{q}"] = r["summary"]
    if a.clients:
        f = os.path.join(a.out, f"closed{a.clients}.json")
        r = bench(f, d.router_url, "load", "--concurrency", str(a.clients), "--n", "128",
                  "--prompt-len", str(a.prompt_len), "--max-tokens", str(a.max_tokens),
                  "--seed", "2")
        res[f"closed{a.clients}"] = r["summary"]
    return {"perf": res, "files": files}


def cmd_kill(a, d):
    victim = next(i for i in d.instances if i.name == a.victim)
    timeline = {}
    f = os.path.join(a.out, "kill.json")
    t_start = time.time()

    def chaos():
        time.sleep(a.at)
        pids = victim.rank_pids()
        timeline["rank_pids"] = pids
        if a.rank < 0:      # the whole instance (all its processes) at once
            if isinstance(victim, RemoteInstance):
                t = victim.kill_group()
            else:
                t = time.time()
                os.killpg(victim.proc.pid, signal.SIGKILL)
        else:
            t = victim.kill_rank(a.rank)
        timeline["kill"] = t
        if isinstance(victim, RemoteInstance) and victim.kill_times:
            t0, _, t1 = victim.kill_times[-1]
            timeline["kill_rtt"] = t1 - t0     # uncertainty of the kill time
        if t is None:
            timeline["error"] = f"rank {a.rank} of {victim.name} not found ({pids})"
            return
        # when does the router stop listing the group, and does the rest of
        # the group exit (vLLM/SGLang take the engine down with a dead rank)?
        while time.time() - t < 30:
            try:
                p = _pools(d)
                urls = [e["url"] for e in p[victim.role] if e["healthy"]]
            except Exception:  # noqa: BLE001
                urls = None
            if urls is not None and victim.url not in urls and "router_drop" not in timeline:
                timeline["router_drop"] = time.time() - t
            if not victim.alive() and "group_exit" not in timeline:
                timeline["group_exit"] = time.time() - t
            if "router_drop" in timeline and "group_exit" in timeline:
                break
            time.sleep(0.05)
        if a.restart_at:
            time.sleep(max(0.0, t_start + a.restart_at - time.time()))
            victim.kill(signal.SIGKILL)          # rest of the group, if still up
            tw = time.time()
            ok = _wait_gpus_free(victim)     # the dead group's memory is released late
            timeline["gpu_free_wait"] = time.time() - tw
            if not ok and not isinstance(victim, RemoteInstance):
                timeline["gpu_not_free"] = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
                     "--format=csv,noheader"], capture_output=True, text=True).stdout[-2000:]
                timeline["victim_procs"] = subprocess.run(
                    ["ps", "-o", "pid,stat,etime,args", "-g", str(victim.proc.pid)],
                    capture_output=True, text=True).stdout[-2000:]
            t2 = time.time()
            victim.start()
            try:
                wait_http(victim.url, 900, victim.proc)   # proc is None for a remote one
                timeline["restart_ready"] = time.time() - t2
                while time.time() - t2 < 900:
                    p = _pools(d)
                    if any(e["url"] == victim.url and e["healthy"] for e in p[victim.role]):
                        timeline["rejoined"] = time.time() - t2
                        break
                    time.sleep(0.2)
            except Exception as e:  # noqa: BLE001
                timeline["restart_error"] = str(e)

    th = threading.Thread(target=chaos, daemon=True)
    th.start()
    r = bench(f, d.router_url, "load", "--qps", str(a.kill_qps), "--n", str(a.n),
              "--prompt-len", str(a.prompt_len), "--max-tokens", str(a.max_tokens),
              "--keep-ids", "--seed", "3")
    th.join(timeout=1200)
    # requests that started after the restarted instance rejoined: did it serve?
    stats = _router_stats(d)
    return {"summary": r["summary"], "timeline": timeline, "router": stats,
            "kill_offset_s": (timeline.get("kill") or t_start) - t_start, "files": [f]}


def _wait_gpus_free(inst, need_mib=30000, timeout=600):
    gpus = str(inst.gpu).split("+")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if isinstance(inst, RemoteInstance):
            if inst.gpus_free(need_mib):
                return True
            time.sleep(1)
            continue
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        free = dict(x.replace(" ", "").split(",") for x in out.stdout.strip().splitlines())
        if all(int(free.get(g, 0)) >= need_mib for g in gpus):
            return True
        time.sleep(1)
    return False


def cmd_scale(a, d):
    """Pool scaling under load: add an instance, drain one, stop it."""
    import urllib.request

    role, spec = a.add.split("=", 1)
    victim = next(i for i in d.instances if i.name == a.victim)
    timeline = {}
    f = os.path.join(a.out, "scale.json")
    t_start = time.time()

    def post(path, body):
        req = urllib.request.Request(d.router_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode()

    def chaos():
        time.sleep(a.at)
        new = d.add(role, spec)
        wait_http(new.url, 900, new.proc)
        timeline["added_ready"] = time.time() - t_start
        d.set_pool_file(role, [i.url for i in d.instances if i.role == role])
        time.sleep(max(0.0, t_start + a.drain_at - time.time()))
        timeline["drain"] = time.time() - t_start
        timeline["drain_resp"] = post("/pd/drain", {"url": victim.url, "role": victim.role})
        time.sleep(max(0.0, t_start + a.stop_at - time.time()))
        d.set_pool_file(victim.role, [i.url for i in d.instances
                                      if i.role == victim.role and i is not victim])
        time.sleep(1.5)   # router re-reads the pool files every second
        timeline["stop"] = time.time() - t_start
        victim.kill(signal.SIGTERM)

    th = threading.Thread(target=chaos, daemon=True)
    th.start()
    r = bench(f, d.router_url, "load", "--qps", str(a.kill_qps), "--n", str(a.n),
              "--prompt-len", str(a.prompt_len), "--max-tokens", str(a.max_tokens),
              "--keep-ids", "--seed", "3")
    th.join(timeout=1200)
    return {"summary": r["summary"], "timeline": timeline, "router": _router_stats(d),
            "pools_end": _pools(d), "files": [f]}


def _router_stats(d):
    import urllib.request

    try:
        with urllib.request.urlopen(d.router_url + "/proxy/stats", timeout=2) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["correct", "perf", "kill", "scale"])
    ap.add_argument("--hosts", default=None, help="JSON file of remote hosts (local_pd.py)")
    ap.add_argument("--add", default="decode=1", help="scale: ROLE=GPU_SPEC of the added instance")
    ap.add_argument("--drain-at", type=float, default=50)
    ap.add_argument("--stop-at", type=float, default=80)
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--connector", default="connex")
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--prefill", default="1")
    ap.add_argument("--decode", default="2")
    ap.add_argument("--score-gpus", default="", help="GPU(s) of the scoring instance")
    ap.add_argument("--score-engine", default="", help="engine of the scoring instance "
                    "(default --engine; SGLang runs are scored on vLLM)")
    ap.add_argument("--score-python", default="")
    ap.add_argument("--front-router", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--retain-s", type=float, default=60.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--qps", default="1,2,4")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--clients", type=int, default=32)
    ap.add_argument("--victim", default="decode1")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--at", type=float, default=30)
    ap.add_argument("--restart-at", type=float, default=0)
    ap.add_argument("--kill-qps", type=float, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logdir = os.path.join(a.out, "logs")
    d = deploy(a, logdir)
    t0 = time.time()
    try:
        res = {"correct": cmd_correct, "perf": cmd_perf, "kill": cmd_kill,
               "scale": cmd_scale}[a.mode](a, d)
    finally:
        d.down()
    res["args"] = vars(a)
    res["wall_s"] = time.time() - t0
    res["router_log_tail"] = _tail(os.path.join(logdir, "router.log"))
    files = res.pop("files", [])
    if files:
        res["score"] = score(a, files, logdir)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "router_log_tail"}, indent=1))


def _tail(path, n=40):
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return []
    keep = [x for x in lines if re.search(r"resum|fail|blacklist|pool change|re-prefill", x)]
    return keep[-n:]


if __name__ == "__main__":
    main()
