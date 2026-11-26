# SPDX-License-Identifier: Apache-2.0
"""Front-router registration cost at scale.

Two measurements, both against connex.front_router as found on PYTHONPATH
(so the same script can time an older checkout):

  cost   API-thread work per register (handler + reply encoding) with N
         routes of one model already present, for an old client (no
         ``snapshot`` field: full route table), a new client
         (``snapshot: "compact"``) and ``snapshot: "none"``.
  storm  A real router process with N routes; R client threads re-register
         every endpoint once (e.g. a fleet restart) while a separate
         process sends heartbeats at a fixed rate. Reports storm wall time
         and heartbeat latency (p50/p99/max) during the storm.

    python benchmarks/register_bench.py cost --endpoints 500,1000,2000,5000
    python benchmarks/register_bench.py storm --endpoints 2000 --mode old
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import subprocess
import sys
import time

import zmq

MODEL = "bench"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def lid(i: int, stages: int = 4) -> dict:
    return {"model_name": MODEL, "stage_idx": i % stages, "shard_idx": 0,
            "replica_idx": i // stages}


def reg_params(i: int, mode: str) -> dict:
    p = {"lid": lid(i), "addresses": [f"tcp://10.0.{i // 250}.{i % 250}:7000"],
         "capabilities": [0, 1]}
    if mode != "old":
        p["snapshot"] = mode
        p["stage_idx"] = lid(i)["stage_idx"]
    return p


def prefill(fr, n: int):
    from connex.front_router import LogicalEndpointID, TransportCapability
    for i in range(n):
        p = reg_params(i, "old")
        fr.route_manager.add_route(LogicalEndpointID.from_dict(p["lid"]), p["addresses"],
                                   [TransportCapability(c) for c in p["capabilities"]], 0)


# ------------------------------------------------------------------ cost

class _NullPub:
    def send_multipart(self, parts):
        pass


def encode(fr, resp) -> bytes:
    enc = getattr(fr, "_encode_reply", None)
    return enc(resp) if enc else json.dumps(resp).encode("utf8")


def run_cost(args):
    from connex.front_router import FrontRouter, FrontRouterConfig
    rows = []
    for n in [int(x) for x in args.endpoints.split(",")]:
        fr = FrontRouter(FrontRouterConfig())
        fr.event_bus._pub_socket = _NullPub()
        fr.event_bus._running = True
        prefill(fr, n)
        routes = fr.route_manager.get_all_routes()
        for label in ("old", "old-hb", "compact", "none"):
            # old-hb: every route heartbeated since the previous reply (the
            # worst case for the full form, whose timestamps then all change).
            mode = "old" if label == "old-hb" else label
            ts, size = [], 0
            calls = max(20, args.calls // 5) if label == "old-hb" else args.calls
            for k in range(calls):
                i = (k * 7919) % n
                if label == "old-hb":
                    now = time.time()
                    for r in routes:
                        r.last_heartbeat = now
                t0 = time.perf_counter()
                resp = fr._handle_request({"method": "register", "params": reg_params(i, mode)})
                wire = encode(fr, resp)
                ts.append(time.perf_counter() - t0)
                size = len(wire)
            rows.append((n, label, statistics.median(ts) * 1e3, pct(ts, 99) * 1e3, size))
    print(f"{'routes':>7} {'client':>8} {'median_ms':>10} {'p99_ms':>8} {'reply_B':>9}")
    for n, mode, med, p99, size in rows:
        print(f"{n:>7} {mode:>8} {med:>10.3f} {p99:>8.3f} {size:>9}")


# ----------------------------------------------------------------- storm

def serve(n: int, api: str, ev: str):
    from connex.front_router import FrontRouter, FrontRouterConfig
    import logging
    logging.disable(logging.WARNING)
    fr = FrontRouter(FrontRouterConfig(api_address=f"tcp://{api}",
                                       event_pub_address=f"tcp://{ev}"))
    prefill(fr, n)
    fr.start()
    print("ready", flush=True)
    while True:
        time.sleep(3600)


def heartbeater(api, n, rate, stop, out):
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{api}")
    lat = []
    k = 0
    period = 1.0 / rate
    nxt = time.perf_counter()
    while not stop.is_set():
        t0 = time.perf_counter()
        s.send_json({"method": "heartbeat", "params": {"lid": lid(k % n)}})
        s.recv()
        lat.append(time.perf_counter() - t0)
        k += 1
        nxt += period
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.perf_counter()
    out.put(lat)


def registrar(api, n, mode, r, nproc, go, ready):
    """One registering client process: endpoints r, r + nproc, ... re-register."""
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{api}")
    ready.put(r)
    go.wait()
    for i in range(r, n, nproc):
        s.send_json({"method": "register", "params": reg_params(i, mode)})
        # The reply is received in full but not decoded: decoding is the
        # endpoint's own (parallel) cost, not the router's.
        assert s.recv().startswith(b'{"success": true')
    s.close()
    ctx.term()


def run_storm(args):
    n = args.endpoints_storm
    api, ev = f"127.0.0.1:{free_port()}", f"127.0.0.1:{free_port()}"
    # One heartbeater cannot reach every one of N endpoints each second as a
    # real fleet does; a long lease keeps the routes it misses from expiring
    # (and flapping UNHEALTHY/ACTIVE) mid-run, which would change the table.
    env = dict(os.environ, CONNEX_HEARTBEAT_TIMEOUT_S="3600")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", str(n), api, ev],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        assert proc.stdout.readline().strip() == "ready"
        mpc = mp.get_context("spawn")
        stop, out = mpc.Event(), mpc.Queue()
        hb = mpc.Process(target=heartbeater, args=(api, n, args.hb_rate, stop, out))
        hb.start()
        time.sleep(1.0)       # heartbeat baseline before the storm
        go, ready = mpc.Event(), mpc.Queue()
        regs = [mpc.Process(target=registrar, args=(api, n, args.mode, r, args.registrars, go, ready))
                for r in range(args.registrars)]
        for p in regs:
            p.start()
        for _ in regs:
            ready.get(timeout=60)
        t0 = time.perf_counter()
        go.set()
        for p in regs:
            p.join()
        storm_s = time.perf_counter() - t0
        time.sleep(0.2)
        stop.set()
        lat = out.get(timeout=30)
        hb.join()
        ms = [x * 1e3 for x in lat]
        print(f"routes={n} mode={args.mode} registrars={args.registrars} "
              f"storm_s={storm_s:.2f} regs_per_s={n / storm_s:.0f} heartbeats={len(ms)} "
              f"hb_p50_ms={pct(ms, 50):.2f} hb_p99_ms={pct(ms, 99):.2f} hb_max_ms={max(ms):.2f}")
    finally:
        proc.kill()
        proc.wait()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(int(sys.argv[2]), sys.argv[3], sys.argv[4])
        return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--endpoints", default="500,1000,2000,5000")
    c.add_argument("--calls", type=int, default=200)
    s = sub.add_parser("storm")
    s.add_argument("--endpoints", dest="endpoints_storm", type=int, default=2000)
    s.add_argument("--mode", choices=["old", "compact", "none"], default="old")
    s.add_argument("--registrars", type=int, default=8)
    s.add_argument("--hb-rate", type=float, default=500.0)
    args = ap.parse_args()
    run_cost(args) if args.cmd == "cost" else run_storm(args)


if __name__ == "__main__":
    main()
