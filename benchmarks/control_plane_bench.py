# SPDX-License-Identifier: Apache-2.0
"""Control-plane stress test (control-plane scalability).

Registers N endpoints, then for --duration seconds runs U updater processes
issuing route updates as fast as possible while one client measures
route-refresh (lookup) latency and one subscriber measures update -> event
notification delay. Two backends run the same workload:

  connex      connex.directory_server.DirectoryServer (msgpack, ROUTER/PUB)
  frontrouter a JSON REP/PUB front-router (default connex/front_router.py;
              override with FRONT_ROUTER_PY=/path/to/front_router.py)

    python benchmarks/control_plane_bench.py --backend connex --endpoints 2000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import threading
import time

import msgpack
import zmq

MODEL = "bench"
FRONT_ROUTER = os.environ.get(
    "FRONT_ROUTER_PY",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "connex", "front_router.py"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


# ---------------------------------------------------------------- clients

class Client:
    """Synchronous request client for either backend."""

    def __init__(self, backend: str, addr: str):
        self.backend = backend
        ctx = zmq.Context.instance()
        if backend == "connex":
            self.sock = ctx.socket(zmq.DEALER)
        else:
            self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{addr}")

    def call(self, msg):
        if self.backend == "connex":
            self.sock.send(msgpack.packb(msg, use_bin_type=True))
            return msgpack.unpackb(self.sock.recv(), raw=False)
        self.sock.send_json(msg)
        return self.sock.recv_json()

    def register(self, i: int, addr: str):
        if self.backend == "connex":
            return self.call({"type": "register", "event": "JOIN", "endpoint": {
                "lid": [MODEL, 1, 0, i], "addresses": [addr], "transport_caps": [], "epoch": 0}})
        return self.call({"method": "register", "params": {
            "lid": {"model_name": MODEL, "stage_idx": 1, "shard_idx": 0, "replica_idx": i},
            "addresses": [addr], "capabilities": [0]}})

    def update(self, i: int, addr: str):
        if self.backend == "connex":
            return self.call({"type": "register", "event": "MIGRATE", "endpoint": {
                "lid": [MODEL, 1, 0, i], "addresses": [addr], "transport_caps": [], "epoch": 0}})
        return self.call({"method": "update_route", "params": {
            "lid": {"model_name": MODEL, "stage_idx": 1, "shard_idx": 0, "replica_idx": i},
            "addresses": [addr]}})

    def lookup(self, i: int):
        if self.backend == "connex":
            return self.call({"type": "lookup", "lid": [MODEL, 1, 0, i]})
        return self.call({"method": "get_route", "params": {
            "lid": {"model_name": MODEL, "stage_idx": 1, "shard_idx": 0, "replica_idx": i}}})


def updater(backend, addr, n, duration, out, start_evt):
    c = Client(backend, addr)
    start_evt.wait()
    t_end = time.time() + duration
    k = 0
    while time.time() < t_end:
        i = k % n
        c.update(i, f"10.0.0.1:{time.time_ns()}")
        k += 1
    out.put(k)


def subscriber(backend, event_addr, delays, stop, ready):
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://{event_addr}")
    sub.poll(200)
    ready.set()
    while not stop.is_set():
        if not sub.poll(50):
            continue
        frames = sub.recv_multipart()
        now = time.time_ns()
        try:
            if backend == "connex":
                ev = msgpack.unpackb(frames[-1], raw=False)
                addrs = ev["endpoint"]["addresses"]
            else:
                ev = json.loads(frames[-1])
                p = ev.get("payload", {})
                addrs = p.get("new_addresses") or p.get("addresses") or []
            ts = int(addrs[0].rsplit(":", 1)[1])
            if ts > 1e15:  # an update address, not a registration
                delays.append((now - ts) / 1e6)
        except Exception:
            pass


# ----------------------------------------------------------------- server

def _serve_connex(api, ev):
    from connex.directory_server import DirectoryServer
    DirectoryServer(api, ev, lease_s=3600).serve_forever()


def start_server(backend):
    api, ev = f"127.0.0.1:{free_port()}", f"127.0.0.1:{free_port()}"
    if backend == "connex":
        p = mp.get_context("spawn").Process(target=_serve_connex, args=(api, ev), daemon=True)
        p.start()
        stop = p.kill
    else:
        proc = subprocess.Popen(
            [sys.executable, FRONT_ROUTER, "--api-address", f"tcp://{api}",
             "--event-address", f"tcp://{ev}", "--log-level", "WARNING"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop = proc.kill
        p = proc
    time.sleep(1.5)
    return api, ev, stop, p


def cpu_seconds(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return float("nan")


def run(args):
    api, ev, stop_server, proc = start_server(args.backend)
    pid = proc.pid
    try:
        c = Client(args.backend, api)
        for i in range(args.endpoints):
            c.register(i, f"10.0.0.2:{6000 + i % 50000}")

        delays, stop, ready = [], threading.Event(), threading.Event()
        threading.Thread(target=subscriber, args=(args.backend, ev, delays, stop, ready),
                         daemon=True).start()
        ready.wait(2)

        ctx = mp.get_context("spawn")
        out, start_evt = ctx.Queue(), ctx.Event()
        ups = [ctx.Process(target=updater, args=(args.backend, api, args.endpoints,
                                                 args.duration, out, start_evt))
               for _ in range(args.updaters)]
        for u in ups:
            u.start()
        time.sleep(1.0)
        cpu0 = cpu_seconds(pid)
        start_evt.set()
        t0 = time.time()
        lat = []
        k = 0
        while time.time() - t0 < args.duration:
            a = time.perf_counter()
            c.lookup(k % args.endpoints)
            lat.append((time.perf_counter() - a) * 1000)
            k += 1
            time.sleep(0.002)
        updates = sum(out.get(timeout=30) for _ in ups)
        elapsed = time.time() - t0
        cpu = (cpu_seconds(pid) - cpu0) / elapsed * 100
        time.sleep(0.5)
        stop.set()
        for u in ups:
            u.join(5)
        return {
            "backend": args.backend, "endpoints": args.endpoints, "updaters": args.updaters,
            "updates_per_s": round(updates / elapsed, 1),
            "lookup_p50_ms": round(pct(lat, 50), 3), "lookup_p99_ms": round(pct(lat, 99), 3),
            "notify_p50_ms": round(pct(delays, 50), 3), "notify_p99_ms": round(pct(delays, 99), 3),
            "notified_frac": round(len(delays) / max(updates, 1), 3),
            "server_cpu_pct": round(cpu, 1),
        }
    finally:
        stop_server()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["connex", "frontrouter"], default="connex")
    ap.add_argument("--endpoints", type=int, default=2000)
    ap.add_argument("--updaters", type=int, default=4)
    ap.add_argument("--duration", type=float, default=10)
    args = ap.parse_args()
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
