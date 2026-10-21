# SPDX-License-Identifier: Apache-2.0
"""Elastic scaling benchmark for Connex.

One upstream sender (stage 0) streams activations to a pool of downstream
replicas (stage 1) while replicas join and leave. Every replica is a separate
process with its own UCXTransport; membership goes through DirectoryServer.

Default timeline (seconds from start):
    0   replicas r0, r1 running
    T/4 scale-out: r2 joins
    T/2 graceful scale-in: r0 deregisters, drains, exits
    3T/4 abrupt scale-in: r1 is SIGKILLed (no deregister)

Reported per event:
    scale-out ready   JOIN -> first message delivered to the new replica
    graceful leave    messages lost, sends after LEAVE
    abrupt kill       messages lost, delivery gap, sender stall
and p50/p99 one-way latency and throughput for the steady state.

    python benchmarks/elastic_bench.py --duration 20 --size 1048576 --rate 500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time

HDR = struct.Struct("!Qd")  # seq, send wall time
MODEL = "bench"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------- receivers

def replica_main(idx: int, port: int, directory: str, log_path: str,
                 ready, leave, mode: str, heartbeat: bool) -> None:
    from connex.directory import DirectoryClient
    from connex.ucx_transport import TransportMode, UCXTransport

    tm = TransportMode.UCX_TCP if mode == "ucx" else TransportMode.ZEROMQ
    t = UCXTransport("127.0.0.1", port, mode=tm, device="cpu",
                     buffer_size=1 << 16, num_buffers=1)
    t.start_sync()
    d = DirectoryClient(directory, MODEL, stage_idx=1, replica_idx=idx,
                        local_addresses=[f"127.0.0.1:{port}"])
    d.start()
    log = open(log_path, "w", buffering=1)
    log.write(f"J {time.time()!r}\n")
    d.register()
    if heartbeat:
        d.start_heartbeat(0.2)
    ready.set()

    def recv(timeout):
        if tm == TransportMode.ZEROMQ:
            return t.recv_sync(timeout_ms=int(timeout * 1000))
        got = t.recv_from_queue(timeout=timeout)
        return got[1] if got else None

    leaving_since = None
    while True:
        data = recv(0.05)
        if data is not None:
            seq, ts = HDR.unpack_from(data)
            log.write(f"R {seq} {ts!r} {time.time()!r}\n")
        if leave.is_set() and leaving_since is None:
            log.write(f"L {time.time()!r}\n")
            d.deregister()
            leaving_since = time.time()
        if leaving_since is not None and data is None and time.time() - leaving_since > 0.5:
            break  # drained
    log.close()
    d.stop()
    t.stop_sync()
    os._exit(0)


# ------------------------------------------------------------------ sender

class PollingRouter:
    """Baseline: poll ``list_stage``, connect lazily on first send to a peer."""

    def __init__(self, directory: str, tx, poll_ms: int):
        from connex.directory import DirectoryClient
        self.tx, self.poll_s = tx, poll_ms / 1000
        self.members: dict[tuple, str] = {}
        self.eps: dict[tuple, object] = {}
        self.lock = threading.Lock()
        self.rr = 0
        self.client = DirectoryClient(directory, MODEL, stage_idx=0)
        self.client.start()
        self._stop = threading.Event()
        self._refresh()
        threading.Thread(target=self._poll, daemon=True).start()

    def _refresh(self):
        eps = self.client.list_stage(1)
        with self.lock:
            self.members = {e.lid: e.primary_address for e in eps}

    def _poll(self):
        while not self._stop.wait(self.poll_s):
            self._refresh()

    def pick(self):
        with self.lock:
            members = sorted(self.members)
            if not members:
                return None
            self.rr += 1
            return members[self.rr % len(members)]

    def send(self, lid, payload):
        try:
            ep = self.eps.get(lid)
            if ep is None:
                with self.lock:
                    addr = self.members[lid]
                host, port = addr.rsplit(":", 1)
                ep = self.eps[lid] = self.tx.connect_sync(host, int(port))
            self.tx.send_sync(ep, payload)
        except Exception:
            with self.lock:
                self.members.pop(lid, None)
            ep = self.eps.pop(lid, None)
            if ep is not None:
                try:
                    self.tx.close_endpoint(ep)
                except Exception:
                    pass
            raise

    def stop(self):
        self._stop.set()
        self.client.stop()


class ElasticRouter:
    """Optimized: ElasticPeerSet (push events, pre-connect, failure reporting)."""

    def __init__(self, directory: str, tx):
        from connex.directory import DirectoryClient
        from connex.elastic import ElasticPeerSet
        self.client = DirectoryClient(directory, MODEL, stage_idx=0)
        self.client.start()
        self.peers = ElasticPeerSet(self.client, tx, stage_idx=1)
        self.peers.start()
        self.peers.wait_ready(2, timeout=30)

    def pick(self):
        return self.peers.pick()

    def send(self, lid, payload):
        self.peers.send(lid, payload)

    def stop(self):
        self.peers.stop()
        self.client.stop()


def run(args) -> dict:
    from connex.directory_server import DirectoryServer
    from connex.ucx_transport import TransportMode, UCXTransport

    server = DirectoryServer(f"127.0.0.1:{free_port()}", f"127.0.0.1:{free_port()}",
                             lease_s=args.lease)
    server.start()
    ctx = mp.get_context("spawn")
    tmp = tempfile.mkdtemp(prefix="elastic-")
    reps = {}

    def spawn(i):
        ready, leave = ctx.Event(), ctx.Event()
        port = free_port()
        log = os.path.join(tmp, f"r{i}.log")
        p = ctx.Process(target=replica_main,
                        args=(i, port, server.bind_addr, log, ready, leave, args.transport,
                              args.membership == "elastic"))
        p.start()
        reps[i] = dict(proc=p, ready=ready, leave=leave, log=log, lid=(MODEL, 1, 0, i))
        return reps[i]

    for i in (0, 1):
        spawn(i)
    for r in reps.values():
        r["ready"].wait(60)

    tm = TransportMode.UCX_TCP if args.transport == "ucx" else TransportMode.ZEROMQ
    tx = UCXTransport("127.0.0.1", free_port(), mode=tm, device="cpu",
                      buffer_size=1 << 16, num_buffers=1)
    tx.start_sync()
    if args.membership == "elastic":
        router = ElasticRouter(server.bind_addr, tx)
    else:
        router = PollingRouter(server.bind_addr, tx, args.poll_ms)
    payload = bytearray(args.size)

    T = args.duration
    events = ([] if args.no_churn else
              [(T / 4, "join", 2), (T / 2, "leave", 0), (3 * T / 4, "kill", 1)])
    marks = {}
    sends = []  # (seq, lid, t_start, t_end, ok)
    t0 = time.time()
    seq = 0
    interval = 1.0 / args.rate if args.rate else 0
    next_t = t0
    while time.time() - t0 < T:
        now = time.time() - t0
        while events and now >= events[0][0]:
            _, kind, i = events.pop(0)
            marks[kind] = time.time()
            if kind == "join":
                spawn(i)
            elif kind == "leave":
                reps[i]["leave"].set()
            else:
                os.kill(reps[i]["proc"].pid, signal.SIGKILL)
        lid = router.pick()
        if lid is None:
            time.sleep(0.001)
            continue
        HDR.pack_into(payload, 0, seq, time.time())
        ts = time.time()
        ok = True
        try:
            router.send(lid, payload)
        except Exception:
            ok = False
        sends.append((seq, lid[3], ts, time.time(), ok))
        seq += 1
        if interval:
            next_t += interval
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()

    # Let graceful leaver finish, then collect.
    for r in reps.values():
        r["proc"].join(5)
        if r["proc"].is_alive():
            r["proc"].kill()
    router.stop()
    tx.stop_sync()
    server.stop()
    return analyse(args, reps, sends, marks)


# ---------------------------------------------------------------- analysis

def read_log(path):
    join = leave = None
    recv = {}
    try:
        for line in open(path):
            f = line.split()
            if f[0] == "J":
                join = float(f[1])
            elif f[0] == "L":
                leave = float(f[1])
            elif f[0] == "R" and len(f) == 4:
                recv[int(f[1])] = (float(f[2]), float(f[3]))
    except FileNotFoundError:
        pass
    return join, leave, recv


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def analyse(args, reps, sends, marks) -> dict:
    logs = {i: read_log(r["log"]) for i, r in reps.items()}
    delivered = {}
    for i, (_, _, recv) in logs.items():
        for s, v in recv.items():
            delivered[s] = (i, *v)
    ok_sends = [s for s in sends if s[4]]
    lost = [s for s in ok_sends if s[0] not in delivered]
    lat = [(d[2] - d[1]) * 1000 for d in delivered.values()]
    send_lat = [(s[3] - s[2]) * 1000 for s in sends]
    dur = sends[-1][3] - sends[0][2] if sends else 1

    def window(a, b):
        return [(d[2] - d[1]) * 1000 for d in delivered.values() if a <= d[1] < b]

    out = {
        "config": vars(args),
        "sent": len(sends), "send_errors": len(sends) - len(ok_sends),
        "delivered": len(delivered), "lost": len(lost),
        "throughput_msg_s": len(delivered) / dur,
        "throughput_MBps": len(delivered) * args.size / dur / 1e6,
        "lat_ms_p50": pct(lat, 50), "lat_ms_p99": pct(lat, 99), "lat_ms_max": max(lat or [0]),
        "send_ms_max": max(send_lat or [0]),
    }
    t_start = sends[0][2] if sends else 0
    worst = sorted(sends, key=lambda s: s[3] - s[2])[-3:]
    out["worst_sends"] = [
        {"t_s": round(s[2] - t_start, 3), "replica": s[1], "ms": round((s[3] - s[2]) * 1000, 1), "ok": s[4]}
        for s in reversed(worst)
    ]
    out["events_t_s"] = {k: round(v - t_start, 3) for k, v in marks.items()}
    # scale-out: JOIN -> first delivery to r2
    j, _, recv2 = logs.get(2, (None, None, {}))
    if j and recv2:
        out["scaleout_ready_ms"] = (min(v[1] for v in recv2.values()) - j) * 1000
    # graceful leave of r0
    _, l0, _ = logs[0]
    if l0:
        after = [s for s in ok_sends if s[1] == 0 and s[2] > l0]
        out["leave_sends_after_LEAVE"] = len(after)
        out["leave_lost"] = len([s for s in lost if s[1] == 0])
        out["leave_view_lag_ms"] = ((max(s[2] for s in after) - l0) * 1000) if after else 0.0
    # abrupt kill of r1
    k = marks.get("kill")
    if k:
        after = [s for s in sends if s[1] == 1 and s[2] >= k]
        out["kill_lost"] = len([s for s in lost if s[1] == 1 and s[2] >= k - 0.05])
        out["kill_sends_to_dead"] = len(after)
        out["kill_send_errors"] = len([s for s in after if not s[4]])
        out["kill_sender_stall_ms"] = max([(s[3] - s[2]) * 1000 for s in after] or [0])
        ts = sorted(d[2] for d in delivered.values() if d[2] >= k - 1)
        gaps = [b - a for a, b in zip(ts, ts[1:]) if a >= k - 0.1]
        out["kill_delivery_gap_ms"] = max(gaps or [0]) * 1000
        out["kill_window_lat_p99_ms"] = pct(window(k, k + 2), 99)
    steady = window(marks.get("join", 0) - 2, marks.get("join", 0)) if marks.get("join") else lat
    out["steady_lat_p99_ms"] = pct(steady, 99)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--size", type=int, default=1 << 20)
    ap.add_argument("--rate", type=float, default=500, help="msgs/s, 0 = closed loop")
    ap.add_argument("--transport", choices=["ucx", "zmq"], default="ucx")
    ap.add_argument("--membership", choices=["poll", "elastic"], default="poll")
    ap.add_argument("--poll-ms", type=int, default=100)
    ap.add_argument("--no-churn", action="store_true", help="steady state, no scale events")
    ap.add_argument("--lease", type=float, default=1.0, help="directory lease (elastic mode)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args)
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
