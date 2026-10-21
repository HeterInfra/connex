"""Kill/rejoin churn against a live connex-rs engine (leak check, G6).

Each cycle starts a child process that brings up a connex_rs.Engine under
the SAME logical id and port, links to the target URL, reads a few bytes of
a region the target registered (optional), waits for the link to be up, and
is then SIGKILLed. The target sees PeerFailed then a rejoin, every cycle.

  python3 churn_peer.py --target <url> --ip <my ip> --cycles 100 \
      [--id model,stage,shard,replica] [--port 18800]
Prints one JSON line per cycle: link-up time after start, and whether the
child saw the link up.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def child(a):
    import connex_rs
    m, st, sh, rp = a.id.split(",")
    e = connex_rs.Engine(m, int(st), int(sh), int(rp), listen_ip=a.ip, port=a.port)
    e.connect(a.target)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10:
        ev = e.wait_event(100)
        if ev and ev["type"] == "up":
            print(json.dumps({"up_ms": round((time.monotonic() - t0) * 1e3, 1)}), flush=True)
            break
    time.sleep(3600)  # until killed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="parent")
    ap.add_argument("--target", required=True)
    ap.add_argument("--ip", required=True)
    ap.add_argument("--id", default="churn,7,0,0")
    ap.add_argument("--port", type=int, default=18800)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--hold", type=float, default=0.5)
    a = ap.parse_args()
    if a.mode == "child":
        return child(a)
    args = [sys.executable, __file__, "child", "--target", a.target, "--ip", a.ip,
            "--id", a.id, "--port", str(a.port)]
    ok = 0
    for i in range(a.cycles):
        t = time.monotonic()
        p = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline()
        up = json.loads(line)["up_ms"] if line.strip() else None
        ok += up is not None
        time.sleep(a.hold)
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        print(json.dumps({"cycle": i, "up_ms": up, "cycle_s": round(time.monotonic() - t, 2)}),
              flush=True)
    print(json.dumps({"cycles": a.cycles, "linked": ok}))


if __name__ == "__main__":
    main()
