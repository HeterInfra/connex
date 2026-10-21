# SPDX-License-Identifier: Apache-2.0
"""Churn driver for a 2-replica pipeline (replica 1 goes down/up every interval).

  soft  deregister all of replica 1's stage routes at the front-router, later
        re-register them; processes keep running.
  hard  kill replica 1's pods (scale to 0), later scale back up; the replica
        rejoins after a real cold start.

Clients learn about stage-0 endpoints through --endpoints-file (read by
trace_load.py every second): the replica-1 entry is removed when it goes
down and added back once it is serving again.

    kubectl -n $NAMESPACE port-forward svc/connex-front-router 15555:15555 &   # or your --router-name
    python churn.py --prefix cte --mode soft --interval 60 --duration 300 \\
        --r0 http://<stage0-replica0-host>:5100 --r1 http://<stage0-replica1-host>:5101 \\
        --endpoints-file eps.txt --events-out events.json

Endpoints may also come from CONNEX_R0_URL / CONNEX_R1_URL, the namespace
from CONNEX_NAMESPACE (default "connex").
"""
import argparse
import json
import subprocess
import time
import urllib.request

import zmq

import os



class Router:
    def __init__(self, addr):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 5000)
        self.sock.connect(f"tcp://{addr}")

    def call(self, method, **params):
        self.sock.send_json({"method": method, "params": params})
        return self.sock.recv_json()


def write_eps(path, eps):
    with open(path, "w") as f:
        f.write("\n".join(eps) + "\n")


def serving(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(url + "/v1/models", timeout=2).status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--router", default="127.0.0.1:15555")
    ap.add_argument("--endpoints-file", required=True)
    ap.add_argument("--events-out", required=True)
    ap.add_argument("--r0", default=os.environ.get("CONNEX_R0_URL"),
                    help="stage-0 API URL of replica 0 (env CONNEX_R0_URL)")
    ap.add_argument("--r1", default=os.environ.get("CONNEX_R1_URL"),
                    help="stage-0 API URL of replica 1 (env CONNEX_R1_URL)")
    ap.add_argument("--r1-base-port", default="25510",
                    help="replica 1's data-plane base port (deploy/k8s/manifests.py: 25500 + 10*replica)")
    ap.add_argument("--namespace", default=os.environ.get("CONNEX_NAMESPACE", "connex"))
    ap.add_argument("--model", default=None, help="served model name (default PREFIX-model)")
    a = ap.parse_args()
    if not (a.r0 and a.r1):
        ap.error("--r0/--r1 (or CONNEX_R0_URL/CONNEX_R1_URL) are required")
    R0, R1, R1_BASE_PORT, NS = a.r0, a.r1, ":" + a.r1_base_port.lstrip(":"), a.namespace
    os.environ["CONNEX_NAMESPACE"] = NS  # for wait_ready.sh

    model = a.model or f"{a.prefix}-model"
    router = Router(a.router)
    write_eps(a.endpoints_file, [R0, R1])
    t0 = time.time()
    events, saved, state = [], [], "up"   # up -> down -> (hard: rejoining) -> up
    next_t = t0 + a.interval

    def log(kind, **kw):
        events.append({"t": round(time.time() - t0, 2), "kind": kind, **kw})
        print(json.dumps(events[-1]), flush=True)

    while time.time() - t0 < a.duration:
        time.sleep(0.5)
        if state == "rejoining" and serving(R1) and subprocess.run(
                [os.path.join(os.path.dirname(os.path.abspath(__file__)), "wait_ready.sh"), a.prefix, "1", "replica=1"],
                capture_output=True).returncode == 0:  # all replica-1 stages serving
            write_eps(a.endpoints_file, [R0, R1])
            state = "up"
            log("r1_serving")
        if time.time() < next_t:
            continue
        next_t += a.interval
        if state == "up":
            write_eps(a.endpoints_file, [R0])
            if a.mode == "soft":
                routes = router.call("get_routes", model_name=model).get("routes", [])
                saved = [r for r in routes if r["lid"].get("replica_idx") == 1]  # 3-node layout: ports differ per stage
                for r in saved:
                    router.call("deregister", lid=r["lid"])
            else:
                subprocess.run(["kubectl", "-n", NS, "scale", "deploy", "-l",
                                f"pipeline={a.prefix},replica=1", "--replicas=0"],
                               check=False, capture_output=True)
            state = "down"
            log("down", routes=len(saved))
        elif state == "down":
            if a.mode == "soft":
                for r in saved:
                    router.call("register", lid=r["lid"], addresses=r["addresses"],
                                capabilities=r.get("capabilities", [0]))
                write_eps(a.endpoints_file, [R0, R1])
                state = "up"
            else:
                subprocess.run(["kubectl", "-n", NS, "scale", "deploy", "-l",
                                f"pipeline={a.prefix},replica=1", "--replicas=1"],
                               check=False, capture_output=True)
                state = "rejoining"
            log("up")
    # Leave replica 1 out if it is still down or rejoining; the load generator
    # may outlive this script and must not be pointed at a dead address.
    write_eps(a.endpoints_file, [R0, R1] if state == "up" else [R0])
    json.dump({"t0": t0, "events": events, "config": vars(a)}, open(a.events_out, "w"), indent=1)


if __name__ == "__main__":
    main()
