#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start, kill and restart a multi-replica external-PP deployment on one host,
one docker container per stage (the image built from docker/Dockerfile).

    python benchmarks/pp/local_pp.py --image connex-vllm:dev --model /models/Llama-3-8B \\
        --chains "0+1,2+3" --hold
    # two replica chains of two stages, TP 1:  --chains "0,1;2,3"
    # one chain, two stages of TP 2:           --chains "0+1,2+3"

``--chains``: chains separated by ``;``, stages by ``,``, the GPUs of one
stage by ``+`` (their count is the stage's TP size). Replica r's stage 0
serves the OpenAI API on port 8000 + 10 r. ``--vllm-dir`` bind-mounts a
patched fork package over the image's (development), ``--connex-dir`` the
Connex checkout over /opt/connex. Docker resolves bind-mount paths on the
host: ``--host-prefix OLD=NEW`` rewrites them when this shell sees a
different mount namespace.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_http(url, timeout=900, alive=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if alive is not None and not alive():
            raise RuntimeError(f"{url}: container exited")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise TimeoutError(url)


class Stage:
    def __init__(self, dep, replica, stage, gpus):
        self.dep, self.replica, self.stage, self.gpus = dep, replica, stage, gpus
        self.tp = len(gpus.split("+"))
        self.name = f"{dep.prefix}-r{replica}-s{stage}"
        self.api_port = 8000 + 10 * replica + stage
        self.url = f"http://127.0.0.1:{self.api_port}"

    def env(self):
        d = self.dep
        e = {
            "STAGE": str(self.stage), "NUM_STAGES": str(d.num_stages), "REPLICA": str(self.replica),
            "ROUTER_ADDR": f"127.0.0.1:{d.router_port}", "HOST_IP": d.host_ip,
            "MODEL_PATH": "/model", "MODEL_NAME": "model", "NUM_LAYERS": str(d.num_layers),
            "API_PORT": str(self.api_port),
            "BASE_PORT": str(25500 + 100 * self.replica + 10 * self.stage),
            "MOONCAKE_BASE_PORT": str(50000 + 1000 * self.replica + 100 * self.stage),
            "GPU_SELECT": "none", "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(self.tp)),
            "TP": str(self.tp), "GPU_MEM_UTIL": str(d.gpu_mem),
            "BACKEND": "connexte", "MOONCAKE_PROTOCOL": d.protocol,
            "MOONCAKE_METADATA_SERVER": "P2PHANDSHAKE",
            "CONNEX_HEARTBEAT_INTERVAL_S": "0.5",
            "CONNEXTE_SEND_TIMEOUT_MS": "3000", "CONNEXTE_RECV_TIMEOUT_MS": "3000",
            "CONNEXTE_SEND_RETRIES": "3", "CONNEXTE_RECV_RETRIES": "15",
            "CONNEX_PP_INFLIGHT": "1", "UCX_WARN_UNUSED_ENV_VARS": "n",
            "NO_PROXY": "*", "no_proxy": "*",
        }
        e.update(d.extra_env)
        if d.extra_args:
            e["EXTRA_ARGS"] = d.extra_args
        return e

    def start(self):
        d = self.dep
        if subprocess.run(["docker", "inspect", self.name], capture_output=True).returncode == 0:
            # keep the previous incarnation's log (restart after a kill)
            n = len([f for f in os.listdir(d.logdir) if f.startswith(self.name + ".prev")])
            with open(os.path.join(d.logdir, f"{self.name}.prev{n}.log"), "w") as f:
                f.write(self.logs())
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", self.name, "--network", "host", "--ipc", "host",
               "--shm-size", "16g", "--gpus", f'"device={self.gpus.replace("+", ",")}"',
               "-v", f"{d.host(d.model)}:/model:ro", "-v", f"{d.host(d.connex_dir)}:/opt/connex"]
        if d.vllm_dir:
            cmd += ["-v", f"{d.host(d.vllm_dir)}:/usr/local/lib/python3.12/dist-packages/vllm"]
        if d.rdma:
            cmd += ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK"]
        for k, v in self.env().items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["--entrypoint", "bash", d.image, "/opt/connex/docker/launch_stage.sh"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"docker run {self.name}: {r.stderr}")
        return self

    def alive(self):
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name],
                           capture_output=True, text=True)
        return r.stdout.strip() == "true"

    def ready(self, timeout=900):
        """Stage 0: API up. Stages 1+: the pipeline loop is running."""
        if self.stage == 0:
            return wait_http(self.url, timeout, self.alive)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.alive():
                raise RuntimeError(f"{self.name} exited")
            if "pipeline loop thread started" in self.logs() or \
                    "entering tight pipeline loop" in self.logs():
                return time.time() - t0
            time.sleep(2)
        raise TimeoutError(self.name)

    def logs(self, tail=None):
        cmd = ["docker", "logs", self.name] + (["--tail", str(tail)] if tail else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout + r.stderr

    def pids(self):
        """{TP rank: host pid} of the stage's worker processes."""
        r = subprocess.run(["docker", "top", self.name, "-eo", "pid,args"],
                           capture_output=True, text=True)
        import re

        out = {}
        for line in r.stdout.splitlines()[1:]:
            pid, _, args = line.strip().partition(" ")
            m = re.search(r"Worker(?:_PP\d+)?_TP(\d+)", args)
            if m:
                out.setdefault(int(m.group(1)), int(pid))
            elif self.tp == 1 and "EngineCore" in args:
                out.setdefault(0, int(pid))
        return out

    def kill_rank(self, rank):
        """SIGKILL one TP rank's worker process (inside the container: the
        host pid may not be visible from this shell's pid namespace)."""
        if rank not in self.pids():
            return None
        prog = ("import os,re,signal\n"
                "for p in os.listdir('/proc'):\n"
                " try: a=open(f'/proc/{p}/cmdline','rb').read().decode(errors='replace')\n"
                " except Exception: continue\n"
                f" m=re.search(r'Worker(?:_PP\\d+)?_TP(\\d+)',a)\n"
                f" if m and int(m.group(1))=={rank} and p.isdigit(): os.kill(int(p),signal.SIGKILL); print(p)\n")
        t = time.time()
        r = subprocess.run(["docker", "exec", self.name, "python3", "-c", prog],
                           capture_output=True, text=True)
        return t if r.stdout.strip() else None

    def kill(self):
        subprocess.run(["docker", "kill", self.name], capture_output=True)

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


class Deployment:
    def __init__(self, a):
        self.image, self.model, self.num_layers = a.image, a.model, a.num_layers
        self.connex_dir, self.vllm_dir = a.connex_dir, a.vllm_dir
        self.prefix, self.router_port = a.prefix, a.router_port
        self.gpu_mem, self.protocol, self.rdma = a.gpu_mem, a.protocol, a.protocol == "rdma"
        self.host_ip = a.host_ip
        self.extra_args = a.extra_args
        self.extra_env = dict(kv.split("=", 1) for kv in a.env)
        self.prefixes = [p.split("=", 1) for p in a.host_prefix]
        self.logdir = a.logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.chains = [[s for s in c.split(",") if s] for c in a.chains.split(";") if c]
        self.num_stages = len(self.chains[0])
        self.stages = [Stage(self, r, s, g) for r, c in enumerate(self.chains)
                       for s, g in enumerate(c)]
        self.fr = None

    def host(self, path):
        path = os.path.abspath(path)
        for old, new in self.prefixes:
            if path.startswith(old):
                return new + path[len(old):]
        return path

    def stage(self, replica, stage):
        return next(s for s in self.stages if (s.replica, s.stage) == (replica, stage))

    def start_router(self):
        env = dict(os.environ, CONNEX_HEARTBEAT_TIMEOUT_S="2", CONNEX_UNHEALTHY_GRACE_S="10",
                   PYTHONPATH=ROOT)
        self.fr = subprocess.Popen(
            [sys.executable, "-m", "connex.front_router",
             "--api-address", f"tcp://127.0.0.1:{self.router_port}",
             "--event-address", f"tcp://127.0.0.1:{self.router_port + 5}"],
            env=env, start_new_session=True,
            stdout=open(os.path.join(self.logdir, "front_router.log"), "ab"),
            stderr=subprocess.STDOUT)
        time.sleep(1)

    def up(self):
        self.start_router()
        # one at a time: vLLM memory profiling and NCCL port picking race otherwise
        for s in sorted(self.stages, key=lambda s: (-s.stage, s.replica)):
            s.start()
            s.ready()

    def save_logs(self):
        for s in self.stages:
            with open(os.path.join(self.logdir, f"{s.name}.log"), "w") as f:
                f.write(s.logs())

    def down(self):
        self.save_logs()
        for s in self.stages:
            s.remove()
        if self.fr is not None and self.fr.poll() is None:
            os.killpg(self.fr.pid, signal.SIGKILL)
            self.fr.wait()


def add_args(ap):
    ap.add_argument("--image", default="connex-vllm:dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-layers", type=int, default=32)
    ap.add_argument("--chains", default="0+1,2+3")
    ap.add_argument("--connex-dir", default=ROOT)
    ap.add_argument("--vllm-dir", default="")
    ap.add_argument("--host-prefix", action="append", default=[], help="OLD=NEW path rewrite")
    ap.add_argument("--prefix", default="cxpp")
    ap.add_argument("--router-port", type=int, default=15555)
    ap.add_argument("--host-ip", default="127.0.0.1")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--protocol", default="tcp", choices=["tcp", "rdma"])
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every stage")
    ap.add_argument("--logdir", default="pp-logs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--hold", action="store_true")
    a = ap.parse_args()
    d = Deployment(a)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        d.up()
        for s in d.stages:
            print(f"{s.name} gpus={s.gpus} tp={s.tp} " + (s.url if s.stage == 0 else ""), flush=True)
        if a.hold:
            signal.pause()
    finally:
        d.down()


if __name__ == "__main__":
    main()
