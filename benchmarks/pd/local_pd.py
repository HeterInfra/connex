#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start, kill and scale a PD deployment on one host (vLLM or SGLang).

Used by run_pd_experiments.py; also handy by hand:

    python benchmarks/pd/local_pd.py --engine vllm --connector connex \\
        --model /models/llama3-8b --prefill-gpus 1 --decode-gpus 2 --hold

Every instance is a subprocess with its own GPU(s), port and log file in
``--logdir``. A GPU spec of ``1+2`` starts a tensor-parallel instance (TP 2)
on GPUs 1 and 2; ``Deployment.kill_rank`` kills one TP rank's process. The PD router (connex.serving.pd_router) runs in-process or as
a subprocess; instances register in a Connex front-router when
``--front-router`` is on, else the router gets static URL lists.

Several hosts: ``--hosts hosts.json`` names hosts and how to run a shell on
each, and a GPU spec ``B:2+3`` starts the instance on host ``B``::

    {"B": {"ip": "10.0.0.2", "exec": ["ssh", "node-b"]},
     "C": {"ip": "10.0.0.3", "exec": ["kubectl", "exec", "agent-c", "--"],
           "env": {"CONNEX_IB_DEVICE": "mlx5_1"}}}

``ip`` is the address peers use (the RDMA NIC's IP); ``exec`` is a command
prefix that runs its arguments on the host (``ssh``, ``kubectl exec`` into an
agent pod, ...). The model path, ``--python`` and this repository must exist
at the same paths there (``python``/``root`` override them per host). The
routers run here and must be reachable from the hosts (``CONNEX_HOST_IP``).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def wait_http(url, timeout=900, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{url}: process exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise TimeoutError(url)


def _rdma_ip():
    from connex.pp.transports import _if_ipv4

    try:
        for d in sorted(os.listdir("/sys/class/infiniband")):
            for n in sorted(os.listdir(f"/sys/class/infiniband/{d}/device/net")):
                ip = _if_ipv4(n)
                if ip:
                    return ip
    except OSError:
        pass
    return None


class Instance:
    host = None

    def __init__(self, name, role, gpu, port, cmd, env, log):
        self.name, self.role, self.gpu, self.port = name, role, gpu, port
        self.url = f"http://127.0.0.1:{port}"
        self.cmd, self.env, self.logpath = cmd, env, log
        self.proc = None

    def start(self):
        self.log = open(self.logpath, "ab")
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        return self

    def kill(self, sig=signal.SIGKILL):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, sig)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def tp(self):
        return len(str(self.gpu).split("+"))

    def rank_pids(self):
        """{TP rank: pid} of this instance's worker processes (vLLM names
        them ``VLLM::Worker_TP<r>``, SGLang ``sglang::scheduler_TP<r>``)."""
        import re

        out = {}
        if not self.alive():
            return out
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if os.getpgid(int(d)) != self.proc.pid:
                    continue
                with open(f"/proc/{d}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            except (OSError, ProcessLookupError):
                continue
            m = re.search(r"(?:Worker|scheduler)_TP(\d+)", cmd)
            if m:
                out[int(m.group(1))] = int(d)
        return out

    def kill_rank(self, rank, sig=signal.SIGKILL):
        """Kill one TP rank's process (not the rest of the group); returns the
        time of the kill or None if the rank was not found."""
        pid = self.rank_pids().get(rank)
        if pid is None:
            return None
        t = time.time()
        os.kill(pid, sig)
        return t


class RemoteInstance(Instance):
    """An instance on another host, started and killed through ``host["exec"]``.

    The process runs in its own session there (``setsid``), so its pid is
    also its process group; its log stays on the host until ``fetch_log``."""

    def __init__(self, name, role, gpu, port, cmd, env, log, host, hostname):
        super().__init__(name, role, gpu, port, cmd, env, log)
        self.host, self.hostname = host, hostname
        self.url = f"http://{host['ip']}:{port}"
        self.remote_log = f"/tmp/connex-pd-{name}-{port}.log"
        self.pid = None
        self.kill_times = []      # (local time before, host time of the kill, local after)

    def _sh(self, script, timeout=60):
        r = subprocess.run([*self.host["exec"], "bash", "-c", script], capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout

    def start(self):
        import shlex

        envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in self.env.items())
        cmd = " ".join(shlex.quote(c) for c in self.cmd)
        pidf = f"{self.remote_log}.pid"
        if not getattr(self, "_started", False):   # a fresh log per deployment
            self._sh(f"rm -f {self.remote_log}")
            self._started = True
        inner = f"echo $$ > {pidf}; cd /tmp; exec env -i {envs} {cmd}"
        out = self._sh(f"rm -f {pidf}; setsid -f bash -c {shlex.quote(inner)} >> {self.remote_log} "
                       f"2>&1 < /dev/null; for i in $(seq 50); do [ -s {pidf} ] && break; "
                       f"sleep 0.1; done; cat {pidf}")
        self.pid = int(out.strip().splitlines()[-1])
        return self

    def kill(self, sig=signal.SIGKILL):
        if self.pid is None:
            return
        n = int(sig)
        self._sh(f"kill -{n} -{self.pid} 2>/dev/null; for i in $(seq 60); do "
                 f"kill -0 -{self.pid} 2>/dev/null || exit 0; sleep 0.5; done; "
                 f"kill -9 -{self.pid} 2>/dev/null", timeout=90)

    def alive(self):
        if self.pid is None:
            return False
        return self._sh(f"kill -0 {self.pid} 2>/dev/null && echo y").strip() == "y"

    def rank_pids(self):
        import re

        out = {}
        if self.pid is None:
            return out
        ps = self._sh(f"ps -o pid=,args= -g {self.pid}")
        for line in ps.splitlines():
            m = re.search(r"(?:Worker|scheduler)_TP(\d+)", line)
            if m:
                out[int(m.group(1))] = int(line.split()[0])
        return out

    def kill_rank(self, rank, sig=signal.SIGKILL):
        pid = self.rank_pids().get(rank)
        if pid is None:
            return None
        t0 = time.time()
        out = self._sh(f"date +%s.%N; kill -{int(sig)} {pid}")
        t1 = time.time()
        th = float(out.split()[0])
        self.kill_times.append((t0, th, t1))
        # the host's clock may be off; the local midpoint of the round trip is
        # within half of it of the kill
        return (t0 + t1) / 2

    def kill_group(self, sig=signal.SIGKILL):
        """SIGKILL the whole instance at once; returns the local kill time."""
        t0 = time.time()
        self._sh(f"kill -{int(sig)} -{self.pid}")
        t1 = time.time()
        self.kill_times.append((t0, None, t1))
        return (t0 + t1) / 2

    def gpus_free(self, need_mib=30000):
        out = self._sh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
        free = dict(x.replace(" ", "").split(",") for x in out.strip().splitlines())
        return all(int(free.get(g, 0)) >= need_mib for g in str(self.gpu).split("+"))

    def fetch_log(self):
        try:
            data = subprocess.run([*self.host["exec"], "cat", self.remote_log],
                                  capture_output=True, timeout=120).stdout
            with open(self.logpath, "ab") as f:
                f.write(data)
        except Exception:  # noqa: BLE001
            pass


def split_spec(spec):
    """``"B:2+3"`` -> ``("B", "2+3")``; ``"2+3"`` -> ``(None, "2+3")``."""
    spec = str(spec)
    if ":" in spec:
        h, g = spec.split(":", 1)
        return h, g
    return None, spec


class Deployment:
    """One host, one model, prefill and decode pools plus the PD router."""

    def __init__(self, engine, connector, model, logdir, python, gpu_mem=0.85,
                 max_model_len=4096, base_port=8100, router_port=8000, front_router=False,
                 extra_args=(), retain_s=60.0, hosts=None):
        self.engine, self.connector, self.model = engine, connector, model
        self.logdir, self.python = logdir, python
        self.gpu_mem, self.max_model_len = gpu_mem, max_model_len
        # CONNEX_PD_PORT_OFFSET shifts every port, so that two deployments can
        # share hosts (instances, routers, control and data-plane ports)
        self.po = int(os.getenv("CONNEX_PD_PORT_OFFSET", "0"))
        self.base_port, self.router_port = base_port + self.po, router_port + self.po
        self.front_router = front_router
        self.extra_args = list(extra_args)
        self.retain_s = retain_s
        self.hosts = hosts or {}
        self.instances: list[Instance] = []
        # data-plane address: the RDMA (IPoIB) interface if there is one; UCX
        # cannot reach 127.0.0.1 over an RDMA device
        self.host_ip = os.getenv("CONNEX_HOST_IP") or _rdma_ip() or "127.0.0.1"
        # where the instances reach the routers: loopback unless some are remote
        self.ctrl_ip = self.host_ip if self.hosts else "127.0.0.1"
        self.router = None
        self.fr = None
        self.n = 0
        os.makedirs(logdir, exist_ok=True)

    # -- instance commands -------------------------------------------------------
    def _env(self, gpu, host=None):
        if host is None:
            env = dict(os.environ)
            root = ROOT
        else:   # a clean environment on the host: only what the instance needs
            keep = ("HF_HOME", "HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL")
            env = {k: os.environ[k] for k in keep if k in os.environ}
            env["PATH"] = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            env["HOME"] = "/root"
            env.update({k: v for k, v in os.environ.items()
                        if k.startswith(("CONNEX_", "UCX_", "SGLANG_", "VLLM_", "NCCL_"))
                        and k != "CONNEX_HOST_IP"})
            env["CONNEX_HOST_IP"] = host["ip"]
            env.update(host.get("env", {}))
            root = host.get("root", ROOT)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).replace("+", ",")
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("CONNEX_HOST_IP", self.host_ip)
        # host nvcc may be older than the wheel's CUDA: avoid FlashInfer JIT kernels
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        env.pop("http_proxy", None), env.pop("https_proxy", None)
        env.pop("HTTP_PROXY", None), env.pop("HTTPS_PROXY", None)
        env["NO_PROXY"] = "*"
        if self.engine == "sglang":
            env.setdefault("SGLANG_HOST_IP", env["CONNEX_HOST_IP"])   # KV manager / bootstrap address
            # sglang 0.5.22 wheels pin flashinfer 0.7.1 but flashinfer-cubin 0.6.3
            env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        return env

    def _ip(self, host):
        return host["ip"] if host else "127.0.0.1"

    def _vllm_cmd(self, role, port, idx, tp=1, host=None):
        cmd = [(host or {}).get("python", self.python), "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
               "--served-model-name", "model", "--port", str(port),
               "--gpu-memory-utilization", str(self.gpu_mem),
               "--max-model-len", str(self.max_model_len),
               *(["--tensor-parallel-size", str(tp)] if tp > 1 else []),
               *self.extra_args]
        if role == "single":
            return cmd
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        if self.connector == "connex":
            extra = {"ctrl_port": 17100 + self.po + 10 * idx, "retain_s": self.retain_s,
                     "replica": idx, "http_url": f"http://{self._ip(host)}:{port}"}
            if self.front_router:
                extra["front_router"] = f"tcp://{self.ctrl_ip}:{25555 + self.po}"
            cfg = {"kv_connector": "ConnexConnector",
                   "kv_connector_module_path": "connex.adapters.vllm.pd_connector",
                   "kv_role": kv_role, "kv_connector_extra_config": extra}
        elif self.connector == "nixl":
            cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        else:
            raise ValueError(self.connector)
        return cmd + ["--kv-transfer-config", json.dumps(cfg)]

    def _sglang_cmd(self, role, port, idx, tp=1, host=None):
        cmd = [(host or {}).get("python", self.python), "-m", "sglang.launch_server",
               "--model-path", self.model,
               "--served-model-name", "model", "--port", str(port), "--host", self._ip(host),
               "--mem-fraction-static", str(self.gpu_mem),
               "--context-length", str(self.max_model_len),
               "--attention-backend", "triton", "--sampling-backend", "pytorch",
               "--page-size", "16",
               *(["--tp-size", str(tp)] if tp > 1 else []),
               *self.extra_args]
        if role == "single":
            return cmd
        cmd += ["--disaggregation-mode", role,
                "--disaggregation-transfer-backend", self.connector]
        if role == "prefill":
            cmd += ["--disaggregation-bootstrap-port", str(8998 + self.po + idx)]
        if self.connector == "mooncake":
            cmd += ["--disaggregation-ib-device", (host or {}).get("env", {}).get(
                "CONNEX_IB_DEVICE", os.getenv("CONNEX_IB_DEVICE", "mlx5_0"))]
        return cmd

    def add(self, role, gpu, start=True):
        idx = self.n
        self.n += 1
        port = self.base_port + idx
        hostname, gpu = split_spec(gpu)
        host = self.hosts[hostname] if hostname else None
        ip = self._ip(host)
        tp = len(str(gpu).split("+"))
        cmd = (self._vllm_cmd if self.engine == "vllm" else self._sglang_cmd)(
            role, port, idx, tp, host)
        env = self._env(gpu, host)
        if self.connector == "nixl":
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5600 + self.po + idx)
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = ip   # default localhost: no cross-host handshake
        if self.engine == "sglang":
            # fixed connex-rs ports: a random free port can later collide with
            # one SGLang hands to its own components
            env.setdefault("CONNEX_RS_PORT", str(52000 + self.po + 10 * idx))
        if self.engine == "sglang" and self.front_router:
            env.update(CONNEX_PD_FRONT_ROUTER=f"tcp://{self.ctrl_ip}:{25555 + self.po}",
                       CONNEX_PD_HTTP_URL=f"http://{ip}:{port}", CONNEX_PD_MODEL="model",
                       CONNEX_PD_REPLICA=str(idx), CONNEX_PD_BOOTSTRAP=f"{ip}:{8998 + self.po + idx}")
        log = os.path.join(self.logdir, f"{role}{idx}.log")
        if host is None:
            inst = Instance(f"{role}{idx}", role, gpu, port, cmd, env, log)
        else:
            inst = RemoteInstance(f"{role}{idx}", role, gpu, port, cmd, env, log, host, hostname)
        self.instances.append(inst)
        if start:
            inst.start()
        return inst

    def start_front_router(self):
        self.fr = subprocess.Popen([sys.executable, "-m", "connex.front_router", "--api-address",
                                    f"tcp://*:{25555 + self.po}", "--event-address",
                                    f"tcp://*:{25560 + self.po}"],
                                   # members' lease as in production setups: 2 s
                                   env={**self._env(""), "CONNEX_HEARTBEAT_TIMEOUT_S":
                                        os.getenv("CONNEX_HEARTBEAT_TIMEOUT_S", "2")},
                                   start_new_session=True,
                                   stdout=open(os.path.join(self.logdir, "front_router.log"), "ab"),
                                   stderr=subprocess.STDOUT)

    def start_router(self, static=True):
        pre = [i.url for i in self.instances if i.role == "prefill"] if static else []
        dec = [i.url for i in self.instances if i.role == "decode"] if static else []
        cmd = [sys.executable, "-m", "connex.serving.pd_router", "--port", str(self.router_port),
               "--engine", self.engine, "--prefill", ",".join(pre), "--decode", ",".join(dec),
               "--prefill-file", os.path.join(self.logdir, "prefill.txt"),
               "--decode-file", os.path.join(self.logdir, "decode.txt")]
        if self.engine == "sglang":
            cmd += ["--bootstrap", ",".join(
                f"{i.url}={self._ip(i.host)}:{8998 + self.po + int(i.name[len(i.role):])}"
                for i in self.instances if i.role == "prefill")]
        if self.front_router:
            cmd += ["--front-router", f"tcp://{self.ctrl_ip}:{25555 + self.po}", "--model", "model",
                    "--front-router-events", f"tcp://{self.ctrl_ip}:{25560 + self.po}"]
        for f in ("prefill.txt", "decode.txt"):
            open(os.path.join(self.logdir, f), "a").close()   # keep files written before
        self.router = subprocess.Popen(cmd, env=self._env(""), start_new_session=True,
                                       stdout=open(os.path.join(self.logdir, "router.log"), "ab"),
                                       stderr=subprocess.STDOUT)
        self.router_url = f"http://127.0.0.1:{self.router_port}"
        return self.router_url

    def wait(self, insts=None, timeout=900):
        for i in insts or self.instances:
            wait_http(i.url, timeout, i.proc)

    def set_pool_file(self, role, urls):
        # atomic: the router re-reads the file every second and must never
        # see it truncated (an empty pool fails requests)
        path = os.path.join(self.logdir, f"{role}.txt")
        with open(path + ".tmp", "w") as f:
            f.write("\n".join(urls) + "\n")
        os.replace(path + ".tmp", path)

    def down(self):
        for i in self.instances:
            i.kill(signal.SIGTERM)
            if isinstance(i, RemoteInstance):
                i.fetch_log()
        for p in (self.router, self.fr):
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


def load_hosts(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--connector", default="connex")
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--prefill-gpus", default="1")
    ap.add_argument("--decode-gpus", default="2")
    ap.add_argument("--logdir", default="pd-logs")
    ap.add_argument("--front-router", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--router-port", type=int, default=8000)
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--extra", default="", help="extra engine args (space separated)")
    ap.add_argument("--single", default=None, help="also start a non-disaggregated instance on this GPU")
    ap.add_argument("--pool-files", action="store_true",
                    help="router reads pools only from prefill.txt / decode.txt in --logdir "
                         "(initially the first prefill and decode instance); edit them to scale")
    ap.add_argument("--hold", action="store_true", help="stay up until Ctrl-C")
    ap.add_argument("--hosts", default=None,
                    help="JSON file of remote hosts; GPU specs HOST:GPUS start there")
    a = ap.parse_args()
    d = Deployment(a.engine, a.connector, a.model, a.logdir, a.python, gpu_mem=a.gpu_mem,
                   max_model_len=a.max_model_len, base_port=a.base_port,
                   router_port=a.router_port, front_router=a.front_router,
                   extra_args=a.extra.split(), hosts=load_hosts(a.hosts))
    if a.front_router:
        d.start_front_router()
    gpus = [("prefill", g) for g in a.prefill_gpus.split(",")] + \
        [("decode", g) for g in a.decode_gpus.split(",")] + \
        ([("single", a.single)] if a.single is not None else [])
    used = [(split_spec(g)[0], x) for _, g in gpus for x in split_spec(g)[1].split("+")]
    shared = len(set(used)) < len(used)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # run the finally below
    try:
        for role, g in gpus:
            inst = d.add(role, g)
            if shared:  # vLLM's memory profiling breaks if a GPU neighbour starts meanwhile
                d.wait([inst])
        d.wait()
        if a.pool_files:
            pre = [i for i in d.instances if i.role == "prefill"]
            dec = [i for i in d.instances if i.role == "decode"]
            d.set_pool_file("prefill", [pre[0].url])
            d.set_pool_file("decode", [dec[0].url])
        print("router:", d.start_router(static=not a.pool_files), flush=True)
        for i in d.instances:
            print(f"instance {i.name} {i.url} gpu={i.gpu} pid={i.proc.pid}", flush=True)
        if a.hold:
            signal.pause()
    finally:
        d.down()


if __name__ == "__main__":
    main()
