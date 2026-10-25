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


class Deployment:
    """One host, one model, prefill and decode pools plus the PD router."""

    def __init__(self, engine, connector, model, logdir, python, gpu_mem=0.85,
                 max_model_len=4096, base_port=8100, router_port=8000, front_router=False,
                 extra_args=(), retain_s=60.0):
        self.engine, self.connector, self.model = engine, connector, model
        self.logdir, self.python = logdir, python
        self.gpu_mem, self.max_model_len = gpu_mem, max_model_len
        self.base_port, self.router_port = base_port, router_port
        self.front_router = front_router
        self.extra_args = list(extra_args)
        self.retain_s = retain_s
        self.instances: list[Instance] = []
        # data-plane address: the RDMA (IPoIB) interface if there is one; UCX
        # cannot reach 127.0.0.1 over an RDMA device
        self.host_ip = os.getenv("CONNEX_HOST_IP") or _rdma_ip() or "127.0.0.1"
        self.router = None
        self.fr = None
        self.n = 0
        os.makedirs(logdir, exist_ok=True)

    # -- instance commands -------------------------------------------------------
    def _env(self, gpu):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).replace("+", ",")
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("CONNEX_HOST_IP", self.host_ip)
        # host nvcc may be older than the wheel's CUDA: avoid FlashInfer JIT kernels
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        env.pop("http_proxy", None), env.pop("https_proxy", None)
        env.pop("HTTP_PROXY", None), env.pop("HTTPS_PROXY", None)
        env["NO_PROXY"] = "*"
        if self.engine == "sglang":
            env.setdefault("SGLANG_HOST_IP", self.host_ip)   # KV manager / bootstrap address
            # sglang 0.5.22 wheels pin flashinfer 0.7.1 but flashinfer-cubin 0.6.3
            env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        return env

    def _vllm_cmd(self, role, port, idx, tp=1):
        cmd = [self.python, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
               "--served-model-name", "model", "--port", str(port),
               "--gpu-memory-utilization", str(self.gpu_mem),
               "--max-model-len", str(self.max_model_len),
               *(["--tensor-parallel-size", str(tp)] if tp > 1 else []),
               *self.extra_args]
        if role == "single":
            return cmd
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        if self.connector == "connex":
            extra = {"ctrl_port": 17100 + 10 * idx, "retain_s": self.retain_s,
                     "replica": idx, "http_url": f"http://127.0.0.1:{port}"}
            if self.front_router:
                extra["front_router"] = "tcp://127.0.0.1:25555"
            cfg = {"kv_connector": "ConnexConnector",
                   "kv_connector_module_path": "connex.adapters.vllm.pd_connector",
                   "kv_role": kv_role, "kv_connector_extra_config": extra}
        elif self.connector == "nixl":
            cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        else:
            raise ValueError(self.connector)
        return cmd + ["--kv-transfer-config", json.dumps(cfg)]

    def _sglang_cmd(self, role, port, idx, tp=1):
        cmd = [self.python, "-m", "sglang.launch_server", "--model-path", self.model,
               "--served-model-name", "model", "--port", str(port), "--host", "127.0.0.1",
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
            cmd += ["--disaggregation-bootstrap-port", str(8998 + idx)]
        if self.connector == "mooncake":
            cmd += ["--disaggregation-ib-device", os.getenv("CONNEX_IB_DEVICE", "mlx5_0")]
        return cmd

    def add(self, role, gpu, start=True):
        idx = self.n
        self.n += 1
        port = self.base_port + idx
        tp = len(str(gpu).split("+"))
        cmd = (self._vllm_cmd if self.engine == "vllm" else self._sglang_cmd)(role, port, idx, tp)
        env = self._env(gpu)
        if self.connector == "nixl":
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5600 + idx)
        if self.engine == "sglang":
            # fixed connex-rs ports: a random free port can later collide with
            # one SGLang hands to its own components
            env.setdefault("CONNEX_RS_PORT", str(52000 + 10 * idx))
        if self.engine == "sglang" and self.front_router:
            env.update(CONNEX_PD_FRONT_ROUTER="tcp://127.0.0.1:25555",
                       CONNEX_PD_HTTP_URL=f"http://127.0.0.1:{port}", CONNEX_PD_MODEL="model",
                       CONNEX_PD_REPLICA=str(idx), CONNEX_PD_BOOTSTRAP=f"127.0.0.1:{8998 + idx}")
        inst = Instance(f"{role}{idx}", role, gpu, port, cmd, env,
                        os.path.join(self.logdir, f"{role}{idx}.log"))
        self.instances.append(inst)
        if start:
            inst.start()
        return inst

    def start_front_router(self):
        self.fr = subprocess.Popen([sys.executable, "-m", "connex.front_router", "--api-address",
                                    "tcp://*:25555", "--event-address", "tcp://*:25560"],
                                   env=self._env(""), start_new_session=True,
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
            cmd += ["--bootstrap", ",".join(f"{i.url}=127.0.0.1:{8998 + int(i.name[len(i.role):])}"
                                            for i in self.instances if i.role == "prefill")]
        if self.front_router:
            cmd += ["--front-router", "tcp://127.0.0.1:25555", "--model", "model"]
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
        with open(os.path.join(self.logdir, f"{role}.txt"), "w") as f:
            f.write("\n".join(urls) + "\n")

    def down(self):
        for i in self.instances:
            i.kill(signal.SIGTERM)
        for p in (self.router, self.fr):
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


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
    a = ap.parse_args()
    d = Deployment(a.engine, a.connector, a.model, a.logdir, a.python, gpu_mem=a.gpu_mem,
                   max_model_len=a.max_model_len, base_port=a.base_port,
                   router_port=a.router_port, front_router=a.front_router,
                   extra_args=a.extra.split())
    if a.front_router:
        d.start_front_router()
    gpus = [("prefill", g) for g in a.prefill_gpus.split(",")] + \
        [("decode", g) for g in a.decode_gpus.split(",")] + \
        ([("single", a.single)] if a.single is not None else [])
    used = [x for _, g in gpus for x in str(g).split("+")]
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
