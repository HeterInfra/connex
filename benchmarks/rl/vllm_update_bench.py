# SPDX-License-Identifier: Apache-2.0
"""End-to-end weight update time through vLLM's weight-transfer API:
the built-in "nccl" backend vs "connex", one engine, same trainer code.

Start the engine with the backend under test, e.g.

    VLLM_SERVER_DEV_MODE=1 vllm serve MODEL --port 8141 \\
        --weight-transfer-config '{"backend": "nccl"}'      # or "connex"

then

    python vllm_update_bench.py --backend nccl --engine http://127.0.0.1:8141 --model MODEL

Times ``send_weights()`` (start + transfer + load + finish, as the trainer
sees it) over ``--rounds`` rounds after one warm-up, and checks the engine
still answers.
"""
import argparse
import json
import socket
import time

import requests
import torch


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    from transformers import AutoModelForCausalLM
    from vllm.distributed.weight_transfer.base import ModuleSource
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from connex.rl.transport import advertise_ip

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nccl", "connex"], required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--buffers", type=int, default=2, help="connex publisher arenas")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device)
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    client = HTTPVLLMWeightSyncClient(a.engine)
    host = advertise_ip()
    srv = None
    t0 = time.time()
    if a.backend == "nccl":
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
        ws = requests.get(f"{a.engine}/get_world_size", timeout=10,
                          proxies={"http": None}).json()["world_size"] + 1
        info = NCCLTrainerInitInfo(rank=0, master_address=host, master_port=free_port(),
                                   world_size=ws)
    else:
        from connex.rl import RegistryServer, WeightRegistry
        from connex.rl.vllm_backend import ConnexTrainerInitInfo, register
        register()
        srv = RegistryServer(WeightRegistry())
        info = ConnexTrainerInitInfo(rank=0, registry_url=f"http://{host}:{srv.port}",
                                     device=a.device, buffers=a.buffers)
    eng = WeightTransferTrainerFactory.trainer_init(info, client=client,
                                                    source=ModuleSource(model))
    init_s = time.time() - t0
    times = []
    for r in range(a.rounds + 1):
        torch.cuda.synchronize()
        t = time.time()
        eng.send_weights()
        times.append(time.time() - t)
    r = requests.post(f"{a.engine}/v1/completions", json={
        "model": a.model, "prompt": "Hello", "max_tokens": 4, "temperature": 0},
        proxies={"http": None}, timeout=60)
    rec = {"backend": a.backend, "model": a.model.rstrip("/").split("/")[-1],
           "gbytes": round(nbytes / 1e9, 3), "init_s": round(init_s, 3),
           "first_s": round(times[0], 3), "send_weights_s": [round(x, 3) for x in times[1:]],
           "mean_s": round(sum(times[1:]) / a.rounds, 3), "engine_ok": r.status_code == 200}
    if r.status_code != 200:
        rec["engine_error"] = r.text[:300]
    print(json.dumps(rec), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    eng.shutdown()
    if srv:
        srv.close()


if __name__ == "__main__":
    main()
