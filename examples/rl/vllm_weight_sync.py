# SPDX-License-Identifier: Apache-2.0
"""Connex weight sync into running vLLM servers (vLLM's weight-transfer API,
backend "connex").

Start one or more engines (connex installed in their environment, so the
backend is registered through the vllm.general_plugins entry point):

    VLLM_SERVER_DEV_MODE=1 vllm serve MODEL --port 8100 --enforce-eager \\
        --weight-transfer-config '{"backend": "connex"}'

then run the trainer side:

    python vllm_weight_sync.py --model MODEL --engines http://HOST:8100,...

It publishes version 1 (the original weights), checks greedy outputs match
the pre-sync baseline, publishes version 2 (perturbed weights) and checks
the outputs changed, then version 3 (original again) and checks they match
the baseline again. Every engine reports its version to the registry.
"""
import argparse
import json
import time

import requests
import torch

from connex.rl import RegistryServer, WeightRegistry
from connex.rl.vllm_backend import (ConnexTrainerInitInfo, ConnexTrainerWeightTransferEngine,
                                    PoolClient)

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Three primary colors are"]


def generate(url, model, n=24):
    out = []
    for p in PROMPTS:
        r = requests.post(f"{url}/v1/completions", json={
            "model": model, "prompt": p, "max_tokens": n, "temperature": 0}, timeout=120)
        r.raise_for_status()
        out.append(r.json()["choices"][0]["text"])
    return out


def main():
    from transformers import AutoModelForCausalLM
    from vllm.distributed.weight_transfer.base import ModuleSource
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--registry-port", type=int, default=29700)
    ap.add_argument("--registry-host", default=None, help="address engines use (default: advertise ip)")
    ap.add_argument("--transport", default="connex-rs")
    a = ap.parse_args()
    from connex.rl.transport import advertise_ip
    host = a.registry_host or advertise_ip()
    srv = RegistryServer(WeightRegistry(fanout=2), port=a.registry_port)
    reg_url = f"http://{host}:{srv.port}"
    urls = [u.rstrip("/") for u in a.engines.split(",")]
    name = a.served_name or a.model

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device)
    base = {u: generate(u, name) for u in urls}
    client = PoolClient({u: HTTPVLLMWeightSyncClient(u) for u in urls})
    t0 = time.time()
    eng = ConnexTrainerWeightTransferEngine.trainer_init(
        ConnexTrainerInitInfo(rank=0, registry_url=reg_url, transport=a.transport,
                              device=a.device),
        client=client, source=ModuleSource(model))
    print(json.dumps({"init_s": time.time() - t0}))

    def sync(tag):
        t = time.time()
        eng.send_weights()
        dt = time.time() - t
        st = srv.reg.status()
        print(json.dumps({"step": tag, "version": eng.version, "send_weights_s": round(dt, 3),
                          "members": st["members"], "failed": client.failed}), flush=True)

    sync("v1 original")
    for u in urls:
        assert generate(u, name) == base[u], "v1 must reproduce the baseline"
    saved = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "mlp" in n:
                saved[n] = p.detach().clone()
                p.mul_(0.5)
    sync("v2 perturbed")
    changed = [generate(u, name) != base[u] for u in urls]
    assert all(changed), "v2 must change the outputs"
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    sync("v3 original")
    for u in urls:
        assert generate(u, name) == base[u], "v3 must reproduce the baseline"
    print(json.dumps({"ok": True, "engines": len(urls)}))
    eng.shutdown()
    srv.close()


if __name__ == "__main__":
    main()
