# SPDX-License-Identifier: Apache-2.0
"""Connex weight sync into running SGLang servers (custom weight loader).

Start one or more servers (connex installed in their environment):

    python -m sglang.launch_server --model-path MODEL --port 30000 \\
        --custom-weight-loader connex.rl.sglang_ext.connex_load

then run:

    python sglang_weight_sync.py --model MODEL --engines http://HOST:30000,...

Version 1 (original weights) sets the greedy reference, version 2
(perturbed) changes it, version 3 (original) reproduces version 1 exactly;
every response reports the weight version SGLang served it with.
"""
import argparse
import asyncio
import json
import time

import torch

from connex.rl import RegistryServer, WeightPublisher, WeightRegistry, make_transport
from connex.rl.engines import SGLangInstance
from connex.rl.transport import advertise_ip

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Three primary colors are"]


async def main_async(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device)
    srv = RegistryServer(WeightRegistry(fanout=2), port=a.registry_port)
    reg_url = f"http://{advertise_ip()}:{srv.port}"
    pub = WeightPublisher(srv.reg, make_transport("connex-rs", lid=("sgl", 0, 0, 0)),
                          device=a.device)
    insts = [SGLangInstance(f"s{i}", u, a.model, reg_url, member=f"s{i}")
             for i, u in enumerate(a.engines.split(","))]
    reqs = [{"prompt_ids": tok(p).input_ids, "max_tokens": 24} for p in PROMPTS]

    async def gen(inst):
        outs = [await inst.generate(r) for r in reqs]
        return [o["tokens"] for o in outs], {o["weight_version"] for o in outs}

    base = [(await gen(i))[0] for i in insts]

    async def sync(tag):
        v = pub.publish([(n, p.detach()) for n, p in model.named_parameters()])
        t = time.time()
        await asyncio.gather(*(i.update(v) for i in insts))
        dt = time.time() - t
        print(json.dumps({"step": tag, "version": v, "update_s": round(dt, 3),
                          "members": srv.reg.status()["members"]}), flush=True)
        return v

    v = await sync("v1 original")
    # Reference: outputs after an in-place load of the original weights. They
    # can differ from the startup outputs (``base``) by a near-tie token: SGLang's
    # own update_weights_from_tensor with the same tensors shows the same flip.
    ref = []
    for i, b in zip(insts, base):
        toks, vers = await gen(i)
        assert vers == {str(v)}, (vers, v)
        ref.append(toks)
        print(json.dumps({"v1_equals_startup_output": toks == b}))
    base = ref
    saved = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "mlp" in n:
                saved[n] = p.detach().clone()
                p.mul_(0.5)
    v = await sync("v2 perturbed")
    for i, b in zip(insts, base):
        toks, vers = await gen(i)
        assert toks != b and vers == {str(v)}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    v = await sync("v3 original")
    for i, b in zip(insts, base):
        toks, vers = await gen(i)
        assert toks == b and vers == {str(v)}
    print(json.dumps({"ok": True, "engines": len(insts)}))
    for i in insts:
        await i.close()
    srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--registry-port", type=int, default=29720)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
