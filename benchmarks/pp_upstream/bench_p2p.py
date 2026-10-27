# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Per-hop latency of vLLM's PP tensor-dict transfer: NCCL (vLLM default)
vs the Connex PP group (connex-rs), through the same GroupCoordinator API
vLLM's worker uses (isend_tensor_dict / irecv_tensor_dict).

Two ranks ping-pong a {hidden_states, residual} dict of [tokens, hidden]
bf16 tensors; one-way latency = round trip / 2. Run on a vLLM that has
--pp-group-cls:

    python bench_p2p.py --gpus 0,1 --backend nccl
    python bench_p2p.py --gpus 0,1 --backend connex
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

import torch
import torch.multiprocessing as mp

# no model config here: size the same-host channel slots for the largest dict
os.environ.setdefault("CONNEX_PP_IPC_SLOT_MB", "40")


def _run_remote(rank: int, args, q) -> None:
    _run(rank, args, None, q, device=0)


def _run(rank: int, args, init_file: str | None, q, device: int | None = None) -> None:
    torch.cuda.set_device(rank if device is None else device)
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_pp_group, init_distributed_environment, initialize_model_parallel)

    group_cls = "" if args.backend == "nccl" else "connex.adapters.vllm.pp.group.ConnexPPGroup"
    extra = {}
    if device is not None:
        host, port = args.init.removeprefix("tcp://").rsplit(":", 1)
        extra = {"nnodes": 2, "node_rank": rank, "master_addr": host,
                 "master_port": int(port)}
    cfg = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=2,
                                                    pp_group_cls=group_cls, **extra))
    with set_current_vllm_config(cfg):
        init_method = args.init or f"file://{init_file}"
        init_distributed_environment(2, rank, init_method,
                                     rank if device is None else device, "nccl")
        initialize_model_parallel(1, 2)
        pp = get_pp_group()
        results = {}
        for tokens in [int(t) for t in args.tokens.split(",")]:
            x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
            td = {"hidden_states": x, "residual": x.clone()}
            lat = []
            for i in range(args.iters + args.warmup):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if rank == 0:
                    pp.isend_tensor_dict(td)
                    got, handles, post = pp.irecv_tensor_dict()
                else:
                    got, handles, post = pp.irecv_tensor_dict()
                for h in handles:
                    h.wait()
                for p in post:
                    p()
                if rank == 1:
                    pp.isend_tensor_dict(got)
                torch.cuda.synchronize()
                if i >= args.warmup:
                    lat.append(time.perf_counter() - t0)
            if rank == 0:
                lat.sort()
                results[tokens] = {
                    "bytes": 2 * x.numel() * x.element_size(),
                    "one_way_median_us": statistics.median(lat) / 2 * 1e6,
                    "one_way_p99_us": lat[int(len(lat) * 0.99) - 1] / 2 * 1e6,
                }
        if rank == 0:
            q.put(results)
        torch.distributed.barrier()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--backend", choices=["nccl", "connex"], default="nccl")
    ap.add_argument("--tokens", default="1,8,32,64,128,256,512,2048")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--init", default="", help="tcp://host:port: one rank per host "
                    "(with --rank), instead of two local processes")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    if args.init:
        import queue

        q = queue.Queue()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus.split(",")[0])
        _run_remote(args.rank, args, q)
        if args.rank == 0:
            out = {"backend": args.backend, "hosts": 2, "hidden": args.hidden,
                   "results": q.get()}
            print(json.dumps(out, indent=1))
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=1)
        return
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(init_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, args, init_file, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = q.get(timeout=600)
    for p in procs:
        p.join(60)
    out = {"backend": args.backend, "gpus": args.gpus, "hidden": args.hidden, "results": res}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    sys.exit(0)


if __name__ == "__main__":
    main()
