# SPDX-License-Identifier: Apache-2.0
"""Weight-sync time: Connex (one-sided pulls, tree fan-out) vs NCCL broadcast.

One trainer process and N receiver processes, one GPU each. The model is
synthetic (bf16 tensors of ``--tensor-mb`` each) sized like Qwen2.5-0.5B /
1.5B or Llama-3-8B; the transfer path does not look at values. Both modes
start with the weights already on the trainer GPU and stop when every
receiver holds the full model on its GPU.

* ``connex``: trainer packs into a registered arena and publishes; receivers
  pull concurrently (registry assigns sources, fan-out ``--fanout``).
* ``nccl``: one process group (trainer rank 0); the trainer broadcasts the
  packed arena in ``--chunk-mb`` buckets (what vLLM's packed NCCL path does).

``--kill-at`` (connex only): receiver 1 is killed this many seconds into
the pull, to check survivors finish. Distributed runs: start ``registry``
once, then ``trainer`` and ``receiver`` processes on any hosts.

  python weight_sync_bench.py local --mode connex --size 1.5B --gpus 1,2,3
  python weight_sync_bench.py local --mode nccl --size 1.5B --gpus 1,2,3
"""
import argparse
import json
import os
import socket
import sys
import time

import torch
import torch.multiprocessing as mp

SIZES = {"0.5B": 494_032_768, "1.5B": 1_543_714_304, "8B": 8_030_261_248}  # params


def tensors(size, tensor_mb, device):
    n = SIZES[size] * 2                    # bf16 bytes
    per = tensor_mb << 20
    out, i = [], 0
    while n > 0:
        k = min(per, n)
        out.append((f"t{i}", torch.full((k // 2,), float(i % 7), dtype=torch.bfloat16,
                                        device=device)))
        n -= k
        i += 1
    return out


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# -- connex -------------------------------------------------------------------
def connex_trainer(a, reg_url, q, rounds):
    from connex.rl import RegistryClient, WeightPublisher, make_transport
    torch.cuda.set_device(a.trainer_dev)
    torch.zeros(1, device="cuda")
    reg = RegistryClient(reg_url)
    pub = WeightPublisher(reg, make_transport("connex-rs", lid=("bench", 0, 0, 0)), device="cuda")
    named = tensors(a.size, a.tensor_mb, "cuda")
    torch.cuda.synchronize()
    for r in range(rounds):
        q.get()                             # receivers ready
        t0 = time.time()
        v = pub.publish(named)
        t1 = time.time()
        print(json.dumps({"role": "trainer", "round": r, "version": v, "t_start": t0,
                          "pack_s": t1 - t0}), flush=True)
        q.get()                             # round finished
    q.get()


def connex_receiver(a, reg_url, idx, dev, q_ready, q_res, rounds):
    from connex.rl import RegistryClient, WeightSubscriber, make_transport
    torch.cuda.set_device(dev)
    torch.zeros(1, device="cuda")
    reg = RegistryClient(reg_url)
    sub = WeightSubscriber(reg, make_transport("connex-rs", lid=("bench", 1, 0, idx)), f"rx{idx}",
                           device="cuda", chunk_bytes=a.chunk_mb << 20, window=a.window)
    last = 0
    for r in range(rounds):
        q_ready.put(idx)
        reg.get_latest(wait_newer_than=last, timeout=600)
        st = sub.sync(lambda it: None)
        last = st["version"]
        q_res.put({"idx": idx, "round": r, "t_end": time.time(), **st})


def run_connex(a, gpus):
    from connex.rl import RegistryServer, WeightRegistry
    ctx = mp.get_context("spawn")
    srv = RegistryServer(WeightRegistry(fanout=a.fanout))
    url = f"http://127.0.0.1:{srv.port}"
    q_t, q_ready, q_res = ctx.Queue(), ctx.Queue(), ctx.Queue()
    rx = [ctx.Process(target=connex_receiver, args=(a, url, i, g, q_ready, q_res, a.rounds))
          for i, g in enumerate(gpus[1:])]
    a.trainer_dev = gpus[0]
    tr = ctx.Process(target=connex_trainer, args=(a, url, q_t, a.rounds))
    tr.start()
    for p in rx:
        p.start()
    n = len(rx)
    results = []
    for r in range(a.rounds):
        for _ in range(n):
            q_ready.get(timeout=600)
        t0 = time.time()
        q_t.put("go")
        got, killed = [], None
        while len(got) < n - (1 if killed is not None else 0):
            if a.kill_at and killed is None and r == a.rounds - 1:
                try:
                    got.append(q_res.get(timeout=max(0.0, t0 + a.kill_at - time.time())))
                    continue
                except Exception:
                    rx[1].kill()
                    killed = 1
                    continue
            got.append(q_res.get(timeout=600))
        t_end = max(g["t_end"] for g in got)
        res = {"mode": "connex", "size": a.size, "receivers": n, "round": r,
               "total_s": t_end - t0, "per_receiver_s": sorted(g["pull_s"] for g in got),
               "sources": {g["idx"]: g["sources"] for g in got},
               "source_failures": sum(g["source_failures"] for g in got),
               "killed": killed, "gbytes": got[0]["bytes"] / 1e9}
        results.append(res)
        print(json.dumps(res), flush=True)
        q_t.put("done")
    q_t.put("exit")
    tr.join(60)
    for p in rx:
        p.join(30)
        if p.is_alive():
            p.kill()
    srv.close()
    return results


# -- nccl -------------------------------------------------------------------
def nccl_worker(a, rank, world, dev, port, q):
    import torch.distributed as dist
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=torch.device("cuda", torch.cuda.current_device()))
    named = tensors(a.size, a.tensor_mb, "cuda") if rank == 0 else None
    total = sum(t.numel() * 2 for _, t in tensors(a.size, a.tensor_mb, "meta"))
    chunk = a.chunk_mb << 20
    arena = torch.empty(total, dtype=torch.uint8, device="cuda")
    dist.barrier()
    for r in range(a.rounds):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        if rank == 0:     # pack, like the publisher does
            off = 0
            for _, t in named:
                n = t.numel() * 2
                arena[off:off + n].copy_(t.view(-1).view(torch.uint8))
                off += n
        for off in range(0, total, chunk):
            dist.broadcast(arena[off:off + chunk], src=0)
        torch.cuda.synchronize()
        t1 = time.time()
        q.put({"rank": rank, "round": r, "t0": t0, "t1": t1})
    dist.destroy_process_group()


def run_nccl(a, gpus):
    ctx = mp.get_context("spawn")
    port, q = free_port(), ctx.Queue()
    ps = [ctx.Process(target=nccl_worker, args=(a, i, len(gpus), g, port, q))
          for i, g in enumerate(gpus)]
    for p in ps:
        p.start()
    out = [q.get(timeout=900) for _ in range(len(gpus) * a.rounds)]
    for p in ps:
        p.join(60)
    results = []
    for r in range(a.rounds):
        rs = [o for o in out if o["round"] == r]
        t0 = min(o["t0"] for o in rs)
        res = {"mode": "nccl", "size": a.size, "receivers": len(gpus) - 1, "round": r,
               "total_s": max(o["t1"] for o in rs if o["rank"] > 0) - t0}
        results.append(res)
        print(json.dumps(res), flush=True)
    return results


# -- distributed (one process per pod/host; rank 0 = trainer) ---------------
def run_dist(a):
    """Both modes on the same processes: NCCL rounds, then Connex rounds.
    Timing is taken on the trainer: round start to the barrier after every
    receiver has the full model (the barrier adds well under a ms)."""
    import torch.distributed as dist
    from connex.rl import (RegistryClient, RegistryServer, WeightPublisher, WeightRegistry,
                           WeightSubscriber, make_transport)
    rank, world = a.rank, a.world
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    dist.init_process_group("nccl", init_method=f"tcp://{a.master}:{a.port}", rank=rank,
                            world_size=world, device_id=torch.device("cuda", 0))
    named = tensors(a.size, a.tensor_mb, "cuda") if rank == 0 else None
    total = sum(t.numel() * 2 for _, t in tensors(a.size, a.tensor_mb, "meta"))
    chunk = a.chunk_mb << 20
    out = []
    # NCCL broadcast of the packed arena
    arena = torch.empty(total, dtype=torch.uint8, device="cuda")
    for r in range(a.rounds):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        if rank == 0:
            off = 0
            for _, t in named:
                n = t.numel() * 2
                arena[off:off + n].copy_(t.view(-1).view(torch.uint8))
                off += n
        for off in range(0, total, chunk):
            dist.broadcast(arena[off:off + chunk], src=0)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            out.append({"mode": "nccl", "size": a.size, "receivers": world - 1, "round": r,
                        "total_s": time.time() - t0})
            print(json.dumps(out[-1]), flush=True)
    del arena
    torch.cuda.empty_cache()
    # Connex: registry in the trainer process
    if rank == 0:
        srv = RegistryServer(WeightRegistry(fanout=a.fanout), port=a.port + 1)
    dist.barrier()
    reg = RegistryClient(f"http://{a.master}:{a.port + 1}")
    if rank == 0:
        pub = WeightPublisher(reg, make_transport("connex-rs", lid=("bench", 0, 0, 0)),
                              device="cuda", buffers=a.pub_buffers)
    else:
        sub = WeightSubscriber(reg, make_transport("connex-rs", lid=("bench", 1, 0, rank)),
                               f"rx{rank}", device="cuda", chunk_bytes=chunk, window=a.window)
    for r in range(a.rounds):
        dist.barrier()
        t0 = time.time()
        if rank == 0:
            v = pub.publish(named)
            pack = time.time() - t0
        else:
            reg.get_latest(wait_newer_than=r, timeout=600)
            st = sub.sync(lambda it: None)
        stats = [None] * world
        dist.all_gather_object(stats, None if rank == 0 else st)
        if rank == 0:
            rx = [s for s in stats[1:]]
            out.append({"mode": "connex", "size": a.size, "receivers": world - 1, "round": r,
                        "version": v, "total_s": time.time() - t0, "pack_s": pack,
                        "per_receiver_s": sorted(round(s["pull_s"], 4) for s in rx),
                        "sources": [s["sources"][0] for s in rx], "fanout": a.fanout})
            print(json.dumps(out[-1]), flush=True)
    dist.barrier()
    if rank == 0 and a.out:
        with open(a.out, "a") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        srv.close()


# -- churn: no collective group; a receiver dies in the middle of a round -----
def run_churn(a):
    """Rank 0 = trainer (+ registry); ranks 1.. = receivers that start
    ``--stagger`` s apart (so later ones read from earlier ones: a tree).
    In round ``--kill-round``, rank ``--kill-rank`` exits after ``--kill-after-chunks``
    chunk batches -- typically while another receiver is reading from it.
    The trainer reports when every live receiver holds the version."""
    from connex.rl import (RegistryClient, RegistryServer, WeightPublisher, WeightRegistry,
                           WeightSubscriber, make_transport)
    rank = a.rank
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    url = f"http://{a.master}:{a.port + 1}"
    if rank == 0:
        srv = RegistryServer(WeightRegistry(fanout=a.fanout, member_ttl_s=3.0), port=a.port + 1)
        reg = srv.reg
        pub = WeightPublisher(reg, make_transport("connex-rs", lid=("churn", 0, 0, 0)),
                              device="cuda", buffers=a.pub_buffers)
        named = tensors(a.size, a.tensor_mb, "cuda")
        while len(reg.status()["members"]) < a.world - 1:
            time.sleep(0.2)
        for r in range(a.rounds):
            t0 = time.time()
            v = pub.publish(named)
            while True:
                st = reg.status()
                live = st["members"]
                if live and all(x == v for x in live.values()):
                    break
                time.sleep(0.01)
            rec = {"mode": "connex-churn", "size": a.size, "round": r, "version": v,
                   "total_s": time.time() - t0, "live": sorted(live),
                   "events": [e[1:] for e in reg.events if e[0] >= t0 and e[1] in
                              ("source_failed", "expired", "leave")]}
            print(json.dumps(rec), flush=True)
            time.sleep(4.0)            # let the registry expire a dead member
        srv.close()
        return
    while True:
        try:
            reg = RegistryClient(url, timeout=600)
            reg.status()
            break
        except Exception:
            time.sleep(0.5)
    t = make_transport("connex-rs", lid=("churn", 1, 0, rank))
    sub = WeightSubscriber(reg, t, f"rx{rank}", device="cuda", chunk_bytes=a.chunk_mb << 20,
                           window=a.window, heartbeat_s=0.5)
    last = 0
    for r in range(a.rounds):
        reg.get_latest(wait_newer_than=last, timeout=600)
        time.sleep(a.stagger * (rank - 1))
        if r == a.kill_round and rank == a.kill_rank and a.kill_mode == "puller":
            orig, n = t.read_batch, [0]

            def dying(*args, **kw):
                n[0] += 1
                if n[0] > a.kill_after_chunks:
                    print(json.dumps({"rank": rank, "event": "exit", "t": time.time()}),
                          flush=True)
                    os._exit(1)
                return orig(*args, **kw)
            t.read_batch = dying
        st = sub.sync(lambda it: None)
        last = st["version"]
        print(json.dumps({"rank": rank, "round": r, **{k: st[k] for k in (
            "version", "pull_s", "sources", "source_failures")}}), flush=True)
        if r == a.kill_round and rank == a.kill_rank and a.kill_mode == "source":
            # die while later receivers read this copy
            time.sleep(a.kill_delay)
            print(json.dumps({"rank": rank, "event": "exit", "t": time.time()}), flush=True)
            os._exit(1)
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["local", "dist", "churn"])
    ap.add_argument("--stagger", type=float, default=0.0)
    ap.add_argument("--kill-round", type=int, default=-1)
    ap.add_argument("--kill-rank", type=int, default=-1)
    ap.add_argument("--kill-after-chunks", type=int, default=2)
    ap.add_argument("--kill-mode", choices=["puller", "source"], default="source")
    ap.add_argument("--kill-delay", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--mode", choices=["connex", "nccl"], default="connex")
    ap.add_argument("--size", choices=list(SIZES), default="0.5B")
    ap.add_argument("--gpus", default="1,2,3", help="first = trainer")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tensor-mb", type=int, default=64)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--fanout", type=int, default=2)
    ap.add_argument("--pub-buffers", type=int, default=2)
    ap.add_argument("--kill-at", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "dist":
        return run_dist(a)
    if a.cmd == "churn":
        return run_churn(a)
    gpus = [int(g) for g in a.gpus.split(",")]
    res = (run_connex if a.mode == "connex" else run_nccl)(a, gpus)
    if a.out:
        with open(a.out, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    main()
