#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-call cost of PD host staging copies: one copy per byte range (old)
vs one gather + scatter through the mapped mirror (new).

    CUDA_VISIBLE_DEVICES=0 python benchmarks/pd/staging_bench.py

For 32 layers x K/V and {64 fragmented, 64 contiguous, 512 fragmented}
blocks it prints, per direction and path:

* ``call``: time the calling thread spends in stage_out / the copy enqueue
  (what the engine thread pays per finished prefill);
* ``done``: time until the copy has finished on the GPU.

``--plan-only`` runs without a GPU and times only the row plan.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np


def _layouts(starts, nblocks, seg):
    from connex.pd.kv_layout import TensorLayout
    return [TensorLayout(base=int(s), outer=2, outer_stride=nblocks * seg, block_stride=seg,
                         seg_bytes=seg, num_blocks=nblocks) for s in starts]


def _cases(nblocks):
    rnd = random.Random(0)
    return {"64 fragmented": sorted(rnd.sample(range(1, nblocks), 64)),
            "64 contiguous": list(range(100, 164)),
            "512 fragmented": sorted(rnd.sample(range(1, nblocks), 512))}


def plan_only(args):
    from connex.pd import host_staging as hs
    seg, nb = args.seg_bytes, args.num_blocks
    span = 2 * nb * seg
    starts = np.arange(args.layers, dtype=np.int64) * (span + (2 << 20)) + (1 << 40)
    lens = np.full(args.layers, span, dtype=np.int64)
    mptrs = np.arange(args.layers, dtype=np.int64) * (span + (4 << 20)) + (1 << 36)
    for name, blocks in _cases(nb).items():
        ranges = hs.block_ranges(_layouts(starts, nb, seg), blocks)
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            hs.plan_rows(ranges, starts, lens, starts, mptrs, int(starts.min()), int(mptrs.min()))
            ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            hs.block_ranges(_layouts(starts, nb, seg), blocks)
        br = (time.perf_counter() - t0) / args.iters
        print(f"{name:>15}: {len(ranges):6d} ranges  plan {statistics.median(ts) * 1e6:8.1f} us"
              f"  block_ranges {br * 1e6:8.1f} us")


def gpu(args):
    import torch
    from connex.pd import host_staging as hs

    seg, nb = args.seg_bytes, args.num_blocks
    kvs = [torch.randint(0, 256, (2, nb, seg), dtype=torch.uint8, device="cuda")
           for _ in range(args.layers)]
    st = hs.HostStaging([(t.data_ptr(), t.numel()) for t in kvs])
    if st._vec is None:
        raise SystemExit("vectorised staging unavailable on this host (see log)")
    los = _layouts([t.data_ptr() for t in kvs], nb, seg)
    print(f"{args.layers} layers x K/V, {seg} B segments, {nb} blocks/layer, "
          f"{sum(t.numel() for t in kvs) / 2**30:.1f} GiB KV")

    def run(path, ranges, to_host):
        copy = st._copy if path == "new" else st._copy_ranges
        calls, dones = [], []
        for i in range(args.warmup + args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if to_host:                       # engine thread, engine stream
                copy(ranges, True)
                ev = torch.cuda.Event()
                ev.record()
            else:                             # pull thread, side stream
                with torch.cuda.stream(st._stream):
                    st._stream.wait_stream(st._engine_stream)
                    copy(ranges, False)
                    ev = torch.cuda.Event()
                    ev.record()
            t1 = time.perf_counter()
            ev.synchronize()
            t2 = time.perf_counter()
            if i >= args.warmup:
                calls.append(t1 - t0)
                dones.append(t2 - t0)
        return statistics.median(calls), statistics.median(dones)

    for name, blocks in _cases(nb).items():
        ranges = hs.block_ranges(los, blocks)
        mb = sum(n for _, n in ranges) / 2**20
        for to_host in (True, False):
            res = {p: run(p, ranges, to_host) for p in ("old", "new")}
            # byte check of the new path against the device
            if to_host:
                mirror = {start: m for start, _, _, m in st._spans}   # spans are sorted by address
                for t in kvs:
                    m = mirror[t.data_ptr()]
                    assert torch.equal(m.view(t.shape)[:, blocks], t[:, blocks].cpu())
            d = "out" if to_host else "in "
            (oc, od), (nc, nd) = res["old"], res["new"]
            print(f"{name:>15} stage_{d} {mb:7.1f} MB  old call {oc * 1e3:8.3f} ms done "
                  f"{od * 1e3:8.3f} ms | new call {nc * 1e3:8.3f} ms done {nd * 1e3:8.3f} ms "
                  f"({st._vec.launches} launches)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=2048)
    ap.add_argument("--seg-bytes", type=int, default=32 << 10,
                    help="bytes of one block in one of K/V (16 tokens x 8 heads x 128 x 2 B)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.plan_only:
        plan_only(args)
    else:
        gpu(args)


if __name__ == "__main__":
    main()
