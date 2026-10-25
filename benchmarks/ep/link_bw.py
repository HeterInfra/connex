# SPDX-License-Identifier: Apache-2.0
"""Reference bandwidth of the GPU-to-GPU path that expert moves use.

Copies ``--mb`` MB from cuda:0 to cuda:1 (as numbered by
CUDA_VISIBLE_DEVICES) with ``Tensor.copy_`` (peer-to-peer when the driver
allows it, staged through the host otherwise) and with connex-rs one-sided
reads between two engines in this process (GPU memory registered on both
sides; UCX picks cuda_ipc / RDMA / staging). Prints GB/s for each.

    CUDA_VISIBLE_DEVICES=2,3 python benchmarks/ep/link_bw.py
"""
import argparse
import json
import time

import torch


def bw(fn, nbytes, reps):
    fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    return nbytes * reps / (time.perf_counter() - t0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=192)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--chunk-mb", type=int, default=12)
    a = ap.parse_args()
    n = a.mb << 20
    src = torch.randn(n // 2, dtype=torch.bfloat16, device="cuda:0")
    dst = torch.empty_like(src, device="cuda:1")
    out = {"p2p_access": torch.cuda.can_device_access_peer(1, 0),
           "torch_copy_GBps": bw(lambda: dst.copy_(src), n, a.reps)}
    try:
        from connex.rl.transport import ConnexRsTransport
        ta = ConnexRsTransport(lid=("bw", 0, 0, 0))
        tb = ConnexRsTransport(lid=("bw", 0, 1, 0))
        sb, db = src.view(torch.uint8), dst.view(torch.uint8)
        ta.register(sb); tb.register(db)
        ck = a.chunk_mb << 20
        ops = [(sb.data_ptr() + o, o, min(ck, n - o)) for o in range(0, n, ck)]
        out["connex_rs_read_GBps"] = bw(lambda: tb.read_batch(ta.url, ops, db), n, a.reps)
        assert torch.equal(dst.cpu(), src.cpu())
        ta.close(); tb.close()
    except ImportError:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
