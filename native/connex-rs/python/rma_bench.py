"""One-sided read benchmark and GPU check for connex_rs.Engine.

Reader and owner are separate processes (cross-host: run `owner` on one
host and `reader` on the other). The owner registers a buffer and prints
its URL and address; the reader registers its own buffer, reads with
transfer_sync_read and verifies the bytes.

  python3 rma_bench.py local [--dev cuda:1 --peer-dev cuda:2] [--sizes ...]
  python3 rma_bench.py owner --dev cuda:2 --ip <owner ip> --port 18500
  python3 rma_bench.py reader --dev cuda:3 --ip <reader ip> --url <url> --addr <addr> --len <bytes>

Output: one JSON line per size (p50/p99 latency of a blocking read, GB/s).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import connex_rs
import torch

SIZES = [16 << 10, 1 << 20, 16 << 20]


def buf(dev, n, fill=None):
    t = torch.empty(n, dtype=torch.uint8, device=dev)
    if fill is not None:
        g = torch.Generator(device="cpu").manual_seed(fill)
        t.copy_(torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g))
    return t


def owner(a):
    if a.dev.startswith("cuda"):
        torch.cuda.set_device(a.dev)
        torch.zeros(1, device=a.dev)  # create the CUDA context before the engine
    e = connex_rs.Engine("bench", 0, 0, 0, listen_ip=a.ip, port=a.port)
    n = max(a.sizes)
    src = buf(a.dev, n, fill=1234)
    torch.cuda.synchronize() if a.dev.startswith("cuda") else None
    assert e.register_memory(src.data_ptr(), n) == 0
    print(json.dumps({"url": e.local_url, "addr": src.data_ptr(), "len": n}), flush=True)
    if a.hold:
        time.sleep(a.hold)  # cross-host: keep serving for a.hold seconds
    else:
        sys.stdin.readline()  # reader done
    print(json.dumps({"owner_stats": e.stats()}), flush=True)
    e.close()


def reader(a, url, addr, n):
    if a.dev.startswith("cuda"):
        torch.cuda.set_device(a.dev)
        torch.zeros(1, device=a.dev)  # create the CUDA context before the engine
    e = connex_rs.Engine("bench", 1, 0, 0, listen_ip=a.ip, port=0)
    dst = buf(a.dev, n)
    if os.environ.get("PRECONNECT"):
        e.connect(url)
        time.sleep(0.5)
    assert e.register_memory(dst.data_ptr(), n) == 0
    want = buf("cpu", n, fill=1234)
    out = []
    for size in a.sizes:
        rc = e.transfer_sync_read(url, dst.data_ptr(), addr, size)
        assert rc == 0, f"read failed {rc}"
        assert torch.equal(dst[:size].cpu(), want[:size]), f"payload mismatch at {size}"
        lat = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            rc = e.transfer_sync_read(url, dst.data_ptr(), addr, size)
            lat.append(time.perf_counter() - t0)
            assert rc == 0
        lat.sort()
        p50 = lat[len(lat) // 2]
        r = {"type": "rma_read", "dev": a.dev, "size": size,
             "p50_us": round(p50 * 1e6, 1), "p99_us": round(lat[len(lat) * 99 // 100] * 1e6, 1),
             "gbps": round(size / p50 / 1e9, 2), "tls": os.environ.get("UCX_TLS", "")}
        print(json.dumps(r), flush=True)
        out.append(r)
    st = e.stats()
    print(json.dumps({"reader_stats": st}), flush=True)
    e.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["local", "owner", "reader"])
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--peer-dev", default="cuda:2")
    ap.add_argument("--ip", default=os.environ.get("IP", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--url")
    ap.add_argument("--addr", type=int)
    ap.add_argument("--len", type=int)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--hold", type=float, default=0, help="owner: serve this long, then exit")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=SIZES)
    a = ap.parse_args()
    if a.mode == "owner":
        return owner(a)
    if a.mode == "reader":
        return reader(a, a.url, a.addr, a.len)
    args = [sys.executable, __file__, "owner", "--dev", a.peer_dev, "--ip", a.ip,
            "--sizes", ",".join(map(str, a.sizes))]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    info = json.loads(p.stdout.readline())
    try:
        reader(a, info["url"], info["addr"], info["len"])
    finally:
        p.stdin.write("\n")
        p.stdin.flush()
        print(p.stdout.readline().strip())
        p.wait(timeout=30)


if __name__ == "__main__":
    main()
