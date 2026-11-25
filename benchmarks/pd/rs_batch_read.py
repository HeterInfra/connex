# SPDX-License-Identifier: Apache-2.0
"""Fragmented KV-style read through ConnexRsPlane: per-op get_async vs the
connex-rs batch API (transfer_batch_sync, connex-rs >= 0.1.7).

usage: UCX_TLS=tcp python benchmarks/pd/rs_batch_read.py [n_ops] [op_bytes]
Prints the median wall time of one plane.read() of n_ops reads.
"""
import os
import statistics
import sys
import time

import connex_rs

import connex.pd.dataplane as dp


def main():
    n_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    a = dp.ConnexRsPlane("rsb", 0, 0, 0, "127.0.0.1")
    b = connex_rs.Engine("rsb", 1, listen_ip="127.0.0.1")
    size = 2 * n_ops * nb  # every other slot: no two ops adjacent (fragmented)
    src = b.allocate_managed_buffer(size)
    dst = a.e.allocate_managed_buffer(size)
    b.write_bytes_to_buffer(src, os.urandom(size), size)
    ops = [(dst + 2 * i * nb, src + 2 * i * nb, nb) for i in range(n_ops)]
    print(f"connex_rs {getattr(connex_rs, '__version__', '?')}, batch API "
          f"{hasattr(a.e, 'transfer_batch_sync')}, {n_ops} ops x {nb} B")
    for mode in ("per_op", "batch"):
        dp._RS_BATCH = mode == "batch"
        assert a.read(b.local_url, ops) == 0  # warm-up (connect)
        ts, cpu = [], []
        for _ in range(5):
            t0, c0 = time.perf_counter(), time.thread_time()
            assert a.read(b.local_url, ops) == 0
            ts.append(time.perf_counter() - t0)
            cpu.append(time.thread_time() - c0)
        # caller CPU = Python time spent posting/waiting while holding the GIL
        print(f"{mode:7s} wall {statistics.median(ts) * 1e3:8.1f} ms, caller CPU "
              f"{statistics.median(cpu) * 1e3:8.1f} ms per read() of {n_ops} ops")
    assert a.e.read_bytes_from_buffer(dst, nb) == b.read_bytes_from_buffer(src, nb)
    a.close()
    b.close()


if __name__ == "__main__":
    main()
