#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Markdown table from pd_bench.py result files.

    python benchmarks/pd/summarize.py connex=runs/perf-connex nixl=runs/perf-nixl
"""
import glob
import json
import os
import sys


def row(name, f):
    s = json.load(open(f))["summary"]

    def ms(k, q):
        v = s[k][q]
        return "-" if v is None else f"{v * 1000:.0f}"

    return (f"| {name} | {os.path.basename(f)[:-5]} | {s['ok']}/{s['requests']} | "
            f"{ms('ttft', 'p50')} / {ms('ttft', 'p99')} | {ms('tpot', 'p50')} / {ms('tpot', 'p99')} | "
            f"{s['e2e']['mean']:.2f} / {s['e2e']['p99']:.2f} | {s['throughput_tok_s']:.0f} |")


def main():
    print("| connector | load | ok | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | E2E mean / p99 (s) | tok/s |")
    print("|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "qps*.json")),
                       key=lambda f: float(os.path.basename(f)[3:-5])) + \
            sorted(glob.glob(os.path.join(d, "closed*.json")))
        for f in files:
            print(row(name, f))


if __name__ == "__main__":
    main()
