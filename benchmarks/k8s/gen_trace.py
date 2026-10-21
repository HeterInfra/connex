# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic request trace for trace_load.py.

Arrivals follow a gamma process with mean rate --qps and coefficient of
variation --cv (cv=1 is Poisson). Prompt/output lengths are drawn uniformly
from the given ranges. Output columns: TIMESTAMP,ContextTokens,GeneratedTokens.

    python gen_trace.py --qps 4 --cv 1.0 --duration 300 --out traces/gamma_qps4.0.csv
"""
import argparse
import csv
import os
import random
from datetime import datetime, timedelta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--cv", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=300, help="seconds")
    ap.add_argument("--prompt", default="128-1024", help="min-max context tokens")
    ap.add_argument("--output", default="32-256", help="min-max generated tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pl, ph = map(int, a.prompt.split("-"))
    ol, oh = map(int, a.output.split("-"))
    shape = 1.0 / (a.cv ** 2)
    scale = 1.0 / (a.qps * shape)
    t0 = datetime(2026, 1, 1)
    t, rows = 0.0, []
    while t < a.duration:
        rows.append((t0 + timedelta(seconds=int(t)), rng.randint(pl, ph), rng.randint(ol, oh)))
        t += rng.gammavariate(shape, scale)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        for ts, c, g in rows:
            w.writerow([ts.isoformat(sep=" "), c, g])
    print(f"wrote {len(rows)} requests to {a.out}")


if __name__ == "__main__":
    main()
