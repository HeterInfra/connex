# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Merge CONNEX_PP_TRACE files of one host into per-step timelines.

usage: python trace_steps.py /dev/shm/cxtrace-*.txt [--min-gap-ms 0]

Prints, for every event tag, the median offset (ms) from the step's first
event ("h_exec_enc", stage 0 encoding EXEC), over all steps that have it,
plus the difference to the previous event in that order. Only steps that
reached "h_wake!" are counted; decode steps by default, --prefill for the
steps that schedule a prefill.
"""
import argparse
import collections
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--prefill", action="store_true",
                    help="only steps that schedule a prefill (else only decode steps)")
    a = ap.parse_args()
    steps = collections.defaultdict(dict)
    for f in a.files:
        for line in open(f):
            p = line.split()
            if len(p) < 3:
                continue
            tag, seq, t = p[0], int(p[1]), int(p[2])
            steps[seq].setdefault(tag, t)  # first occurrence per step (TP ranks)
    offs = collections.defaultdict(list)
    for seq, ev in steps.items():
        if "h_exec_enc" not in ev or "h_wake!" not in ev:
            continue
        if ("h_prefill" in ev) != a.prefill:
            continue
        t0 = ev["h_exec_enc"]
        for tag, t in ev.items():
            offs[tag].append((t - t0) / 1e6)
    order = sorted(offs, key=lambda k: statistics.median(offs[k]))
    prev = None
    print(f"{'event':22s} {'n':>5s} {'median ms':>10s} {'p90 ms':>8s} {'step':>8s}")
    for tag in order:
        v = sorted(offs[tag])
        med = statistics.median(v)
        p90 = v[int(0.9 * (len(v) - 1))]
        d = med - prev if prev is not None else 0.0
        print(f"{tag:22s} {len(v):5d} {med:10.3f} {p90:8.3f} {d:8.3f}")
        prev = med


if __name__ == "__main__":
    main()
