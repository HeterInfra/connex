# SPDX-License-Identifier: Apache-2.0
"""Median summary of elastic_bench.py JSON results grouped by config prefix."""

import glob
import json
import os
import statistics
import sys

KEYS = [
    ("throughput_msg_s", "msg/s"),
    ("lat_ms_p50", "p50 ms"),
    ("lat_ms_p99", "p99 ms"),
    ("scaleout_ready_ms", "join ready ms"),
    ("leave_sends_after_LEAVE", "sends after LEAVE"),
    ("kill_sends_to_dead", "sends to dead"),
    ("kill_sender_stall_ms", "kill stall ms"),
    ("send_ms_max", "max send ms"),
    ("lost", "lost"),
]


def main(out):
    groups = {}
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        groups.setdefault(os.path.basename(f).rsplit("_", 1)[0], []).append(json.load(open(f)))
    print("| config | runs | " + " | ".join(h for _, h in KEYS) + " |")
    print("|---" * (len(KEYS) + 2) + "|")
    for name, rs in groups.items():
        cells = []
        for k, _ in KEYS:
            vals = [r[k] for r in rs if k in r]
            cells.append(f"{statistics.median(vals):.1f}" if vals else "-")
        print(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
