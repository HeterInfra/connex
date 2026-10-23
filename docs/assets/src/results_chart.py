#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate docs/assets/results-connexrs-vs-mooncake.svg from the numbers in
docs/results.md, section 6 (Llama-3-8B, 4 pipeline stages on A40 GPUs,
100 Gb InfiniBand, GPUDirect; only CONNEX_PP_BACKEND differs).

    python docs/assets/src/results_chart.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results-connexrs-vs-mooncake.svg")
RS, MC, INK, GRID = "#2563eb", "#ea580c", "#6b7280", "#9ca3af"

# kill -> first resumed token, median per run, range over runs (seconds)
HANDOVER = [  # label, connex-rs (lo, hi), Mooncake (lo, hi)
    ("K=1, stages 1-3", (0.02, 0.03), (2.0, 3.0)),
    ("K=4, stage 1", (0.01, 0.02), (2.0, 3.0)),
    ("K=4, stage 2", (0.03, 0.04), (2.0, 3.0)),
    ("K=4, stage 3", (0.03, 0.83), (2.0, 3.0)),
]
# K=4 steady state, mean E2E latency (s)
QPS = [2, 4, 8]
STEADY_RS = [1.61, 2.01, 2.61]
STEADY_MC = [1.60, 2.04, 2.59]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": GRID, "svg.fonttype": "none"})
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
fig.patch.set_alpha(0)

for i, (lab, rs, mc) in enumerate(HANDOVER):
    y = len(HANDOVER) - 1 - i
    for (lo, hi), c, dy in ((rs, RS, 0.15), (mc, MC, -0.15)):
        a.plot([lo, hi], [y + dy, y + dy], color=c, lw=6, solid_capstyle="round")
a.set_yticks(range(len(HANDOVER)), [h[0] for h in reversed(HANDOVER)])
a.set_xscale("log")
a.set_xlim(0.005, 6)
a.set_xticks([0.01, 0.1, 1, 3], ["0.01 s", "0.1 s", "1 s", "3 s"])
a.set_title("Handover: kill a stage -> next token", color=INK, fontsize=11, loc="left")
a.text(0.035, -0.75, "connex-rs", color=RS, fontweight="bold")
a.text(1.1, -0.75, "Mooncake TE", color=MC, fontweight="bold")
a.set_ylim(-1.0, len(HANDOVER) - 0.5)

w = 0.36
x = range(len(QPS))
b.bar([i - w / 2 for i in x], STEADY_RS, w, color=RS, label="connex-rs")
b.bar([i + w / 2 for i in x], STEADY_MC, w, color=MC, label="Mooncake TE")
b.set_xticks(list(x), [f"QPS {q}" for q in QPS])
b.set_ylabel("mean E2E latency (s)")
b.set_title("Steady state, K=4", color=INK, fontsize=11, loc="left")
b.legend(frameon=False, loc="upper left", labelcolor=INK)
b.set_ylim(0, 3.4)

for ax in (a, b):
    ax.grid(axis="x" if ax is a else "y", color=GRID, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    ax.patch.set_alpha(0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "Llama-3-8B, 4 pipeline stages on A40, 100 Gb InfiniBand, GPUDirect.\nHandover: 2 replicas, range of "
         "per-run medians (Mooncake: 2-3 s, section 2). Steady state: 1 replica. Source: docs/results.md, section 6", fontsize=8, color=INK)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT, format="svg", metadata={"Date": None, "Creator": None})
# Text stays text (svg.fonttype none); render it with the viewer's system font.
STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, 'DejaVu Sans', sans-serif"
with open(OUT) as f:
    svg = f.read().replace("'DejaVu Sans'", STACK)
with open(OUT, "w") as f:
    f.write(svg)
print(OUT)
