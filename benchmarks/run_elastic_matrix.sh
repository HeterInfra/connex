#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Compare baseline vs optimized Connex under scale-out / graceful leave / kill.
# usage: benchmarks/run_elastic_matrix.sh OUTDIR [RUNS]
set -u
OUT=${1:?outdir}; RUNS=${2:-5}
mkdir -p "$OUT"
declare -A CFG=(
  [baseline]="--membership poll|UCX_TLS=all UCX_RNDV_THRESH=auto"
  [transport_opt]="--membership poll|"
  [full_opt]="--membership elastic|"
)
for name in baseline transport_opt full_opt; do
  args=${CFG[$name]%%|*}; envs=${CFG[$name]#*|}
  for i in $(seq "$RUNS"); do
    env $envs timeout -s KILL 90 python "$(dirname "$0")/elastic_bench.py" --duration 12 \
      $args --out "$OUT/${name}_$i.json" > "$OUT/${name}_$i.log" 2>&1 \
      || echo "run failed: $name $i (rc=$?)"
  done
done
python "$(dirname "$0")/summarize.py" "$OUT"
