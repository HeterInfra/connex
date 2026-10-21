#!/usr/bin/env bash
# Quick 5-cycle kill/rejoin smoke test over each local transport.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
IB_IP=${IB_IP:-$(ip -4 -o addr show ibs102 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}
for t in ${TLS_LIST:-tcp rc_mlx5 sm,tcp}; do
  echo "== $t"
  ip=127.0.0.1; case $t in rc*|ud*|dc*) true;; *) false;; esac && ip=$IB_IP
  UCX_TLS=$t IP=$ip timeout ${SMOKE_TIMEOUT:-120} bench/run_local.sh "results/smoke${TAG:-}_$t.jsonl" --reps ${REPS:-5} --size ${SIZE:-65536} "$@" 2>/dev/null
  cut -c1-300 "results/smoke${TAG:-}_$t.jsonl"
  tail -3 "results/smoke_$t.jsonl.victim.log" 2>/dev/null
done
