#!/usr/bin/env bash
# Heartbeat timeout sweep for false positives: receiver CPU-saturated and
# 64 MB steps (the receiver's single thread verifies each step for
# hundreds of ms without calling progress). Results: results/local/fpsweep_*.jsonl
cd "$(dirname "$0")/.." || exit 1
IB_IP=${IB_IP:-$(ip -4 -o addr show ibs102 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}
for to in ${TIMEOUTS:-1000 3000}; do
  for t in tcp rc_mlx5; do
    ip=127.0.0.1; [ $t = rc_mlx5 ] && ip=$IB_IP
    hb="--hb-ms 100 --hb-timeout-ms $to"
    echo "== fpsweep $t $to $(date +%T)"
    UCX_TLS=$t IP=$ip VICTIM_BURN=8 BURN_HOST=8 VICTIM_ARGS="$hb" timeout 900 \
      bench/run_local.sh "results/local/fpsweep_${to}ms_64m_$t.jsonl" --reps 0 --duration-s ${FP_SECS:-120} --size 67108864 $hb 2>/dev/null
    grep '"done"' "results/local/fpsweep_${to}ms_64m_$t.jsonl"
  done
done
