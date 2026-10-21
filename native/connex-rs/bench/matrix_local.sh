#!/usr/bin/env bash
# Full single-host matrix. Results: results/local/<name>.jsonl
# usage: bench/matrix_local.sh [phase...]   phases: kill idle burn big stop fp lat (default: all)
# Pinned to a few cores (CPUS_SURV / CPUS_VICT) so a shared host is not disturbed.
cd "$(dirname "$0")/.." || exit 1
OUT=results/local
mkdir -p $OUT
IB_IP=${IB_IP:-$(ip -4 -o addr show ibs102 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}
REPS=${REPS:-100}
PHASES=${*:-kill idle burn big stop fp lat}
HB="--hb-ms 50 --hb-timeout-ms 300"

ipfor() { case $1 in rc*|ud*|dc*) echo "$IB_IP";; *) echo 127.0.0.1;; esac; }

run() {  # name tls [env...] -- survivor args
  local name=$1 tls=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "== $name ($tls) $(date +%T)"
  env UCX_TLS="$tls" IP="$(ipfor "$tls")" "${envs[@]}" timeout "${RUN_TIMEOUT:-1800}" \
    bench/run_local.sh "$OUT/$name.jsonl" "$@" 2>/dev/null
  grep -E '"done"|"error"' "$OUT/$name.jsonl" | tail -2
}

for ph in $PHASES; do
  case $ph in
  kill)  # SIGKILL while streaming 1 MB steps (2 in flight); 100 kill/rejoin cycles
    for t in tcp rc_mlx5 sm,tcp; do run "kill_${t/,/+}" $t -- --reps $REPS --size 1048576; done ;;
  idle)  # SIGKILL with no traffic in flight
    for t in tcp rc_mlx5; do run "idle_${t}" $t -- --reps 50 --size 65536 --idle; done ;;
  burn)  # receiver CPU saturated: 8 busy threads in the victim + 8 in a burner, on its 4 cores
    for t in tcp rc_mlx5; do
      run "burn_${t}" $t VICTIM_BURN=8 BURN_HOST=8 "VICTIM_ARGS=$HB" -- --reps 50 --size 1048576 $HB
    done ;;
  big)   # 64 MB steps in flight (depth 2) when the fault hits
    for t in tcp rc_mlx5; do run "big64m_${t}" $t -- --reps 30 --size 67108864 --warmup-ms 1500; done ;;
  stop)  # hung peer (SIGSTOP): UCX keepalive 100 ms alone, then + app heartbeat 50/300 ms
    for t in tcp rc_mlx5; do
      run "stop_ka_${t}" $t UCX_KEEPALIVE_INTERVAL=100ms -- --reps 5 --fault stop --local --detect-timeout-ms 10000
      run "stop_hb_${t}" $t "VICTIM_ARGS=$HB" -- --reps 30 --fault stop --local $HB
    done ;;
  fp)    # false positives: 5 min streaming, no faults, heartbeat 50/300 + keepalive 100 ms,
         # receiver CPU-saturated; 1 MB and 64 MB steps
    for t in tcp rc_mlx5; do
      run "fp_burn_1m_${t}" $t VICTIM_BURN=8 BURN_HOST=8 UCX_KEEPALIVE_INTERVAL=100ms "VICTIM_ARGS=$HB" -- --reps 0 --duration-s ${FP_SECS:-300} --size 1048576 $HB
      run "fp_burn_64m_${t}" $t VICTIM_BURN=8 BURN_HOST=8 UCX_KEEPALIVE_INTERVAL=100ms "VICTIM_ARGS=$HB" -- --reps 0 --duration-s ${FP_SECS:-300} --size 67108864 $HB
    done ;;
  lat)   # steady state (echo, half round trip), host memory
    for t in tcp rc_mlx5 sm,tcp; do
      run "lat_${t/,/+}" $t -- --lat-sizes 16384,1048576,16777216 --lat-iters 2000
    done ;;
  esac
done
