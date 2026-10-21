#!/usr/bin/env bash
# Two processes on this host: survivor + a victim that a loop restarts after
# every fault (same logical id each time).
#
# usage: bench/run_local.sh OUT.jsonl [survivor args...]
# env:   UCX_TLS / UCX_* (passed to both), VICTIM_BURN=N (busy threads in the
#        victim), CPUS_SURV / CPUS_VICT (taskset lists), CXBENCH (binary),
#        BURN_HOST=N (extra burner process pinned to the victim's cores),
#        IP (address to use; default 127.0.0.1), VICTIM_ARGS (extra victim args)
set -euo pipefail
OUT=${1:?out file}; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CXBENCH:-$HERE/target/release/cxbench}
IP=${IP:-127.0.0.1}
PORT=${PORT:-$((20000 + RANDOM % 20000))}
CTL=$((PORT + 1))
CPUS_SURV=${CPUS_SURV:-100-101}
CPUS_VICT=${CPUS_VICT:-102-105}
VICTIM_BURN=${VICTIM_BURN:-0}

taskset -c "$CPUS_SURV" "$BIN" survivor --listen "$IP" --port "$PORT" --ctl-port "$CTL" --local "$@" >"$OUT" &
SPID=$!
trap 'kill $SPID 2>/dev/null || true; kill $LOOP 2>/dev/null || true; kill $BPID 2>/dev/null || true; pkill -9 -f "cxbench victim --connect $IP:$PORT" 2>/dev/null || true' EXIT
for _ in $(seq 100); do grep -q ready "$OUT" 2>/dev/null && break; sleep 0.05; done
grep -q ready "$OUT" || { echo "survivor did not start"; exit 1; }

BPID=
if [ "${BURN_HOST:-0}" -gt 0 ]; then
  taskset -c "$CPUS_VICT" "$BIN" burn --threads "$BURN_HOST" --seconds 100000 & BPID=$!
fi

(
  while kill -0 $SPID 2>/dev/null; do
    rc=0
    taskset -c "$CPUS_VICT" "$BIN" victim --connect "$IP:$PORT" --ctl "$IP:$CTL" --burn "$VICTIM_BURN" ${VICTIM_ARGS:-} 2>>"$OUT.victim.log" || rc=$?
    [ $rc -eq 0 ] && break
  done
) &
LOOP=$!
wait $SPID
