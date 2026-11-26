#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Greedy correctness gate for pipeline parallelism with expert parallelism inside each stage, on one host
# with 4 GPUs.  usage: run_gate.sh MODEL_DIR TAG [N_PROMPTS] [ARMS]
# ARMS (space separated, default "ref1 ref2 pp2 pp1"), each started fresh and stopped after its run:
#   ref*  PP=1, TP=2 + EP on GPU 0,1 (the reference; two of them give the null)
#   pp2   Connex, one instance per stage: stage 0 on GPU 0,1, stage 1 on GPU 2,3 (each TP=2 + EP)
#   pp1   Connex, one instance with -pp 2 on GPU 0-3
#   ppn   stock vLLM PP (NCCL), -pp 2 on GPU 0-3
#   score PP=1 reference that re-scores every pp*.json and ref2.json by teacher forcing (tf_score.py)
# Env: EXTRA (e.g. --enforce-eager, see README), EP (empty: no expert parallelism), BLOCKS, GMU, OUT_ROOT.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M=$1; TAG=$2; N=${3:-20}; ARMS=${4:-"ref1 ref2 pp2 pp1"}
OUT=${OUT_ROOT:-$PWD/out}/$TAG; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
BLOCKS=${BLOCKS:-2048}
IP=$(hostname -i | awk '{print $1}')
COMMON="--served-model-name m --max-model-len 4096 --num-gpu-blocks-override $BLOCKS --gpu-memory-utilization ${GMU:-0.85} \
 -tp 2 ${EP---enable-expert-parallel} --seed 0 ${EXTRA:-}"
PP="--pp-group-cls connex.adapters.vllm.pp.group.ConnexPPGroup --worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker"
PIDS=()
export CONNEX_PP_SECRET=${CONNEX_PP_SECRET:-ppep-gate-$(date +%s)}

ready() {  # port timeout_s pid...
  local port=$1 t=$2; shift 2
  for ((i = 0; i < t; i += 5)); do
    curl -sf http://127.0.0.1:$port/v1/models >/dev/null && return 0
    for p in "$@"; do kill -0 $p 2>/dev/null || { echo "process $p died"; return 1; }; done
    sleep 5
  done
  echo "timeout on $port"; return 1
}
stop() {  # kill exactly the process groups we started
  for p in "${PIDS[@]}"; do kill -TERM -- -$p 2>/dev/null; done
  for p in "${PIDS[@]}"; do for ((i = 0; i < 60; i++)); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -KILL -- -$p 2>/dev/null; done
  PIDS=(); sleep 5
}
arm() {
  local a=$1 port=8000 t0=$(date +%s)
  case $a in
    ref*) CUDA_VISIBLE_DEVICES=0,1 setsid vllm serve $M --port $port $COMMON > $OUT/$a.log 2>&1 & PIDS+=($!) ;;
    pp2)
      CUDA_VISIBLE_DEVICES=2,3 CONNEX_PP_DATA_PORT=48001 setsid python3 -m connex.adapters.vllm.pp.stage --model $M $COMMON $PP \
        --pipeline-parallel-size 2 --distributed-executor-backend connex.adapters.vllm.pp.executor.ConnexPPExecutor \
        --pipeline-stage-rank 1 --connex-control-port 48000 --connex-replica 0 > $OUT/$a.s1.log 2>&1 & PIDS+=($!)
      CUDA_VISIBLE_DEVICES=0,1 CONNEX_PP_STAGES="[[\"$IP:48000\"]]" setsid vllm serve $M --port $port $COMMON $PP \
        --pipeline-parallel-size 2 --distributed-executor-backend connex.adapters.vllm.pp.executor.ConnexPPExecutor \
        --pipeline-stage-rank 0 > $OUT/$a.log 2>&1 & PIDS+=($!) ;;
    ppn) CUDA_VISIBLE_DEVICES=0,1,2,3 setsid vllm serve $M --port $port $COMMON --pipeline-parallel-size 2 > $OUT/$a.log 2>&1 & PIDS+=($!) ;;
    score) CUDA_VISIBLE_DEVICES=0,1 setsid vllm serve $M --port $port $COMMON > $OUT/$a.log 2>&1 & PIDS+=($!) ;;
    pp1) CUDA_VISIBLE_DEVICES=0,1,2,3 setsid vllm serve $M --port $port $COMMON $PP --pipeline-parallel-size 2 > $OUT/$a.log 2>&1 & PIDS+=($!) ;;
  esac
  if ready $port ${READY_S:-1500} "${PIDS[@]}"; then
    echo "$a ready after $(( $(date +%s) - t0 )) s"
    if [ $a = score ]; then
      for c in $OUT/pp*.json $OUT/ref2.json; do [ -f $c ] && { echo -n "tf $(basename $c): "; (python3 $HERE/tf_score.py http://127.0.0.1:$port $c); }; done
    else
      python3 $HERE/gate.py run http://127.0.0.1:$port m $OUT/$a.json $N || echo "$a: client failed"
    fi
  else
    echo "$a: FAILED to start"; for f in $OUT/$a*.log; do grep -m3 -n "Error:" $f; done
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > $OUT/$a.mem
  stop
}
for a in $ARMS; do arm $a; done
cd $OUT
[ -f ref1.json ] && [ -f ref2.json ] && { echo -n "null ref1 vs ref2: "; python3 $HERE/gate.py cmp ref1.json ref2.json; }
for a in pp2 pp1 ppn; do [ -f ref1.json ] && [ -f $a.json ] && { echo -n "$a vs ref1: "; python3 $HERE/gate.py cmp ref1.json $a.json; }; done
echo DONE $TAG
