#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Churn matrix: configuration variants x QPS list, clean restart
# (router + both replicas) before every run. Deploys with deploy/k8s/manifests.py.
# usage: churn_matrix.sh MODE OUT_DIR [VARIANTS] [QPS_LIST]
#   MODE      soft (deregister/re-register replica 1) | hard (scale it to 0 and back)
#   VARIANTS  "NAME[:KEY=VAL,KEY=VAL...] ..." stage env per variant
#             (default "k4:CONNEX_PP_INFLIGHT=4,CONNEX_PP_DOWNSTREAM_GRAPHS=1")
#   QPS_LIST  "QPS:REPEAT ..." (default "2.0:4 4.0:2 8.0:2")
# Set PROXY=local to send the load through the resume proxy (see run_churn.sh).
# env:
#   CONNEX_NAMESPACE   k8s namespace (default connex)
#   NODES              comma-separated node names for replica 0, stage 0..N-1 (required)
#   NODES_R1           nodes for replica 1 (default: NODES rotated by one)
#   TRACE_DIR          directory with gamma_qps<Q>.csv traces (default benchmarks/traces; see gen_trace.py)
#   MANIFEST_ARGS      extra args for manifests.py (--image, --model-host-path, ...)
#   CONNEX_R0_URL / CONNEX_R1_URL  stage-0 API URLs (see run_churn.sh)
#   PREFIX             pipeline name (default cte)
#   CONNEX_ROUTER_NAME router Deployment/Service name (default connex-front-router;
#                      also passed to manifests.py as --router-name)
set -u
cd "$(dirname "$0")" || exit 1
MODE=$1 OUT=$2 VARIANTS=${3:-"k4:CONNEX_PP_INFLIGHT=4,CONNEX_PP_DOWNSTREAM_GRAPHS=1"} QPSL=${4:-"2.0:4 4.0:2 8.0:2"}
NS=${CONNEX_NAMESPACE:-connex}
T=${TRACE_DIR:-$(cd ../traces && pwd)}
: "${NODES:?set NODES=node-a,node-b,...}"
IFS=, read -ra N0 <<< "$NODES"
if [ -n "${NODES_R1:-}" ]; then IFS=, read -ra N1 <<< "$NODES_R1"; else N1=("${N0[@]:1}" "${N0[0]}"); fi
GEN=../../deploy/k8s/manifests.py
P=${PREFIX:-cte}
RN=${CONNEX_ROUTER_NAME:-connex-front-router}
export CONNEX_ROUTER_NAME=$RN
deploy() {  # $1 = comma-separated KEY=VAL list (may be empty)
  # shellcheck disable=SC2206
  local extra=(--namespace "$NS" --router-name "$RN" --num-stages ${#N0[@]} --set VLLM_PATCHES=all ${MANIFEST_ARGS:-})
  local kv
  IFS=, read -ra kvs <<< "$1"
  for kv in "${kvs[@]}"; do [ -n "$kv" ] && extra+=(--set "$kv"); done
  python3 $GEN pipeline --backend connexte --prefix "$P" --nodes "$NODES" "${extra[@]}" | kubectl apply -f - >/dev/null
  for i in "${!N1[@]}"; do
    python3 $GEN stage --backend connexte --prefix "$P" --stage "$i" --replica 1 \
      --node "${N1[$i]}" "${extra[@]}" | kubectl apply -f - >/dev/null
  done
}
for variant in $VARIANTS; do
  comm=${variant%%:*}
  envs=""; [ "$variant" != "$comm" ] && envs=${variant#*:}
  deploy "$envs"
  for qr in $QPSL; do
    q=${qr%%:*}; r=${qr#*:}
    # a previous hard-churn run may have left replica 1 scaled to 0
    kubectl -n "$NS" scale deploy -l pipeline="$P" --replicas=1 >/dev/null
    kubectl -n "$NS" delete pod -l router="$RN" >/dev/null; sleep 8
    kubectl -n "$NS" delete pod -l pipeline="$P" >/dev/null; sleep 25
    ./wait_ready.sh "$P" 1200 >/dev/null || { echo "$comm $q: not ready"; continue; }
    pkill -f "port-forward svc/$RN " || true; sleep 1
    res=$(./run_churn.sh "$P" "$MODE" "$T/gamma_qps$q.csv" "$r" 240 "$OUT/$comm" 2>&1 | grep '^{"requests"')
    echo "$comm qps$q $res"
  done
done
