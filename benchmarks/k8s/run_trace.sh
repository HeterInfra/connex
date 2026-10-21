#!/bin/bash
# Run one trace replay against a deployed pipeline and record run metadata.
# usage: [PROXY=local|URL] run_trace.sh PREFIX TRACE_CSV OUT_DIR [ENDPOINTS]
# env:   CONNEX_API_URL     default endpoint list (comma-separated stage-0 URLs)
#        CONNEX_NAMESPACE   k8s namespace for pod metadata (default connex; skipped without kubectl)
#        MODEL_NAME         served model name (default PREFIX-model)
set -euo pipefail
cd "$(dirname "$0")"
PREFIX=$1 TRACE=$2 OUT=$3
EP=${4:-${CONNEX_API_URL:?pass ENDPOINTS or set CONNEX_API_URL}}
NS=${CONNEX_NAMESPACE:-connex}
MODEL=${MODEL_NAME:-$PREFIX-model}
mkdir -p "$OUT"
name="${PREFIX}_$(basename "$TRACE" .csv)_$(date +%H%M%S)"
pods="[]"
if command -v kubectl >/dev/null; then
  pods=$(kubectl -n "$NS" get pods -l pipeline="$PREFIX" -o json 2>/dev/null | python3 -c 'import json,sys;print(json.dumps([[p["metadata"]["name"],p["spec"].get("nodeName"),[e.get("value") for e in p["spec"]["containers"][0]["env"] if e["name"] in ("BACKEND","NUM_STAGES")]] for p in json.load(sys.stdin)["items"]]))' 2>/dev/null || echo "[]")
fi
{
  echo "{"
  echo "\"connex_head\": \"$(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo unknown)\","
  echo "\"connexte_md5\": \"$(md5sum ../../integrations/vllm/connexte_communicator.py | cut -c1-12)\","
  echo "\"connex_md5\": \"$(md5sum ../../integrations/vllm/connex_communicator.py | cut -c1-12)\","
  echo "\"pods\": $pods,"
  echo "\"trace\": \"$TRACE\", \"endpoints\": \"$EP\""
  echo "}"
} > "$OUT/$name.meta.json"
# Optional resume proxy (connex/serving/proxy.py), off by default.
#   PROXY=local   start one on this host (reads the same endpoints) and send load through it
#   PROXY=URL     send load through an already running proxy (e.g. deploy/k8s/manifests.py proxy)
start_proxy() {  # args: proxy arguments; sets EP_PROXY, PROXY_PID
  PROXY_PID=""
  case "${PROXY:-}" in
    "") EP_PROXY="" ;;
    local)
      PYTHONPATH=../.. python3 -m connex.serving.proxy --host 127.0.0.1 --port "${PROXY_PORT:-5199}" "$@" \
        > "$OUT/$name.proxy.log" 2>&1 &
      PROXY_PID=$!
      EP_PROXY="http://127.0.0.1:${PROXY_PORT:-5199}"
      for _ in $(seq 50); do curl -sf --noproxy '*' "$EP_PROXY/health" >/dev/null && break; sleep 0.2; done ;;
    *) EP_PROXY=$PROXY ;;
  esac
}
stop_proxy() {
  [ -n "$EP_PROXY" ] && curl -s --noproxy '*' "$EP_PROXY/proxy/stats" > "$OUT/$name.proxy_stats.json" || true
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" || true
}
start_proxy --endpoints "$EP"
python3 trace_load.py --trace "$TRACE" --endpoints "${EP_PROXY:-$EP}" --model "$MODEL" --out "$OUT/$name.json"
stop_proxy
