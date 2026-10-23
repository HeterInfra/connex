#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Trace replay under churn against a 2-replica pipeline deployed with
# deploy/k8s/manifests.py.
# usage: [PROXY=local|URL] run_churn.sh PREFIX MODE TRACE REPEAT DURATION OUT_DIR
# env:   CONNEX_R0_URL, CONNEX_R1_URL  stage-0 API URLs of replica 0 / 1 (required)
#        CONNEX_NAMESPACE              k8s namespace (default connex)
#        CONNEX_ROUTER                 front-router API host:port (default: port-forward to 127.0.0.1:15555)
#        CONNEX_ROUTER_NAME            router Service to port-forward (manifests.py --router-name; default connex-front-router)
#        CHURN_INTERVAL_S              seconds between down/up events (default 60)
#        MODEL_NAME                    served model name (default PREFIX-model)
set -euo pipefail
cd "$(dirname "$0")"
PREFIX=$1 MODE=$2 TRACE=$3 REPEAT=$4 DUR=$5 OUT=$6
NS=${CONNEX_NAMESPACE:-connex}
: "${CONNEX_R0_URL:?set CONNEX_R0_URL}" "${CONNEX_R1_URL:?set CONNEX_R1_URL}"
INTERVAL=${CHURN_INTERVAL_S:-60}
MODEL=${MODEL_NAME:-$PREFIX-model}
mkdir -p "$OUT"
name="${PREFIX}_${MODE}_$(basename "$TRACE" .csv)_$(date +%H%M%S)"
if [ -z "${CONNEX_ROUTER:-}" ]; then
  RN=${CONNEX_ROUTER_NAME:-connex-front-router}
  pgrep -f "port-forward svc/$RN " >/dev/null || \
    (kubectl -n "$NS" port-forward "svc/$RN" 15555:15555 >/dev/null 2>&1 &)
  sleep 2
fi
EPS="$OUT/$name.eps"
printf "%s\n%s\n" "$CONNEX_R0_URL" "$CONNEX_R1_URL" > "$EPS"
{
  echo "{"
  echo "\"connex_head\": \"$(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo unknown)\","
  echo "\"mode\": \"$MODE\", \"interval_s\": $INTERVAL, \"trace\": \"$TRACE\", \"repeat\": $REPEAT,"
  echo "\"pods\": $(kubectl -n "$NS" get pods -l pipeline="$PREFIX" -o json | python3 -c 'import json,sys;print(json.dumps([[p["metadata"]["name"],p["spec"].get("nodeName"),{e["name"]:e.get("value") for e in p["spec"]["containers"][0]["env"] if e["name"] in ("BACKEND","VLLM_PATCHES","CONNEX_PP_INFLIGHT","CONNEX_PP_DOWNSTREAM_GRAPHS")}] for p in json.load(sys.stdin)["items"]]))')"
  echo "}"
} > "$OUT/$name.meta.json"
python3 churn.py --prefix "$PREFIX" --mode "$MODE" --interval "$INTERVAL" --duration "$DUR" \
  --router "${CONNEX_ROUTER:-127.0.0.1:15555}" --namespace "$NS" --model "$MODEL" \
  --r0 "$CONNEX_R0_URL" --r1 "$CONNEX_R1_URL" \
  --endpoints-file "$EPS" --events-out "$OUT/$name.events.json" > "$OUT/$name.churn.log" 2>&1 &
CH=$!
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
start_proxy --endpoints-file "$EPS"
if [ -n "$EP_PROXY" ]; then
  python3 trace_load.py --trace "$TRACE" --repeat "$REPEAT" --endpoints "$EP_PROXY" \
    --model "$MODEL" --out "$OUT/$name.json"
else
  python3 trace_load.py --trace "$TRACE" --repeat "$REPEAT" --endpoints "$(paste -sd, "$EPS")" \
    --endpoints-file "$EPS" --model "$MODEL" --out "$OUT/$name.json"
fi
stop_proxy
wait $CH || true
cat "$OUT/$name.churn.log"
