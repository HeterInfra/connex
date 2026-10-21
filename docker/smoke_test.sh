#!/bin/bash
# Wait for the stage-0 API and send one greedy completion.
# usage: smoke_test.sh [API_URL] [MODEL_NAME]
# env:   TIMEOUT_S (default 900)
set -euo pipefail
URL=${1:-${CONNEX_API_URL:-http://127.0.0.1:8000}}
MODEL=${2:-${MODEL_NAME:-connex-demo}}
end=$((SECONDS + ${TIMEOUT_S:-900}))
until curl -sf --noproxy '*' "$URL/v1/models" >/dev/null; do
  [ $SECONDS -lt $end ] || { echo "API at $URL not ready"; exit 1; }
  sleep 5
done
# Stage 0 serves before downstream stages finish loading; retry the request.
for _ in $(seq 30); do
  out=$(curl -sf --noproxy '*' -m 120 "$URL/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" || true)
  if echo "$out" | grep -q '"text"'; then
    echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("OK:", repr(d["choices"][0]["text"]))'
    exit 0
  fi
  sleep 10
done
echo "FAILED: no completion from $URL"; exit 1
