#!/bin/bash
# CPU-only end-to-end demo, no GPU, RDMA or vLLM needed:
#   1. two pipeline stages find each other through the Connex directory and
#      exchange a tensor (ZeroMQ, or UCX over TCP if ucx-py is installed);
#   2. the request retry/resume proxy in front of two deterministic fake
#      OpenAI backends: one backend is killed mid-stream, the client still
#      receives the full, correct completion.
# usage: examples/cpu_demo.sh        (needs: pip install -e ".[proxy]", or pip install "connex-ai[proxy]")
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PYTHON:-python3}
BASE=${DEMO_BASE_PORT:-18100}
A=$BASE B=$((BASE + 1)) P=$((BASE + 99))
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "== 1. two stages over the Connex directory"
$PY examples/two_stage_pipeline.py

echo "== 2. resume proxy: kill a backend mid-stream"
$PY -m connex.serving.fake_backend --port "$A" --tpot-ms 20 >/dev/null 2>&1 & pids+=($!); PID_A=$!
$PY -m connex.serving.fake_backend --port "$B" --tpot-ms 20 >/dev/null 2>&1 & pids+=($!); PID_B=$!
$PY -m connex.serving.proxy --host 127.0.0.1 --port "$P" --log-level WARNING \
    --endpoints "http://127.0.0.1:$A,http://127.0.0.1:$B" >/dev/null & pids+=($!)
for _ in $(seq 50); do curl -sf --noproxy '*' "http://127.0.0.1:$P/health" >/dev/null && break; sleep 0.2; done

$PY - "$P" "$A" "$B" "$PID_A" "$PID_B" <<'PYEOF'
import json, os, signal, sys, threading, time, urllib.request
port, a, b, pid_a, pid_b = sys.argv[1:]
body = {"model": "fake", "prompt": "w5 w7 w11", "max_tokens": 100, "temperature": 0}

def post(url, d):
    req = urllib.request.Request(url, json.dumps(d).encode(), {"Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60)

ref = json.load(post(f"http://127.0.0.1:{a}/v1/completions", body))["choices"][0]["text"]
# Kill the backend that serves the stream after ~1 s (20 ms/token -> ~50 tokens in).
def killer():
    time.sleep(1.0)
    stats = json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        f"http://127.0.0.1:{port}/proxy/stats"))
    busy = [e["url"] for e in stats["endpoints"] if e["inflight"]][0]
    victim = pid_a if busy.endswith(":" + a) else pid_b
    os.kill(int(victim), signal.SIGKILL)
    print(f"killed the backend serving the stream ({busy})")
threading.Thread(target=killer, daemon=True).start()
text, n = "", 0
with post(f"http://127.0.0.1:{port}/v1/completions", dict(body, stream=True)) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                text += c.get("text", ""); n += 1
stats = json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
    f"http://127.0.0.1:{port}/proxy/stats"))
print("proxy stats after:", {k: stats[k] for k in ("requests", "retries", "resumes", "failed")})
ok = text == ref and stats["resumes"] >= 1
print(f"streamed {n} chunks, {len(text.split())} tokens; identical to uninterrupted reference: {ok}")
sys.exit(0 if ok else 1)
PYEOF
echo "== demo OK"
