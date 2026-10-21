# SPDX-License-Identifier: Apache-2.0
"""Per-hop communication cost of decode steps: metric deltas around one
idle streaming request.
usage: hop_profile.py MODEL STAGE_HOSTS [API_PORT]
  STAGE_HOSTS  comma-separated host of every stage (stage 0 first), or env CONNEX_STAGE_HOSTS"""
import os
import collections
import re
import sys
import time
import json
import urllib.request

urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))  # direct, bypass any HTTP proxy

model = sys.argv[1]
ips = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CONNEX_STAGE_HOSTS", "127.0.0.1")).split(",")
port = sys.argv[3] if len(sys.argv) > 3 else "5100"


def snap(ip):
    d = collections.defaultdict(float)
    txt = urllib.request.urlopen(f"http://{ip}:{port}/metrics", timeout=10).read().decode()
    for l in txt.splitlines():
        m = re.match(r"vllm:pipeline_comm_([a-z_]+)_(sum|count)\{([^}]*)\} ([0-9.e+-]+)", l)
        if m and "tensor" not in m.group(1):
            d[(m.group(1), m.group(2))] += float(m.group(4))
    return d


before = {ip: snap(ip) for ip in ips}
body = json.dumps({"model": model, "prompt": "Tell me a story about a cat.", "max_tokens": 64,
                   "temperature": 0, "ignore_eos": True}).encode()
t0 = time.time()
urllib.request.urlopen(urllib.request.Request(f"http://{ips[0]}:{port}/v1/completions", body,
                                              {"Content-Type": "application/json"}), timeout=120).read()
print(f"request {time.time() - t0:.2f}s for 64 tokens -> {1000 * (time.time() - t0) / 64:.1f} ms/token")
for i, ip in enumerate(ips):
    after = snap(ip)
    parts = []
    for (name, kind) in sorted(after):
        if kind != "sum":
            continue
        n = after[(name, "count")] - before[ip][(name, "count")]
        if n:
            parts.append(f"{name}={1000 * (after[(name, 'sum')] - before[ip][(name, 'sum')]) / n:.2f}ms")
    print(f"stage {i}: " + " ".join(parts))
