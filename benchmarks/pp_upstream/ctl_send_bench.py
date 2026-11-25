# SPDX-License-Identifier: Apache-2.0
"""Engine-thread time per step of stage 0's EXEC/SAMPLE sends, inline
(CONNEX_PP_ASYNC_CTRL=0) vs queued (the default), on CPU with a slow
stand-in for connex-rs send_msg.

    python benchmarks/pp_upstream/ctl_send_bench.py --send-us 300 --peers 3

Each step calls the executor's execute_model and sample_tokens (the local
stage does no work) and then sleeps --gap-ms, standing in for the GPU step
during which the sender threads drain. The stand-in send releases the GIL
(time.sleep), as connex-rs does. Without vLLM installed, the executor is
imported with the stand-ins of tests/_fake_vllm.py.
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]


def load():
    try:
        from connex.adapters.vllm.pp import executor
        return executor
    except ImportError:
        from _fake_vllm import load_executor
        return load_executor()[0]


def head(ex, peers, send, async_ctrl, payload):
    from connex.adapters.vllm.pp.ctltx import CtlSender

    h = ex.ConnexPPExecutor.__new__(ex.ConnexPPExecutor)
    h.head = True
    h.lock = threading.RLock()
    h.cv = threading.Condition(h.lock)
    h.num_stages = peers + 1
    h.replicas = [[ex.Replica(k, 0, f"ctl{k}", f"d{k}", alive=True)] for k in range(1, peers + 1)]
    h.route = [x[0] for x in h.replicas]
    h.epoch, h.seq, h.replica_id, h.owner = 1, 0, 0, "o"
    h.pending, h.live_reqs, h.failovers, h.await_sample = {}, set(), [], deque()
    h.lost_pending, h.replace_timeout, h.max_tokens, h.local_data_url = False, 1.0, 8192, "d0"

    def inline(peer, msg, raw=None, quiet=False):
        send(peer, raw)
        return True
    # the real encode pickles and seals the message: same order of cost
    h.ctl = SimpleNamespace(encode=lambda m: pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL),
                            send=inline)
    h.ctl_tx = CtlSender(send) if async_ctrl else None
    h._payload = payload
    return h


def run(ex, args, async_ctrl):
    calls = []

    def send(peer, data):
        t = time.perf_counter()
        time.sleep(args.send_us * 1e-6)
        calls.append(time.perf_counter() - t)

    h = head(ex, args.peers, send, async_ctrl, b"x" * args.bytes)
    so = SimpleNamespace(total_num_scheduled_tokens=512, num_scheduled_tokens={"r": 512},
                         scheduled_new_reqs=[], finished_req_ids=set(), blob=h._payload)
    per_step = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        h.execute_model(so)
        h.sample_tokens(None, non_block=True)
        per_step.append(time.perf_counter() - t0)
        h.pending.clear()
        time.sleep(args.gap_ms * 1e-3)
    if h.ctl_tx is not None:
        h.ctl_tx.close(drain_s=5.0)
    per_step = sorted(per_step[args.steps // 10:])     # drop warm-up
    q = lambda f: per_step[min(len(per_step) - 1, int(f * len(per_step)))] * 1e6  # noqa: E731
    return {"p50_us": q(0.5), "p99_us": q(0.99), "mean_us": statistics.mean(per_step) * 1e6,
            "send_call_us": statistics.median(calls) * 1e6, "sends": len(calls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send-us", type=float, default=300)
    ap.add_argument("--peers", type=int, default=3)
    ap.add_argument("--bytes", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--gap-ms", type=float, default=5.0)
    args = ap.parse_args()
    ex = load()
    for name, a in (("inline (CONNEX_PP_ASYNC_CTRL=0)", False), ("queued (default)", True)):
        r = run(ex, args, a)
        print(f"{name:34s} engine thread per step: p50 {r['p50_us']:8.1f} us  "
              f"p99 {r['p99_us']:8.1f} us  mean {r['mean_us']:8.1f} us   "
              f"(stand-in send_msg median {r['send_call_us']:.0f} us, {r['sends']} sends)")


if __name__ == "__main__":
    main()
