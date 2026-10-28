# Connex in real vLLM under replica churn

Setup (2026-10-16, one 4-GPU host): Llama-3-8B, 2-stage external pipeline (external-PP vLLM
fork `380186d`), stage 0 on GPU1, stage-1 replicas A (GPU2) and B (GPU3) in one
container, front-router in memory mode. Load: 8 closed-loop clients,
32 tokens each, temperature 0. At t=20 s, replica A, which carries stage 0's
traffic, is either SIGKILLed or SIGTERMed. B is a warm standby.

Reproduce:

The original single-container harness was tied to a local environment and is
not shipped. The same experiment can be run with `docker/docker-compose.fork.yml`
(add a second stage-1 service with `REPLICA=1`) and `benchmarks/k8s/closed_loop.py`
while killing or stopping one stage-1 container.

## Results

3 fresh-pipeline runs per cell; "before" is commit 9d52994.

| version | event | recovered | stage-0 crash | failed requests (median) | recovery (median) | tok/s after / before |
|---|---|---|---|---|---|---|
| before | kill -9 | 0/3 | 0/3 | 8 (all in-flight, then hang) | never | 0.00 |
| before | SIGTERM | 1/3* | 3/3 | ~92k** | — | 0.00 |
| after | kill -9 | 3/3 | 0/3 | 0 | 3.6 s | 1.00 |
| after | SIGTERM | 3/3 | 0/3 | 0 | 3.4 s | 1.00 |

\* One request that started after the event happened to finish before stage 0 crashed.
\*\* The closed-loop clients hammer a dead API server with fast connection refusals.
"Recovery" is the time from the event until the first request that *started after
it* completes, so it includes about 1.8 s of normal request latency.

## Failure chain before the fix

**kill -9 → permanent hang.** In the default mode (ZMQ signal + UCX data),
stage 0 waited for the return with `router_socket.recv_multipart()`, which has
no timeout. The fork's stage-0 loop already has a retry contract ("return None
on timeout/peer failure, I re-send the step"), but Connex never returned None.
The router flags the dead replica only after its heartbeat timeout (30 s +
grace), and even that does not unblock a recv that is already waiting.

**SIGTERM → stage 0 crashes with a CUDA device-side assert.** This happens in three steps.
1. The dying replica's EngineCore interpreter starts shutting down, but its
   non-daemon threads keep running. UCX sends then fail with "cannot schedule new
   futures after interpreter shutdown", and the communicator silently falls back to
   inline ZMQ, *still sending* returns.
2. Stage 0 has already abandoned that step and re-sent it to B.
3. The receiver paired messages by queue position only. The late reply from A
   was taken as the answer to the retry, every later step was off by one,
   and stage 0 sampled from garbage, which triggered the assert in
   `_bookkeeping_sync`.

**Restarted replica silently half-broken.** The UCX listener bind failed with
"Address already in use", from a TIME_WAIT socket of the previous incarnation or a
leftover `VLLM::` process. The transport only logged a warning, and the replica
registered its UCX address anyway. Upstream connects then timed out forever.

## Fixes (integrations/vllm/connex_communicator.py, connex/ucx_transport.py)

| Problem | Fix |
|---|---|
| recv blocks forever | polled wait; returns None on peer loss or after `CONNEX_RECV_TIMEOUT_MS` (8 s) |
| loss detected only after router heartbeat timeout | transport `peer_closed_callbacks`: the connection that delivered our last return dies → peer lost **immediately** |
| router keeps routing to the dead replica | stage 0 deregisters the dead replica's LID at the router (`_report_peer_failure`); a falsely reported live replica re-registers on its next heartbeat |
| late reply mistaken for retry's reply | correlation id frame: stage 0 stamps each send, stage 1 echoes it, stage 0 drops mismatches |
| UCX signal ↔ payload paired by position | the signal carries the header send timestamp; stale payloads are dropped and early ones stashed |
| SIGTERM leaves a registered zombie | `threading._register_atexit` hook deregisters and stops the heartbeat before non-daemon threads are joined |
| listener bind failure ignored | `UCXListenerError` is raised and the communicator refuses to start; optional `port_fallback` (communicator: 16) binds a nearby port and advertises it |
| `_close_ucx_endpoint_for_stage` leaked | called `ep.close()` (an un-awaited coroutine); now `transport.close_endpoint` |
| rendezvous segfault on peer death | RDMA mode with a non-RDMA `UCX_TLS` gets the eager default too |

## Caveats

- **KV state of in-flight requests.** A decode step that was re-sent to B
  runs without that request's earlier KV cache on B. The 8 in-flight outputs
  read coherently, but they are not verified as exact. A bit-exact check is
  impossible here, because batching already changes greedy outputs run to run
  (100/272 unaffected requests differed from a sequential reference).
  Proper handling (re-prefill on failover) belongs in the fork's scheduler.
- Real RDMA (`UCX_TLS` with rc/dc/ud) still uses rendezvous, so the UCX
  segfault on peer death can still happen there.
- Harness pitfalls: vLLM renames processes to `VLLM::...`, so
  `pkill -f vllm` misses them, and they keep holding ports. Run the container
  with `--init`, or zombies pile up.
