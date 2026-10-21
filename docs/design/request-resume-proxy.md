# Request retry/resume proxy for stage-0 deaths

Code: `connex/serving/proxy.py`, fake backend `connex/serving/fake_backend.py`,
tests `tests/test_resume_proxy.py`, overhead `benchmarks/proxy_overhead.py`.
It is optional and off by default. To turn it on, set `PROXY=local` (or
`PROXY=<url>`) for `benchmarks/k8s/run_churn.sh` or `run_trace.sh`, or deploy
it with `python deploy/k8s/manifests.py proxy --proxy-endpoints URL[,URL...]`.

## Why requests were lost

Under hard churn (replica 1 scaled to 0), about 99.1% goodput remained. The
lost requests were of two kinds. A few were zero-token resets in the window
where the endpoints file still listed the dying replica. The rest were streams
cut part-way through, and every one of these failed about 15 s after the kill
(76.2 s and 256.4 s, for kills at 61.08 s and 241.24 s).

The 15 s is `terminationGracePeriodSeconds: 15` in `manifests.py`, not a client
timeout. This explanation comes from reading the code, and the cluster has not
confirmed it yet. The chain is as follows:

1. `kubectl scale --replicas=0` makes the kubelet send SIGTERM to PID 1.
   `docker/launch_stage.sh` ends in `exec python -m vllm.entrypoints.openai.api_server`,
   so PID 1 is the API server itself.
2. uvicorn handles SIGTERM gracefully. It closes the listening socket, then
   waits for open connections to finish. vLLM does not set
   `timeout_graceful_shutdown`, so uvicorn waits without a limit.
3. The other stages of replica 1 receive SIGTERM at the same moment, so the
   pipeline produces no more tokens. The in-flight SSE streams stay open but
   silent.
4. After 15 s the kubelet sends SIGKILL. The kernel then resets the sockets,
   and the client sees `ClientPayloadError`.

The client (`trace_load.py`) has only `ClientTimeout(total=60)`, with no read or
idle timeout. TCP keepalive (2 h by default) never matters here, because the
peer kernel is alive and the connection really is open. Only the RST at SIGKILL
ends the stream. `CONNEXTE_RECV_RETRIES=15` is a coincidence: it controls how
long stage 0 waits for a downstream stage, not how long the HTTP stream stays
open.

There are two consequences. First, an idle timeout set after the first token
(2.5 s by default) cuts the detection lag from 15 s to about 2.5 s. Second, a
draining uvicorn refuses new connections at once, so a prober that opens a
fresh connection to `GET /v1/models` notices the death within one probe
interval (1 s). That also unblocks requests still waiting for their first
token, without any timer that could fire on a slow prefill.

## How resume works

- The proxy sends every backend call with `stream=true` and
  `stream_options={include_usage, continuous_usage_stats}`. For completions it
  also sets `logprobs=0` and `return_tokens_as_token_ids=true`, which gives the
  proxy the exact generated token ids. It strips these fields before the client
  sees them.
- When a request fails after k tokens, the proxy re-sends
  `prompt = tokenize(prompt) + generated_ids` (one `/tokenize` call on the new
  replica, made only on this failure path), with `max_tokens -= k`,
  `min_tokens -= k` and `echo=false`. There is no re-tokenisation of generated
  text, so greedy output is bit-identical.
- Chat requests resume from text: `messages + [{"role": "assistant",
  "content": partial}]` with `continue_final_message=true` and
  `add_generation_prompt=false`. Here a re-tokenisation drift is possible at the
  splice point.
- On splicing, the proxy keeps the first stream's `id`, `created` and `model`.
  It drops the resumed stream's role preamble, and emits a single usage block
  with `prompt_tokens` from the first attempt and `completion_tokens` equal to
  the total emitted.
- A failure before the first token (refused, reset, 5xx, or the replica
  blacklisted while waiting) is retried on another replica. 4xx responses are
  passed through unchanged.
- An idle timeout after the first token resumes elsewhere. It blacklists the
  replica only if a probe of that replica also fails, so a replica that is slow
  but alive keeps its other requests.

## Steady-state cost (CPU, fake backend, 64 tokens/request, TPOT 10 ms)

Run `python benchmarks/proxy_overhead.py --concurrency C --requests N`. The
backend, the proxy and the client are separate processes on one host.
The added per-chunk latency is (client receive time - backend send time),
through the proxy minus direct.

| concurrency | added p50 / chunk | added p99 / chunk | added p99, past the first 4 chunks | added TTFT p50 |
|---|---|---|---|---|
| 1  | 0.25 ms | 0.52 ms | 0.40 ms | 2.1 ms |
| 16 | 0.18 ms | 1.9 ms | 0.54 ms | 5.9 ms |
| 64 | 0.13 ms | 2.6 ms | 1.2 ms | 13.9 ms |

All requests start in lockstep, so the TTFT and overall p99 numbers include a
burst of 16 or 64 simultaneous arrivals. Each arrival costs the proxy a JSON
parse and a backend POST, about 0.2 ms each in Python. The first version used
`asyncio.wait_for` per chunk for the idle timeout and stdlib json. At c=64 its
p99 was about 64 to 100 ms, because each `wait_for` creates a Task per chunk
(6,400 chunks/s). One watchdog task per stream plus orjson fixed this. These
costs are tiny next to real TPOT (tens of ms) and TTFT (hundreds of ms).

A process-level smoke test SIGKILLed one of two fake backends while 20 streams
of 100 tokens were in flight. All 20 finished bit-identical to the reference,
1.4 s after the kill, with 10 resumes.

## Cluster validation (details in docs/results.md)

- The vLLM fork accepts `return_tokens_as_token_ids`, `stream_options.continuous_usage_stats` and `POST /tokenize`, so resume uses token ids, not the text fallback. 58 resumes in 6 hard-churn runs, 0 failed.
- Hard churn, K=4, 2 replicas: goodput 99.4 / 99.6 / 99.2 % without the proxy and **100 / 100 / 100 %** with `PROXY=local` at QPS 2 / 4 / 8 (QPS 8: 2 reps, 12 and 14 resumes, 1 retry each).
- The 15 s lag is confirmed. In the pod watch, SIGTERM (Terminating) came at +0, then uvicorn logged `Waiting for connections to close` at +0.9 s, every cut stream ended at +15.1 s, and the pod went to Error at +15.5 s.
- Churn-free K=4 QPS 8: 0 resumes and 0 retries in 3 runs. The 2.5 s idle timeout never fired on a live replica.
- **`logprobs=0` used to cost a cold-start stall.** vLLM compiles `batched_count_greater_than` (`vllm/v1/sample/ops/logprobs.py`, `@torch.compile(dynamic=True)`) lazily, on the first request with logprobs, so every cold engine stalled about 8.5-9 s once (proxy mean TTFT 545 ms against 112 ms without it). Patch 15 (`patch_external_pp_warmup_logprobs.py`, `CONNEX_WARMUP_LOGPROBS=1` by default) moves that compile into engine warm-up; with it the proxy's steady-state mean / P99 E2E is 2.68 / 10.10 s against 2.64 / 10.03 s without the proxy. `--no-token-ids` avoids logprobs entirely, at the price of re-tokenisation drift on resume.
- The image ships aiohttp 3.13.2 but not orjson, so an in-cluster proxy pod runs with stdlib json. The cluster runs above used `PROXY=local` on the client host.
- With a single endpoint, the retry budget (4 attempts) does not span a ~1 s connection-reset blip of that replica (seen once: 15 requests failed with HTTP 502 in a churn-free run).

## Still open

- Membership comes from the endpoints file or the prober. The proxy does not
  yet subscribe to front-router events (`front_router.py`, PUB 15560), because
  stage routes do not carry the HTTP URL.
- The proxy pod (`deploy/k8s/manifests.py proxy`) has not been measured, and neither has
  orjson in the image.
