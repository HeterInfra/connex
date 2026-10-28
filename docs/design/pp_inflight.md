# External PP pipelining (CONNEX_PP_INFLIGHT=K)

## Why only one step is in flight today

Stage 0 is a single-rank vLLM (pipeline_parallel_size=1 inside vLLM, the
stages are separate pods). UniProcExecutor.max_concurrent_batches is 1 unless
async scheduling is on, so EngineCore uses step() and never the batch queue,
and the external-PP branch of gpu_model_runner.execute_model blocks on
recv_tensor_dict until the last stage returns. With 4 equal stages each GPU
is busy about 1/4 of the time.

Upstream V1 already has everything on the scheduler side: with
max_concurrent_batches > 1 EngineCore runs step_with_batch_queue, and
Scheduler.schedule skips a request whose num_new_tokens is 0 (scheduler.py
~227/263), which is the case for a decode request whose sampled token has not
come back. So consecutive in-flight batches are disjoint for free.

## What is implemented

Communicator (integrations/vllm/connexte_communicator.py), active only with
K > 1: new_step_id / send_step / recv_step / abandon_step on stage 0 (several
outstanding ids, replies matched by id in any order, other replies buffered,
per-step retry under the same id), recv_forward / send_reply on later stages
(thread-local per-step context so a prefetching receiver does not overwrite
the corr id / originator of the step being answered), a ring of the last K
replies for dedupe (GPU ring regions pinned), and a stale window of K in
_CorrFilter. Wire format unchanged. K=1 keeps the original code paths.

gpu_worker patch (patches/patch_external_pp_inflight.py): stages 1+ prefetch
step n+1 in a thread while computing step n.

## Stage 0 (patch 7, patches/patch_external_pp_stage0_inflight.py)

Why this shape: the least invasive correct design keeps all runner work in
ONE thread. Instead of AsyncModelRunnerOutput on UniProcExecutor's async
thread (which would race with the next execute_model on self.input_batch),
sample_tokens of an outstanding step returns a deferred object and the
patched UniProcExecutor wraps it in a lazy Future whose result() does
recv_step + logits + sampling + bookkeeping in the EngineCore thread when
step_with_batch_queue pops that step.

* executor: max_concurrent_batches = K in external-PP mode with
  CONNEX_PP_INFLIGHT=K>1 (EngineCore then uses step_with_batch_queue).
* scheduler: per step at most ceil(len(running)/K) RUNNING requests are
  scheduled. Without this cap the scheduler puts every ready request in one
  step, the next step is empty, and nothing overlaps (all 32 requests decode
  in lockstep).
* runner: K InputBatch objects (built from the recorded constructor kwargs
  with fresh logits processors). A row is only trusted in the slot that last
  scheduled the request (_CxSlots.home); stale rows are dropped before
  _update_states so it re-adds them from CachedRequestState. Rows of an
  outstanding step are never touched (chunked prefill can be scheduled again
  while its previous chunk is outstanding; that row is a discarded sample).
  Requests aborted while outstanding keep their state aside until that
  step's bookkeeping ran. Forward dicts are cloned for retries because
  cudagraph output buffers are reused by the next forward.

Two bugs found on GPU (2 x A40, Llama-3-8B, 2 stages), both reproducible
with the 32-prompt concurrency check (sequential K=1 vs concurrent):

* Cross-request contamination at K>1 (a haiku prompt continuing with Paris
  text, 0/32 outputs equal). Cause: cudagraph_mode is FULL_AND_PIECEWISE by
  default, the full decode graphs bake in the GPU addresses of slot 0's
  BlockTable.block_table / slot_mapping buffers, so steps built in slots 1..K-1
  attended through stale tables. Fix: all slots share slot 0's GPU buffers
  (CPU rows stay per slot). Any future "K persistent batches" design must do
  the same for every tensor a captured graph reads.
* Fork bug, K independent (patch 8, patch_external_pp_mixed_prefill.py):
  stages 1+ set seq_len = query length for every request of a step labelled
  prefill, so decodes mixed with a new prompt lose their context there. Fix
  uses the num_computed_tokens stage 0 already sends.

Even with both fixes, concurrent greedy outputs differ from sequential ones
for about 1/3 of the prompts at K=1 too. Those diffs are plausible text
that diverges after some shared prefix (batch-shape dependent bf16 numerics
flipping a near tie). K=2/4 differences are the same kind, and many of them
equal the K=1 concurrent output exactly.

Measured (2026-10-17, GPUs 0/1 of the A40 node, 2 stages 0-16/16-32,
MOONCAKE_PROTOCOL=tcp, closed loop 32 clients, max_tokens 64, ignore_eos,
60 s): K=1 635 tok/s, K=2 806 tok/s (1.27x), K=4 428 tok/s (0.67x).
Correctness: single greedy prompts identical for K=1/2/4; 32 concurrent vs
sequential K=1: K=1 20,23/32, K=2 22,24/32, K=4 21,20/32 (two runs each;
two K=1 concurrent runs agree 25/32 with each other, so K>1 sits inside the
existing batching-nondeterminism floor). K > number of stages does not help:
each microbatch gets smaller and the per-step stage-0 overhead (tcp send with
.cpu(), serialization, Python) dominates, so use K = number of stages. Why
K=2 is only 1.27x and not ~2x has not been profiled yet. The suspects are
stage-0 per-step CPU time and the synchronous send_step on the critical path.

Repro: `CONNEX_PP_INFLIGHT=K MOONCAKE_PROTOCOL=tcp docker compose -f
docker/docker-compose.fork.yml up` (the image applies apply_all.py at build time),
then `benchmarks/k8s/closed_loop.py --url http://127.0.0.1:8000`. P2PHANDSHAKE needs the communicator fix that advertises
ip:get_rpc_port() (the TE listens on a random RPC port, so ip:MOONCAKE_BASE_PORT
got connection refused).

Limitations: connexte backend only; spec decode / async scheduling make the
runner fall back to serial; structured output unsupported with K>1; stage-0
TTFT/TPOT pipeline metrics are not recorded for K>1 steps.

Enable: CONNEX_PP_INFLIGHT=K on every stage (launch_stage.sh passes the environment through),
apply_all.py inside the image (it patches runner, worker, uniproc_executor and
scheduler). No extra vLLM args.

## Why K=2 is only ~1.27x (profiled 2026-10-17, same setup)

Tools: py-spy from a helper container (`docker run --pid container:<c>
--cap-add SYS_PTRACE`, run py-spy as root there; inside the pod's own
container ptrace is denied), `nvidia-smi dmon -s u`, CONNEXTE_PROFILE=1,
CONNEX_PP_PROFILE=1 (patch 9, stage-0 phases + microbatch histogram) and
CONNEXTE_TRACE=/tmp/tr (per-step send/signal/arrive/recv timestamps of both
stages, joined by correlation id).

K=1, 32 tok per step, period ~49 ms, GPU0 35% / GPU1 31% busy. Stage-0
engine thread: 31% (15 ms) blocked in `logits_indices.cpu()` waiting for its
own forward, 54% (26 ms) waiting for the reply, ~10% (5 ms) scheduler +
prep + sampling. Stage 1: 57% idle, forward ~15 ms.

K=2, 16 tok per microbatch, period ~19.5 ms, GPU0 85% / GPU1 76% busy.
Cross-stage timeline (median per step):

    stage-0 forward (16 layers, full cudagraph)      ~13.5 ms
    send + Mooncake tcp read on stage 1               ~1.2 ms
    stage-1 forward (no cudagraphs, see below)       ~15.9 ms
    return + read on stage 0                          ~0.6 ms (+ wake-up)
    stage-0 turnaround: recv, lm_head + sampling      ~3.9 ms (1.9 ms of it GPU,
      (_to_list waits), bookkeeping, update,                   ~1 ms scheduler/prep)
      schedule, input prep
    one microbatch round trip                        ~38 ms -> period = 38/2

Conclusions:
* Decode is memory bound: a 16-request microbatch costs a stage almost the
  same GPU time as 32 requests (15 ms either way). Halving the batch halves
  the tokens per step, so ~2x is not reachable at 32 clients. With every
  CPU and transfer cost at zero the bound is (13.5+15.9)/2 = 14.7 ms per 16
  tokens, about 1090 tok/s (1.7x).
* K=2 is round-trip (ring) bound: stage 0's recv waits only 0.12 ms, i.e.
  the reply of microbatch Y is already there when the forward of X ends.
* Rejected hypotheses: microbatch imbalance (histogram: 224/256 steps are
  exactly 16), Mooncake tcp transfer (0.3-1 ms), stage-1 recv/compute overlap
  (the prefetch works, stage 1 idles 22%), scheduler CPU (<0.3 ms).
* K=3+ is worse: the per-microbatch GPU time does not shrink, so stage 0
  saturates at ~15 ms per 11 tokens.

## Overlap attempt (patch 9, CONNEX_PP_OVERLAP=1, default OFF)

Implemented and measured, slower, so it is opt-in only. Stage 0 builds
logits_indices from the CPU copy (no sync), the send goes to a sender thread
(send_step_async / send_reply_async), recv + logits + sampling run on a side
stream waiting only on the step's own event, and CONNEXTE_GDR_HOST=1 stages
the GPUDirect buffers in pinned host memory. That last part is needed
because Mooncake tcp moves GPU memory with a synchronous cudaMemcpy, which
waits for the other microbatch's forward on the legacy stream (r_gdr 2.1
ms). Results at K=2: 740-757 tok/s vs 822 for the plain path. The reply
arrives only ~1 ms before the other forward ends, so nothing is left to
overlap. Running sampling concurrently with a forward makes the turnaround
longer (6.8 vs ~4.5 ms), and since that turnaround is on the ring, the
period grows. A lower GIL switch interval (CONNEXTE_SWITCH_INTERVAL_S,
applied only with K>1) changed nothing. Host staging with synchronous sends
was also slower (795 vs 822). Greedy outputs stay correct in overlap mode
(fixed prompts identical, 13/32 concurrent-vs-sequential, inside the floor).

Final numbers with defaults (overlap off): K=1 686 tok/s, K=2 809-825 tok/s
(1.2x to 1.27x; the K=1 run scored higher than the earlier 635/651 runs on
the same code, so expect a few percent of run-to-run noise). Fixed greedy
prompts identical K=1/K=2. 32-concurrent vs sequential K=1: K=1 14-15/32,
K=2 17/32 (new prompt set; same batching-nondeterminism floor as before).

What would actually move K=2 (ring latency, ~38 ms):
1. CUDA graphs on stages 1+ (execute_external_pipeline_step calls
   set_forward_context without a batch descriptor, so stage 1 runs
   inductor code eagerly: 15.9 ms vs 13.5 ms for the same 16 layers on
   stage 0). Piecewise graphs need inputs copied into persistent buffers
   and padding to capture sizes. Estimated 1-2 ms per step.
2. More concurrency per microbatch (64+ clients): tokens per step grow and
   GPU time per step barely does.
3. Native code would only help the ~1 ms scheduler/prep Python and ~0.5 ms
   of recv glue in the turnaround (a few percent). The rest is GPU (lm_head
   + sampling, forwards) or wire time.
4. A layer split that moves 1-2 layers off the stage holding the lm_head
   helps only when one stage is the bottleneck, not in the ring-bound regime.

Container note: with a bridge network the container gets its own network
namespace; useful when the default front-router port 15555 is already taken
on the host.

## CUDA graphs on stages 1+ (patch 10, CONNEX_PP_DOWNSTREAM_GRAPHS=1, default OFF)

Why stage 1 was eager: stages 1+ already capture the normal graphs at
startup (FULL_AND_PIECEWISE: 51 piecewise sizes + 35 FULL decode sizes, 0.31
GiB, capture_model -> _dummy_run with self.intermediate_tensors,
self.positions, self.seq_lens, self.query_start_loc and
input_batch.block_table buffers). execute_external_pipeline_step just never
used them: set_forward_context got no cudagraph mode / batch descriptor and
the inputs were the received tensors, not the captured buffers.

What patch 10 does (patches/patch_external_pp_downstream_graphs.py): for a
pure decode step (step_type decode, num_actual_tokens == num_reqs, padded
size <= max capture size, one KV group, no LoRA/mrope) it copies
hidden_states/residual into self.intermediate_tensors and positions into
self.positions.gpu (pad rows zeroed, junk positions could index past the
rope cache), asks the dispatcher for BatchDescriptor(pad, uniform_decode=True)
and, for FULL (the FA2 default on A40), copies the block table rows, slot
mapping and seq_lens that the unchanged minimal builder produced into the
persistent buffers. Padding rows get seq_len 0, an empty query range and slot
-1, the same as the stock runner. PIECEWISE (if FULL is not captured) only
needs the layer inputs in persistent buffers, attention runs eagerly with the
minimal metadata. The output is a graph-pool tensor that the next replay
overwrites, so it is sliced to the input length and cloned (16x4096 bf16,
negligible). Prefill / mixed steps stay eager. Allocator bookkeeping is
untouched because the minimal builder still runs first. CONNEX_PP_PROFILE=1
also logs stage 1+ decode forward GPU time (CUDA events from before the input
copies to after the forward, read back lazily, median/p90 per 256 steps).

Measured 2026-10-17, GPUs 0/1 (A40), 2 stages, connexte, MOONCAKE_PROTOCOL=tcp,
32 closed-loop clients, max_tokens 64, ignore_eos, 60 s:

    config        stage-1 decode fwd (median)     tok/s (two runs)
    K=1 off       19.4-21.3 ms (launch gaps)      685, 666, 696
    K=1 on        15.0 ms                         720, 738   (+6%)
    K=2 off       15.1 ms                         805, 813
    K=2 on        14.7 ms                         810, 821   (+1%)

Correctness: fixed prompts ("The capital of France is", "Write a haiku about
the sea:", 24 tokens, temp 0) identical in every config. 32 prompts sent
sequentially, graphs on vs off: 32/32 identical. 32 concurrent: on vs off
16/32 and 22/32, off vs off 21/32 (floor), K=2 on vs K=2 off 18/32; the
diffs are near-tie divergences after a long shared prefix.

Why the gain is small at K=2: the event time includes CPU launch gaps, and at
K=1 (32-token steps) eager stage 1 leaves the GPU waiting for launches
(~20 ms vs 15 ms), but at K=2 eager stage 1 is already GPU bound (15.1 ms).
The replayed 16-layer decode costs 14.7 ms for 16 tokens, close to the
weight-streaming bound (~7 GB of layer weights at ~700 GB/s is ~10 ms), so
launches were only ~0.4 ms of it. The earlier "15.9 vs 13.5 ms" gap between
stage 1 and stage 0 is therefore mostly not launch overhead. K=2 stays ring
bound; graphs shave ~0.4 ms off a ~38 ms round trip.

Risks: the graph reads input_batch.block_table / seq_lens / query_start_loc
of stage 1's runner, which nothing else uses on stages 1+ (they never run
execute_model); if a later change makes stages 1+ run real InputBatch steps,
those writes would race. FA2 FULL graphs bake max_seqlen_k=max_model_len
from capture (same as stage 0). K>1 on stage 1 is safe because compute is
single-threaded (only recv is prefetched).

Repro: set CONNEX_PP_DOWNSTREAM_GRAPHS=1 CONNEX_PP_PROFILE=1
CONNEX_PP_INFLIGHT=K BACKEND=connexte MOONCAKE_PROTOCOL=tcp on every stage
(docker-compose environment). The image must contain the Connex
communicators: with the fork's own older connexte communicator the stages
hang in _register_stage (DEALER frames vs the front router's REP socket).

## Churn with K>1: lost re-sent steps and how they were fixed (2026-10-17)

Symptom: with K=4, at the first churn event (replica 1 leaves) stage 0 of
replica 0 stopped receiving the reply of one step, re-sent it 5 times (3 s
each) and then the engine died ("no reply for pipelined step ... after 5
attempts"). K=4 goodput was 37/23/28% soft and 43/28/80% hard at QPS 2/4/8,
while K=1 stayed at about 100%.

Root cause (the lost step). The re-sends did reach every stage: stages 1-3
logged "duplicate step e85fdc4e00001521 (sender retry); re-sending cached
reply" on every attempt, so the reply was lost on the return hop. At the
same moment stage 3 logged "Reset return_socket", because its
deregistration handler closed *every* return DEALER, including the one to
the surviving stage 0 of replica 0. The next reply went out on a new DEALER
with the same fixed ZMQ identity ("rk-return-3"). A ROUTER that still holds
a connection with identity X (the old one, draining its K in-flight replies
under LINGER) treats a second connection announcing X as anonymous and
silently drops everything it sends, for the life of that connection
(ROUTER_HANDOVER=0). tests/test_churn_fixes.py reproduces this with plain
pyzmq. K=1 rarely hits it because nothing is queued when the socket is
closed, so the old connection is gone before the new one connects. The same
trap existed on the forward hop: stage 0 closed and re-created its "rk-1"
DEALER on every recv timeout.

Root cause (the 0 ms burst). recv_step has a monotonic deadline and passes
the remaining time to recv_tensor_dict, which keeps its own wall-clock
deadline and truncates to whole ms. When the inner timeout fired a bit before
the outer deadline, recv_step looped on a sub-ms remainder, and every call
logged "recv timed out after 0 ms" and closed the forward socket again.

Heartbeats. The router log shows real false alarms: replicas that were still
alive marked unhealthy and then "recovered (heartbeat resumed)" (e.g.
1:0:1 at 08:14:59 to 08:15:01), mostly during startup and heavy load,
because a busy engine process starves its heartbeat thread past the 3 s
lease. Since f85713e UNHEALTHY was handled like DEREGISTERED.

Fixes:
- DEALER identities are unique per socket (_dealer_identity), as the older
  connex_communicator already did, so a re-created socket or two upstream
  replicas sharing a downstream peer can never collide.
- A stage-0 deregistration no longer closes the last stage's return sockets
  (each reply names its own stage-0 origin); socket close/reset in the event
  and invalidation paths runs under the send lock (no cross-thread close).
- UNHEALTHY only marks the downstream peer we actually route to as suspect.
  The failover happens when a step to it then times out
  (confirm_peer_failure), or when the router deregisters it.
- recv_step treats less than 5 ms remaining as a timeout and logs one line per
  attempt. On K>1 it keeps the forward socket on the first timeout of a step
  and reconnects only on the second timeout or for a suspect peer.
- Stage 0 (patch 7) keeps re-sending past CONNEXTE_RECV_RETRIES while the
  downstream peer is healthy, up to CONNEX_PP_STEP_BUDGET_S (60 s), and only
  then gives up the engine.
- Router lease CONNEX_HEARTBEAT_TIMEOUT_S was raised to 6 s (1 s heartbeats)
  at this point; fast handover later brought it down to 2 s (0.5 s
  heartbeats) with a TCP probe before any failover (see
  reprefill-on-failover.md).

Repro: benchmarks/k8s/churn_matrix.sh soft|hard OUT
"k4:CONNEX_PP_INFLIGHT=4,CONNEX_PP_DOWNSTREAM_GRAPHS=1" "4.0:2"; grep the
stage logs for "no reply for step" and "duplicate step".

Results after the fix (4 stages x 2
replicas, churn every 60 s, goodput / mean / P99 e2e):

| config | QPS 2 | QPS 4 | QPS 8 |
|---|---|---|---|
| K=4 soft | 100% / 1.60 s / 5.6 s | 100% / 2.14 s / 7.4 s | 100% / 2.80 s / 11.1 s |
| K=4 hard | 99.6% / 1.61 s / 5.9 s | 99.8% / 1.98 s / 7.2 s | 89.5% / 2.44 s / 9.3 s |
| K=1 soft | n/a | 100% / 2.92 s / 10.5 s | n/a |
| K=1 hard | n/a | 99.1% / 2.96 s / 10.6 s | n/a |

Before the fix, K=4 goodput was 37/23/28% soft and 43/28/80% hard. The
QPS 8 hard losses are client connection errors to the killed replica (K=1
was at 89% there too); most of them were a harness bug (the churn driver
put the dead replica back into the endpoints file at exit), and with it
fixed hard churn at QPS 8 is 99.2% (docs/results.md). Steady-state K=4 at QPS 8 without churn: 100%,
mean 2.57 s, P99 9.9 s (it was 2.61 s and 10.0 s before).

The front-router is now pinned to a pipeline node (`--router-node`). When it was
scheduled on a node outside the pipeline, the host-network pods could not open TCP connections to
it, every stage logged "Registration failed: epc-api: no reply to
'register'" and stage 0 died with "Peer stage 1 not found".

## Expected gain once stage 0 is done

Steady-state decode throughput up to min(K, 4)x for 4 balanced stages when
there are enough concurrent requests to fill K disjoint microbatches (each
microbatch is smaller, so per-step efficiency drops; realistic 2.5-3.5x at
K=4). Per-request TPOT does not improve (it may rise slightly).
With only the current patches the gain is about zero.

## How to reproduce the tests

    VLLM_FORK_DIR=/path/to/vllm-fork python -m pytest -q -p no:cacheprovider tests/test_pipelining.py tests/test_stage0_inflight.py tests/test_downstream_graphs.py
