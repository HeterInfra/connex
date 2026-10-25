# EP recovery and elasticity: what we learned

Notes behind `connex.ep` (expert parallelism), with the reasoning and how to
reproduce each observation. Numbers: [`results.md`](../results.md#10-expert-parallelism-moe).

## 1. The engine already has the right hook: EPLB's expert maps

vLLM's EP routes every token through `logical_to_physical_map` and
`logical_replica_count` (one replica chosen per token by a hash of the token
index), read by a Triton kernel each step. EPLB writes these maps when it
rebalances. That is exactly the indirection a mobility control plane needs:
Connex writes the same tensors from its placement table, in place (CUDA
graphs keep their pointers), inside a `collective_rpc` that vLLM executes on
every worker between two steps. No engine patch is needed. The only model
gap was OLMoE, whose MoE layer is built without EPLB; `connex.ep.vllm_ext`
adds vLLM's `MixtureOfExperts` interface to it at import time (worker
extension), and with `--enable-eplb` but without it vLLM silently runs
without redundant experts (check `Local/global number of experts` in the
log, and the weight memory: 6.7 GiB per rank instead of 13 GiB).

Reproduce: start OLMoE with `--enable-eplb` and `num_redundant_experts=64`
with and without `--worker-extension-cls connex.ep.vllm_ext.ConnexEPWorker`
and compare the logged weight memory.

## 2. Redundant experts are not free unless they are standby

With 64 redundant experts on 2 ranks (every expert on both ranks) decode
throughput dropped by about a third against vLLM's default EP, and by a
sixth with 32. EPLB with 0 redundant experts plus the Connex extension
matched the default, so the cost is the replicas, not the control path. Why:
vLLM hashes every token over all replicas, so in each decode step both copies
of a popular expert are read from HBM, and small-batch MoE decode is bound
by weight reads. Routing at most one replica per expert
(`PlacementTable.active_replicas = 1`) and keeping the others as warm
standby restores the default's throughput. The standby copies still pay
off: when a rank drains or fails they become routable in the next epoch with
no weight movement.

Reproduce: `ep_bench.py run` against servers with `num_redundant_experts`
0 / 32 / 64, then 64 with `--active-replicas 1`.

## 3. Copy in the background, never inside the step loop

A `collective_rpc` blocks the engine loop while it runs. Copying 12.9 GB of
expert weights inside it (a full rejoin of one rank) stalled every stream
for 1.6 s. The copy is safe to run concurrently because its destination
slots are unroutable in the current epoch, so the worker extension starts it
in a background thread (one-sided reads land directly in the slots, local
copies on a side stream) and returns; the controller polls and commits the
next epoch when every rank is done. The stall disappears; the copy itself
takes longer because it shares PCIe and HBM with serving.

## 4. vLLM's padded replica dimension made every cutover slow

`logical_to_physical_map` is `[layers, experts, MAX_EXPERT_REDUNDANCY + 1]`
(1024 wide). Building that from Python lists for each epoch took 80-90 ms
per worker, all of it inside the step loop. Writing only the real width
(replicas per expert) and filling the rest with -1 on the device once brings
the worker-side apply to a few milliseconds; most of what remains is the
HTTP `collective_rpc` round trip.

## 5. In-group rank death is the engine's boundary

A vLLM EP group shares collectives every step; when one rank dies the group
stops, and vLLM's elastic EP only rescales on request. So a dead rank is
handled one level up: each EP group is an endpoint behind the resume proxy,
in-flight requests resume on another group from the tokens they already
have, and `EPGroupSupervisor` rebuilds and re-adds the group. Two details
dominated recovery time:

* vLLM's API server keeps answering `/health` for about 10 s after a worker
  was killed. Watching the rank processes directly (a killed worker is a
  zombie until its parent reaps it, so check `/proc/<pid>/stat`, not
  `kill(pid, 0)`) detects the death within one probe interval.
* The proxy only noticed the dead streams after its idle timeout (2.5 s).
  Removing an endpoint from the endpoints file now marks it dead, which
  closes its streams at once and resumes them elsewhere.

SGLang behaves the same way: one dead scheduler process takes down the
server. SGLang's elastic EP (Mooncake backend) is the in-group alternative
(it masks dead ranks out of the all-to-all); it needs Mooncake EP kernels
built for the installed torch, which the test environment did not have.

## 6. SGLang: same maps, a different cutover point

SGLang's `ExpertLocationMetadata` plays the role of vLLM's EPLB maps: each
MoE layer's top-k reads views of `logical_to_all_physical_map` and its
`num_valid` counts, so writing them in place reroutes tokens, inside CUDA
graphs too. What SGLang lacks is vLLM's `collective_rpc` from HTTP: its RPC
socket is only reachable from the offline `Engine`, and the HTTP control
requests have fixed types. The cutover point is the scheduler loop itself:
at the top of every iteration TP rank 0 pulls new requests (plus the
timeout aborts it decided locally) and broadcasts the list to the other TP
ranks. The `connex_ep` plugin appends queued control commands to rank 0's
local list, so every rank sees each command in the same iteration, before
the next batch is scheduled; a TP-CPU-group all-gather returns the
per-rank results to rank 0's HTTP endpoint. No new collective runs in
iterations without a command.

Two SGLang details mattered:

* **The overlap scheduler.** Input handling for batch N+1 runs while batch N
  is still on the GPU, so a map write there could race with batch N. The
  first version synchronized the device before and after the write, which
  blocked the scheduler for one decode step (22 ms) per epoch. The writes
  are now ordered on the GPU: the scheduler thread runs on SGLang's
  schedule stream, which waits for the forward stream before the writes,
  and every forward waits for the schedule stream anyway. A move into a
  slot the epoch staged out waits for an event recorded after the writes
  (in its background thread, so nothing blocks the loop).
* **Replicas were silently unused.** SGLang's Qwen2-MoE passes the dispatch
  info to `TopK` only on its DeepEP path; without an all-to-all backend the
  top-k ids stayed logical and were used as physical slot ids. With 60
  redundant experts every token went to slots 0-59, all on EP rank 0, and
  decode ran a third slower than the default EP even with
  `active_replicas=1`. A per-rank kernel profile shows it at once
  (`fused_moe_kernel`: 1,027 ms on rank 0, 14 ms on rank 1 over 30 decode
  steps). The plugin wraps `TopK.forward_cuda` to build the dispatch info
  from the layer id when the caller passed none.

Reproduce: start Qwen1.5-MoE with `--ep 2 --ep-num-redundant-experts 60
--ep-dispatch-algorithm dynamic` without `CONNEX_EP=1`, run `ep_bench.py run
--engine sglang`, then `POST /start_profile` with `num_steps` and sum the
kernel times per rank in the two traces.

**Attaching under load stalled serving.** The first controller attach
stalled every running request for 0.3-2.3 s even after the registration
moved to a background thread. Timing the parts in the server showed that
the registration itself is short (130-250 ms for 24.9 GB) and creating the
connex-rs endpoint is not (0.2 s on rank 0, 1.1 s on rank 1), and the
constructor holds the interpreter, so the scheduler thread cannot run.
A standalone thread test does not reproduce the full stall, so test it in
the server. The endpoint is now created when the scheduler starts. Reproduce:
attach with the in-process transport (`SGLangEPEngine(..., transport="local")`):
no stall; with connex-rs created at attach time: the stall is back.

**Scoring on SGLang.** Its default kernels are not batch invariant, so a
teacher-forced score flips 0-2 near-ties per 32,768 tokens even in runs
without any event, at random positions. To show that an event never routes
a token to a wrong or half-written slot, run both the experiment and the
reference with `--enable-deterministic-inference`: then every run scores 0,
including the drain whose drained slots are NaN.

**Health.** SGLang's `/health` generates one token and answers after at
least a second, so a 1 s probe marks every group down. Use
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` for the supervisor and rely on
the rank process watch for death.

## 7. Operating note: GPU lock and orphaned servers

Experiments share GPUs through `flock /tmp/connex-gpu.lock`. A server started
by a locked script inherits the lock's file descriptor; when the script is
killed but the server survives (its own session), the orphan keeps both the
GPUs and the lock. Use `flock -o` (close the descriptor before running the
command) and kill servers by process group. Never clean up with a pattern
kill (`pkill -f benchmarks/ep/`): it also kills every other experiment
driver whose command line matches, including ones of a parallel session.
