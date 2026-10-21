# SGLang adapter: where Connex plugs in, what is missing, plan

Studied: SGLang main at commit `14ad64c411eb22b46da051135f622a1e28fb9b73`
(2026-10-18); paths below are relative to `python/sglang/srt/`. The
installed release 0.5.11 has the same `GroupCoordinator` signatures (checked
by `tests/test_sglang_adapter.py` against whichever source is present).

## 1. How SGLang does pipeline parallelism today

**A ring of replicated schedulers.** With `pp_size > 1` every PP rank runs a
full `Scheduler` and `SchedulerPPMixin.event_loop_pp()`
(`managers/scheduler_pp_mixin.py`). Each rank makes the same scheduling
decision on the same requests:

* rank 0 reads requests from the tokenizer (ZeroMQ) and relays the list to
  rank 1, which relays it on (`_pp_send_pyobj_to_next_stage` /
  `_pp_recv_pyobj_from_prev_stage`, pickled through
  `utils/common.py::point_to_point_pyobj` on the world gloo group);
* every rank runs `get_next_batch_to_run` locally;
* hidden states go forward as `"proxy"` dicts
  (`_pp_send_proxy_to_next_stage` -> `pp_group.send_tensor_dict(..., all_gather_group=attn_tp_group, async_send=True)`);
* the last rank's sampled tokens (`_pp_prepare_tensor_dict`:
  `next_token_ids`, logprobs, spec-decode fields) go to rank 0 and are then
  relayed around the whole ring as `"output"` dicts, so every rank runs
  `process_batch_result` itself and keeps identical request state.

Messages are typed by a `"__msg_type__"` key; `_pp_recv_typed_dict` stashes
messages of the other kind in an inbox. Even ranks send first and odd ranks
receive first to avoid an NCCL deadlock.

**Microbatches = K in flight.** `init_pp_loop_state` creates
`pp_loop_size = pp_size + pp_async_batch_depth` slots, each with its own
running batch (`running_mbs[mb_id]`); the loop cycles through them, so up to
`pp_loop_size` microbatches are in the pipe. Results are consumed in slot
order.

**Model side.** `ModelRunner.forward(forward_batch, ..., pp_proxy_tensors)`;
non-first ranks read `hidden_states` / `residual` from the
`PPProxyTensors` (`model_executor/forward_batch_info.py`), non-last ranks
return a `PPProxyTensors`. Layers per rank come from
`distributed/utils.py::get_pp_indices(num_layers, pp_rank, pp_size)`
(override: `SGLANG_PP_LAYER_PARTITION`); missing layers are `PPMissingLayer`.

**KV ownership.** Every rank has its own `req_to_token_pool`,
`token_to_kv_pool_allocator` and radix `tree_cache` for its own layers.
There is no exchange of slot indices: they agree because the schedulers are
deterministic replicas (same request stream, same outputs, same seed, pool
size from a MIN all-reduce at start-up). Nothing checks this at run time
outside the disaggregation paths.

**Retract / recompute.** `ScheduleBatch.retract_decode()` (when
`check_decode_mem` fails) frees requests, calls `Req.reset_for_retract()`
and re-queues them; re-admission recomputes `origin_input_ids + output_ids`
as a normal prefill (with radix-tree prefix reuse). Aborts arrive through
the same relayed request stream. Each rank retracts independently, which
again relies on identical pools.

## 2. What SGLang lacks for elastic external PP

1. **Fixed process groups.** `init_distributed_environment` /
   `initialize_model_parallel` build `pp_group` and `world_group` once;
   neighbours are `(pp_rank +- 1) % pp_size` ranks. No join, leave or
   replacement.
2. **Ring dependency.** Outputs travel through every stage, so losing any
   stage stalls all of them, and a stage cannot be served by a different
   replica of its neighbour.
3. **Replicated scheduler state.** A replacement stage would need the full
   scheduler state of its predecessor (running microbatches, radix tree,
   pools, RNG). This is exactly what elastic PP wants to avoid.
4. **No per-request KV accounting across stages.** Nothing tells a stage
   that it is asked to decode a request it holds no KV for (the silent
   wrong-output failure Connex's handover check exists for).
5. **Static layer split and pool sizing** at start-up (`get_pp_indices`, the
   MIN all-reduce of free memory).
6. **Ordering assumptions** of NCCL/gloo point-to-point (parity ordering,
   CUDA-graph fences on the send stream).

## 3. Where Connex plugs in

All scheduler-side PP traffic goes through `scheduler.pp_group`
(`send_tensor_dict` / `recv_tensor_dict`, plus `is_first_rank`,
`is_last_rank`, `next_rank`, `prev_rank`, `world_size`) and the three
pyobj helpers. So the communication side is a drop-in object:
`connex.adapters.sglang.ConnexPPGroup`, a `GroupCoordinator`-compatible
group on top of a `connex.pp` endpoint. The topology changes to the one
Connex supports (and the vLLM fork uses):

* **stage 0 is the only scheduler.** Its `send_tensor_dict(proxy)` starts a
  step (`Stage0Link.submit`: correlation id, `request_ids`, `cx_ctx`,
  `finished_req_ids`); its `recv_tensor_dict()` returns the `"output"` of
  the oldest outstanding step, re-sending on timeout and calling the
  engine's `reprefill` for `cx_missing` (rows listed under
  `__cx_reprefilled__` must be dropped from that result);
* **stages 1.. are executors without a scheduler.** `recv_tensor_dict()`
  returns the next step's `"proxy"` dict after the KV check (rows lacking KV
  are released and listed under `__cx_missing_here__`); the following
  `send_tensor_dict()` answers that step, forward or, on the last stage, as
  the `"output"` back to the originating stage 0;
* `send_object` / `recv_object` raise: requests are not relayed.

## 4. Plan

| # | piece | where | status |
|---|---|---|---|
| 1 | `ConnexPPGroup` (pp_group over Connex, typed proxy/output, K in flight, retries, failover, handover) | `connex/adapters/sglang/pp_group.py` | **done**, unit tested with fakes and with real Connex endpoints on localhost (`tests/test_sglang_adapter.py`) |
| 2 | `make_pp_group()` from environment | `connex/adapters/sglang/setup.py` | done, not run inside SGLang |
| 3 | external stage mode: launch each stage as its own server with `pp_size=1`, load only layers `[start, end)` given by flags (patch `get_pp_indices` / `make_layers`), embed on stage 0, norm + lm_head on the last stage | SGLang engine patch | not started |
| 4 | stage-0 loop: `event_loop_pp`-like loop with K slots but sends only proxies and receives only outputs (no pyobj relay, no output relay); `row_info()` = (rids, `req.kv_committed_len` / `len(fill_ids) - extend_len`) per row of the batch | SGLang scheduler patch | not started |
| 5 | executor loop on stages 1..: receive proxy + batch metadata, build a `ForwardBatch` without a scheduler (positions, seq lens, `out_cache_loc` from a local per-request allocator), run layers, send | new executor (analogue of the fork's `execute_external_pipeline_step`) | not started; the largest piece |
| 6 | per-request KV allocator on executors (`kv_held`, `release`, TTL/LRU reclaim, finished ids), the analogue of `ConnexPPBlockAllocator` | executor | not started; `KVLedger` covers the accounting, not the slot allocation |
| 7 | `reprefill(rids)` on stage 0: for running requests, free the stage-0 KV, `reset_for_retract()`, re-queue as retracted (recompute prompt + output ids); skip the token sampled in the failed step; `abort` via `FINISH_ABORT` | SGLang scheduler patch | not started |
| 8 | batch metadata in the step dict (seq lens, extend lens, positions, sampling info needed on the last stage) | stage 0 + executor | not started |
| 9 | attention-TP inside a stage: Connex sends full tensors from one rank; TP peers of a stage need a local broadcast | executor | not started |

Out of scope for the first version: speculative decoding over PP
(`spec_*` fields), PD disaggregation PP paths, overlap scheduling.

## 5. What is claimed and what is not

Implemented and tested without GPUs: the pp_group object, its typed message
handling, K-slot ordering, retry and failover behaviour, `cx_ctx` /
`cx_missing` round trip to an engine hook, and the interface match with
SGLang's `GroupCoordinator` signatures (parsed from source, SGLang never
imported). Not done: everything in rows 3-9. SGLang has **not** been run on
Connex end to end; no throughput or correctness claim is made for it.
