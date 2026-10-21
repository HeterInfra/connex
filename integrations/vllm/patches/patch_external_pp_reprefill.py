# SPDX-License-Identifier: Apache-2.0
"""Patch 14 (correctness, all stages): re-prefill in-flight requests whose KV
a downstream stage does not hold (CONNEX_PP_REPREFILL, default on).

    python patch_external_pp_reprefill.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Edits the runner (the path given) and ../core/sched/scheduler.py (skipped with
a message when absent). Needs patches 1, 7, 12 (allocator, stage-0 inflight,
finished ids); runs before patch 10 (downstream graphs, kept last).

Bug: KV is per request on every stage. When a downstream replica dies or
leaves, its upstream neighbour fails over to a surviving replica of that
stage (sticky peer) and stage 0 re-sends the step. The new replica never saw
the requests of that step: it decodes them against an empty context (the
allocator creates a fresh request, the block table points at the null block)
and returns plausible but wrong hidden states. Nothing told stage 0. The same
happens after a replica restart, when the allocator reclaimed a request
(TTL/LRU), and when a stage-0 prefix-cache hit reaches a downstream replica
that does not hold that prefix.

Fix (pure logic in external_pp_reprefill.py, inlined):
  * stage 0 sends "cx_ctx" (num_computed_tokens per request) with every step;
  * stages 1+ (ConnexPPKVGuard) compare it with the context they hold
    (tokens they wrote for a request still live in the allocator, or computed
    prefix-cache blocks). Missing requests are released and blanked (no token
    ids, so the step writes their garbage only to the null block / a private
    block and marks nothing computed), and reported in "cx_missing", which
    later stages union and forward and the last stage returns;
  * stage 0 attaches the ids to its ModelRunnerOutput (cx_reprefill_req_ids);
    Scheduler.update_from_output drops the token sampled for them this step and
    rewinds num_computed_tokens to 0 (the request stays RUNNING and keeps its
    stage-0 blocks: the stock KV-load-failure recovery pattern). The next
    schedule() gives it prompt + generated tokens as a (possibly chunked)
    prefill, the runner truncates the dropped token (stock "discarded output
    tokens" alignment), and decoding continues from exactly where it was;
  * a request rewound more than CONNEX_PP_REPREFILL_MAX times (default 8) is
    finished with finish_reason=abort instead (clean failure, never a silent
    wrong decode);
  * stage 0 classifies a multi-token step of a request past its prompt (a
    later chunk of a re-prefill) as prefill, so token ids go downstream.
CONNEX_PP_REPREFILL=0 disables sending cx_ctx (stage 0) and the check
(stages 1+); old peers ignore the keys.
"""
import os
import sys

MARK = "# connex: re-prefill on failover (CONNEX_PP_REPREFILL)"
S_MARK = "# connex: re-prefill on failover, scheduler (CONNEX_PP_REPREFILL)"
FLAG_ANCHOR = '_CONNEX_PP_DEBUG = os.environ.get("VLLM_EXTERNAL_PP_DEBUG", "") == "1"\n'
MOD_ANCHOR = "# END connex external-pp engine fixes\n"
MOD_BEGIN = "# BEGIN connex external-pp reprefill"
MOD_END = "# END connex external-pp reprefill"


def _module_block() -> str:
    me = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "external_pp_reprefill.py")).read()
    return me[me.index(MOD_BEGIN):me.index(MOD_END)] + MOD_END + "\n"


R_EDITS = [
    # flag
    (FLAG_ANCHOR, FLAG_ANCHOR + MARK + "\n"
     '_CONNEX_PP_REPREFILL = os.environ.get("CONNEX_PP_REPREFILL", "1") != "0"\n'),
    # stage 0: a multi-token step past the prompt (later re-prefill chunk) is a prefill
    ("                    if req_state.num_computed_tokens < req_state.num_prompt_tokens:\n"
     "                        prefill_req_ids.add(rid)\n",
     "                    if req_state.num_computed_tokens < req_state.num_prompt_tokens or (\n"
     "                            _CONNEX_PP_REPREFILL  # connex: re-prefill chunk past the prompt\n"
     "                            and scheduler_output.num_scheduled_tokens.get(rid, 0) > 1\n"
     "                            and not scheduler_output.scheduled_spec_decode_tokens.get(rid)):\n"
     "                        prefill_req_ids.add(rid)\n"),
    # stage 0: context per request on every step
    ("                # v1.7.4.8 OPTIMIZATION: Only send token IDs in prefill phase\n",
     "                if _CONNEX_PP_REPREFILL:  # connex: re-prefill on failover\n"
     "                    forward_tensor_dict[\"cx_ctx\"] = [\n"
     "                        (self.requests[r].num_computed_tokens if r in self.requests else 0)\n"
     "                        for r in req_ids]\n"
     "                # v1.7.4.8 OPTIMIZATION: Only send token IDs in prefill phase\n"),
    # stage 0 K=1: reply
    ("                # Record TTFT metric for prefill steps and update TPOT tracking.\n",
     "                self._cx_reprefill_ids = return_tensor_dict.get(\"cx_missing\") or None\n"
     "                # Record TTFT metric for prefill steps and update TPOT tracking.\n"),
    # stage 0 K>1: reply
    ("            final_hidden_states = ret[\"hidden_states\"]\n",
     "            self._cx_reprefill_ids = ret.get(\"cx_missing\") or None\n"
     "            final_hidden_states = ret[\"hidden_states\"]\n"),
    # stage 0: hand the ids to the scheduler with this step's output
    ("                num_nans_in_logits=num_nans_in_logits,\n            )\n",
     "                num_nans_in_logits=num_nans_in_logits,\n            )\n"
     "            _cx_rp = getattr(self, \"_cx_reprefill_ids\", None)\n"
     "            if _cx_rp:  # connex: re-prefill on failover\n"
     "                self._cx_reprefill_ids = None\n"
     "                output.cx_reprefill_req_ids = set(_cx_rp)\n"
     "                logger.warning(\"Stage 0: %d request(s) lack downstream KV; \"\n"
     "                               \"re-prefilling them\", len(_cx_rp))\n"),
    # stages 1+: check before metadata / allocation
    ("        num_scheduled_tokens_per_req = tensor_dict.get(\"num_scheduled_tokens_per_req\")\n",
     "        num_scheduled_tokens_per_req = tensor_dict.get(\"num_scheduled_tokens_per_req\")\n"
     "        " + MARK + "\n"
     "        _cx_ctx = tensor_dict.get(\"cx_ctx\") if _CONNEX_PP_REPREFILL else None\n"
     "        _cx_miss = []\n"
     "        if _cx_ctx is not None and request_ids:\n"
     "            _cx_g = getattr(self, \"_cx_kv_guard\", None)\n"
     "            if _cx_g is None:\n"
     "                _cx_g = self._cx_kv_guard = ConnexPPKVGuard(connex_pp_hash_block)\n"
     "            _cx_miss = _cx_g.check(getattr(self, \"_connex_pp_alloc\", None), request_ids,\n"
     "                                   _cx_ctx, step_type, all_token_ids,\n"
     "                                   self.cache_config.block_size,\n"
     "                                   num_scheduled_tokens_per_req)\n"
     "            if _cx_miss:\n"
     "                if all_token_ids is not None:\n"
     "                    all_token_ids = connex_pp_blank_rows(all_token_ids, request_ids, _cx_miss)\n"
     "                logger.warning(\n"
     "                    \"External PP stage %d: %d/%d request(s) of a %s step have no KV here \"\n"
     "                    \"(failover/restart/reclaim); asking stage 0 to re-prefill (total %d)\",\n"
     "                    get_pp_group().rank_in_group, len(_cx_miss), len(request_ids),\n"
     "                    step_type, _cx_g.stats[\"missing\"])\n"
     "        _cx_miss_all = connex_pp_merge_missing(tensor_dict.get(\"cx_missing\"), _cx_miss)\n"),
    # stages 1+: commit after the forward, before marking prefix blocks computed
    ("        # CRITICAL: After successful prefill, mark blocks as computed\n",
     "        if _cx_ctx is not None and request_ids:\n"
     "            self._cx_kv_guard.commit(getattr(self, \"_connex_pp_alloc\", None), request_ids,\n"
     "                                     _cx_ctx, step_type, num_scheduled_tokens_per_req,\n"
     "                                     _cx_miss)\n"
     "        # CRITICAL: After successful prefill, mark blocks as computed\n"),
    # last stage: return the ids
    ("            return return_tensor_dict\n",
     "            if _cx_miss_all:\n"
     "                return_tensor_dict[\"cx_missing\"] = _cx_miss_all\n"
     "            return return_tensor_dict\n"),
    # intermediate stages: forward ctx and ids
    ("            if _cx_fin:\n                forward_tensor_dict[\"finished_req_ids\"] = _cx_fin\n",
     "            if _cx_fin:\n                forward_tensor_dict[\"finished_req_ids\"] = _cx_fin\n"
     "            if _cx_ctx is not None:\n"
     "                forward_tensor_dict[\"cx_ctx\"] = _cx_ctx\n"
     "            if _cx_miss_all:\n"
     "                forward_tensor_dict[\"cx_missing\"] = _cx_miss_all\n"),
]

S_EDITS = [
    ('_CX_PP_INFLIGHT = max(1, int(_cx_os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))\n',
     '_CX_PP_INFLIGHT = max(1, int(_cx_os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))\n'
     + S_MARK + "\n"
     '_CX_PP_REPREFILL_MAX = int(_cx_os.environ.get("CONNEX_PP_REPREFILL_MAX", "8") or "8")\n'),
    ("        stopped_running_reqs: set[Request] = set()\n",
     "        stopped_running_reqs: set[Request] = set()\n"
     "        _cx_rp = getattr(model_runner_output, \"cx_reprefill_req_ids\", None)\n"),
    ("            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:\n",
     "            if _cx_rp and req_id in _cx_rp:\n"
     "                # connex: a downstream stage lacked this request's KV; the\n"
     "                # token sampled now is garbage. Rewind and re-prefill.\n"
     "                self._cx_reprefill_one(req_id, outputs, stopped_running_reqs)\n"
     "                continue\n"
     "            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:\n"),
    ("    def _update_request_with_output(\n",
     '''    def _cx_reprefill_one(self, req_id, outputs, stopped_running_reqs) -> None:
        """connex: drop this step's token of `req_id` and recompute its KV on
        the current downstream chain (num_computed_tokens -> 0; the request
        stays RUNNING and keeps its blocks, like the KV-load-failure path).
        Past CONNEX_PP_REPREFILL_MAX rewinds the request is aborted."""
        request = self.requests.get(req_id)
        if request is None or request.status != RequestStatus.RUNNING:
            return
        cnt = getattr(self, "_cx_rp_counts", None)
        if cnt is None or len(cnt) > 65536:
            cnt = self._cx_rp_counts = {}
        n = cnt[req_id] = cnt.get(req_id, 0) + 1
        self._cx_rp_total = getattr(self, "_cx_rp_total", 0) + 1
        if n > _CX_PP_REPREFILL_MAX:
            cnt.pop(req_id, None)
            logger.error("connex: request %s lost its downstream KV %d times; "
                         "aborting it", req_id, n)
            request.status = RequestStatus.FINISHED_ABORTED
            self._free_request(request)
            stopped_running_reqs.add(request)
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                )
            )
            return
        request.num_computed_tokens = 0

    def _update_request_with_output(\n'''),
]


def patch_runner(src: str):
    if MARK in src:
        return src, "already patched"
    for old, _ in R_EDITS:
        if src.count(old) != 1:
            return None, f"anchor not found/unique ({src.count(old)}x): {old.strip()[:60]!r}"
    if src.count(MOD_ANCHOR) != 1:
        return None, "engine-fixes block not found (needs patch_external_pp_finished_ids.py)"
    if "def _cx_finish_step" not in src:
        return None, "stage-0 inflight not applied (needs patch_external_pp_stage0_inflight.py)"
    for old, new in R_EDITS:
        src = src.replace(old, new)
    src = src.replace(MOD_ANCHOR, MOD_ANCHOR + "\n\n" + _module_block(), 1)
    return src, "patched"


def patch_scheduler(src: str):
    if S_MARK in src:
        return src, "already patched"
    for old, _ in S_EDITS:
        if src.count(old) != 1:
            return None, f"anchor not found/unique ({src.count(old)}x): {old.strip()[:60]!r}"
    for old, new in S_EDITS:
        src = src.replace(old, new)
    return src, "patched"


def scheduler_path(runner_path: str) -> str:
    v1 = os.path.dirname(os.path.dirname(os.path.abspath(runner_path)))
    return os.path.join(v1, "core", "sched", "scheduler.py")


def main(argv) -> int:
    runner = argv[1]
    msgs, rc = [], 0
    for path, fn, name in ((runner, patch_runner, "runner"),
                           (scheduler_path(runner), patch_scheduler, "scheduler.py")):
        if name != "runner" and not os.path.exists(path):
            msgs.append(f"{name} skipped (not found)")
            continue
        src = open(path).read()
        new, msg = fn(src)
        if new is None:
            rc = 1
        elif new != src:
            open(path, "w").write(new)
        msgs.append(f"{name} {msg}")
    print(f"{os.path.basename(__file__)}: {'FAILED ' if rc else ''}{'; '.join(msgs)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
