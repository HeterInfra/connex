# SPDX-License-Identifier: Apache-2.0
"""Patch 12 (memory, all stages): forward finished request ids downstream.

    python patch_external_pp_finished_ids.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Bug (fork 380186d57): only stage 0 sees scheduler_output.finished_req_ids
(_update_states). Stages 1+ run execute_external_pipeline_step on the received
tensor_dict, which carries no finish information, so their per-request KV
blocks (ConnexPPBlockAllocator, patch 1) are only reclaimed after
CONNEX_PP_REQ_TTL_S (120 s) unseen, or LRU-first once the pool is exhausted:
live_reqs grows with the request rate x TTL and the prefix cache is evicted
by dead requests.

Fix:
  * stage 0 (_update_states) puts finished ids into a ConnexPPFinishedOutbox;
    the next send (K=1 and K>1, before the retry loops) attaches them as the
    non-tensor field "finished_req_ids" (each id on 2 consecutive sends,
    CONNEX_PP_FINISHED_REPEAT). A retried step re-sends the same dict, so
    ids are never dropped with it.
  * stages 1+ apply them with ConnexPPFinishTracker before the step's own
    allocation (release now, or defer while the step itself references the
    id; a re-sent older step that re-creates a finished request is released
    again at the next step) and intermediate stages forward them.
  * TTL/LRU reclamation stays as the fallback.
CONNEX_PP_SEND_FINISHED=0 disables sending (stage 0); old peers ignore the key.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_pp_engine_fixes import inline_module, run_patch  # noqa: E402

MARK = "# connex: finished_req_ids piggybacking (CONNEX_PP_SEND_FINISHED)"
FLAG_ANCHOR = '_CONNEX_PP_DEBUG = os.environ.get("VLLM_EXTERNAL_PP_DEBUG", "") == "1"\n'
FLAGS = ('_CONNEX_PP_SEND_FINISHED = os.environ.get("CONNEX_PP_SEND_FINISHED", "1") != "0"\n'
         '_CONNEX_PP_FINISHED_REPEAT = int(os.environ.get("CONNEX_PP_FINISHED_REPEAT", "2") or "2")\n')


def _ins_before(anchor, text):
    return (anchor, text + anchor)


def _ins_after(anchor, text):
    return (anchor, anchor + text)


EDITS = [
    _ins_after(FLAG_ANCHOR, FLAGS),
    # stage 0: collect
    _ins_before(
        "        # Remove finished requests from the cached states.\n"
        "        for req_id in scheduler_output.finished_req_ids:\n",
        "        " + MARK + "\n"
        "        if (_CONNEX_PP_SEND_FINISHED and scheduler_output.finished_req_ids\n"
        "                and self.parallel_config.pipeline_stage_mode == \"external\"\n"
        "                and get_pp_group().is_first_rank):\n"
        "            _cx_ob = getattr(self, \"_cx_fin_outbox\", None)\n"
        "            if _cx_ob is None:\n"
        "                _cx_ob = self._cx_fin_outbox = ConnexPPFinishedOutbox(\n"
        "                    _CONNEX_PP_FINISHED_REPEAT)\n"
        "            _cx_ob.add(scheduler_output.finished_req_ids)\n"),
    # stage 0: attach to the step about to be sent (both K=1 and K>1 paths)
    _ins_before(
        "                _debug_pp = _CONNEX_PP_DEBUG\n",
        "                _cx_ob = getattr(self, \"_cx_fin_outbox\", None)\n"
        "                if _cx_ob is not None and len(_cx_ob):\n"
        "                    forward_tensor_dict[\"finished_req_ids\"] = _cx_ob.take()\n"),
    # stages 1+: apply before this step's allocation
    _ins_after(
        "        request_ids = tensor_dict.get(\"request_ids\", [])\n",
        "        " + MARK + "\n"
        "        _cx_fin = tensor_dict.get(\"finished_req_ids\") or None\n"
        "        _cx_ft = getattr(self, \"_cx_fin_tracker\", None)\n"
        "        if _cx_fin is not None and _cx_ft is None:\n"
        "            _cx_ft = self._cx_fin_tracker = ConnexPPFinishTracker()\n"
        "        if _cx_ft is not None:\n"
        "            _cx_alloc = getattr(self, \"_connex_pp_alloc\", None)\n"
        "            _cx_ft.on_step(_cx_alloc, _cx_fin, request_ids)\n"
        "            if _cx_fin and _cx_ft.stats[\"notices\"] - getattr(_cx_ft, \"_logged\", 0) >= 256:\n"
        "                _cx_ft._logged = _cx_ft.stats[\"notices\"]\n"
        "                logger.info(\n"
        "                    \"External PP stage %d: finished notices=%d released=%d deferred=%d \"\n"
        "                    \"revived=%d live_reqs=%d free=%d\",\n"
        "                    get_pp_group().rank_in_group, _cx_ft.stats[\"notices\"],\n"
        "                    _cx_ft.stats[\"released\"], _cx_ft.stats[\"deferred\"],\n"
        "                    _cx_ft.stats[\"revived\"],\n"
        "                    len(_cx_alloc.reqs) if _cx_alloc is not None else 0,\n"
        "                    _cx_alloc.num_free() if _cx_alloc is not None else -1)\n"),
    # intermediate stages: forward
    _ins_before(
        "            # OPTIMIZATION: Only pass all_token_ids in prefill phase (for prefix caching)\n",
        "            if _cx_fin:\n"
        "                forward_tensor_dict[\"finished_req_ids\"] = _cx_fin\n"),
]


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    for old, new in EDITS:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
        src = src.replace(old, new)
    src = inline_module(src)
    if src is None:
        return None, "allocator block not found (needs patch_external_pp_allocator.py)"
    return src, "patched"


if __name__ == "__main__":
    sys.exit(run_patch(__file__, patch, sys.argv))
