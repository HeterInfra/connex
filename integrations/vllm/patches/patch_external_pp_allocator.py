# SPDX-License-Identifier: Apache-2.0
"""Patch 1 (correctness): refcounted block allocator for external-PP Stage-1+.

Problem (fork 380186d57, _allocate_blocks_with_prefix_cache_for_external_pp):
every prefill gives *every* slot up to max_num_blocks_per_seq a block keyed by
f"padding_{req_id}_{j}". That is O(num_reqs x max_blocks) string/dict/allocation
work per prefill, the keys are never evicted (_external_pp_prefix_cache grows
forever), and the block ring (_external_pp_next_block_id) wraps after
num_blocks / max_blocks requests and hands out blocks that live requests are
still reading/writing -> silent KV corruption under sustained load. The decode
path additionally caches a cloned GPU block table per tuple(request_ids) that is
never evicted on downstream stages (they never see finished_req_ids).

Mechanism: ConnexPPBlockAllocator (external_pp_alloc.py, inlined verbatim):
  * full prompt blocks are content-addressed with refcounts; unreferenced cached
    blocks sit in an LRU and are evicted only when the free list is empty;
  * the partial prompt block and decode growth get private blocks, allocated
    lazily one block at a time as the host-tracked position advances;
  * unreached slots are padded with block 0 (vLLM's null block) without
    allocating anything. Safe because attention reads stop at seq_len and
    slot_mapping only targets slots covering real positions, which are always
    allocated (we keep one block of headroom past the current position);
  * requests are released on finished_req_ids (stage 0 / if ever forwarded), after
    CONNEX_PP_REQ_TTL_S seconds unseen (default 120), or LRU-first on pool
    exhaustion (never a request in the current batch);
  * the per-batch GPU cache is gone: each step stacks the per-request numpy rows
    and does one H2D copy (no growth, no clone).
  * mark_computed only marks blocks fully covered by computed tokens
    (stage0 computed + scheduled), fixing chunked-prefill over-marking.
Expected gain: prefill block-table work drops from O(reqs x max_blocks) to
O(reqs x prompt_blocks); decode avoids the O(reqs) row loop on batch-change
misses; no GPU memory leak; no live-KV overwrite.

    python patch_external_pp_allocator.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
Idempotent; exit 0 if applied/already applied, 1 if anchors are not found.
"""
import os
import sys

MARK = "# connex: refcounted external-PP block allocator"


def _alloc_source() -> str:
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_pp_alloc.py")).read()
    b = src.index("# BEGIN connex external-pp allocator")
    e = src.index("# END connex external-pp allocator")
    return src[b:e] + "# END connex external-pp allocator\n\n\n"


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    edits = []
    # A. module-level allocator
    a = "_external_pp_send_executor: ThreadPoolExecutor | None = None\n"
    edits.append((a, f"{MARK}\n" + _alloc_source() + a))
    # B. state
    a = "        self._external_pp_computed_blocks: set[bytes] = set()\n"
    edits.append((a, a + "        self._connex_pp_alloc = None  # ConnexPPBlockAllocator, created lazily\n"))
    # C. release on finished requests
    a = "            # Clean up external PP request blocks for prefix caching\n"
    edits.append((a, "            if getattr(self, \"_connex_pp_alloc\", None) is not None:\n"
                     "                self._connex_pp_alloc.release(req_id)\n" + a))
    # D. host upload helper (patch_external_pp_pinned_upload.py swaps its body)
    a = "    def _compute_block_hashes_for_external_pp(\n"
    edits.append((a, "    def _connex_pp_upload_block_table(self, rows):\n"
                     "        # connex: block-table upload (pageable, synchronous)\n"
                     "        return torch.from_numpy(rows).to(self.device)\n\n" + a))
    # E. mark computed through the allocator
    a = "        as having their KV computed and stored in cache.\n        \"\"\"\n"
    edits.append((a, a + "        if getattr(self, \"_connex_pp_alloc\", None) is not None:\n"
                         "            self._connex_pp_alloc.mark_computed(request_ids, block_size)\n"
                         "            return\n"))
    # F. one allocator call per step, shared across KV cache groups
    a = "        # Build minimal block_table and slot_mapping\n"
    edits.append((a, "        _cx_rows = None\n" + a))
    for old, new in edits:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
        src = src.replace(old, new)

    # G. replace the decode batch cache + ring allocation region
    start = "                    # v1.7.4.10 OPTIMIZATION: Use batch block table cache for decode\n"
    end = "                else:\n                    # Fallback: assign sequential block IDs (non-persistent)\n"
    if src.count(start) != 1 or src.count(end) != 1 or src.index(start) > src.index(end):
        return None, "block-table region anchors not found"
    i, j = src.index(start), src.index(end)
    body = '''                    if self._connex_pp_alloc is None:
                        self._connex_pp_alloc = ConnexPPBlockAllocator(
                            total_kv_blocks,
                            ttl_s=float(os.environ.get("CONNEX_PP_REQ_TTL_S", "120")),
                        )
                    _cx = self._connex_pp_alloc
                    if _cx_rows is None:
                        _cx_ids = list(request_ids)
                        if all_token_ids is not None and len(all_token_ids) == num_reqs:
                            _cx_s0 = (list(num_computed_tokens_list)
                                      if num_computed_tokens_list is not None else None)
                            _cx_rows, cache_hits, cache_queries, prefix_computed_per_req = _cx.prefill(
                                _cx_ids, all_token_ids, _cx_s0, num_scheduled_tokens_per_req,
                                block_size, max_num_blocks_per_seq,
                            )
                            # Stage-1+ uses its own computed-prefix view (as the fork does)
                            if step_type == "prefill" and num_computed_tokens_list is not None:
                                for i, computed in enumerate(prefix_computed_per_req):
                                    if i < len(num_computed_tokens_list):
                                        num_computed_tokens_list[i] = computed
                            _cx_st = _cx.stats
                            if _cx_st["queries"] - getattr(_cx, "_logged_q", 0) >= 500:
                                _cx._logged_q = _cx_st["queries"]
                                logger.info(
                                    "External PP stage %d: prefix cache hits=%d misses=%d "
                                    "evictions=%d reclaimed_reqs=%d exhausted=%d free=%d live_reqs=%d",
                                    stage_idx, _cx_st["hits"], _cx_st["misses"], _cx_st["evictions"],
                                    _cx_st["reclaimed_reqs"], _cx_st["exhausted"], _cx.num_free(),
                                    len(_cx.reqs),
                                )
                        else:
                            _cx_counts = None
                            if step_type != "decode":
                                _cx_counts = (list(num_scheduled_tokens_per_req)
                                              if num_scheduled_tokens_per_req is not None
                                              else [num_actual_tokens or num_tokens] * num_reqs)
                            _cx_rows, _ = _cx.decode(
                                _cx_ids, block_size, max_num_blocks_per_seq, _cx_counts
                            )
                    blk_table_tensor = self._connex_pp_upload_block_table(_cx_rows)
'''
    src = src[:i] + body + src[j:]
    return src, "patched"


def main(path: str) -> int:
    src = open(path).read()
    new, status = patch(src)
    print(f"{os.path.basename(__file__)}: {status}")
    if new is None:
        return 1
    if new is not src:
        open(path, "w").write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
