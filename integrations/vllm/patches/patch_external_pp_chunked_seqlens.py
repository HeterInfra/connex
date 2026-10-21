# SPDX-License-Identifier: Apache-2.0
"""Patch 11 (correctness, stages 1+): context-consistent prefill metadata.

    python patch_external_pp_chunked_seqlens.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Needs patches 1, 2 and 8 (allocator, host metadata, mixed prefill).

Bugs (fork 380186d57, _build_minimal_attention_metadata_for_external_pp and
execute_external_pipeline_step):
  a. prefill-type steps set seq_len = num_scheduled_tokens, dropping the
     cached context (num_computed_tokens) of chunked-prefill chunks and of
     decode requests riding in a prefill batch. Patch 8 adds the context to
     seq_lens; this patch keeps the rest of the metadata consistent with it.
  b. num_computed_tokens_cpu was taken from num_computed_tokens_list AFTER the
     allocator overwrote that list in place with this stage's block-granular
     prefix-cache view, so seq_len - query_len != num_computed_tokens for any
     partially cached context. Now num_computed_tokens_cpu = seq_len - query_len
     (exactly vLLM's own invariant), which equals stage 0's view.
  c. the overwrite happened on the list object taken from the received
     tensor_dict, and an intermediate stage forwards that same object, so
     stage 2+ received stage 1's prefix view instead of stage 0's
     num_computed_tokens (wrong seq_lens on >2-stage pipelines). The runner
     now works on a private copy.
positions come from stage 0 (true absolute positions); slot_mapping indexes
the allocator row with position // block_size, and the allocator row covers
every token of all_token_ids plus one block, so a later chunk / decode token
always maps to an owned block.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_pp_engine_fixes import run_patch  # noqa: E402

MARK = "# connex: context-consistent prefill metadata"

EDITS = [
    # c. private copy of stage 0's num_computed_tokens
    ('        num_computed_tokens_list = tensor_dict.get("num_computed_tokens")\n',
     '        num_computed_tokens_list = tensor_dict.get("num_computed_tokens")\n'
     '        ' + MARK + ' (private copy; the allocator rewrites it)\n'
     '        if num_computed_tokens_list is not None:\n'
     '            num_computed_tokens_list = list(num_computed_tokens_list)\n'),
    # b. num_computed_tokens_cpu = seq_len - query_len
    ('            elif num_computed_tokens_list is not None and len(num_computed_tokens_list) == num_reqs:\n'
     '                # Prefill with prefix cache: use num_computed_tokens from Stage-0\n',
     '            elif (num_scheduled_tokens_per_req is not None\n'
     '                  and len(num_scheduled_tokens_per_req) == num_reqs and num_reqs):\n'
     '                ' + MARK + '\n'
     '                # context = seq_len - query_len, consistent with seq_lens_cpu\n'
     '                num_computed_tokens_cpu = (\n'
     '                    seq_lens_cpu - (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])\n'
     '                ).clamp_(min=0).to(torch.int32)\n'
     '            elif num_computed_tokens_list is not None and len(num_computed_tokens_list) == num_reqs:\n'
     '                # Prefill with prefix cache: use num_computed_tokens from Stage-0\n'),
]


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    if "# connex: mixed prefill/decode seq_lens" not in src:
        return None, "needs patch_external_pp_mixed_prefill.py"
    for old, new in EDITS:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
        src = src.replace(old, new)
    return src, "patched"


if __name__ == "__main__":
    sys.exit(run_patch(__file__, patch, sys.argv))
