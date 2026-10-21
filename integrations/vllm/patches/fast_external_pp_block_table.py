# SPDX-License-Identifier: Apache-2.0
"""Patch the external-PP vLLM fork (380186d57): build the external-PP block
table on the host and copy it to the GPU once per step.

The fork fills `blk_table_tensor` (a CUDA tensor) element by element from a
Python loop, i.e. one tiny H2D copy + kernel per block slot per request per
step on every downstream stage. Under load that dominates step time
(~200 ms/step measured with py-spy at QPS 4). Behaviour is unchanged: the same
block ids end up in the same slots.

    python fast_external_pp_block_table.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
Idempotent; exits 0 if already applied, 1 if the expected code is not found.
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "# connex: host-built block table"
if MARK in src:
    print("already patched")
    sys.exit(0)

old_alloc = """        blk_table_tensor = torch.zeros(
            (num_reqs, max_num_blocks_per_seq), dtype=torch.int32, device=self.device"""
new_alloc = f"""        {MARK}
        _blk_rows = [[0] * max_num_blocks_per_seq for _ in range(num_reqs)]
        blk_table_tensor = torch.zeros(
            (num_reqs, max_num_blocks_per_seq), dtype=torch.int32, device=self.device"""
old_fill = """            # Fill block table tensor
            for j, bid in enumerate(block_ids[:max_num_blocks_per_seq]):
                blk_table_tensor[i, j] = bid
        
        return blk_table_tensor, total_cache_hits, total_queries, num_computed_per_req"""
new_fill = """            # Fill block table row on the host (copied to the GPU once below)
            _ids = block_ids[:max_num_blocks_per_seq]
            _blk_rows[i][:len(_ids)] = _ids

        if num_reqs:
            blk_table_tensor.copy_(torch.tensor(_blk_rows, dtype=torch.int32), non_blocking=False)
        return blk_table_tensor, total_cache_hits, total_queries, num_computed_per_req"""
for old, new in ((old_alloc, new_alloc), (old_fill, new_fill)):
    if src.count(old) != 1:
        print("expected code not found; not patching")
        sys.exit(1)
    src = src.replace(old, new)
open(path, "w").write(src)
print("patched")
