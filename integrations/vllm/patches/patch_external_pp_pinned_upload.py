# SPDX-License-Identifier: Apache-2.0
"""Patch 3 (decode block table): pinned, non-blocking block-table upload.

Requires patch_external_pp_allocator.py, which already removed the fork's
per-tuple(request_ids) GPU block-table cache (unbounded GPU memory, O(num_reqs)
row copies + clone on every batch change) and replaced it with one stacked
host table per step. This patch makes that single upload asynchronous:
  * the numpy table is copied into a pinned buffer from PyTorch's caching host
    allocator (which keeps the block alive until the copy's event fires, so
    reuse is race-free) and sent with non_blocking=True;
  * the throw-away torch.zeros((num_reqs, max_blocks), device=cuda) built every
    step before the allocator overwrote it becomes torch.empty (the only other
    consumer, the non-persistent fallback, writes every row in full).
Expected gain: the host no longer blocks on a pageable H2D copy (a stream sync)
each step, and one memset kernel per KV group per step disappears.
No semantic change: same block ids, same stream order.

    python patch_external_pp_pinned_upload.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
"""
import os
import sys

MARK = "# connex: block-table upload (pinned, non_blocking)"


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    edits = [
        ("""        # connex: block-table upload (pageable, synchronous)
        return torch.from_numpy(rows).to(self.device)
""", f"""        {MARK}
        return torch.from_numpy(rows).pin_memory().to(self.device, non_blocking=True)
"""),
        ("""                max_num_blocks_per_seq = (self.max_model_len + block_size - 1) // block_size
                blk_table_tensor = torch.zeros(
""", """                max_num_blocks_per_seq = (self.max_model_len + block_size - 1) // block_size
                blk_table_tensor = torch.empty(
"""),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            return None, f"anchor not found/unique (needs patch_external_pp_allocator.py first): {old.strip()[:60]!r}"
        src = src.replace(old, new)
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
