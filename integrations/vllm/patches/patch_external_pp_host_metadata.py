# SPDX-License-Identifier: Apache-2.0
"""Patch 2 (per-step syncs): build external-PP attention metadata on the host.

Problem (_build_minimal_attention_metadata_for_external_pp): every decode step
does seq_lens.max().item(), query_start_loc.cpu() and seq_lens.cpu() -> three
D2H copies, each a full device sync that stalls the CPU until all queued GPU
work (incl. the previous layer stack) drains, so host-side prep of the next
step can never overlap GPU execution. Prefill does torch.tensor(..., cuda)
followed by .item()/.cpu() of the same data (H2D then D2H round trips).

Mechanism:
  * decode: query_start_loc_cpu is a host arange. If patch 1's allocator knows
    every request, seq_lens_cpu / max_seq_len come from its host-tracked
    positions (position = len(all_token_ids) after prefill, +1 per decode step);
    the GPU seq_lens tensor is still derived from the real positions tensor (no
    sync), so the kernels see ground truth. max_seq_len gets +1 slack (an
    overestimate is only a launch heuristic for FlashAttention). Unknown
    requests fall back to the original sync path.
    CONNEX_PP_VERIFY_SEQLENS=1 keeps the sync, compares, and logs mismatches.
  * prefill: seq_lens / query_start_loc are built on the CPU (max from the
    Python list, cumsum on the host) and shipped with one pinned non_blocking
    H2D copy.
Expected gain: removes 3 device syncs per decode step (typically 0.1-1 ms each
of pure stall, more when the GPU queue is deep) and 2-3 per prefill.

Risk: if the host-tracked position drifts from Stage-0 (e.g. Stage-0 async
scheduling lags output_token_ids), seq_lens_cpu/max_seq_len are off; run once
with CONNEX_PP_VERIFY_SEQLENS=1 to confirm zero mismatches. Works without
patch 1 (falls back to the sync path for decode).

    python patch_external_pp_host_metadata.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
"""
import os
import sys

MARK = "# connex: host-built external-PP metadata"


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    edits = []
    a = "_external_pp_send_executor: ThreadPoolExecutor | None = None\n"
    edits.append((a, f"{MARK}\n_CONNEX_PP_VERIFY_SEQLENS = os.environ.get(\"CONNEX_PP_VERIFY_SEQLENS\", \"0\") == \"1\"\n\n\n" + a))

    # decode: host seq_lens
    edits.append((
        """                actual_positions = positions[:num_reqs]
                seq_lens = (actual_positions + 1).int()
                max_seq_len = int(seq_lens.max().item())
""",
        """                actual_positions = positions[:num_reqs]
                seq_lens = (actual_positions + 1).int()
                _cx_alloc = getattr(self, "_connex_pp_alloc", None)
                _cx_pos = None
                if _cx_alloc is not None and request_ids is not None and len(request_ids) == num_reqs:
                    _cx_pos = _cx_alloc.peek_positions(request_ids)
                if _cx_pos is not None and num_reqs > 0:
                    seq_lens_cpu = torch.tensor([p + 1 for p in _cx_pos], dtype=torch.int32)
                    max_seq_len = min(max(_cx_pos) + 2, self.max_model_len)
                    if _CONNEX_PP_VERIFY_SEQLENS:
                        _cx_true = seq_lens.cpu()
                        if not torch.equal(_cx_true, seq_lens_cpu):
                            logger.warning("connex: host seq_lens mismatch host=%s gpu=%s",
                                           seq_lens_cpu.tolist()[:8], _cx_true.tolist()[:8])
                            seq_lens_cpu = _cx_true
                            max_seq_len = int(_cx_true.max())
                else:
                    max_seq_len = int(seq_lens.max().item())
"""))
    edits.append((
        "            if positions is not None and len(positions) >= num_reqs:\n",
        "            seq_lens_cpu = None\n            if positions is not None and len(positions) >= num_reqs:\n"))
    edits.append((
        """            max_query_len = 1
            # CPU copies only needed for CommonAttentionMetadata
            query_start_loc_cpu = query_start_loc.cpu()
            seq_lens_cpu = seq_lens.cpu()
""",
        """            max_query_len = 1
            # Host copies built without a device sync
            query_start_loc_cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.cpu()
"""))
    # prefill with per-request counts
    edits.append((
        """                seq_lens = torch.tensor(num_scheduled_tokens_per_req, dtype=torch.int32, device=self.device)
                max_seq_len = int(seq_lens.max().item())
                max_query_len = max_seq_len
{B}
                # Build query_start_loc from cumulative sum on GPU
                query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
                query_start_loc[1:] = torch.cumsum(seq_lens, dim=0)
{B}
                # CPU copies only for CommonAttentionMetadata
                query_start_loc_cpu = query_start_loc.cpu()
                seq_lens_cpu = seq_lens.cpu()
""".replace("{B}", " " * 16),
        """                _cx_host = torch.zeros(2 * num_reqs + 1, dtype=torch.int32)
                seq_lens_cpu = _cx_host[:num_reqs]
                seq_lens_cpu.copy_(torch.tensor(num_scheduled_tokens_per_req, dtype=torch.int32))
                query_start_loc_cpu = _cx_host[num_reqs:]
                query_start_loc_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0)
                max_seq_len = int(max(num_scheduled_tokens_per_req)) if num_reqs else 0
                max_query_len = max_seq_len
                _cx_dev = _cx_host.pin_memory().to(self.device, non_blocking=True)
                seq_lens = _cx_dev[:num_reqs]
                query_start_loc = _cx_dev[num_reqs:]
"""))
    # single-request prefill
    edits.append((
        """                seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device=self.device)
                seq_lens_cpu = seq_lens.cpu()
                query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)
                query_start_loc_cpu = query_start_loc.cpu()
""",
        """                _cx_host = torch.tensor([max_seq_len, 0, num_tokens], dtype=torch.int32)
                seq_lens_cpu = _cx_host[:1]
                query_start_loc_cpu = _cx_host[1:]
                _cx_dev = _cx_host.pin_memory().to(self.device, non_blocking=True)
                seq_lens = _cx_dev[:1]
                query_start_loc = _cx_dev[1:]
"""))
    for old, new in edits:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
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
