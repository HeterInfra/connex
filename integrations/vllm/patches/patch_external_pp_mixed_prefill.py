# SPDX-License-Identifier: Apache-2.0
"""Patch 8 (correctness, stages 1+): seq_lens of mixed prefill/decode steps.

    python patch_external_pp_mixed_prefill.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Bug (fork, independent of CONNEX_PP_INFLIGHT): stage 0 labels a step
"prefill" as soon as ONE request in it is prefilling. Stages 1+ then build
attention metadata with seq_len = num_scheduled_tokens for every request,
i.e. they assume no computed context. A decode request (or a later
prefill chunk) that shares the step with a new prompt therefore attends only
to its own new token on stages 1+, and its output drifts. It shows up
whenever requests arrive while others decode (32 concurrent requests at
temperature 0: about half differ from their sequential output at K=1), and
much more with stage-0 pipelining, whose microbatches mix new prompts with
decodes on almost every step.

Fix: stage 0 already sends num_computed_tokens per request in prefill steps
(its view, before stage-1's own prefix-cache adjustment), so
seq_len = num_computed + num_scheduled; query_start_loc / max_query_len stay
the query lengths. CONNEX_PP_FIX_MIXED_PREFILL=0 restores the old metadata.
"""
import os
import sys

MARK = "# connex: mixed prefill/decode seq_lens (CONNEX_PP_FIX_MIXED_PREFILL)"

OLD = '''                seq_lens_cpu.copy_(torch.tensor(num_scheduled_tokens_per_req, dtype=torch.int32))
                query_start_loc_cpu = _cx_host[num_reqs:]
                query_start_loc_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0)
                max_seq_len = int(max(num_scheduled_tokens_per_req)) if num_reqs else 0
                max_query_len = max_seq_len
'''
NEW = '''                seq_lens_cpu.copy_(torch.tensor(num_scheduled_tokens_per_req, dtype=torch.int32))
                query_start_loc_cpu = _cx_host[num_reqs:]
                query_start_loc_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0)
                max_seq_len = int(max(num_scheduled_tokens_per_req)) if num_reqs else 0
                max_query_len = max_seq_len
                ''' + MARK + '''
                if (os.environ.get("CONNEX_PP_FIX_MIXED_PREFILL", "1") != "0"
                        and num_computed_tokens_list is not None
                        and len(num_computed_tokens_list) == num_reqs and num_reqs):
                    seq_lens_cpu += torch.tensor(num_computed_tokens_list, dtype=torch.int32)
                    max_seq_len = int(seq_lens_cpu.max())
'''


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    if src.count(OLD) != 1:
        return None, f"anchor not found/unique (needs patch_external_pp_host_metadata.py): {OLD.strip()[:60]!r}"
    return src.replace(OLD, NEW), "patched"


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
