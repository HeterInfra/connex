# SPDX-License-Identifier: Apache-2.0
"""Apply the external-PP patches to gpu_model_runner.py in order.

    python apply_all.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py [--skip name ...]

Order: fast_external_pp_block_table (base), then allocator (1), host metadata
(2), pinned upload (3, needs 1), fast hash (4), lazy logging (5), inflight (6,
edits the sibling gpu_worker.py, CONNEX_PP_INFLIGHT gated), stage-0 inflight
(7, runner + ../executor/uniproc_executor.py + ../core/sched/scheduler.py,
CONNEX_PP_INFLIGHT gated), mixed prefill/decode seq_lens fix (8, needs 2,
CONNEX_PP_FIX_MIXED_PREFILL=0 disables it at runtime), overlap (9, needs 6+7,
runner + gpu_worker.py, K>1 only, opt-in with CONNEX_PP_OVERLAP=1), downstream
graphs (10, stages 1+ replay captured decode cudagraphs, opt-in with
CONNEX_PP_DOWNSTREAM_GRAPHS=1; kept last). Engine fixes 11-13 run before 10:
chunked-prefill metadata (needs 8), finished_req_ids piggybacking, host
logits_indices (see their docstrings); re-prefill on failover (14,
runner + ../core/sched/scheduler.py, CONNEX_PP_REPREFILL=0 disables it) also
runs before 10, as does the logprobs compile warm-up (15,
CONNEX_WARMUP_LOGPROBS=0 disables it), and tensor parallelism per stage (16,
../../config/parallel.py + ../../distributed/parallel_state.py; TP rank 0 is
the stage's pipeline endpoint). Each patch is
idempotent. Exits 1 if any selected patch failed (later ones still run, a
dependent patch then reports its own anchor failure).
Patches can also be skipped via CONNEX_PP_SKIP_PATCHES="name1,name2".
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = [
    "fast_external_pp_block_table.py",
    "patch_external_pp_allocator.py",
    "patch_external_pp_host_metadata.py",
    "patch_external_pp_pinned_upload.py",
    "patch_external_pp_fast_hash.py",
    "patch_external_pp_lazy_logging.py",
    "patch_external_pp_inflight.py",
    "patch_external_pp_stage0_inflight.py",
    "patch_external_pp_mixed_prefill.py",
    "patch_external_pp_overlap.py",
    "patch_external_pp_chunked_seqlens.py",
    "patch_external_pp_finished_ids.py",
    "patch_external_pp_host_logits_idx.py",
    "patch_external_pp_reprefill.py",
    "patch_external_pp_warmup_logprobs.py",
    "patch_external_pp_tp.py",
    "patch_external_pp_downstream_graphs.py",
]


def main(argv) -> int:
    path = argv[1]
    skip = set(argv[argv.index("--skip") + 1:]) if "--skip" in argv else set()
    skip |= {s for s in os.environ.get("CONNEX_PP_SKIP_PATCHES", "").split(",") if s}
    rc = 0
    for name in ORDER:
        if name in skip or name[:-3] in skip:
            print(f"{name}: skipped")
            continue
        p = subprocess.run([sys.executable, os.path.join(HERE, name), path],
                           capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip().splitlines()
        print(f"{name}: rc={p.returncode} {out[-1] if out else ''}")
        rc |= p.returncode != 0
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
