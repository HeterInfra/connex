# SPDX-License-Identifier: Apache-2.0
"""Patch 13 (performance, stage 0): logits_indices without a device sync.

    python patch_external_pp_host_logits_idx.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Bug (fork 380186d57, stage-0 branch of execute_model): every external-PP step
does forward_tensor_dict["logits_indices"] = logits_indices.cpu().tolist(),
a blocking D2H copy that waits for the whole stage-0 forward before the send
can even be enqueued. Without spec decode vLLM sets
logits_indices = query_start_loc[1:] - 1 (_prepare_inputs), and the host copy
self.query_start_loc.np is already filled, so the same list is computed on the
host. Spec decode keeps the original path. Patch 9 (overlap) already does this
for its K>1 branch; this covers K=1 and non-overlap K>1.
CONNEX_PP_HOST_LOGITS_IDX=0 restores the device path.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_pp_engine_fixes import inline_module, run_patch  # noqa: E402

MARK = "# connex: host logits_indices (CONNEX_PP_HOST_LOGITS_IDX)"
LINE = re.compile(r'^( *)forward_tensor_dict\["logits_indices"\] = logits_indices\.cpu\(\)\.tolist\(\)\n',
                  re.M)
FLAG_ANCHOR = '_CONNEX_PP_DEBUG = os.environ.get("VLLM_EXTERNAL_PP_DEBUG", "") == "1"\n'


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    ms = list(LINE.finditer(src))
    if len(ms) != 1:
        return None, f"logits_indices .cpu().tolist() line found {len(ms)}x (want 1)"
    if src.count(FLAG_ANCHOR) != 1:
        return None, "flag anchor not found (needs patch_external_pp_lazy_logging.py)"
    m = ms[0]
    ind = m.group(1)
    new = (f"{ind}{MARK}\n"
           f"{ind}if _CONNEX_PP_HOST_LOGITS_IDX and spec_decode_metadata is None:\n"
           f"{ind}    forward_tensor_dict[\"logits_indices\"] = connex_pp_host_logits_indices(\n"
           f"{ind}        self.query_start_loc.np, num_reqs)\n"
           f"{ind}else:\n"
           f"{ind}    forward_tensor_dict[\"logits_indices\"] = logits_indices.cpu().tolist()\n")
    src = src[:m.start()] + new + src[m.end():]
    src = src.replace(FLAG_ANCHOR, FLAG_ANCHOR + "_CONNEX_PP_HOST_LOGITS_IDX = "
                      "os.environ.get(\"CONNEX_PP_HOST_LOGITS_IDX\", \"1\") != \"0\"\n", 1)
    src = inline_module(src)
    if src is None:
        return None, "allocator block not found (needs patch_external_pp_allocator.py)"
    return src, "patched"


if __name__ == "__main__":
    sys.exit(run_patch(__file__, patch, sys.argv))
