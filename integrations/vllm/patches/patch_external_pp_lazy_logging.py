# SPDX-License-Identifier: Apache-2.0
"""Patch 5 (hot-path overhead): Stage-0 per-step env lookups and eager logging.

Every Stage-0 step re-reads VLLM_EXTERNAL_PP_DEBUG, VLLM_EXTERNAL_PP_ASYNC_SEND
and CONNEXTE_RECV_RETRIES from os.environ and formats an f-string for a
logger.debug call that is normally discarded. This reads the env vars once at
import (module constants) and makes the debug log lazy (%-style args, guarded by
isEnabledFor). Semantic change: changing those env vars after the worker
process started no longer has an effect (they are set at pod start anyway).
Expected gain: small (~5-20 us/step); mainly removes noise from py-spy profiles.

    python patch_external_pp_lazy_logging.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
"""
import os
import sys

MARK = "# connex: external-PP env read once"


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    edits = [
        ("_external_pp_send_executor: ThreadPoolExecutor | None = None\n",
         f"""{MARK}
_CONNEX_PP_DEBUG = os.environ.get("VLLM_EXTERNAL_PP_DEBUG", "") == "1"
_CONNEX_PP_ASYNC_SEND = os.environ.get("VLLM_EXTERNAL_PP_ASYNC_SEND", "1") != "0"
_CONNEX_PP_RECV_RETRIES = int(os.environ.get("CONNEXTE_RECV_RETRIES", "3"))


_external_pp_send_executor: ThreadPoolExecutor | None = None
"""),
        ("""                _debug_pp = os.environ.get("VLLM_EXTERNAL_PP_DEBUG", "") == "1"
                _async_send = os.environ.get("VLLM_EXTERNAL_PP_ASYNC_SEND", "1") != "0"
""", """                _debug_pp = _CONNEX_PP_DEBUG
                _async_send = _CONNEX_PP_ASYNC_SEND
"""),
        ("""                else:
                    logger.debug(
                        f"Stage 0: Sending forward tensor dict with {len(hidden_states.tensors)} tensors, "
                        f"{num_input_tokens} tokens, {num_reqs} requests, step_type={step_type_str}"
                    )
""", """                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Stage 0: Sending forward tensor dict with %d tensors, "
                        "%s tokens, %s requests, step_type=%s",
                        len(hidden_states.tensors), num_input_tokens, num_reqs, step_type_str,
                    )
"""),
        ("""                _max_pipeline_retries = int(os.environ.get("CONNEXTE_RECV_RETRIES", "3"))
""", """                _max_pipeline_retries = _CONNEX_PP_RECV_RETRIES
"""),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
        src = src.replace(old, new)
    if "\nimport logging\n" not in src:
        src = src.replace("\nimport os\n", "\nimport logging\nimport os\n", 1)
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
