# SPDX-License-Identifier: Apache-2.0
"""Patch 4 (prefill hashing): cheaper external-PP block hashes.

The fork hashes each full block as sha256(prev + str(tuple(tokens))) and does it
twice per prefill (allocate + mark_computed). Patch 1 already caches the hash
chain per request (computed once, extended incrementally when a request is
re-sent with more tokens, reused by mark_computed). This patch switches the hash
to blake2b(prev + int64 token bytes, digest_size=16): no str() formatting of
the token tuple (the dominant cost, ~10x the hashing itself) and a faster digest.
The hashes are only compared inside one stage process, so the format change is
safe; 128-bit digests keep collision probability negligible.
Without patch 1 the fork's _compute_block_hashes_for_external_pp is patched instead.
Expected gain: roughly 5-10x less CPU per prefilled block (measure with
tests/test_vllm_patches.py::test_fast_hash_is_faster).

    python patch_external_pp_fast_hash.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
"""
import os
import sys

MARK = "# connex: fast external-PP block hash"


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    flag = "_CONNEX_PP_FAST_HASH = False\n"
    legacy = """            hash_input = prev_hash + str(block_tokens).encode('utf-8')
            block_hash = hashlib.sha256(hash_input).digest()
"""
    done = False
    if src.count(flag) == 1:
        src = src.replace(flag, f"_CONNEX_PP_FAST_HASH = True  {MARK}\n")
        done = True
    if src.count(legacy) == 1:
        src = src.replace(legacy, f"""            {MARK}
            from array import array as _cx_legacy_array
            block_hash = hashlib.blake2b(
                prev_hash + _cx_legacy_array("q", block_tokens).tobytes(), digest_size=16
            ).digest()
""")
        done = True
    if not done:
        return None, "anchors not found"
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
