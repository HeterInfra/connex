# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin (entry point ``vllm.general_plugins``): registers the
``connex`` weight-transfer backend in every vLLM process."""


def register():
    try:
        from .vllm_backend import register as _r
    except ImportError:   # vLLM without the pluggable weight-transfer API
        return
    _r()
