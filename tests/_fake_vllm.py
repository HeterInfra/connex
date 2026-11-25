# SPDX-License-Identifier: Apache-2.0
"""Import the PP executor without vLLM: the few vLLM names it uses at import
time, as stand-ins, so its stage-0 send path can be tested and timed on CPU.

``load_executor()`` returns ``(module, restore)``; call ``restore()`` to
remove the stand-ins and the executor module again, so tests that skip
without vLLM still skip."""
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field

_MODS = ("connex.adapters.vllm.pp.executor", "connex.adapters.vllm.pp.worker")


class MultiprocExecutor:
    """execute_model / sample_tokens of the local stage: no work."""

    def execute_model(self, scheduler_output, non_block=False):
        return None

    def sample_tokens(self, grammar_output, non_block=False):
        return _Done()

    def shutdown(self):
        pass


class _Done:
    def result(self):
        return None


@dataclass
class KVConnectorOutput:
    lost_all_kv: bool = False


@dataclass
class ModelRunnerOutput:
    req_ids: list = field(default_factory=list)
    req_id_to_index: dict = field(default_factory=dict)
    sampled_token_ids: list = field(default_factory=list)
    kv_connector_output: KVConnectorOutput | None = None


def _fakes() -> dict:
    m = {n: types.ModuleType(n) for n in (
        "vllm", "vllm.v1", "vllm.v1.executor", "vllm.v1.executor.multiproc_executor",
        "vllm.v1.outputs", "vllm.distributed", "vllm.distributed.parallel_state",
        "vllm.v1.worker", "vllm.v1.worker.gpu_worker")}
    m["vllm.v1.executor.multiproc_executor"].MultiprocExecutor = MultiprocExecutor
    m["vllm.v1.outputs"].KVConnectorOutput = KVConnectorOutput
    m["vllm.v1.outputs"].ModelRunnerOutput = ModelRunnerOutput
    m["vllm.distributed.parallel_state"].get_pp_group = lambda: None
    m["vllm.v1.worker.gpu_worker"].Worker = type("Worker", (), {})
    return m


def load_executor():
    saved = {n: sys.modules.get(n) for n in list(_fakes()) + list(_MODS)}
    for n in _MODS:
        sys.modules.pop(n, None)
    sys.modules.update(_fakes())
    mod = importlib.import_module("connex.adapters.vllm.pp.executor")

    def restore():
        import connex.adapters.vllm.pp as pkg

        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v
        for n in ("executor", "worker"):
            if saved[f"connex.adapters.vllm.pp.{n}"] is None and hasattr(pkg, n):
                delattr(pkg, n)

    return mod, restore
