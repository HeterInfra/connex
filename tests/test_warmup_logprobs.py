# SPDX-License-Identifier: Apache-2.0
"""Patch 15: logprobs compile warm-up in _dummy_sampler_run (CPU checks)."""

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from _fork import FORK_V1

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
FORK = os.path.join(FORK_V1, "worker", "gpu_model_runner.py")
NAME = "patch_external_pp_warmup_logprobs.py"


def _run(path):
    return subprocess.run([sys.executable, os.path.join(PATCHES, NAME), path],
                          capture_output=True, text=True)


@pytest.fixture
def patched(tmp_path):
    if not os.path.exists(FORK):
        pytest.skip("fork source not available")
    dst = tmp_path / "gpu_model_runner.py"
    shutil.copy(FORK, dst)
    for _ in range(2):
        r = _run(str(dst))
        assert r.returncode == 0, r.stdout
    return open(dst).read()


def _block(src):
    fn = src[src.index("def _dummy_sampler_run("):]
    b = fn.index("# connex: logprobs compile warm-up")
    e = fn.index("if self.speculative_config:")
    return textwrap.dedent("        " + fn[b:e])


def test_idempotent_and_inside_dummy_sampler_run(patched):
    assert patched.count("# connex: logprobs compile warm-up") == 1
    fn = patched[patched.index("def _dummy_sampler_run("):]
    fn = fn[:fn.index("\n    def ", 10)]
    assert fn.index("self.sampler(") < fn.index("connex: logprobs") < fn.index("if self.speculative_config")


def test_unknown_source_fails(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("def foo():\n    pass\n")
    assert _run(str(p)).returncode == 1


def _exec_block(src, sampler, logits, num_reqs, env):
    class Self:
        pass
    s = Self()
    s.sampler = sampler
    import logging
    ns = {"os": os, "time": time, "torch": torch, "logger": logging.getLogger("t"),
          "self": s, "logits": logits, "num_reqs": num_reqs}
    old = os.environ.get("CONNEX_WARMUP_LOGPROBS")
    if env is None:
        os.environ.pop("CONNEX_WARMUP_LOGPROBS", None)
    else:
        os.environ["CONNEX_WARMUP_LOGPROBS"] = env
    try:
        exec(compile(_block(src), "block", "exec"), ns)
    finally:
        if old is None:
            os.environ.pop("CONNEX_WARMUP_LOGPROBS", None)
        else:
            os.environ["CONNEX_WARMUP_LOGPROBS"] = old


def test_warmup_covers_dynamic_shapes_and_flag(patched):
    """Same decorator as the fork (dynamic=True). After the warm-up, row counts
    1..40 and echo-sized prompts must not trigger a new dynamo compile."""
    torch._dynamo.reset()
    compiles = []

    def backend(gm, inputs):
        compiles.append([tuple(i.shape) if hasattr(i, "shape") else i for i in inputs])
        return gm.forward

    @torch.compile(dynamic=True, backend=backend)
    def bcgt(x, values):
        return (x >= values).sum(-1)

    class Sampler:  # mirrors vllm.v1.sample.sampler.Sampler (static methods)
        calls = []

        @staticmethod
        def compute_logprobs(logits):
            return logits.log_softmax(dim=-1, dtype=torch.float32)

        @staticmethod
        def gather_logprobs(logprobs, num_logprobs, token_ids):
            assert token_ids.dtype == torch.int64
            Sampler.calls.append((logprobs.shape[0], num_logprobs))
            topk_lp, topk_idx = torch.topk(logprobs, num_logprobs, dim=-1)
            token_ids = token_ids.unsqueeze(-1)
            tok_lp = logprobs.gather(-1, token_ids)
            return bcgt(logprobs, tok_lp)

    V, N = 1000, 16
    logits = torch.randn(N, V, dtype=torch.bfloat16)
    _exec_block(patched, Sampler, logits, N, "0")
    assert Sampler.calls == [] and compiles == []

    _exec_block(patched, Sampler, logits, N, None)  # default on
    assert sorted({n for n, _ in Sampler.calls}) == [1, 2, N]
    n_warm = len(compiles)
    assert n_warm >= 1
    for n in list(range(1, 41)) + [300, 1500]:
        lp = torch.randn(n, V).log_softmax(-1)
        Sampler.gather_logprobs(lp, 1, lp.argmax(-1))
    assert len(compiles) == n_warm, compiles[n_warm:]
