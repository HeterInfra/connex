# SPDX-License-Identifier: Apache-2.0
"""Patch 10 (stages 1+ replay decode cudagraphs): application on top of the
full apply_all chain, idempotency, gating, and anchor failure."""

import ast
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
V1 = FORK_V1
FILES = ["worker/gpu_model_runner.py", "worker/gpu_worker.py",
         "executor/uniproc_executor.py", "core/sched/scheduler.py"]
P10 = "patch_external_pp_downstream_graphs.py"
MARK = "# connex: downstream-stage CUDA graphs (CONNEX_PP_DOWNSTREAM_GRAPHS)"


@pytest.fixture
def tree(tmp_path):
    if not os.path.exists(os.path.join(V1, FILES[0])):
        pytest.skip("fork source not available")
    for f in FILES:
        (tmp_path / os.path.dirname(f)).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(V1, f), tmp_path / f)
    return tmp_path


def _run(args):
    return subprocess.run([sys.executable] + args, capture_output=True, text=True)


def test_registered_last_in_apply_all():
    src = open(os.path.join(PATCHES, "apply_all.py")).read()
    tree = ast.parse(src)
    order = next(n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ORDER")
    names = [e.value for e in order.elts]
    assert names[-1] == P10


def test_applies_after_chain_and_is_idempotent(tree):
    runner = str(tree / FILES[0])
    p = _run([os.path.join(PATCHES, "apply_all.py"), runner])
    assert p.returncode == 0, p.stdout + p.stderr
    assert any(l.startswith(P10) and l.endswith(": patched") for l in p.stdout.splitlines()), p.stdout
    src = open(runner).read()
    ast.parse(src)
    assert src.count(MARK) == 3
    # default off, the eager call is kept as the fallback
    assert '"CONNEX_PP_DOWNSTREAM_GRAPHS", "0") == "1"' in src
    assert "def _connex_pp_graph_forward(" in src
    assert "if model_output is None:" in src
    # graph path uses the persistent buffers the capture used
    for s in ("self.intermediate_tensors", "self.positions.gpu", "self.seq_lens.gpu",
              "self.query_start_loc.gpu", "blk.slot_mapping.gpu", "uniform_decode=True"):
        assert s in src, s
    p = _run([os.path.join(PATCHES, "apply_all.py"), runner])
    assert p.returncode == 0
    assert open(runner).read() == src
    r = _run([os.path.join(PATCHES, P10), runner])
    assert r.returncode == 0 and "already patched" in r.stdout


def test_anchor_failure_on_unpatched_fork_runner_is_reported(tmp_path):
    # all three anchors must match; on a foreign file it fails and writes nothing
    f = tmp_path / "x.py"
    f.write_text("print('not a runner')\n")
    r = _run([os.path.join(PATCHES, P10), str(f)])
    assert r.returncode == 1 and "anchor not found" in r.stdout
    assert f.read_text() == "print('not a runner')\n"


def test_patched_function_has_eager_fallback_shape(tree):
    runner = str(tree / FILES[0])
    assert _run([os.path.join(PATCHES, "apply_all.py"), runner]).returncode == 0
    mod = ast.parse(open(runner).read())
    cls = next(n for n in ast.walk(mod) if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    meths = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    g = meths["_connex_pp_graph_forward"]
    # every early exit returns None (eager), the only other return is the cloned output
    rets = [n for n in ast.walk(g) if isinstance(n, ast.Return)]
    assert sum(isinstance(r.value, ast.Constant) and r.value.value is None for r in rets) >= 4
    assert "_connex_pp_fwd_profile" in meths
