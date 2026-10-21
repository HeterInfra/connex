# SPDX-License-Identifier: Apache-2.0
"""Patch 7 (stage-0 pipelining): application/idempotency on copies of the fork
files, and the pure pieces it inserts (_CxSlots, _CxLazyFuture)."""

import ast
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
V1 = FORK_V1
FILES = ["worker/gpu_model_runner.py", "worker/gpu_worker.py",
         "executor/uniproc_executor.py", "core/sched/scheduler.py"]
P7 = "patch_external_pp_stage0_inflight.py"


@pytest.fixture
def tree(tmp_path):
    if not os.path.exists(os.path.join(V1, FILES[0])):
        pytest.skip("fork source not available")
    for f in FILES:
        (tmp_path / os.path.dirname(f)).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(V1, f), tmp_path / f)
    return tmp_path


def _apply_all(runner):
    return subprocess.run([sys.executable, os.path.join(PATCHES, "apply_all.py"), runner],
                          capture_output=True, text=True)


def _read(tree):
    return {f: (tree / f).read_text() for f in FILES}


def test_apply_all_patches_three_files_idempotently(tree):
    runner = str(tree / FILES[0])
    p = _apply_all(runner)
    assert p.returncode == 0, p.stdout + p.stderr
    first = _read(tree)
    p = _apply_all(runner)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _read(tree) == first
    last = next(l for l in p.stdout.splitlines() if l.startswith(P7))
    assert last.count("already patched") == 3
    for f, src in first.items():
        ast.parse(src)
    r = first[FILES[0]]
    assert r.count("_cx_input_batch(") == 3  # def + two constructions
    assert "self._cx_begin_step(scheduler_output)" in r
    assert "return self._cx_send_step(" in r
    # the K=1 serial send/recv loop is still right after the gate
    gate = r.index("return self._cx_send_step(")
    assert r.index("for _pipeline_attempt in range(_max_pipeline_retries):", gate) > gate
    assert "return _CxLazyFuture(_cx_run, single_value)" in first[FILES[2]]
    assert "break  # connex: leave the rest for the next microbatch" in first[FILES[3]]


def test_runner_only_copy_skips_siblings(tmp_path):
    if not os.path.exists(os.path.join(V1, FILES[0])):
        pytest.skip("fork source not available")
    dst = tmp_path / "gpu_model_runner.py"
    shutil.copy(os.path.join(V1, FILES[0]), dst)
    p = _apply_all(str(dst))
    assert p.returncode == 0, p.stdout
    line = next(l for l in p.stdout.splitlines() if l.startswith(P7))
    assert line.count("skipped (not found)") == 2 and "runner patched" in line


def _extract(src, names, ns):
    tree = ast.parse(src)
    body = [n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), "x", "exec"), ns)
    return ns


@pytest.fixture
def patched(tree):
    assert _apply_all(str(tree / FILES[0])).returncode == 0
    return _read(tree)


def test_slots_logic(patched):
    ns = _extract(patched[FILES[0]], ["_CxSlots"], {})
    S = ns["_CxSlots"](2)
    a = S.acquire()
    assert a == 0
    S.set_home(0, {"r1": 1, "r2": 1})
    S.busy[0] = {"req_ids": {"r1", "r2"}}
    assert S.acquire() == 1
    # r3 new; r1 cannot be scheduled (outstanding) unless chunked prefill
    S.set_home(1, {"r3": 1, "r1": 4})
    S.busy[1] = {"req_ids": {"r3", "r1"}}
    with pytest.raises(RuntimeError):
        S.acquire()
    # slot 0 finishes; r2 aborted while slot 0 outstanding? no: it is done now
    S.release(0)
    assert S.acquire() == 0
    # batch 0 still holds r1 (stale: last scheduled in slot 1) and r2 (home 0)
    assert S.stale_rows(0, ["r1", None, "r2"], {"r1": 1, "r2": 1}) == ["r1"]
    assert S.stale_rows(0, ["r1", "r2"], {"r2": 1}) == []  # unscheduled: stock path


def test_slots_zombies(patched):
    ns = _extract(patched[FILES[0]], ["_CxSlots"], {})
    S = ns["_CxSlots"](2)
    S.busy[1] = {"req_ids": {"a", "b"}}
    S.set_home(1, {"a": 1, "b": 1})
    states = {"a": "A", "b": "B", "c": "C"}
    S.note_finished({"a", "c"}, states.get)
    assert S.zombies == {"a": (1, "A")}  # c not outstanding -> plain finish
    assert "a" not in S.home and S.home["b"] == 1
    assert S.zombies_of(1) == {"a": "A"} and S.zombies_of(0) == {}
    S.release(1)
    assert S.zombies == {} and S.busy == {}


def test_lazy_future_runs_once_in_caller(patched):
    ns = _extract(patched[FILES[2]], ["_CxLazyFuture"], {"Future": Future})
    LF = ns["_CxLazyFuture"]
    calls = []
    f = LF(lambda: calls.append(1) or "out")
    assert not f.done() and calls == []
    seen = []
    f.add_done_callback(lambda fut: seen.append(fut.result()))
    assert f.result() == "out" and f.result() == "out" and calls == [1]
    assert f.done() and seen == ["out"]
    g = LF(lambda: "x", single_value=False)
    assert g.result() == ["x"]

    def boom():
        raise ValueError("no reply")
    h = LF(boom)
    with pytest.raises(ValueError):
        h.result()
    with pytest.raises(ValueError):
        h.result()


def test_executor_k1_unchanged(patched):
    src = patched[FILES[2]]
    assert "return 2 if self.scheduler_config.async_scheduling else 1" in src
    assert '_CX_PP_INFLIGHT = max(1, int(os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))' in src


def test_scheduler_cap_math(patched):
    src = patched[FILES[3]]
    line = next(l for l in src.splitlines() if "_cx_cap = (" in l)
    assert "-(-len(self.running) // _CX_PP_INFLIGHT)" in line
    for running, k, cap in [(32, 2, 16), (33, 2, 17), (32, 4, 8), (1, 4, 1), (0, 2, 0)]:
        assert -(-running // k) == cap


def test_mixed_prefill_fix_applied_and_gated(patched):
    src = patched[FILES[0]]
    i = src.index("# connex: mixed prefill/decode seq_lens (CONNEX_PP_FIX_MIXED_PREFILL)")
    blk = src[i:i + 600]
    assert 'os.environ.get("CONNEX_PP_FIX_MIXED_PREFILL", "1") != "0"' in blk
    assert "seq_lens_cpu += torch.tensor(num_computed_tokens_list" in blk
    # query_start_loc is still built from query lengths, before the fix
    assert src.rindex("query_start_loc_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0)", 0, i) < i
    # and the device copy of seq_lens happens after it
    assert src.index("_cx_dev = _cx_host.pin_memory().to(self.device, non_blocking=True)", i) > i


def test_mixed_prefill_seq_lens_math():
    import torch
    sched, computed = [5, 1, 1], [0, 17, 30]
    host = torch.zeros(2 * 3 + 1, dtype=torch.int32)
    sl, qsl = host[:3], host[3:]
    sl.copy_(torch.tensor(sched, dtype=torch.int32))
    qsl[1:] = torch.cumsum(sl, dim=0)
    sl += torch.tensor(computed, dtype=torch.int32)
    assert qsl.tolist() == [0, 5, 6, 7] and sl.tolist() == [5, 18, 31]


def test_extra_batches_share_cudagraph_buffers(patched):
    src = patched[FILES[0]]
    i = src.index("    def _cx_new_input_batch(self):")
    body = src[i:src.index("    def _cx_begin_step", i)]
    assert "mine.block_table.gpu = base.block_table.gpu" in body
    assert "mine.slot_mapping.gpu = base.slot_mapping.gpu" in body
