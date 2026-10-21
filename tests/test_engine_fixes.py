# SPDX-License-Identifier: Apache-2.0
"""External-PP engine fixes (patches 11-13): application on a copy of the
fork, and the pure logic they inline (external_pp_engine_fixes.py)."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
NEW = ["patch_external_pp_chunked_seqlens.py", "patch_external_pp_finished_ids.py",
       "patch_external_pp_host_logits_idx.py"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PATCHES, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load("external_pp_engine_fixes")
A = _load("external_pp_alloc")


@pytest.fixture
def v1_copy(tmp_path):
    if not os.path.isdir(FORK_V1):
        pytest.skip("fork source not available")
    for d in ("worker", "executor", "core"):
        shutil.copytree(os.path.join(FORK_V1, d), tmp_path / "v1" / d)
    return str(tmp_path / "v1" / "worker" / "gpu_model_runner.py")


def _apply_all(path):
    return subprocess.run([sys.executable, os.path.join(PATCHES, "apply_all.py"), path],
                          capture_output=True, text=True)


# ---------------------------------------------------------------- patch files

def test_full_chain_applies_idempotent(v1_copy):
    p = _apply_all(v1_copy)
    assert p.returncode == 0, p.stdout + p.stderr
    for n in NEW:
        assert f"{n}: rc=0 {n}: patched" in p.stdout
    first = open(v1_copy).read()
    p = _apply_all(v1_copy)
    assert p.returncode == 0, p.stdout
    for n in NEW:
        assert f"{n}: rc=0 {n}: already patched" in p.stdout
    src = open(v1_copy).read()
    assert src == first
    ast.parse(src)
    assert src.count("# BEGIN connex external-pp engine fixes") == 1
    assert "logits_indices.cpu().tolist()" in src  # spec-decode fallback only
    assert "connex_pp_host_logits_indices(" in src
    assert 'forward_tensor_dict["finished_req_ids"] = _cx_ob.take()' in src
    assert 'forward_tensor_dict["finished_req_ids"] = _cx_fin' in src
    assert "list(num_computed_tokens_list)" in src


def test_inlined_block_runs_standalone(v1_copy):
    assert _apply_all(v1_copy).returncode == 0
    src = open(v1_copy).read()
    b = src.index("# BEGIN connex external-pp engine fixes")
    e = src.index("# END connex external-pp engine fixes")
    ns = {}
    exec(compile(src[b:e], "inlined", "exec"), ns)
    assert ns["connex_pp_prefill_lens"]([2], [3])[0] == [5]


def test_new_patches_need_their_base(tmp_path):
    if not os.path.isdir(FORK_V1):
        pytest.skip("fork source not available")
    dst = tmp_path / "gpu_model_runner.py"
    shutil.copy(os.path.join(FORK_V1, "worker", "gpu_model_runner.py"), dst)
    before = dst.read_text()
    for n in NEW:
        p = subprocess.run([sys.executable, os.path.join(PATCHES, n), str(dst)],
                           capture_output=True, text=True)
        assert p.returncode == 1, (n, p.stdout)
    assert dst.read_text() == before


# ------------------------------------------------------- bug 1: prefill lens

def test_prefill_lens_chunked_and_mixed():
    # req0: 2nd chunk of a 1000-token prompt (512 cached, 488 now)
    # req1: decode request inside a prefill batch (40 cached, 1 now)
    # req2: brand new prompt
    seq, qsl, ctx, mx, mq = F.connex_pp_prefill_lens([488, 1, 7], [512, 40, 0])
    assert seq == [1000, 41, 7]
    assert qsl == [0, 488, 489, 496]
    assert ctx == [512, 40, 0]
    assert (mx, mq) == (1000, 488)
    # invariant used by the patch: context = seq_len - query_len
    assert [s - (qsl[i + 1] - qsl[i]) for i, s in enumerate(seq)] == ctx


def test_prefill_lens_without_context():
    seq, qsl, ctx, mx, mq = F.connex_pp_prefill_lens([3, 4], None)
    assert seq == [3, 4] and ctx == [0, 0] and qsl == [0, 3, 7] and mx == 4
    assert F.connex_pp_prefill_lens([], [])[3:] == (0, 0)


def test_chunked_positions_map_to_owned_blocks():
    """Slot mapping on stages 1+ = row[pos // bs] * bs + pos % bs: every
    position of a later chunk / decode token must hit an allocated block,
    and the context blocks must not change between chunks."""
    bs, width = 16, 64
    al = A.ConnexPPBlockAllocator(200)
    toks = list(range(100))
    rows, *_ = al.prefill(["r"], [toks], [0], [48], bs, width)
    first = rows[0].copy()
    rows, *_ = al.prefill(["r"], [toks], [48], [52], bs, width)
    assert (rows[0][:3] == first[:3]).all()          # context blocks kept
    for pos in range(48, 100):
        assert rows[0][pos // bs] != 0
    # first decode position after the prompt
    assert al.peek_positions(["r"]) == [100]
    rows, pos = al.decode(["r"], bs, width)
    assert pos == [100] and rows[0][100 // bs] != 0


def test_seq_lens_patch_math_matches_torch():
    torch = pytest.importorskip("torch")
    sched, comp = [488, 1, 7], [512, 40, 0]
    host = torch.zeros(2 * 3 + 1, dtype=torch.int32)
    seq = host[:3]
    seq.copy_(torch.tensor(sched, dtype=torch.int32))
    qsl = host[3:]
    qsl[1:] = torch.cumsum(seq, dim=0)
    seq += torch.tensor(comp, dtype=torch.int32)          # patch 8
    ctx = (seq - (qsl[1:] - qsl[:-1])).clamp_(min=0)     # patch 11
    exp = F.connex_pp_prefill_lens(sched, comp)
    assert seq.tolist() == exp[0] and qsl.tolist() == exp[1] and ctx.tolist() == exp[2]


# ------------------------------------------------- bug 2: finished ids

def test_outbox_repeats_then_drains():
    ob = F.ConnexPPFinishedOutbox(repeat=2)
    assert ob.take() == [] and len(ob) == 0
    ob.add(["a", "b"])
    assert ob.take() == ["a", "b"]
    ob.add(["c"])
    assert ob.take() == ["a", "b", "c"]
    assert ob.take() == ["c"]
    assert ob.take() == []


def test_outbox_bounded():
    ob = F.ConnexPPFinishedOutbox(repeat=1, max_pending=3)
    ob.add([str(i) for i in range(10)])
    assert ob.take() == ["7", "8", "9"]


def _alloc_with(ids, bs=4):
    al = A.ConnexPPBlockAllocator(64, ttl_s=0)
    for rid in ids:
        al.prefill([rid], [list(range(10))], [0], [10], bs, 16)
    return al


def test_tracker_releases_and_frees_blocks():
    al = _alloc_with(["a", "b"])
    free0 = al.num_free()
    tr = F.ConnexPPFinishTracker()
    tr.on_step(al, ["a"], ["b"])
    assert "a" not in al.reqs and "b" in al.reqs
    assert al.num_free() > free0
    assert tr.stats["released"] == 1
    tr.on_step(al, ["a"], ["b"])  # duplicate notice (repeat / re-send): no-op
    assert tr.stats["released"] == 1 and "b" in al.reqs


def test_tracker_defers_while_referenced():
    al = _alloc_with(["a"])
    tr = F.ConnexPPFinishTracker()
    tr.on_step(al, ["a"], ["a"])      # the step itself still uses "a"
    assert "a" in al.reqs and "a" in tr.deferred
    tr.on_step(al, None, ["a"])       # still referenced
    assert "a" in al.reqs
    tr.on_step(al, None, ["x"])
    assert "a" not in al.reqs and not tr.deferred


def test_tracker_reordered_stale_step_is_released_again():
    """Step n (refs "a") is lost and re-sent after step n+1 (finish "a")."""
    bs, width = 4, 16
    al = _alloc_with(["a", "b"])
    tr = F.ConnexPPFinishTracker()
    tr.on_step(al, ["a"], ["b"])
    al.decode(["b"], bs, width)
    assert "a" not in al.reqs
    # the stale step arrives: it re-creates "a" in the allocator
    tr.on_step(al, None, ["a"])
    al.decode(["a"], bs, width)
    assert "a" in al.reqs and "a" in tr.revived
    tr.on_step(al, None, ["b"])
    assert "a" not in al.reqs and tr.stats["revived"] == 1


def test_tracker_without_allocator_and_bounded_tombstones():
    tr = F.ConnexPPFinishTracker(max_tombstones=2)
    tr.on_step(None, ["a", "b", "c"], [])
    assert list(tr.tombstones) == ["b", "c"]
    assert tr.stats["released"] == 0


def test_end_to_end_k_inflight_with_retry():
    """Stage 0 with K=2 steps in flight, a lost+re-sent step and the
    outbox/tracker pair: every finished request is released downstream and
    nothing live is touched."""
    bs, width = 4, 16
    al = A.ConnexPPBlockAllocator(256, ttl_s=0)
    ob, tr = F.ConnexPPFinishedOutbox(repeat=1), F.ConnexPPFinishTracker()

    def make_step(reqs, finished_now):
        ob.add(finished_now)
        return {"request_ids": reqs, "finished_req_ids": ob.take()}

    def process(td):
        tr.on_step(al, td.get("finished_req_ids"), td["request_ids"])
        for rid in td["request_ids"]:
            if rid not in al.reqs:
                al.prefill([rid], [list(range(6))], [0], [6], bs, width)
            else:
                al.decode([rid], bs, width)

    s1 = make_step(["a", "b"], [])
    s2 = make_step(["c"], [])
    process(s1)
    # s2 is lost; stage 0 finishes "a" and sends s3, then re-sends s2
    s3 = make_step(["b"], ["a"])
    process(s3)
    process(s2)               # re-sent (same dict, its own finished ids)
    s4 = make_step(["c"], ["b"])
    process(s4)
    s5 = make_step([], ["c"])
    process(s5)
    process({"request_ids": []})
    assert al.reqs == {}
    assert al.live_blocks() == set()


# ------------------------------------------------- bug 3: host logits indices

def test_host_logits_indices_match_device_formula():
    np = pytest.importorskip("numpy")
    qsl = np.array([0, 5, 6, 9, 9, 0, 0], dtype=np.int32)  # padded buffer
    assert F.connex_pp_host_logits_indices(qsl, 3) == [4, 5, 8]
    torch = pytest.importorskip("torch")
    li = torch.from_numpy(qsl[:4])[1:] - 1               # vLLM: query_start_loc[1:] - 1
    assert li.tolist() == F.connex_pp_host_logits_indices(qsl, 3)
    assert all(type(x) is int for x in F.connex_pp_host_logits_indices(qsl, 3))
