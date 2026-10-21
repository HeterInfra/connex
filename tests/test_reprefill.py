# SPDX-License-Identifier: Apache-2.0
"""Re-prefill on failover (patch 14): application on copies of the fork after
the full chain, idempotency, and the pure logic it inlines
(external_pp_reprefill.py) together with the real block allocator."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
P14 = "patch_external_pp_reprefill.py"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PATCHES, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RP = _load("external_pp_reprefill")
A = _load("external_pp_alloc")
P = _load("patch_external_pp_reprefill")
BS = 4


@pytest.fixture
def v1_copy(tmp_path):
    if not os.path.isdir(FORK_V1):
        pytest.skip("fork source not available")
    for d in ("worker", "executor", "core"):
        shutil.copytree(os.path.join(FORK_V1, d), tmp_path / "v1" / d)
    return str(tmp_path / "v1" / "worker" / "gpu_model_runner.py")


def _apply_all(path, *extra):
    return subprocess.run([sys.executable, os.path.join(PATCHES, "apply_all.py"), path, *extra],
                          capture_output=True, text=True)


# ---------------------------------------------------------------- patch files

def test_registered_before_downstream_graphs():
    src = open(os.path.join(PATCHES, "apply_all.py")).read()
    order = next(n.value for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ORDER")
    names = [e.value for e in order.elts]
    assert names[-1] == "patch_external_pp_downstream_graphs.py"
    assert names.index(P14) > names.index("patch_external_pp_finished_ids.py")
    assert names.index(P14) > names.index("patch_external_pp_stage0_inflight.py")


def test_full_chain_applies_idempotent(v1_copy):
    p = _apply_all(v1_copy)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"{P14}: rc=0 {P14}: runner patched; scheduler.py patched" in p.stdout
    sched = os.path.join(os.path.dirname(os.path.dirname(v1_copy)), "core", "sched", "scheduler.py")
    first = open(v1_copy).read(), open(sched).read()
    p = _apply_all(v1_copy)
    assert p.returncode == 0, p.stdout
    assert f"{P14}: rc=0 {P14}: runner already patched; scheduler.py already patched" in p.stdout
    assert (open(v1_copy).read(), open(sched).read()) == first
    src, ssrc = first
    ast.parse(src)
    ast.parse(ssrc)
    assert src.count("# BEGIN connex external-pp reprefill") == 1
    assert src.count(P.MARK) == 2  # flag + stages 1+ check
    for s in ('forward_tensor_dict["cx_ctx"] = [',
              'self._cx_reprefill_ids = return_tensor_dict.get("cx_missing") or None',
              'self._cx_reprefill_ids = ret.get("cx_missing") or None',
              "output.cx_reprefill_req_ids = set(_cx_rp)",
              'return_tensor_dict["cx_missing"] = _cx_miss_all',
              'forward_tensor_dict["cx_missing"] = _cx_miss_all',
              'forward_tensor_dict["cx_ctx"] = _cx_ctx',
              "ConnexPPKVGuard(connex_pp_hash_block)",
              '"CONNEX_PP_REPREFILL", "1") != "0"'):
        assert s in src, s
    # check runs before the step builds metadata; commit before mark-computed
    assert src.index("_cx_g.check(") < src.index("attn_metadata = self._build_minimal_attention_metadata_for_external_pp(")
    assert src.index("self._cx_kv_guard.commit(") < src.index("# CRITICAL: After successful prefill, mark blocks")
    # downstream graphs (kept last) still applied on top
    assert "def _connex_pp_graph_forward(" in src
    assert "def _cx_reprefill_one(self, req_id, outputs, stopped_running_reqs)" in ssrc
    assert ssrc.index("if _cx_rp and req_id in _cx_rp:") < ssrc.index(
        "if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:")


def test_pyflakes_clean_names(v1_copy):
    pf = pytest.importorskip("pyflakes.api")
    assert _apply_all(v1_copy).returncode == 0
    sched = os.path.join(os.path.dirname(os.path.dirname(v1_copy)), "core", "sched", "scheduler.py")
    from pyflakes.reporter import Reporter
    import io
    for f in (v1_copy, sched):
        out = io.StringIO()
        pf.check(open(f).read(), f, Reporter(out, out))
        bad = [l for l in out.getvalue().splitlines() if "undefined name" in l]
        assert not bad, bad


def test_inlined_block_runs_standalone(v1_copy):
    assert _apply_all(v1_copy).returncode == 0
    src = open(v1_copy).read()
    b = src.index("# BEGIN connex external-pp reprefill")
    e = src.index("# END connex external-pp reprefill")
    ns = {}
    exec(compile(src[b:e], "inlined", "exec"), ns)
    assert ns["connex_pp_merge_missing"](["a"], ["b", "a"]) == ["a", "b"]


def test_skip_leaves_files_alone_and_needs_base(v1_copy):
    p = _apply_all(v1_copy, "--skip", P14)
    assert p.returncode == 0, p.stdout
    assert "cx_ctx" not in open(v1_copy).read()
    # on the stock fork the anchors of the base patches are missing
    fresh = os.path.join(os.path.dirname(v1_copy), "fresh_runner.py")
    shutil.copy(os.path.join(FORK_V1, "worker", "gpu_model_runner.py"), fresh)
    new, msg = P.patch_runner(open(fresh).read())
    assert new is None and "anchor" in msg


def test_scheduler_rewind_and_abort(v1_copy):
    """Run the inserted Scheduler._cx_reprefill_one on fake requests."""
    assert _apply_all(v1_copy).returncode == 0
    sched = os.path.join(os.path.dirname(os.path.dirname(v1_copy)), "core", "sched", "scheduler.py")
    tree = ast.parse(open(sched).read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_cx_reprefill_one")
    code = textwrap.dedent(ast.get_source_segment(open(sched).read(), fn))

    class St:
        RUNNING, FINISHED_ABORTED = "running", "aborted"

    class Out:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    ns = {"RequestStatus": St, "EngineCoreOutput": Out, "_CX_PP_REPREFILL_MAX": 2,
          "logger": types.SimpleNamespace(error=lambda *a: None)}
    exec(code, ns)

    class Req:
        def __init__(self, rid):
            self.request_id, self.status, self.num_computed_tokens = rid, St.RUNNING, 57
            self.client_index, self.trace_headers, self.num_cached_tokens = 0, None, 0

        def get_finished_reason(self):
            return "abort" if self.status == St.FINISHED_ABORTED else None

        def take_events(self):
            return None

    freed = []
    s = types.SimpleNamespace(requests={"x": Req("x"), "y": Req("y")},
                              _free_request=lambda r: freed.append(r.request_id))
    from collections import defaultdict
    outputs, stopped = defaultdict(list), set()
    f = lambda rid: ns["_cx_reprefill_one"](s, rid, outputs, stopped)  # noqa: E731
    f("x")
    assert s.requests["x"].num_computed_tokens == 0 and not outputs and not stopped
    s.requests["x"].num_computed_tokens = 60
    f("x")
    assert s.requests["x"].num_computed_tokens == 0
    f("x")  # third time > max 2: clean failure
    assert freed == ["x"] and s.requests["x"] in stopped
    assert outputs[0][0].finish_reason == "abort" and outputs[0][0].new_token_ids == []
    s.requests["y"].status = "preempted"
    f("y")
    assert s.requests["y"].num_computed_tokens == 57  # not running: untouched
    f("gone")  # unknown id: no-op


# ---------------------------------------------------------------- pure logic

def _alloc():
    return A.ConnexPPBlockAllocator(256, ttl_s=0)


def _run_step(alloc, guard, rids, ctx, step_type, toks=None, nsched=None):
    """What execute_external_pipeline_step does around the forward."""
    miss = guard.check(alloc, rids, ctx, step_type, toks, BS, nsched)
    if toks is not None:
        toks = RP.connex_pp_blank_rows(toks, rids, miss)
        s0 = list(ctx)
        alloc.prefill(list(rids), toks, s0, nsched, BS, 64)
    else:
        alloc.decode(list(rids), BS, 64)
    guard.commit(alloc, rids, ctx, step_type, nsched, miss)
    if toks is not None:
        alloc.mark_computed(list(rids), BS)
    return miss


def test_healthy_prefill_then_decode_never_missing():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    toks = list(range(10))
    assert _run_step(al, g, ["a"], [0], "prefill", [toks], [10]) == []
    for c in range(10, 20):
        assert _run_step(al, g, ["a"], [c], "decode") == []
    assert g.kv_len["a"] == 20


def test_chunked_prefill_continuation_ok():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    toks = list(range(30))
    assert _run_step(al, g, ["a"], [0], "prefill", [toks], [12]) == []
    assert _run_step(al, g, ["a"], [12], "prefill", [toks], [12]) == []
    assert _run_step(al, g, ["a"], [24], "prefill", [toks], [6]) == []
    assert _run_step(al, g, ["a"], [30], "decode") == []


def test_failover_replica_without_state_reports_missing_and_recovers():
    """Replica A served the request; replica B (fresh) gets a decode step."""
    b, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    miss = _run_step(b, g, ["a", "n"], [25, 0], "decode")
    assert miss == ["a"]
    assert "a" not in b.reqs and "a" not in g.kv_len  # released, nothing kept
    # stage 0 rewinds 'a' and re-prefills prompt + generated tokens
    toks = list(range(26))
    assert _run_step(b, g, ["a"], [0], "prefill", [toks], [26]) == []
    assert _run_step(b, g, ["a"], [26], "decode") == []


def test_stale_replica_with_older_state_is_missing():
    """Back to a replica that served the request before it moved away."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    _run_step(al, g, ["a"], [0], "prefill", [list(range(8))], [8])
    _run_step(al, g, ["a"], [8], "decode")          # holds 9 tokens
    assert _run_step(al, g, ["a"], [15], "decode") == ["a"]


def test_reclaimed_request_is_missing():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    _run_step(al, g, ["a"], [0], "prefill", [list(range(8))], [8])
    al.release("a")  # TTL / LRU / finished-notice race
    assert _run_step(al, g, ["a"], [8], "decode") == ["a"]


def test_missing_prefill_row_marks_nothing_computed():
    """A chunk continuation on a replica without the first chunk must not
    poison the prefix cache with blocks computed without context."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    toks = list(range(24))
    assert _run_step(al, g, ["a"], [12], "prefill", [toks], [12]) == ["a"]
    assert not al.computed and "a" not in al.reqs


def test_stage0_prefix_hit_served_by_own_prefix_cache():
    """New request with stage-0 prefix-cache context: fine when this stage
    holds the same computed prefix blocks, missing when it does not."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    prompt = list(range(16))
    _run_step(al, g, ["a"], [0], "prefill", [prompt], [16])
    # 'b' shares 12 tokens (3 blocks) with 'a'; stage 0 had them cached
    pb = prompt[:12] + [99, 98, 97]
    assert _run_step(al, g, ["b"], [12], "prefill", [pb], [3]) == []
    other, g2 = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    assert _run_step(other, g2, ["b"], [12], "prefill", [pb], [3]) == ["b"]


def test_mixed_batch_only_flags_affected_rows():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    _run_step(al, g, ["a"], [0], "prefill", [list(range(8))], [8])
    miss = _run_step(al, g, ["a", "b", "c"], [8, 40, 0], "prefill",
                     [list(range(9)), list(range(41)), list(range(5))], [1, 1, 5])
    assert miss == ["b"]
    assert g.kv_len["a"] == 9 and g.kv_len["c"] == 5


def test_resent_older_step_does_not_shrink_or_flag():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    _run_step(al, g, ["a"], [0], "prefill", [list(range(8))], [8])
    _run_step(al, g, ["a"], [8], "decode")
    _run_step(al, g, ["a"], [9], "decode")
    assert _run_step(al, g, ["a"], [8], "decode") == []  # duplicate re-executed
    assert _run_step(al, g, ["a"], [10], "decode") == []


def test_no_ctx_means_no_check():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    assert g.check(al, ["a"], None, "decode", None, BS) == []
    assert g.check(None, [], [3], "decode", None, BS) == []
    assert g.check(None, ["a"], [3], "decode", None, BS) == ["a"]  # nothing allocated yet


def test_prune_forgets_released_requests():
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block, prune_every=2)
    _run_step(al, g, ["a"], [0], "prefill", [list(range(8))], [8])
    al.release("a")
    _run_step(al, g, ["b"], [0], "prefill", [list(range(4))], [4])
    assert "a" not in g.kv_len and "b" in g.kv_len


def test_merge_and_blank_helpers():
    assert RP.connex_pp_merge_missing(None, []) is None
    assert RP.connex_pp_merge_missing(None, ["a"]) == ["a"]
    assert RP.connex_pp_merge_missing(["a"], None) == ["a"]
    assert RP.connex_pp_blank_rows([[1], [2], [3]], ["a", "b", "c"], ["b"]) == [[1], [], [3]]


# ---- communicator: a confirmed-dead replica is not re-selected ---------------

def test_failover_skips_departed_replica_still_listed_by_router():
    """Stage 0 confirmed stage-1 replica 1 dead (suspect + data-path timeout)
    while the router still lists it: the re-selection must pick replica 0,
    not the rank-mapped dead replica 1 (seen on the cluster: stage 0 retried
    the dead replica until its engine gave up)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_connexte_protocol import FakeEPC, _route, _route_comm
    epc = FakeEPC([_route(1, 0, "s1r0"), _route(1, 1, "s1r1"),
                   _route(2, 0, "s2r0:1"), _route(2, 1, "s2r1:1")])
    c = _route_comm(epc)  # stage 1, our replica 1 -> rank 1 -> picks replica 1
    assert c._get_peer_endpoint(2) == "s2r1:1"
    c._deregistered_replicas.add((2, 1))
    c._sticky_peer.pop(2)
    assert c._get_peer_endpoint(2) == "s2r0:1"
    assert c._sticky_peer[2] == 0
    # nothing else left: fall back to the listed one rather than to nothing
    epc.routes = [_route(1, 1, "s1r1"), _route(2, 1, "s2r1:1")]
    c._sticky_peer.pop(2)
    assert c._get_peer_endpoint(2) == "s2r1:1"


# ---- same-step shared prefix (stage 0 caches blocks at schedule time)

def _pair(bs):
    shared = list(range(1000, 1000 + bs))
    return shared + [1, 2, 3], shared + [7, 8, 9, 10]


def test_same_step_shared_prefix_accepted():
    """Two rows share the first block in ONE step: stage 0 ships the second
    with ctx=BS and only its suffix; the first row writes that block now."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    pa, pb = _pair(BS)
    miss = _run_step(al, g, ["a", "b"], [0, BS], "prefill", [pa, pb],
                     [len(pa), len(pb) - BS])
    assert miss == []
    # both rows map the shared block to the same physical block
    assert al.reqs["a"].blocks[0] == al.reqs["b"].blocks[0] != 0
    # order inside the step does not matter
    al2, g2 = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    assert _run_step(al2, g2, ["b", "a"], [BS, 0], "prefill", [pb, pa],
                     [len(pb) - BS, len(pa)]) == []


def test_same_step_chain_of_producers():
    """c relies on b's block 2, b relies on a's block 1 (fixpoint)."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    base = list(range(500, 500 + 3 * BS))
    pa = base[:BS] + [1]
    pb = base[:2 * BS] + [2]
    pc = base[:3 * BS] + [3]
    assert _run_step(al, g, ["c", "b", "a"], [2 * BS, BS, 0], "prefill",
                     [pc, pb, pa], [BS + 1, BS + 1, BS + 1]) == []


def test_true_miss_after_failover_still_flagged():
    """Fresh replica (empty cache): a shared-prefix row with no producer in
    the step, or whose producer is itself missing, must be flagged."""
    pa, pb = _pair(BS)
    fresh, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    assert _run_step(fresh, g, ["b"], [BS], "prefill", [pb], [len(pb) - BS]) == ["b"]
    fresh2, g2 = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    # 'a' is a chunk continuation (ctx=BS) this replica never saw -> missing;
    # it cannot vouch for the shared block, so 'b' is missing too
    assert sorted(_run_step(fresh2, g2, ["a", "b"], [BS, BS], "prefill", [pa, pb],
                            [len(pa) - BS, len(pb) - BS])) == ["a", "b"]
    assert not fresh2.computed
    # a producer that only schedules part of the shared block produces nothing
    fresh3, g3 = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    assert _run_step(fresh3, g3, ["a", "b"], [0, BS], "prefill", [pa, pb],
                     [BS - 1, len(pb) - BS]) == ["b"]


def test_k_gt_1_inflight_steps_no_cascade():
    """K>1: steps N..N+3 are in flight. The same-step pair at N must not be
    flagged, so its later in-flight decode steps are not flagged either; a
    consumer in step N+1 (stage 0 cached the block when scheduling N) is
    served by alloc.computed because this stage finishes N before checking N+1."""
    al, g = _alloc(), RP.ConnexPPKVGuard(A.connex_pp_hash_block)
    pa, pb = _pair(BS)
    assert _run_step(al, g, ["a", "b"], [0, BS], "prefill", [pa, pb],
                     [len(pa), len(pb) - BS]) == []
    pc = pa[:BS] + [42, 43]
    assert _run_step(al, g, ["a", "b", "c"], [len(pa), len(pb), BS], "prefill",
                     [pa + [5], pb + [6], pc], [1, 1, len(pc) - BS]) == []
    for k in range(3):
        assert _run_step(al, g, ["a", "b", "c"],
                         [len(pa) + 1 + k, len(pb) + 1 + k, len(pc) + k],
                         "decode") == []
    assert g.stats["missing"] == 0
