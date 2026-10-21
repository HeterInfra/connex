# SPDX-License-Identifier: Apache-2.0
"""External-PP vLLM patches: apply/idempotency on a copy of the fork, and the
pure allocator logic that patch 1 inlines."""

import ast
import hashlib
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
FORK = os.path.join(FORK_V1, "worker", "gpu_model_runner.py")
BASE = "fast_external_pp_block_table.py"
MINE = [
    "patch_external_pp_allocator.py",
    "patch_external_pp_host_metadata.py",
    "patch_external_pp_pinned_upload.py",
    "patch_external_pp_fast_hash.py",
    "patch_external_pp_lazy_logging.py",
]


def _load_alloc():
    spec = importlib.util.spec_from_file_location("external_pp_alloc", os.path.join(PATCHES, "external_pp_alloc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_alloc()


def _run(name, path):
    return subprocess.run([sys.executable, os.path.join(PATCHES, name), path],
                          capture_output=True, text=True)


@pytest.fixture
def runner_copy(tmp_path):
    if not os.path.exists(FORK):
        pytest.skip("fork source not available")
    dst = tmp_path / "gpu_model_runner.py"
    shutil.copy(FORK, dst)
    assert _run(BASE, str(dst)).returncode == 0
    return str(dst)


# ---------------------------------------------------------------- patch files

def test_apply_all_and_idempotent(runner_copy):
    for _ in range(2):
        p = subprocess.run([sys.executable, os.path.join(PATCHES, "apply_all.py"), runner_copy],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
    src = open(runner_copy).read()
    ast.parse(src)
    before = src
    for name in MINE:
        r = _run(name, runner_copy)
        assert r.returncode == 0 and "already patched" in r.stdout, r.stdout
    assert open(runner_copy).read() == before
    # the fork's padding allocation and batch cache are gone from the live path
    assert "ConnexPPBlockAllocator(" in src
    assert "self._external_pp_batch_blk_table_cache[batch_key][kv_cache_gid]" not in src
    assert "seq_lens = torch.tensor(num_scheduled_tokens_per_req, dtype=torch.int32, device=self.device)" not in src
    assert "_CONNEX_PP_FAST_HASH = True" in src


def test_inlined_allocator_compiles_and_runs(runner_copy):
    assert _run(MINE[0], runner_copy).returncode == 0
    src = open(runner_copy).read()
    b = src.index("# BEGIN connex external-pp allocator")
    e = src.index("# END connex external-pp allocator")
    ns = {}
    exec(compile(src[b:e], "inlined", "exec"), ns)
    al = ns["ConnexPPBlockAllocator"](32)
    rows, *_ = al.prefill(["r"], [list(range(40))], [0], [40], 16, 8)
    assert rows.shape == (1, 8)


@pytest.mark.parametrize("name", [MINE[0], MINE[1], MINE[3], MINE[4]])
def test_each_patch_alone(runner_copy, name):
    for _ in range(2):
        r = _run(name, runner_copy)
        assert r.returncode == 0, r.stdout
    ast.parse(open(runner_copy).read())


def test_pinned_upload_requires_allocator(runner_copy):
    assert _run(MINE[2], runner_copy).returncode == 1
    assert _run(MINE[0], runner_copy).returncode == 0
    assert _run(MINE[2], runner_copy).returncode == 0


def test_unknown_source_fails(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("print(1)\n")
    for name in MINE:
        assert _run(name, str(f)).returncode == 1
    assert f.read_text() == "print(1)\n"


# ---------------------------------------------------------------- allocator

BS, W = 16, 64


def _check_no_sharing(al):
    """Private blocks are owned by one request; hashed blocks only shared by hash."""
    owner = {}
    for rid, r in al.reqs.items():
        for b, h in zip(r.blocks, r.hashes):
            if b == 0:
                continue
            key = ("h", h) if h is not None else ("p", rid)
            assert owner.setdefault(b, key) == key, f"block {b} shared: {owner[b]} vs {key}"
            if h is None:
                assert b not in al.block_to_hash
    free = set(al._free) | set(al.evictable)
    assert not (free & al.live_blocks())


def test_padding_is_null_block_and_no_overallocation():
    al = A.ConnexPPBlockAllocator(1000)
    rows, hits, q, comp = al.prefill(["a"], [list(range(40))], [0], [40], BS, W)
    # 2 full hashed blocks + 1 private partial block; the rest padding 0
    assert list(rows[0, :3]) == [1, 2, 3] and not rows[0, 3:].any()
    assert (hits, q, comp) == (0, 2, [0])
    assert al.num_free() == 999 - 3


def test_decode_grows_lazily_across_boundary():
    al = A.ConnexPPBlockAllocator(1000)
    al.prefill(["a"], [list(range(30))], [0], [30], BS, W)
    assert len(al.reqs["a"].blocks) == 2          # positions 0..30
    rows, pos = al.decode(["a"], BS, W)           # writes pos 30, covers 31
    assert pos == [30] and len(al.reqs["a"].blocks) == 2
    rows, pos = al.decode(["a"], BS, W)           # pos 31, covers 32 -> new block
    assert pos == [31] and len(al.reqs["a"].blocks) == 3 and rows[0, 2] != 0
    assert al.peek_positions(["a"]) == [32]
    assert al.peek_positions(["a", "zz"]) is None


def test_prefix_hit_refcount_and_eviction():
    al = A.ConnexPPBlockAllocator(1000)
    p = list(range(48))
    r1, *_ = al.prefill(["a"], [p + [1]], [0], [49], BS, W)
    al.mark_computed(["a"], BS)
    r2, hits, q, comp = al.prefill(["b"], [p + [2]], [0], [49], BS, W)
    assert hits == 3 and comp == [48]
    assert list(r1[0, :3]) == list(r2[0, :3]) and r1[0, 3] != r2[0, 3]
    blk = r1[0, 0]
    al.release("a")
    assert al.ref[blk] == 1
    al.release("b")
    assert al.ref[blk] == 0 and blk in al.evictable
    # still a hit while cached
    _, hits, _, comp = al.prefill(["c"], [p], [0], [48], BS, W)
    assert hits == 3 and comp == [48]
    _check_no_sharing(al)


def test_mark_computed_respects_chunked_prefill():
    al = A.ConnexPPBlockAllocator(1000)
    toks = list(range(64))
    al.prefill(["a"], [toks], [0], [20], BS, W)   # first chunk: only 20 tokens computed
    al.mark_computed(["a"], BS)
    assert len(al.computed) == 1
    al.prefill(["a"], [toks], [20], [44], BS, W)
    al.mark_computed(["a"], BS)
    assert len(al.computed) == 4


def test_eviction_drops_computed_and_mapping():
    al = A.ConnexPPBlockAllocator(5)                # blocks 1..4
    al.prefill(["a"], [list(range(32))], [0], [32], BS, W)   # 2 hashed + 1 private
    al.mark_computed(["a"], BS)
    al.release("a")
    assert len(al.evictable) == 2
    al.prefill(["b"], [list(range(100, 164))], [0], [64], BS, W)
    assert al.stats["evictions"] == 2
    assert not al.computed and len(al.hash_to_block) == 4
    _check_no_sharing(al)


def test_exhaustion_reclaims_idle_not_current_batch():
    al = A.ConnexPPBlockAllocator(7)                # 6 usable
    al.prefill(["old"], [list(range(40))], [0], [40], BS, W)   # 3 blocks
    al.prefill(["cur"], [list(range(200, 240))], [0], [40], BS, W)
    assert al.num_free() == 0
    # cur grows while old is idle -> old reclaimed, cur keeps its blocks
    cur_blocks = list(al.reqs["cur"].blocks)
    for _ in range(10):
        al.decode(["cur"], BS, W)
    assert "old" not in al.reqs and al.stats["reclaimed_reqs"] == 1
    assert al.reqs["cur"].blocks[:3] == cur_blocks
    _check_no_sharing(al)
    # a single batch bigger than the pool degrades to null blocks, never steals
    rows, _ = al.decode(["cur"] * 1, BS, W, counts=[200])
    assert al.stats["exhausted"] > 0 and 0 in set(al.reqs["cur"].blocks)


def test_ttl_release():
    t = [0.0]
    al = A.ConnexPPBlockAllocator(100, ttl_s=10, clock=lambda: t[0])
    al.prefill(["a"], [list(range(20))], [0], [20], BS, W)
    t[0] = 100
    for _ in range(64):
        al.decode(["b"], BS, W)
    assert "a" not in al.reqs and "b" in al.reqs


def test_randomized_never_hands_out_live_blocks():
    rng = random.Random(0)
    al = A.ConnexPPBlockAllocator(120, ttl_s=0)
    prompts = [[rng.randrange(50) for _ in range(rng.randrange(1, 70))] for _ in range(8)]
    active = {}
    for step in range(3000):
        op = rng.random()
        if op < 0.2 and len(active) < 12:
            rid = f"r{step}"
            toks = list(rng.choice(prompts)) + [rng.randrange(50) for _ in range(rng.randrange(0, 20))]
            al.prefill([rid], [toks], [0], [len(toks)], 8, 32)
            al.mark_computed([rid], 8)
            active[rid] = toks
        elif op < 0.3 and active:
            rid = rng.choice(list(active))
            al.release(rid)
            del active[rid]
        elif active:
            batch = rng.sample(list(active), min(len(active), 4))
            al.decode(batch, 8, 32)
        for rid in list(active):
            if rid not in al.reqs:     # reclaimed under pressure
                del active[rid]
        _check_no_sharing(al)
        assert len(al.computed) <= len(al.hash_to_block)


def test_hash_cache_matches_fork_scheme():
    toks = list(range(100))
    # fork: sha256(prev + str(tuple(block)))
    prev, ref = b"NONE_HASH", []
    for i in range(len(toks) // BS):
        prev = hashlib.sha256(prev + str(tuple(toks[i * BS:(i + 1) * BS])).encode()).digest()
        ref.append(prev)
    al = A.ConnexPPBlockAllocator(100)
    r = al._get_req("x", W)
    assert al.token_hashes(r, toks[:40], BS) == ref[:2]
    assert al.token_hashes(r, toks, BS) == ref          # incremental extension
    assert al.token_hashes(r, toks[:20], BS) == ref[:1]  # shrink recomputes tail


def test_fast_hash_is_faster_and_distinct():
    toks = list(range(16))
    old = A._CONNEX_PP_FAST_HASH
    try:
        A._CONNEX_PP_FAST_HASH = False
        t0 = time.perf_counter()
        for _ in range(3000):
            slow = A.connex_pp_hash_block(b"x", toks)
        t_slow = time.perf_counter() - t0
        A._CONNEX_PP_FAST_HASH = True
        t0 = time.perf_counter()
        for _ in range(3000):
            fast = A.connex_pp_hash_block(b"x", toks)
        t_fast = time.perf_counter() - t0
        assert fast != slow and len(fast) == 16
        assert A.connex_pp_hash_block(b"x", toks[:15] + [99]) != fast
        assert t_fast < t_slow
    finally:
        A._CONNEX_PP_FAST_HASH = old
