# SPDX-License-Identifier: Apache-2.0
"""Tensor parallelism in PD (connex.pd) on CPU: TP plans, KV head re-layout
between different prefill and decode TP sizes, multi-reader releases."""
import time

import pytest
import torch

from connex.pd.dataplane import LoopbackPlane
from connex.pd.kv_layout import TPSource, head_read_ops, read_ops, tensor_layout, tp_plan
from connex.pd.transfer import KVConsumer, KVProducer

NB, BS, H, D = 8, 4, 8, 16   # blocks, tokens per block, total KV heads, head dim
LAYERS = 2


def full_cache(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, NB, BS, H, D, generator=g).to(torch.float16) for _ in range(LAYERS)]


def shard(full, rank, tp, hnd):
    """The rank's KV cache in vLLM's logical shape [2, NB, BS, H/tp, D], stored
    head-major (HND, LBHNC) or token-major (NHD)."""
    h = H // tp
    out = []
    for t in full:
        s = t[:, :, :, rank * h:(rank + 1) * h, :]
        if hnd:   # physical [2, NB, h, BS, D], viewed back as logical
            out.append(s.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4))
        else:
            out.append(s.contiguous())
    return out


def layouts(ts):
    return [tensor_layout(t, NB, heads_dims=(-3, -2)) for t in ts]


# -- plans --------------------------------------------------------------------

def test_tp_plan_equal_sizes_reads_own_rank():
    for r in range(4):
        assert tp_plan(r, 4, 4, 8) == [TPSource(r, 2, 0, 0, True, 1)]


def test_tp_plan_decode_smaller_reads_every_prefill_rank():
    assert tp_plan(0, 1, 2, 8) == [TPSource(0, 4, 0, 0, True, 1), TPSource(1, 4, 4, 0, True, 1)]
    plan = tp_plan(1, 2, 4, 8)
    assert [(s.remote_rank, s.nheads, s.local_head0) for s in plan] == [(2, 2, 0), (3, 2, 2)]


def test_tp_plan_decode_larger_shares_a_prefill_rank():
    p0, p1 = tp_plan(0, 2, 1, 8), tp_plan(1, 2, 1, 8)
    assert p0 == [TPSource(0, 4, 0, 0, True, 2)]
    assert p1 == [TPSource(0, 4, 0, 4, False, 2)]   # second half of rank 0, not the releaser


def test_tp_plan_covers_every_head_exactly_once():
    for lt in (1, 2, 4, 8):
        for rt in (1, 2, 4, 8):
            seen = []
            for r in range(lt):
                for s in tp_plan(r, lt, rt, 8):
                    seen += [s.remote_rank * (8 // rt) + s.remote_head0 + i for i in range(s.nheads)]
            assert sorted(seen) == list(range(8)), (lt, rt)


def test_tp_plan_rejects_replicated_heads():
    with pytest.raises(ValueError):
        tp_plan(0, 4, 1, 2)
    with pytest.raises(ValueError):
        tp_plan(2, 2, 1, 8)


# -- geometry and ops ---------------------------------------------------------

def test_head_geometry_of_hnd_and_nhd():
    hnd = layouts(shard(full_cache(), 0, 2, True))[0]
    assert hnd.head_major and hnd.heads == 4 and hnd.head_stride == BS * D * 2
    nhd = layouts(shard(full_cache(), 0, 2, False))[0]
    assert not nhd.head_major and nhd.head_stride == D * 2 and nhd.token_stride == 4 * D * 2
    d = hnd.to_dict()
    assert type(hnd).from_dict(d) == hnd
    d.pop("heads")
    d["unknown_future_key"] = 1
    assert type(hnd).from_dict(d).heads == 0   # older / newer peers still parse


def test_hnd_head_range_is_one_read_per_block_segment():
    lo = layouts(shard(full_cache(), 0, 1, True))
    ro = layouts(shard(full_cache(), 1, 2, True))
    ops = head_read_ops(lo, ro, [0, 1], [3, 4], local_head0=4, remote_head0=0, nheads=4)
    assert len(ops) == LAYERS * 2 * 2          # layers x K/V x blocks
    assert all(n == 4 * BS * D * 2 for _, _, n in ops)


def test_full_shard_ops_merge_like_read_ops():
    a, b = layouts(shard(full_cache(), 0, 2, True)), layouts(shard(full_cache(), 0, 2, True))
    assert head_read_ops(a, b, [0, 1, 2], [4, 5, 6], 0, 0, 4) == read_ops(a, b, [0, 1, 2], [4, 5, 6])


def test_head_ops_reject_mismatched_geometry():
    a = layouts(shard(full_cache(), 0, 2, True))
    with pytest.raises(ValueError):
        head_read_ops(a, a, [0], [0], 2, 0, 4)    # past the shard
    plain = [tensor_layout(t, NB) for t in shard(full_cache(), 0, 2, True)]
    with pytest.raises(ValueError):
        head_read_ops(plain, a, [0], [0], 0, 0, 4)


# -- end to end over the loopback plane --------------------------------------

def wait_done(c, n=1, timeout=5.0):
    out, t0 = [], time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        out += c.take_done()
        time.sleep(0.005)
    return out


def run_pd(p_tp, d_tp, p_hnd=True, d_hnd=True, release_after_pull=False):
    """Prefill group of TP p_tp holds blocks 1..3 of transfer T; every decode
    rank pulls them into its blocks 4..6. Returns (full, decode shards,
    producers, consumers, addrs)."""
    full = full_cache(seed=p_tp * 10 + d_tp)
    prods, addrs = [], []
    for r in range(p_tp):
        ts = shard(full, r, p_tp, p_hnd)
        prod = KVProducer(LoopbackPlane(), layouts(ts), bind_ip="127.0.0.1", retain_s=30,
                          block_size=BS)
        prod._keep = ts
        prod.hold("req", "T", [1, 2, 3])
        prods.append(prod)
        addrs.append(f"127.0.0.1:{prod.port}")
    dsh, cons = [], []
    for r in range(d_tp):
        ts = [torch.zeros_like(t) for t in shard(full, r, d_tp, d_hnd)]
        if d_hnd:
            ts = [t.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4) for t in ts]
        c = KVConsumer(LoopbackPlane(), layouts(ts), block_size=BS, ctrl_timeout_ms=500,
                       acquire_wait_s=0.2, release_after_pull=release_after_pull,
                       who=f"d/{r}")
        srcs = [(addrs[s.remote_rank], s) for s in tp_plan(r, d_tp, p_tp, H)]
        c.pull("req", "", "T", [4, 5, 6], 0, sources=srcs)
        dsh.append(ts)
        cons.append(c)
    return full, dsh, prods, cons, addrs


@pytest.mark.parametrize("p_tp,d_tp", [(1, 1), (2, 2), (2, 1), (1, 2), (4, 2), (2, 4), (8, 1)])
@pytest.mark.parametrize("p_hnd,d_hnd", [(True, True), (False, False), (True, False)])
def test_heterogeneous_tp_pull_relayouts_heads(p_tp, d_tp, p_hnd, d_hnd):
    full, dsh, prods, cons, _ = run_pd(p_tp, d_tp, p_hnd, d_hnd)
    try:
        for r, c in enumerate(cons):
            (res,) = wait_done(c)
            assert res.ok, res.why
            want = shard(full, r, d_tp, False)
            for got, w in zip(dsh[r], want):
                assert torch.equal(got[:, 4:7], w[:, 1:4])
                assert not got[:, :4].any() and not got[:, 7:].any()
    finally:
        for x in prods + cons:
            x.close()


def test_hold_freed_only_after_every_reader_released():
    full, dsh, prods, cons, addrs = run_pd(1, 2)
    try:
        for c in cons:
            assert wait_done(c)[0].ok
        plan = [tp_plan(r, 2, 1, H)[0] for r in range(2)]
        cons[0].release(addrs[0], "T", plan[0])
        time.sleep(0.1)
        assert prods[0].held() == 1 and prods[0].take_finished() == set()
        cons[0].release(addrs[0], "T", plan[0])     # same reader twice: still held
        time.sleep(0.1)
        assert prods[0].held() == 1
        cons[1].release(addrs[0], "T", plan[1])
        time.sleep(0.1)
        assert prods[0].held() == 0 and prods[0].take_finished() == {"req"}
    finally:
        for x in prods + cons:
            x.close()


def test_release_after_pull_with_shared_prefill_rank():
    full, dsh, prods, cons, _ = run_pd(1, 2, release_after_pull=True)
    try:
        for c in cons:
            assert wait_done(c)[0].ok
        time.sleep(0.1)
        assert prods[0].take_finished() == {"req"}
    finally:
        for x in prods + cons:
            x.close()


def test_one_dead_prefill_rank_fails_the_whole_pull():
    full, dsh, prods, cons, _ = run_pd(2, 1)
    try:
        assert wait_done(cons[0])[0].ok
        prods[1].close()                 # rank 1 of the prefill group dies
        cons[0].pull("req2", "", "T", [0], 0,
                     sources=[(f"127.0.0.1:{p.port}", s)
                              for p, s in zip(prods, tp_plan(0, 1, 2, H))])
        (res,) = wait_done(cons[0])
        assert not res.ok and "TP rank 1" in res.why and res.failed_blocks == [0]
    finally:
        for x in prods + cons:
            x.close()


def _block_first(full, rank, tp, head_major):
    """vLLM 0.32 per-layer tensors: LBHNC [NB, h, BS, 2D] or LBNHC [NB, BS, h, 2D]
    (K and V of a head and token side by side)."""
    h = H // tp
    out = []
    for t in full:
        s = t[:, :, :, rank * h:(rank + 1) * h, :]                 # [2, NB, BS, h, D]
        kv = torch.cat([s[0], s[1]], dim=-1)                       # [NB, BS, h, 2D]
        out.append(kv.permute(0, 2, 1, 3).contiguous() if head_major else kv.contiguous())
    return out


@pytest.mark.parametrize("p_tp,d_tp", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("p_hm,d_hm", [(True, True), (True, False)])
def test_block_first_kv_concat_layouts(p_tp, d_tp, p_hm, d_hm):
    full = full_cache(seed=7)

    def lay(ts, hm):
        return [tensor_layout(t, NB, heads_dims=(2, 1) if hm else (1, 2)) for t in ts]

    prods, addrs = [], []
    for r in range(p_tp):
        ts = _block_first(full, r, p_tp, p_hm)
        prod = KVProducer(LoopbackPlane(), lay(ts, p_hm), bind_ip="127.0.0.1", block_size=BS)
        prod._keep = ts
        prod.hold("req", "T", [1, 2, 3])
        prods.append(prod)
        addrs.append(f"127.0.0.1:{prod.port}")
    try:
        for r in range(d_tp):
            ts = [torch.zeros_like(t) for t in _block_first(full, r, d_tp, d_hm)]
            lo = lay(ts, d_hm)
            assert lo[0].head_bytes == 2 * D * 2 and lo[0].heads == H // d_tp
            c = KVConsumer(LoopbackPlane(), lo, block_size=BS, acquire_wait_s=0.2, who=f"d{r}")
            c.pull("req", "", "T", [4, 5, 6], 0,
                   sources=[(addrs[s.remote_rank], s) for s in tp_plan(r, d_tp, p_tp, H)])
            (res,) = wait_done(c)
            c.close()
            assert res.ok, res.why
            want = _block_first(full, r, d_tp, d_hm)
            for got, w in zip(ts, want):
                assert torch.equal(got[4:7], w[1:4]) and not got[:4].any()
    finally:
        for p in prods:
            p.close()


def test_find_heads_dims():
    from connex.pd.kv_layout import find_heads_dims

    assert find_heads_dims((5140, 8, 16, 256), 5140, 8, 16, "LBHNC") == (2, 1)
    assert find_heads_dims((5140, 16, 8, 256), 5140, 8, 16, "LBNHC") == (1, 2)
    assert find_heads_dims((2, 5140, 16, 8, 128), 5140, 8, 16) == (2, 3)
    assert find_heads_dims((5140, 16, 16, 256), 5140, 16, 16, "LBHNC") == (2, 1)
    assert find_heads_dims((5140, 16, 16, 256), 5140, 16, 16, "LBNHC") == (1, 2)
    assert find_heads_dims((5140, 4, 16, 256), 5140, 8, 16) is None
    assert find_heads_dims((5140, 8, 16, 256), 999, 8, 16) is None
