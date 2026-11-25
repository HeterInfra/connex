# SPDX-License-Identifier: Apache-2.0
"""Regression tests: PD hold/release race, hold expiry scan, ordering of
direct reads after the engine stream, early refusal of sub-threshold head
reads, EP move poll backoff and the per-layer replica index of plan_repair."""
import random
import time

import pytest
import torch

import connex.pd.transfer as transfer
from connex.ep.placement import UP, Move, PlacementTable
from connex.pd.dataplane import ZCOPY_THRESH, LoopbackPlane
from connex.pd.kv_layout import TensorLayout, head_read_min, head_read_ops, tensor_layout
from connex.pd.transfer import KVConsumer, KVProducer

NB, BS, H, D = 8, 4, 8, 16


def caches(n=2, fill=False):
    out = []
    for _ in range(n):
        t = torch.zeros(2, NB, BS, H, D, dtype=torch.float16)
        if fill:
            t.copy_(torch.randn(t.shape).to(torch.float16))
        out.append(t)
    return out


def wait_done(c, n=1, timeout=5.0):
    out, t0 = [], time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        out += c.take_done()
        time.sleep(0.005)
    return out


# -- KVProducer.hold vs a release while stage_out runs --------------------------

class _ReleasingStaging:
    """stage_out stand-in that lets a release arrive while hold() has dropped
    its lock (the decode side released before the hold was recorded)."""

    def __init__(self):
        self.prod = None
        self.tid = None

    def stage_out(self, ranges):
        assert self.prod._handle({"op": "release", "tid": self.tid}) == {"ok": True}
        return None

    def translate(self, lo):
        return lo


def test_release_during_stage_out_frees_the_hold():
    ts = caches()
    plane = LoopbackPlane()
    st = _ReleasingStaging()
    plane._connex_staging = st
    prod = KVProducer(plane, [tensor_layout(t, NB) for t in ts], bind_ip="127.0.0.1",
                      retain_s=60, register=False)
    try:
        st.prod, st.tid = prod, "T"
        prod.hold("req", "T", [1, 2])
        assert prod.held() == 0                      # not pinned for retain_s
        assert prod.take_finished() == {"req"}
        assert "T" not in prod._released            # nor left behind
    finally:
        prod.close()


# -- take_finished: scan only when a deadline may have passed -------------------

def test_take_finished_expiry_with_renewed_deadlines():
    ts = caches()
    prod = KVProducer(LoopbackPlane(), [tensor_layout(t, NB) for t in ts], bind_ip="127.0.0.1",
                      retain_s=0.4)
    try:
        prod.hold("a", "A", [0])                     # deadline 0.4
        time.sleep(0.2)
        prod.hold("b", "B", [1])                     # deadline 0.6
        assert prod.take_finished() == set()
        time.sleep(0.15)
        assert prod._handle({"op": "acquire", "tid": "A"})["ok"]   # A renewed to 0.75
        time.sleep(0.3)
        assert prod.take_finished() == {"b"}         # t=0.65: A's first deadline passed
        time.sleep(0.15)
        assert prod.take_finished() == {"a"} and prod.held() == 0
    finally:
        prod.close()


def test_take_finished_does_not_scan_live_holds():
    ts = caches()
    prod = KVProducer(LoopbackPlane(), [tensor_layout(t, NB) for t in ts], bind_ip="127.0.0.1",
                      retain_s=60)
    try:
        for i in range(20000):
            prod.hold(f"r{i}", f"t{i}", [0])
        t0 = time.perf_counter()
        for _ in range(100):
            assert prod.take_finished() == set()
        per_call = (time.perf_counter() - t0) / 100
        assert per_call < 1e-3, per_call     # a full scan of 20k holds takes ~5 ms
    finally:
        prod.close()


# -- direct device reads wait for the engine stream -----------------------------

class _Ev:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("sync")


class _LoggingPlane(LoopbackPlane):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def read(self, peer_url, ops):
        self.log.append("read")
        return super().read(peer_url, ops)


@pytest.mark.parametrize("tp_sources", [False, True])
def test_direct_reads_posted_after_engine_event(tp_sources):
    src, dst = caches(fill=True), caches()
    log = []
    prod = KVProducer(LoopbackPlane(), [tensor_layout(t, NB, heads_dims=(-3, -2)) for t in src],
                      bind_ip="127.0.0.1", retain_s=30, block_size=BS)
    cons = KVConsumer(_LoggingPlane(log), [tensor_layout(t, NB, heads_dims=(-3, -2)) for t in dst],
                      block_size=BS, ctrl_timeout_ms=500, acquire_wait_s=0.2)
    try:
        assert cons._order_reads is False            # host tensors: nothing to order
        cons._order_reads = True                     # as for device KV memory
        cons._engine_event = lambda: (log.append("record"), _Ev(log))[1]
        prod.hold("req", "T", [1, 2])
        addr = f"127.0.0.1:{prod.port}"
        srcs = None
        if tp_sources:
            srcs = [(addr, transfer.TPSource(0, H, 0, 0, True, 1))]
        cons.pull("req", addr, "T", [5, 6], sources=srcs)
        (res,) = wait_done(cons)
        assert res.ok, res.why
        assert log[0] == "record" and log.index("sync") < log.index("read")
        assert torch.equal(dst[0][:, 5:7], src[0][:, 1:3])
    finally:
        prod.close()
        cons.close()


# -- head reads below the connex-rs zero-copy threshold -------------------------

def _nhd(base, heads, tokens=16, head_bytes=256, nb=64):
    seg = heads * tokens * head_bytes
    return TensorLayout(base=base, outer=2, outer_stride=nb * seg, block_stride=seg,
                        seg_bytes=seg, num_blocks=nb, heads=heads, head_stride=head_bytes,
                        tokens=tokens, token_stride=heads * head_bytes, head_bytes=head_bytes)


def _hnd(base, heads, tokens=16, head_bytes=256, nb=64):
    seg = heads * tokens * head_bytes
    return TensorLayout(base=base, outer=2, outer_stride=nb * seg, block_stride=seg,
                        seg_bytes=seg, num_blocks=nb, heads=heads,
                        head_stride=tokens * head_bytes, tokens=tokens,
                        token_stride=head_bytes, head_bytes=head_bytes)


@pytest.mark.parametrize("lmk,rmk", [(_nhd, _nhd), (_hnd, _hnd), (_hnd, _nhd), (_nhd, _hnd)])
@pytest.mark.parametrize("lh0,rh0,nh", [(0, 0, 4), (0, 4, 4), (2, 0, 2), (0, 0, 8)])
def test_head_read_min_is_a_read_that_is_emitted(lmk, rmk, lh0, rh0, nh):
    lo = [lmk(1 << 30, 8), lmk(1 << 32, 8)]
    ro = [rmk(1 << 34, 8), rmk(1 << 36, 8)]
    for local, remote in (([3], [9]), ([0, 1, 2, 7], [4, 5, 6, 8]), (list(range(10)), list(range(10)))):
        ops = head_read_ops(lo, ro, local, remote, lh0, rh0, nh)
        m = head_read_min(lo, ro, lh0, rh0, nh)
        lens = {n for _, _, n in ops}
        if m is not None:
            assert m in lens and min(lens) <= m


class _RsLikePlane(LoopbackPlane):
    name = "connex-rs"


def test_sub_threshold_head_read_refused_before_building_ops(monkeypatch):
    # NHD on both sides, half of the heads: one 128 B read per token
    full = [torch.randn(2, NB, BS, H, D).to(torch.float16) for _ in range(2)]
    dst = [torch.zeros(2, NB, BS, H // 2, D, dtype=torch.float16) for _ in range(2)]
    prod = KVProducer(LoopbackPlane(), [tensor_layout(t, NB, heads_dims=(-3, -2)) for t in full],
                      bind_ip="127.0.0.1", retain_s=30, block_size=BS)
    cons = KVConsumer(_RsLikePlane(), [tensor_layout(t, NB, heads_dims=(-3, -2)) for t in dst],
                      block_size=BS, ctrl_timeout_ms=500, acquire_wait_s=0.2)

    def boom(*a, **k):
        raise AssertionError("op list built")

    monkeypatch.setattr(transfer, "head_read_ops", boom)
    try:
        prod.hold("req", "T", [1, 2])
        addr = f"127.0.0.1:{prod.port}"
        cons.pull("req", addr, "T", [5, 6], sources=[(addr, transfer.TPSource(0, 4, 0, 4, True, 1))])
        (res,) = wait_done(cons)
        assert not res.ok and "zero-copy threshold" in res.why and str(ZCOPY_THRESH) in res.why
        assert cons.plane.reads == 0
    finally:
        prod.close()
        cons.close()


# -- EP: async move polling backs off ---------------------------------------------

def test_vllm_ep_move_poll_backs_off():
    from connex.ep.vllm_ext import VLLMEPEngine

    eng = VLLMEPEngine("http://127.0.0.1:1", timeout=10)
    calls = []
    t_done = time.time() + 0.6

    def rpc(method, *args):
        calls.append(method)
        if method == "connex_ep_move_poll":
            done = time.time() >= t_done
            return [{"rank": 0, "done": done, "seconds": 0.6, "results": [], "error": None}]
        return [{"rank": 0}]

    eng.rpc = rpc
    t0 = time.time()
    assert eng.move([], {}, 4) == []
    assert time.time() - t0 < 0.6 + 0.15            # the backoff cap bounds the extra wait
    polls = calls.count("connex_ep_move_poll")
    assert polls <= 15, polls                        # a fixed 5 ms poll makes ~100 calls


# -- EP: plan_repair with a per-layer replica index ----------------------------------

def _plan_repair_reference(t, min_replicas=1, prefer_src_rank=None):
    """plan_repair as it was (one scan of the row per expert)."""
    moves = []
    for layer in range(t.num_layers):
        row = t.slot_map[layer]
        reserved = set()
        up_reps = {e: [p for p, x in enumerate(row) if x == e and (layer, p) not in t.staging
                       and t.rank_state[t.rank_of(p)] == UP] for e in range(t.num_experts)}
        need = sorted((e for e in range(t.num_experts) if len(up_reps[e]) < min_replicas),
                      key=lambda e: len(up_reps[e]))
        for e in need:
            while len(up_reps[e]) < min_replicas:
                dst = t._pick_dst(layer, e, up_reps, reserved)
                if dst is None:
                    break
                src = t._pick_src(layer, e, dst, prefer_src_rank)
                old = row[dst]
                if old >= 0 and dst in up_reps.get(old, []):
                    up_reps[old].remove(dst)
                reserved.add(dst)
                up_reps[e].append(dst)
                moves.append(Move(layer, e, dst, src, old))
    return moves


@pytest.mark.parametrize("seed", range(6))
def test_plan_repair_matches_reference(seed):
    rnd = random.Random(seed)
    E, R, S, L = 16, 4, 6, 3
    t = PlacementTable.spread(E, R, S, num_layers=L)
    for layer in range(L):
        for p in rnd.sample(range(R * S), 4):
            t.slot_map[layer][p] = -1
    t.staging = {(rnd.randrange(L), rnd.randrange(R * S)) for _ in range(3)}
    t.mark_down(rnd.randrange(R))
    t.mark_draining(rnd.randrange(R))
    for k in (1, 2):
        assert t.plan_repair(k, prefer_src_rank=1) == _plan_repair_reference(t, k, 1)


def test_plan_repair_without_moves_is_cheap():
    t = PlacementTable.linear(256, 32, 9, num_layers=58)   # DeepSeek-V3-like
    t0 = time.perf_counter()
    assert t.plan_repair() == []
    dt = time.perf_counter() - t0
    assert dt < 0.05, dt     # the per-expert scan took ~150 ms
