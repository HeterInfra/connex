"""Host staging copies as one gather + scatter (issue #39): byte-identical
to the per-range copies for fragmented block sets, both directions. The CPU
tests drive the same row plan and kernels over CPU memory; the CUDA tests
check the real device path, its stream use and that stage_out does not wait
for the GPU."""
import ctypes
import random
import time

import numpy as np
import pytest
import torch

from connex.pd import host_staging as hs
from connex.pd.kv_layout import TensorLayout


def _cpu_view(ptr, nbytes, e):
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    t = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
    return t.view(hs._torch_dtype(e)) if e > 1 else t


class _CpuStaging(hs.HostStaging):
    """HostStaging over CPU memory: the "device" spans are carved out of one
    buffer with gaps between them (like separately allocated KV tensors),
    the mirrors out of another with different gaps (like separate pinned
    allocations). Nothing else lives in either view's address range, as with
    device memory vs. pinned host memory, unless ``alloc_in_gap`` puts the
    scratch and index tensors into the gaps between device spans. One copy
    per instance in that mode (the gaps are not reused)."""

    def __init__(self, span_sizes, gap=8192, alloc_in_gap=False, vector=True, mgap=12288):
        self.torch = torch
        self._big = torch.zeros(sum(span_sizes) + gap * (len(span_sizes) + 1) + 8192,
                                dtype=torch.uint8)
        self._mbig = torch.zeros(sum(span_sizes) + mgap * len(span_sizes) + 8192,
                                 dtype=torch.uint8)
        off = (-self._big.data_ptr()) % 4096 + gap
        moff = (-self._mbig.data_ptr()) % 4096
        self._starts, self._spans = [], []
        for n in span_sizes:
            dev = self._big[off:off + n]
            mirror = self._mbig[moff:moff + n]
            self._starts.append(dev.data_ptr())
            self._spans.append((dev.data_ptr(), n, dev, mirror))
            off += n + gap
            moff += n + mgap
        self._alloc_in_gap = alloc_in_gap
        self._used = set()
        self._vec = None
        if vector:
            starts = np.asarray(self._starts, dtype=np.int64)
            lens = np.asarray(span_sizes, dtype=np.int64)
            mptrs = np.asarray([m.data_ptr() for *_, m in self._spans], dtype=np.int64)
            self._vec = hs._Vector(starts, lens, mptrs, _cpu_view, self._alloc, self._upload)

    def _alloc(self, nbytes):
        if not self._alloc_in_gap:
            return torch.empty(nbytes, dtype=torch.uint8)
        # bump-allocate in the gaps between spans, one tensor per gap
        lo = self._big.data_ptr()
        for i in range(len(self._spans) - 1):
            s0, n0 = self._spans[i][:2]
            o = (s0 + n0 - lo + 63) // 64 * 64 + 64
            if i not in self._used and o + nbytes <= self._spans[i + 1][0] - lo:
                self._used.add(i)
                return self._big[o:o + nbytes]
        raise AssertionError("gap too small for the test")

    def _upload(self, idx, out):
        out.copy_(torch.from_numpy(idx))

    def fill(self, seed):
        g = torch.Generator().manual_seed(seed)
        for _, n, dev, m in self._spans:
            dev.copy_(torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g))
            m.copy_(torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g))

    def snapshot(self):
        return [(dev.clone(), m.clone()) for _, _, dev, m in self._spans]


def _layouts(st, nblocks, seg, outer=2):
    """One layout per span: [outer, nblocks] segments of ``seg`` bytes."""
    return [TensorLayout(base=s, outer=outer, outer_stride=nblocks * seg, block_stride=seg,
                         seg_bytes=seg, num_blocks=nblocks) for s in st._starts]


def _reference(st, snap, ranges, to_host):
    """Byte-level expectation: every byte of every range copied across."""
    out = [(d.clone(), m.clone()) for d, m in snap]
    for a, n in ranges:
        i = max(j for j, s in enumerate(st._starts) if s <= a)
        o = a - st._starts[i]
        d, m = out[i]
        if to_host:
            m[o:o + n] = d[o:o + n]
        else:
            d[o:o + n] = m[o:o + n]
    return out


def _same(a, b):
    return all(torch.equal(x, y) and torch.equal(p, q) for (x, p), (y, q) in zip(a, b))


@pytest.mark.parametrize("to_host", [True, False])
@pytest.mark.parametrize("alloc_in_gap", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_vector_matches_per_range_on_random_fragmented_blocks(to_host, alloc_in_gap, seed):
    rnd = random.Random(seed)
    nblocks, seg, layers = 64, rnd.choice([256, 512, 4096]), rnd.choice([3, 8])
    st = _CpuStaging([2 * nblocks * seg] * layers, gap=(4 << 20) if alloc_in_gap else 8192,
                     alloc_in_gap=alloc_in_gap)
    blocks = rnd.sample(range(nblocks), rnd.randint(1, nblocks))
    ranges = hs.block_ranges(_layouts(st, nblocks, seg), blocks)
    if seed % 2:   # stage_in style: pieces of segments (head slices), overlapping
        ranges = [(a + seg // 4, seg // 2) for a, _ in ranges.tolist()] + ranges[:5].tolist()
    st.fill(seed)
    snap = st.snapshot()
    st._copy(ranges, to_host)
    new = st.snapshot()
    assert st._vec.launches > 0
    if alloc_in_gap:
        assert st._vec.pieces >= 2      # the device view was cut around them
    # the old path on the same input
    old = _CpuStaging([2 * nblocks * seg] * layers, vector=False)
    old.fill(seed)
    oranges = hs.block_ranges(_layouts(old, nblocks, seg), blocks)
    if seed % 2:
        oranges = [(a + seg // 4, seg // 2) for a, _ in oranges.tolist()] + oranges[:5].tolist()
    old._copy(oranges, to_host)
    assert _same(new, old.snapshot())
    assert _same(new, _reference(st, snap, ranges, to_host))


def test_launches_are_constant_in_the_number_of_ranges():
    nblocks, seg = 512, 4096
    st = _CpuStaging([2 * nblocks * seg] * 32, gap=1 << 20)
    st.fill(1)
    counts = []
    for nb in (8, 64, 256):   # up to 64 MB: one scratch chunk
        blocks = list(range(0, nblocks, nblocks // nb))[:nb]
        st._copy(hs.block_ranges(_layouts(st, nblocks, seg), blocks), True)
        counts.append(st._vec.launches)
    assert counts[0] == counts[1] == counts[2] == 3   # index upload, gather, scatter


def test_chunks_bound_the_scratch(monkeypatch):
    monkeypatch.setenv("CONNEX_PD_STAGING_CHUNK_MB", "1")
    nblocks, seg = 64, 64 << 10
    st = _CpuStaging([2 * nblocks * seg] * 2)
    st.fill(3)
    snap = st.snapshot()
    ranges = hs.block_ranges(_layouts(st, nblocks, seg), list(range(0, 64, 2)))
    st._copy(ranges, True)
    total = sum(n for _, n in ranges)
    assert st._vec.launches == 1 + 2 * (total >> 20)
    assert _same(st.snapshot(), _reference(st, snap, ranges, True))


def test_adjacent_spans_do_not_merge_and_odd_ranges_fall_back():
    st = _CpuStaging([1 << 16, 1 << 16], gap=0)          # adjacent tensors
    st.fill(5)
    snap = st.snapshot()
    a = st._starts[0] + (1 << 16) - 4096
    ranges = [(a, 4096), (st._starts[1], 4096)]           # touch across the boundary
    st._copy(ranges, True)
    assert st._vec.launches == 3
    assert _same(st.snapshot(), _reference(st, snap, ranges, True))
    # rows below 64 bytes: per-range copies
    st.fill(6)
    snap = st.snapshot()
    ranges = [(st._starts[0] + 3, 5), (st._starts[1] + 7, 11)]
    assert hs.plan_rows(ranges, st._vec.starts, st._vec.lens, st._vec.starts,
                        st._vec.mirror_ptrs, st._vec.dev_base, st._vec.m_base) is None
    st._copy(ranges, False)
    assert _same(st.snapshot(), _reference(st, snap, ranges, False))


def test_per_range_fallback_splits_a_merge_across_adjacent_spans():
    st = _CpuStaging([1 << 12, 1 << 12], gap=0, vector=False)
    st.fill(7)
    snap = st.snapshot()
    ranges = [(st._starts[0] + 2048, 2048), (st._starts[1], 1024)]   # merged by _merge
    st._copy(ranges, True)
    assert _same(st.snapshot(), _reference(st, snap, ranges, True))


def test_env_switch_disables_the_vector_path(monkeypatch):
    monkeypatch.setenv("CONNEX_PD_STAGING_VECTOR", "0")
    assert not hs.vector_enabled()
    monkeypatch.setenv("CONNEX_PD_STAGING_VECTOR", "1")
    assert hs.vector_enabled()


def test_plan_build_is_cheap_on_the_engine_thread():
    """32 layers x K/V x 64 fragmented blocks: the plan costs well under a
    millisecond-scale budget compared to thousands of copy launches."""
    nblocks, seg = 4096, 32 << 10
    starts = np.arange(32, dtype=np.int64) * (2 * nblocks * seg + (2 << 20)) + (1 << 40)
    lens = np.full(32, 2 * nblocks * seg, dtype=np.int64)
    mptrs = np.arange(32, dtype=np.int64) * (4 * nblocks * seg) + (1 << 36)
    los = [TensorLayout(base=int(s), outer=2, outer_stride=nblocks * seg, block_stride=seg,
                        seg_bytes=seg, num_blocks=nblocks) for s in starts]
    blocks = random.Random(0).sample(range(nblocks), 64)
    ranges = hs.block_ranges(los, blocks)
    t0 = time.perf_counter()
    for _ in range(20):
        u, e, idx = hs.plan_rows(ranges, starts, lens, starts, mptrs, int(starts[0]),
                                 int(mptrs[0]))
    dt = (time.perf_counter() - t0) / 20
    assert u == seg and e == 8 and idx.shape == (2, 32 * 2 * 64)
    assert dt < 0.02, dt


# -- GPU ------------------------------------------------------------------------
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _mirrors(st, kvs):
    """HostStaging keeps its spans sorted by address; the allocator need not
    hand out the KV tensors in that order. Spans matching ``kvs`` one to one."""
    by_start = {s[0]: s for s in st._spans}
    return [by_start[t.data_ptr()] for t in kvs]


def _gpu_setup(layers=8, nblocks=64, seg=16 << 10):
    kvs = [torch.randint(0, 256, (2, nblocks, seg), dtype=torch.uint8, device="cuda")
           for _ in range(layers)]
    spans = [(t.data_ptr(), t.numel()) for t in kvs]
    los = [TensorLayout(base=t.data_ptr(), outer=2, outer_stride=nblocks * seg,
                        block_stride=seg, seg_bytes=seg, num_blocks=nblocks) for t in kvs]
    return kvs, spans, los


@cuda
@pytest.mark.parametrize("vector", ["1", "0"])
def test_gpu_stage_out_and_in_match(monkeypatch, vector):
    monkeypatch.setenv("CONNEX_PD_STAGING_VECTOR", vector)
    kvs, spans, los = _gpu_setup()
    st = hs.HostStaging(spans)
    assert (st._vec is not None) == (vector == "1")
    blocks = random.Random(1).sample(range(64), 37)
    ranges = hs.block_ranges(los, blocks)
    st.stage_out(ranges).synchronize()
    for t, (_, _, _, m) in zip(kvs, _mirrors(st, kvs)):
        mv = m.view(t.shape)
        assert torch.equal(mv[:, blocks], t[:, blocks].cpu())
    # consumer direction: new mirror contents land in the blocks only
    for _, _, _, m in st._spans:
        m.copy_(torch.randint(0, 256, m.shape, dtype=torch.uint8))
    before = [t.clone() for t in kvs]
    st.stage_in(ranges)
    other = [b for b in range(64) if b not in blocks]
    for t, b, (_, _, _, m) in zip(kvs, before, _mirrors(st, kvs)):
        mv = m.view(t.shape)
        assert torch.equal(t[:, blocks].cpu(), mv[:, blocks])
        assert torch.equal(t[:, other], b[:, other])


@cuda
def test_gpu_stage_out_does_not_wait_for_the_engine_stream():
    kvs, spans, los = _gpu_setup()
    st = hs.HostStaging(spans)
    assert st._vec is not None
    ranges = hs.block_ranges(los, list(range(0, 64, 3)))
    st.stage_out(ranges).synchronize()               # warm up allocators
    torch.cuda.synchronize()
    torch.cuda._sleep(int(0.3 * 1.4e9))              # engine busy ~300 ms
    kvs[0].fill_(9)                                   # the prefill writes the KV
    t0 = time.perf_counter()
    ev = st.stage_out(ranges)
    dt = time.perf_counter() - t0
    assert dt < 0.1, dt                               # returned while the GPU is busy
    assert not ev.query()                             # ordered after the engine work
    ev.synchronize()
    m0 = st._spans[0][3].view(kvs[0].shape)
    assert int(m0[0, 0, 0]) == 9 and int(m0[1, 63, -1]) == 9


@cuda
def test_gpu_stage_in_is_ordered_after_engine_writes_vectorised():
    kvs, spans, los = _gpu_setup(layers=2)
    st = hs.HostStaging(spans)
    assert st._vec is not None
    for _, _, _, m in st._spans:
        m.fill_(7)
    torch.cuda._sleep(int(0.2 * 1.4e9))
    kvs[1].fill_(1)                                   # stale step writes the blocks
    import threading
    ranges = hs.block_ranges(los, [0, 5, 63])
    th = threading.Thread(target=st.stage_in, args=(ranges,))
    th.start(), th.join()
    torch.cuda.synchronize()
    assert int(kvs[1][0, 5, 0]) == 7 and int(kvs[1][1, 63, -1]) == 7
    assert int(kvs[1][0, 4, 0]) == 1
