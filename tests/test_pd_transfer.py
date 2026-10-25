# SPDX-License-Identifier: Apache-2.0
"""PD KV transfer core (connex.pd) on CPU: layouts, holds, pulls, failures."""
import time

import pytest
import torch

from connex.pd.dataplane import LoopbackPlane
from connex.pd.kv_layout import block_runs, read_ops, register_spans, tensor_layout
from connex.pd.transfer import KVConsumer, KVProducer

NB, BS, H, D = 16, 4, 2, 8


def kv_caches(n_layers=3, fill=None):
    """vLLM FlashAttention-style per-layer caches: [2, num_blocks, block, heads, dim]."""
    out = []
    for i in range(n_layers):
        t = torch.zeros(2, NB, BS, H, D, dtype=torch.float16)
        if fill is not None:
            t.copy_(torch.randn_like(t, dtype=torch.float32).to(torch.float16))
        out.append(t)
    return out


def wait_done(c, n=1, timeout=5.0):
    out, t0 = [], time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        out += c.take_done()
        time.sleep(0.005)
    return out


def test_layout_of_kv_first_and_block_first_tensors():
    t = torch.zeros(2, NB, BS, H, D, dtype=torch.float16)
    lo = tensor_layout(t, NB)
    assert (lo.outer, lo.seg_bytes, lo.block_stride) == (2, BS * H * D * 2, BS * H * D * 2)
    assert lo.outer_stride == NB * BS * H * D * 2
    assert lo.span == (t.data_ptr(), t.numel() * 2)
    u = torch.zeros(NB, 2, BS, H, D, dtype=torch.bfloat16)
    lu = tensor_layout(u, NB)
    assert (lu.outer, lu.seg_bytes) == (1, 2 * BS * H * D * 2)
    with pytest.raises(ValueError):
        tensor_layout(torch.zeros(3, 5), NB)


def test_block_runs_coalesce_only_when_both_sides_are_consecutive():
    assert block_runs([3, 4, 5, 9], [7, 8, 9, 10]) == [(3, 7, 3), (9, 10, 1)]
    assert block_runs([1, 2], [5, 7]) == [(1, 5, 1), (2, 7, 1)]
    with pytest.raises(ValueError):
        block_runs([1], [1, 2])


def test_read_ops_and_register_spans():
    a, b = kv_caches(2), kv_caches(2)
    la = [tensor_layout(t, NB) for t in a]
    lb = [tensor_layout(t, NB) for t in b]
    ops = read_ops(la, lb, [0, 1, 2], [5, 6, 7])
    assert len(ops) == 2 * 2           # layers x (K, V), one run each
    assert all(n == 3 * la[0].seg_bytes for _, _, n in ops)
    assert len(register_spans(la + la)) == 2
    bad = [tensor_layout(torch.zeros(2, NB, BS * 2, H, D, dtype=torch.float16), NB)] * 2
    with pytest.raises(ValueError):
        read_ops(la, bad, [0], [0])


def make_pair(retain_s=30.0, release_after_pull=False, block_size=BS):
    src, dst = kv_caches(fill=True), kv_caches()
    p_plane, d_plane = LoopbackPlane(), LoopbackPlane()
    prod = KVProducer(p_plane, [tensor_layout(t, NB) for t in src], bind_ip="127.0.0.1",
                      retain_s=retain_s, engine_id="p0", block_size=block_size)
    cons = KVConsumer(d_plane, [tensor_layout(t, NB) for t in dst], block_size=BS,
                      ctrl_timeout_ms=500, release_after_pull=release_after_pull,
                      acquire_wait_s=0.2)
    return src, dst, prod, cons, f"127.0.0.1:{prod.port}"


def test_pull_copies_exactly_the_requested_blocks():
    src, dst, prod, cons, addr = make_pair()
    try:
        prod.hold("r1", "t1", [9, 10, 11, 2])
        cons.pull("r1", addr, "t1", [0, 1, 7, 8], first_block=0)
        (res,) = wait_done(cons)
        assert res.ok and res.nbytes > 0, res
        for s, d in zip(src, dst):
            for lb, rb in zip([0, 1, 7, 8], [9, 10, 11, 2]):
                assert torch.equal(d[:, lb], s[:, rb])
            assert torch.count_nonzero(d[:, 3]) == 0     # untouched
        # held until released
        assert prod.take_finished() == set() and prod.held() == 1
        cons.release(addr, "t1")
        t0 = time.monotonic()
        fin = set()
        while not fin and time.monotonic() - t0 < 2:
            fin = prod.take_finished()
        assert fin == {"r1"}
    finally:
        cons.close(), prod.close()


def test_repull_by_a_second_consumer_and_partial_suffix():
    """Decode instance died: another one acquires the same transfer again,
    and with a local prefix hit pulls only the missing suffix."""
    src, dst, prod, cons, addr = make_pair()
    dst2 = kv_caches()
    cons2 = KVConsumer(LoopbackPlane(), [tensor_layout(t, NB) for t in dst2], ctrl_timeout_ms=500)
    try:
        prod.hold("r1", "t1", [4, 5, 6])
        cons.pull("r1", addr, "t1", [0, 1, 2])
        assert wait_done(cons)[0].ok
        cons2.pull("r1b", addr, "t1", [12], first_block=2)   # blocks 0,1 hit locally
        (res,) = wait_done(cons2)
        assert res.ok
        for s, d in zip(src, dst2):
            assert torch.equal(d[:, 12], s[:, 6])
        assert prod.stats["acquires"] == 2
    finally:
        cons2.close(), cons.close(), prod.close()


def test_failures_are_reported_per_request_not_raised():
    src, dst, prod, cons, addr = make_pair(retain_s=0.2)
    try:
        cons.pull("unknown", addr, "nope", [0])
        r = wait_done(cons)[0]
        assert not r.ok and "unknown" in r.why and r.failed_blocks == [0]
        prod.hold("r1", "t1", [1])
        cons.pull("short", addr, "t1", [0, 1])            # producer holds fewer blocks
        assert not wait_done(cons)[0].ok
        time.sleep(0.3)                                    # lease expires without acquires
        assert prod.take_finished() == {"r1"} and prod.stats["expired"] == 1
        prod.close()                                       # prefill instance gone
        cons.pull("dead", addr, "t1", [0])
        r = wait_done(cons)[0]
        assert not r.ok and r.why == "producer unreachable"
    finally:
        cons.close()


def test_release_after_pull_and_release_before_hold():
    src, dst, prod, cons, addr = make_pair(release_after_pull=True)
    try:
        prod.hold("r1", "t1", [3])
        cons.pull("r1", addr, "t1", [3])
        assert wait_done(cons)[0].ok
        t0 = time.monotonic()
        fin = set()
        while not fin and time.monotonic() - t0 < 2:
            fin = prod.take_finished()
        assert fin == {"r1"}
        # a release that overtakes the engine's hold frees it at once
        cons.release(addr, "t2")
        time.sleep(0.1)
        prod.hold("r2", "t2", [4])
        assert prod.take_finished() == {"r2"} and prod.held() == 0
    finally:
        cons.close(), prod.close()


def test_acquire_waits_for_a_late_hold():
    """The decode request may reach the decode instance before the prefill
    worker has registered the hold: the acquire is retried briefly."""
    src, dst, prod, cons, addr = make_pair()
    try:
        cons.pull("r1", addr, "t-late", [0])
        time.sleep(0.05)
        prod.hold("r1", "t-late", [5])
        (res,) = wait_done(cons)
        assert res.ok, res
        assert torch.equal(dst[0][:, 0], src[0][:, 5])
    finally:
        cons.close(), prod.close()


def test_block_size_mismatch_rejected():
    src, dst, prod, cons, addr = make_pair(block_size=BS * 2)
    try:
        prod.hold("r", "t", [0])
        cons.pull("r", addr, "t", [0])
        r = wait_done(cons)[0]
        assert not r.ok and "block size" in r.why
    finally:
        cons.close(), prod.close()


class _FakeStaging:
    """HostStaging on CPU: the "device" is host memory, copies are memmoves."""

    def __init__(self, spans, device=None):
        import bisect
        import ctypes
        from connex.pd import host_staging as hs

        self._bisect, self._mm, self._merge = bisect, ctypes.memmove, hs._merge
        self.spans = sorted(spans)
        self.mirrors = [torch.zeros(n, dtype=torch.uint8) for _, n in self.spans]
        self.staged_in = 0

    def _find(self, addr):
        i = self._bisect.bisect_right([s for s, _ in self.spans], addr) - 1
        return self.spans[i][0], self.mirrors[i]

    def mirror(self, addr):
        s, m = self._find(addr)
        return m.data_ptr() + addr - s

    def mirror_spans(self):
        return [(m.data_ptr(), n) for m, (_, n) in zip(self.mirrors, self.spans)]

    def translate(self, lo):
        from dataclasses import replace
        return replace(lo, base=self.mirror(lo.base))

    def mirror_ops(self, ops):
        return [(self.mirror(d), s, n) for d, s, n in ops]

    def stage_out(self, ranges):
        for a, n in self._merge(ranges):
            self._mm(self.mirror(a), a, n)
        return type("Ev", (), {"synchronize": lambda self: None})()

    def stage_in(self, ranges):
        self.staged_in += 1
        for a, n in self._merge(ranges):
            self._mm(a, self.mirror(a), n)


def test_host_staging_reads_go_through_the_mirrors(monkeypatch):
    from connex.pd import host_staging

    monkeypatch.setenv("CONNEX_PD_HOST_STAGING", "1")
    monkeypatch.setattr(host_staging, "HostStaging", _FakeStaging)
    src, dst, prod, cons, addr = make_pair()
    try:
        assert prod._staging is not None and cons._staging is not None
        # only the mirrors are registered with the data planes
        assert prod.plane.regions == prod._staging.mirror_spans()
        assert cons.plane.regions == cons._staging.mirror_spans()
        want = [s[:, [9, 10]].clone() for s in src]
        prod.hold("r1", "t1", [9, 10])            # staged out here
        for s in src:
            s.zero_()                             # the reader must not see the device copy
        cons.pull("r1", addr, "t1", [3, 4], first_block=0)
        (res,) = wait_done(cons)
        assert res.ok, res
        for w, d in zip(want, dst):
            assert torch.equal(d[:, [3, 4]], w)
        assert cons._staging.staged_in == 1
    finally:
        cons.close(), prod.close()
