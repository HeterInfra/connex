# SPDX-License-Identifier: Apache-2.0
"""connex-rs batch reads (issue #39): plane and RL transport use
Engine.transfer_batch_sync when the engine has it, per-op get_async otherwise."""

import pytest

import connex.pd.dataplane as dp


class _H:
    def __init__(self, ok=True):
        self.ok = ok

    def wait(self, ms):
        return self.ok


class PerOpEngine:
    """connex-rs < 0.1.7: no batch API."""

    def __init__(self):
        self.calls = []

    def get_async(self, url, dst, src, n, to):
        self.calls.append((dst, src, n))
        return _H()


class BatchEngine(PerOpEngine):
    def __init__(self, rc=0):
        super().__init__()
        self.rc = rc
        self.batches = []

    def transfer_batch_sync(self, url, is_put, ops, timeout_ms):
        assert is_put is False and isinstance(ops, list)
        self.batches.append(list(ops))
        return self.rc


def _plane(engine):
    p = dp.ConnexRsPlane.__new__(dp.ConnexRsPlane)
    p.e, p.timeout_ms = engine, 1000
    return p


OPS = [(0x1000, 0x9000, 64), (0x2000, 0xa000, 64), (0x3000, 0xb000, 128)]


def test_plane_uses_one_batch_call():
    e = BatchEngine()
    assert _plane(e).read("peer", iter(OPS)) == 0
    assert e.batches == [OPS] and not e.calls


def test_plane_falls_back_per_op_without_batch_api():
    e = PerOpEngine()
    assert _plane(e).read("peer", OPS) == 0
    assert e.calls == OPS


@pytest.mark.parametrize("rc,want", [(-2, dp.PEER_FAILED), (-3, dp.TIMEOUT), (-1, dp.PEER_FAILED),
                                     (-4, -4)])
def test_plane_maps_batch_errors(rc, want):
    assert _plane(BatchEngine(rc)).read("peer", OPS) == want


def test_batch_disabled_by_env_switch(monkeypatch):
    monkeypatch.setattr(dp, "_RS_BATCH", False)
    e = BatchEngine()
    assert _plane(e).read("peer", OPS) == 0
    assert not e.batches and e.calls == OPS


def test_empty_batch_is_a_noop():
    e = BatchEngine()
    assert dp.batch_read(e, "peer", [], 1000) == 0 and not e.batches


def test_rl_transport_batches_and_raises():
    import torch

    from connex.rl.transport import ConnexRsTransport, TransferError

    t = ConnexRsTransport.__new__(ConnexRsTransport)
    t.e, t.timeout_ms, t.reads, t.bytes = BatchEngine(), 1000, 0, 0
    a, b = torch.zeros(256, dtype=torch.uint8), torch.zeros(256, dtype=torch.uint8)
    t.read_many("peer", [(a, [(0x9000, 0, 64), (0xa000, 64, 64)]), (b, [(0xb000, 8, 16)])])
    assert t.e.batches == [[(a.data_ptr(), 0x9000, 64), (a.data_ptr() + 64, 0xa000, 64),
                            (b.data_ptr() + 8, 0xb000, 16)]]
    assert t.reads == 3 and t.bytes == 144
    t.read_batch("peer", [(0xc000, 0, 32)], a)
    assert t.e.batches[-1] == [(a.data_ptr(), 0xc000, 32)] and t.reads == 4
    t.e.rc = -3
    with pytest.raises(TransferError, match="rc=-3"):
        t.read_batch("peer", [(0xc000, 0, 32)], a)
