# SPDX-License-Identifier: Apache-2.0
"""SGLang Connex transfer backend: plugin registration (needs SGLang) and the
connex-rs adapter behind SGLang's Mooncake transfer-engine surface (no GPU)."""
import pytest

from connex.adapters.sglang.pd_backend import ConnexRsTransferEngine


class FakeTransfer:
    def __init__(self, ok=True):
        self.ok = ok

    def wait(self, timeout_ms=None):
        if not self.ok:
            raise RuntimeError("peer failed")
        return True


class FakeEngine:
    def __init__(self, model, stage, shard, replica, **kw):
        self.local_url = f"{kw['listen_ip']}:5000#{model}|{stage}|{shard}|{replica}"
        self.puts, self.regions, self.fail_after = [], {}, None

    def register_memory(self, ptr, n):
        self.regions[ptr] = n
        return 0

    def unregister_memory(self, ptr):
        self.regions.pop(ptr, None)
        return 0

    def put_async(self, url, src, dst, n, timeout_ms=None):
        self.puts.append((url, src, dst, n))
        return FakeTransfer(self.fail_after is None or len(self.puts) <= self.fail_after)

    def transfer_sync_write(self, url, src, dst, n, timeout_ms=None):
        self.puts.append((url, src, dst, n))
        return 0

    def connect(self, url):
        if "dead" in url:
            raise RuntimeError("unreachable")


@pytest.fixture(autouse=True)
def _restore_env():
    """The adapter sets UCX_* defaults for its process; keep them out of the
    other tests (UCX reads them at initialisation)."""
    import os

    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def engine():
    return ConnexRsTransferEngine("10.0.0.1", "prefill", gpu_id=2, engine_cls=FakeEngine)


def test_session_id_is_the_logical_endpoint_url():
    e = engine()
    assert e.get_session_id().startswith("10.0.0.1:5000#sglang-pd|1000|2|")
    assert e.batch_register([1, 2], [10, 20]) == 0 and e.engine.regions == {1: 10, 2: 20}


def test_batch_writes_are_posted_together_and_failures_reported():
    e = engine()
    n = 1 << 16
    assert e.batch_transfer_sync("peer", [1, 2, 3], [11, 12, 13], [n, n, n]) == 0
    assert [p[1:] for p in e.engine.puts] == [(1, 11, n), (2, 12, n), (3, 13, n)]
    assert e.batch_transfer_sync("peer", [1], [2], [100]) == -4   # below zero-copy size
    e.engine.puts, e.engine.fail_after = [], 1
    assert e.batch_transfer_sync("peer", [1, 2], [11, 12], [n, n]) < 0
    assert e.send_probe("10.0.0.2:1#x|1|0|0") == 0 and e.send_probe("dead:1#x|1|0|0") == -1


def test_plugin_registers_connex_backend():
    pytest.importorskip("sglang.srt.disaggregation.utils")
    import sglang.srt.disaggregation.utils as du
    from connex.adapters.sglang.pd_backend import register

    register()
    register()   # idempotent
    tb = du.TransferBackend("connex")
    assert tb is du.TransferBackend.CONNEX
    mgr = du.get_kv_class(tb, du.KVClassType.MANAGER)
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    assert issubclass(mgr, MooncakeKVManager)
    assert du.get_kv_class(du.TransferBackend.MOONCAKE, du.KVClassType.MANAGER) is MooncakeKVManager
