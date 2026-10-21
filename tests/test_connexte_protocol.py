# SPDX-License-Identifier: Apache-2.0
"""Mock tests for the ConnexTE data-path protocol logic.

integrations/vllm/connexte_communicator.py runs inside the vLLM fork, so the
vllm.* modules it imports are stubbed here; no GPU, Mooncake or cluster.
"""

import importlib.util
import pickle
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import zmq

ROOT = Path(__file__).resolve().parents[1]


def _load():
    stubs = {
        "vllm": {},
        "vllm.distributed": {},
        "vllm.distributed.device_communicators": {},
        "vllm.distributed.device_communicators.zeromq_communicator": {
            "ZeroMQPPCommunicator": type("ZeroMQPPCommunicator", (), {})},
        "vllm.logger": {"init_logger": lambda name: __import__("logging").getLogger(name)},
        "vllm.utils": {},
        "vllm.utils.network_utils": {
            "get_ip": lambda: "127.0.0.1",
            "get_tcp_uri": lambda h, p: f"tcp://{h}:{p}",
            "join_host_port": lambda h, p: f"{h}:{p}",
            "split_host_port": lambda s: (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])),
        },
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
    try:
        spec = importlib.util.spec_from_file_location(
            "connexte_under_test", ROOT / "integrations/vllm/connexte_communicator.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


cx = _load()


# ---- ring allocation / ack / free ---------------------------------------------

def test_ring_does_not_reuse_unacked_region_even_after_hold():
    t = cx._RegionTable(1024, token_base=1 << 32)
    off0, tok0 = t.alloc(1024, now=0.0, hold_s=10.0, want_ack=True)
    assert off0 == 0
    # 5 s later (well past the old 1 s hold): still unacked -> no overwrite
    assert t.alloc(512, now=5.0, hold_s=10.0, want_ack=True) is None
    assert t.ack(tok0)
    off1, _ = t.alloc(512, now=5.0, hold_s=10.0, want_ack=True)
    assert off1 == 0


def test_ring_wraps_and_skips_live_region():
    t = cx._RegionTable(1000)
    a = t.alloc(400, 0.0, 10.0, True)
    b = t.alloc(400, 0.0, 10.0, True)
    assert (a[0], b[0]) == (0, 400)
    # 400 more does not fit at head (800) -> wraps to 0, which is still live
    assert t.alloc(400, 0.0, 10.0, True) is None
    t.ack(a[1])
    assert t.alloc(400, 0.0, 10.0, True)[0] == 0


def test_ring_safety_timeout_frees_and_counts():
    t = cx._RegionTable(100)
    _, tok = t.alloc(100, 0.0, 10.0, True)
    assert t.alloc(100, 9.9, 10.0, True) is None
    assert t.alloc(100, 10.1, 10.0, True) is not None
    t.collect(10.1)
    assert t.expired_unacked == 1
    assert tok not in t.leases


def test_ring_legacy_time_hold_without_ack():
    t = cx._RegionTable(100)
    t.alloc(100, 0.0, 1.0, False)
    assert not t.awaiting_ack()
    assert t.alloc(100, 0.5, 1.0, False) is None
    assert t.alloc(100, 1.5, 1.0, False) is not None
    t.collect(100.0)
    assert t.expired_unacked == 0


def test_pinned_region_survives_ack_and_timeout_until_unpinned():
    t = cx._RegionTable(100)
    _, tok = t.alloc(100, 0.0, 1.0, True)
    t.pin(tok)
    t.ack(tok)
    assert t.alloc(100, 50.0, 1.0, True) is None
    assert t.collect(50.0) == []
    t.unpin(tok)
    assert t.alloc(100, 50.0, 1.0, True) is not None


def test_temp_buffers_released_on_ack_only():
    t = cx._RegionTable(100)
    obj = object()
    tok = t.add_temp(obj, 0.0, 10.0, True)
    assert t.collect(1.0) == []
    t.ack(tok)
    assert t.collect(1.0) == [obj]
    assert t.ack(tok) is False  # late duplicate ack is ignored


def test_tokens_unique_and_carry_base():
    t = cx._RegionTable(10, token_base=7 << 32)
    toks = {t.add_temp(None, 0, 1, False) for _ in range(100)}
    assert len(toks) == 100 and all(x >> 32 == 7 for x in toks)


# ---- stale correlation ids ------------------------------------------------------

def test_corr_filter_drops_older_same_base_only():
    f = cx._CorrFilter()
    a, b = 0xAAAA << 32, 0xBBBB << 32
    assert f.accept(a + 5)
    assert f.accept(a + 5)          # same id (dup handled elsewhere) is not stale
    assert not f.accept(a + 4)      # older step of the same stage 0
    assert f.accept(b + 1)          # restarted stage 0: new random base
    assert f.accept(a + 6)
    assert f.accept(None) and f.accept(0)  # old senders without ids


def test_corr_filter_lru_bounded():
    f = cx._CorrFilter(cap=4)
    for h in range(10):
        f.accept((h << 32) + 1)
    assert len(f.hi) == 4


# ---- header codec ---------------------------------------------------------------

LAYOUT = [("hidden_states", "bfloat16", (7, 4096), 0, 7 * 4096 * 2),
          ("residual", "bfloat16", (7, 4096), 57344, 7 * 4096 * 2),
          ("positions", "int64", (7,), 114688, 56)]


def test_layout_roundtrip():
    b = cx._encode_layout(LAYOUT)
    assert cx._decode_layout(b) == LAYOUT


def test_layout_unknown_dtype_falls_back():
    assert cx._encode_layout([("x", "complex64", (2,), 0, 16)]) is None


def test_g2_roundtrip_and_layout_cache():
    codec = cx._LayoutCodec()
    key = (("hidden_states", "bfloat16", (7, 4096)), ("positions", "int64", (7,)))
    layout, total, lb = codec.layout_for(key, [57344, 56])
    assert layout[1][3] == 57344 and total == cx._align(57344 + 56)
    assert codec.layout_for(key, [57344, 56])[2] is lb  # cached, not re-encoded
    nt = pickle.dumps({"foo": 1})
    tail = cx._encode_g2(123, lb, b"tcp://1.2.3.4:5", b"tcp://9.9.9.9:1", nt)
    assert tail[:2] == b"G2"
    tok, lb2, ack_ep, origin, nt_b = cx._decode_g2(tail[2:])
    assert (tok, ack_ep, origin) == (123, "tcp://1.2.3.4:5", "tcp://9.9.9.9:1")
    assert codec.parse(lb2) == layout and pickle.loads(nt_b) == {"foo": 1}
    # no non-tensor fields -> no pickle bytes at all
    assert cx._decode_g2(cx._encode_g2(0, lb, b"", b"", b"")[2:])[4] == b""


# ---- communicator-level behaviour with fakes -------------------------------------

class FakeRouter:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    def poll(self, timeout, flags=None):
        return 1 if self.msgs else 0

    def recv_multipart(self, flags=0):
        if not self.msgs:
            raise zmq.Again()
        return [b"id", self.msgs.pop(0)]


def _sig(corr, tail=cx._encode_g2(0, cx._encode_layout([]), b"", b"tcp://s0:1", b"")):
    url = b"10.0.0.1:50000"
    return b"CONNEXTE" + struct.pack("!QQI", 0, 0, len(url)) + url + struct.pack("!Q", corr) + tail


def _comm(stage, n=4, msgs=()):
    c = cx.ConnexTECommunicator.__new__(cx.ConnexTECommunicator)
    c.stage_idx, c.num_stages = stage, n
    c.is_first, c.is_last = stage == 0, stage == n - 1
    c._mooncake_available = True
    c._prof = None
    c._stash = []
    c._gdr = cx._RegionTable(1 << 20)
    c._corr_filter = cx._CorrFilter()
    c._layouts = cx._LayoutCodec()
    c._pending_corr = None
    c._corr_in = None
    c._last_done = None
    c._recent_step_s = []
    c._current_originator_endpoint = None
    c.router_socket = FakeRouter(msgs)
    c._mcache = {}
    sent = []
    c._send_via_zeromq = lambda data, dst: sent.append(data)
    c._invalidate_stage_connection = lambda s: None
    return c, sent


def test_middle_stage_drops_stale_and_accepts_new(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    base = 0x1234 << 32
    c, _ = _comm(1, msgs=[_sig(base + 3), _sig(base + 2), _sig(base + 4)])
    c.recv_tensor_dict()
    assert c._corr_in == base + 3
    c.recv_tensor_dict()  # base+2 is stale -> skipped, base+4 delivered
    assert c._corr_in == base + 4
    assert c._current_originator_endpoint == "tcp://s0:1"


def test_ack_messages_are_consumed_not_delivered(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    c, _ = _comm(2)
    _, tok = c._gdr.alloc(100, 0.0, 10.0, True)
    c.router_socket = FakeRouter([cx._ACK_MARK + cx._ACK.pack(tok), _sig(5)])
    out = c.recv_tensor_dict()
    assert out == {"__stage0_endpoint__": "tcp://s0:1"}
    assert c._gdr.leases[tok].acked


def test_drain_acks_stashes_data_for_recv(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    c, _ = _comm(1)
    _, tok = c._gdr.alloc(100, 0.0, 10.0, True)
    c.router_socket = FakeRouter([_sig(7), cx._ACK_MARK + cx._ACK.pack(tok)])
    c._drain_acks()
    assert c._gdr.leases[tok].acked and len(c._stash) == 1
    c.recv_tensor_dict()
    assert c._corr_in == 7 and not c._stash


def test_duplicate_resends_cached_signal_bytes(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    c, sent = _comm(1, msgs=[_sig(9), _sig(10)])
    c._last_done = (9, ("sig", b"ORIGINAL-SIGNAL"), 2)
    c.recv_tensor_dict()
    assert sent == [b"ORIGINAL-SIGNAL"] and c._corr_in == 10


def test_old_g1_header_still_accepted(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    hdr = pickle.dumps({"layout": [], "nt": {"__stage0_endpoint__": "tcp://old:1", "k": 3}})
    c, _ = _comm(1, msgs=[_sig(1, b"G1" + hdr)])
    assert c.recv_tensor_dict() == {"__stage0_endpoint__": "tcp://old:1", "k": 3}


# ---- route lookups ----------------------------------------------------------------

class FakeEPC:
    def __init__(self, routes):
        self.routes = routes
        self.refreshes = 0
        self.lid = SimpleNamespace(replica_idx=1)

    def refresh_routes(self):
        self.refreshes += 1

    def get_pipeline_peers(self):
        return list(self.routes)


def _route(stage, rep, addr):
    return SimpleNamespace(lid=SimpleNamespace(stage_idx=stage, replica_idx=rep), addresses=[addr])


def _route_comm(epc):
    c = cx.ConnexTECommunicator.__new__(cx.ConnexTECommunicator)
    c.stage_idx, c.num_stages = 1, 4
    c._epc_client = epc
    c._sticky_peer, c._route_dirty, c._deregistered_replicas = {}, set(), set()
    return c


def test_sticky_peer_avoids_route_refresh():
    epc = FakeEPC([_route(1, 1, "a:1"), _route(2, 5, "b:1"), _route(2, 6, "c:1")])
    c = _route_comm(epc)
    first = c._get_peer_endpoint(2)
    assert epc.refreshes == 1
    for _ in range(10):
        assert c._get_peer_endpoint(2) == first
    assert epc.refreshes == 1
    c._route_dirty.add(2)  # e.g. a send failed
    c._get_peer_endpoint(2)
    assert epc.refreshes == 2


def test_sticky_peer_gone_triggers_refresh_and_reselect():
    epc = FakeEPC([_route(1, 1, "a:1"), _route(2, 5, "b:1")])
    c = _route_comm(epc)
    assert c._get_peer_endpoint(2) == "b:1"
    epc.routes = [_route(1, 1, "a:1"), _route(2, 6, "c:1")]
    assert c._get_peer_endpoint(2) == "c:1"
    assert epc.refreshes == 2


def test_filtered_stage_api_used_when_present():
    epc = FakeEPC([_route(1, 1, "a:1"), _route(2, 5, "stale:1")])
    calls = []
    epc.get_stage_peers = lambda s, refresh="auto": calls.append(refresh) or [_route(2, 6, "fresh:1")]
    c = _route_comm(epc)
    assert c._get_peer_endpoint(2) == "fresh:1"
    assert epc.refreshes == 0 and calls == ["always"]


# ---- end-to-end GPU path with a fake Mooncake engine (skipped without CUDA) -----

class FakeEngine:
    """register_memory/transfer_sync_read over real CUDA tensors by pointer."""

    def __init__(self):
        self.regs = {}  # ptr -> tensor

    def track(self, t):
        self.regs[t.data_ptr()] = t

    def register_memory(self, ptr, n):
        return 0

    def unregister_memory(self, ptr):
        self.regs.pop(ptr, None)
        return 0

    def _view(self, ptr, n):
        import torch

        for base, t in self.regs.items():
            if base <= ptr and ptr + n <= base + t.numel():
                return t[ptr - base:ptr - base + n]
        # temp buffers are torch.empty'd inside the communicator: find via torch
        raise KeyError(hex(ptr))

    def transfer_sync_read(self, url, dst, src, n):
        self._view(dst, n).copy_(self._view(src, n))
        import torch

        torch.cuda.synchronize()
        return 0


def _gpu_comm(stage, eng, slots=1):
    import torch

    c, sent = _comm(stage)
    c._mooncake_engine = eng
    c._local_url = "10.0.0.1:5000%d" % stage
    c.router_endpoint = "tcp://10.0.0.1:600%d" % stage
    c._gdr_buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    c._gdr_recv = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    eng.track(c._gdr_buf)
    eng.track(c._gdr_recv)
    c._gdr = cx._RegionTable(c._gdr_buf.numel())
    c._gdr_send_evt = torch.cuda.Event()
    c._gdr_ack_on = True
    c._gdr_slot_size = (c._gdr_recv.numel() // slots) // 256 * 256
    c._gdr_slot_evts = [torch.cuda.Event() for _ in range(slots)]
    c._gdr_slot_used = [False] * slots
    c._gdr_slot_cur = None
    c._last_token = None
    c._corr_base, c._corr_seq = 0, 0
    acks = []
    c._send_ack = lambda ep, tok: acks.append((ep, tok))
    return c, sent, acks


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_gpu_send_recv_ack_and_resend(monkeypatch):
    import torch

    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    eng = FakeEngine()
    tx, sent, _ = _gpu_comm(1, eng)
    rx, _, acks = _gpu_comm(2, eng)
    tx._corr_in = (5 << 32) + 1
    h = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(3, device="cuda")
    tx._send_gdr({"h": h, "pos": pos, "extra": 7}, 2, 0.0)
    want_h, want_pos = h.clone(), pos.clone()
    h.fill_(0)  # vLLM reuses its buffers right after send
    tok = tx._last_token
    assert tx._gdr.leases[tok].pinned and not tx._gdr.leases[tok].acked
    rx.router_socket = FakeRouter([sent[0]])
    out = rx.recv_tensor_dict()
    assert torch.equal(out["h"], want_h) and torch.equal(out["pos"], want_pos)
    assert out["extra"] == 7
    assert acks == [("tcp://10.0.0.1:6001", tok)]
    tx._gdr.ack(tok)
    # duplicate request -> re-signal original region, data still intact
    tx._resend_last()
    rx2, _, _ = _gpu_comm(2, eng)
    rx2.router_socket = FakeRouter([sent[1]])
    out2 = rx2.recv_tensor_dict()
    assert torch.equal(out2["h"], want_h)
    # a second send moves the pin; the old region becomes reusable
    tx._corr_in += 1
    tx._send_gdr({"h": h}, 2, 0.0)
    assert not tx._gdr.leases[tok].pinned


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_gpu_ring_full_uses_temp_buffer(monkeypatch):
    import torch

    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    eng = FakeEngine()
    orig_reg = eng.register_memory
    tx, sent, _ = _gpu_comm(0, eng)
    tx.router_socket = FakeRouter([])
    big = torch.ones(600 * 1024, dtype=torch.uint8, device="cuda")
    tx._send_gdr({"x": big}, 1, 0.0)
    tx._pending_corr = None
    tx._send_gdr({"x": big}, 1, 0.0)  # first region unacked -> temp, no overwrite
    temps = [l for l in tx._gdr.leases.values() if l.obj is not None]
    assert len(temps) == 1
    eng.register_memory = lambda p, n: -1
    tx._pending_corr = None
    with pytest.raises(RuntimeError, match="register_memory failed"):
        tx._send_gdr({"x": big}, 1, 0.0)
    eng.register_memory = orig_reg


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_gpu_recv_slots_return_views(monkeypatch):
    import torch

    monkeypatch.setattr(cx, "METRICS_ENABLED", False)
    eng = FakeEngine()
    tx, sent, _ = _gpu_comm(1, eng)
    rx, _, _ = _gpu_comm(2, eng, slots=2)
    outs = []
    for i in range(3):
        tx._corr_in = (9 << 32) + i + 1
        tx._send_gdr({"h": torch.full((4,), float(i), device="cuda")}, 2, 0.0)
        tx._gdr.ack(tx._last_token)
        rx.router_socket = FakeRouter([sent[-1]])
        outs.append(rx.recv_tensor_dict()["h"])
    assert outs[0].data_ptr() == outs[2].data_ptr() != outs[1].data_ptr()  # rotated views
    assert float(outs[1][0]) == 1.0 and float(outs[2][0]) == 2.0
