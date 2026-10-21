# SPDX-License-Identifier: Apache-2.0
"""Differential test: the refactored vLLM adapter (connex.adapters.vllm on top
of connex.pp) against a frozen copy of the self-contained communicator it
replaced (tests/reference/connexte_communicator_reference.py).

Both are driven through identical scenarios with the same fake transfer
engine and captured ZeroMQ sends; every byte put on the wire, every value
returned to the engine and the protocol state must be identical.
"""

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_connexte_protocol import FakeEPC, FakeRouter, _route, _sig, cx as NEW

ROOT = Path(__file__).resolve().parent


def _load_reference():
    stubs = {
        "vllm": {}, "vllm.distributed": {}, "vllm.distributed.device_communicators": {},
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
            "connexte_reference", ROOT / "reference/connexte_communicator_reference.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


OLD = _load_reference()
BASE = 0x5EED << 32


class FakeTE:
    """Deterministic transfer engine: pointers count up, memory is a dict."""

    def __init__(self):
        self.next, self.mem, self.reads = 0x1000, {}, []

    def allocate_managed_buffer(self, n):
        p, self.next = self.next, self.next + ((n + 0xFFF) & ~0xFFF) + 0x1000
        self.mem[p] = bytearray(n)
        return p

    def free_managed_buffer(self, p, n):
        self.mem.pop(p, None)

    def write_bytes_to_buffer(self, p, data, n):
        self.mem[p][:n] = data[:n]

    def read_bytes_from_buffer(self, p, n):
        return bytes(self.mem[p][:n])

    def transfer_sync_read(self, url, dst, src, n):
        self.reads.append((url, src, n))
        peer = PEERS[url]
        if dst in self.mem:  # host path
            self.mem[dst][:n] = peer.mem[src][:n]
        else:                # GPUDirect path: dst is a device pointer
            DEV_READS.append((dst, bytes(peer.dev(src, n))))
        return 0

    def register_memory(self, p, n):
        return 0

    def unregister_memory(self, p):
        return 0


PEERS: dict = {}
DEV_READS: list = []


def _mk(mod, stage, n=4, k=1, msgs=(), url="10.0.0.1:50000", engine=None):
    cls = mod.ConnexTECommunicator
    c = cls.__new__(cls)
    c.stage_idx, c.num_stages = stage, n
    c.is_first, c.is_last = stage == 0, stage == n - 1
    c.device = "cpu"
    c._mooncake_available = True
    c._mooncake_engine = engine or FakeTE()
    c._local_url = url
    c._buffer_cleaner = None
    c._prof = None
    c._stash = []
    c._gdr = None
    c._corr_filter = mod._CorrFilter()
    c._layouts = mod._LayoutCodec()
    c._corr_base, c._corr_seq = BASE, 0
    c._pending_corr = c._corr_in = c._last_done = None
    c._last_token = None
    c._recent_step_s = []
    c._current_originator_endpoint = None
    c.router_endpoint = f"tcp://127.0.0.1:{5550 + stage}"
    c.router_socket = FakeRouter(list(msgs))
    c._mcache = {}
    c._sticky_peer, c._route_dirty, c._deregistered_replicas = {}, set(), set()
    c._suspect_replicas = set()
    sent = []
    c._send_via_zeromq = lambda data, dst: sent.append((data, dst))
    c._invalidate_stage_connection = lambda s: sent.append(("invalidate", s))
    if k > 1:
        c._init_inflight(k)
    return c, sent


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    for mod in (OLD, NEW):
        monkeypatch.setattr(mod, "METRICS_ENABLED", False)
    import connex.pp.endpoint as core
    for m in (OLD, core):
        monkeypatch.setattr(m.time, "time", lambda: 1700000000.25)
    PEERS.clear()
    DEV_READS.clear()


def _pair(**kw):
    return _mk(OLD, **kw), _mk(NEW, **kw)


def _td(i=0):
    return {"hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4) + i,
            "positions": torch.arange(3, dtype=torch.int64),
            "request_ids": ["a", "b", "c"], "cx_ctx": [5, 0, 9], "step": i}


def _same_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])
        else:
            assert a[k] == b[k]


# ---- host-buffer path: identical signal and payload bytes ------------------------

@pytest.mark.parametrize("stage", [0, 1, 3])
def test_host_send_bytes_identical(stage):
    (o, so), (n, sn) = _pair(stage=stage)
    for c in (o, n):
        c._current_originator_endpoint = "tcp://10.0.0.9:5550"
    for i in range(3):
        o._corr_in = n._corr_in = BASE + 7 + i
        o.send_tensor_dict(_td(i))
        n.send_tensor_dict(_td(i))
    assert so == sn and len(so) == 3
    assert o._mooncake_engine.mem == n._mooncake_engine.mem
    assert o._last_done == n._last_done
    assert o._pending_corr == n._pending_corr


def test_host_roundtrip_identical():
    (o0, so), (n0, sn) = _pair(stage=0, url="s0:1")
    PEERS["s0:1"] = o0._mooncake_engine
    o0.send_tensor_dict(_td(1))
    sig = so[0][0]
    (o1, _), (n1, _) = _pair(stage=1, msgs=[sig])
    ro, rn = o1.recv_tensor_dict(), n1.recv_tensor_dict()
    _same_dict(ro, rn)
    assert ro["__stage0_endpoint__"] == o0.router_endpoint
    assert o1._corr_in == n1._corr_in == BASE + 1
    assert o1._current_originator_endpoint == n1._current_originator_endpoint


# ---- receive rules: duplicate / stale / abandoned / ack / G1 ------------------------

def _g2(corr, origin=b"tcp://s0:1", nt=None):
    nt_b = pickle.dumps(nt) if nt else b""
    return _sig(corr, NEW._encode_g2(0, NEW._encode_layout([]), b"", origin, nt_b))


def _run_recv(mod, stage, msgs, k=1, prep=None, calls=1):
    c, sent = _mk(mod, stage=stage, msgs=msgs, k=k)
    if prep:
        prep(c)
    outs = [c.recv_tensor_dict() for _ in range(calls)]
    return outs, sent, (c._corr_in, c._pending_corr, c._current_originator_endpoint,
                        list(c.router_socket.msgs))


def _both(stage, msgs, **kw):
    a = _run_recv(OLD, stage, msgs, **kw)
    b = _run_recv(NEW, stage, msgs, **kw)
    assert a == b
    return a


def test_recv_stale_dup_rules_identical():
    msgs = [_g2(BASE + 3), _g2(BASE + 2), _g2(BASE + 4, nt={"x": 1})]
    (outs, sent, state) = _both(1, msgs, calls=2)
    assert outs[1]["x"] == 1 and state[0] == BASE + 4

    def dup(c):
        c._last_done = (BASE + 9, ("sig", b"CACHED"), 2)
    _both(1, [_g2(BASE + 9), _g2(BASE + 10)], prep=dup)

    def pending(c):
        c._pending_corr = BASE + 5
    _both(0, [_g2(BASE + 4), _g2(BASE + 5)], prep=pending)
    hdr = pickle.dumps({"layout": [], "nt": {"__stage0_endpoint__": "tcp://old:1", "k": 3}})
    _both(1, [_sig(1, b"G1" + hdr)])


def test_recv_k_window_identical():
    msgs = [_g2(BASE + 10), _g2(BASE + 8), _g2(BASE + 6), _g2(BASE + 11)]
    _both(2, msgs, k=4, calls=3)


# ---- K > 1 step API -------------------------------------------------------------------

def _steps(mod):
    c, sent = _mk(mod, stage=0, k=3)
    a, b, d = (c.new_step_id() for _ in range(3))
    c.abandon_step(b)
    c.router_socket = FakeRouter([_g2(d, nt={"mb": 3}), _g2(b, nt={"mb": 2}),
                                  _g2(a, nt={"mb": 1})])
    r = [c.recv_step(a, timeout_s=1)["mb"], c.recv_step(d, timeout_s=1)["mb"]]
    return r, sorted(c._mb_outstanding), dict(c._mb_done), (a, b, d)


def test_stage0_step_api_identical():
    assert _steps(OLD) == _steps(NEW)


def _replies(mod):
    c, sent = _mk(mod, stage=3, k=2)
    out = []
    for corr in (BASE + 1, BASE + 2):
        c.router_socket = FakeRouter([_g2(corr, origin=b"tcp://s0r1:1")])
        td, ctx = c.recv_forward()
        out.append(ctx)
        c.send_reply({"t": torch.ones(2), "cx_missing": ["r1"]}, ctx)
    # duplicate of step 1 -> cached reply re-sent, no delivery
    c.router_socket = FakeRouter([_g2(BASE + 1), _g2(BASE + 3)])
    td, ctx = c.recv_forward()
    out.append(ctx)
    return out, sent, list(c._done_ring)


def test_last_stage_reply_cache_identical():
    o, n = _replies(OLD), _replies(NEW)
    assert o == n
    assert len(o[1]) == 3  # two replies + one re-send of the cached first


# ---- GPUDirect path (G2 header, ring, ack, pinning) ----------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("stage", [0, 2])
def test_gdr_send_identical(stage):
    res = []
    for mod in (OLD, NEW):
        c, sent = _mk(mod, stage=stage)
        c.device = "cuda"
        c._gdr_buf = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda")
        c._gdr = mod._RegionTable(c._gdr_buf.numel(), token_base=BASE)
        c._gdr_send_evt = torch.cuda.Event()
        c._gdr_ack_on = True
        c._current_originator_endpoint = "tcp://s0:1"
        for i in range(3):
            c._corr_in = BASE + 1 + i
            td = _td(i)
            td["hidden_states"] = td["hidden_states"].cuda()
            c.send_tensor_dict(td)
        res.append((sent, c._gdr_buf.cpu().tolist(),
                    {t: (l.start, l.end, l.pinned, l.want_ack) for t, l in c._gdr.leases.items()},
                    c._last_done, c._last_token))
    assert res[0] == res[1]


# ---- routing ------------------------------------------------------------------------

def _route_run(mod):
    epc = FakeEPC([_route(1, 0, "s1r0:1"), _route(1, 1, "s1r1:1"), _route(2, 0, "s2r0:1"),
                   _route(2, 1, "s2r1:1"), _route(2, 2, "dup:1"), _route(3, 0, "dup:1")])
    c, _ = _mk(mod, stage=1)
    c._epc_client = epc
    out = [c._get_peer_endpoint(2)]
    c._deregistered_replicas.add((2, 1))
    c._sticky_peer.pop(2)
    out.append(c._get_peer_endpoint(2))
    out.append(c._get_peer_endpoint(2, exclude_replica=0))
    epc.routes = [_route(1, 1, "s1r1:1")]
    c._route_dirty.add(2)
    out.append(c._get_peer_endpoint(2))
    return out, dict(c._sticky_peer), epc.refreshes


def test_peer_selection_identical():
    assert _route_run(OLD) == _route_run(NEW)


def _events(mod):
    c, sent = _mk(mod, stage=1)
    c.forward_socket = c.return_socket = None
    c._peer_dealer_cache, c._last_failed_peer, c._remote_urls = {}, {}, {}
    c._peer_lock = __import__("threading").Lock()
    c._sticky_peer[2] = 1

    def ev(st, rep, **kw):
        return SimpleNamespace(payload={"lid": {"stage_idx": st, "replica_idx": rep}, **kw})
    c._on_endpoint_unhealthy(ev(2, 1))
    s1 = (set(c._suspect_replicas), c.downstream_suspect())
    c._on_endpoint_registered(ev(2, 1))
    c._on_endpoint_unhealthy(ev(2, 1))
    confirmed = c.confirm_peer_failure()
    return s1, confirmed, set(c._deregistered_replicas), dict(c._sticky_peer), \
        c.downstream_healthy()


def test_membership_events_identical():
    assert _events(OLD) == _events(NEW)


# ---- fast handover ----------------------------------------------------------------------

def _handover(mod, monkeypatch):
    import connex.pp.endpoint as core
    for m in (mod, core):
        monkeypatch.setattr(m, "_REMOTE_FAILOVER_DELAY_S", 0.0, raising=False)
    c, sent = _mk(mod, stage=0, k=4)
    c._init_handover()
    c._sticky_peer = {1: 0}
    c.forward_socket = c.return_socket = None
    c._peer_dealer_cache, c._last_failed_peer, c._remote_urls = {}, {}, {}
    c._peer_lock = __import__("threading").Lock()
    steps = [c.new_step_id() for _ in range(4)]
    for s in steps:
        c._mark_sent(s)
    c._sent_at[steps[0]] = (0, 0, -1e9)               # overdue
    c.on_peer_dead(2, 1)                               # farther down: remote epoch
    remote = [c.step_stale(s) for s in steps]
    c.on_peer_dead(1, 0)                               # our peer: chain epoch
    chain = [c.step_stale(s) for s in steps]
    c.router_socket = FakeRouter([])
    t = [c.recv_step(s, timeout_s=5.0) for s in steps]
    for s in steps:
        c._mark_sent(s)
    after = [c.step_stale(s) for s in steps]
    return (remote, chain, t, after, c._chain_epoch, c._remote_epoch,
            dict(c._sticky_peer), set(c._deregistered_replicas))


def test_fast_handover_state_identical(monkeypatch):
    a, b = _handover(OLD, monkeypatch), _handover(NEW, monkeypatch)
    assert a == b
    assert a[0] == [True, False, False, False] and a[1] == [True] * 4 and a[3] == [False] * 4


def test_k1_duplicate_reply_dropped_identically():
    def answered(c):
        c._last_reply_corr = BASE + 5
    _both(0, [_g2(BASE + 5), _g2(BASE + 4), _g2(BASE + 6, nt={"x": 6})], prep=answered)
