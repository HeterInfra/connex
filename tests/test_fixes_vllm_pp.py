# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Regression tests for the vLLM PP adapter (CPU only, no connex-rs)."""

import importlib
import queue
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from connex.adapters.vllm.pp import link as L  # noqa: E402
from connex.adapters.vllm.pp.hostwait import HostFlagWaiter  # noqa: E402



class _Engine:
    def __init__(self):
        self.sent = []

    def send_msg(self, peer, data):
        self.sent.append(peer)


def _link():
    """A StageLink without connex-rs, GPU or threads (as in
    test_pp_upstream_vllm, which needs vllm to import)."""
    lk = object.__new__(L.StageLink)
    lk.engine = _Engine()
    lk.url = "10.0.0.1:1#cxpp|1|0|0"
    lk.inbox = L._Inbox()
    lk.push_fallback = {}
    lk.min_epoch = 0
    lk.stats = {"sent": 0, "recv": 0, "bytes_sent": 0, "bytes_recv": 0, "read_s": 0.0,
                "chain_err": 0, "fallback_fill": 0}
    lk._push_rx, lk._rx_chans, lk._tx_chans = {}, {}, {}
    lk._chan_lock = threading.RLock()
    lk._retired_cids = deque(maxlen=256)
    lk._graveyard = None
    lk._announced = {}
    lk._tx = queue.Queue()
    lk._reap = queue.Queue()
    lk._closed = False
    lk._fatal = False
    lk.fatals = []
    lk.on_fatal = lk.fatals.append
    lk.abort_callbacks = []
    lk._post_lock, lk._out_lock = threading.Lock(), threading.Lock()
    lk._posted, lk._ready, lk._outstanding = {}, {}, {}
    lk._waiter = None
    lk._prev_off = None
    lk._rx_done = None
    lk.send_ring = L._Ring(1 << 20)
    lk.ipc_slot_bytes = 0
    return lk


class _Chan:
    def __init__(self, cid=None, key=None):
        self.cid, self.key, self.retired = cid, key, False
        self.from_key = None

    def retire(self):
        self.retired = True
        return []

    def close(self):
        pass


def _cpu_waiter(num_slots=8, slot_bytes=64):
    """A HostFlagWaiter on plain host memory (no CUDA): only the CPU-side
    slot protocol (no_wait / fill / wait_filled) is used."""
    w = object.__new__(HostFlagWaiter)
    w.num_slots, w.slot_bytes = num_slots, slot_bytes
    w.flags_np = np.zeros(num_slots, dtype=np.int32)
    w.staging_np = np.zeros(num_slots * slot_bytes, dtype=np.uint8)
    w.gen = np.zeros(num_slots, dtype=np.int64)
    w.busy, w.next, w.lock = {}, 0, threading.Lock()
    return w


def _recv_link(monkeypatch):
    lk = _link()
    lk.device = torch.device("cpu")
    lk.recv_ring = L._Ring(1 << 20)
    lk.recv_buf = torch.zeros(1 << 20, dtype=torch.uint8)
    lk._waiter = _cpu_waiter()
    monkeypatch.setattr(L.torch.cuda, "current_stream", lambda dev=None: None)
    return lk


def _header():
    return L._Header([("h", "torch.float32", (4,), 0, 16)], 16, None, None,
                     ("cxpp", 0, 0, 0))


def _run(fn, timeout=3.0):
    out = {}

    def body():
        out["ret"] = fn()

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout)
    return t.is_alive(), out.get("ret", "unset")


def test_abort_between_take_and_post_does_not_hang(monkeypatch):
    """An ABORT that lands after take() returned the header but before the
    receive posted its slot left a post that nobody fills (abort had already
    swept _posted); the blocking wait spun forever."""
    lk = _recv_link(monkeypatch)
    key = (1, 7, -1)
    lk.inbox.items[key] = _header()
    orig_take = L.StageLink.take

    def take_then_abort(key, timeout=None):
        item = orig_take(lk, key, timeout)
        lk.abort(key[0] + 1)          # the sender failed over meanwhile
        return item

    lk.take = take_then_abort
    alive, ret = _run(lambda: lk.recv_tensors(key, blocking=True))
    assert not alive, "receive hangs on a post that the abort missed"
    assert ret is None
    assert key not in lk._posted
    assert not lk.fatals


def test_abort_between_take_and_post_releases_stream_wait(monkeypatch):
    """The same race on the stream-wait path: the slot is released at once
    (its stream wait would otherwise never be satisfied)."""
    lk = _recv_link(monkeypatch)
    w = lk._waiter
    posted = []
    w.post_wait = lambda: posted.append(w.no_wait()) or posted[-1]
    key = (1, 8, -1)
    lk.inbox.items[key] = _header()
    orig_take = L.StageLink.take

    def take_then_abort(key, timeout=None):
        item = orig_take(lk, key, timeout)
        lk.abort(key[0] + 1)
        return item

    lk.take = take_then_abort
    assert lk.recv_tensors(key, blocking=False) is None
    slot = posted[0]
    assert w.flags_np[slot] == int(w.gen[slot]) & 0x7FFFFFFF   # released
    assert key not in lk._posted


def test_blocking_receive_without_data_is_fatal_after_deadline(monkeypatch):
    """A live step whose READY never comes used to block on the CPU forever."""
    lk = _recv_link(monkeypatch)
    monkeypatch.setattr(L, "RECV_WAIT_S", 0.2)
    key = (1, 9, -1)
    lk.inbox.items[key] = _header()
    alive, ret = _run(lambda: lk.recv_tensors(key, blocking=True))
    assert not alive and ret is None
    assert lk.fatals and "did not arrive" in lk.fatals[0]
    assert key not in lk._posted


def test_blocking_receive_released_by_abort_returns_none(monkeypatch):
    lk = _recv_link(monkeypatch)
    key = (1, 10, -1)
    lk.inbox.items[key] = _header()
    threading.Timer(0.1, lambda: lk.abort(2)).start()
    alive, ret = _run(lambda: lk.recv_tensors(key, blocking=True))
    assert not alive and ret is None and not lk.fatals


def test_blocking_receive_still_delivers(monkeypatch):
    lk = _recv_link(monkeypatch)
    key = (1, 11, -1)
    lk.inbox.items[key] = _header()
    reads = []

    def read(src, dst, src_ptr, n, timeout_ms):
        reads.append(n)
        return 0

    lk.engine.transfer_sync_read = read
    lk.engine.send_msg = lambda *a: None
    lk._ready[key] = (("cxpp", 0, 0, 0), (L.READY, key, "src", 0, 3, 16))
    alive, ret = _run(lambda: lk.recv_tensors(key, blocking=True))
    assert not alive and ret is not None and reads
    assert ret[0]["h"].shape == (4,)


def test_rx_channels_of_peers_off_the_route_are_retired():
    """Only tx channels were retired when a peer left the route; the receive
    channels (and their rings) of a departed upstream stayed forever."""
    lk = _link()
    a, b = ("m", 0, 0, 0), ("m", 0, 0, 1)
    ra, rb, rb_old, rb_ipc = _Chan(1, (a, ("p", "h"))), _Chan(2, (b, ("p", "h"))), \
        _Chan(3, (b, ("p", "h"))), _Chan()
    lk._rx_chans = {(a, ("p", "h")): ra, (b, ("p", "h")): rb, (b, ("h",)): rb_ipc}
    lk._push_rx = {1: ra, 2: rb, 3: rb_old}   # 3: b's previous push channel
    lk.retain_rx({a})
    assert rb.retired and rb_old.retired and rb_ipc.retired and not ra.retired
    assert list(lk._rx_chans) == [(a, ("p", "h"))] and list(lk._push_rx) == [1]


def _group_module(monkeypatch):
    """connex.adapters.vllm.pp.group, with a stand-in for vllm's
    parallel_state when vllm itself does not import."""
    try:
        return importlib.import_module("connex.adapters.vllm.pp.group")
    except ImportError:
        pass
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.GroupCoordinator = type("GroupCoordinator", (), {})
    ps.get_tp_group = lambda: None
    dist = types.ModuleType("vllm.distributed")
    dist.parallel_state = ps
    vllm = types.ModuleType("vllm")
    vllm.distributed = dist
    for name, mod in (("vllm", vllm), ("vllm.distributed", dist),
                      ("vllm.distributed.parallel_state", ps)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "connex.adapters.vllm.pp.group", raising=False)
    mod = importlib.import_module("connex.adapters.vllm.pp.group")
    # not left behind for tests that need the real vllm
    monkeypatch.delitem(sys.modules, "connex.adapters.vllm.pp.group", raising=False)
    return mod


def test_sampled_filler_does_not_wait_forever_on_a_lost_key(monkeypatch):
    """_SampledFiller took each key without a timeout: one lost key blocked
    the stream wait of that step and every receive queued behind it."""
    G = _group_module(monkeypatch)
    monkeypatch.setattr(G, "SAMPLED_WAIT_S", 0.2)
    lk = _link()
    filled = []
    g = SimpleNamespace(link=lk, _pending_slots={},
                        _waiter=SimpleNamespace(fill=lambda slot, d: filled.append((slot, d))))
    f = object.__new__(G._SampledFiller)
    f.group = g
    f.q = queue.Queue()
    threading.Thread(target=f._loop, daemon=True).start()
    lost, ok = (1, 5, 0), (1, 6, 0)
    g._pending_slots.update({lost: 3, ok: 4})
    lk.inbox.items[ok] = b"tok"
    f.add(lost, 3)
    f.add(ok, 4)
    deadline = time.monotonic() + 3
    while len(filled) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert lk.fatals and "did not arrive" in lk.fatals[0]
    assert filled == [(3, None), (4, b"tok")]
    # an aborted key is released without a failure
    lk.fatals.clear()
    lk.min_epoch = 2
    g._pending_slots[(1, 7, 0)] = 5
    f.add((1, 7, 0), 5)
    deadline = time.monotonic() + 3
    while len(filled) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert filled[-1] == (5, None) and not lk.fatals


class _Ev:
    def __init__(self):
        self.done, self.synced = False, 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synced += 1
        self.done = True


class _Dst:
    def __init__(self):
        self.copies = []

    def copy_(self, src, non_blocking=False):
        self.copies.append((bytes(src.numpy()), non_blocking))


def test_pinned_staging_reuses_buffers_and_never_overwrites_a_pending_copy(monkeypatch):
    """The CPU-filled sampled tokens were copied from pageable memory with a
    blocking copy, which waits for all work queued on the stream every step.
    Now: a pinned buffer, a non-blocking copy, and a buffer is rewritten only
    after the copy that read it has run."""
    G = _group_module(monkeypatch)
    st = G._PinnedStaging()
    allocs, events = [], []
    st._alloc = lambda n: allocs.append(n) or torch.zeros(n, dtype=torch.uint8)
    st._event = lambda stream: events.append(_Ev()) or events[-1]
    d = _Dst()
    st.copy(d, b"abcd", None)
    st.copy(d, b"efgh", None)               # first copy still pending: second buffer
    assert len(allocs) == 2
    assert d.copies == [(b"abcd", True), (b"efgh", True)]
    events[0].done = True
    st.copy(d, b"ijkl", None)               # first buffer reused
    assert len(allocs) == 2 and bytes(st.bufs[0][0][:4].numpy()) == b"ijkl"
    assert bytes(st.bufs[1][0][:4].numpy()) == b"efgh"   # pending: untouched
    # all buffers in flight: wait for the oldest copy, never overwrite it early
    for _ in range(G._PinnedStaging.MAX_BUFS):
        st.copy(d, b"zzzz", None)
    assert len(st.bufs) == G._PinnedStaging.MAX_BUFS
    assert sum(e.synced for e in events) >= 1
