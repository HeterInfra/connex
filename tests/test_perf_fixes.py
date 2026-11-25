# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the performance fixes of issues #32, #35, #36, #37, #38."""

import random
import socket
import threading
import time

import pytest
import torch

from connex.pp import Stage0Link, StageWorker, StaticMembership
from connex.pp.routing import Lid, Route
from connex.pp.testing import LoopbackTransferEngine


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Hooks:
    def kv_held(self, rid):
        return 0

    def release(self, rids):
        pass

    def reprefill(self, rids):
        pass

    def abort(self, rids):
        pass


@pytest.fixture
def chain(monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 3000)
    made = []

    def make(n, path="engine"):
        if path == "zmq":
            monkeypatch.setenv("CONNEX_PP_BACKEND", "zmq")
        routes, eps, engines = [], [], []
        for st in range(n):
            port = _port()
            mem = StaticMembership(st, 0)
            mem.routes = routes
            kw = {}
            if path == "engine":
                url = f"loop-{st}-{port}"
                eng = LoopbackTransferEngine(url)
                engines.append(eng)
                kw = dict(transfer_engine=eng, local_url=url)
            ep = core.ConnexTEEndpoint(st, n, port, "127.0.0.1:1", device="cpu",
                                       membership=mem, **kw)
            made.append(ep)
            routes.append(Route(Lid(st, 0), [ep.router_endpoint]))
            eps.append(ep)
        return eps, engines

    yield make
    for ep in made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass


def _serve(eps, stop, compute=lambda td, missing: {"h": td["h"] + 1}):
    threads = []
    for ep in eps[1:]:
        w = StageWorker(ep, _Hooks(), compute)

        def run(w=w):
            while not stop.is_set():
                try:
                    w.serve_one()
                except Exception:  # noqa: BLE001
                    if stop.is_set():
                        return
                    raise
        t = threading.Thread(target=run, daemon=True)
        t.start()
        threads.append(t)
    return threads


def _join(threads):
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads)


# --- #32: host-path send buffers are freed as their hold expires --------------

def test_host_path_buffers_bounded(chain, monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setattr(core, "_HOST_FREE_HOLD_S", 0.05)
    eps, engines = chain(2)
    assert eps[0]._gdr is None  # host path
    stop = threading.Event()
    threads = _serve(eps, stop)
    s0 = Stage0Link(eps[0], _Hooks(), recv_timeout_s=3.0, send_ctx=False)
    for i in range(300):
        r = s0.step({"h": torch.full((2,), float(i))})
        assert torch.equal(r["h"], torch.full((2,), float(i + 1)))
    time.sleep(0.1)
    s0.step({"h": torch.zeros(2)})  # a send reaps every expired buffer
    stop.set()
    _join(threads)
    # the old one-free-per-second worker left ~300 live per stage here
    assert len(eps[0]._deferred_frees) <= 2
    assert sum(len(e.mem) for e in engines) <= 8
    for ep in eps:
        ep.close()
    assert all(not ep._deferred_frees for ep in eps)  # close frees the rest


# --- #35: recv_step honours the caller's timeout at K=1 ----------------------

def test_k1_recv_step_uses_caller_timeout(chain):
    eps, _ = chain(2)
    s0 = eps[0]
    corr = s0.new_step_id()
    s0.send_step({"h": torch.zeros(2)}, corr)  # nobody serves stage 1
    t = time.monotonic()
    assert s0.recv_step(corr, 0.3) is None
    took = time.monotonic() - t
    assert took < 1.5, took  # the endpoint default here is 3 s


# --- #36: the loopback engine tolerates concurrent alloc and read -------------

def test_loopback_engine_concurrent_find():
    a, b = LoopbackTransferEngine("cc-a"), LoopbackTransferEngine("cc-b")
    src = a.allocate_managed_buffer(64)
    a.write_bytes_to_buffer(src, b"x" * 64, 64)
    dst = b.allocate_managed_buffer(64)
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            p = a.allocate_managed_buffer(16)
            a.free_managed_buffer(p, 16)

    t = threading.Thread(target=churn)
    t.start()
    try:
        for _ in range(20000):
            assert b.transfer_sync_read("cc-a", dst, src, 64) == 0
    finally:
        stop.set()
        t.join()


# --- #37: raw tensor frames on the ZeroMQ fallback ---------------------------

@pytest.mark.parametrize("raw_send", [True, False])
def test_zmq_fallback_roundtrip_raw_and_legacy(chain, monkeypatch, raw_send):
    import connex.pp.zmq_base as zb
    monkeypatch.setattr(zb, "_ZMQ_RAW", raw_send)  # legacy senders still read
    eps, _ = chain(2, "zmq")
    assert not eps[0]._mooncake_available
    sent = {
        "f32": torch.randn(3, 5),
        "bf16": torch.randn(4, 2).to(torch.bfloat16),
        "i64": torch.arange(7),
        "mask": torch.tensor([True, False, True]),
        "empty": torch.empty(0, 4),
        "scalar": torch.tensor(2.5),
        "strided": torch.randn(6, 4)[:, 1],
        "meta": {"num_tokens": 3, "ids": ["a", "b"]},
    }
    eps[0].send_tensor_dict(dict(sent))
    got = eps[1].recv_tensor_dict()
    for k, v in sent.items():
        if isinstance(v, torch.Tensor):
            assert got[k].dtype == v.dtype and got[k].shape == v.shape, k
            assert torch.equal(got[k], v), k
        else:
            assert got[k] == v


def test_zmq_raw_frames_are_cheaper_than_torch_save():
    import connex.pp.zmq_base as zb
    t = torch.randn(64)
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        zb._bytes_to_tensor(zb._tensor_to_bytes(t), t.dtype, t.shape)
    raw = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(n):
        import io
        torch.load(io.BytesIO(zb._torch_save(t)), weights_only=True)
    saved = time.perf_counter() - start
    assert raw * 3 < saved, (raw, saved)


# --- #38 (PD): registered-span check by bisection ----------------------------

def _brute(spans, ops):
    for dst, src, n in ops:
        if not any(s <= dst and dst + n <= s + ln for s, ln in spans):
            return (hex(dst), n)
    return None


def test_pd_outside_matches_linear_scan():
    from connex.pd.transfer import KVConsumer
    rng = random.Random(7)
    for _ in range(200):
        spans, at = [], 0
        for _ in range(rng.randint(1, 12)):
            at += rng.randint(0, 64)
            ln = rng.randint(1, 256)
            spans.append((at, ln))
            at += ln
        rng.shuffle(spans)
        ops = []
        for _ in range(rng.randint(1, 20)):
            s, ln = rng.choice(spans)
            dst = s + rng.randint(-8, ln)
            ops.append((dst, 0, rng.randint(1, 32)))
        obj = KVConsumer.__new__(KVConsumer)
        obj._spans = spans
        assert obj._outside(ops) == _brute(spans, ops), (spans, ops)
