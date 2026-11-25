# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ fallback carries correlation ids (issues #33, #34).

A step that times out and is re-sent must not shift later replies by one, the
downstream stage must answer the retry from its reply cache instead of
computing the step again, and K>1 must work without a transfer engine.
"""

import socket
import threading
import time

import pytest
import torch

from connex.pp import Stage0Link, StageWorker, StaticMembership
from connex.pp.routing import Lid, Route


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
def zchain(monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("CONNEX_PP_BACKEND", "zmq")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 3000)
    made = []

    def make(n, k=1):
        routes, eps = [], []
        for st in range(n):
            port = _port()
            mem = StaticMembership(st, 0)
            mem.routes = routes
            ep = core.ConnexTEEndpoint(st, n, port, "127.0.0.1:1", device="cpu",
                                       membership=mem, inflight=k)
            assert not ep._mooncake_available
            made.append(ep)
            routes.append(Route(Lid(st, 0), [ep.router_endpoint]))
            eps.append(ep)
        return eps

    yield make
    for ep in made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass


def _serve(eps, compute, stop):
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


def _stop(stop, threads):
    stop.set()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads)


def test_retry_does_not_shift_later_replies(zchain):
    eps = zchain(3)
    calls = {"n": 0}

    def compute(td, missing):
        calls["n"] += 1
        if float(td["h"][0]) == 5.0 and not calls.get("slept"):
            calls["slept"] = True
            time.sleep(0.6)   # stage 0 times out and re-sends this step
        return {"h": td["h"] + 1}

    stop = threading.Event()
    threads = _serve(eps, compute, stop)
    s0 = Stage0Link(eps[0], _Hooks(), recv_timeout_s=0.3, send_ctx=False)
    wrong = 0
    for i in range(20):
        r = s0.step({"h": torch.full((2,), float(i))})
        wrong += not torch.equal(r["h"], torch.full((2,), float(i + 2)))
    _stop(stop, threads)
    assert s0.stats["retries"] >= 1
    assert wrong == 0
    # the re-sent step was answered from the reply cache, not computed again
    assert calls["n"] <= 2 * 20 + 1


def test_inflight_two_without_transfer_engine(zchain):
    eps = zchain(3, k=2)
    stop = threading.Event()
    threads = _serve(eps, lambda td, m: {"h": td["h"] + 1}, stop)
    s0 = Stage0Link(eps[0], _Hooks(), recv_timeout_s=2.0, send_ctx=False)
    pend, wrong = [], 0
    for i in range(60):
        pend.append((s0.submit({"h": torch.full((2,), float(i))}), i))
        if len(pend) >= 2:
            c, v = pend.pop(0)
            wrong += not torch.equal(s0.wait(c)["h"], torch.full((2,), float(v + 2)))
    for c, v in pend:
        wrong += not torch.equal(s0.wait(c)["h"], torch.full((2,), float(v + 2)))
    _stop(stop, threads)
    assert wrong == 0 and s0.stats["retries"] == 0


def test_corr_key_is_not_handed_to_the_engine(zchain):
    eps = zchain(2)
    eps[0].send_tensor_dict({"h": torch.zeros(2), "num_tokens": 2})
    got = eps[1].recv_tensor_dict()
    assert set(got) == {"h", "num_tokens", "__stage0_endpoint__"}
