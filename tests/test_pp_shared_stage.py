# SPDX-License-Identifier: Apache-2.0
"""Two stage-0 replicas sharing one downstream chain (real endpoints, CPU).

Stage-0 r0 and r1 both forward to the only stage-1 replica, which forwards to
the only stage-2 replica. Each step must come back to the stage 0 that sent
it: the origin travels with the step (``__stage0_endpoint__``) and the last
stage replies to it. Each replica sends its own base value, so a reply routed
to the wrong replica fails the value check (or never arrives).
Covered on the transfer-engine path and on the ZeroMQ-only fallback.
"""

import socket
import threading

import pytest
import torch

from connex.pp import Stage0Link, StageWorker, StaticMembership
from connex.pp.routing import Lid, Route
from connex.pp.testing import LoopbackTransferEngine

from test_pp_core import DownstreamEngine, Pump, Stage0Engine


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mk(monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 400)
    made = []

    def make(stage, n, replica, routes, path, k=1):
        port = _port()
        mem = StaticMembership(stage, replica)
        mem.routes = routes
        kw = {}
        if path == "engine":
            url = f"loop-{stage}-{replica}-{port}"
            kw = dict(transfer_engine=LoopbackTransferEngine(url), local_url=url)
        else:
            monkeypatch.setenv("CONNEX_PP_BACKEND", "zmq")
        ep = core.ConnexTEEndpoint(stage, n, port, "127.0.0.1:1", device="cpu",
                                   membership=mem, inflight=k, **kw)
        made.append(ep)
        routes.append(Route(Lid(stage, replica), [ep.router_endpoint]))
        return ep

    yield make
    for ep in made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass


def _drive(s0, rids, base, steps, out, errors):
    try:
        n = {r: 0 for r in rids}
        res = s0.step({"h": torch.full((len(rids),), base), "nsched": [4] * len(rids)},
                      rids, [0] * len(rids))
        out.append(res["h"].tolist())
        for r in rids:
            n[r] = 4
        for _ in range(steps):
            res = s0.step({"h": torch.full((len(rids),), base)}, rids, [n[r] for r in rids])
            out.append(res["h"].tolist())
            for r in rids:
                n[r] += 1
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


@pytest.mark.parametrize("path,k", [("engine", 1), ("engine", 2), ("zmq", 1)])
def test_two_stage0_replicas_share_downstream(mk, path, k):
    if path == "zmq" and k > 1:
        pytest.skip("in-flight K>1 needs the transfer-engine step API")
    routes = []
    s0a = mk(0, 3, 0, routes, path, k)
    s0b = mk(0, 3, 1, routes, path, k)
    s1 = mk(1, 3, 0, routes, path, k)
    s2 = mk(2, 3, 0, routes, path, k)
    assert s0a._mooncake_available == (path == "engine")
    e1, e2 = DownstreamEngine("1"), DownstreamEngine("2")
    links = [Stage0Link(s0a, Stage0Engine(), recv_timeout_s=3.0),
             Stage0Link(s0b, Stage0Engine(), recv_timeout_s=3.0)]
    outs, errors = [[], []], []
    pump = Pump([StageWorker(s1, e1, e1.compute, commit=e1.commit),
                 StageWorker(s2, e2, e2.compute, commit=e2.commit)])
    with pump:
        ts = [threading.Thread(target=_drive, args=(links[0], ["a0", "a1"], 0.0, 6, outs[0], errors)),
              threading.Thread(target=_drive, args=(links[1], ["b0", "b1"], 100.0, 6, outs[1], errors))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    # Pump joins for 5 s only; a worker still inside recv when the fixture closes
    # the endpoints makes ZeroMQ abort the process (seen on one slow CPU).
    for t in pump.threads:
        t.join(30)
    assert not any(t.is_alive() for t in pump.threads)
    assert not errors, errors
    assert outs[0] == [[2.0, 2.0]] * 7
    assert outs[1] == [[102.0, 102.0]] * 7
    # the shared stages hold KV for both replicas' requests
    assert set(e2.ledger.kv_len) == {"a0", "a1", "b0", "b1"}
    # replies went back over one return socket per origin
    assert {key for key in s2._peer_dealer_cache if key.startswith("return_")} == {
        f"return_{s0a.router_endpoint}", f"return_{s0b.router_endpoint}"}
