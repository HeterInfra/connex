"""One stage per instance (mode 2): fixes from a heterogeneous-node run.

* A stage server built its vLLM config in ENGINE_CONTEXT, whose
  max_num_seqs default (128) is below `vllm serve`'s (256): stage 0
  scheduled more requests than stage k had slots ("No free indices").
* After a stage loss the route of a step can hold None (no live replica);
  the sampled-token send crashed on it.
* Without a peer-memory driver, GPU memory over RDMA made UCX fall back to
  CPU access of GPU pointers (SIGSEGV): fail closed instead.
"""

import ast
import pathlib
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from connex.adapters.vllm.pp import link as L  # noqa: E402

from test_fixes_vllm_pp import _group_module, _link  # noqa: E402

PP = pathlib.Path(L.__file__).parent


def test_stage_server_uses_the_serve_usage_context():
    tree = ast.parse((PP / "stage.py").read_text())
    ctx = [n.args[0].attr for n in ast.walk(tree)
           if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "create_engine_config"]
    assert ctx == ["OPENAI_API_SERVER"]


def test_require_gdr_is_fatal_without_peer_memory():
    lk = _link()
    lk._gdr = False
    assert lk.require_gdr("sending x") is False
    assert lk.fatals and "GPUDirect" in lk.fatals[0] and "sending x" in lk.fatals[0]
    lk2 = _link()
    assert lk2.require_gdr("sending x") is True and not lk2.fatals


def test_cross_host_send_without_gdr_does_not_touch_the_ring():
    lk = _link()
    lk._gdr = False
    lk.ipc_slot_bytes = 0          # not a same-host (IPC) peer
    used = []
    lk.send_ring = SimpleNamespace(alloc=lambda *a, **k: used.append(a))
    lk.send_tensors("10.0.0.2:5#cxpp|2|0|0", (1, 1, 0), {"h": torch.zeros(4)})
    assert lk.fatals and not used and not lk.engine.sent


def test_send_sampled_skips_a_route_entry_without_replica(monkeypatch):
    G = _group_module(monkeypatch)
    lk = _link()
    lk._ip = "10.0.0.1"
    lk.ipc_slot_bytes = 0
    sent = []
    lk.send_bytes = lambda url, key, payload, event=None: sent.append(url)
    g = object.__new__(G.ConnexPPGroup)
    g.link = lk
    g.is_endpoint = True
    g.external = False
    g.world_size = 3
    g.tp_rank = 0
    g.device = torch.device("cpu")
    g._route = ["10.0.0.3:5#cxpp|0|0|0", None, "10.0.0.1:5#cxpp|2|0|0"]   # stage 1 lost
    monkeypatch.setattr(G.torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(G.ConnexPPGroup, "_bkey", lambda self: (1, 7, 0), raising=False)

    class _Ev:
        def record(self, stream):
            pass
    monkeypatch.setattr(G.torch.cuda, "Event", _Ev)
    monkeypatch.setattr(G.torch, "empty",
                        lambda shape, dtype=None, pin_memory=False: torch.zeros(shape, dtype=dtype))
    g._send_sampled(torch.zeros(2, dtype=torch.int64))
    assert sent == ["10.0.0.3:5#cxpp|0|0|0"]
