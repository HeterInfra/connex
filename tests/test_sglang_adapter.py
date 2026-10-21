# SPDX-License-Identifier: Apache-2.0
"""ConnexPPGroup (SGLang pp_group replacement) with fake engines.

Drives the group the way SGLang's PP scheduler loop does (send "proxy"
forward, the last stage sends "output", stage 0 receives it), over the
in-process fake chain and over real Connex endpoints. SGLang itself is not
imported; its GroupCoordinator signatures are checked by parsing its source
when it is available.
"""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from connex.adapters.sglang import MISSING_HERE, MSG_TYPE, REPREFILLED, ConnexPPGroup
from connex.pp import KVLedger
from connex.pp.testing import fake_chain


class S0Hooks:
    def __init__(self):
        self.reprefilled, self.aborted = [], []

    def kv_held(self, r):
        return 0

    def release(self, rids):
        pass

    def reprefill(self, rids):
        self.reprefilled.extend(rids)

    def abort(self, rids):
        self.aborted.extend(rids)


class ExecHooks(S0Hooks):
    def __init__(self):
        super().__init__()
        self.ledger, self.released = KVLedger(), []

    def kv_held(self, r):
        return self.ledger.kv_held(r)

    def release(self, rids):
        rids = list(rids)
        self.released += rids
        self.ledger.forget(rids)


def _executor_step(group, hooks):
    """What a passive SGLang executor stage would do per microbatch."""
    td = group.recv_tensor_dict()
    assert td[MSG_TYPE] == "proxy"
    out = {"hidden_states": td["hidden_states"] + 1, "nsched": td.get("nsched")}
    if not group.is_last_rank:
        out[MSG_TYPE] = "proxy"
    else:
        out = {"next_token_ids": (td["hidden_states"] + 1).long()}
    hooks.ledger.commit(td.get("request_ids"), td.get("cx_ctx"), td.get("nsched"),
                        td[MISSING_HERE])
    group.send_tensor_dict(out)


class Rows:
    def __init__(self):
        self.rows = (None, None)

    def __call__(self):
        return self.rows


def _setup(links):
    rows = Rows()
    h0 = S0Hooks()
    g0 = ConnexPPGroup(links[0], h0, row_info=rows, recv_timeout_s=0.5)
    ex = [(ConnexPPGroup(l, h), h) for l, h in ((l, ExecHooks()) for l in links[1:])]
    return g0, h0, rows, ex


def _step(g0, rows, ex, rids, ncomp, nsched=None, ex_override=None):
    rows.rows = (rids, ncomp)
    td = {"hidden_states": torch.zeros(len(rids)), MSG_TYPE: "proxy"}
    if nsched:
        td["nsched"] = nsched
    work = g0.send_tensor_dict(td, all_gather_group=None, async_send=True)
    for w in work:
        w.work.wait()
    for g, h in (ex_override or ex):
        _executor_step(g, h)
    return g0.recv_tensor_dict()


def test_group_properties_and_roundtrip():
    links = fake_chain(3)
    g0, h0, rows, ex = _setup(links)
    assert g0.is_first_rank and not g0.is_last_rank and ex[-1][0].is_last_rank
    assert (g0.world_size, g0.next_rank, g0.prev_rank) == (3, 1, 2)
    out = _step(g0, rows, ex, ["a", "b"], [0, 0], nsched=[4, 4])
    assert out[MSG_TYPE] == "output" and out["next_token_ids"].tolist() == [2, 2]
    for _ in range(3):
        n = ex[0][1].ledger.kv_len["a"]
        out = _step(g0, rows, ex, ["a", "b"], [n, n])
        assert REPREFILLED not in out
    assert ex[1][1].ledger.kv_len == {"a": 7, "b": 7}


def test_microbatches_received_in_launch_order():
    links = fake_chain(2, k=2)
    g0, h0, rows, ex = _setup(links)
    for i in range(2):
        rows.rows = ([f"r{i}"], [0])
        g0.send_tensor_dict({"hidden_states": torch.full((1,), float(i)), MSG_TYPE: "proxy"})
    for _ in range(2):
        _executor_step(*ex[0])
    assert [g0.recv_tensor_dict()["next_token_ids"].item() for _ in range(2)] == [1, 2]


def test_failover_missing_reaches_stage0_reprefill():
    links = fake_chain(3)
    spare = fake_chain(3)[1]
    spare.head, spare.downstream = links[0], links[2]
    g0, h0, rows, ex = _setup(links)
    _step(g0, rows, ex, ["a"], [0], nsched=[4])
    _step(g0, rows, ex, ["a"], [4])
    links[0].downstream = spare                       # stage 1 replica replaced
    spare_hooks = ExecHooks()
    spare_ex = (ConnexPPGroup(spare, spare_hooks), spare_hooks)
    out = _step(g0, rows, ex, ["a", "b"], [5, 0], ex_override=[spare_ex, ex[1]])
    assert out[REPREFILLED] == ["a"] and h0.reprefilled == ["a"]
    assert spare_ex[1].released == ["a"]
    out = _step(g0, rows, ex, ["a", "b"], [0, 1], nsched=[6, 1], ex_override=[spare_ex, ex[1]])
    assert REPREFILLED not in out


def test_send_without_recv_on_executor_is_an_error():
    links = fake_chain(2)
    g1 = ConnexPPGroup(links[1], ExecHooks())
    with pytest.raises(RuntimeError):
        g1.send_tensor_dict({"x": 1})
    with pytest.raises(NotImplementedError):
        g1.send_object({}, dst=0)


def _sglang_parallel_state() -> Path | None:
    spec = importlib.util.find_spec("sglang")  # locates, does not import
    if spec and spec.submodule_search_locations:
        p = Path(list(spec.submodule_search_locations)[0]) / "srt/distributed/parallel_state.py"
        if p.exists():
            return p
    return None


@pytest.mark.skipif(_sglang_parallel_state() is None, reason="SGLang source not available")
def test_signatures_match_sglang_group_coordinator():
    tree = ast.parse(_sglang_parallel_state().read_text())
    gc = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator")
    theirs = {f.name: [a.arg for a in f.args.args + f.args.kwonlyargs]
              for f in gc.body if isinstance(f, ast.FunctionDef)}
    for name in ("send_tensor_dict", "recv_tensor_dict", "send_object", "recv_object"):
        ours = list(inspect.signature(getattr(ConnexPPGroup, name)).parameters)
        # every argument SGLang passes by name must be accepted
        assert set(theirs[name]) <= set(ours), (name, theirs[name], ours)
    for prop in ("is_first_rank", "is_last_rank", "next_rank", "prev_rank"):
        assert prop in theirs and isinstance(getattr(ConnexPPGroup, prop), property)
    assert "sglang" not in sys.modules


def test_real_endpoints(monkeypatch):
    import socket as _s
    import connex.pp.endpoint as core
    from connex.pp.routing import StaticMembership
    from connex.pp.testing import LoopbackTransferEngine
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 2000)
    routes, eps = [], []
    for st in range(3):
        with _s.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mem = StaticMembership(st, 0)
        mem.routes = routes
        url = f"sgl-{st}-{port}"
        ep = core.ConnexTEEndpoint(st, 3, port, "127.0.0.1:1", device="cpu", membership=mem,
                                   transfer_engine=LoopbackTransferEngine(url), local_url=url)
        mem.add(st, 0, ep.router_endpoint)
        eps.append(ep)
    try:
        import threading
        g0, h0, rows, ex = _setup(eps)
        # executors block in recv; run them for exactly the steps we send
        n_steps = 4
        ths = [threading.Thread(target=lambda g=g, h=h: [_executor_step(g, h) for _ in range(n_steps)],
                                daemon=True) for g, h in ex]
        for t in ths:
            t.start()
        rows.rows = (["a"], [0])
        g0.send_tensor_dict({"hidden_states": torch.zeros(1), "nsched": [4], MSG_TYPE: "proxy"})
        assert g0.recv_tensor_dict()["next_token_ids"].tolist() == [2]
        for i in range(n_steps - 1):
            rows.rows = (["a"], [4 + i])
            g0.send_tensor_dict({"hidden_states": torch.zeros(1), MSG_TYPE: "proxy"})
            out = g0.recv_tensor_dict()
            assert REPREFILLED not in out and out[MSG_TYPE] == "output"
        for t in ths:
            t.join(5)
        assert ex[1][1].ledger.kv_len == {"a": 7}
    finally:
        for ep in eps:
            ep.close()
