# SPDX-License-Identifier: Apache-2.0
"""connex.ep SGLang glue without SGLang: the placement view is written into
tensors shaped like SGLang's ExpertLocationMetadata, and the control endpoint
runs commands on a scheduler stand-in between its "steps"."""
import json
import socket
import threading
import time

import torch

from connex.ep import EPController, ExpertStore, PlacementTable, logical_to_physical
from connex.ep import sglang_ext
from connex.rl.transport import LocalTransport

E, R, S, L, H = 4, 2, 4, 2, 3


class FakeMD:
    """The fields of sglang's ExpertLocationMetadata that Connex writes."""

    def __init__(self, rows):
        p2l = torch.tensor(rows)
        self.physical_to_logical_map = p2l.clone()
        self.physical_to_logical_map_cpu = p2l.clone()
        P = p2l.shape[1]
        l2p = torch.full((L, E, P), -1)
        for l in range(L):
            for e in range(E):
                ps = [p for p in range(P) if rows[l][p] == e]
                l2p[l, e, :len(ps)] = torch.tensor(ps)
        self.logical_to_all_physical_map = l2p
        self.logical_to_all_physical_map_cpu = l2p.clone()
        self.logical_to_all_physical_map_num_valid = (l2p != -1).sum(-1)
        self.logical_to_rank_dispatch_physical_map = None
        self.ep_size = R
        self.num_layers = L
        self.num_logical_experts = E
        self.num_local_physical_experts = P // R


class FakeWorker(sglang_ext.SGLangEPWorker):
    def __init__(self, md, rank, layers):
        self.md, self.rank, self.layers = md, rank, layers

    def _md(self):
        return self.md

    def _ep_rank(self):
        return self.rank

    def _dispatch(self):
        return "dynamic"

    def _expert_layers(self):
        return self.layers


def linear_rows():
    return [[p % E for p in range(R * S)] for _ in range(L)]


def pick(md, layer, expert, row):
    """SGLang's dynamic, rank-invariant dispatch: replica row % num_valid."""
    n = int(md.logical_to_all_physical_map_num_valid[layer, expert])
    return int(md.logical_to_all_physical_map[layer, expert, row % n])


def test_apply_writes_sglang_maps_in_place():
    md = FakeMD(linear_rows())
    ptrs = [md.logical_to_all_physical_map.data_ptr(),
            md.logical_to_all_physical_map_num_valid.data_ptr()]
    w = FakeWorker(md, 0, [])
    t = PlacementTable(E, R, S, linear_rows())
    t.mark_down(1)
    t.commit()
    v = t.view()
    out = w.connex_ep_apply(json.dumps(v))
    assert out["applied"] and out["epoch"] == 1
    # same storage (CUDA graphs keep their pointers), new contents
    assert ptrs == [md.logical_to_all_physical_map.data_ptr(),
                    md.logical_to_all_physical_map_num_valid.data_ptr()]
    for l in range(L):
        a, c = logical_to_physical(v, l)
        assert md.logical_to_all_physical_map_num_valid[l].tolist() == c == [1] * E
        for e in range(E):
            assert md.logical_to_all_physical_map[l, e, 0] == a[e][0]
            assert (md.logical_to_all_physical_map[l, e, 1:] == -1).all()
            for row in range(16):
                assert pick(md, l, e, row) // S == 0          # never the down rank
    # a stale or duplicate view is ignored
    assert not w.connex_ep_apply(json.dumps(v))["applied"]
    # rank 1 back: both replicas routed again, the padding stays consistent
    t.mark_up(1)
    t.commit()
    w.connex_ep_apply(t.view())
    assert md.logical_to_all_physical_map_num_valid.tolist() == [[2] * E] * L
    assert {pick(md, 0, 1, r) for r in range(4)} == {1, 5}


def test_apply_static_dispatch_prefers_local_replica():
    md = FakeMD(linear_rows())
    md.logical_to_rank_dispatch_physical_map = torch.zeros(L, E, dtype=torch.long)
    w = FakeWorker(md, 1, [])
    t = PlacementTable(E, R, S, linear_rows())
    t.commit()
    w.connex_ep_apply(t.view())
    assert md.logical_to_rank_dispatch_physical_map[0].tolist() == [4, 5, 6, 7]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeScheduler:
    """Runs the patched-scheduler logic: pull queued commands with the aborts
    of an iteration, then execute them before the batch."""

    def __init__(self, worker):
        self._connex_ep = worker
        self.steps = 0
        self.stop = False

    def loop(self):
        while not self.stop:
            cmds = []
            while not self._connex_ep_q.empty():
                cmds.append(self._connex_ep_q.get_nowait())
            for c in cmds:
                sglang_ext._run_cmd(self, c)
            self.steps += 1
            time.sleep(0.001)


def test_control_endpoint_drives_epcontroller_end_to_end():
    W = torch.randn(E, H, H, generator=torch.Generator().manual_seed(0))
    rows = linear_rows()
    md = FakeMD(rows)
    # one process stands in for both ranks: rank 0's worker holds every slot
    layers = [[torch.stack([W[rows[l][p]] for p in range(R * S)]).contiguous()]
              for l in range(L)]
    w = FakeWorker(md, 0, layers)
    w.connex_ep_init = lambda cfg="{}": ExpertStore(0, layers, LocalTransport("ep-r0")).describe()
    sched = FakeScheduler(w)
    import queue
    sched._connex_ep_q = queue.Queue()
    sched._connex_ep_wait = {}
    port = _free_port()
    sglang_ext._serve(sched, port)
    th = threading.Thread(target=sched.loop, daemon=True)
    th.start()
    try:
        eng = sglang_ext.SGLangEPEngine(f"http://127.0.0.1:{port}", timeout=10)
        info = eng.info()[0]
        assert (info["num_ranks"], info["slots_per_rank"], info["num_experts"]) == (R, S, E)
        ctl = EPController(eng.table(), eng)
        ctl.push(force=True)
        rep = ctl.drain_rank(1)
        assert rep["uncovered"] == 0 and md.logical_to_all_physical_map_num_valid.tolist() == \
            [[1] * E] * L
        assert eng.last["apply"][0]["applied"]
        # an engine error comes back as an exception, not a silent skip
        try:
            eng.rpc("connex_ep_apply", {"epoch": 99, "num_ranks": 9, "slots_per_rank": 1,
                                        "slot_map": [[0]]})
            raise AssertionError("expected an error")
        except RuntimeError as ex:
            assert "geometry" in str(ex)
    finally:
        sched.stop = True
        sched._connex_ep_srv.shutdown()


def test_init_registers_in_the_background_and_polls():
    layers = [[torch.zeros(S, H, H)] for _ in range(L)]
    w = FakeWorker(FakeMD(linear_rows()), 0, layers)
    out = w.connex_ep_init(json.dumps({"transport": "local"}))
    deadline = time.time() + 10
    while out.get("pending") and time.time() < deadline:
        time.sleep(0.01)
        out = w.connex_ep_init(json.dumps({"transport": "local"}))
    assert out["rank"] == 0 and len(out["layers"]) == L
    assert w.connex_ep_init("{}") == out                 # registered once
