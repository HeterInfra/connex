# SPDX-License-Identifier: Apache-2.0
"""The external-PP tensor-parallel patch (integrations/vllm/patches/
patch_external_pp_tp.py) on a synthetic fork tree: applies once, is
idempotent, and the patched group coordinator lets only TP rank 0 talk to
other stages while the others take the step from a TP broadcast."""
import importlib.util
import os
import sys
import types

import pytest

PATCH = os.path.join(os.path.dirname(__file__), "..", "integrations", "vllm", "patches",
                     "patch_external_pp_tp.py")


def load():
    spec = importlib.util.spec_from_file_location("ptp", PATCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fork_tree(tmp_path, m):
    root = tmp_path / "vllm"
    (root / "config").mkdir(parents=True)
    (root / "distributed").mkdir()
    (root / "v1" / "worker").mkdir(parents=True)
    (root / "v1" / "worker" / "gpu_model_runner.py").write_text("")
    (root / "config" / "parallel.py").write_text(
        "class P:\n    def check(self):\n        if True:\n" + m.CONFIG_OLD + "        return 1\n")
    ps = ("_TP = None\n\ndef get_tp_group():\n    return _TP\n\n"
          "def init_external_pp_group(stage_idx, CommunicatorClass, num_stages=2, router_port=1,\n"
          "                           front_router_addr='x', device='cpu', logger=None):\n"
          "    class ExternalPPGroupCoordinator:\n"
          "        zeromq_comm = None\n"
          + m.PS_SEND_OLD + m.PS_RECV_OLD +
          "    pp_group = ExternalPPGroupCoordinator()\n"
          + m.PS_COMM_OLD + "    return pp_group\n")
    (root / "distributed" / "parallel_state.py").write_text(ps)
    return root


class Comm:
    def __init__(self, **kw):
        self.sent, self.inbox = [], []

    def send_tensor_dict(self, td, dst=None):
        self.sent.append(td)

    def recv_tensor_dict(self, src=None):
        return self.inbox.pop(0) if self.inbox else None


class TP:
    def __init__(self, size, rank, bus):
        self.world_size, self.rank_in_group, self.bus = size, rank, bus
        self.device = types.SimpleNamespace(type="cpu")

    def broadcast_tensor_dict(self, td=None, src=0):
        if self.rank_in_group == src:
            self.bus.append(td)
            return td
        return self.bus.pop(0)


def test_patch_applies_once_and_routes_through_rank0(tmp_path, monkeypatch):
    m = load()
    root = fork_tree(tmp_path, m)
    runner = str(root / "v1" / "worker" / "gpu_model_runner.py")
    assert m.main(["x", runner]) == 0
    first = (root / "distributed" / "parallel_state.py").read_text()
    assert m.main(["x", runner]) == 0          # idempotent
    assert (root / "distributed" / "parallel_state.py").read_text() == first

    members = []

    class FakeMember:
        def __init__(self, stage, rank, size, front_router_api=None):
            members.append((stage, rank, size))

        def start(self):
            return True

    monkeypatch.setitem(sys.modules, "connex.pp.group",
                        types.SimpleNamespace(TPShardMember=FakeMember))
    monkeypatch.setenv("CONNEX_PP_TP_SIZE", "1")   # restored after the test
    ns = {}
    exec(compile(first, "parallel_state", "exec"), ns)

    class Log:
        def info(self, *a):
            pass

    bus = []
    groups = []
    for rank in (0, 1):
        ns["_TP"] = TP(2, rank, bus)
        groups.append(ns["init_external_pp_group"](1, Comm, logger=Log()))
    g0, g1 = groups
    assert g0.zeromq_comm is not None and g1.zeromq_comm is None   # one endpoint per group
    assert members == [(1, 1, 2)]                  # rank 1 registers as shard 1 of 2
    assert os.environ["CONNEX_PP_TP_SIZE"] == "2"  # rank 0's endpoint: shard 0 of 2
    g0.send_tensor_dict({"h": 1})
    g1.send_tensor_dict({"h": 1})
    assert g0.zeromq_comm.sent == [{"h": 1}]
    g0.zeromq_comm.inbox.append({"step": 7})
    ns["_TP"] = TP(2, 0, bus)
    assert g0.recv_tensor_dict() == {"step": 7}
    ns["_TP"] = TP(2, 1, bus)
    assert g1.recv_tensor_dict() == {"step": 7}
    ns["_TP"] = TP(2, 0, bus)
    assert g0.recv_tensor_dict() is None                       # timeout: every rank sees None
    ns["_TP"] = TP(2, 1, bus)
    assert g1.recv_tensor_dict() is None

    cfg = (root / "config" / "parallel.py").read_text()
    ns2 = {}
    exec(compile(cfg, "parallel", "exec"), ns2)
    p = ns2["P"]()
    p.tensor_parallel_size = 2
    monkeypatch.setenv("CONNEX_PP_INFLIGHT", "1")
    assert p.check() == 1
    monkeypatch.setenv("CONNEX_PP_INFLIGHT", "4")
    with pytest.raises(ValueError):
        p.check()


def test_patch_reports_missing_anchor(tmp_path):
    m = load()
    root = fork_tree(tmp_path, m)
    (root / "distributed" / "parallel_state.py").write_text("nothing here\n")
    with pytest.raises(SystemExit):
        m.main(["x", str(root / "v1" / "worker" / "gpu_model_runner.py")])
