# SPDX-License-Identifier: Apache-2.0
"""Regression tests for EP / RL weight sync / agent session fixes."""
import os
import pickle

import pytest
import torch

from connex.agents.session import SessionProxy
from connex.ep import EPController, PlacementTable
from connex.ep.mover import _load_ipc
from connex.rl.registry import PUBLISHER, WeightRegistry
from connex.rl.sync import WeightPublisher, WeightSubscriber
from connex.rl.transport import LocalTransport


def test_dead_reader_load_is_released_on_leave():
    reg = WeightRegistry(fanout=2)
    reg.publish(1, {}, "u", 0)
    reg.join("r")
    assert reg.assign(1, "r")["member"] == PUBLISHER
    reg.leave("r", "expired")                      # died mid-read, never released
    assert reg.wait_idle(1, timeout=0.1)["ok"]
    # a late release of the same assignment does not go negative / double count
    reg.join("r2")
    reg.assign(1, "r2")
    reg.release(1, PUBLISHER, member="r")
    assert reg.status()["holders"]["1"][PUBLISHER] == 1


def test_overwrite_during_read_is_reported_stale():
    reg = WeightRegistry(fanout=2)
    reg.publish(1, {}, "u", 0)
    reg.join("r")
    reg.assign(1, "r")
    assert not reg.wait_idle(1, timeout=0.05)["ok"]  # publisher overwrites anyway
    assert reg.release(1, PUBLISHER, member="r").get("stale")


def test_subscriber_rereads_when_source_was_overwritten():
    reg = WeightRegistry(fanout=2)
    pub = WeightPublisher(reg, LocalTransport("rv-trainer"))
    named = [("w", torch.arange(64, dtype=torch.float32))]
    pub.publish(named)
    sub = WeightSubscriber(reg, LocalTransport("rv-sub"), "s", heartbeat_s=0)
    orig, calls = reg.release, []

    def release(version, source, member=None):
        out = orig(version, source, member)
        calls.append(1)
        return dict(out, stale=True) if len(calls) == 1 else out
    reg.release = release
    sub.pull()
    assert len(calls) == 2
    assert torch.equal(dict(sub.tensors())["w"], named[0][1])


def test_failed_pull_does_not_expose_old_manifest():
    reg = WeightRegistry(fanout=2)
    pub = WeightPublisher(reg, LocalTransport("fp-trainer"))
    pub.publish([("w", torch.ones(8))])
    sub = WeightSubscriber(reg, LocalTransport("fp-sub"), "s", heartbeat_s=0)
    sub.sync(lambda it: list(it))
    pub.publish([("w", torch.zeros(8))])
    reg.assign = lambda **kw: None                 # no source: pull fails midway
    with pytest.raises(RuntimeError):
        sub.pull()
    with pytest.raises(Exception):
        list(sub.tensors())


def test_ipc_handles_reject_arbitrary_globals():
    class Evil:
        def __reduce__(self):
            return (os.system, ("true",))
    with pytest.raises(pickle.UnpicklingError):
        _load_ipc(pickle.dumps([[Evil()]]))


def test_engine_exception_does_not_leak_staging():
    t = PlacementTable(4, 2, 4, [[0, 1, 2, 3, 0, 1, 2, 3]])

    class Eng:
        def apply(self, view):
            return True

        def describe(self):
            return {}

        def move(self, *a):
            raise RuntimeError("rpc down")
    c = EPController(t, Eng())
    with pytest.raises(RuntimeError):
        c.migrate(3, 1)
    assert t.staging == set() and t.slot_map[0][1] == -1


def test_idle_sessions_expire():
    p = SessionProxy.__new__(SessionProxy)
    p.sessions, p.ttl, p._next_sweep, p.stats = {}, 10.0, 0.0, {"sessions": 0}
    s = p.get_session("a")
    s.last_used -= 100
    p._next_sweep = 0.0
    p.get_session("b")
    assert set(p.sessions) == {"b"}
