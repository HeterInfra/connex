# SPDX-License-Identifier: Apache-2.0
"""Group-level failure of tensor-parallel instances (CPU only): a TP group is
one endpoint at the front-router; any rank failing, missing its lease or
leaving takes the whole group out of the PD pools, and it rejoins under the
same identity once all ranks are back."""
import itertools
import time

import pytest
import zmq

from connex.front_router import FrontRouter, FrontRouterConfig
from connex.pd.group import GroupFailureTracker, PeerDeathReporter
from connex.pd.membership import PD_STAGE, PDMember, complete_groups, fetch_pools, pd_lid
from connex.protocol import PDRole

DEC = PD_STAGE[PDRole.DECODE]
_ports = itertools.count(25700, 10)


def route(stage, shard, replica, n=1, state=1):
    r = {"lid": {"model_name": "m", "stage_idx": stage, "shard_idx": shard,
                 "replica_idx": replica}, "addresses": [f"http://r{replica}:80"], "state": state}
    if n > 1:
        r["group_size"] = n
    return r


def test_complete_groups_lists_only_full_groups():
    rs = [route(DEC, 0, 0), route(DEC, 0, 1, 2), route(DEC, 1, 1, 2),
          route(DEC, 0, 2, 2), route(DEC, 1, 3, 2), route(DEC, 0, 4, 2), route(DEC, 1, 4, 2, state=2)]
    out = complete_groups(rs)
    assert [(r["lid"]["replica_idx"], r["lid"]["shard_idx"]) for r in out] == [(0, 0), (1, 0)]
    meta = route(DEC, 0, 5)
    meta["metadata"] = {"group_size": 2}
    assert complete_groups([meta]) == []


def test_tracker_reports_each_dead_rank_once_per_incarnation():
    t = GroupFailureTracker("m", [DEC])
    peer = ("m", DEC, 1, 3)
    assert t.on_event({"type": "failed", "peer": peer, "reason": "reset"}) is None   # never up
    t.on_event({"type": "up", "peer": peer})
    assert t.on_event({"type": "failed", "peer": peer, "reason": "timeout"}) is None  # suspect
    lid = t.on_event({"type": "failed", "peer": peer, "reason": "connection reset"})
    assert (lid.stage_idx, lid.shard_idx, lid.replica_idx) == (DEC, 1, 3)
    assert t.on_event({"type": "failed", "peer": peer, "reason": "reset"}) is None    # once
    t.on_event({"type": "joined", "peer": peer})                                      # rejoined
    assert t.on_event({"type": "failed", "peer": peer, "reason": "reset"}) is not None
    assert t.on_event({"type": "failed", "peer": ("other", DEC, 0, 0)}) is None
    assert t.on_event({"type": "failed", "peer": ("m", 3, 0, 0), "reason": "x"}) is None
    assert t.on_event({"type": "failed", "peer": None}) is None


@pytest.fixture
def fr(monkeypatch):
    monkeypatch.setenv("CONNEX_HEARTBEAT_TIMEOUT_S", "0.6")
    monkeypatch.setenv("CONNEX_HEALTH_CHECK_INTERVAL_S", "0.1")
    p = next(_ports)
    router = FrontRouter(FrontRouterConfig(api_address=f"tcp://127.0.0.1:{p}",
                                           event_pub_address=f"tcp://127.0.0.1:{p + 1}"))
    router.start()
    router.api, router.events = f"tcp://127.0.0.1:{p}", f"tcp://127.0.0.1:{p + 1}"
    yield router
    router.stop()


def member(fr, role, replica, shard=0, n=1):
    m = PDMember(fr.api, "m", role, replica, f"http://{role[0]}{replica}:80",
                 kv_addr="h:17100" if role == PDRole.PREFILL else None, events=fr.events,
                 heartbeat_interval_s=0.1, shard=shard, group_size=n)
    assert m.start()
    return m


def raw_register(fr, role, replica, shard, n):
    """A rank that registers once and never heartbeats (dies right away)."""
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(fr.api)
    lid = pd_lid("m", role, replica, shard)
    s.send_json({"method": "register", "params": {
        "lid": {"model_name": "m", "stage_idx": lid.stage_idx, "shard_idx": shard,
                "replica_idx": replica},
        "addresses": [f"http://{role[0]}{replica}:80"], "group_size": n}})
    assert s.recv_json()["success"]
    s.close(0)


def has(fr, role, replica, shard):
    return f"m:{PD_STAGE[role]}:{shard}:{replica}" in fr.route_manager._routes


def decode_urls(fr):
    return [x.url for x in fetch_pools(fr.api, "m")[PDRole.DECODE]]


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_group_listed_only_with_all_ranks(fr):
    ms = [member(fr, PDRole.DECODE, 0)]
    try:
        ms.append(member(fr, PDRole.DECODE, 1, shard=0, n=2))
        assert decode_urls(fr) == ["http://d0:80"]          # rank 1 not there yet
        ms.append(member(fr, PDRole.DECODE, 1, shard=1, n=2))
        assert decode_urls(fr) == ["http://d0:80", "http://d1:80"]
        info = fetch_pools(fr.api, "m")[PDRole.DECODE][1]
        assert info.group_size == 2
    finally:
        for m in ms:
            m.stop()


def test_report_failure_fails_whole_group_and_it_rejoins(fr):
    ms = [member(fr, PDRole.DECODE, 0), member(fr, PDRole.DECODE, 1, 0, 2),
          member(fr, PDRole.DECODE, 1, 1, 2)]
    try:
        ms[2].client.stop = lambda: None      # rank 1 "dies": keep its client from deregistering
        ms[2].client._running = False
        failed = fr.fail_group(pd_lid("m", PDRole.DECODE, 1, 1), "test")
        assert sorted(failed) == [f"m:{DEC}:0:1", f"m:{DEC}:1:1"]
        assert decode_urls(fr) == ["http://d0:80"]
        # rank 0 is alive: it re-registers on its next heartbeat, but the
        # group stays out until rank 1 is back
        assert wait_for(lambda: has(fr, PDRole.DECODE, 1, 0))
        assert decode_urls(fr) == ["http://d0:80"]
        ms.append(member(fr, PDRole.DECODE, 1, 1, 2))      # restarted rank, same identity
        assert decode_urls(fr) == ["http://d0:80", "http://d1:80"]
    finally:
        for m in ms:
            m.stop()


def test_missed_lease_of_one_rank_fails_the_group(fr):
    ms = [member(fr, PDRole.DECODE, 0), member(fr, PDRole.DECODE, 1, 0, 2)]
    try:
        raw_register(fr, PDRole.DECODE, 1, 1, 2)            # never heartbeats
        assert decode_urls(fr) == ["http://d0:80", "http://d1:80"]
        assert wait_for(lambda: decode_urls(fr) == ["http://d0:80"], timeout=3)
        assert fr._stats.get("group_failures", 0) >= 1
    finally:
        for m in ms:
            m.stop()


def test_one_rank_leaving_takes_the_group_out(fr):
    ms = [member(fr, PDRole.PREFILL, 0, 0, 2), member(fr, PDRole.PREFILL, 0, 1, 2)]
    try:
        assert [x.url for x in fetch_pools(fr.api, "m")[PDRole.PREFILL]] == ["http://p0:80"]
        ms[1].stop()
        assert not has(fr, PDRole.PREFILL, 0, 0)
        assert fetch_pools(fr.api, "m")[PDRole.PREFILL] == []
    finally:
        ms[0].stop()


class FakeEngine:
    def __init__(self, events):
        self.events = list(events)

    def wait_event(self, ms):
        if self.events:
            return self.events.pop(0)
        time.sleep(ms / 1000)
        return None


def test_transport_peer_death_reports_the_group(fr):
    ms = [member(fr, PDRole.PREFILL, 0, 0, 2), member(fr, PDRole.PREFILL, 0, 1, 2),
          member(fr, PDRole.PREFILL, 1)]
    peer = ("m", PD_STAGE[PDRole.PREFILL], 1, 0)
    rep = PeerDeathReporter(FakeEngine([{"type": "up", "peer": peer},
                                        {"type": "failed", "peer": peer,
                                         "reason": "Connection reset by remote peer"}]),
                            fr.api, "m")
    try:
        ms[1].client._running = False         # the dead rank sends nothing any more
        rep.start()
        assert wait_for(lambda: rep.reports, timeout=3)
        _, lid, res = rep.reports[0]
        assert (lid.shard_idx, lid.replica_idx) == (1, 0) and len(res) == 2
        assert [x.url for x in fetch_pools(fr.api, "m")[PDRole.PREFILL]] == ["http://p1:80"]
    finally:
        rep.stop()
        for m in ms:
            m.stop()


def test_pipeline_peers_see_only_complete_tp_groups():
    from connex.epc_client import RouteEntry, serving_routes

    def r(stage, shard, replica, n):
        d = {"lid": {"model_name": "m", "stage_idx": stage, "shard_idx": shard,
                     "replica_idx": replica}, "addresses": [f"a{stage}{shard}{replica}"],
             "capabilities": [], "epoch": 0}
        if n > 1:
            d["metadata"] = {"group_size": n}
        return RouteEntry.from_dict(d)

    routes = [r(0, 0, 0, 1), r(1, 0, 0, 2), r(1, 1, 0, 2), r(1, 0, 1, 2)]
    keys = [(x.lid.stage_idx, x.lid.shard_idx, x.lid.replica_idx) for x in serving_routes(routes)]
    assert keys == [(0, 0, 0), (1, 0, 0)]      # shard 1 hidden; replica 1 lacks rank 1


def test_tp_shard_member_fails_pipeline_stage_group(fr):
    """A pipeline stage of TP 2: rank 0's endpoint (shard 0) and a liveness
    shard for rank 1. Rank 1 going silent deregisters the stage's endpoint
    for every peer."""
    from connex.epc_client import EPCV2Client, EPCV2ClientConfig
    from connex.pp.group import TPShardMember

    ep = EPCV2Client(EPCV2ClientConfig(front_router_api=fr.api, front_router_events=fr.events,
                                       model_name="pp", stage_idx=1, replica_idx=0,
                                       group_size=2, local_addresses=["tcp://s1:1"],
                                       heartbeat_interval_s=0.1, ucx_enabled=False))
    sh = TPShardMember(1, 1, 2, replica_idx=0, model="pp", front_router_api=fr.api,
                       events=fr.events, heartbeat_interval_s=0.1)
    peer = EPCV2Client(EPCV2ClientConfig(front_router_api=fr.api, front_router_events=fr.events,
                                         model_name="pp", stage_idx=0, replica_idx=0,
                                         local_addresses=["tcp://s0:1"],
                                         heartbeat_interval_s=0.1, ucx_enabled=False))
    try:
        assert ep.start() and peer.start()
        assert [x.lid.stage_idx for x in peer.get_pipeline_peers()] == [0]   # stage 1 incomplete
        assert sh.start()
        assert wait_for(lambda: sorted(x.lid.stage_idx for x in peer.get_pipeline_peers())
                        == [0, 1])
        sh.client._running = False          # rank 1 dies: no more heartbeats
        assert wait_for(lambda: [x.lid.stage_idx for x in peer.get_pipeline_peers()] == [0],
                        timeout=4)
        assert fr._stats.get("group_failures", 0) >= 1
    finally:
        for c in (ep, peer):
            c.stop()
