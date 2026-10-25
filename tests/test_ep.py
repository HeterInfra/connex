# SPDX-License-Identifier: Apache-2.0
"""connex.ep: placement epochs, repair planning, one-sided expert moves, and
a toy MoE layer that keeps producing the reference output while ranks fail,
drain and join (CPU)."""
import pytest
import torch

from connex.ep import (DOWN, DRAINING, UP, EPController, ExpertStore, LocalEngine, Move,
                       PlacementError, PlacementTable, logical_to_physical)
from connex.rl.transport import LocalTransport

E, H = 8, 4


def test_linear_matches_vllm_default():
    t = PlacementTable.linear(64, 2, 64)
    v = t.view()
    assert v["slot_map"][0][:64] == list(range(64)) and v["slot_map"][0][64:] == list(range(64))
    l2p, cnt = logical_to_physical(v, 0)
    assert cnt == [2] * 64 and l2p[5] == [5, 69]


def test_spread_puts_replicas_on_distinct_ranks():
    t = PlacementTable.spread(8, 4, 4)
    for e in range(8):
        ranks = [t.rank_of(p) for p in t.replicas(0, e)]
        assert len(ranks) == 2 and len(set(ranks)) == 2


def test_epoch_only_on_commit_and_masks():
    t = PlacementTable.linear(4, 2, 4)          # every expert on both ranks
    assert t.commit() == 0
    t.mark_down(1)
    assert t.epoch == 0 and t.commit() == 1
    assert t.routable_mask(0) == [True] * 4 + [False] * 4
    assert t.uncovered() == []


def test_draining_rank_serves_only_its_unique_experts():
    t = PlacementTable(4, 2, 4, [[0, 1, 2, 0, 3, 1, 2, -1]])
    t.mark_draining(1)
    # expert 3 lives only on rank 1: still routable; 1 and 2 served by rank 0
    assert t.routable_mask(0) == [True] * 4 + [True, False, False, False]
    moves = t.plan_repair()
    assert [(m.expert, m.src, m.evicts) for m in moves] == [(3, 4, 0)] and moves[0].dst in (0, 3)


def test_repair_is_best_effort_when_capacity_runs_out():
    t = PlacementTable(4, 2, 3, [[0, 1, 2, 3, 0, 1]])
    t.mark_down(1)
    assert t.plan_repair() == []                # rank 0 has no evictable slot
    assert t.uncovered() == [(0, 3)]


def test_plan_repair_never_strands_and_uses_checkpoint_when_no_copy():
    t = PlacementTable(4, 2, 4, [[0, 1, 2, 0, 3, 2, 1, -1]])
    t.mark_down(1)
    moves = t.plan_repair()
    assert [(m.expert, m.src) for m in moves] == [(3, None)]   # no live copy
    t.begin_moves(moves)
    t.finish_moves(moves)
    t.commit()
    assert t.uncovered() == []


def test_staging_slot_not_routable_until_finished():
    t = PlacementTable.linear(4, 2, 4)
    m = Move(0, 2, 7, 2, 3)
    t.begin_moves([m]); t.commit()
    assert not t.routable_mask(0)[7]
    t.finish_moves([m]); t.commit()
    assert t.routable_mask(0)[7] and t.slot_map[0][7] == 2


def test_view_roundtrip_and_max_replicas():
    t = PlacementTable.linear(4, 2, 4, num_layers=2)
    t2 = PlacementTable.from_dict(t.to_dict())
    assert t2.view() == t.view()
    with pytest.raises(PlacementError):
        logical_to_physical(t.view(), 0, max_replicas=1)


# -- toy EP group -------------------------------------------------------------

def weights():
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, H, H, generator=g)


def group(slot_map, R, S):
    W = weights()
    stores = []
    for r in range(R):
        w = torch.zeros(S, H, H)
        for i in range(S):
            e = slot_map[r * S + i]
            if e >= 0:
                w[i] = W[e]
        stores.append(ExpertStore(r, [[w]], LocalTransport(f"ep-r{r}")))
    eng = LocalEngine(stores, E)
    eng.loader = lambda layer, e: [W[e]]
    t = PlacementTable(E, R, S, [slot_map])
    ctl = EPController(t, eng)
    ctl.push()
    return W, eng, ctl


def reference(W, x, topk):
    return [sum(row @ W[e] for e in ids) for row, ids in zip(x, topk)]


def check(W, eng, n=64):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, H, generator=g)
    topk = torch.randint(0, E, (n, 2), generator=g).tolist()
    for a, b in zip(eng.forward(x, topk), reference(W, x, topk)):
        assert torch.allclose(a, b)


def test_fail_rank_reroutes_and_restores_coverage():
    # 3 ranks x 4 slots, experts 0..7 once + 4 redundant: rank 2 holds 0..3 copies
    R, S = 3, 4
    W, eng, ctl = group(list(range(8)) + [0, 1, 2, 3], R, S)
    check(W, eng)
    eng.kill(1)                                  # experts 4..7 only lived there
    with pytest.raises(RuntimeError):
        check(W, eng)
    rep = ctl.fail_rank(1)
    assert rep["uncovered"] == 0 and rep["moves"] == 4
    assert eng.applied[-2:] == [rep["epoch_cutover"], rep["epoch"]]
    check(W, eng)                                 # loaded from the checkpoint


def test_drain_copies_unique_experts_from_the_leaving_rank():
    R, S = 3, 4
    W, eng, ctl = group(list(range(8)) + [0, 1, 2, 3], R, S)
    rep = ctl.drain_rank(1)
    assert rep["moves"] == 4 and rep["uncovered"] == 0
    assert eng.stores[0].stats["checkpoint_loads"] == 0
    assert ctl.table.rank_load()[1] == 0          # no token goes to rank 1
    ctl.left(1); eng.kill(1)
    check(W, eng)


def test_source_dies_mid_move_replans():
    R, S = 3, 4
    W, eng, ctl = group(list(range(8)) + [0, 1, 2, 3], R, S)
    ctl.refresh_peers()
    ctl.table.mark_draining(1)                    # 4..7 must be copied from rank 1
    eng.kill(1)                                   # ... which dies before the copy
    rep = ctl.reconcile("drain then die", prefer_src_rank=1)
    assert rep["failed_moves"] == 4 and rep["moves"] == 4 and rep["rounds"] == 2
    assert ctl.table.rank_state[1] == DOWN and ctl.table.uncovered() == []
    check(W, eng)


def test_join_rank_adds_replicas_and_migrate():
    R, S = 2, 8
    W, eng, ctl = group(list(range(8)) + [-1] * 8, R, S)
    ctl.min_replicas = 2
    rep = ctl.reconcile("replicate")
    assert rep["moves"] == 8 and all(len(ctl.table.routable_replicas(0, e)) == 2 for e in range(E))
    check(W, eng)
    rep = ctl.migrate(3, 0)                       # slot 0 held expert 0 (replicated)
    assert ctl.table.slot_map[0][0] == 3
    assert eng.stores[0].stats["local_copies"] >= 1
    check(W, eng)


def test_stale_view_ignored():
    W, eng, ctl = group(list(range(8)), 2, 4)
    v = ctl.table.view()
    assert eng.apply(v) is False


def test_group_supervisor_removes_rebuilds_and_rejoins(tmp_path):
    import time as _t
    from connex.ep.groups import EPGroup, EPGroupSupervisor
    alive = {"http://a": True, "http://b": True}
    restarted = []

    def restart():
        restarted.append(1)
        alive["http://a"] = True                 # comes back healthy
        return None

    f = str(tmp_path / "eps")
    sup = EPGroupSupervisor([EPGroup("a", "http://a", restart), EPGroup("b", "http://b")], f,
                            probe_s=0.01, fail_after=2, probe=lambda u: alive[u])
    assert open(f).read().split() == ["http://a", "http://b"]
    alive["http://a"] = False
    sup.check_once(); sup.check_once()
    assert open(f).read().split() == ["http://b"] or restarted
    for _ in range(200):
        if any(e["kind"] == "rejoin" for e in sup.events):
            break
        _t.sleep(0.01)
    assert [e["kind"] for e in sup.events] == ["down", "rebuild-start", "rejoin"]
    assert open(f).read().split() == ["http://a", "http://b"]


def test_active_replicas_standby_flips_on_failure_without_copies():
    t = PlacementTable.linear(64, 2, 64)
    t.active_replicas = 1
    load = t.rank_load()
    assert load == {0: 32, 1: 32}                 # one routed copy per expert, balanced
    assert all(len(t.routable_replicas(0, e)) == 1 for e in range(64))
    t.mark_down(1)
    assert t.plan_repair() == []                  # standby copies cover rank 1's experts
    assert t.rank_load() == {0: 64, 1: 0} and t.uncovered() == []
    t.mark_up(1)
    assert t.rank_load() == {0: 32, 1: 32}
    assert PlacementTable.from_dict(t.to_dict()).view() == t.view()


def test_group_supervisor_detects_dead_rank_process(tmp_path):
    import subprocess
    from connex.ep.groups import EPGroup, EPGroupSupervisor, pid_alive
    p = subprocess.Popen(["sleep", "30"])
    sup = EPGroupSupervisor([EPGroup("a", "http://a", rank_pids=lambda: [p.pid]),
                             EPGroup("b", "http://b")], str(tmp_path / "eps"),
                            probe=lambda u: True)         # /health still answers
    sup.check_once()
    assert open(tmp_path / "eps").read().split() == ["http://a", "http://b"]
    p.kill()                                               # zombie until reaped
    import time as _t
    for _ in range(100):
        if not pid_alive(p.pid):
            break
        _t.sleep(0.01)
    assert not pid_alive(p.pid)
    sup.check_once()
    assert [e["kind"] for e in sup.events] == ["down"]
    assert sup.events[0]["why"] == "rank process exited"
    assert open(tmp_path / "eps").read().split() == ["http://b"]
    p.wait()


def test_migrate_with_standby_replicas():
    W, eng, ctl = group(list(range(8)) * 2, 2, 8)
    ctl.table.active_replicas = 1
    ctl.push()
    victim = ctl.table.slot_map[0][8]
    rep = ctl.migrate(3, 8)                        # slot 8's expert keeps its copy on rank 0
    assert rep["moves"] == 1 and ctl.table.slot_map[0][8] == 3
    assert ctl.table.replicas(0, victim) == [victim]
    check(W, eng)
    with pytest.raises(PlacementError):
        ctl.migrate(5, victim)                     # that is now the last copy of victim


def test_stale_controller_is_refused_not_silently_ignored():
    W, eng, ctl = group(list(range(8)) * 2, 2, 8)
    ctl.drain_rank(1)                             # engine now at a newer epoch
    fresh = EPController(PlacementTable(E, 2, 8, [list(range(8)) * 2]), eng)
    with pytest.raises(PlacementError):
        fresh.push()
    fresh.table.epoch = eng.view["epoch"]          # adopt the engine's epoch
    fresh.drain_rank(0)
    assert eng.view["epoch"] == fresh.table.epoch


def test_rank_death_watcher_fails_the_group_of_a_dead_remote_rank(tmp_path):
    from connex.ep.groups import EPGroup, EPGroupSupervisor, RankDeathWatcher

    class Engine:
        def __init__(self):
            self.connected = []

        def connect(self, url):
            self.connected.append(url)

    sup = EPGroupSupervisor([EPGroup("a", "http://h1:1"), EPGroup("b", "http://h2:2")],
                            str(tmp_path / "eps"))
    w = RankDeathWatcher(sup, engine=Engine())
    w.watch("a", ["10.0.0.1:5#connex-ep|0|0|0", "10.0.0.2:6#connex-ep|0|1|0"])
    w.watch("b", ["10.0.0.1:7#connex-ep|0|0|1", "10.0.0.2:8#connex-ep|0|1|1"])
    assert len(w.engine.connected) == 4
    up = lambda r, g: {"type": "up", "peer": ["connex-ep", 0, r, g]}        # noqa: E731
    dead = lambda r, g, why="reset": {"type": "failed", "peer": ["connex-ep", 0, r, g],  # noqa: E731
                                      "reason": why}
    for r in (0, 1):
        for g in (0, 1):
            w.handle(up(r, g))
    assert w.handle(dead(1, 0, "timeout")) is None           # suspicion only
    assert sup.groups["a"].state == "up"
    assert w.handle(dead(1, 0)) == "a"                        # rank 1 of group a (remote host)
    assert sup.groups["a"].state == "down" and sup.groups["b"].state == "up"
    assert open(tmp_path / "eps").read().split() == ["http://h2:2"]
    assert w.handle(dead(0, 0)) is None                       # group already out
