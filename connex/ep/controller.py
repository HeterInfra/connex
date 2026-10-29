# SPDX-License-Identifier: Apache-2.0
"""EP control plane: owns the placement table and drives engines through
epochs.

Every reconfiguration (a rank fails, drains, joins; an expert migrates)
runs the same two-epoch protocol::

    epoch k+1  route away from down / draining ranks, and away from the
               destination slots of the repair moves  -> engines apply
    copy       destination ranks pull expert weights (one-sided, from a live
               replica; the checkpoint when none is left)
    epoch k+2  destination slots routable with their new expert -> apply

Engines apply a view only at a step boundary, so a step never mixes two
placements and in-flight requests keep running on the surviving replicas.
A move whose source dies mid-copy is re-planned from another source.

An *engine* is anything with the small duck type below; one handle may stand
for all ranks of one EP group (vLLM: ``collective_rpc`` to every worker)::

    apply(view) -> any                     install the view at a step boundary
    describe() -> {rank: descriptor}       ExpertStore.describe() per rank
    move(moves, peers, slots_per_rank) -> [(Move, ok, err)]
"""
from __future__ import annotations

import logging
import threading
import time

from .placement import DOWN, DRAINING, UP, Move, PlacementError, PlacementTable

log = logging.getLogger("connex.ep")


class EPController:
    def __init__(self, table: PlacementTable, engine, min_replicas=1, max_rounds=4):
        self.table = table
        self.engine = engine
        self.min_replicas = min_replicas
        self.max_rounds = max_rounds
        self.lock = threading.RLock()
        self.peers = {}
        self.history = []          # reports of every reconfiguration

    def refresh_peers(self):
        self.peers = {int(k): v for k, v in (self.engine.describe() or {}).items()}
        return self.peers

    def push(self, force=False):
        """Commit pending changes and install the view; returns seconds.
        Nothing is sent when the engines already run the current epoch."""
        dirty = self.table._dirty
        self.table.commit()
        if not dirty and not force and getattr(self, "_pushed", None) == self.table.epoch:
            return 0.0
        self._pushed = self.table.epoch
        t0 = time.perf_counter()
        if self.engine.apply(self.table.view()) is False:
            raise PlacementError(f"engine rejected epoch {self.table.epoch} as stale: it runs a "
                                 "newer one (another controller?)")
        return time.perf_counter() - t0

    # -- events ---------------------------------------------------------------
    def fail_rank(self, rank):
        """A rank died: stop routing to it, then restore coverage."""
        with self.lock:
            self.table.mark_down(rank)
            self.peers.pop(rank, None)
            return self.reconcile(f"fail rank {rank}")

    def drain_rank(self, rank):
        """Planned leave: copy what only this rank holds elsewhere, then stop
        routing to it. Afterwards it can be stopped (call ``left``)."""
        with self.lock:
            self.table.mark_draining(rank)
            return self.reconcile(f"drain rank {rank}", prefer_src_rank=rank)

    def left(self, rank):
        with self.lock:
            self.table.mark_down(rank)
            self.peers.pop(rank, None)
            t = self.push()
            return {"event": f"left rank {rank}", "epoch": self.table.epoch, "apply_s": t}

    def join_rank(self, rank, slot_map_rows=None):
        """A rank (re)joins with ``slot_map_rows`` loaded (None: keep the
        table's rows for it, i.e. it came back with the same experts)."""
        with self.lock:
            self.table.mark_up(rank, slot_map_rows)
            self.refresh_peers()
            return self.reconcile(f"join rank {rank}")

    def migrate(self, expert, dst_slot, layers=None):
        """Copy ``expert`` into ``dst_slot`` (all layers by default)."""
        with self.lock:
            layers = range(self.table.num_layers) if layers is None else layers
            moves = []
            for l in layers:
                old = self.table.slot_map[l][dst_slot]
                if old == expert:
                    continue
                # standby replicas count: they become routable when needed
                if old >= 0 and len(self.table.replicas(l, old, live_only=True)) <= 1:
                    raise PlacementError(f"slot {dst_slot} holds the last replica of {old}")
                src = self.table._pick_src(l, expert, dst_slot)
                moves.append(Move(l, expert, dst_slot, src, old))
            return self._run_moves(f"migrate e{expert}->slot{dst_slot}", moves)

    # -- core ---------------------------------------------------------------
    def reconcile(self, event="reconcile", prefer_src_rank=None):
        moves = self.table.plan_repair(self.min_replicas, prefer_src_rank=prefer_src_rank)
        return self._run_moves(event, moves)

    def _run_moves(self, event, moves):
        rep = {"event": event, "epoch_before": self.table.epoch, "moves": 0, "failed_moves": 0,
               "rounds": 0, "cutover_s": 0.0, "copy_s": 0.0, "commit_s": 0.0, "bytes": 0}
        t_start = time.perf_counter()
        self.table.begin_moves(moves)
        rep["cutover_s"] = self.push()            # epoch k+1: route around
        rep["epoch_cutover"] = self.table.epoch
        if not self.peers:
            self.refresh_peers()
        pending = list(moves)
        try:
            while pending and rep["rounds"] < self.max_rounds:
                rep["rounds"] += 1
                t0 = time.perf_counter()
                res = self.engine.move(pending, self.peers, self.table.slots_per_rank)
                rep["copy_s"] += time.perf_counter() - t0
                done = [m for m, ok, _ in res if ok]
                failed = [(m, err) for m, ok, err in res if not ok]
                # results for moves whose destination rank did not answer count as failed
                seen = {(m.layer, m.dst) for m, _, _ in res}
                failed += [(m, "no result") for m in pending if (m.layer, m.dst) not in seen]
                self.table.finish_moves(done, ok=True)
                rep["moves"] += len(done)
                rep["bytes"] += sum(self._move_bytes(m) for m in done)
                pending = []
                for m, err in failed:
                    log.warning("move %s failed: %s", m, err)
                    rep["failed_moves"] += 1
                    src_rank = None if m.src is None else self.table.rank_of(m.src)
                    if src_rank is not None and src_rank in self.peers and "unreachable" in str(err):
                        self.table.mark_down(src_rank)
                    if not self.table.live(self.table.rank_of(m.dst)):
                        self.table.finish_moves([m], ok=False)
                        continue
                    src = self.table._pick_src(m.layer, m.expert, m.dst, None)
                    if src == m.src:
                        src = None                    # fall back to the checkpoint
                    pending.append(Move(m.layer, m.expert, m.dst, src, m.evicts))
        except BaseException:
            # the engine call failed outright: the staged slots may be half
            # written, so release them as empty instead of leaking them as
            # staging forever (unroutable and never repaired)
            self.table.finish_moves(pending, ok=False)
            try:
                self.push()
            except Exception:  # noqa: BLE001
                log.exception("ep %s: push after a failed move round failed", event)
            raise
        if pending:
            self.table.finish_moves(pending, ok=False)
        rep["commit_s"] = self.push()             # epoch k+2: new slots live
        rep["epoch"] = self.table.epoch
        rep["uncovered"] = len(self.table.uncovered())
        rep["total_s"] = time.perf_counter() - t_start
        self.history.append(rep)
        log.info("ep %s: %s", event, rep)
        return rep

    def _move_bytes(self, m):
        r = self.table.rank_of(m.dst)
        d = self.peers.get(r)
        if not d:
            return 0
        return sum(rb for _, rb in d["layers"][m.layer])


class LocalEngine:
    """In-process EP group for tests and CPU demos: one ExpertStore per rank;
    ``forward`` runs a toy MoE layer with the installed routing."""

    def __init__(self, stores, num_experts):
        self.stores = {s.rank: s for s in stores}
        self.num_experts = num_experts
        self.view = None
        self.applied = []
        self.dead = set()

    def apply(self, view):
        if self.view is not None and view["epoch"] <= self.view["epoch"]:
            return False                         # stale or duplicate
        self.view = view
        self.applied.append(view["epoch"])
        return True

    def describe(self):
        return {r: s.describe() for r, s in self.stores.items() if r not in self.dead}

    def move(self, moves, peers, slots_per_rank, loader=None):
        out = []
        for r, s in self.stores.items():
            if r in self.dead:
                continue
            out += s.apply_moves(moves, peers, slots_per_rank, loader=loader or self.loader)
        return out

    loader = None

    def kill(self, rank):
        self.dead.add(rank)
        t = self.stores[rank].t
        if t is not None and hasattr(t, "kill"):
            t.kill()

    def expert_weight(self, slot, layer=0, param=0):
        S = self.view["slots_per_rank"]
        return self.stores[slot // S].layers[layer][param][slot % S]

    def forward(self, x, topk, layer=0):
        """y[i] = sum over chosen experts e of x[i] @ W[e] via a routable
        replica (token-hashed like vLLM). Raises if a token hits a dead or
        unroutable slot."""
        from .placement import logical_to_physical
        l2p, cnt = logical_to_physical(self.view, layer)
        S = self.view["slots_per_rank"]
        y = []
        for i, (row, ids) in enumerate(zip(x, topk)):
            acc = 0
            for e in ids:
                if cnt[e] == 0:
                    raise RuntimeError(f"expert {e} has no routable replica")
                p = l2p[e][(i * 2654435769 & 0xFFFFFFFF) % cnt[e]]
                if p // S in self.dead:
                    raise RuntimeError(f"token routed to dead rank {p // S}")
                acc = acc + row @ self.expert_weight(p, layer)
            y.append(acc)
        return y


__all__ = ["EPController", "LocalEngine", "UP", "DOWN", "DRAINING"]
