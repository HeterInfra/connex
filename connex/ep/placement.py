# SPDX-License-Identifier: Apache-2.0
"""Versioned expert placement table (engine neutral, pure Python).

An EP deployment has ``num_ranks`` EP ranks, each with ``slots_per_rank``
physical expert slots; physical slot ``p`` lives on rank
``p // slots_per_rank``. Per MoE layer, ``slot_map[layer][p]`` is the logical
expert whose weights the slot holds (``-1``: empty). A logical expert with
several slots has replicas.

Ranks are mobile endpoints with a state:

* ``up``: serves tokens for the experts it holds;
* ``draining``: about to leave; its slots stay routable only for experts that
  have no replica on an ``up`` rank, so a drain never strands a token;
* ``down``: gone (failed or left); never routed to, never a copy source.

A slot is also unroutable while it is the destination of a weight move
(``staging``): its old contents are being overwritten.

Every change becomes visible only through :meth:`commit`, which bumps the
``epoch``. Engines apply a :meth:`view` only at a step boundary and only if
its epoch is newer than the one they run (like ``connex.pp`` chain epochs),
so all ranks switch placement between the same two steps and a stale or
duplicated view is ignored.

Repair (:meth:`plan_repair`) restores coverage after a rank goes down or
drains: for each (layer, expert) without a routable replica, or below
``min_replicas``, it picks a destination slot on an ``up`` rank (an empty
slot, else a slot whose expert keeps another routable replica) and a source
(a live slot holding the expert, else ``None``: load from the checkpoint).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

UP, DRAINING, DOWN = "up", "draining", "down"


@dataclass(frozen=True)
class Move:
    """Copy ``expert``'s weights of ``layer`` into physical slot ``dst``.

    ``src`` is a physical slot holding the expert on a live rank, or ``None``
    when no live copy exists (load from the checkpoint). ``evicts`` is the
    expert the destination held before (``-1``: empty)."""
    layer: int
    expert: int
    dst: int
    src: Optional[int]
    evicts: int = -1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class PlacementError(RuntimeError):
    pass


class PlacementTable:
    def __init__(self, num_experts: int, num_ranks: int, slots_per_rank: int,
                 slot_map: list, num_layers: Optional[int] = None, epoch: int = 0,
                 rank_state: Optional[dict] = None, active_replicas: Optional[int] = None):
        self.num_experts = int(num_experts)
        self.num_ranks = int(num_ranks)
        self.slots_per_rank = int(slots_per_rank)
        if slot_map and not isinstance(slot_map[0], (list, tuple)):
            slot_map = [list(slot_map) for _ in range(num_layers or 1)]
        self.slot_map = [list(map(int, row)) for row in slot_map]
        self.num_layers = len(self.slot_map)
        for row in self.slot_map:
            if len(row) != self.num_slots:
                raise PlacementError(f"slot_map row has {len(row)} slots, want {self.num_slots}")
            if any(e < -1 or e >= self.num_experts for e in row):
                raise PlacementError("slot_map holds an out-of-range expert id")
        self.rank_state = {r: UP for r in range(self.num_ranks)}
        if rank_state:
            self.rank_state.update({int(k): v for k, v in rank_state.items()})
        self.staging = set()          # (layer, slot) being overwritten
        # at most this many replicas of an expert are routed; the others are
        # warm standby (None: route to every replica)
        self.active_replicas = active_replicas
        self.epoch = int(epoch)
        self._dirty = False

    # -- construction -------------------------------------------------------
    @classmethod
    def linear(cls, num_experts, num_ranks, slots_per_rank, num_layers=1):
        """vLLM's initial EPLB layout: slot p holds expert p, redundant slots
        (p >= num_experts) hold p mod num_experts. Matches what the engine
        loads at start-up, so no weights move."""
        P = num_ranks * slots_per_rank
        if P < num_experts:
            raise PlacementError(f"{P} slots cannot hold {num_experts} experts")
        row = [p if p < num_experts else (p - num_experts) % num_experts for p in range(P)]
        return cls(num_experts, num_ranks, slots_per_rank, [row] * num_layers)

    @classmethod
    def spread(cls, num_experts, num_ranks, slots_per_rank, num_layers=1):
        """Replicas of one expert on distinct ranks where capacity allows:
        copy c of expert e goes to rank (e + c) mod R (round-robin fill)."""
        R, S, E = num_ranks, slots_per_rank, num_experts
        if R * S < E:
            raise PlacementError(f"{R * S} slots cannot hold {E} experts")
        free = {r: list(range(r * S, (r + 1) * S)) for r in range(R)}
        row = [-1] * (R * S)
        c = 0
        while any(free.values()):
            placed = False
            for e in range(E):
                for k in range(R):
                    r = (e + c + k) % R
                    if free[r] and e not in [row[p] for p in range(r * S, (r + 1) * S)]:
                        row[free[r].pop(0)] = e
                        placed = True
                        break
            c += 1
            if not placed:
                break
        return cls(E, R, S, [row] * num_layers)

    # -- geometry -----------------------------------------------------------
    @property
    def num_slots(self):
        return self.num_ranks * self.slots_per_rank

    def rank_of(self, slot):
        return slot // self.slots_per_rank

    def local_index(self, slot):
        return slot % self.slots_per_rank

    def rank_slots(self, rank):
        return range(rank * self.slots_per_rank, (rank + 1) * self.slots_per_rank)

    # -- queries ------------------------------------------------------------
    def live(self, rank):
        return self.rank_state.get(rank) in (UP, DRAINING)

    def replicas(self, layer, expert, live_only=False):
        return [p for p, e in enumerate(self.slot_map[layer]) if e == expert
                and (layer, p) not in self.staging and (not live_only or self.live(self.rank_of(p)))]

    def routable_mask(self, layer):
        """Per slot: True if tokens may be routed to it in this layer."""
        row = self.slot_map[layer]
        on_up = set()
        for p, e in enumerate(row):
            if e >= 0 and (layer, p) not in self.staging and self.rank_state[self.rank_of(p)] == UP:
                on_up.add(e)
        mask = []
        for p, e in enumerate(row):
            st = self.rank_state[self.rank_of(p)]
            ok = e >= 0 and (layer, p) not in self.staging and (
                st == UP or (st == DRAINING and e not in on_up))
            mask.append(ok)
        if self.active_replicas:
            mask = self._limit_active(row, mask, self.active_replicas)
        return mask

    def _limit_active(self, row, mask, k):
        """Keep at most k routed replicas per expert, spreading the routed
        copies evenly over the ranks; the rest stay as warm standby."""
        cands = {}
        for p, ok in enumerate(mask):
            if ok:
                cands.setdefault(row[p], []).append(p)
        load = {r: 0 for r in range(self.num_ranks)}
        keep = set()
        for e in sorted(cands, key=lambda e: (len(cands[e]), e)):
            ps = list(cands[e])
            for _ in range(min(k, len(ps))):
                p = min(ps, key=lambda q: (load[self.rank_of(q)], (q + e) % len(row)))
                ps.remove(p)
                keep.add(p)
                load[self.rank_of(p)] += 1
        return [p in keep for p in range(len(row))]

    def routable_replicas(self, layer, expert):
        m = self.routable_mask(layer)
        return [p for p, e in enumerate(self.slot_map[layer]) if e == expert and m[p]]

    def uncovered(self):
        """(layer, expert) pairs with no routable replica."""
        out = []
        for layer in range(self.num_layers):
            m = self.routable_mask(layer)
            have = {e for p, e in enumerate(self.slot_map[layer]) if m[p]}
            out.extend((layer, e) for e in range(self.num_experts) if e not in have)
        return out

    def rank_load(self, layer=0):
        """Number of routable slots per rank (a proxy for token share)."""
        m = self.routable_mask(layer)
        load = {r: 0 for r in range(self.num_ranks)}
        for p, ok in enumerate(m):
            if ok:
                load[self.rank_of(p)] += 1
        return load

    # -- mutations (visible after commit) ----------------------------------
    def set_rank_state(self, rank, state):
        if state not in (UP, DRAINING, DOWN):
            raise ValueError(state)
        if rank >= self.num_ranks:
            raise PlacementError(f"rank {rank} out of range")
        if self.rank_state[rank] != state:
            self.rank_state[rank] = state
            if state == DOWN:   # its in-progress copies are void
                self.staging = {(l, p) for l, p in self.staging if self.rank_of(p) != rank}
            self._dirty = True

    def mark_down(self, rank):
        self.set_rank_state(rank, DOWN)

    def mark_draining(self, rank):
        self.set_rank_state(rank, DRAINING)

    def mark_up(self, rank, slot_map_rows=None):
        """A rank (re)joins. ``slot_map_rows``: what its slots hold now, per
        layer (a fresh process loads the default layout; empty slots -1)."""
        if slot_map_rows is not None:
            for layer, row in enumerate(slot_map_rows):
                for i, e in enumerate(row):
                    self.slot_map[layer][rank * self.slots_per_rank + i] = int(e)
        self.set_rank_state(rank, UP)

    def begin_moves(self, moves: Iterable[Move]):
        """Destination slots stop being routable (commit before copying)."""
        for m in moves:
            if not self.live(self.rank_of(m.dst)):
                raise PlacementError(f"move to slot {m.dst} on a dead rank")
            self.staging.add((m.layer, m.dst))
        self._dirty = True

    def finish_moves(self, moves: Iterable[Move], ok=True):
        """After the copy: the destination holds ``expert`` (ok) or is empty."""
        for m in moves:
            self.staging.discard((m.layer, m.dst))
            if self.live(self.rank_of(m.dst)):
                self.slot_map[m.layer][m.dst] = m.expert if ok else -1
        self._dirty = True

    def commit(self):
        """Publish pending changes as a new epoch. Returns the epoch."""
        if self._dirty:
            self.epoch += 1
            self._dirty = False
        return self.epoch

    # -- planning -----------------------------------------------------------
    def plan_repair(self, min_replicas=1, prefer_src_rank=None):
        """Moves that give every (layer, expert) at least ``min_replicas``
        routable replicas on ``up`` ranks (best effort when slots run out).
        Does not change the table; apply with begin_moves / finish_moves."""
        moves = []
        for layer in range(self.num_layers):
            row = self.slot_map[layer]
            reserved = set()
            # replicas on up ranks (draining ranks are leaving)
            up_reps = {e: [p for p, x in enumerate(row) if x == e and (layer, p) not in self.staging
                           and self.rank_state[self.rank_of(p)] == UP] for e in range(self.num_experts)}
            need = sorted((e for e in range(self.num_experts) if len(up_reps[e]) < min_replicas),
                          key=lambda e: len(up_reps[e]))
            for e in need:
                while len(up_reps[e]) < min_replicas:
                    dst = self._pick_dst(layer, e, up_reps, reserved)
                    if dst is None:
                        break
                    src = self._pick_src(layer, e, dst, prefer_src_rank)
                    old = row[dst]
                    if old >= 0 and dst in up_reps.get(old, []):
                        up_reps[old].remove(dst)
                    reserved.add(dst)
                    up_reps[e].append(dst)
                    moves.append(Move(layer, e, dst, src, old))
        return moves

    def _pick_dst(self, layer, expert, up_reps, reserved):
        row = self.slot_map[layer]
        load = {r: 0 for r in range(self.num_ranks)}
        for p, x in enumerate(row):
            if x >= 0:
                load[self.rank_of(p)] += 1
        have_rank = {self.rank_of(p) for p in up_reps[expert]}
        best, best_key = None, None
        for p, x in enumerate(row):
            r = self.rank_of(p)
            if self.rank_state[r] != UP or p in reserved or (layer, p) in self.staging:
                continue
            if x == expert:
                continue
            if x >= 0 and len(up_reps.get(x, [])) <= 1:
                continue          # would strand the expert it holds
            # empty first, then the most replicated victim, then a rank that
            # does not hold the expert yet, then the least loaded rank
            key = (x >= 0, -len(up_reps.get(x, [])) if x >= 0 else 0, r in have_rank, load[r], p)
            if best_key is None or key < best_key:
                best, best_key = p, key
        return best

    def _pick_src(self, layer, expert, dst, prefer_rank=None):
        cands = [p for p, x in enumerate(self.slot_map[layer]) if x == expert
                 and (layer, p) not in self.staging and self.live(self.rank_of(p)) and p != dst]
        if not cands:
            return None
        dst_rank = self.rank_of(dst)

        def key(p):
            r = self.rank_of(p)
            return (r != dst_rank, prefer_rank is not None and r != prefer_rank,
                    self.rank_state[r] != UP, p)
        return min(cands, key=key)

    # -- serialisation ------------------------------------------------------
    def view(self):
        """What an engine applies: epoch, geometry, per-layer slot contents
        and routable masks."""
        return {
            "epoch": self.epoch,
            "num_experts": self.num_experts,
            "num_ranks": self.num_ranks,
            "slots_per_rank": self.slots_per_rank,
            "slot_map": copy.deepcopy(self.slot_map),
            "routable": [self.routable_mask(l) for l in range(self.num_layers)],
            "rank_state": {str(r): s for r, s in self.rank_state.items()},
            "active_replicas": self.active_replicas,
        }

    def to_dict(self):
        d = self.view()
        d["staging"] = sorted(self.staging)
        return d

    @classmethod
    def from_dict(cls, d):
        t = cls(d["num_experts"], d["num_ranks"], d["slots_per_rank"], d["slot_map"],
                epoch=d.get("epoch", 0), rank_state=d.get("rank_state"),
                active_replicas=d.get("active_replicas"))
        t.staging = {tuple(x) for x in d.get("staging", [])}
        return t


def logical_to_physical(view, layer, max_replicas=None):
    """Routing lists of one layer from a view: ``l2p[e]`` = routable slots of
    expert ``e``, padded with -1 to ``max_replicas``; plus replica counts."""
    row, mask = view["slot_map"][layer], view["routable"][layer]
    l2p = [[] for _ in range(view["num_experts"])]
    for p, (e, ok) in enumerate(zip(row, mask)):
        if ok and e >= 0:
            l2p[e].append(p)
    width = max_replicas or max(1, max(len(x) for x in l2p))
    counts = [len(x) for x in l2p]
    if max(counts) > width:
        raise PlacementError(f"an expert has {max(counts)} replicas, engine allows {width}")
    return [x + [-1] * (width - len(x)) for x in l2p], counts
