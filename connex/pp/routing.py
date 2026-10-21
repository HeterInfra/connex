# SPDX-License-Identifier: Apache-2.0
"""Membership and peer selection for elastic pipelines (engine neutral).

A *membership* source answers "which replicas of stage s exist and where".
In production it is :class:`connex.epc_client.EPCV2Client` (Front Router
registration, leases, PUB/SUB events). Anything with the same small duck
type works, e.g. :class:`StaticMembership` for tests or for a fixed
deployment without a router::

    lid.replica_idx                       our replica index
    get_pipeline_peers() -> [Route]       cached routes of all stages
    refresh_routes()                      re-read the routes from the source
    get_stage_peers(stage, refresh=...)   optional: routes of one stage
    on_event(event_type, callback)        optional: membership events
    start() -> bool / stop()              optional

Route = object with ``.lid.stage_idx``, ``.lid.replica_idx``, ``.addresses``.

Peer choice (:func:`select_peer`) is *sticky*: once a stage picked replica r
of its neighbour stage it keeps r while r is listed and not known dead, so
in-flight requests keep finding their KV. A new choice is rank based (our
rank among the replicas of our stage, modulo the number of target replicas),
so chains spread 1:1 when the replica counts match.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lid:
    stage_idx: int
    replica_idx: int
    model_name: str = "default"


@dataclass
class Route:
    lid: Lid
    addresses: list = field(default_factory=list)


def select_peer(all_peers: list, target_stage: int, my_stage: int, my_replica: int,
                current: Optional[int], deregistered: set, exclude_replica: Optional[int] = None,
                log: logging.Logger = _log):
    """Choose the replica of ``target_stage`` to talk to.

    Returns ``(address, replica_idx, is_new_choice)`` or ``(None, None, False)``.
    Rules, in order: drop addresses listed under more than one stage (stale
    artefacts of an earlier deployment; fall back to the unfiltered list if
    that leaves nothing); drop ``exclude_replica``; skip replicas known to
    have departed unless nothing else is left; keep ``current`` if still
    listed; else the only one; else rank based.
    """
    addr_to_stages: dict[str, set[int]] = defaultdict(set)
    for r in all_peers:
        for addr in r.addresses:
            addr_to_stages[addr].add(r.lid.stage_idx)
    ambiguous = {a for a, st in addr_to_stages.items() if len(st) > 1}
    if ambiguous:
        log.warning("Stage %s: Detected %d ambiguous address(es) across stages "
                    "(stale route artefact), excluding from peer selection: %s",
                    my_stage, len(ambiguous), {a: addr_to_stages[a] for a in ambiguous})

    peers = [r for r in all_peers
             if r.lid.stage_idx == target_stage and r.addresses
             and r.addresses[0] not in ambiguous
             and (exclude_replica is None or r.lid.replica_idx != exclude_replica)]
    if not peers:
        log.warning("Stage %s: No clean peers for stage %s after ambiguity "
                    "filter, falling back to unfiltered list", my_stage, target_stage)
        peers = [r for r in all_peers
                 if r.lid.stage_idx == target_stage and r.addresses
                 and (exclude_replica is None or r.lid.replica_idx != exclude_replica)]

    live = [r for r in peers if (r.lid.stage_idx, r.lid.replica_idx) not in deregistered]
    if live:
        peers = live
    if not peers:
        return None, None, False
    for r in peers:
        if r.lid.replica_idx == current:
            return r.addresses[0], current, False
    if len(peers) == 1:
        return peers[0].addresses[0], peers[0].lid.replica_idx, True

    mine = sorted((r for r in all_peers if r.lid.stage_idx == my_stage and r.addresses),
                  key=lambda r: r.lid.replica_idx)
    log.info("Stage %s: my_replica_idx=%s, my_stage_peers=%s",
             my_stage, my_replica, [p.lid.replica_idx for p in mine])
    my_rank = 0
    for i, p in enumerate(mine):
        if p.lid.replica_idx == my_replica:
            my_rank = i
            break
    peers.sort(key=lambda r: r.lid.replica_idx)
    sel_i = my_rank % len(peers)
    sel = peers[sel_i]
    log.info("Stage %s (replica %s, rank %d/%d): Selected peer stage %s replica %s (idx %d/%d), addr=%s%s",
             my_stage, my_replica, my_rank, len(mine), target_stage, sel.lid.replica_idx,
             sel_i, len(peers), sel.addresses[0] if sel.addresses else "N/A",
             f" (excluded R{exclude_replica})" if exclude_replica else "")
    return sel.addresses[0], sel.lid.replica_idx, True


class StaticMembership:
    """In-process membership source (tests, fixed deployments). Mutate it
    with :meth:`add` / :meth:`remove`; registered callbacks receive events
    shaped like the EPC client's (``event.payload["lid"]`` dict)."""

    def __init__(self, stage_idx: int, replica_idx: int = 0, routes: Iterable = (),
                 model_name: str = "default"):
        self.lid = Lid(stage_idx, replica_idx, model_name)
        self.routes: list = list(routes)
        self.refreshes = 0
        self._handlers: dict = defaultdict(list)

    # EPC duck type
    def start(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def refresh_routes(self) -> None:
        self.refreshes += 1

    def get_pipeline_peers(self) -> list:
        return list(self.routes)

    def get_stage_peers(self, stage_idx: int, refresh: str = "auto") -> list:
        if refresh == "always":
            self.refreshes += 1
        return [r for r in self.routes if r.lid.stage_idx == stage_idx]

    def on_event(self, event_type, callback: Callable) -> None:
        self._handlers[event_type].append(callback)

    # mutation
    def add(self, stage_idx: int, replica_idx: int, address: str) -> Route:
        r = Route(Lid(stage_idx, replica_idx), [address])
        self.routes.append(r)
        return r

    def remove(self, stage_idx: int, replica_idx: int) -> None:
        self.routes = [r for r in self.routes
                       if (r.lid.stage_idx, r.lid.replica_idx) != (stage_idx, replica_idx)]

    def emit(self, event_type, stage_idx: int, replica_idx: int, **payload) -> None:
        class _E:
            pass
        e = _E()
        e.payload = {"lid": {"stage_idx": stage_idx, "replica_idx": replica_idx}, **payload}
        for cb in self._handlers.get(event_type, ()):
            cb(e)
