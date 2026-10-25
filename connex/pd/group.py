# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Group-level failure detection for tensor-parallel instances.

A tensor-parallel (TP) instance is one mobile endpoint made of ``n`` shards,
one per TP rank (``LID(model, stage, shard=rank, replica)``, registered with
``group_size=n``). All-reduce inside the group stays on NCCL/NVLink; Connex
only needs to know when the group as a whole can no longer serve. Two
signals say so, whichever comes first:

* a rank's lease lapses at the front-router (the rank's process is gone or
  hung); the front-router fails the whole group (``fail_group``);
* a peer's data plane sees the rank die: connex-rs raises a peer-death event
  within milliseconds when the kernel tears down the dead process's
  connections. ``PeerDeathReporter`` turns that event into
  ``report_failure`` for the rank's LID, and the front-router fails the
  group the same way.

Either way every shard of the group is removed and announced as
deregistered, ``fetch_pools`` stops listing the instance, and the PD router
hands its requests over. Ranks that are still alive re-register on their
next heartbeat; the instance is routable again (same identity) once all of
its ranks are registered, e.g. after it is restarted.

Engine neutral; the connex-rs engine is duck-typed (``wait_event``).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Optional

from connex.protocol import LogicalEndpointID

log = logging.getLogger("connex.pd")


class GroupFailureTracker:
    """Pure logic: which peer LIDs to report for a stream of transport
    events. A link must have been up before its failure counts (a failed
    reconnect to a peer already known dead is not news), timeouts are only
    suspicion (hung or partitioned, left to the lease), and each peer is
    reported once per incarnation (until its link comes up again).

    ``stages``: the stage indices this tracker cares about (e.g. the PD
    stages); events of other identities are ignored."""

    def __init__(self, model: Optional[str], stages: Iterable[int]):
        self.model = model
        self.stages = set(int(s) for s in stages)
        self.up: set[tuple] = set()
        self.reported: set[tuple] = set()

    def on_event(self, ev: dict) -> Optional[LogicalEndpointID]:
        peer = ev.get("peer")
        if not peer or len(peer) < 4:
            return None
        model, stage, rank, replica = peer[0], int(peer[1]), int(peer[2]), int(peer[3])
        if stage not in self.stages or (self.model and model != self.model):
            return None
        key = (model, stage, rank, replica)
        if ev.get("type") in ("up", "joined"):
            self.up.add(key)
            self.reported.discard(key)
            return None
        if ev.get("type") != "failed" or key not in self.up or key in self.reported:
            return None
        if "timeout" in str(ev.get("reason", "")):
            return None
        self.up.discard(key)
        self.reported.add(key)
        return LogicalEndpointID(model, stage, rank, replica)


class PeerDeathReporter:
    """Reads a connex-rs engine's peer events and reports dead peers (one
    TP rank each) to the front-router, which fails their whole group."""

    def __init__(self, engine, front_router_api: str, model: Optional[str],
                 stages: Optional[Iterable[int]] = None,
                 report: Optional[Callable[[LogicalEndpointID, str], object]] = None):
        if stages is None:
            from connex.pd.membership import PD_STAGE

            stages = PD_STAGE.values()
        self.engine = engine
        self.tracker = GroupFailureTracker(model, stages)
        self.front_router = front_router_api
        if report is None:
            from connex.pd.membership import report_failure

            def report(lid, reason):
                return report_failure(front_router_api, lid, reason)
        self._report = report
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reports: list[tuple[float, LogicalEndpointID, object]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="connex-pd-peer-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self.engine.wait_event(100)
            except Exception:  # noqa: BLE001  (engine closed)
                return
            if ev is None:
                continue
            self.handle(ev)

    def handle(self, ev: dict) -> Optional[LogicalEndpointID]:
        lid = self.tracker.on_event(ev)
        if lid is None:
            return None
        reason = f"peer died ({ev.get('reason', 'transport event')})"
        log.warning("connex-rs: %s rank %d of replica %d died: reporting its group",
                    lid.model_name, lid.shard_idx, lid.replica_idx)
        try:
            res = self._report(lid, reason)
        except Exception as e:  # noqa: BLE001
            res = e
        self.reports.append((time.time(), lid, res))
        return lid

    def stop(self) -> None:
        self._stop.set()
