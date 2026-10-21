# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Elastic peer set: the live, connected replicas of one downstream stage.

``ElasticPeerSet`` turns directory membership into a set of ready-to-use
transport endpoints, keeping scale events off the data path:

- JOIN / MIGRATE: the new address is connected on a background thread and
  only becomes routable once the connection is up, so the sender never blocks
  on connection setup.
- LEAVE (deregister, expired lease, or reported failure): the peer is removed
  from routing immediately; its endpoint is closed after ``drain_grace_s`` so
  in-flight messages are not cut off.
- Send failure: the peer is removed locally, reported to the directory (which
  evicts it and tells every other sender), and its endpoint is closed.

Membership comes from directory push events when the directory publishes
them, otherwise from polling ``list_stage`` every ``poll_s``.

Typical use on an upstream stage::

    peers = ElasticPeerSet(directory_client, transport, stage_idx=1)
    peers.start()
    peers.wait_ready(1)
    lid = peers.pick()
    peers.send(lid, payload)

``send``, ``pick`` and ``mark_failed`` must be called from one thread, the
same one that owns ``directory_client``'s request socket.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from connex.directory import DirectoryClient, EndpointInfo, MembershipEvent

logger = logging.getLogger(__name__)


class ElasticPeerSet:
    def __init__(
        self,
        directory: DirectoryClient,
        transport: Any,
        stage_idx: int,
        drain_grace_s: float = 1.0,
        poll_s: float = 0.1,
        connect_workers: int = 4,
    ):
        self.directory = directory
        self.transport = transport
        self.stage_idx = stage_idx
        self.drain_grace_s = drain_grace_s
        self.poll_s = poll_s

        self._lock = threading.Lock()
        self._ready: dict[tuple, tuple[Any, str]] = {}  # lid -> (endpoint, address)
        self._wanted: dict[tuple, str] = {}  # lid -> address we should be connected to
        self._order: list[tuple] = []  # routable lids, sorted
        self._rr = 0
        self._ready_cv = threading.Condition(self._lock)
        self._pool = ThreadPoolExecutor(max_workers=connect_workers,
                                        thread_name_prefix="connex-peer")
        self._stop = threading.Event()
        self._push = False
        self.stats = {"connects": 0, "connect_ms_total": 0.0, "failures": 0, "leaves": 0}

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.directory.subscribe_events(self._on_event)
        self._push = self.directory.start_event_listener()
        # Snapshot after subscribing so no change is missed in between.
        self._sync(self.directory.list_stage(self.stage_idx))
        if not self._push:
            logger.info("Directory has no event stream; polling every %.0f ms",
                        self.poll_s * 1000)
            threading.Thread(target=self._poll_loop, daemon=True,
                             name="connex-peer-poll").start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            eps = [ep for ep, _ in self._ready.values()]
            self._ready.clear()
            self._wanted.clear()
            self._order = []
        for ep in eps:
            self._close(ep)
        self._pool.shutdown(wait=False)

    # --------------------------------------------------------------- query

    def peers(self) -> list[tuple]:
        with self._lock:
            return list(self._order)

    def wait_ready(self, n: int = 1, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._ready_cv:
            while len(self._order) < n:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._ready_cv.wait(left)
            return True

    def pick(self) -> Optional[tuple]:
        """Round-robin over routable peers; None if there are none."""
        with self._lock:
            if not self._order:
                return None
            lid = self._order[self._rr % len(self._order)]
            self._rr += 1
            return lid

    def endpoint(self, lid: tuple) -> Optional[Any]:
        with self._lock:
            entry = self._ready.get(lid)
            return entry[0] if entry else None

    # ---------------------------------------------------------------- data

    def send(self, lid: tuple, data: Any, tag: int = 0) -> None:
        """Send to ``lid``; on failure the peer is marked failed and the error re-raised."""
        ep = self.endpoint(lid)
        if ep is None:
            raise KeyError(f"peer {lid} is not routable")
        try:
            self.transport.send_sync(ep, data, tag)
        except Exception:
            self.mark_failed(lid)
            raise

    def mark_failed(self, lid: tuple) -> None:
        with self._lock:
            entry = self._ready.pop(lid, None)
            self._wanted.pop(lid, None)
            self._reorder_locked()
        self.stats["failures"] += 1
        logger.warning("Peer %s failed; removed and reported", lid)
        self.directory.report_failure(lid)
        if entry is not None:
            self._pool.submit(self._close, entry[0])

    # ------------------------------------------------------------ internals

    def _reorder_locked(self) -> None:
        self._order = sorted(self._ready)
        self._ready_cv.notify_all()

    def _on_event(self, event: MembershipEvent, info: EndpointInfo) -> None:
        if info.stage != self.stage_idx or info.model != self.directory.model_name:
            return
        if event == MembershipEvent.LEAVE:
            self._remove(info.lid)
        elif info.addresses:
            self._want(info.lid, info.primary_address)

    def _sync(self, members: list[EndpointInfo]) -> None:
        current = {m.lid: m.primary_address for m in members if m.addresses}
        with self._lock:
            gone = [lid for lid in self._wanted if lid not in current]
        for lid in gone:
            self._remove(lid)
        for lid, addr in current.items():
            self._want(lid, addr)

    def _poll_loop(self) -> None:
        # Uses its own client: the owner's request socket is not thread-safe.
        client = DirectoryClient(self.directory.front_router_addr,
                                 self.directory.model_name, stage_idx=-1,
                                 replica_idx=id(self) & 0xFFFF)
        client.start()
        try:
            while not self._stop.wait(self.poll_s):
                self._sync(client.list_stage(self.stage_idx))
        finally:
            client.stop()

    def _want(self, lid: tuple, addr: str) -> None:
        with self._lock:
            if self._wanted.get(lid) == addr:
                return  # already connected or connecting to this address
            self._wanted[lid] = addr
        self._pool.submit(self._connect, lid, addr)

    def _connect(self, lid: tuple, addr: str) -> None:
        host, port = addr.rsplit(":", 1)
        t0 = time.monotonic()
        try:
            ep = self.transport.connect_sync(host, int(port))
        except Exception as e:
            logger.warning("Connect to %s (%s) failed: %s", lid, addr, e)
            with self._lock:
                if self._wanted.get(lid) == addr:
                    del self._wanted[lid]
            return
        old = None
        with self._lock:
            if self._wanted.get(lid) != addr:
                stale = True  # left or migrated while we were connecting
            else:
                stale = False
                old = self._ready.get(lid)
                self._ready[lid] = (ep, addr)
                self._reorder_locked()
        if stale:
            self._close(ep)
            return
        self.stats["connects"] += 1
        self.stats["connect_ms_total"] += (time.monotonic() - t0) * 1000
        logger.info("Peer %s ready at %s (%.1f ms)", lid, addr,
                    (time.monotonic() - t0) * 1000)
        if old is not None and old[0] is not ep:
            self._close_later(old[0])  # migrated: drain the old path

    def _remove(self, lid: tuple) -> None:
        with self._lock:
            self._wanted.pop(lid, None)
            entry = self._ready.pop(lid, None)
            self._reorder_locked()
        if entry is not None:
            self.stats["leaves"] += 1
            self._close_later(entry[0])

    def _close_later(self, ep: Any) -> None:
        def run():
            with self._lock:
                # The transport caches endpoints by address, so a peer that
                # rejoined at the same address may be using this endpoint again.
                in_use = any(e is ep for e, _ in self._ready.values())
            if not in_use and not self._stop.is_set():
                self._close(ep)
        timer = threading.Timer(self.drain_grace_s, run)
        timer.daemon = True
        timer.start()

    def _close(self, ep: Any) -> None:
        try:
            self.transport.close_endpoint(ep)
        except Exception as e:
            logger.debug("close_endpoint failed: %s", e)
