# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Reference directory service for DirectoryClient.

A small single-process ZeroMQ server that keeps the membership table in
memory. It speaks the same msgpack protocol as DirectoryClient:

- ``register`` with event JOIN / MIGRATE / LEAVE
- ``lookup`` by LID tuple ``(model, stage, shard, replica)``
- ``list_stage`` returning every member of a stage
- ``request_assignment`` returning the first replica of the target stage
- ``heartbeat`` refreshing a member's lease (``not_found`` if it was evicted)
- ``report_failure`` evicting a member a peer could not reach
- ``info`` returning the event PUB address

Membership changes are published on a PUB socket (one msgpack frame per
event: ``{"event", "endpoint", "epoch", "reason"}``) so that clients learn
about JOIN/LEAVE without polling. Members that stop heartbeating for
``lease_s`` seconds are evicted with a LEAVE event (reason ``expired``).
Leases only apply to members that have heartbeated at least once, so clients
that never call ``start_heartbeat`` are kept until they deregister.

Every membership change bumps a global epoch, which is returned in responses.
This is meant for tests, examples and small deployments; production setups
usually embed the same protocol in their own router.

Run standalone::

    python -m connex.directory_server --bind 0.0.0.0:5555
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any, Optional

import msgpack
import zmq

from connex.directory import MembershipEvent

logger = logging.getLogger(__name__)


class DirectoryServer:
    """In-memory directory served over ZeroMQ ROUTER (requests) + PUB (events)."""

    def __init__(
        self,
        bind_addr: str = "127.0.0.1:5555",
        event_bind_addr: Optional[str] = None,
        lease_s: float = 2.0,
    ):
        self.bind_addr = bind_addr
        host, port = bind_addr.rsplit(":", 1)
        self.event_bind_addr = event_bind_addr or f"{host}:{int(port) + 1}"
        self.lease_s = lease_s
        self.epoch = 0
        self._endpoints: dict[tuple, dict[str, Any]] = {}
        self._last_seen: dict[tuple, float] = {}  # only members that heartbeat
        self._pending_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        """Serve in a background thread; returns once the sockets are bound."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError(f"DirectoryServer failed to bind {self.bind_addr}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)
            self._thread = None

    def members(self) -> dict[tuple, dict[str, Any]]:
        with self._lock:
            return dict(self._endpoints)

    def serve_forever(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.bind_addr}")
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://{self.event_bind_addr}")
        self._ready.set()
        logger.info("DirectoryServer listening on %s (events on %s)",
                    self.bind_addr, self.event_bind_addr)
        tick = min(0.05, self.lease_s / 4)
        try:
            while not self._stop.is_set():
                if sock.poll(int(tick * 1000)):
                    # Drain everything queued so a burst of heartbeats does
                    # not delay event publication.
                    while True:
                        try:
                            frames = sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        ident, data = frames[0], frames[-1]
                        try:
                            resp = self.handle(msgpack.unpackb(data, raw=False))
                        except Exception as e:  # malformed request
                            resp = {"status": "error", "error": str(e)}
                        sock.send_multipart([ident, msgpack.packb(resp, use_bin_type=True)])
                self._expire_leases()
                with self._lock:
                    events, self._pending_events = self._pending_events, []
                for ev in events:
                    pub.send(msgpack.packb(ev, use_bin_type=True))
        finally:
            sock.close()
            pub.close()

    # -------------------------------------------------------------- handlers

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        kind = req.get("type")
        if kind == "register":
            return self._handle_register(req)
        if kind == "heartbeat":
            return self._handle_heartbeat(tuple(req["lid"]))
        if kind == "report_failure":
            return self._handle_report_failure(tuple(req["lid"]))
        if kind == "lookup":
            return self._handle_lookup(tuple(req["lid"]))
        if kind == "list_stage":
            return self._handle_list_stage(req["model"], req["stage"])
        if kind == "request_assignment":
            return self._handle_assignment(req)
        if kind == "info":
            return {"status": "ok", "event_address": self.event_bind_addr,
                    "lease_s": self.lease_s, "epoch": self.epoch}
        return {"status": "error", "error": f"unknown request type {kind!r}"}

    # All _locked helpers expect self._lock to be held.

    def _remove_locked(self, lid: tuple, reason: str) -> bool:
        ep = self._endpoints.pop(lid, None)
        self._last_seen.pop(lid, None)
        if ep is None:
            return False
        self.epoch += 1
        self._pending_events.append({"event": MembershipEvent.LEAVE.value,
                                     "endpoint": ep, "epoch": self.epoch,
                                     "reason": reason})
        logger.info("LEAVE %s (%s, epoch=%d)", lid, reason, self.epoch)
        return True

    def _handle_register(self, req: dict[str, Any]) -> dict[str, Any]:
        ep = req["endpoint"]
        lid = tuple(ep["lid"])
        event = req.get("event", MembershipEvent.JOIN.value)
        with self._lock:
            if event == MembershipEvent.LEAVE.value:
                if not self._remove_locked(lid, "deregister"):
                    return {"status": "not_found", "epoch": self.epoch}
                return {"status": "ok", "epoch": self.epoch}
            self._endpoints[lid] = ep
            if lid in self._last_seen:
                self._last_seen[lid] = time.monotonic()
            self.epoch += 1
            self._pending_events.append({"event": event, "endpoint": ep,
                                         "epoch": self.epoch, "reason": ""})
            logger.info("%s %s (epoch=%d)", event, lid, self.epoch)
            return {"status": "ok", "epoch": self.epoch}

    def _handle_heartbeat(self, lid: tuple) -> dict[str, Any]:
        with self._lock:
            if lid not in self._endpoints:
                return {"status": "not_found", "epoch": self.epoch}
            self._last_seen[lid] = time.monotonic()
            return {"status": "ok", "epoch": self.epoch}

    def _handle_report_failure(self, lid: tuple) -> dict[str, Any]:
        with self._lock:
            removed = self._remove_locked(lid, "reported")
            return {"status": "ok" if removed else "not_found", "epoch": self.epoch}

    def _expire_leases(self) -> None:
        now = time.monotonic()
        with self._lock:
            for lid in [l for l, t in self._last_seen.items() if now - t > self.lease_s]:
                self._remove_locked(lid, "expired")

    def _handle_lookup(self, lid: tuple) -> dict[str, Any]:
        with self._lock:
            ep = self._endpoints.get(lid)
            if ep is None:
                return {"status": "not_found", "epoch": self.epoch}
            return {"status": "ok", "endpoint": ep, "epoch": self.epoch}

    def _handle_list_stage(self, model: str, stage: int) -> dict[str, Any]:
        with self._lock:
            eps = [ep for lid, ep in sorted(self._endpoints.items())
                   if lid[0] == model and lid[1] == stage]
            return {"status": "ok", "endpoints": eps, "epoch": self.epoch}

    def _handle_assignment(self, req: dict[str, Any]) -> dict[str, Any]:
        model, stage = req.get("model"), req.get("target_stage")
        with self._lock:
            candidates = sorted(
                lid for lid in self._endpoints if lid[0] == model and lid[1] == stage
            )
            if not candidates:
                return {"status": "not_found", "epoch": self.epoch}
            ep = self._endpoints[candidates[0]]
            return {
                "status": "ok",
                "endpoint": ep["addresses"][0] if ep["addresses"] else "",
                "routing_key": "-".join(str(x) for x in candidates[0]),
                "epoch": self.epoch,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Connex reference directory server")
    parser.add_argument("--bind", default="0.0.0.0:5555", help="host:port for requests")
    parser.add_argument("--event-bind", default=None,
                        help="host:port for the event PUB socket (default: port+1)")
    parser.add_argument("--lease", type=float, default=2.0,
                        help="seconds without heartbeat before a member is evicted")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = DirectoryServer(args.bind, args.event_bind, args.lease)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
