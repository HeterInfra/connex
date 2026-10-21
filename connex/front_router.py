#!/usr/bin/env python3
"""
Front Router Service - Connex EPC V2 Control Plane.

Independent control plane service that manages:
- Authoritative RouteTable with persistent state
- Global Epoch coordination
- Membership registry with heartbeat tracking
- Migration orchestration
- Event distribution via ZeroMQ PUB

The control plane is separated from vLLM instances, surviving instance recycles.
"""

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default


# Failure-detection knobs (router side). Defaults keep the historical
# behaviour (30 s timeout, 60 s grace). For fast failover set e.g.
#   CONNEX_HEARTBEAT_TIMEOUT_S=3  (with client CONNEX_HEARTBEAT_INTERVAL_S=1)
# The check interval defaults to min(5 s, timeout / 4) so detection latency
# is bounded by roughly timeout * 1.25 instead of timeout + 5 s.
ENV_HEARTBEAT_TIMEOUT = "CONNEX_HEARTBEAT_TIMEOUT_S"
ENV_HEALTH_CHECK_INTERVAL = "CONNEX_HEALTH_CHECK_INTERVAL_S"
ENV_UNHEALTHY_GRACE = "CONNEX_UNHEALTHY_GRACE_S"


# ============================================================================
# Data Types
# ============================================================================

class RouteState(IntEnum):
    """State of a route entry."""
    UNKNOWN = 0
    ACTIVE = 1
    DRAINING = 2
    MIGRATING = 3
    UNHEALTHY = 4
    

class EventType(IntEnum):
    """Types of control plane events."""
    ENDPOINT_REGISTERED = 0
    ENDPOINT_DEREGISTERED = 1
    ENDPOINT_UNHEALTHY = 2
    EPOCH_BUMPED = 3
    PREPARE_MIGRATION = 4
    SHADOW_READY = 5
    CUTOVER_START = 6
    CUTOVER_COMPLETE = 7
    MIGRATION_FAILED = 8
    ROUTE_UPDATED = 9


class TransportCapability(IntEnum):
    """Transport capabilities of an endpoint."""
    ZEROMQ = 0
    UCX_TCP = 1
    UCX_RDMA = 2
    GPU_DIRECT = 3
    MOONCAKE_TE = 4


@dataclass
class LogicalEndpointID:
    """Logical Endpoint Identifier - stable identity across migrations."""
    model_name: str
    stage_idx: int
    shard_idx: int = 0
    replica_idx: int = 0
    
    def to_key(self) -> str:
        return f"{self.model_name}:{self.stage_idx}:{self.shard_idx}:{self.replica_idx}"
    
    @classmethod
    def from_key(cls, key: str) -> "LogicalEndpointID":
        parts = key.split(":")
        return cls(
            model_name=parts[0],
            stage_idx=int(parts[1]),
            shard_idx=int(parts[2]) if len(parts) > 2 else 0,
            replica_idx=int(parts[3]) if len(parts) > 3 else 0,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogicalEndpointID":
        return cls(**d)


@dataclass
class RouteEntry:
    """A route entry in the global route table."""
    lid: LogicalEndpointID
    addresses: List[str]
    capabilities: List[TransportCapability]
    epoch: int
    state: RouteState = RouteState.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lid": self.lid.to_dict(),
            "addresses": self.addresses,
            "capabilities": [int(c) for c in self.capabilities],
            "epoch": self.epoch,
            "state": int(self.state),
            "last_heartbeat": self.last_heartbeat,
            "registered_at": self.registered_at,
            "metadata": self.metadata,
        }
    
    def to_compact_dict(self) -> Dict[str, Any]:
        """Only the fields data-plane clients need (no timestamps/metadata)."""
        return {
            "lid": self.lid.to_dict(),
            "addresses": list(self.addresses),
            "capabilities": [int(c) for c in self.capabilities],
            "epoch": self.epoch,
            "state": int(self.state),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteEntry":
        return cls(
            lid=LogicalEndpointID.from_dict(d["lid"]),
            addresses=d["addresses"],
            capabilities=[TransportCapability(c) for c in d["capabilities"]],
            epoch=d["epoch"],
            state=RouteState(d.get("state", RouteState.ACTIVE)),
            last_heartbeat=d.get("last_heartbeat", time.time()),
            registered_at=d.get("registered_at", time.time()),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Event:
    """Control plane event for distribution."""
    type: EventType
    epoch: int
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source_lid: Optional[str] = None
    # Monotonic per-router publish sequence number (assigned by EventBus).
    # Additive wire field: old clients ignore it, new clients use it to
    # detect missed PUB/SUB messages and fall back to a full refresh.
    seq: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": int(self.type),
            "epoch": self.epoch,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source_lid": self.source_lid,
        }
        if self.seq is not None:
            d["seq"] = self.seq
        return d
    
    def serialize(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")
    
    @classmethod
    def deserialize(cls, data: bytes) -> "Event":
        d = json.loads(data.decode("utf-8"))
        return cls(
            type=EventType(d["type"]),
            epoch=d["epoch"],
            payload=d["payload"],
            timestamp=d.get("timestamp", time.time()),
            source_lid=d.get("source_lid"),
            seq=d.get("seq"),
        )


@dataclass
class MigrationRequest:
    """A pending migration request."""
    migration_id: str
    lid: LogicalEndpointID
    old_address: str
    new_address: str
    old_epoch: int
    new_epoch: int
    state: str = "pending"  # pending, preparing, ready, cutting_over, complete, failed
    shadows_ready: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ============================================================================
# Route Manager
# ============================================================================

class RouteManager:
    """Manages the authoritative global route table.

    Besides the primary key -> route map it keeps two secondary structures:

    * ``_by_stage``: (model_name, stage_idx) -> {key: route}, so stage-scoped
      queries cost O(routes in that stage) instead of O(all routes).
    * ``_hb_order``: an OrderedDict key -> last touch time kept in
      heartbeat order (oldest first).  Expiry scans walk it from the front
      and stop at the first non-expired entry, so a health check costs
      O(expired + 1) instead of O(N).
    """
    
    def __init__(self, persistence: Optional["PersistenceBackend"] = None):
        self._routes: Dict[str, RouteEntry] = {}
        self._by_stage: Dict[Tuple[str, int], Dict[str, RouteEntry]] = {}
        self._hb_order: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.RLock()
        self._persistence = persistence
        
        # Load from persistence if available
        if self._persistence:
            self._load_from_persistence()
    
    # -- index helpers (caller holds the lock) ------------------------------
    
    def _index_add(self, key: str, route: RouteEntry):
        self._routes[key] = route
        self._by_stage.setdefault(
            (route.lid.model_name, route.lid.stage_idx), {}
        )[key] = route
        self._touch_locked(key, route.last_heartbeat)
    
    def _index_remove(self, key: str) -> Optional[RouteEntry]:
        route = self._routes.pop(key, None)
        self._hb_order.pop(key, None)
        if route is not None:
            skey = (route.lid.model_name, route.lid.stage_idx)
            bucket = self._by_stage.get(skey)
            if bucket is not None:
                bucket.pop(key, None)
                if not bucket:
                    del self._by_stage[skey]
        return route
    
    def _touch_locked(self, key: str, ts: float):
        """Record a heartbeat time, keeping _hb_order sorted by time."""
        self._hb_order[key] = ts
        self._hb_order.move_to_end(key)
    
    def _load_from_persistence(self):
        """Load routes from persistent storage."""
        try:
            data = self._persistence.get_all("routes")
            entries = sorted(
                (RouteEntry.from_dict(v) for v in data.values()),
                key=lambda r: r.last_heartbeat,
            )
            for route in entries:
                self._index_add(route.lid.to_key(), route)
            logger.info("Loaded %d routes from persistence", len(self._routes))
        except Exception as e:
            logger.warning("Failed to load routes from persistence: %s", e)
    
    def add_route(
        self,
        lid: LogicalEndpointID,
        addresses: List[str],
        capabilities: List[TransportCapability],
        epoch: int,
    ) -> RouteEntry:
        """Add or update a route entry."""
        key = lid.to_key()
        
        with self._lock:
            existing = self._routes.get(key)
            now = time.time()
            
            if existing:
                # Update existing route
                existing.addresses = addresses
                existing.capabilities = capabilities
                existing.epoch = epoch
                existing.state = RouteState.ACTIVE
                existing.last_heartbeat = now
                self._touch_locked(key, now)
                route = existing
            else:
                # Create new route
                route = RouteEntry(
                    lid=lid,
                    addresses=addresses,
                    capabilities=capabilities,
                    epoch=epoch,
                    last_heartbeat=now,
                )
                self._index_add(key, route)
            
            # Persist
            if self._persistence:
                self._persistence.set(f"routes/{key}", route.to_dict())
            
            return route
    
    def get_route(self, lid: LogicalEndpointID) -> Optional[RouteEntry]:
        """Get a route entry by LID."""
        with self._lock:
            return self._routes.get(lid.to_key())
    
    def get_routes_for_model(self, model_name: str) -> List[RouteEntry]:
        """Get all routes for a model."""
        with self._lock:
            return [
                r for r in self._routes.values()
                if r.lid.model_name == model_name
            ]
    
    def get_routes_for_stage(
        self, model_name: str, stage_idx: int
    ) -> List[RouteEntry]:
        """Get routes of one pipeline stage (O(stage size)), sorted by replica."""
        with self._lock:
            bucket = self._by_stage.get((model_name, int(stage_idx)))
            routes = list(bucket.values()) if bucket else []
        routes.sort(key=lambda r: (r.lid.shard_idx, r.lid.replica_idx))
        return routes
    
    def get_all_routes(self) -> List[RouteEntry]:
        """Get all routes."""
        with self._lock:
            return list(self._routes.values())
    
    def remove_route(self, lid: LogicalEndpointID) -> Optional[RouteEntry]:
        """Remove a route entry."""
        key = lid.to_key()
        
        with self._lock:
            route = self._index_remove(key)
            
            if route and self._persistence:
                self._persistence.delete(f"routes/{key}")
            
            return route
    
    def update_route_state(
        self, 
        lid: LogicalEndpointID, 
        state: RouteState
    ) -> Optional[RouteEntry]:
        """Update the state of a route."""
        with self._lock:
            route = self._routes.get(lid.to_key())
            if route:
                route.state = state
                if self._persistence:
                    self._persistence.set(f"routes/{lid.to_key()}", route.to_dict())
            return route
    
    def update_addresses(
        self, lid: LogicalEndpointID, addresses: List[str]
    ) -> Optional[Tuple[RouteEntry, List[str]]]:
        """Replace a route's addresses, mark it ACTIVE and refresh its heartbeat.

        Returns (route, old_addresses) or None if the route does not exist.
        """
        key = lid.to_key()
        with self._lock:
            route = self._routes.get(key)
            if route is None:
                return None
            old = list(route.addresses)
            now = time.time()
            route.addresses = addresses
            route.last_heartbeat = now
            route.state = RouteState.ACTIVE
            self._touch_locked(key, now)
            if self._persistence:
                self._persistence.set(f"routes/{key}", route.to_dict())
            return route, old
    
    def heartbeat(self, lid: LogicalEndpointID) -> Tuple[bool, bool]:
        """Record a heartbeat. Returns (found, recovered_from_unhealthy)."""
        key = lid.to_key()
        with self._lock:
            route = self._routes.get(key)
            if not route:
                return False, False
            now = time.time()
            route.last_heartbeat = now
            self._touch_locked(key, now)
            recovered = route.state == RouteState.UNHEALTHY
            if recovered:
                route.state = RouteState.ACTIVE
            return True, recovered
    
    def update_heartbeat(self, lid: LogicalEndpointID) -> bool:
        """Update heartbeat timestamp for a route."""
        return self.heartbeat(lid)[0]
    
    def collect_expired(
        self, timeout_s: float, grace_s: float, now: Optional[float] = None
    ) -> Tuple[List[RouteEntry], List[RouteEntry]]:
        """Expire routes in O(expired + 1).

        Marks ACTIVE routes whose heartbeat is older than ``timeout_s`` as
        UNHEALTHY and removes UNHEALTHY routes older than
        ``timeout_s + grace_s``.  Returns (newly_unhealthy, removed).
        """
        now = time.time() if now is None else now
        cutoff = now - timeout_s
        remove_cutoff = cutoff - grace_s
        newly_unhealthy: List[RouteEntry] = []
        to_remove: List[str] = []
        retouch: List[Tuple[str, float]] = []
        with self._lock:
            for key, ts in self._hb_order.items():
                if ts >= cutoff:
                    break  # ordered by touch time: the rest are fresh
                route = self._routes.get(key)
                if route is None:
                    to_remove.append(key)
                    continue
                if route.last_heartbeat >= cutoff:
                    # last_heartbeat was refreshed outside touch(); re-sort.
                    retouch.append((key, route.last_heartbeat))
                    continue
                if route.state == RouteState.ACTIVE:
                    route.state = RouteState.UNHEALTHY
                    newly_unhealthy.append(route)
                    if self._persistence:
                        self._persistence.set(f"routes/{key}", route.to_dict())
                elif (route.state == RouteState.UNHEALTHY
                      and route.last_heartbeat < remove_cutoff):
                    to_remove.append(key)
            for key, ts in retouch:
                self._touch_locked(key, ts)
            removed: List[RouteEntry] = []
            for key in to_remove:
                route = self._index_remove(key)
                if route is not None:
                    removed.append(route)
                    if self._persistence:
                        self._persistence.delete(f"routes/{key}")
        return newly_unhealthy, removed
    
    def get_unhealthy_routes(self, timeout_s: float = 30.0) -> List[RouteEntry]:
        """Get ACTIVE routes that haven't sent heartbeat within timeout."""
        cutoff = time.time() - timeout_s
        with self._lock:
            out = []
            for key, ts in self._hb_order.items():
                if ts >= cutoff:
                    break
                r = self._routes.get(key)
                if r is not None and r.state == RouteState.ACTIVE:
                    out.append(r)
            return out


# ============================================================================
# Epoch Manager
# ============================================================================

class EpochManager:
    """Manages the global epoch counter with atomic increments."""
    
    def __init__(self, persistence: Optional["PersistenceBackend"] = None):
        self._epoch = 0
        self._lock = threading.Lock()
        self._persistence = persistence
        self._history: List[Tuple[int, float, str]] = []  # (epoch, timestamp, reason)
        
        # Load from persistence
        if self._persistence:
            try:
                data = self._persistence.get("epoch")
                if data:
                    self._epoch = data.get("value", 0)
                    logger.info("Loaded epoch %d from persistence", self._epoch)
            except Exception as e:
                logger.warning("Failed to load epoch: %s", e)
    
    @property
    def current(self) -> int:
        """Get current epoch."""
        with self._lock:
            return self._epoch
    
    def bump(self, reason: str = "manual") -> int:
        """Atomically increment epoch and return new value."""
        with self._lock:
            self._epoch += 1
            new_epoch = self._epoch
            
            # Record history
            self._history.append((new_epoch, time.time(), reason))
            if len(self._history) > 1000:
                self._history = self._history[-1000:]
            
            # Persist
            if self._persistence:
                self._persistence.set("epoch", {
                    "value": new_epoch,
                    "updated_at": time.time(),
                    "reason": reason,
                })
            
            logger.info("Epoch bumped to %d (reason: %s)", new_epoch, reason)
            return new_epoch
    
    def get_history(self, limit: int = 100) -> List[Tuple[int, float, str]]:
        """Get recent epoch history."""
        with self._lock:
            return self._history[-limit:]


# ============================================================================
# Member Manager
# ============================================================================

class MemberManager:
    """Manages endpoint membership with heartbeat tracking.

    Timing is configurable per instance (constructor args) or via env:
    ``CONNEX_HEARTBEAT_TIMEOUT_S`` (default 30), ``CONNEX_UNHEALTHY_GRACE_S``
    (default 60) and ``CONNEX_HEALTH_CHECK_INTERVAL_S`` (default
    min(5, timeout/4)).  Worst-case detection latency is
    ``timeout + check_interval``.  Recommended fast-failover setting:
    client interval 1 s, router timeout 3 s (check interval 0.75 s).
    """
    
    HEARTBEAT_TIMEOUT_S = 30.0
    GRACE_PERIOD_S = 60.0  # Keep route after unhealthy before removal
    
    def __init__(
        self,
        route_manager: RouteManager,
        on_unhealthy: Optional[Callable[[LogicalEndpointID], None]] = None,
        heartbeat_timeout_s: Optional[float] = None,
        grace_period_s: Optional[float] = None,
        check_interval_s: Optional[float] = None,
        on_removed: Optional[Callable[[LogicalEndpointID], None]] = None,
    ):
        self._route_manager = route_manager
        self._on_unhealthy = on_unhealthy
        self._on_removed = on_removed
        self._check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Instance attributes shadow the class defaults (still overridable
        # by assigning MemberManager.HEARTBEAT_TIMEOUT_S before construction).
        self.HEARTBEAT_TIMEOUT_S = (
            heartbeat_timeout_s if heartbeat_timeout_s is not None
            else _env_float(ENV_HEARTBEAT_TIMEOUT, type(self).HEARTBEAT_TIMEOUT_S)
        )
        self.GRACE_PERIOD_S = (
            grace_period_s if grace_period_s is not None
            else _env_float(ENV_UNHEALTHY_GRACE, type(self).GRACE_PERIOD_S)
        )
        default_interval = min(5.0, max(0.05, self.HEARTBEAT_TIMEOUT_S / 4.0))
        self.check_interval_s = (
            check_interval_s if check_interval_s is not None
            else _env_float(ENV_HEALTH_CHECK_INTERVAL, default_interval)
        )
    
    def start(self):
        """Start the health check thread."""
        self._stop_event.clear()
        self._check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="member-health-check",
        )
        self._check_thread.start()
        logger.info(
            "Member health check started (timeout=%.2fs, interval=%.2fs, grace=%.1fs)",
            self.HEARTBEAT_TIMEOUT_S, self.check_interval_s, self.GRACE_PERIOD_S,
        )
    
    def stop(self):
        """Stop the health check thread."""
        self._stop_event.set()
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
    
    def _health_check_loop(self):
        """Periodic health check loop."""
        while not self._stop_event.wait(timeout=self.check_interval_s):
            try:
                self._check_health()
            except Exception as e:
                logger.error("Health check error: %s", e)
    
    def _check_health(self):
        """Check health of endpoints in O(expired + 1)."""
        now = time.time()
        unhealthy, removed = self._route_manager.collect_expired(
            self.HEARTBEAT_TIMEOUT_S, self.GRACE_PERIOD_S, now=now
        )
        
        for route in unhealthy:
            logger.warning(
                "Endpoint %s is unhealthy (last heartbeat: %.1fs ago)",
                route.lid.to_key(),
                now - route.last_heartbeat
            )
            if self._on_unhealthy:
                try:
                    self._on_unhealthy(route.lid)
                except Exception as e:
                    logger.error("on_unhealthy callback error: %s", e)
        
        for route in removed:
            logger.info("Removing stale route: %s", route.lid.to_key())
            if self._on_removed:
                try:
                    self._on_removed(route.lid)
                except Exception as e:
                    logger.error("on_removed callback error: %s", e)


# ============================================================================
# Migration Orchestrator
# ============================================================================

class MigrationOrchestrator:
    """Orchestrates endpoint migrations with shadow warmup."""
    
    SHADOW_TIMEOUT_S = 30.0
    CUTOVER_TIMEOUT_S = 10.0
    
    def __init__(
        self,
        route_manager: RouteManager,
        epoch_manager: EpochManager,
        event_bus: "EventBus",
    ):
        self._route_manager = route_manager
        self._epoch_manager = epoch_manager
        self._event_bus = event_bus
        self._pending: Dict[str, MigrationRequest] = {}
        self._lock = threading.Lock()
    
    def prepare_migration(
        self,
        lid: LogicalEndpointID,
        new_address: str,
    ) -> MigrationRequest:
        """Prepare a migration - bump epoch and notify all endpoints."""
        route = self._route_manager.get_route(lid)
        if not route:
            raise ValueError(f"No route found for {lid.to_key()}")
        
        # Bump epoch
        new_epoch = self._epoch_manager.bump(f"migration:{lid.to_key()}")
        
        # Create migration request
        migration_id = f"mig-{lid.to_key()}-{new_epoch}"
        migration = MigrationRequest(
            migration_id=migration_id,
            lid=lid,
            old_address=route.addresses[0] if route.addresses else "",
            new_address=new_address,
            old_epoch=route.epoch,
            new_epoch=new_epoch,
        )
        
        with self._lock:
            self._pending[migration_id] = migration
        
        # Update route state
        self._route_manager.update_route_state(lid, RouteState.MIGRATING)
        
        # Broadcast PREPARE_MIGRATION
        self._event_bus.publish(Event(
            type=EventType.PREPARE_MIGRATION,
            epoch=new_epoch,
            payload={
                "migration_id": migration_id,
                "lid": lid.to_dict(),
                "old_address": migration.old_address,
                "new_address": new_address,
                "old_epoch": route.epoch,
                "new_epoch": new_epoch,
            },
        ))
        
        logger.info(
            "Prepared migration %s: %s -> %s (epoch %d -> %d)",
            migration_id, migration.old_address, new_address,
            route.epoch, new_epoch
        )
        
        return migration
    
    def notify_shadow_ready(
        self,
        migration_id: str,
        from_lid: LogicalEndpointID,
    ) -> bool:
        """Notify that an endpoint has completed shadow warmup."""
        with self._lock:
            migration = self._pending.get(migration_id)
            if not migration:
                return False
            
            migration.shadows_ready.add(from_lid.to_key())
            
            # Check if all endpoints are ready
            # For now, we just need the involved endpoints
            if migration.state == "pending":
                migration.state = "preparing"
            
            return True
    
    def execute_cutover(self, migration_id: str) -> bool:
        """Execute the cutover for a migration."""
        with self._lock:
            migration = self._pending.get(migration_id)
            if not migration:
                return False
            
            migration.state = "cutting_over"
        
        # Broadcast CUTOVER_START
        self._event_bus.publish(Event(
            type=EventType.CUTOVER_START,
            epoch=migration.new_epoch,
            payload={
                "migration_id": migration_id,
                "lid": migration.lid.to_dict(),
            },
        ))
        
        # Update route with new address
        route = self._route_manager.get_route(migration.lid)
        if route:
            route.addresses = [migration.new_address]
            route.epoch = migration.new_epoch
            route.state = RouteState.ACTIVE
        
        # Complete migration
        with self._lock:
            migration.state = "complete"
            migration.completed_at = time.time()
        
        # Broadcast CUTOVER_COMPLETE
        self._event_bus.publish(Event(
            type=EventType.CUTOVER_COMPLETE,
            epoch=migration.new_epoch,
            payload={
                "migration_id": migration_id,
                "lid": migration.lid.to_dict(),
                "new_address": migration.new_address,
                **({"route": route.to_compact_dict()} if route else {}),
            },
        ))
        
        logger.info("Migration %s completed", migration_id)
        return True
    
    def abort_migration(self, migration_id: str, reason: str) -> bool:
        """Abort a migration."""
        with self._lock:
            migration = self._pending.get(migration_id)
            if not migration:
                return False
            
            migration.state = "failed"
        
        # Restore route state
        self._route_manager.update_route_state(migration.lid, RouteState.ACTIVE)
        
        # Broadcast MIGRATION_FAILED
        self._event_bus.publish(Event(
            type=EventType.MIGRATION_FAILED,
            epoch=migration.new_epoch,
            payload={
                "migration_id": migration_id,
                "lid": migration.lid.to_dict(),
                "reason": reason,
            },
        ))
        
        logger.warning("Migration %s aborted: %s", migration_id, reason)
        return True


# ============================================================================
# Event Bus
# ============================================================================

class EventBus:
    """ZeroMQ-based event distribution."""
    
    def __init__(self, pub_address: str = "tcp://*:15560"):
        self._pub_address = pub_address
        self._context: Optional[zmq.Context] = None
        self._pub_socket: Optional[zmq.Socket] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Event history for late subscribers
        self._history: List[Event] = []
        self._max_history = 1000
        # Publish sequence number (see Event.seq).
        self._seq = 0
    
    @property
    def seq(self) -> int:
        """Sequence number of the last published event."""
        with self._lock:
            return self._seq
    
    def start(self):
        """Start the event bus."""
        self._context = zmq.Context()
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(self._pub_address)
        self._running = True
        
        logger.info("Event bus started on %s", self._pub_address)
    
    def stop(self):
        """Stop the event bus."""
        self._running = False
        
        if self._pub_socket:
            self._pub_socket.close()
            self._pub_socket = None
        
        if self._context:
            self._context.term()
            self._context = None
    
    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if not self._running or not self._pub_socket:
            logger.warning("Event bus not running, dropping event")
            return
        
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            # Store in history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            
            # Serialize and send
            topic = f"connex.{EventType(event.type).name.lower()}"
            data = event.serialize()
            
            self._pub_socket.send_multipart([
                topic.encode("utf-8"),
                data,
            ])
            
            logger.debug("Published event: %s (epoch=%d)", topic, event.epoch)
    
    def get_history(self, since_epoch: int = 0) -> List[Event]:
        """Get events since a given epoch."""
        with self._lock:
            return [e for e in self._history if e.epoch >= since_epoch]


# ============================================================================
# Persistence Backend (Abstract)
# ============================================================================

class PersistenceBackend:
    """Abstract persistence backend."""
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    
    def set(self, key: str, value: Dict[str, Any]):
        raise NotImplementedError
    
    def delete(self, key: str):
        raise NotImplementedError
    
    def get_all(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


class InMemoryPersistence(PersistenceBackend):
    """In-memory persistence for testing."""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        with self._lock:
            self._data[key] = value
    
    def delete(self, key: str):
        with self._lock:
            self._data.pop(key, None)
    
    def get_all(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                k: v for k, v in self._data.items()
                if k.startswith(prefix)
            }


# ============================================================================
# Front Router Service
# ============================================================================

class FrontRouterConfig:
    """Configuration for Front Router service."""
    
    def __init__(
        self,
        api_address: str = "tcp://*:15555",
        event_pub_address: str = "tcp://*:15560",
        persistence_type: str = "memory",  # memory, etcd, redis
        etcd_endpoints: Optional[List[str]] = None,
    ):
        self.api_address = api_address
        self.event_pub_address = event_pub_address
        self.persistence_type = persistence_type
        self.etcd_endpoints = etcd_endpoints or ["localhost:2379"]


class FrontRouter:
    """Front Router - Connex EPC V2 Control Plane Service."""
    
    def __init__(self, config: FrontRouterConfig):
        self.config = config
        
        # Initialize persistence
        if config.persistence_type == "memory":
            self._persistence = InMemoryPersistence()
        else:
            # TODO: Implement etcd/Redis backends
            self._persistence = InMemoryPersistence()
        
        # Core components
        self.route_manager = RouteManager(self._persistence)
        self.epoch_manager = EpochManager(self._persistence)
        self.event_bus = EventBus(config.event_pub_address)
        
        self.member_manager = MemberManager(
            self.route_manager,
            on_unhealthy=self._on_endpoint_unhealthy,
            on_removed=self._on_endpoint_expired,
        )
        
        self.migration_orchestrator = MigrationOrchestrator(
            self.route_manager,
            self.epoch_manager,
            self.event_bus,
        )
        
        # API server
        self._context: Optional[zmq.Context] = None
        self._api_socket: Optional[zmq.Socket] = None
        self._running = False
        self._api_thread: Optional[threading.Thread] = None
        
        # Stats
        self._stats = {
            "registrations": 0,
            "deregistrations": 0,
            "heartbeats": 0,
            "migrations": 0,
            "api_requests": 0,
        }
    
    def start(self):
        """Start the Front Router service."""
        logger.info("Starting Front Router...")
        
        # Start event bus
        self.event_bus.start()
        
        # Start member manager
        self.member_manager.start()
        
        # Start API server
        self._context = zmq.Context()
        self._api_socket = self._context.socket(zmq.REP)
        self._api_socket.bind(self.config.api_address)
        
        self._running = True
        self._api_thread = threading.Thread(
            target=self._api_loop,
            daemon=True,
            name="front-router-api",
        )
        self._api_thread.start()
        
        logger.info(
            "Front Router started (API: %s, Events: %s)",
            self.config.api_address,
            self.config.event_pub_address
        )
    
    def stop(self):
        """Stop the Front Router service."""
        logger.info("Stopping Front Router...")
        
        self._running = False
        
        # Stop member manager
        self.member_manager.stop()
        
        # Stop API server
        if self._api_socket:
            self._api_socket.close()
            self._api_socket = None
        
        if self._context:
            self._context.term()
            self._context = None
        
        # Stop event bus
        self.event_bus.stop()
        
        logger.info("Front Router stopped")
    
    def _api_loop(self):
        """Main API request loop."""
        while self._running:
            try:
                if self._api_socket.poll(timeout=1000):
                    request = self._api_socket.recv_json()
                    response = self._handle_request(request)
                    self._api_socket.send_json(response)
                    self._stats["api_requests"] += 1
            except Exception as e:
                logger.error("API error: %s", e)
                try:
                    self._api_socket.send_json({"error": str(e)})
                except:
                    pass
    
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an API request."""
        method = request.get("method", "")
        params = request.get("params", {})
        
        handlers = {
            "register": self._handle_register,
            "deregister": self._handle_deregister,
            "heartbeat": self._handle_heartbeat,
            "get_routes": self._handle_get_routes,
            "get_stage_routes": self._handle_get_stage_routes,
            "get_route": self._handle_get_route,
            "get_epoch": self._handle_get_epoch,
            "prepare_migration": self._handle_prepare_migration,
            "scale_out": self._handle_scale_out,
            "shadow_ready": self._handle_shadow_ready,
            "execute_cutover": self._handle_execute_cutover,
            "get_stats": self._handle_get_stats,
            "get_topology": self._handle_get_topology,
            "update_route": self._handle_update_route,
            "force_refresh": self._handle_force_refresh,
        }
        
        handler = handlers.get(method)
        if not handler:
            return {"error": f"Unknown method: {method}"}
        
        try:
            return handler(params)
        except Exception as e:
            logger.error("Handler error for %s: %s", method, e)
            return {"error": str(e)}
    
    def _handle_register(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle endpoint registration."""
        lid = LogicalEndpointID.from_dict(params["lid"])
        addresses = params["addresses"]
        capabilities = [TransportCapability(c) for c in params.get("capabilities", [])]
        
        # Check if this is a reconnection
        existing = self.route_manager.get_route(lid)
        is_reconnect = existing is not None
        address_changed = False
        old_addresses = []
        
        if is_reconnect:
            # Use existing epoch, just update address
            epoch = existing.epoch
            old_addresses = existing.addresses.copy()
            # Check if address changed (pod restart with new IP)
            address_changed = set(old_addresses) != set(addresses)
            if address_changed:
                logger.info(
                    "Reconnection from %s with address change: %s -> %s (epoch=%d)",
                    lid.to_key(), old_addresses, addresses, epoch
                )
            else:
                logger.info("Reconnection from %s (epoch=%d)", lid.to_key(), epoch)
        else:
            # New registration - use current epoch
            epoch = self.epoch_manager.current
            logger.info("New registration: %s (epoch=%d)", lid.to_key(), epoch)
        
        # Add/update route
        route = self.route_manager.add_route(lid, addresses, capabilities, epoch)
        
        # Event seq at snapshot time (read before the snapshot; see get_routes)
        snapshot_seq = self.event_bus.seq
        
        # Get full route table for response
        all_routes = self.route_manager.get_routes_for_model(lid.model_name)
        
        # Broadcast registration event
        self.event_bus.publish(Event(
            type=EventType.ENDPOINT_REGISTERED,
            epoch=epoch,
            payload={
                "lid": lid.to_dict(),
                "addresses": addresses,
                "is_reconnect": is_reconnect,
                "address_changed": address_changed,
                "old_addresses": old_addresses if address_changed else [],
                # Additive: full route so clients can update incrementally.
                "route": route.to_compact_dict(),
            },
            source_lid=lid.to_key(),
        ))
        
        # If address changed, also broadcast ROUTE_UPDATED event for upstream/downstream
        # This notifies other stages to refresh their peer connections
        if address_changed:
            self._broadcast_route_updated(
                lid, old_addresses, addresses, epoch, route=route
            )
        
        self._stats["registrations"] += 1
        
        return {
            "success": True,
            "epoch": epoch,
            "is_reconnect": is_reconnect,
            "address_changed": address_changed,
            "route_table": [r.to_dict() for r in all_routes],
            "seq": snapshot_seq,
        }
    
    def _broadcast_route_updated(
        self,
        lid: LogicalEndpointID,
        old_addresses: List[str],
        new_addresses: List[str],
        epoch: int,
        route: Optional[RouteEntry] = None,
    ):
        """Broadcast ROUTE_UPDATED event when an endpoint's address changes.
        
        This is critical for pod recovery scenarios where:
        1. A pod crashes and is restarted with a new IP
        2. Upstream stages need to update their downstream peer address
        3. Downstream stages need to update their upstream peer address
        """
        # force_refresh emits one of these per route; keep no-op updates quiet.
        log = logger.info if old_addresses != new_addresses else logger.debug
        log(
            "Broadcasting route update for %s: %s -> %s",
            lid.to_key(), old_addresses, new_addresses
        )
        
        if route is None:
            route = self.route_manager.get_route(lid)
        payload = {
            "lid": lid.to_dict(),
            "old_addresses": old_addresses,
            "new_addresses": new_addresses,
            "stage_idx": lid.stage_idx,
            "replica_idx": lid.replica_idx,
        }
        if route is not None:
            payload["route"] = route.to_compact_dict()
        self.event_bus.publish(Event(
            type=EventType.ROUTE_UPDATED,
            epoch=epoch,
            payload=payload,
            source_lid=lid.to_key(),
        ))

    def _handle_deregister(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle endpoint deregistration."""
        lid = LogicalEndpointID.from_dict(params["lid"])
        
        route = self.route_manager.remove_route(lid)
        if not route:
            return {"success": False, "error": "Route not found"}
        
        # Broadcast deregistration
        self.event_bus.publish(Event(
            type=EventType.ENDPOINT_DEREGISTERED,
            epoch=self.epoch_manager.current,
            payload={"lid": lid.to_dict()},
            source_lid=lid.to_key(),
        ))
        
        self._stats["deregistrations"] += 1
        
        return {"success": True}
    
    def _handle_heartbeat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heartbeat from endpoint."""
        lid = LogicalEndpointID.from_dict(params["lid"])
        
        success, recovered = self.route_manager.heartbeat(lid)
        
        self._stats["heartbeats"] += 1
        
        if recovered:
            # UNHEALTHY -> ACTIVE used to be silent, leaving event-driven
            # caches stale.  Announce it as a reconnect without address change
            # (existing clients treat that as a benign membership refresh).
            route = self.route_manager.get_route(lid)
            logger.info("Endpoint %s recovered (heartbeat resumed)", lid.to_key())
            if route is not None:
                self.event_bus.publish(Event(
                    type=EventType.ENDPOINT_REGISTERED,
                    epoch=self.epoch_manager.current,
                    payload={
                        "lid": lid.to_dict(),
                        "addresses": list(route.addresses),
                        "is_reconnect": True,
                        "address_changed": False,
                        "old_addresses": [],
                        "recovered": True,
                        "route": route.to_compact_dict(),
                    },
                    source_lid=lid.to_key(),
                ))
        
        return {
            "success": success,
            "epoch": self.epoch_manager.current,
            # Additive: lets clients detect missed events.
            "seq": self.event_bus.seq,
        }
    
    def _handle_get_routes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get routes, optionally filtered by model and stage.

        Optional params (all additive / backward compatible):
        - ``stage_idx``: int or list of ints; requires ``model_name``.
        - ``compact``: if true, omit timestamps/metadata per route.
        The response also carries ``seq`` (event sequence number at snapshot
        time) so clients can resume incremental updates from events.
        """
        # Read seq BEFORE the routes: every event with seq <= this value was
        # published after its mutation, so it is reflected in the snapshot.
        seq = self.event_bus.seq
        model_name = params.get("model_name")
        stage_idx = params.get("stage_idx")
        
        if model_name and stage_idx is not None:
            stages = stage_idx if isinstance(stage_idx, list) else [stage_idx]
            routes = []
            for st in stages:
                routes.extend(self.route_manager.get_routes_for_stage(model_name, int(st)))
        elif model_name:
            routes = self.route_manager.get_routes_for_model(model_name)
        else:
            routes = self.route_manager.get_all_routes()
        
        if params.get("compact"):
            out = [r.to_compact_dict() for r in routes]
        else:
            out = [r.to_dict() for r in routes]
        return {
            "routes": out,
            "epoch": self.epoch_manager.current,
            "seq": seq,
        }
    
    def _handle_get_stage_routes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cheap stage-scoped query: compact routes of one (model, stage)."""
        if params.get("model_name") is None or params.get("stage_idx") is None:
            return {"error": "model_name and stage_idx required"}
        q = dict(params)
        q.setdefault("compact", True)
        return self._handle_get_routes(q)
    
    def _handle_get_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific route with fallback on failure.
        
        Route is fixed by default (worker caches it). Worker only re-requests
        when it detects request failure or high latency and invalidates cache;
        then we return fallback to another active replica if the requested one
        is not found or unhealthy.
        """
        lid = LogicalEndpointID.from_dict(params["lid"])
        allow_fallback = params.get("allow_fallback", True)  # Default to True for backward compat
        
        route = self.route_manager.get_route(lid)
        
        # If exact route found and active, return it
        if route and route.state == RouteState.ACTIVE:
            return {"route": route.to_dict()}
        
        # If fallback is not allowed or route is draining/migrating (sticky), return as-is
        if not allow_fallback:
            if route:
                return {"route": route.to_dict()}
            return {"error": "Route not found"}
        
        # v2.0: Try to find an active replica of the same stage (load balancing)
        # Get all routes for the model
        all_routes = self.route_manager.get_routes_for_model(lid.model_name)
        
        # Filter to same stage, same shard, but different replica (active only)
        fallback_candidates = [
            r for r in all_routes
            if r.lid.stage_idx == lid.stage_idx
            and r.lid.shard_idx == lid.shard_idx
            and r.lid.replica_idx != lid.replica_idx
            and r.state == RouteState.ACTIVE
        ]
        
        if fallback_candidates:
            # Round-robin selection among fallback candidates
            # Use a simple counter based on current time for distribution
            import time
            idx = int(time.time() * 1000) % len(fallback_candidates)
            fallback_route = fallback_candidates[idx]
            
            logger.info(
                "Route fallback: %s (not found/unhealthy) -> %s (replica %d)",
                lid.to_key(), fallback_route.lid.to_key(), fallback_route.lid.replica_idx
            )
            
            return {
                "route": fallback_route.to_dict(),
                "is_fallback": True,
                "original_replica": lid.replica_idx,
                "fallback_replica": fallback_route.lid.replica_idx,
            }
        
        # No fallback available
        if route:
            # Return the original route even if not active (let client decide)
            return {"route": route.to_dict(), "warning": "Route not in ACTIVE state"}
        
        return {"error": f"Route not found for stage {lid.stage_idx} (no fallback available)"}
    
    def _handle_get_epoch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current epoch."""
        return {"epoch": self.epoch_manager.current}
    
    def _handle_prepare_migration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a migration."""
        lid = LogicalEndpointID.from_dict(params["lid"])
        new_address = params["new_address"]
        
        try:
            migration = self.migration_orchestrator.prepare_migration(lid, new_address)
            self._stats["migrations"] += 1
            
            return {
                "success": True,
                "migration_id": migration.migration_id,
                "new_epoch": migration.new_epoch,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_scale_out(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scale-out trigger from scheduler. Prepare migration for lid to new_address.

        Called by an external scheduler when a new pod is placed. Parameters:
        - model_name: Model name
        - stage_idx: Pipeline stage index
        - new_address: Address of new endpoint (e.g. from K8s pod placement)
        - shard_idx: Optional, default 0
        - replica_idx: Optional, default 0
        """
        model_name = params.get("model_name")
        stage_idx = params.get("stage_idx")
        new_address = params.get("new_address")
        shard_idx = params.get("shard_idx", 0)
        replica_idx = params.get("replica_idx", 0)

        if model_name is None or stage_idx is None or new_address is None:
            return {"success": False, "error": "model_name, stage_idx, new_address required"}

        lid = LogicalEndpointID(
            model_name=model_name,
            stage_idx=int(stage_idx),
            shard_idx=int(shard_idx),
            replica_idx=int(replica_idx),
        )

        try:
            migration = self.migration_orchestrator.prepare_migration(lid, new_address)
            self._stats["migrations"] += 1

            return {
                "success": True,
                "migration_id": migration.migration_id,
                "new_epoch": migration.new_epoch,
                "lid": lid.to_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_get_topology(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current topology for scheduler (cache-aware scheduling).

        Returns routes grouped by model for scheduler placement decisions.
        """
        model_name = params.get("model_name")

        if model_name:
            routes = self.route_manager.get_routes_for_model(model_name)
        else:
            routes = self.route_manager.get_all_routes()

        return {
            "routes": [r.to_dict() for r in routes],
            "epoch": self.epoch_manager.current,
        }

    def _handle_shadow_ready(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shadow ready notification."""
        migration_id = params["migration_id"]
        from_lid = LogicalEndpointID.from_dict(params["from_lid"])
        
        success = self.migration_orchestrator.notify_shadow_ready(migration_id, from_lid)
        
        return {"success": success}
    
    def _handle_execute_cutover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute migration cutover."""
        migration_id = params["migration_id"]
        
        success = self.migration_orchestrator.execute_cutover(migration_id)
        
        return {"success": success}
    
    def _handle_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "stats": self._stats,
            "routes_count": len(self.route_manager.get_all_routes()),
            "current_epoch": self.epoch_manager.current,
        }
    
    def _handle_update_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manually update a route's addresses (for pod IP recovery).
        
        This can be called when a pod restarts with a new IP and needs
        to update the route table immediately without full re-registration.
        """
        lid = LogicalEndpointID.from_dict(params["lid"])
        new_addresses = params["addresses"]
        
        # Update route (also refreshes heartbeat and marks ACTIVE)
        res = self.route_manager.update_addresses(lid, new_addresses)
        if res is None:
            return {"success": False, "error": "Route not found"}
        existing, old_addresses = res
        
        logger.info(
            "Route updated manually: %s: %s -> %s",
            lid.to_key(), old_addresses, new_addresses
        )
        
        # Broadcast route update
        self._broadcast_route_updated(
            lid, old_addresses, new_addresses, existing.epoch, route=existing
        )
        
        return {
            "success": True,
            "old_addresses": old_addresses,
            "new_addresses": new_addresses,
        }
    
    def _handle_force_refresh(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Force all clients to refresh their route tables.
        
        This broadcasts a ROUTE_UPDATED event for each active route,
        causing all clients to re-fetch the route table.
        """
        model_name = params.get("model_name")
        
        if model_name:
            routes = self.route_manager.get_routes_for_model(model_name)
        else:
            routes = self.route_manager.get_all_routes()
        
        refreshed = 0
        for route in routes:
            if route.state == RouteState.ACTIVE:
                self._broadcast_route_updated(
                    route.lid,
                    route.addresses,  # old = new (just trigger refresh)
                    route.addresses,
                    route.epoch,
                    route=route,
                )
                refreshed += 1
        
        logger.info("Force refreshed %d routes", refreshed)
        
        return {
            "success": True,
            "refreshed_count": refreshed,
        }
    
    def _on_endpoint_unhealthy(self, lid: LogicalEndpointID):
        """Callback when an endpoint becomes unhealthy."""
        self.event_bus.publish(Event(
            type=EventType.ENDPOINT_UNHEALTHY,
            epoch=self.epoch_manager.current,
            payload={"lid": lid.to_dict()},
        ))
    
    def _on_endpoint_expired(self, lid: LogicalEndpointID):
        """Callback when an UNHEALTHY route is garbage-collected after grace.

        Previously removal was silent, so event-driven caches kept the dead
        route forever.  Announce it as a deregistration.
        """
        self.event_bus.publish(Event(
            type=EventType.ENDPOINT_DEREGISTERED,
            epoch=self.epoch_manager.current,
            payload={"lid": lid.to_dict(), "reason": "expired"},
            source_lid=lid.to_key(),
        ))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for Front Router service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Connex Front Router Service")
    parser.add_argument(
        "--api-address",
        default="tcp://*:15555",
        help="API server address"
    )
    parser.add_argument(
        "--event-address",
        default="tcp://*:15560",
        help="Event bus PUB address"
    )
    parser.add_argument(
        "--persistence",
        default="memory",
        choices=["memory", "etcd", "redis"],
        help="Persistence backend"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Create config
    config = FrontRouterConfig(
        api_address=args.api_address,
        event_pub_address=args.event_address,
        persistence_type=args.persistence,
    )
    
    # Create and start service
    router = FrontRouter(config)
    router.start()
    
    logger.info("Front Router running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        router.stop()


if __name__ == "__main__":
    main()
