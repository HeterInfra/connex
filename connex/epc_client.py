# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Connex EPC V2 Client - Lightweight client for vLLM integration.

This is the client-side library that vLLM instances use to connect to
the Front Router control plane. It provides:
- Registration with the control plane
- Heartbeat maintenance
- Route table subscription
- Migration event handling
- Data plane management (UCX/RDMA)

The client is designed to be lightweight, with all control plane state
managed by the Front Router service.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

import zmq

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default


# Client-side knobs.  Defaults keep historical behaviour (10 s heartbeat,
# 5 s API timeout).  For fast failover use CONNEX_HEARTBEAT_INTERVAL_S=1 on
# clients together with CONNEX_HEARTBEAT_TIMEOUT_S=3 on the Front Router.
ENV_HEARTBEAT_INTERVAL = "CONNEX_HEARTBEAT_INTERVAL_S"
ENV_API_TIMEOUT_MS = "CONNEX_EPC_API_TIMEOUT_MS"
ENV_HEARTBEAT_TIMEOUT_MS = "CONNEX_EPC_HEARTBEAT_TIMEOUT_MS"


# ============================================================================
# Shared Types
# ============================================================================

class RouteState(IntEnum):
    UNKNOWN = 0
    ACTIVE = 1
    DRAINING = 2
    MIGRATING = 3
    UNHEALTHY = 4


class EventType(IntEnum):
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
    ZEROMQ = 0
    UCX_TCP = 1
    UCX_RDMA = 2
    GPU_DIRECT = 3
    MOONCAKE_TE = 4


@dataclass
class LogicalEndpointID:
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
        return {
            "model_name": self.model_name,
            "stage_idx": self.stage_idx,
            "shard_idx": self.shard_idx,
            "replica_idx": self.replica_idx,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogicalEndpointID":
        return cls(**d)


@dataclass
class RouteEntry:
    lid: LogicalEndpointID
    addresses: List[str]
    capabilities: List[TransportCapability]
    epoch: int
    state: RouteState = RouteState.ACTIVE
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteEntry":
        return cls(
            lid=LogicalEndpointID.from_dict(d["lid"]),
            addresses=d["addresses"],
            capabilities=[TransportCapability(c) for c in d["capabilities"]],
            epoch=d["epoch"],
            state=RouteState(d.get("state", RouteState.ACTIVE)),
        )


@dataclass
class Event:
    type: EventType
    epoch: int
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source_lid: Optional[str] = None
    # Router publish sequence number (None with older routers).
    seq: Optional[int] = None
    # Set by the client when the local route cache already reflects this
    # event (applied incrementally or covered by a refresh), so built-in
    # handlers can skip the legacy full refresh_routes().
    cache_synced: bool = False
    
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


# ============================================================================
# EPC V2 Client Configuration
# ============================================================================

@dataclass
class EPCV2ClientConfig:
    """Configuration for EPC V2 Client."""
    # Front Router addresses
    front_router_api: str = "tcp://localhost:15555"
    front_router_events: str = "tcp://localhost:15560"
    
    # Local endpoint identity
    model_name: str = "default"
    stage_idx: int = 0
    shard_idx: int = 0
    replica_idx: int = 0
    
    # Local addresses to advertise
    local_addresses: List[str] = field(default_factory=list)
    capabilities: List[TransportCapability] = field(default_factory=list)
    
    # Heartbeat configuration (env CONNEX_HEARTBEAT_INTERVAL_S overrides the
    # default; an explicit constructor value always wins).
    heartbeat_interval_s: float = field(
        default_factory=lambda: _env_float(ENV_HEARTBEAT_INTERVAL, 10.0)
    )
    
    # Request timeouts.  On timeout the REQ socket is recreated (a REQ socket
    # that timed out waiting for a reply is otherwise stuck in EFSM).
    api_timeout_ms: int = field(
        default_factory=lambda: int(_env_float(ENV_API_TIMEOUT_MS, 5000))
    )
    # None -> min(api_timeout_ms, max(500, heartbeat_interval_s * 1000)).
    heartbeat_timeout_ms: Optional[int] = field(
        default_factory=lambda: (
            int(_env_float(ENV_HEARTBEAT_TIMEOUT_MS, 0)) or None
        )
    )
    
    # Transport configuration
    ucx_enabled: bool = True
    gpu_direct_enabled: bool = False
    mooncake_enabled: bool = False
    
    @property
    def lid(self) -> LogicalEndpointID:
        return LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=self.stage_idx,
            shard_idx=self.shard_idx,
            replica_idx=self.replica_idx,
        )


# ============================================================================
# Route Table Cache
# ============================================================================

class RouteTableCache:
    """Local cache of the route table from Front Router."""
    
    def __init__(self):
        self._routes: Dict[str, RouteEntry] = {}
        self._epoch = 0
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[RouteEntry], None]] = []
    
    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch
    
    def update_from_list(self, routes: List[Dict[str, Any]], epoch: int):
        """Update cache from route list."""
        with self._lock:
            self._routes.clear()
            for r in routes:
                entry = RouteEntry.from_dict(r)
                self._routes[entry.lid.to_key()] = entry
            self._epoch = epoch
    
    def get(self, lid: LogicalEndpointID) -> Optional[RouteEntry]:
        """Get a route from cache."""
        with self._lock:
            return self._routes.get(lid.to_key())
    
    def get_all(self) -> List[RouteEntry]:
        """Get all routes."""
        with self._lock:
            return list(self._routes.values())
    
    def get_for_model(self, model_name: str) -> List[RouteEntry]:
        """Get routes for a specific model."""
        with self._lock:
            return [r for r in self._routes.values() if r.lid.model_name == model_name]
    
    def update_single(self, route: RouteEntry, epoch: int):
        """Update a single route entry."""
        with self._lock:
            self._routes[route.lid.to_key()] = route
            if epoch > self._epoch:
                self._epoch = epoch
            
            # Notify callbacks
            for cb in self._callbacks:
                try:
                    cb(route)
                except Exception as e:
                    logger.error("Route callback error: %s", e)
    
    def remove(self, lid: LogicalEndpointID):
        """Remove a route from cache."""
        with self._lock:
            self._routes.pop(lid.to_key(), None)
    
    def upsert(self, route: RouteEntry, epoch: int = 0):
        """Insert/replace a route without firing callbacks."""
        with self._lock:
            self._routes[route.lid.to_key()] = route
            if epoch > self._epoch:
                self._epoch = epoch
    
    def set_state(self, lid: LogicalEndpointID, state: RouteState) -> bool:
        with self._lock:
            r = self._routes.get(lid.to_key())
            if r is None:
                return False
            r.state = state
            return True
    
    def get_for_stage(self, model_name: str, stage_idx: int) -> List[RouteEntry]:
        """Routes of one stage, sorted by (shard, replica)."""
        with self._lock:
            out = [
                r for r in self._routes.values()
                if r.lid.model_name == model_name and r.lid.stage_idx == stage_idx
            ]
        out.sort(key=lambda r: (r.lid.shard_idx, r.lid.replica_idx))
        return out
    
    def replace_stage(
        self, model_name: str, stage_idx: int, routes: List[RouteEntry], epoch: int
    ):
        """Replace every cached route of one stage with ``routes``."""
        with self._lock:
            for k in [
                k for k, r in self._routes.items()
                if r.lid.model_name == model_name and r.lid.stage_idx == stage_idx
            ]:
                del self._routes[k]
            for r in routes:
                self._routes[r.lid.to_key()] = r
            if epoch > self._epoch:
                self._epoch = epoch
    
    def add_callback(self, callback: Callable[[RouteEntry], None]):
        """Add a callback for route updates."""
        self._callbacks.append(callback)


# ============================================================================
# Data Plane Manager (Stub)
# ============================================================================

class DataPlaneManager:
    """Manages data plane connections (UCX/RDMA)."""
    
    def __init__(self, config: EPCV2ClientConfig):
        self._config = config
        self._connections: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def get_or_create_connection(self, lid: LogicalEndpointID, address: str) -> Any:
        """Get or create a data plane connection."""
        key = f"{lid.to_key()}@{address}"
        
        with self._lock:
            if key not in self._connections:
                # TODO: Create actual UCX/RDMA connection
                self._connections[key] = {
                    "lid": lid,
                    "address": address,
                    "created_at": time.time(),
                }
            return self._connections[key]
    
    def close_connection(self, lid: LogicalEndpointID):
        """Close connections to a specific endpoint."""
        with self._lock:
            keys_to_remove = [
                k for k in self._connections.keys()
                if k.startswith(lid.to_key())
            ]
            for k in keys_to_remove:
                # TODO: Close actual connection
                del self._connections[k]
    
    def close_all(self):
        """Close all data plane connections."""
        with self._lock:
            self._connections.clear()


# ============================================================================
# Shadow Connection Manager
# ============================================================================

class ShadowConnectionManager:
    """Manages shadow connections for hitless migration."""
    
    def __init__(self, data_plane: DataPlaneManager):
        self._data_plane = data_plane
        self._shadows: Dict[str, Dict[str, Any]] = {}  # migration_id -> shadow info
        self._lock = threading.Lock()
    
    def prepare_shadow(
        self,
        migration_id: str,
        lid: LogicalEndpointID,
        new_address: str,
    ) -> bool:
        """Prepare a shadow connection for migration."""
        with self._lock:
            if migration_id in self._shadows:
                return True  # Already prepared
            
            # Create shadow connection
            conn = self._data_plane.get_or_create_connection(lid, new_address)
            
            self._shadows[migration_id] = {
                "lid": lid,
                "new_address": new_address,
                "connection": conn,
                "ready": True,
                "created_at": time.time(),
            }
            
            return True
    
    def activate_shadow(self, migration_id: str) -> bool:
        """Activate shadow connection (replace old connection)."""
        with self._lock:
            shadow = self._shadows.get(migration_id)
            if not shadow:
                return False
            
            # TODO: Swap connections atomically
            del self._shadows[migration_id]
            return True
    
    def discard_shadow(self, migration_id: str):
        """Discard a shadow connection."""
        with self._lock:
            shadow = self._shadows.pop(migration_id, None)
            if shadow:
                # TODO: Close shadow connection
                pass


# ============================================================================
# EPC V2 Client
# ============================================================================

class APITimeoutError(RuntimeError):
    """A Front Router request did not get a reply in time."""


class _ReqChannel:
    """A thread-safe, self-healing REQ channel to the Front Router.

    ZMQ REQ sockets are not thread-safe and enforce strict send/recv
    alternation: once a recv times out the socket is stuck (EFSM) and every
    later call fails.  This wrapper serializes calls with a lock and, on any
    timeout or ZMQ error, closes the socket (LINGER=0) and lazily creates a
    fresh one, so one slow reply cannot wedge the channel.
    """

    def __init__(self, context: zmq.Context, endpoint: str, timeout_ms: int, name: str):
        self._context = context
        self._endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._name = name
        self._lock = threading.Lock()
        self._socket: Optional[zmq.Socket] = None
        self._closed = False
        self.resets = 0  # number of socket recreations (for tests/metrics)

    def _ensure(self) -> zmq.Socket:
        if self._socket is None:
            sock = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.connect(self._endpoint)
            self._socket = sock
        return self._socket

    def _reset_locked(self):
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
            self.resets += 1

    def call(self, request: Dict[str, Any], timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        with self._lock:
            if self._closed:
                raise RuntimeError("Not connected to Front Router")
            sock = self._ensure()
            try:
                sock.send_json(request)
                if not sock.poll(timeout, zmq.POLLIN):
                    self._reset_locked()
                    raise APITimeoutError(
                        f"{self._name}: no reply to {request.get('method')!r} "
                        f"within {timeout} ms"
                    )
                return sock.recv_json()
            except APITimeoutError:
                raise
            except zmq.ZMQError as e:
                self._reset_locked()
                raise RuntimeError(f"{self._name}: ZMQ error: {e}") from e

    def close(self):
        with self._lock:
            self._closed = True
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
                self._socket = None


class EPCV2Client:
    """
    Lightweight EPC V2 Client for vLLM integration.
    
    Connects to Front Router for control plane operations while
    maintaining data plane connections locally.

    Threading model: API calls go through a locked, self-healing REQ
    channel; heartbeats use a *separate* channel so a slow route query can
    never delay them.  The local route cache is kept current from events
    (incremental upserts/removals); a full refresh is only needed when the
    event sequence (``Event.seq``) shows a gap.  Use
    :meth:`get_stage_peers` for cheap per-stage lookups.
    """
    
    def __init__(self, config: EPCV2ClientConfig):
        self.config = config
        self.lid = config.lid
        
        # ZMQ context and channels
        self._context: Optional[zmq.Context] = None
        self._api: Optional[_ReqChannel] = None
        self._hb: Optional[_ReqChannel] = None
        self._sub_socket: Optional[zmq.Socket] = None
        
        # Route table cache
        self.route_table = RouteTableCache()
        
        # Data plane
        self._data_plane = DataPlaneManager(config)
        self._shadow_manager = ShadowConnectionManager(self._data_plane)
        
        # State
        self._registered = False
        self._running = False
        self._local_epoch = 0
        self._stop_event = threading.Event()
        
        # Event-sequence tracking for incremental cache maintenance.
        # _last_seq is None until we know which event the cache reflects.
        self._sync_lock = threading.RLock()
        self._last_seq: Optional[int] = None
        self._needs_full_refresh = False
        self._hb_seen_seq: Optional[int] = None  # router seq seen on previous heartbeat
        self._stage_query_supported = True
        self.stats: Dict[str, int] = {
            "full_refreshes": 0,
            "stage_queries": 0,
            "incremental_applied": 0,
            "seq_gaps": 0,
            "heartbeat_failures": 0,
        }
        
        # Threads
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._event_thread: Optional[threading.Thread] = None
        
        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        
        # Migration state
        self._pending_migrations: Dict[str, Dict[str, Any]] = {}
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def _heartbeat_timeout_ms(self) -> int:
        if self.config.heartbeat_timeout_ms:
            return int(self.config.heartbeat_timeout_ms)
        return int(min(
            self.config.api_timeout_ms,
            max(500, self.config.heartbeat_interval_s * 1000),
        ))
    
    def start(self) -> bool:
        """Start the client and register with Front Router."""
        try:
            # Initialize ZMQ
            self._context = zmq.Context()
            self._stop_event.clear()
            
            # API channel (shared by register/refresh/etc., serialized) and a
            # dedicated heartbeat channel.
            self._api = _ReqChannel(
                self._context, self.config.front_router_api,
                self.config.api_timeout_ms, "epc-api",
            )
            self._hb = _ReqChannel(
                self._context, self.config.front_router_api,
                self._heartbeat_timeout_ms(), "epc-heartbeat",
            )
            
            # Connect event subscriber
            self._sub_socket = self._context.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.LINGER, 0)
            self._sub_socket.connect(self.config.front_router_events)
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "connex.")
            
            self._running = True
            
            # Register with Front Router
            if not self._register():
                logger.error("Failed to register with Front Router")
                self.stop()
                return False
            
            # Start event subscriber thread
            self._event_thread = threading.Thread(
                target=self._event_loop,
                daemon=True,
                name=f"epc-events-{self.lid.stage_idx}",
            )
            self._event_thread.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"epc-heartbeat-{self.lid.stage_idx}",
            )
            self._heartbeat_thread.start()
            
            logger.info("EPC V2 Client started: %s", self.lid.to_key())
            return True
            
        except Exception as e:
            logger.error("Failed to start EPC V2 Client: %s", e)
            self.stop()
            return False
    
    def stop(self):
        """Stop the client and deregister."""
        logger.info("Stopping EPC V2 Client: %s", self.lid.to_key())
        
        self._running = False
        self._stop_event.set()
        
        # Let background threads exit before closing the sockets they use.
        me = threading.current_thread()
        for t in (self._heartbeat_thread, self._event_thread):
            if t is not None and t is not me and t.is_alive():
                t.join(timeout=2.0)
        self._heartbeat_thread = None
        self._event_thread = None
        
        # Deregister
        if self._registered and self._api is not None:
            try:
                self._deregister()
            except Exception as e:
                logger.warning("Deregister error: %s", e)
        
        # Close sockets
        if self._sub_socket:
            self._sub_socket.close(linger=0)
            self._sub_socket = None
        
        for ch in (self._api, self._hb):
            if ch is not None:
                ch.close()
        self._api = None
        self._hb = None
        
        if self._context:
            self._context.term()
            self._context = None
        
        # Close data plane
        self._data_plane.close_all()
        
        self._registered = False
    
    # -------------------------------------------------------------------------
    # Control Plane API
    # -------------------------------------------------------------------------
    
    def _call_api(
        self,
        method: str,
        params: Dict[str, Any],
        channel: Optional[_ReqChannel] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make an API call to Front Router (thread-safe, timeout-safe)."""
        ch = channel if channel is not None else self._api
        if ch is None:
            raise RuntimeError("Not connected to Front Router")
        
        response = ch.call({"method": method, "params": params}, timeout_ms)
        
        if "error" in response:
            raise RuntimeError(f"API error: {response['error']}")
        
        return response
    
    def _register(self, channel: Optional[_ReqChannel] = None) -> bool:
        """Register with Front Router."""
        try:
            response = self._call_api("register", {
                "lid": self.lid.to_dict(),
                "addresses": self.config.local_addresses,
                "capabilities": [int(c) for c in self.config.capabilities],
            }, channel=channel)
            
            if response.get("success"):
                self._local_epoch = response.get("epoch", 0)
                self._registered = True
                
                # Update route table
                route_table = response.get("route_table", [])
                self._apply_snapshot(route_table, self._local_epoch, response.get("seq"))
                
                is_reconnect = response.get("is_reconnect", False)
                logger.info(
                    "Registered with Front Router (epoch=%d, reconnect=%s, routes=%d)",
                    self._local_epoch, is_reconnect, len(route_table)
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("Registration failed: %s", e)
            return False
    
    def _deregister(self):
        """Deregister from Front Router."""
        try:
            self._call_api("deregister", {"lid": self.lid.to_dict()})
            self._registered = False
        except Exception as e:
            logger.warning("Deregistration failed: %s", e)
    
    def _send_heartbeat(self) -> bool:
        """Send heartbeat to Front Router (on the dedicated heartbeat channel)."""
        try:
            response = self._call_api(
                "heartbeat", {"lid": self.lid.to_dict()},
                channel=self._hb if self._hb is not None else None,
            )
            
            # Check for epoch bump
            server_epoch = response.get("epoch", 0)
            if server_epoch > self._local_epoch:
                logger.info(
                    "Epoch advanced: %d -> %d",
                    self._local_epoch, server_epoch
                )
                self._local_epoch = server_epoch
            
            self._check_seq_from_heartbeat(response.get("seq"))
            
            ok = response.get("success", False)
            if not ok and self._running:
                # The router no longer knows us (e.g. router restart with
                # in-memory state, or our route was garbage-collected):
                # re-register so peers can find us again.
                logger.warning("Heartbeat rejected (unknown endpoint); re-registering")
                if self._register(channel=self._hb):
                    return True
            return ok
            
        except Exception as e:
            self.stats["heartbeat_failures"] += 1
            logger.warning("Heartbeat failed: %s", e)
            return False
    
    def _check_seq_from_heartbeat(self, server_seq: Optional[int]):
        """Detect silently missed events (e.g. PUB/SUB slow joiner).

        If the router's event seq seen on the previous heartbeat is still
        ahead of what this client has processed, events were lost: schedule
        a full refresh.  (One heartbeat of slack absorbs in-flight events.)
        """
        if server_seq is None:
            return
        with self._sync_lock:
            prev = self._hb_seen_seq
            self._hb_seen_seq = server_seq
            if prev is not None and (self._last_seq is None or self._last_seq < prev):
                if not self._needs_full_refresh:
                    logger.info(
                        "Event seq lag detected (processed=%s, router=%s); "
                        "scheduling route refresh", self._last_seq, prev,
                    )
                self._needs_full_refresh = True
    
    def _apply_snapshot(
        self, routes: List[Dict[str, Any]], epoch: int, seq: Optional[int]
    ) -> bool:
        """Install a full (model-scoped) snapshot; returns False if stale."""
        with self._sync_lock:
            if seq is not None and self._last_seq is not None and seq < self._last_seq \
                    and not self._needs_full_refresh:
                # Cache already reflects newer events than this snapshot.
                return False
            self.route_table.update_from_list(routes, epoch)
            if seq is not None:
                self._last_seq = seq
                self._needs_full_refresh = False
            return True
    
    def refresh_routes(self) -> List[RouteEntry]:
        """Refresh route table from Front Router (full, model-scoped)."""
        try:
            response = self._call_api("get_routes", {
                "model_name": self.lid.model_name
            })
            
            routes = response.get("routes", [])
            epoch = response.get("epoch", self._local_epoch)
            
            self._apply_snapshot(routes, epoch, response.get("seq"))
            self._local_epoch = max(self._local_epoch, epoch)
            self.stats["full_refreshes"] += 1
            
            return self.route_table.get_all()
            
        except Exception as e:
            logger.error("Failed to refresh routes: %s", e)
            return []
    
    def cache_in_sync(self) -> bool:
        """True if the local cache is maintained incrementally with no known gap."""
        with self._sync_lock:
            return (
                self._last_seq is not None
                and not self._needs_full_refresh
                and self._event_thread is not None
                and self._event_thread.is_alive()
            )
    
    def query_stage_routes(self, stage_idx: int) -> Optional[List[RouteEntry]]:
        """Fetch one stage's routes from the router (cheap) and update the cache.

        Returns None if the query failed.
        """
        model = self.lid.model_name
        try:
            if self._stage_query_supported:
                try:
                    response = self._call_api("get_stage_routes", {
                        "model_name": model, "stage_idx": int(stage_idx),
                    })
                except RuntimeError as e:
                    if "Unknown method" not in str(e):
                        raise
                    # Older router: fall back to a full refresh.
                    self._stage_query_supported = False
                    response = None
            else:
                response = None
            if response is None:
                self.refresh_routes()
                return self.route_table.get_for_stage(model, int(stage_idx))
            self.stats["stage_queries"] += 1
            entries = [RouteEntry.from_dict(r) for r in response.get("routes", [])]
            epoch = response.get("epoch", self._local_epoch)
            seq = response.get("seq")
            with self._sync_lock:
                if not (seq is not None and self._last_seq is not None
                        and seq < self._last_seq and not self._needs_full_refresh):
                    self.route_table.replace_stage(model, int(stage_idx), entries, epoch)
            entries.sort(key=lambda r: (r.lid.shard_idx, r.lid.replica_idx))
            return entries
        except Exception as e:
            logger.warning("Stage %s route query failed: %s", stage_idx, e)
            return None
    
    def get_stage_peers(self, stage_idx: int, refresh: str = "auto") -> List[RouteEntry]:
        """Routes of ``stage_idx`` for this model, sorted by (shard, replica).

        refresh:
          - "never":  local cache only (no RPC).
          - "auto":   local cache if it is event-synced with no known gap,
                      otherwise a cheap stage-scoped query.
          - "always": always a stage-scoped query.
        On query failure the cached entries are returned.
        """
        if refresh not in ("never", "auto", "always"):
            raise ValueError(f"invalid refresh mode: {refresh!r}")
        if refresh == "always" or (refresh == "auto" and not self.cache_in_sync()):
            res = self.query_stage_routes(stage_idx)
            if res is not None:
                return res
        return self.route_table.get_for_stage(self.lid.model_name, int(stage_idx))
    
    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------
    
    def _event_loop(self):
        """Event subscriber loop."""
        last_retry = 0.0
        while self._running:
            try:
                if self._sub_socket.poll(timeout=200):
                    topic, data = self._sub_socket.recv_multipart()
                    event = Event.deserialize(data)
                    self._handle_event(event)
                elif (self._needs_full_refresh and self._running
                      and time.monotonic() - last_retry > 1.0):
                    # Gap flagged by the heartbeat path: resync off the hot path.
                    last_retry = time.monotonic()
                    self.refresh_routes()
            except Exception as e:
                if self._running:
                    logger.error("Event loop error: %s", e)
    
    def _sync_cache_from_event(self, event: Event) -> bool:
        """Keep the route cache current from one event.

        Returns True if the cache reflects the event afterwards (so the
        legacy full refresh can be skipped).
        """
        if event.seq is None:
            return False  # older router: legacy refresh-per-event behaviour
        with self._sync_lock:
            last = self._last_seq
            if last is not None and event.seq <= last and not self._needs_full_refresh:
                return True  # already covered by a newer snapshot
            gap = last is None or self._needs_full_refresh or event.seq != last + 1
        if gap:
            with self._sync_lock:
                self.stats["seq_gaps"] += 1
                self._needs_full_refresh = True
            logger.debug("Event seq gap (last=%s, got=%s): full refresh", last, event.seq)
            self.refresh_routes()
            with self._sync_lock:
                if self._needs_full_refresh or self._last_seq is None:
                    return False  # refresh failed
                if event.seq <= self._last_seq:
                    return True
                if event.seq != self._last_seq + 1:
                    self._needs_full_refresh = True
                    return False
                # Snapshot predates this event: fall through and apply it.
        with self._sync_lock:
            if self._apply_event_locked(event):
                self._last_seq = event.seq
                self.stats["incremental_applied"] += 1
                return True
            self._needs_full_refresh = True
            return False
    
    def _apply_event_locked(self, event: Event) -> bool:
        """Apply one event to the cache. Returns False if it cannot be applied."""
        p = event.payload or {}
        lid_d = p.get("lid")
        if lid_d is None:
            return True  # nothing route-related
        lid = LogicalEndpointID.from_dict(lid_d)
        if lid.model_name != self.lid.model_name:
            return True  # cache is model-scoped
        t = event.type
        if t in (EventType.ENDPOINT_REGISTERED, EventType.ROUTE_UPDATED,
                 EventType.CUTOVER_COMPLETE):
            if "route" not in p:
                return False
            self.route_table.upsert(RouteEntry.from_dict(p["route"]), event.epoch)
        elif t == EventType.ENDPOINT_DEREGISTERED:
            self.route_table.remove(lid)
        elif t == EventType.ENDPOINT_UNHEALTHY:
            self.route_table.set_state(lid, RouteState.UNHEALTHY)
        elif t == EventType.PREPARE_MIGRATION:
            self.route_table.set_state(lid, RouteState.MIGRATING)
        elif t == EventType.MIGRATION_FAILED:
            self.route_table.set_state(lid, RouteState.ACTIVE)
        return True
    
    def _handle_event(self, event: Event):
        """Handle a control plane event."""
        logger.debug("Received event: %s (epoch=%d)", event.type.name, event.epoch)
        
        # Update local epoch
        if event.epoch > self._local_epoch:
            self._local_epoch = event.epoch
        
        # Keep the route cache in sync (also for our own events, so the
        # sequence stays contiguous).
        try:
            event.cache_synced = self._sync_cache_from_event(event)
        except Exception as e:
            logger.error("Route cache sync error: %s", e)
            event.cache_synced = False
        
        # Skip events from ourselves
        if event.source_lid == self.lid.to_key():
            return
        
        # Handle event
        handlers = {
            EventType.ENDPOINT_REGISTERED: self._on_endpoint_registered,
            EventType.ENDPOINT_DEREGISTERED: self._on_endpoint_deregistered,
            EventType.ENDPOINT_UNHEALTHY: self._on_endpoint_unhealthy,
            EventType.PREPARE_MIGRATION: self._on_prepare_migration,
            EventType.CUTOVER_START: self._on_cutover_start,
            EventType.CUTOVER_COMPLETE: self._on_cutover_complete,
            EventType.MIGRATION_FAILED: self._on_migration_failed,
            EventType.ROUTE_UPDATED: self._on_route_updated,
        }
        
        handler = handlers.get(event.type)
        if handler:
            try:
                handler(event)
            except Exception as e:
                logger.error("Event handler error: %s", e)
        
        # Call registered callbacks
        callbacks = self._event_handlers.get(event.type, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Event callback error: %s", e)
    
    def _on_endpoint_registered(self, event: Event):
        """Handle endpoint registration event."""
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        address_changed = event.payload.get("address_changed", False)
        
        (logger.info if address_changed else logger.debug)(
            "Endpoint registered: %s (address_changed=%s)",
            lid.to_key(), address_changed
        )
        
        # Refresh route table (only if the event could not be applied locally)
        if not event.cache_synced:
            self.refresh_routes()
        
        # If address changed, close old connections so they get refreshed
        if address_changed:
            self._data_plane.close_connection(lid)
    
    def _on_endpoint_deregistered(self, event: Event):
        """Handle endpoint deregistration event."""
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        
        logger.info("Endpoint deregistered: %s", lid.to_key())
        
        # Remove from cache
        self.route_table.remove(lid)
        
        # Close data plane connections
        self._data_plane.close_connection(lid)
    
    def _on_endpoint_unhealthy(self, event: Event):
        """Handle endpoint unhealthy event."""
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        logger.warning("Endpoint unhealthy: %s", lid.to_key())
    
    def _on_route_updated(self, event: Event):
        """Handle route updated event (pod IP change after recovery).
        
        This is critical for handling pod crash recovery scenarios:
        1. Close existing data plane connections to the old IP
        2. Refresh route table to get new IP (skipped when the event carried
           the new route and was applied to the cache incrementally)
        3. Next communication will use the new IP automatically
        """
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        old_addresses = event.payload.get("old_addresses", [])
        new_addresses = event.payload.get("new_addresses", [])
        
        # force_refresh emits one no-op update per route: keep those quiet.
        log = logger.info if old_addresses != new_addresses else logger.debug
        log(
            "Route updated for %s: %s -> %s (my stage=%d, their stage=%d)",
            lid.to_key(), old_addresses, new_addresses,
            self.lid.stage_idx, lid.stage_idx
        )
        
        # Check if this affects us (upstream or downstream peer)
        is_upstream = lid.stage_idx == self.lid.stage_idx - 1
        is_downstream = lid.stage_idx == self.lid.stage_idx + 1
        is_same_stage_diff_replica = (
            lid.stage_idx == self.lid.stage_idx and 
            lid.replica_idx != self.lid.replica_idx
        )
        
        if is_upstream or is_downstream or is_same_stage_diff_replica:
            log(
                "Closing stale connection to %s (upstream=%s, downstream=%s)",
                lid.to_key(), is_upstream, is_downstream
            )
            # Close existing connections - they use old IP
            self._data_plane.close_connection(lid)
        
        # Refresh route table to get updated addresses
        if not event.cache_synced:
            self.refresh_routes()

    def _on_prepare_migration(self, event: Event):
        """Handle migration preparation event."""
        migration_id = event.payload["migration_id"]
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        new_address = event.payload["new_address"]
        
        logger.info(
            "Preparing migration %s: %s -> %s",
            migration_id, lid.to_key(), new_address
        )
        
        # Store migration info
        self._pending_migrations[migration_id] = {
            "lid": lid,
            "new_address": new_address,
            "state": "preparing",
        }
        
        # Prepare shadow connection if this is a peer we communicate with
        if lid.model_name == self.lid.model_name:
            self._shadow_manager.prepare_shadow(migration_id, lid, new_address)
            
            # Notify Front Router that shadow is ready
            try:
                self._call_api("shadow_ready", {
                    "migration_id": migration_id,
                    "from_lid": self.lid.to_dict(),
                })
            except Exception as e:
                logger.error("Failed to notify shadow ready: %s", e)
    
    def _on_cutover_start(self, event: Event):
        """Handle cutover start event."""
        migration_id = event.payload["migration_id"]
        logger.info("Cutover starting: %s", migration_id)
        
        # Mark migration as cutting over
        if migration_id in self._pending_migrations:
            self._pending_migrations[migration_id]["state"] = "cutting_over"
    
    def _on_cutover_complete(self, event: Event):
        """Handle cutover complete event."""
        migration_id = event.payload["migration_id"]
        lid = LogicalEndpointID.from_dict(event.payload["lid"])
        new_address = event.payload["new_address"]
        
        logger.info("Cutover complete: %s -> %s", lid.to_key(), new_address)
        
        # Activate shadow connection
        self._shadow_manager.activate_shadow(migration_id)
        
        # Update route table
        if not event.cache_synced:
            self.refresh_routes()
        
        # Clean up
        self._pending_migrations.pop(migration_id, None)
    
    def _on_migration_failed(self, event: Event):
        """Handle migration failure event."""
        migration_id = event.payload["migration_id"]
        reason = event.payload.get("reason", "unknown")
        
        logger.warning("Migration failed: %s (%s)", migration_id, reason)
        
        # Discard shadow
        self._shadow_manager.discard_shadow(migration_id)
        
        # Clean up
        self._pending_migrations.pop(migration_id, None)
    
    def on_event(self, event_type: EventType, callback: Callable[[Event], None]):
        """Register a callback for an event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(callback)
    
    # -------------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------------
    
    def _heartbeat_loop(self):
        """Heartbeat loop (fixed-rate, on its own REQ channel)."""
        interval = max(0.01, float(self.config.heartbeat_interval_s))
        next_t = time.monotonic()
        while self._running:
            if self._registered:
                self._send_heartbeat()
            next_t += interval
            delay = next_t - time.monotonic()
            if delay < 0:  # fell behind (e.g. a timed-out heartbeat): resync
                next_t = time.monotonic()
                delay = 0
            if self._stop_event.wait(delay):
                break
    
    # -------------------------------------------------------------------------
    # Data Plane Access
    # -------------------------------------------------------------------------
    
    def get_peer_address(self, lid: LogicalEndpointID) -> Optional[str]:
        """Get the address for a peer endpoint."""
        route = self.route_table.get(lid)
        if route and route.addresses:
            return route.addresses[0]
        return None
    
    def get_pipeline_peers(self) -> List[RouteEntry]:
        """Get all pipeline peers for the same model."""
        return self.route_table.get_for_model(self.lid.model_name)
    
    def get_next_stage(self) -> Optional[RouteEntry]:
        """Get the next stage in the pipeline."""
        next_stage_idx = self.lid.stage_idx + 1
        for route in self.get_pipeline_peers():
            if route.lid.stage_idx == next_stage_idx:
                return route
        return None
    
    def get_prev_stage(self) -> Optional[RouteEntry]:
        """Get the previous stage in the pipeline."""
        prev_stage_idx = self.lid.stage_idx - 1
        if prev_stage_idx < 0:
            return None
        for route in self.get_pipeline_peers():
            if route.lid.stage_idx == prev_stage_idx:
                return route
        return None
    
    def invalidate_peer_connection(self, lid: LogicalEndpointID):
        """Invalidate cached connection to a peer (force reconnection).
        
        Call this when you detect a communication error with a peer,
        to force the data plane to re-establish the connection.
        """
        logger.info("Invalidating connection to %s", lid.to_key())
        self._data_plane.close_connection(lid)
        # Also refresh routes to get latest addresses
        self.refresh_routes()
    
    def update_my_address(self, new_addresses: List[str]) -> bool:
        """Update this endpoint's advertised addresses (for IP change).
        
        Call this when the local IP changes (e.g., after pod restart)
        to update the route table and notify peers.
        """
        try:
            response = self._call_api("update_route", {
                "lid": self.lid.to_dict(),
                "addresses": new_addresses,
            })
            
            if response.get("success"):
                self.config.local_addresses = new_addresses
                logger.info(
                    "Updated my address: %s -> %s",
                    response.get("old_addresses"),
                    response.get("new_addresses")
                )
                return True
            
            logger.error("Failed to update address: %s", response.get("error"))
            return False
            
        except Exception as e:
            logger.error("Failed to update address: %s", e)
            return False


# ============================================================================
# Factory Function
# ============================================================================

def create_epc_client(
    model_name: str,
    stage_idx: int,
    front_router_api: str = "tcp://localhost:15555",
    front_router_events: str = "tcp://localhost:15560",
    local_addresses: Optional[List[str]] = None,
    **kwargs,
) -> EPCV2Client:
    """
    Create an EPC V2 Client.
    
    Args:
        model_name: Name of the model being served
        stage_idx: Pipeline stage index
        front_router_api: Front Router API address
        front_router_events: Front Router event bus address
        local_addresses: Local addresses to advertise
        **kwargs: Additional config options
    
    Returns:
        Configured EPCV2Client instance
    """
    config = EPCV2ClientConfig(
        front_router_api=front_router_api,
        front_router_events=front_router_events,
        model_name=model_name,
        stage_idx=stage_idx,
        local_addresses=local_addresses or [],
        **kwargs,
    )
    
    return EPCV2Client(config)
