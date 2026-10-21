# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Mobility Control Plane: Versioned Routing and Shadow Connections.

Implements the control plane for endpoint mobility:
- Versioned routing: LID -> {addr, caps, epoch}
- Shadow warmup / dual-path cutover
- Stale-route fast feedback

Benefits:
- Migration becomes local switching instead of global rebuild
- Controllable switch window
- Significantly reduced tail latency spikes

"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from connex.protocol import (
    LogicalEndpointID,
    MessageClass,
    MessageType,
    MobilityMessage,
    ProtocolHeader,
    StreamID,
    create_stale_epoch_feedback,
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection lifecycle states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    WARMING = auto()      # Shadow connection warming up
    READY = auto()        # Ready for traffic
    DRAINING = auto()     # Draining in-flight traffic
    CLOSING = auto()


class RouteState(Enum):
    """Route entry states."""
    ACTIVE = auto()       # Primary route, accepting traffic
    SHADOW = auto()       # Shadow route, warming up
    DUAL = auto()         # Dual-path mode during cutover
    DRAINING = auto()     # Old route, draining
    STALE = auto()        # Marked stale, pending cleanup


@dataclass
class ConnectionCapability:
    """Transport capability information."""
    supports_rdma: bool = False
    supports_gpu_direct: bool = False
    supports_host_staging: bool = False
    supports_proxy: bool = False
    registered_memory_available: bool = False
    max_message_size: int = 64 * 1024 * 1024  # 64MB
    bandwidth_gbps: float = 100.0
    
    def to_dict(self) -> dict:
        return {
            "rdma": self.supports_rdma,
            "gpu_direct": self.supports_gpu_direct,
            "host_staging": self.supports_host_staging,
            "proxy": self.supports_proxy,
            "registered_mem": self.registered_memory_available,
            "max_msg_size": self.max_message_size,
            "bandwidth_gbps": self.bandwidth_gbps,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ConnectionCapability":
        return cls(
            supports_rdma=d.get("rdma", False),
            supports_gpu_direct=d.get("gpu_direct", False),
            supports_host_staging=d.get("host_staging", False),
            supports_proxy=d.get("proxy", False),
            registered_memory_available=d.get("registered_mem", False),
            max_message_size=d.get("max_msg_size", 64 * 1024 * 1024),
            bandwidth_gbps=d.get("bandwidth_gbps", 100.0),
        )


@dataclass
class VersionedRoute:
    """A versioned route entry.
    
    Supports:
    - Epoch-based versioning for safe cutover
    - Multiple addresses for failover
    - Capability negotiation
    - Shadow connection warmup
    """
    # Identity
    lid: LogicalEndpointID
    
    # Physical location
    addresses: List[str]
    capabilities: ConnectionCapability
    
    # Versioning
    epoch: int = 0
    
    # State
    state: RouteState = RouteState.ACTIVE
    
    # Connection handle
    connection: Any = None
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    
    # Metrics
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    # RTT tracking for congestion sensing
    rtt_samples: List[float] = field(default_factory=list)
    rtt_smoothed_us: float = 0.0
    rtt_variance_us: float = 0.0
    
    # Shadow warmup state
    warmup_complete: bool = False
    warmup_start_time: float = 0.0
    
    @property
    def primary_address(self) -> str:
        return self.addresses[0] if self.addresses else ""
    
    @property
    def is_usable(self) -> bool:
        """Check if route can accept traffic."""
        return (
            self.state in (RouteState.ACTIVE, RouteState.DUAL, RouteState.SHADOW) and
            self.connection_state in (ConnectionState.CONNECTED, ConnectionState.READY)
        )
    
    @property
    def is_warming(self) -> bool:
        """Check if route is in warmup phase."""
        return self.state == RouteState.SHADOW and not self.warmup_complete
    
    def update_rtt(self, rtt_us: float) -> None:
        """Update RTT estimate using exponential smoothing."""
        alpha = 0.125  # EWMA factor
        beta = 0.25    # Variance factor
        
        if self.rtt_smoothed_us == 0:
            self.rtt_smoothed_us = rtt_us
            self.rtt_variance_us = rtt_us / 2
        else:
            diff = abs(rtt_us - self.rtt_smoothed_us)
            self.rtt_variance_us = (1 - beta) * self.rtt_variance_us + beta * diff
            self.rtt_smoothed_us = (1 - alpha) * self.rtt_smoothed_us + alpha * rtt_us
        
        # Keep last 100 samples
        self.rtt_samples.append(rtt_us)
        if len(self.rtt_samples) > 100:
            self.rtt_samples = self.rtt_samples[-100:]


class VersionedRouteTable:
    """Thread-safe versioned route table.
    
    Features:
    - Epoch-based versioning for safe cutover
    - Shadow connection support for warmup
    - Dual-path mode during migration
    - Stale route detection and feedback
    - Fast route refresh on stale detection
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        on_stale_route: Optional[Callable[[LogicalEndpointID, int, int], None]] = None,
        on_route_change: Optional[Callable[[LogicalEndpointID, VersionedRoute, str], None]] = None,
    ):
        """Initialize versioned route table.
        
        Args:
            local_lid: Local endpoint LID
            on_stale_route: Callback when stale route detected (lid, stale_epoch, current_epoch)
            on_route_change: Callback when route changes (lid, route, event_type)
        """
        self.local_lid = local_lid
        self.on_stale_route = on_stale_route
        self.on_route_change = on_route_change
        
        # Routes: LID -> List[VersionedRoute] (multiple paths for failover/dual-path)
        self._routes: Dict[LogicalEndpointID, List[VersionedRoute]] = {}
        self._lock = threading.RLock()
        
        # Global epoch counter
        self._epoch = 0
        
        # Stale epoch tracking: LID -> set of stale epochs seen
        self._stale_epochs: Dict[LogicalEndpointID, Set[int]] = defaultdict(set)
        
        # Route refresh pending
        self._refresh_pending: Set[LogicalEndpointID] = set()
        
        logger.info("VersionedRouteTable initialized for %s", local_lid)
    
    @property
    def epoch(self) -> int:
        """Current global epoch."""
        return self._epoch
    
    def increment_epoch(self) -> int:
        """Atomically increment global epoch."""
        with self._lock:
            self._epoch += 1
            logger.info("Global epoch incremented to %d", self._epoch)
            return self._epoch
    
    def add_route(
        self,
        lid: LogicalEndpointID,
        addresses: List[str],
        capabilities: ConnectionCapability,
        epoch: Optional[int] = None,
        as_shadow: bool = False,
    ) -> VersionedRoute:
        """Add a new route.
        
        Args:
            lid: Target logical endpoint ID
            addresses: Physical addresses
            capabilities: Transport capabilities
            epoch: Route epoch (uses current if not specified)
            as_shadow: Add as shadow route for warmup
            
        Returns:
            Created route entry
        """
        if epoch is None:
            epoch = self._epoch
        
        route = VersionedRoute(
            lid=lid,
            addresses=addresses,
            capabilities=capabilities,
            epoch=epoch,
            state=RouteState.SHADOW if as_shadow else RouteState.ACTIVE,
        )
        
        if as_shadow:
            route.warmup_start_time = time.time()
        
        with self._lock:
            if lid not in self._routes:
                self._routes[lid] = []
            
            # Check for duplicate epoch
            existing = [r for r in self._routes[lid] if r.epoch == epoch]
            if existing:
                logger.warning(
                    "Route already exists for %s at epoch %d, updating",
                    lid, epoch
                )
                existing[0].addresses = addresses
                existing[0].capabilities = capabilities
                return existing[0]
            
            self._routes[lid].append(route)
            # Sort by epoch descending
            self._routes[lid].sort(key=lambda r: r.epoch, reverse=True)
        
        logger.info(
            "Added %s route to %s at epoch %d: %s",
            "shadow" if as_shadow else "active",
            lid, epoch, addresses[0] if addresses else "no-addr"
        )
        
        if self.on_route_change:
            self.on_route_change(lid, route, "add")
        
        return route
    
    def get_route(
        self,
        lid: LogicalEndpointID,
        allow_shadow: bool = False,
    ) -> Optional[VersionedRoute]:
        """Get best usable route for LID.
        
        Args:
            lid: Target logical endpoint ID
            allow_shadow: Whether to return shadow routes
            
        Returns:
            Best usable route or None
        """
        with self._lock:
            routes = self._routes.get(lid, [])
            for route in routes:
                if route.state == RouteState.ACTIVE and route.is_usable:
                    return route
                if route.state == RouteState.DUAL and route.is_usable:
                    return route
                if allow_shadow and route.state == RouteState.SHADOW and route.is_usable:
                    return route
            return None
    
    def get_all_routes(self, lid: LogicalEndpointID) -> List[VersionedRoute]:
        """Get all routes for LID."""
        with self._lock:
            return list(self._routes.get(lid, []))
    
    def update_route(
        self,
        lid: LogicalEndpointID,
        new_epoch: int,
        new_addresses: Optional[List[str]] = None,
        new_capabilities: Optional[ConnectionCapability] = None,
    ) -> Optional[VersionedRoute]:
        """Update route with new epoch.
        
        Creates new route entry, marks old as draining.
        
        Args:
            lid: Target LID
            new_epoch: New epoch number
            new_addresses: New addresses (required if epoch changes)
            new_capabilities: New capabilities
            
        Returns:
            Updated route or None
        """
        with self._lock:
            routes = self._routes.get(lid, [])
            old_route = routes[0] if routes else None
            
            if old_route and old_route.epoch >= new_epoch:
                logger.warning(
                    "Cannot update route: new epoch %d <= current %d",
                    new_epoch, old_route.epoch
                )
                return None
            
            # Mark old route as draining
            if old_route and old_route.state == RouteState.ACTIVE:
                old_route.state = RouteState.DRAINING
                logger.info("Marked old route for %s as draining", lid)
            
            # Create new route
            addresses = new_addresses or (old_route.addresses if old_route else [])
            capabilities = new_capabilities or (
                old_route.capabilities if old_route else ConnectionCapability()
            )
            
            return self.add_route(lid, addresses, capabilities, new_epoch)
    
    def check_epoch(
        self,
        lid: LogicalEndpointID,
        message_epoch: int,
    ) -> Tuple[bool, Optional[int]]:
        """Check if message epoch is valid.
        
        Args:
            lid: Source LID of message
            message_epoch: Epoch from message header
            
        Returns:
            (is_valid, current_epoch) - current_epoch only set if stale
        """
        with self._lock:
            routes = self._routes.get(lid, [])
            if not routes:
                # Unknown LID, accept message but note epoch
                return True, None
            
            current_epoch = routes[0].epoch
            
            if message_epoch < current_epoch:
                # Stale epoch detected
                self._stale_epochs[lid].add(message_epoch)
                
                logger.warning(
                    "Stale epoch from %s: msg_epoch=%d, current=%d",
                    lid, message_epoch, current_epoch
                )
                
                if self.on_stale_route:
                    self.on_stale_route(lid, message_epoch, current_epoch)
                
                return False, current_epoch
            
            return True, None
    
    def handle_stale_feedback(
        self,
        lid: LogicalEndpointID,
        stale_epoch: int,
        remote_epoch: int,
    ) -> bool:
        """Handle stale epoch feedback from remote.
        
        Triggers fast route refresh.
        
        Args:
            lid: Remote LID that sent feedback
            stale_epoch: Our stale epoch
            remote_epoch: Remote's current epoch
            
        Returns:
            True if refresh needed
        """
        with self._lock:
            routes = self._routes.get(lid, [])
            if not routes:
                return True  # Need to refresh unknown route
            
            current_route = routes[0]
            if current_route.epoch < remote_epoch:
                logger.warning(
                    "Our route to %s is stale (epoch %d vs %d), requesting refresh",
                    lid, current_route.epoch, remote_epoch
                )
                self._refresh_pending.add(lid)
                return True
            
            return False
    
    def get_pending_refresh(self) -> List[LogicalEndpointID]:
        """Get LIDs pending route refresh."""
        with self._lock:
            result = list(self._refresh_pending)
            self._refresh_pending.clear()
            return result
    
    def mark_route_draining(self, lid: LogicalEndpointID, epoch: int) -> bool:
        """Mark a specific route as draining.
        
        Args:
            lid: Target LID
            epoch: Epoch to mark draining
            
        Returns:
            True if route found and marked
        """
        with self._lock:
            routes = self._routes.get(lid, [])
            for route in routes:
                if route.epoch == epoch:
                    route.state = RouteState.DRAINING
                    logger.info("Route to %s epoch %d marked draining", lid, epoch)
                    if self.on_route_change:
                        self.on_route_change(lid, route, "draining")
                    return True
            return False
    
    def mark_route_stale(self, lid: LogicalEndpointID, epoch: int) -> bool:
        """Mark a route as stale (ready for cleanup)."""
        with self._lock:
            routes = self._routes.get(lid, [])
            for route in routes:
                if route.epoch == epoch:
                    route.state = RouteState.STALE
                    logger.info("Route to %s epoch %d marked stale", lid, epoch)
                    if self.on_route_change:
                        self.on_route_change(lid, route, "stale")
                    return True
            return False
    
    def cleanup_stale_routes(self, max_age_seconds: float = 60.0) -> int:
        """Remove stale routes older than max_age.
        
        Args:
            max_age_seconds: Max age before removal
            
        Returns:
            Number of routes removed
        """
        removed = 0
        now = time.time()
        
        with self._lock:
            for lid in list(self._routes.keys()):
                routes = self._routes[lid]
                original_len = len(routes)
                
                # Keep non-stale routes and recently marked stale ones
                self._routes[lid] = [
                    r for r in routes
                    if r.state != RouteState.STALE or 
                    (now - r.last_used) < max_age_seconds
                ]
                
                removed += original_len - len(self._routes[lid])
                
                # Clean up empty entries
                if not self._routes[lid]:
                    del self._routes[lid]
        
        if removed > 0:
            logger.info("Cleaned up %d stale routes", removed)
        
        return removed
    
    # ==================== PD Separation Support ====================
    
    def lookup_by_role(self, pd_role: str) -> List[VersionedRoute]:
        """Find all active routes with a specific PD role.
        
        Used for PD (Prefill-Decode) separation to find:
        - All decode instances when prefill needs to send KV cache
        - All prefill instances when decode needs to request retry
        
        Args:
            pd_role: "prefill", "decode", or "none"
            
        Returns:
            List of active VersionedRoutes matching the role
        """
        results = []
        with self._lock:
            for lid, routes in self._routes.items():
                if lid.pd_role == pd_role:
                    for route in routes:
                        if route.state == RouteState.ACTIVE and route.is_usable:
                            results.append(route)
                            break  # Only first active route per LID
        return results
    
    def get_decode_endpoints(self) -> List[VersionedRoute]:
        """Get all active decode endpoints (convenience method)."""
        from connex.protocol import PDRole
        return self.lookup_by_role(PDRole.DECODE)
    
    def get_prefill_endpoints(self) -> List[VersionedRoute]:
        """Get all active prefill endpoints (convenience method)."""
        from connex.protocol import PDRole
        return self.lookup_by_role(PDRole.PREFILL)
    
    def register_pd_endpoint(
        self,
        model_name: str,
        pd_role: str,
        replica_idx: int,
        addresses: List[str],
        capabilities: ConnectionCapability,
    ) -> VersionedRoute:
        """Register a PD endpoint (prefill or decode instance).
        
        Args:
            model_name: Model name
            pd_role: "prefill" or "decode"
            replica_idx: Replica index for load balancing
            addresses: Physical addresses
            capabilities: Connection capabilities
            
        Returns:
            Created route entry
        """
        from connex.protocol import PDRole
        
        lid = LogicalEndpointID(
            model_name=model_name,
            stage_idx=0,  # PD doesn't use pipeline stages
            shard_idx=0,
            replica_idx=replica_idx,
            pd_role=pd_role,
        )
        
        return self.add_route(lid, addresses, capabilities)


@dataclass
class DualPathState:
    """State for dual-path cutover.
    
    During migration, both old and new paths may be active briefly.
    This state tracks which messages go to which path.
    """
    # Stream being migrated
    stream_id: StreamID
    
    # Routes
    old_route: VersionedRoute
    new_route: VersionedRoute
    
    # Cutover point
    cutover_seqno: int = 0  # Seqno at which to switch to new path
    
    # State
    started_at: float = field(default_factory=time.time)
    cutover_at: float = 0.0
    completed_at: float = 0.0
    
    # Message tracking
    old_path_last_seqno: int = 0
    new_path_first_seqno: int = 0
    
    # Retransmit buffer for gap handling
    retransmit_buffer: Dict[int, bytes] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if dual-path is still active."""
        return self.cutover_at == 0.0 or self.completed_at == 0.0
    
    def should_use_new_path(self, seqno: int) -> bool:
        """Determine which path to use for given seqno."""
        if self.cutover_at == 0.0:
            return False  # Not yet cut over
        return seqno >= self.cutover_seqno


class ShadowConnectionManager:
    """Manages shadow connections for warmup.
    
    Shadow connections are pre-established before migration:
    - Memory registration warmup
    - Connection handshake
    - Credit initialization
    
    Benefits:
    - Migration becomes local switching
    - Reduced tail latency during switch
    """
    
    def __init__(
        self,
        route_table: VersionedRouteTable,
        on_warmup_complete: Optional[Callable[[LogicalEndpointID, VersionedRoute], None]] = None,
        warmup_timeout_s: float = 30.0,
    ):
        """Initialize shadow connection manager.
        
        Args:
            route_table: The versioned route table
            on_warmup_complete: Callback when shadow warmup completes
            warmup_timeout_s: Timeout for warmup phase
        """
        self.route_table = route_table
        self.on_warmup_complete = on_warmup_complete
        self.warmup_timeout_s = warmup_timeout_s
        
        # Active warmups: LID -> warmup state
        self._warmups: Dict[LogicalEndpointID, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Background thread for warmup monitoring
        self._running = False
        self._warmup_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start shadow connection manager."""
        if self._running:
            return
        
        self._running = True
        self._warmup_thread = threading.Thread(
            target=self._warmup_monitor_loop,
            daemon=True,
            name="ShadowWarmupMonitor"
        )
        self._warmup_thread.start()
        logger.info("ShadowConnectionManager started")
    
    def stop(self) -> None:
        """Stop shadow connection manager."""
        self._running = False
        if self._warmup_thread:
            self._warmup_thread.join(timeout=2.0)
            self._warmup_thread = None
    
    def initiate_warmup(
        self,
        lid: LogicalEndpointID,
        addresses: List[str],
        capabilities: ConnectionCapability,
        new_epoch: int,
        connect_func: Callable[[str], Any],
    ) -> VersionedRoute:
        """Initiate shadow connection warmup.
        
        Args:
            lid: Target LID
            addresses: New addresses
            capabilities: Target capabilities
            new_epoch: Epoch for new connection
            connect_func: Function to establish connection
            
        Returns:
            Shadow route entry
        """
        # Add shadow route
        shadow_route = self.route_table.add_route(
            lid=lid,
            addresses=addresses,
            capabilities=capabilities,
            epoch=new_epoch,
            as_shadow=True,
        )
        
        # Track warmup state
        with self._lock:
            self._warmups[lid] = {
                "route": shadow_route,
                "connect_func": connect_func,
                "started_at": time.time(),
                "steps_complete": set(),
                "required_steps": {"connect", "mr_warmup", "credits"},
            }
        
        # Start async warmup
        threading.Thread(
            target=self._do_warmup,
            args=(lid, shadow_route, connect_func),
            daemon=True,
        ).start()
        
        logger.info("Initiated shadow warmup for %s at epoch %d", lid, new_epoch)
        return shadow_route
    
    def _do_warmup(
        self,
        lid: LogicalEndpointID,
        route: VersionedRoute,
        connect_func: Callable[[str], Any],
    ) -> None:
        """Perform warmup steps."""
        try:
            # Step 1: Establish connection
            route.connection_state = ConnectionState.CONNECTING
            connection = connect_func(route.primary_address)
            route.connection = connection
            route.connection_state = ConnectionState.WARMING
            self._mark_step_complete(lid, "connect")
            
            # Step 2: Memory registration warmup
            # This would be transport-specific
            # For now, just mark complete
            self._mark_step_complete(lid, "mr_warmup")
            
            # Step 3: Credit initialization
            # Would send credit request
            self._mark_step_complete(lid, "credits")
            
        except Exception as e:
            logger.error("Shadow warmup failed for %s: %s", lid, e)
            route.connection_state = ConnectionState.DISCONNECTED
    
    def _mark_step_complete(self, lid: LogicalEndpointID, step: str) -> None:
        """Mark a warmup step as complete."""
        with self._lock:
            state = self._warmups.get(lid)
            if not state:
                return
            
            state["steps_complete"].add(step)
            
            # Check if all steps complete
            if state["steps_complete"] >= state["required_steps"]:
                route = state["route"]
                route.warmup_complete = True
                route.connection_state = ConnectionState.READY
                
                logger.info(
                    "Shadow warmup complete for %s (epoch %d) in %.2fs",
                    lid, route.epoch, time.time() - state["started_at"]
                )
                
                if self.on_warmup_complete:
                    self.on_warmup_complete(lid, route)
                
                del self._warmups[lid]
    
    def _warmup_monitor_loop(self) -> None:
        """Monitor warmup progress and timeouts."""
        while self._running:
            now = time.time()
            
            with self._lock:
                for lid, state in list(self._warmups.items()):
                    elapsed = now - state["started_at"]
                    if elapsed > self.warmup_timeout_s:
                        logger.warning(
                            "Shadow warmup timeout for %s after %.1fs",
                            lid, elapsed
                        )
                        route = state["route"]
                        route.state = RouteState.STALE
                        route.connection_state = ConnectionState.DISCONNECTED
                        del self._warmups[lid]
            
            time.sleep(1.0)
    
    def get_warmup_status(self, lid: LogicalEndpointID) -> Optional[Dict[str, Any]]:
        """Get current warmup status for LID."""
        with self._lock:
            state = self._warmups.get(lid)
            if not state:
                return None
            
            return {
                "elapsed_s": time.time() - state["started_at"],
                "steps_complete": list(state["steps_complete"]),
                "steps_remaining": list(
                    state["required_steps"] - state["steps_complete"]
                ),
            }
    
    def is_warming(self, lid: LogicalEndpointID) -> bool:
        """Check if LID has warmup in progress."""
        with self._lock:
            return lid in self._warmups


class DualPathCutoverManager:
    """Manages dual-path cutover during migration.
    
    Ensures:
    - No message loss during switch
    - In-order delivery via epoch+seqno
    - Idempotent delivery via dedup
    """
    
    def __init__(
        self,
        route_table: VersionedRouteTable,
        on_cutover_complete: Optional[Callable[[StreamID, VersionedRoute], None]] = None,
        dual_path_window_s: float = 5.0,
    ):
        """Initialize dual-path cutover manager.
        
        Args:
            route_table: Versioned route table
            on_cutover_complete: Callback when cutover completes
            dual_path_window_s: Max duration for dual-path mode
        """
        self.route_table = route_table
        self.on_cutover_complete = on_cutover_complete
        self.dual_path_window_s = dual_path_window_s
        
        # Active cutovers: stream_id -> DualPathState
        self._cutovers: Dict[str, DualPathState] = {}
        self._lock = threading.RLock()
    
    def start_cutover(
        self,
        stream_id: StreamID,
        old_route: VersionedRoute,
        new_route: VersionedRoute,
        cutover_seqno: int,
    ) -> DualPathState:
        """Start dual-path cutover.
        
        Args:
            stream_id: Stream to migrate
            old_route: Current active route
            new_route: Warmed-up shadow route
            cutover_seqno: Sequence number at which to switch
            
        Returns:
            Dual-path state
        """
        # Set routes to dual mode
        old_route.state = RouteState.DUAL
        new_route.state = RouteState.DUAL
        
        state = DualPathState(
            stream_id=stream_id,
            old_route=old_route,
            new_route=new_route,
            cutover_seqno=cutover_seqno,
            old_path_last_seqno=cutover_seqno - 1,
            new_path_first_seqno=cutover_seqno,
        )
        
        with self._lock:
            self._cutovers[stream_id.to_str()] = state
        
        logger.info(
            "Started dual-path cutover for %s at seqno %d",
            stream_id, cutover_seqno
        )
        
        return state
    
    def select_path(self, stream_id: StreamID, seqno: int) -> Optional[VersionedRoute]:
        """Select which path to use for a message.
        
        Args:
            stream_id: Message stream
            seqno: Message sequence number
            
        Returns:
            Route to use, or None if no cutover in progress
        """
        with self._lock:
            state = self._cutovers.get(stream_id.to_str())
            if not state:
                return None
            
            if state.should_use_new_path(seqno):
                return state.new_route
            else:
                return state.old_route
    
    def record_retransmit(
        self,
        stream_id: StreamID,
        seqno: int,
        data: bytes,
    ) -> None:
        """Record message for potential retransmit.
        
        Used to handle gaps during cutover.
        
        Args:
            stream_id: Message stream
            seqno: Message sequence number
            data: Serialized message data
        """
        with self._lock:
            state = self._cutovers.get(stream_id.to_str())
            if state and state.is_active:
                state.retransmit_buffer[seqno] = data
    
    def get_retransmit(
        self,
        stream_id: StreamID,
        seqno: int,
    ) -> Optional[bytes]:
        """Get message for retransmit."""
        with self._lock:
            state = self._cutovers.get(stream_id.to_str())
            if state:
                return state.retransmit_buffer.get(seqno)
            return None
    
    def complete_cutover(self, stream_id: StreamID) -> bool:
        """Complete the cutover process.
        
        Args:
            stream_id: Stream that completed cutover
            
        Returns:
            True if successfully completed
        """
        with self._lock:
            key = stream_id.to_str()
            state = self._cutovers.get(key)
            if not state:
                return False
            
            # Finalize routes
            state.old_route.state = RouteState.DRAINING
            state.new_route.state = RouteState.ACTIVE
            state.completed_at = time.time()
            
            logger.info(
                "Cutover complete for %s in %.2fs",
                stream_id, state.completed_at - state.started_at
            )
            
            if self.on_cutover_complete:
                self.on_cutover_complete(stream_id, state.new_route)
            
            # Cleanup
            del self._cutovers[key]
            
        return True
    
    def abort_cutover(self, stream_id: StreamID) -> bool:
        """Abort cutover and revert to old route.
        
        Args:
            stream_id: Stream to abort
            
        Returns:
            True if aborted
        """
        with self._lock:
            key = stream_id.to_str()
            state = self._cutovers.get(key)
            if not state:
                return False
            
            # Revert to old route
            state.old_route.state = RouteState.ACTIVE
            state.new_route.state = RouteState.STALE
            
            logger.warning("Cutover aborted for %s", stream_id)
            
            del self._cutovers[key]
            
        return True
    
    def get_active_cutovers(self) -> List[DualPathState]:
        """Get all active cutovers."""
        with self._lock:
            return [s for s in self._cutovers.values() if s.is_active]
