# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Route Table with Epoch-based Versioned Routing.

Maintains routing information for data plane with:
- Epoch-versioned routes for safe cutover
- Multi-path routing support
- Priority-based route selection
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from connex.directory import EndpointInfo, TransportCapability

logger = logging.getLogger(__name__)


@dataclass
class RouteEntry:
    """Single route entry with version and state."""
    target_lid: tuple                    # Target logical endpoint ID
    endpoint: EndpointInfo               # Physical endpoint info
    epoch: int                           # Route epoch (version)
    transport: TransportCapability       # Selected transport
    priority: int = 0                    # Route priority (higher = preferred)
    state: str = "active"                # active, draining, stale
    
    # Connection handle (set after connection established)
    connection_handle: Any = None
    
    # Metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    last_used: float = 0.0
    
    def is_valid(self, current_epoch: int) -> bool:
        """Check if route is still valid at given epoch."""
        return self.state == "active" and self.epoch >= current_epoch
        
    def mark_draining(self) -> None:
        """Mark route as draining (no new messages, finish in-flight)."""
        self.state = "draining"
        
    def mark_stale(self) -> None:
        """Mark route as stale (should be removed)."""
        self.state = "stale"


class RouteTable:
    """Thread-safe route table with epoch-based routing.
    
    Features:
    - Epoch versioning for safe cutover
    - Multi-path routes per destination
    - Transport capability negotiation
    - Route lifecycle management
    """
    
    def __init__(self, local_lid: tuple, local_caps: list[TransportCapability]):
        """Initialize route table.
        
        Args:
            local_lid: Local endpoint LID
            local_caps: Local transport capabilities
        """
        self.local_lid = local_lid
        self.local_caps = set(local_caps)
        
        # Routes indexed by target LID
        # target_lid -> list[RouteEntry] (multiple paths possible)
        self._routes: dict[tuple, list[RouteEntry]] = {}
        self._lock = threading.RLock()
        
        # Current epoch
        self._epoch = 0
        
        # Route change callbacks
        self._change_handlers: list[callable] = []
        
    @property
    def epoch(self) -> int:
        """Current routing epoch."""
        return self._epoch
        
    def increment_epoch(self) -> int:
        """Atomically increment epoch."""
        with self._lock:
            self._epoch += 1
            return self._epoch
            
    def add_route(
        self,
        target_lid: tuple,
        endpoint: EndpointInfo,
        transport: TransportCapability | None = None,
        priority: int = 0,
    ) -> RouteEntry:
        """Add a route to the table.
        
        Args:
            target_lid: Target endpoint LID
            endpoint: Endpoint information
            transport: Preferred transport (auto-negotiate if None)
            priority: Route priority
            
        Returns:
            Created RouteEntry
        """
        # Negotiate transport if not specified
        if transport is None:
            transport = self._negotiate_transport(endpoint)
            
        entry = RouteEntry(
            target_lid=target_lid,
            endpoint=endpoint,
            epoch=self._epoch,
            transport=transport,
            priority=priority,
            state="active",
        )
        
        with self._lock:
            if target_lid not in self._routes:
                self._routes[target_lid] = []
            self._routes[target_lid].append(entry)
            # Sort by priority (descending)
            self._routes[target_lid].sort(key=lambda r: r.priority, reverse=True)
            
        logger.info(
            "Added route to LID=%s via %s at epoch=%d",
            target_lid, transport.value, self._epoch
        )
        
        self._notify_change("add", entry)
        return entry
        
    def update_route(
        self,
        target_lid: tuple,
        new_endpoint: EndpointInfo,
        new_epoch: int,
    ) -> Optional[RouteEntry]:
        """Update existing route with new endpoint info.
        
        Used during migration/cutover - marks old routes as draining.
        
        Args:
            target_lid: Target endpoint LID
            new_endpoint: New endpoint information
            new_epoch: New epoch for the route
            
        Returns:
            New RouteEntry, or None if no existing route
        """
        with self._lock:
            if target_lid not in self._routes:
                return None
                
            # Mark existing routes as draining
            for entry in self._routes[target_lid]:
                if entry.state == "active":
                    entry.mark_draining()
                    
            # Add new route with higher epoch
            transport = self._negotiate_transport(new_endpoint)
            new_entry = RouteEntry(
                target_lid=target_lid,
                endpoint=new_endpoint,
                epoch=new_epoch,
                transport=transport,
                priority=100,  # New routes get high priority
                state="active",
            )
            self._routes[target_lid].append(new_entry)
            self._routes[target_lid].sort(key=lambda r: r.priority, reverse=True)
            
        logger.info(
            "Updated route to LID=%s at epoch=%d (old routes draining)",
            target_lid, new_epoch
        )
        
        self._notify_change("update", new_entry)
        return new_entry
        
    def get_route(self, target_lid: tuple, epoch: int | None = None) -> Optional[RouteEntry]:
        """Get best active route to target.
        
        Args:
            target_lid: Target endpoint LID
            epoch: Minimum required epoch (uses current if None)
            
        Returns:
            Best active RouteEntry, or None
        """
        epoch = epoch if epoch is not None else self._epoch
        
        with self._lock:
            routes = self._routes.get(target_lid, [])
            for entry in routes:
                if entry.is_valid(epoch):
                    return entry
        return None
        
    def get_all_routes(self, target_lid: tuple) -> list[RouteEntry]:
        """Get all routes to target (including draining)."""
        with self._lock:
            return list(self._routes.get(target_lid, []))
            
    def remove_stale_routes(self) -> int:
        """Remove all stale routes.
        
        Returns:
            Number of routes removed
        """
        removed = 0
        with self._lock:
            for lid in list(self._routes.keys()):
                before = len(self._routes[lid])
                self._routes[lid] = [
                    r for r in self._routes[lid] if r.state != "stale"
                ]
                removed += before - len(self._routes[lid])
                
                # Remove empty entries
                if not self._routes[lid]:
                    del self._routes[lid]
                    
        if removed > 0:
            logger.info("Removed %d stale routes", removed)
        return removed
        
    def mark_routes_stale(self, target_lid: tuple, below_epoch: int) -> int:
        """Mark routes below epoch as stale.
        
        Args:
            target_lid: Target endpoint LID
            below_epoch: Mark routes with epoch < this as stale
            
        Returns:
            Number of routes marked
        """
        marked = 0
        with self._lock:
            routes = self._routes.get(target_lid, [])
            for entry in routes:
                if entry.epoch < below_epoch and entry.state != "stale":
                    entry.mark_stale()
                    marked += 1
                    
        if marked > 0:
            logger.info("Marked %d routes as stale for LID=%s", marked, target_lid)
        return marked
        
    def _negotiate_transport(self, endpoint: EndpointInfo) -> TransportCapability:
        """Negotiate best transport with endpoint.
        
        Priority: RDMA > SHM > TCP > ZeroMQ
        """
        remote_caps = set(endpoint.transport_caps)
        common = self.local_caps & remote_caps
        
        # Priority order
        for cap in [
            TransportCapability.UCX_RDMA,
            TransportCapability.UCX_SHM,
            TransportCapability.UCX_TCP,
            TransportCapability.ZEROMQ,
        ]:
            if cap in common:
                return cap
                
        # Fallback to ZeroMQ
        return TransportCapability.ZEROMQ
        
    def on_change(self, handler: callable) -> None:
        """Register route change handler."""
        self._change_handlers.append(handler)
        
    def _notify_change(self, action: str, entry: RouteEntry) -> None:
        """Notify handlers of route change."""
        for handler in self._change_handlers:
            try:
                handler(action, entry)
            except Exception as e:
                logger.error("Route change handler error: %s", e)
                
    def get_stats(self) -> dict[str, Any]:
        """Get route table statistics."""
        with self._lock:
            total_routes = sum(len(routes) for routes in self._routes.values())
            active_routes = sum(
                1 for routes in self._routes.values()
                for r in routes if r.state == "active"
            )
            draining_routes = sum(
                1 for routes in self._routes.values()
                for r in routes if r.state == "draining"
            )
            
        return {
            "epoch": self._epoch,
            "total_routes": total_routes,
            "active_routes": active_routes,
            "draining_routes": draining_routes,
            "destinations": len(self._routes),
        }
