# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
EPC V3.1: Mobility-Aware Elastic Pipeline Communications with Separated Control Plane.

This is the V2 architecture version that uses Front Router as the control plane.
The vLLM-embedded client is lightweight, with all control plane state managed
externally by the Front Router service.

Key changes from V3.0:
- Control plane state (routes, epoch, membership) managed by Front Router
- Client connects to Front Router via ZeroMQ REQ/REP for API
- Client subscribes to Front Router PUB for events
- Client survives vLLM instance recycles through state recovery
- Data plane (UCX/RDMA) remains embedded in vLLM

"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import zmq

# Try to import CUDA/torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add services path for client import
_SERVICES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../services/front-router"))
if _SERVICES_PATH not in sys.path:
    sys.path.insert(0, _SERVICES_PATH)

try:
    from front_router import (
        LogicalEndpointID,
        RouteEntry,
        RouteState,
        EventType,
        Event,
        TransportCapability,
    )
    from epc_client import (
        EPCV2Client,
        EPCV2ClientConfig,
        RouteTableCache,
    )
except ImportError:
    # Fallback: define locally
    from enum import IntEnum
    
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

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass  
class EPCV31Config:
    """Configuration for EPC V3.1 endpoint with Front Router."""
    
    # Identity
    model_name: str = "default"
    stage_idx: int = 0
    shard_idx: int = 0
    replica_idx: int = 0
    num_stages: int = 1
    
    # Front Router connection (REQUIRED for V2)
    front_router_api: str = "tcp://localhost:15555"
    front_router_events: str = "tcp://localhost:15560"
    
    # Local binding
    local_address: str = ""
    local_port: int = 0
    
    # Transport options
    device: str = "cuda"
    enable_ucx: bool = True
    enable_gpu_direct: bool = False
    enable_mooncake: bool = False
    
    # Buffers
    buffer_size: int = 64 * 1024 * 1024  # 64MB
    num_buffers: int = 4
    
    # Heartbeat
    heartbeat_interval_s: float = 10.0
    
    # Timeouts
    connect_timeout_ms: int = 30000
    recv_timeout_ms: int = -1  # Blocking
    
    @property
    def lid(self) -> "LogicalEndpointID":
        return LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=self.stage_idx,
            shard_idx=self.shard_idx,
            replica_idx=self.replica_idx,
        )
    
    @property
    def local_addresses(self) -> List[str]:
        if self.local_address and self.local_port:
            return [f"{self.local_address}:{self.local_port}"]
        return []
    
    @property
    def capabilities(self) -> List["TransportCapability"]:
        caps = [TransportCapability.ZEROMQ]
        if self.enable_ucx:
            caps.append(TransportCapability.UCX_TCP)
        if self.enable_gpu_direct:
            caps.append(TransportCapability.GPU_DIRECT)
        if self.enable_mooncake:
            caps.append(TransportCapability.MOONCAKE_TE)
        return caps


# ============================================================================
# Lightweight Data Plane
# ============================================================================

class LightweightDataPlane:
    """Lightweight data plane for UCX/ZeroMQ connections.
    
    Manages actual data transfer connections while control plane
    is handled by Front Router.
    """
    
    def __init__(self, config: EPCV31Config):
        self.config = config
        self._connections: Dict[str, Any] = {}  # lid_key -> connection
        self._lock = threading.Lock()
        
        # ZeroMQ for fallback/bootstrap
        self._zmq_context: Optional[zmq.Context] = None
        self._zmq_sockets: Dict[str, zmq.Socket] = {}
        
        # UCX if available
        self._ucx_available = False
        self._ucx_worker = None
        
        self._try_init_ucx()
    
    def _try_init_ucx(self):
        """Try to initialize UCX transport."""
        if not self.config.enable_ucx:
            return
            
        try:
            import ucx
            self._ucx_available = True
            # UCX worker will be created lazily
            logger.info("UCX transport available")
        except ImportError:
            logger.warning("UCX not available, using ZeroMQ fallback")
    
    def get_connection(self, lid_key: str, address: str) -> Any:
        """Get or create connection to a peer."""
        with self._lock:
            if lid_key in self._connections:
                return self._connections[lid_key]
            
            # Create new connection
            conn = self._create_connection(address)
            if conn:
                self._connections[lid_key] = conn
            
            return conn
    
    def _create_connection(self, address: str) -> Any:
        """Create a new connection to address."""
        if self._ucx_available:
            return self._create_ucx_connection(address)
        else:
            return self._create_zmq_connection(address)
    
    def _create_zmq_connection(self, address: str) -> Optional[zmq.Socket]:
        """Create ZeroMQ connection."""
        if not self._zmq_context:
            self._zmq_context = zmq.Context()
        
        try:
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.connect(f"tcp://{address}")
            return socket
        except Exception as e:
            logger.error("Failed to create ZMQ connection to %s: %s", address, e)
            return None
    
    def _create_ucx_connection(self, address: str) -> Any:
        """Create UCX connection."""
        # Placeholder - actual UCX implementation
        try:
            import ucx
            
            if not self._ucx_worker:
                self._ucx_worker = ucx.create_worker()
            
            host, port = address.rsplit(":", 1)
            ep = self._ucx_worker.create_endpoint(host, int(port))
            return ep
            
        except Exception as e:
            logger.error("Failed to create UCX connection to %s: %s", address, e)
            return self._create_zmq_connection(address)  # Fallback
    
    def close_connection(self, lid_key: str):
        """Close a specific connection."""
        with self._lock:
            conn = self._connections.pop(lid_key, None)
            if conn:
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except Exception as e:
                    logger.warning("Error closing connection: %s", e)
    
    def close_all(self):
        """Close all connections."""
        with self._lock:
            for lid_key, conn in list(self._connections.items()):
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except:
                    pass
            self._connections.clear()
        
        # Close ZMQ context
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None
    
    def send(self, lid_key: str, data: bytes) -> bool:
        """Send data to a peer."""
        with self._lock:
            conn = self._connections.get(lid_key)
            if not conn:
                logger.error("No connection for %s", lid_key)
                return False
            
            try:
                if isinstance(conn, zmq.Socket):
                    conn.send(data, zmq.NOBLOCK)
                else:
                    # UCX endpoint
                    conn.send(data)
                return True
            except Exception as e:
                logger.error("Send failed to %s: %s", lid_key, e)
                return False
    
    def recv(self, lid_key: str, timeout_ms: int = -1) -> Optional[bytes]:
        """Receive data from a peer."""
        with self._lock:
            conn = self._connections.get(lid_key)
            if not conn:
                return None
            
            try:
                if isinstance(conn, zmq.Socket):
                    if timeout_ms > 0:
                        if not conn.poll(timeout_ms):
                            return None
                    return conn.recv()
                else:
                    # UCX endpoint
                    return conn.recv()
            except Exception as e:
                logger.error("Recv failed from %s: %s", lid_key, e)
                return None


# ============================================================================
# Shadow Connection Manager
# ============================================================================

class V2ShadowManager:
    """Shadow connection manager for hitless migration (V2 architecture).
    
    Prepares shadow connections when Front Router announces migration,
    and swaps atomically on cutover.
    """
    
    def __init__(self, data_plane: LightweightDataPlane):
        self._data_plane = data_plane
        self._shadows: Dict[str, Dict[str, Any]] = {}  # migration_id -> shadow_info
        self._lock = threading.Lock()
    
    def prepare_shadow(
        self,
        migration_id: str,
        lid_key: str,
        new_address: str,
    ) -> bool:
        """Prepare shadow connection for upcoming migration."""
        with self._lock:
            if migration_id in self._shadows:
                return True  # Already preparing
            
            # Create shadow connection
            shadow_key = f"shadow:{lid_key}"
            conn = self._data_plane._create_connection(new_address)
            
            if not conn:
                logger.error("Failed to create shadow connection for %s", lid_key)
                return False
            
            self._shadows[migration_id] = {
                "lid_key": lid_key,
                "new_address": new_address,
                "shadow_key": shadow_key,
                "connection": conn,
                "ready": True,
                "created_at": time.time(),
            }
            
            logger.info("Shadow connection ready: %s -> %s", lid_key, new_address)
            return True
    
    def activate_shadow(self, migration_id: str) -> bool:
        """Activate shadow connection, replacing old connection."""
        with self._lock:
            shadow = self._shadows.pop(migration_id, None)
            if not shadow:
                logger.warning("No shadow found for migration %s", migration_id)
                return False
            
            lid_key = shadow["lid_key"]
            
            # Close old connection
            self._data_plane.close_connection(lid_key)
            
            # Install shadow as primary
            self._data_plane._connections[lid_key] = shadow["connection"]
            
            logger.info("Shadow activated for %s", lid_key)
            return True
    
    def discard_shadow(self, migration_id: str):
        """Discard shadow connection (migration aborted)."""
        with self._lock:
            shadow = self._shadows.pop(migration_id, None)
            if shadow:
                conn = shadow.get("connection")
                if conn and hasattr(conn, 'close'):
                    try:
                        conn.close()
                    except:
                        pass
                logger.info("Shadow discarded for migration %s", migration_id)


# ============================================================================
# EPC V3.1 Endpoint
# ============================================================================

class EPCV31Endpoint:
    """EPC V3.1 Endpoint with Front Router control plane.
    
    This is the V2 architecture implementation where:
    - Control plane (routes, epoch, membership) is managed by Front Router
    - Client is lightweight, survives vLLM instance recycles
    - Data plane (UCX/ZeroMQ) is embedded in vLLM
    
    Key features:
    - Automatic registration with Front Router on start
    - Event subscription for route/membership updates
    - Shadow connection warmup for hitless migration
    - Recovery of state from Front Router on reconnect
    """
    
    def __init__(self, config: EPCV31Config):
        """Initialize EPC V3.1 endpoint."""
        self.config = config
        self.lid = config.lid
        
        # ZMQ context for control plane
        self._context: Optional[zmq.Context] = None
        self._api_socket: Optional[zmq.Socket] = None
        self._sub_socket: Optional[zmq.Socket] = None
        
        # Local route cache (populated from Front Router)
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._local_epoch = 0
        self._route_lock = threading.Lock()
        
        # Data plane
        self._data_plane = LightweightDataPlane(config)
        self._shadow_manager = V2ShadowManager(self._data_plane)
        
        # State
        self._registered = False
        self._running = False
        
        # Threads
        self._event_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._event_callbacks: Dict[int, List[Callable]] = {}
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "migrations_handled": 0,
            "reconnections": 0,
        }
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self) -> bool:
        """Start the endpoint and register with Front Router."""
        logger.info("Starting EPC V3.1 endpoint: %s", self.lid.to_key())
        
        try:
            # Initialize ZMQ
            self._context = zmq.Context()
            
            # Connect to Front Router API
            self._api_socket = self._context.socket(zmq.REQ)
            self._api_socket.connect(self.config.front_router_api)
            self._api_socket.setsockopt(zmq.RCVTIMEO, self.config.connect_timeout_ms)
            
            # Connect to Front Router events
            self._sub_socket = self._context.socket(zmq.SUB)
            self._sub_socket.connect(self.config.front_router_events)
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "connex.")
            
            self._running = True
            
            # Register with Front Router
            if not self._register():
                logger.error("Failed to register with Front Router")
                self.stop()
                return False
            
            # Start event thread
            self._event_thread = threading.Thread(
                target=self._event_loop,
                daemon=True,
                name=f"epc-v31-events-{self.config.stage_idx}",
            )
            self._event_thread.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"epc-v31-heartbeat-{self.config.stage_idx}",
            )
            self._heartbeat_thread.start()
            
            # Establish data plane connections
            self._establish_data_connections()
            
            logger.info(
                "EPC V3.1 endpoint started (epoch=%d, routes=%d)",
                self._local_epoch, len(self._routes)
            )
            return True
            
        except Exception as e:
            logger.error("Failed to start EPC V3.1 endpoint: %s", e)
            self.stop()
            return False
    
    def stop(self):
        """Stop the endpoint and deregister."""
        logger.info("Stopping EPC V3.1 endpoint: %s", self.lid.to_key())
        
        self._running = False
        
        # Deregister
        if self._registered:
            try:
                self._deregister()
            except Exception as e:
                logger.warning("Deregister error: %s", e)
        
        # Close control plane sockets
        if self._sub_socket:
            self._sub_socket.close()
            self._sub_socket = None
        
        if self._api_socket:
            self._api_socket.close()
            self._api_socket = None
        
        if self._context:
            self._context.term()
            self._context = None
        
        # Close data plane
        self._data_plane.close_all()
        
        self._registered = False
        logger.info("EPC V3.1 endpoint stopped")
    
    # =========================================================================
    # Front Router API
    # =========================================================================
    
    def _call_api(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Front Router API."""
        if not self._api_socket:
            raise RuntimeError("Not connected to Front Router")
        
        request = {"method": method, "params": params}
        self._api_socket.send_json(request)
        response = self._api_socket.recv_json()
        
        if "error" in response:
            raise RuntimeError(f"Front Router API error: {response['error']}")
        
        return response
    
    def _register(self) -> bool:
        """Register with Front Router."""
        try:
            response = self._call_api("register", {
                "lid": self.lid.to_dict(),
                "addresses": self.config.local_addresses,
                "capabilities": [int(c) for c in self.config.capabilities],
            })
            
            if response.get("success"):
                self._local_epoch = response.get("epoch", 0)
                self._registered = True
                
                # Populate route cache
                route_table = response.get("route_table", [])
                self._update_routes(route_table)
                
                is_reconnect = response.get("is_reconnect", False)
                if is_reconnect:
                    self._stats["reconnections"] += 1
                
                logger.info(
                    "Registered with Front Router (epoch=%d, reconnect=%s)",
                    self._local_epoch, is_reconnect
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
    
    def _update_routes(self, route_list: List[Dict[str, Any]]):
        """Update local route cache from Front Router response."""
        with self._route_lock:
            self._routes.clear()
            for r in route_list:
                lid_dict = r.get("lid", {})
                lid_key = f"{lid_dict.get('model_name')}:{lid_dict.get('stage_idx')}:{lid_dict.get('shard_idx', 0)}:{lid_dict.get('replica_idx', 0)}"
                self._routes[lid_key] = r
    
    def _refresh_routes(self):
        """Refresh routes from Front Router."""
        try:
            response = self._call_api("get_routes", {
                "model_name": self.config.model_name
            })
            
            routes = response.get("routes", [])
            epoch = response.get("epoch", self._local_epoch)
            
            self._update_routes(routes)
            self._local_epoch = epoch
            
        except Exception as e:
            logger.error("Failed to refresh routes: %s", e)
    
    def _establish_data_connections(self):
        """Establish data plane connections to pipeline peers."""
        with self._route_lock:
            for lid_key, route in self._routes.items():
                # Skip self
                if lid_key == self.lid.to_key():
                    continue
                
                # Get address
                addresses = route.get("addresses", [])
                if addresses:
                    self._data_plane.get_connection(lid_key, addresses[0])
    
    # =========================================================================
    # Event Handling
    # =========================================================================
    
    def _event_loop(self):
        """Event subscription loop."""
        import json
        
        while self._running:
            try:
                if self._sub_socket.poll(timeout=1000):
                    topic, data = self._sub_socket.recv_multipart()
                    
                    event_dict = json.loads(data.decode("utf-8"))
                    self._handle_event(event_dict)
                    
            except Exception as e:
                if self._running:
                    logger.error("Event loop error: %s", e)
    
    def _handle_event(self, event_dict: Dict[str, Any]):
        """Handle control plane event."""
        event_type = event_dict.get("type", 0)
        event_epoch = event_dict.get("epoch", 0)
        payload = event_dict.get("payload", {})
        source_lid = event_dict.get("source_lid")
        
        # Update local epoch
        if event_epoch > self._local_epoch:
            self._local_epoch = event_epoch
        
        # Skip events from ourselves
        if source_lid == self.lid.to_key():
            return
        
        logger.debug("Received event type=%d epoch=%d", event_type, event_epoch)
        
        # Handle specific events
        if event_type == EventType.ENDPOINT_REGISTERED:
            self._on_endpoint_registered(payload)
        elif event_type == EventType.ENDPOINT_DEREGISTERED:
            self._on_endpoint_deregistered(payload)
        elif event_type == EventType.PREPARE_MIGRATION:
            self._on_prepare_migration(payload)
        elif event_type == EventType.CUTOVER_START:
            self._on_cutover_start(payload)
        elif event_type == EventType.CUTOVER_COMPLETE:
            self._on_cutover_complete(payload)
        elif event_type == EventType.MIGRATION_FAILED:
            self._on_migration_failed(payload)
        
        # Call user callbacks
        callbacks = self._event_callbacks.get(event_type, [])
        for cb in callbacks:
            try:
                cb(event_dict)
            except Exception as e:
                logger.error("Event callback error: %s", e)
    
    def _on_endpoint_registered(self, payload: Dict[str, Any]):
        """Handle new endpoint registration."""
        lid_dict = payload.get("lid", {})
        addresses = payload.get("addresses", [])
        
        lid_key = f"{lid_dict.get('model_name')}:{lid_dict.get('stage_idx')}:{lid_dict.get('shard_idx', 0)}:{lid_dict.get('replica_idx', 0)}"
        
        logger.info("New endpoint registered: %s", lid_key)
        
        # Refresh routes
        self._refresh_routes()
        
        # Establish connection if same model
        if lid_dict.get("model_name") == self.config.model_name and addresses:
            self._data_plane.get_connection(lid_key, addresses[0])
    
    def _on_endpoint_deregistered(self, payload: Dict[str, Any]):
        """Handle endpoint deregistration."""
        lid_dict = payload.get("lid", {})
        lid_key = f"{lid_dict.get('model_name')}:{lid_dict.get('stage_idx')}:{lid_dict.get('shard_idx', 0)}:{lid_dict.get('replica_idx', 0)}"
        
        logger.info("Endpoint deregistered: %s", lid_key)
        
        # Remove from cache and close connection
        with self._route_lock:
            self._routes.pop(lid_key, None)
        
        self._data_plane.close_connection(lid_key)
    
    def _on_prepare_migration(self, payload: Dict[str, Any]):
        """Handle migration preparation - create shadow connection."""
        migration_id = payload.get("migration_id")
        lid_dict = payload.get("lid", {})
        new_address = payload.get("new_address")
        
        lid_key = f"{lid_dict.get('model_name')}:{lid_dict.get('stage_idx')}:{lid_dict.get('shard_idx', 0)}:{lid_dict.get('replica_idx', 0)}"
        
        logger.info("Preparing migration %s: %s -> %s", migration_id, lid_key, new_address)
        
        # Prepare shadow connection
        if lid_dict.get("model_name") == self.config.model_name:
            success = self._shadow_manager.prepare_shadow(
                migration_id, lid_key, new_address
            )
            
            # Notify Front Router that shadow is ready
            if success:
                try:
                    self._call_api("shadow_ready", {
                        "migration_id": migration_id,
                        "from_lid": self.lid.to_dict(),
                    })
                except Exception as e:
                    logger.error("Failed to notify shadow ready: %s", e)
    
    def _on_cutover_start(self, payload: Dict[str, Any]):
        """Handle cutover start."""
        migration_id = payload.get("migration_id")
        logger.info("Cutover starting: %s", migration_id)
    
    def _on_cutover_complete(self, payload: Dict[str, Any]):
        """Handle cutover complete - activate shadow."""
        migration_id = payload.get("migration_id")
        
        logger.info("Cutover complete: %s", migration_id)
        
        # Activate shadow connection
        self._shadow_manager.activate_shadow(migration_id)
        
        # Refresh routes
        self._refresh_routes()
        
        self._stats["migrations_handled"] += 1
    
    def _on_migration_failed(self, payload: Dict[str, Any]):
        """Handle migration failure - discard shadow."""
        migration_id = payload.get("migration_id")
        reason = payload.get("reason", "unknown")
        
        logger.warning("Migration failed: %s (%s)", migration_id, reason)
        
        self._shadow_manager.discard_shadow(migration_id)
    
    # =========================================================================
    # Heartbeat
    # =========================================================================
    
    def _heartbeat_loop(self):
        """Heartbeat maintenance loop."""
        while self._running:
            if self._registered:
                try:
                    response = self._call_api("heartbeat", {"lid": self.lid.to_dict()})
                    
                    # Check epoch
                    server_epoch = response.get("epoch", 0)
                    if server_epoch > self._local_epoch:
                        logger.info("Epoch advanced: %d -> %d", self._local_epoch, server_epoch)
                        self._local_epoch = server_epoch
                        self._refresh_routes()
                        
                except Exception as e:
                    logger.warning("Heartbeat failed: %s", e)
            
            # Wait for next heartbeat
            for _ in range(int(self.config.heartbeat_interval_s * 10)):
                if not self._running:
                    break
                time.sleep(0.1)
    
    # =========================================================================
    # Data Plane API
    # =========================================================================
    
    def get_peer_address(self, stage_idx: int) -> Optional[str]:
        """Get address for a pipeline stage."""
        lid_key = f"{self.config.model_name}:{stage_idx}:0:0"
        
        with self._route_lock:
            route = self._routes.get(lid_key)
            if route:
                addresses = route.get("addresses", [])
                if addresses:
                    return addresses[0]
        
        return None
    
    def get_next_stage_address(self) -> Optional[str]:
        """Get address of next pipeline stage."""
        return self.get_peer_address(self.config.stage_idx + 1)
    
    def get_prev_stage_address(self) -> Optional[str]:
        """Get address of previous pipeline stage."""
        if self.config.stage_idx == 0:
            return None
        return self.get_peer_address(self.config.stage_idx - 1)
    
    def send_to_stage(self, stage_idx: int, data: bytes) -> bool:
        """Send data to a pipeline stage."""
        lid_key = f"{self.config.model_name}:{stage_idx}:0:0"
        
        with self._route_lock:
            route = self._routes.get(lid_key)
            if not route:
                logger.error("No route to stage %d", stage_idx)
                return False
            
            addresses = route.get("addresses", [])
            if not addresses:
                logger.error("No address for stage %d", stage_idx)
                return False
        
        # Ensure connection exists
        self._data_plane.get_connection(lid_key, addresses[0])
        
        # Send
        success = self._data_plane.send(lid_key, data)
        
        if success:
            self._stats["messages_sent"] += 1
            self._stats["bytes_sent"] += len(data)
        
        return success
    
    def recv_from_stage(self, stage_idx: int, timeout_ms: int = -1) -> Optional[bytes]:
        """Receive data from a pipeline stage."""
        lid_key = f"{self.config.model_name}:{stage_idx}:0:0"
        
        data = self._data_plane.recv(lid_key, timeout_ms)
        
        if data:
            self._stats["messages_received"] += 1
            self._stats["bytes_received"] += len(data)
        
        return data
    
    def send_forward(self, data: bytes) -> bool:
        """Send data to next pipeline stage."""
        next_stage = self.config.stage_idx + 1
        if next_stage >= self.config.num_stages:
            return True  # Last stage, no forward
        
        return self.send_to_stage(next_stage, data)
    
    def recv_backward(self, timeout_ms: int = -1) -> Optional[bytes]:
        """Receive data from previous pipeline stage."""
        if self.config.stage_idx == 0:
            return None  # First stage, no backward recv
        
        return self.recv_from_stage(self.config.stage_idx - 1, timeout_ms)
    
    # =========================================================================
    # Event Registration
    # =========================================================================
    
    def on_event(self, event_type: int, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for control plane events."""
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)
    
    # =========================================================================
    # Properties and Stats
    # =========================================================================
    
    @property
    def epoch(self) -> int:
        """Current epoch (from Front Router)."""
        return self._local_epoch
    
    @property
    def is_connected(self) -> bool:
        """Whether connected to Front Router."""
        return self._registered and self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics."""
        return {
            **self._stats,
            "epoch": self._local_epoch,
            "routes_count": len(self._routes),
            "data_connections": len(self._data_plane._connections),
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_epc_v31_endpoint(
    model_name: str,
    stage_idx: int,
    num_stages: int,
    front_router_api: str = "tcp://localhost:15555",
    front_router_events: str = "tcp://localhost:15560",
    local_address: str = "",
    local_port: int = 0,
    **kwargs,
) -> EPCV31Endpoint:
    """Create an EPC V3.1 endpoint.
    
    Args:
        model_name: Name of the model being served
        stage_idx: Pipeline stage index
        num_stages: Total number of pipeline stages
        front_router_api: Front Router API address
        front_router_events: Front Router event bus address
        local_address: Local bind address
        local_port: Local bind port
        **kwargs: Additional config options
    
    Returns:
        Configured EPCV31Endpoint instance
    """
    config = EPCV31Config(
        model_name=model_name,
        stage_idx=stage_idx,
        num_stages=num_stages,
        front_router_api=front_router_api,
        front_router_events=front_router_events,
        local_address=local_address,
        local_port=local_port,
        **kwargs,
    )
    
    return EPCV31Endpoint(config)
