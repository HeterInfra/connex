# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Directory Service Client for Endpoint Mobility.

Maintains logical endpoint (LID) to physical endpoint mapping with versioned routing.
Uses ZeroMQ as control plane transport.

Logical Endpoint ID (LID): (model, stage, shard, replica)
Mapping: LID -> {addr_list, transport_caps, epoch, priority_classes}
Events: JOIN/LEAVE/MIGRATE/PREPARE/CUTOVER/FINALIZE
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import zmq

logger = logging.getLogger(__name__)


class MembershipEvent(Enum):
    """Membership state change events."""
    JOIN = "JOIN"
    LEAVE = "LEAVE"
    MIGRATE = "MIGRATE"
    PREPARE = "PREPARE"
    CUTOVER = "CUTOVER"
    FINALIZE = "FINALIZE"


class TransportCapability(Enum):
    """Available transport capabilities."""
    UCX_RDMA = "ucx_rdma"      # RDMA via UCX
    UCX_TCP = "ucx_tcp"        # TCP via UCX
    UCX_SHM = "ucx_shm"        # Shared memory via UCX
    ZEROMQ = "zeromq"          # ZeroMQ (fallback)


@dataclass
class EndpointInfo:
    """Physical endpoint information."""
    lid: tuple[str, int, int, int]  # (model, stage, shard, replica)
    addresses: list[str]             # Multiple addresses for failover
    transport_caps: list[TransportCapability]
    epoch: int = 0
    priority_classes: list[str] = field(default_factory=lambda: ["token", "control"])
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Connection state
    last_heartbeat: float = 0.0
    state: str = "active"  # active, migrating, leaving
    
    @property
    def model(self) -> str:
        return self.lid[0]
    
    @property
    def stage(self) -> int:
        return self.lid[1]
    
    @property
    def shard(self) -> int:
        return self.lid[2]
    
    @property
    def replica(self) -> int:
        return self.lid[3]
    
    @property
    def primary_address(self) -> str:
        """Get primary address for connection."""
        return self.addresses[0] if self.addresses else ""
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for ZeroMQ transport."""
        return {
            "lid": self.lid,
            "addresses": self.addresses,
            "transport_caps": [c.value for c in self.transport_caps],
            "epoch": self.epoch,
            "priority_classes": self.priority_classes,
            "metadata": self.metadata,
            "state": self.state,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EndpointInfo":
        """Deserialize from dict."""
        return cls(
            lid=tuple(data["lid"]),
            addresses=data["addresses"],
            transport_caps=[TransportCapability(c) for c in data["transport_caps"]],
            epoch=data.get("epoch", 0),
            priority_classes=data.get("priority_classes", ["token", "control"]),
            metadata=data.get("metadata", {}),
            state=data.get("state", "active"),
        )


class DirectoryClient:
    """Client for interacting with the directory service.
    
    Handles:
    - Endpoint registration (JOIN)
    - Endpoint discovery (LOOKUP)
    - Membership events subscription
    - Epoch-based routing updates
    """
    
    def __init__(
        self,
        front_router_addr: str,
        model_name: str,
        stage_idx: int,
        shard_idx: int = 0,
        replica_idx: int = 0,
        local_addresses: list[str] | None = None,
        transport_caps: list[TransportCapability] | None = None,
    ):
        """Initialize directory client.
        
        Args:
            front_router_addr: Address of the front router (control plane)
            model_name: Model name for LID
            stage_idx: Pipeline stage index
            shard_idx: Tensor shard index (for TP)
            replica_idx: Replica index (for redundancy)
            local_addresses: Local endpoint addresses to register
            transport_caps: Supported transport capabilities
        """
        self.front_router_addr = front_router_addr
        self.model_name = model_name
        self.stage_idx = stage_idx
        self.shard_idx = shard_idx
        self.replica_idx = replica_idx
        
        self.lid = (model_name, stage_idx, shard_idx, replica_idx)
        self.local_addresses = local_addresses or []
        self.transport_caps = transport_caps or [
            TransportCapability.UCX_RDMA,
            TransportCapability.UCX_TCP,
            TransportCapability.ZEROMQ,
        ]
        
        # Current epoch for versioned routing
        self._epoch = 0
        self._epoch_lock = threading.Lock()
        
        # Local endpoint cache: LID -> EndpointInfo
        self._endpoint_cache: dict[tuple, EndpointInfo] = {}
        self._cache_lock = threading.RLock()
        
        # Event subscribers
        self._event_handlers: list[Callable[[MembershipEvent, EndpointInfo], None]] = []
        
        # ZeroMQ context and sockets
        self._ctx = zmq.Context.instance()
        self._control_socket: Optional[zmq.Socket] = None
        self._sub_socket: Optional[zmq.Socket] = None
        
        # Background thread for event processing
        self._running = False
        self._event_thread: Optional[threading.Thread] = None

        # Liveness: lease heartbeat thread (uses its own socket)
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        self._registered: Optional[dict[str, Any]] = None
        
    def start(self) -> None:
        """Start the directory client."""
        if self._running:
            return
            
        self._running = True
        
        # Connect control socket to front router
        self._control_socket = self._ctx.socket(zmq.DEALER)
        self._control_socket.setsockopt(zmq.LINGER, 0)
        self._control_socket.setsockopt_string(
            zmq.IDENTITY, 
            f"connex-{self.model_name}-{self.stage_idx}-{self.shard_idx}-{self.replica_idx}"
        )
        self._control_socket.connect(f"tcp://{self.front_router_addr}")
        
        logger.info(
            "DirectoryClient started for LID=%s, connected to %s",
            self.lid, self.front_router_addr
        )
        
    def stop(self) -> None:
        """Stop the directory client."""
        self._running = False
        self._hb_stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=2.0)
            self._hb_thread = None

        if self._event_thread:
            self._event_thread.join(timeout=2.0)
            self._event_thread = None
            
        if self._control_socket:
            self._control_socket.close()
            self._control_socket = None
            
        if self._sub_socket:
            self._sub_socket.close()
            self._sub_socket = None
            
    def register(self, addresses: list[str] | None = None) -> bool:
        """Register this endpoint with the directory.
        
        Args:
            addresses: Addresses to register (uses local_addresses if None)
            
        Returns:
            True if registration successful
        """
        addresses = addresses or self.local_addresses
        if not addresses:
            logger.warning("No addresses to register for LID=%s", self.lid)
            return False
            
        endpoint = EndpointInfo(
            lid=self.lid,
            addresses=addresses,
            transport_caps=self.transport_caps,
            epoch=self._epoch,
            state="active",
        )
        
        # Send registration request
        request = {
            "type": "register",
            "event": MembershipEvent.JOIN.value,
            "endpoint": endpoint.to_dict(),
        }
        
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)
            
            if response and response.get("status") == "ok":
                self._registered = request
                # Update local cache
                with self._cache_lock:
                    self._endpoint_cache[self.lid] = endpoint
                logger.info("Registered endpoint LID=%s at epoch=%d", self.lid, self._epoch)
                return True
            else:
                logger.error("Registration failed for LID=%s: %s", self.lid, response)
                return False
                
        except Exception as e:
            logger.error("Registration error for LID=%s: %s", self.lid, e)
            return False
            
    def lookup(self, target_lid: tuple) -> Optional[EndpointInfo]:
        """Lookup endpoint by LID.
        
        Args:
            target_lid: Target logical endpoint ID
            
        Returns:
            EndpointInfo if found, None otherwise
        """
        # Check cache first
        with self._cache_lock:
            if target_lid in self._endpoint_cache:
                cached = self._endpoint_cache[target_lid]
                # Check if cache is still valid (same epoch)
                if cached.epoch >= self._epoch:
                    return cached
                    
        # Query directory service
        request = {
            "type": "lookup",
            "lid": target_lid,
            "epoch": self._epoch,
        }
        
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)
            
            if response and response.get("status") == "ok":
                endpoint = EndpointInfo.from_dict(response["endpoint"])
                
                # Update cache
                with self._cache_lock:
                    self._endpoint_cache[target_lid] = endpoint
                    
                return endpoint
            else:
                logger.warning("Lookup failed for LID=%s: %s", target_lid, response)
                return None
                
        except Exception as e:
            logger.error("Lookup error for LID=%s: %s", target_lid, e)
            return None
            
    def lookup_stage(self, stage_idx: int) -> Optional[EndpointInfo]:
        """Lookup endpoint by stage index (convenience method).
        
        Args:
            stage_idx: Target stage index
            
        Returns:
            EndpointInfo for the first replica at that stage
        """
        target_lid = (self.model_name, stage_idx, 0, 0)
        return self.lookup(target_lid)
        
    def list_stage(self, stage_idx: int) -> list[EndpointInfo]:
        """List all live replicas/shards registered for a stage of this model.

        Returns an empty list if the stage has no members or the query fails.
        """
        request = {"type": "list_stage", "model": self.model_name, "stage": stage_idx}
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)
        except Exception as e:
            logger.error("list_stage error for stage %d: %s", stage_idx, e)
            return []
        if not response or response.get("status") != "ok":
            return []
        return [EndpointInfo.from_dict(ep) for ep in response.get("endpoints", [])]

    def request_assignment(self, target_stage: int, direction: str) -> Optional[tuple[str, str]]:
        """Request routing assignment from front router.
        
        This is compatible with the existing ZeroMQ front router protocol.
        
        Args:
            target_stage: Target stage index
            direction: "forward" or "return"
            
        Returns:
            (endpoint, routing_key) tuple if successful
        """
        request = {
            "type": "request_assignment",
            "source_stage": self.stage_idx,
            "target_stage": target_stage,
            "direction": direction,
            "model": self.model_name,
        }
        
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)
            
            if response and response.get("status") == "ok":
                endpoint = response.get("endpoint", "")
                routing_key = response.get("routing_key", "")
                return endpoint, routing_key
            else:
                logger.warning("Assignment request failed: %s", response)
                return None
                
        except Exception as e:
            logger.error("Assignment request error: %s", e)
            return None
            
    def notify_migration(
        self,
        new_addresses: list[str],
        target_epoch: int,
    ) -> bool:
        """Notify directory of endpoint migration.
        
        Args:
            new_addresses: New endpoint addresses after migration
            target_epoch: Target epoch for the migration
            
        Returns:
            True if migration notification accepted
        """
        request = {
            "type": "register",
            "event": MembershipEvent.MIGRATE.value,
            "endpoint": {
                "lid": self.lid,
                "addresses": new_addresses,
                "transport_caps": [c.value for c in self.transport_caps],
                "epoch": target_epoch,
                "state": "migrating",
            },
        }
        
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)
            
            if response and response.get("status") == "ok":
                with self._epoch_lock:
                    self._epoch = target_epoch
                logger.info("Migration notified for LID=%s to epoch=%d", self.lid, target_epoch)
                return True
            else:
                logger.error("Migration notification failed: %s", response)
                return False
                
        except Exception as e:
            logger.error("Migration notification error: %s", e)
            return False
            
    def deregister(self) -> bool:
        """Remove this endpoint from the directory (LEAVE).

        Returns:
            True if the directory acknowledged the leave
        """
        request = {
            "type": "register",
            "event": MembershipEvent.LEAVE.value,
            "endpoint": {
                "lid": self.lid,
                "addresses": [],
                "transport_caps": [c.value for c in self.transport_caps],
                "epoch": self._epoch,
                "state": "leaving",
            },
        }

        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=5000)

            if response and response.get("status") == "ok":
                self._registered = None
                self._hb_stop.set()
                self.invalidate_cache(self.lid)
                logger.info("Deregistered endpoint LID=%s", self.lid)
                return True
            logger.error("Deregistration failed for LID=%s: %s", self.lid, response)
            return False

        except Exception as e:
            logger.error("Deregistration error for LID=%s: %s", self.lid, e)
            return False

    def report_failure(self, target_lid: tuple) -> bool:
        """Tell the directory that ``target_lid`` is unreachable.

        The directory evicts it and publishes LEAVE, so every peer stops
        routing to it without waiting for its lease to expire. A member that is
        actually alive re-registers on its next heartbeat.
        """
        request = {"type": "report_failure", "lid": list(target_lid)}
        try:
            self._send_control_message(request)
            response = self._recv_control_message(timeout_ms=2000)
        except Exception as e:
            logger.error("report_failure error for LID=%s: %s", target_lid, e)
            return False
        self.invalidate_cache(target_lid)
        return bool(response and response.get("status") == "ok")

    def start_heartbeat(self, interval_s: float = 0.5) -> None:
        """Keep this endpoint's directory lease alive in a background thread.

        If the directory no longer knows this endpoint (lease expired or a peer
        reported it), it is re-registered with the last registered addresses.
        """
        if self._hb_thread is not None:
            return
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, args=(interval_s,), daemon=True,
            name=f"connex-dir-hb-{self.stage_idx}-{self.replica_idx}",
        )
        self._hb_thread.start()

    def _heartbeat_loop(self, interval_s: float) -> None:
        import msgpack
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.front_router_addr}")

        def call(msg: dict[str, Any]) -> Optional[dict[str, Any]]:
            sock.send(msgpack.packb(msg, use_bin_type=True))
            if sock.poll(int(interval_s * 2000)):
                return msgpack.unpackb(sock.recv(), raw=False)
            return None

        try:
            while not self._hb_stop.is_set():
                registered = self._registered
                if registered is not None:
                    resp = call({"type": "heartbeat", "lid": list(self.lid)})
                    if resp and resp.get("status") == "not_found" and self._registered is not None:
                        logger.warning("LID=%s evicted from directory, re-registering", self.lid)
                        call(registered)
                        call({"type": "heartbeat", "lid": list(self.lid)})
                self._hb_stop.wait(interval_s)
        finally:
            sock.close()

    def start_event_listener(self, event_address: Optional[str] = None) -> bool:
        """Deliver directory membership events to ``subscribe_events`` handlers.

        Events are received on a background thread from the directory's PUB
        socket. Handlers run on that thread and must be quick. Subscribe
        before taking a membership snapshot (``list_stage``) so that no change
        falls between the two.

        Returns False if the directory does not publish events.
        """
        if self._event_thread is not None:
            return True
        if event_address is None:
            try:
                self._send_control_message({"type": "info"})
                info = self._recv_control_message(timeout_ms=2000)
            except Exception as e:
                logger.error("Directory info request failed: %s", e)
                return False
            if not info or info.get("status") != "ok" or not info.get("event_address"):
                return False
            event_address = info["event_address"]
            host, port = event_address.rsplit(":", 1)
            if host in ("0.0.0.0", "*"):
                host = self.front_router_addr.rsplit(":", 1)[0]
            event_address = f"{host}:{port}"

        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.LINGER, 0)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub_socket.connect(f"tcp://{event_address}")
        connected = threading.Event()
        self._event_thread = threading.Thread(
            target=self._event_loop, args=(connected,), daemon=True,
            name=f"connex-dir-events-{self.stage_idx}",
        )
        self._event_thread.start()
        connected.wait(1.0)
        return True

    def _event_loop(self, connected: threading.Event) -> None:
        import msgpack
        sock = self._sub_socket
        # PUB/SUB drops messages sent before the subscription is established;
        # give the connect a moment before callers snapshot membership.
        sock.poll(100)
        connected.set()
        while self._running:
            if not sock.poll(100):
                continue
            try:
                msg = msgpack.unpackb(sock.recv(), raw=False)
                event = MembershipEvent(msg["event"])
                info = EndpointInfo.from_dict(msg["endpoint"])
            except Exception as e:
                logger.warning("Bad directory event: %s", e)
                continue
            epoch = msg.get("epoch")
            if epoch is not None:
                with self._epoch_lock:
                    self._epoch = max(self._epoch, epoch)
            if event == MembershipEvent.LEAVE:
                self.invalidate_cache(info.lid)
            for handler in list(self._event_handlers):
                try:
                    handler(event, info)
                except Exception:
                    logger.exception("Directory event handler failed")

    def subscribe_events(
        self,
        handler: Callable[[MembershipEvent, EndpointInfo], None]
    ) -> None:
        """Subscribe to membership events.
        
        Args:
            handler: Callback function (event, endpoint) -> None
        """
        self._event_handlers.append(handler)
        
    def get_current_epoch(self) -> int:
        """Get current epoch number."""
        with self._epoch_lock:
            return self._epoch
            
    def increment_epoch(self) -> int:
        """Increment and return new epoch."""
        with self._epoch_lock:
            self._epoch += 1
            return self._epoch
            
    def invalidate_cache(self, target_lid: tuple | None = None) -> None:
        """Invalidate endpoint cache.
        
        Args:
            target_lid: Specific LID to invalidate, or None for all
        """
        with self._cache_lock:
            if target_lid:
                self._endpoint_cache.pop(target_lid, None)
            else:
                self._endpoint_cache.clear()
                
    def _send_control_message(self, message: dict[str, Any]) -> None:
        """Send control message via ZeroMQ."""
        import msgpack
        if self._control_socket:
            data = msgpack.packb(message, use_bin_type=True)
            self._control_socket.send(data)
            
    def _recv_control_message(self, timeout_ms: int = 5000) -> Optional[dict[str, Any]]:
        """Receive control message via ZeroMQ."""
        import msgpack
        if self._control_socket:
            if self._control_socket.poll(timeout_ms):
                data = self._control_socket.recv()
                return msgpack.unpackb(data, raw=False)
        return None
