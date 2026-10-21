# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
EPC (Elastic Pipeline Communications) High-Level API.

Provides a clean API for pipeline communication with endpoint mobility:
- epc_register_lid: Register logical endpoint
- epc_connect: Connect to peer
- epc_send: Send tensor/message
- epc_recv: Receive tensor/message
- epc_cutover: Perform endpoint migration
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from connex.directory import (
    DirectoryClient,
    EndpointInfo,
    MembershipEvent,
    TransportCapability,
)
from connex.route_table import RouteEntry, RouteTable
from connex.cutover_fsm import CutoverContext, CutoverFSM, CutoverState
from connex.stream_manager import StreamManager, StreamMessage
from connex.ucx_transport import TransportMode, UCXTransport
from connex.qos_scheduler import QoSScheduler, TrafficClass

logger = logging.getLogger(__name__)


@dataclass
class EPCConfig:
    """Configuration for EPC endpoint."""
    model_name: str
    stage_idx: int
    num_stages: int
    
    # Network
    local_address: str = ""
    local_port: int = 0
    front_router_addr: str = ""
    
    # Transport
    transport_mode: TransportMode = TransportMode.ZEROMQ
    device: str = "cuda"
    
    # Buffers
    buffer_size: int = 64 * 1024 * 1024  # 64MB
    num_buffers: int = 4
    
    # QoS
    migration_pacing_mbps: float = 1000.0
    
    # Timeouts
    connect_timeout_ms: int = 30000
    recv_timeout_ms: int = -1  # Blocking


class EPCEndpoint:
    """High-level EPC endpoint for pipeline communication.
    
    Integrates all EPC components:
    - DirectoryClient for endpoint discovery
    - RouteTable for versioned routing
    - UCXTransport or ZeroMQ for data transfer
    - StreamManager for ordered delivery
    - CutoverFSM for endpoint migration
    - QoSScheduler for traffic prioritization
    """
    
    def __init__(self, config: EPCConfig):
        """Initialize EPC endpoint.
        
        Args:
            config: Endpoint configuration
        """
        self.config = config
        self.model_name = config.model_name
        self.stage_idx = config.stage_idx
        self.num_stages = config.num_stages
        self.device = config.device
        
        # Logical endpoint ID
        self.lid = (config.model_name, config.stage_idx, 0, 0)
        
        # Initialize components
        self.directory = DirectoryClient(
            front_router_addr=config.front_router_addr,
            model_name=config.model_name,
            stage_idx=config.stage_idx,
            local_addresses=[f"{config.local_address}:{config.local_port}"],
            transport_caps=[
                TransportCapability.UCX_RDMA,
                TransportCapability.UCX_TCP,
                TransportCapability.ZEROMQ,
            ],
        )
        
        self.route_table = RouteTable(
            local_lid=self.lid,
            local_caps=[
                TransportCapability.UCX_RDMA,
                TransportCapability.UCX_TCP,
                TransportCapability.ZEROMQ,
            ],
        )
        
        self.transport = UCXTransport(
            local_address=config.local_address,
            local_port=config.local_port,
            mode=config.transport_mode,
            device=config.device,
            buffer_size=config.buffer_size,
            num_buffers=config.num_buffers,
        )
        
        self.stream_manager = StreamManager(
            stage_idx=config.stage_idx,
        )
        
        self.cutover_fsm = CutoverFSM(
            stage_idx=config.stage_idx,
            on_state_change=self._on_cutover_state_change,
        )
        
        self.qos = QoSScheduler(
            on_send=self._do_send,
            migration_pacing_mbps=config.migration_pacing_mbps,
        )
        
        # Connection state
        self._connections: dict[int, Any] = {}  # stage_idx -> connection
        self._lock = threading.RLock()
        self._running = False
        
        # Timing
        self._init_start_time = time.time()
        self._connection_ready_time = 0.0
        
    @property
    def is_first(self) -> bool:
        return self.stage_idx == 0
        
    @property
    def is_last(self) -> bool:
        return self.stage_idx == self.num_stages - 1
        
    @property
    def backend_name(self) -> str:
        return "connex"
        
    def start(self) -> float:
        """Start the EPC endpoint.
        
        Returns:
            Connection ready time in seconds
        """
        if self._running:
            return self._connection_ready_time
            
        self._running = True
        start_time = time.time()
        
        # Start transport
        self.transport.start_sync()
        
        # Start directory client
        self.directory.start()
        
        # Start QoS scheduler
        self.qos.start()
        
        # Register with directory
        self.directory.register()
        
        self._connection_ready_time = time.time() - start_time
        
        logger.info(
            "EPC endpoint started for stage %d/%d (ready in %.3fs)",
            self.stage_idx, self.num_stages, self._connection_ready_time
        )
        
        return self._connection_ready_time
        
    def stop(self) -> None:
        """Stop the EPC endpoint."""
        self._running = False
        
        self.qos.stop()
        self.directory.stop()
        self.transport.stop_sync()
        
        # Close connections
        with self._lock:
            self._connections.clear()
            
    def connect(self, target_stage: int) -> bool:
        """Connect to a peer stage.
        
        Args:
            target_stage: Target stage index
            
        Returns:
            True if connection successful
        """
        if target_stage in self._connections:
            return True
            
        # Lookup target in directory
        target_lid = (self.model_name, target_stage, 0, 0)
        endpoint = self.directory.lookup(target_lid)
        
        if not endpoint:
            # Try requesting assignment from front router
            result = self.directory.request_assignment(target_stage, "forward")
            if result:
                addr, _ = result
                # Parse address
                if ":" in addr:
                    host, port = addr.rsplit(":", 1)
                    endpoint = EndpointInfo(
                        lid=target_lid,
                        addresses=[addr],
                        transport_caps=[TransportCapability.ZEROMQ],
                    )
                    
        if not endpoint:
            logger.error("Could not find endpoint for stage %d", target_stage)
            return False
            
        # Connect via transport
        try:
            addr = endpoint.primary_address
            if ":" in addr:
                host, port = addr.rsplit(":", 1)
                conn = self.transport.connect_sync(host, int(port))
                
                with self._lock:
                    self._connections[target_stage] = conn
                    
                # Add route
                self.route_table.add_route(
                    target_lid=target_lid,
                    endpoint=endpoint,
                )
                
                logger.info("Connected to stage %d at %s", target_stage, addr)
                return True
                
        except Exception as e:
            logger.error("Failed to connect to stage %d: %s", target_stage, e)
            return False
            
        return False
        
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
    ) -> None:
        """Send tensor dictionary to next stage.
        
        Args:
            tensor_dict: Dictionary of tensors and metadata
            dst: Destination stage (default: next stage)
        """
        if dst is None:
            dst = self.stage_idx + 1
            
        if dst >= self.num_stages:
            logger.warning("Cannot send beyond last stage")
            return
            
        # Ensure connected
        if dst not in self._connections:
            self.connect(dst)
            
        # Serialize
        import msgpack
        
        # Process tensors
        serialized = {}
        tensor_meta = {}
        
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                # Store tensor metadata
                tensor_meta[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "device": str(value.device),
                }
                # Serialize tensor data
                if value.is_cuda:
                    value = value.cpu()
                serialized[key] = value.numpy().tobytes()
            else:
                serialized[key] = value
                
        # Build message
        message = {
            "type": "tensor_dict",
            "data": serialized,
            "tensor_meta": tensor_meta,
            "src_stage": self.stage_idx,
            "dst_stage": dst,
            "epoch": self.route_table.epoch,
            "timestamp": time.time(),
        }
        
        data = msgpack.packb(message, use_bin_type=True)
        
        # Enqueue with high priority
        self.qos.enqueue(
            peer_id=str(dst),
            traffic_class=TrafficClass.TOKEN_CRITICAL,
            data=(dst, data),
            size_bytes=len(data),
        )
        
    def recv_tensor_dict(
        self,
        src: int | None = None,
        timeout_ms: int = -1,
    ) -> Optional[dict[str, Any]]:
        """Receive tensor dictionary from previous stage.
        
        Args:
            src: Source stage (default: previous stage)
            timeout_ms: Receive timeout
            
        Returns:
            Tensor dictionary or None if timeout
        """
        if src is None:
            src = self.stage_idx - 1
            
        if src < 0:
            logger.warning("Cannot receive from before first stage")
            return None
            
        # Receive via transport
        data = self.transport.recv_sync(timeout_ms=timeout_ms)
        
        if not data:
            return None
            
        import msgpack
        message = msgpack.unpackb(data, raw=False)
        
        if message.get("type") != "tensor_dict":
            logger.warning("Unexpected message type: %s", message.get("type"))
            return None
            
        # Reconstruct tensors
        result = {}
        tensor_meta = message.get("tensor_meta", {})
        
        for key, value in message["data"].items():
            if key in tensor_meta:
                meta = tensor_meta[key]
                dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
                shape = tuple(meta["shape"])
                
                import numpy as np
                arr = np.frombuffer(value, dtype=torch.tensor([], dtype=dtype).numpy().dtype)
                arr = arr.reshape(shape)
                tensor = torch.from_numpy(arr.copy())
                
                # Move to target device
                if self.device == "cuda":
                    tensor = tensor.cuda()
                    
                result[key] = tensor
            else:
                result[key] = value
                
        return result
        
    def prepare_migrate(
        self,
        stream_id: str,
        new_address: str,
        new_epoch: int,
    ) -> bool:
        """Prepare for endpoint migration.
        
        Args:
            stream_id: Stream to migrate
            new_address: New endpoint address
            new_epoch: Target epoch
            
        Returns:
            True if preparation successful
        """
        old_addr = f"{self.config.local_address}:{self.config.local_port}"
        
        ctx = self.cutover_fsm.start_cutover(
            stream_id=stream_id,
            old_endpoint=old_addr,
            new_endpoint=new_address,
            old_epoch=self.route_table.epoch,
            new_epoch=new_epoch,
        )
        
        # Notify directory
        success = self.directory.notify_migration(
            new_addresses=[new_address],
            target_epoch=new_epoch,
        )
        
        if success:
            self.cutover_fsm.prepare_complete(stream_id, new_connection=None)
        else:
            self.cutover_fsm.prepare_failed(stream_id, "Directory notification failed")
            
        return success
        
    def cutover(self, stream_id: str) -> bool:
        """Perform traffic cutover.
        
        Args:
            stream_id: Stream to cutover
            
        Returns:
            True if cutover successful
        """
        # Get current sequence number
        state = self.stream_manager.table.get_or_create_stream(stream_id)
        last_seqno = state.next_send_seqno - 1
        
        # Start traffic switch
        self.cutover_fsm.start_traffic_switch(stream_id, last_seqno)
        
        # For now, immediately complete (in real impl, would wait for acks)
        self.cutover_fsm.cutover_complete(stream_id)
        
        # Update epoch
        self.route_table.increment_epoch()
        
        # Drain and finalize
        self.cutover_fsm.update_drain_progress(stream_id, last_seqno)
        self.cutover_fsm.finalize(stream_id)
        
        return True
        
    def update_credits(
        self,
        peer_stage: int,
        traffic_class: TrafficClass,
        credits: int,
    ) -> None:
        """Update credits for a peer.
        
        Args:
            peer_stage: Peer stage index
            traffic_class: Traffic class
            credits: Credits to add
        """
        self.qos.process_credits(
            peer_id=str(peer_stage),
            credits_update={traffic_class: credits},
        )
        
    def _do_send(self, msg: Any) -> bool:
        """Internal send callback for QoS scheduler."""
        try:
            dst, data = msg.data
            conn = self._connections.get(dst)
            
            if conn:
                self.transport.send_sync(conn, data)
                return True
            else:
                # Try direct transport send
                self.transport.send_sync(None, data)
                return True
                
        except Exception as e:
            logger.error("Send failed: %s", e)
            return False
            
    def _on_cutover_state_change(
        self,
        state: CutoverState,
        ctx: CutoverContext,
    ) -> None:
        """Handle cutover state changes."""
        logger.info(
            "Stage %d: Cutover state change for %s: %s",
            self.stage_idx, ctx.stream_id, state.name
        )
        
        if state == CutoverState.CUTTING_OVER:
            # Set watermark in stream manager
            self.stream_manager.table.set_cutover_watermark(
                ctx.stream_id,
                ctx.last_old_seqno,
            )
        elif state == CutoverState.IDLE:
            # Clear watermark
            self.stream_manager.table.clear_cutover_watermark(ctx.stream_id)
            
    def get_stats(self) -> dict[str, Any]:
        """Get endpoint statistics."""
        return {
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "backend": self.backend_name,
            "connection_ready_time": self._connection_ready_time,
            "epoch": self.route_table.epoch,
            "connections": len(self._connections),
            "transport": self.transport.get_stats(),
            "routes": self.route_table.get_stats(),
            "streams": self.stream_manager.get_stats(),
            "cutover": self.cutover_fsm.get_stats(),
            "qos": self.qos.get_stats(),
        }


# Convenience functions for simple usage
def create_endpoint(
    model_name: str,
    stage_idx: int,
    num_stages: int,
    front_router_addr: str,
    local_address: str = "",
    local_port: int = 0,
    device: str = "cuda",
) -> EPCEndpoint:
    """Create an EPC endpoint with default configuration.
    
    Args:
        model_name: Model name
        stage_idx: Pipeline stage index
        num_stages: Total pipeline stages
        front_router_addr: Front router address
        local_address: Local bind address
        local_port: Local bind port
        device: Device for tensors
        
    Returns:
        Configured EPCEndpoint
    """
    config = EPCConfig(
        model_name=model_name,
        stage_idx=stage_idx,
        num_stages=num_stages,
        local_address=local_address or "0.0.0.0",
        local_port=local_port or (15550 + stage_idx),
        front_router_addr=front_router_addr,
        device=device,
    )
    
    return EPCEndpoint(config)
