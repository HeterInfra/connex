# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Mooncake Transfer Engine (TE) Integration for KV Cache Transfer.

Provides mobility-aware KV cache transfer with:
- Connex upper-layer interface compatible with Mooncake TE bottom layer
- Endpoint mobility support for KV cache disaggregation
- Prefetch and pipeline optimization

Design:
- Pipeline communication uses Connex
- KV cache transfer uses Mooncake TE (capability integrated into Connex)
- Connex provides mobility-aware transport runtime for Mooncake

Design note: Connex carries pipeline (activation) traffic; Mooncake TE carries KV-cache transfers.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from connex.protocol import (
    LogicalEndpointID,
    MessageClass,
    MessageType,
    MobilityMessage,
    ProtocolHeader,
    StreamID,
)
from connex.mobility_control import (
    ConnectionCapability,
    VersionedRoute,
    VersionedRouteTable,
    ShadowConnectionManager,
)
from connex.mobility_dataplane import (
    MultiQueueScheduler,
    StreamMultiplexer,
)
from connex.gpu_fastpath import (
    GPUAwareFastPath,
    RegistrationCache,
    ChunkedTensorTransfer,
)

logger = logging.getLogger(__name__)


class KVTransferStatus(Enum):
    """Status of KV cache transfer operation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class KVCacheBlock:
    """Represents a KV cache block for transfer."""
    # Block identification
    request_id: str
    layer_idx: int
    head_idx: int
    
    # Cache data
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    
    # Metadata
    seq_len: int = 0
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    
    # Transfer state
    status: KVTransferStatus = KVTransferStatus.PENDING
    transfer_id: Optional[str] = None
    
    @property
    def size_bytes(self) -> int:
        """Total size in bytes."""
        size = 0
        if self.key_cache is not None:
            size += self.key_cache.numel() * self.key_cache.element_size()
        if self.value_cache is not None:
            size += self.value_cache.numel() * self.value_cache.element_size()
        return size
    
    def to_dict(self) -> dict:
        """Serialize metadata (not tensors) to dict."""
        return {
            "request_id": self.request_id,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "seq_len": self.seq_len,
            "dtype": str(self.dtype),
            "device": self.device,
            "status": self.status.name,
            "transfer_id": self.transfer_id,
            "size_bytes": self.size_bytes,
        }


@dataclass
class KVTransferRequest:
    """Request for KV cache transfer."""
    # Request identification
    request_id: str
    transfer_id: str
    
    # Source and destination
    src_lid: LogicalEndpointID
    dst_lid: LogicalEndpointID
    
    # Blocks to transfer
    blocks: List[KVCacheBlock]
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    
    # Status
    status: KVTransferStatus = KVTransferStatus.PENDING
    error: Optional[str] = None
    
    # Progress
    blocks_completed: int = 0
    bytes_transferred: int = 0
    
    @property
    def total_blocks(self) -> int:
        return len(self.blocks)
    
    @property
    def total_bytes(self) -> int:
        return sum(b.size_bytes for b in self.blocks)
    
    @property
    def progress(self) -> float:
        return self.blocks_completed / max(self.total_blocks, 1)
    
    @property
    def elapsed_s(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.completed_at if self.completed_at > 0 else time.time()
        return end - self.started_at
    
    @property
    def throughput_gbps(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return (self.bytes_transferred * 8) / (self.elapsed_s * 1e9)


class MooncakeTEInterface(ABC):
    """Abstract interface matching Mooncake Transfer Engine API.
    
    This interface allows Connex to provide mobility-aware transport
    that is compatible with Mooncake TE's bottom-layer requirements.
    """
    
    @abstractmethod
    def init_transfer_engine(
        self,
        local_addr: str,
        config: Dict[str, Any],
    ) -> bool:
        """Initialize the transfer engine.
        
        Args:
            local_addr: Local endpoint address
            config: Engine configuration
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def register_buffer(
        self,
        buffer: torch.Tensor,
        buffer_id: str,
    ) -> bool:
        """Register a buffer for RDMA transfer.
        
        Args:
            buffer: GPU/CPU tensor buffer
            buffer_id: Unique buffer identifier
            
        Returns:
            True if registered
        """
        pass
    
    @abstractmethod
    def unregister_buffer(self, buffer_id: str) -> bool:
        """Unregister a buffer.
        
        Args:
            buffer_id: Buffer identifier
            
        Returns:
            True if unregistered
        """
        pass
    
    @abstractmethod
    def send_async(
        self,
        buffer_id: str,
        dst_addr: str,
        offset: int,
        size: int,
    ) -> str:
        """Asynchronously send buffer data.
        
        Args:
            buffer_id: Source buffer
            dst_addr: Destination address
            offset: Offset in buffer
            size: Size to send
            
        Returns:
            Transfer ID for tracking
        """
        pass
    
    @abstractmethod
    def recv_async(
        self,
        buffer_id: str,
        src_addr: str,
        offset: int,
        size: int,
    ) -> str:
        """Asynchronously receive into buffer.
        
        Args:
            buffer_id: Destination buffer
            src_addr: Source address
            offset: Offset in buffer
            size: Size to receive
            
        Returns:
            Transfer ID for tracking
        """
        pass
    
    @abstractmethod
    def poll_completion(
        self,
        transfer_id: str,
        timeout_ms: int = 0,
    ) -> Optional[bool]:
        """Poll for transfer completion.
        
        Args:
            transfer_id: Transfer to check
            timeout_ms: Timeout (0 = non-blocking)
            
        Returns:
            True if complete, False if failed, None if pending
        """
        pass
    
    @abstractmethod
    def wait_completion(
        self,
        transfer_id: str,
        timeout_ms: int = -1,
    ) -> bool:
        """Wait for transfer completion.
        
        Args:
            transfer_id: Transfer to wait for
            timeout_ms: Timeout (-1 = infinite)
            
        Returns:
            True if completed successfully
        """
        pass


class ConnexMooncakeAdapter(MooncakeTEInterface):
    """Connex adapter implementing Mooncake TE interface.
    
    Provides mobility-aware transport for Mooncake's KV cache transfers.
    
    Features:
    - Endpoint mobility support (migration without interruption)
    - Registration caching for fast reconnection
    - QoS-aware scheduling (KV_CACHE priority class)
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        route_table: VersionedRouteTable,
        scheduler: MultiQueueScheduler,
        fast_path: GPUAwareFastPath,
        on_transfer_complete: Optional[Callable[[str, bool], None]] = None,
    ):
        """Initialize Connex-Mooncake adapter.
        
        Args:
            local_lid: Local endpoint LID
            route_table: Versioned route table
            scheduler: Multi-queue scheduler
            fast_path: GPU-aware fast path
            on_transfer_complete: Callback when transfer completes
        """
        self.local_lid = local_lid
        self.route_table = route_table
        self.scheduler = scheduler
        self.fast_path = fast_path
        self.on_transfer_complete = on_transfer_complete
        
        # Buffer registry
        self._buffers: Dict[str, torch.Tensor] = {}
        self._buffer_lock = threading.Lock()
        
        # Active transfers
        self._transfers: Dict[str, Dict[str, Any]] = {}
        self._transfer_lock = threading.Lock()
        
        # Transfer ID counter
        self._transfer_counter = 0
        
        # Statistics
        self._stats = {
            "buffers_registered": 0,
            "transfers_started": 0,
            "transfers_completed": 0,
            "transfers_failed": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
    
    def init_transfer_engine(
        self,
        local_addr: str,
        config: Dict[str, Any],
    ) -> bool:
        """Initialize transfer engine with Connex backend."""
        logger.info(
            "Initializing Connex-Mooncake adapter at %s",
            local_addr
        )
        
        # Configure based on provided config
        if "chunk_size" in config:
            self.fast_path.chunked.default_chunk_size = config["chunk_size"]
        
        if "cache_max_entries" in config:
            # Would need to reinitialize cache, skip for now
            pass
        
        return True
    
    def register_buffer(
        self,
        buffer: torch.Tensor,
        buffer_id: str,
    ) -> bool:
        """Register buffer and pre-warm registration cache."""
        with self._buffer_lock:
            if buffer_id in self._buffers:
                logger.warning("Buffer %s already registered", buffer_id)
                return True
            
            self._buffers[buffer_id] = buffer
            
            # Pre-register in cache
            reg = self.fast_path._do_registration(buffer)
            self.fast_path.reg_cache.put(
                buffer.data_ptr(),
                buffer.numel() * buffer.element_size(),
                str(buffer.device),
                reg,
            )
            
            self._stats["buffers_registered"] += 1
            
        logger.debug(
            "Registered buffer %s: %d bytes on %s",
            buffer_id, buffer.numel() * buffer.element_size(), buffer.device
        )
        return True
    
    def unregister_buffer(self, buffer_id: str) -> bool:
        """Unregister buffer and invalidate cache entry."""
        with self._buffer_lock:
            buffer = self._buffers.pop(buffer_id, None)
            if buffer is None:
                return False
            
            # Invalidate cache entry
            self.fast_path.reg_cache.invalidate(
                buffer.data_ptr(),
                buffer.numel() * buffer.element_size(),
                str(buffer.device),
            )
        
        logger.debug("Unregistered buffer %s", buffer_id)
        return True
    
    def _generate_transfer_id(self) -> str:
        """Generate unique transfer ID."""
        self._transfer_counter += 1
        return f"kv-{self.local_lid.stage_idx}-{self._transfer_counter}-{int(time.time()*1000)}"
    
    def send_async(
        self,
        buffer_id: str,
        dst_addr: str,
        offset: int,
        size: int,
    ) -> str:
        """Send buffer data asynchronously."""
        transfer_id = self._generate_transfer_id()
        
        with self._buffer_lock:
            buffer = self._buffers.get(buffer_id)
            if buffer is None:
                raise ValueError(f"Buffer {buffer_id} not registered")
        
        # Parse destination address to LID
        dst_lid = self._addr_to_lid(dst_addr)
        
        # Create stream ID for this transfer
        stream_id = StreamID(
            request_id=transfer_id,
            direction="kv_send",
            token_step=0,
        )
        
        # Extract slice of buffer
        flat = buffer.view(-1)
        elem_size = buffer.element_size()
        start_elem = offset // elem_size
        end_elem = start_elem + (size // elem_size)
        data_slice = flat[start_elem:end_elem]
        
        # Track transfer
        with self._transfer_lock:
            self._transfers[transfer_id] = {
                "type": "send",
                "buffer_id": buffer_id,
                "dst_lid": dst_lid,
                "offset": offset,
                "size": size,
                "stream_id": stream_id,
                "started_at": time.time(),
                "completed": False,
                "success": None,
            }
        
        # Start chunked send
        self.fast_path.prepare_send(data_slice, dst_lid, stream_id)
        
        # Enqueue with KV_CACHE priority
        self._enqueue_transfer(transfer_id, data_slice, dst_lid, stream_id)
        
        self._stats["transfers_started"] += 1
        self._stats["bytes_sent"] += size
        
        logger.debug(
            "Started async send %s: %d bytes to %s",
            transfer_id, size, dst_lid
        )
        
        return transfer_id
    
    def recv_async(
        self,
        buffer_id: str,
        src_addr: str,
        offset: int,
        size: int,
    ) -> str:
        """Receive data into buffer asynchronously."""
        transfer_id = self._generate_transfer_id()
        
        with self._buffer_lock:
            buffer = self._buffers.get(buffer_id)
            if buffer is None:
                raise ValueError(f"Buffer {buffer_id} not registered")
        
        # Parse source address to LID
        src_lid = self._addr_to_lid(src_addr)
        
        # Create stream ID
        stream_id = StreamID(
            request_id=transfer_id,
            direction="kv_recv",
            token_step=0,
        )
        
        # Track transfer
        with self._transfer_lock:
            self._transfers[transfer_id] = {
                "type": "recv",
                "buffer_id": buffer_id,
                "src_lid": src_lid,
                "offset": offset,
                "size": size,
                "stream_id": stream_id,
                "buffer": buffer,
                "started_at": time.time(),
                "completed": False,
                "success": None,
            }
        
        self._stats["transfers_started"] += 1
        self._stats["bytes_received"] += size
        
        logger.debug(
            "Started async recv %s: %d bytes from %s",
            transfer_id, size, src_lid
        )
        
        return transfer_id
    
    def _addr_to_lid(self, addr: str) -> LogicalEndpointID:
        """Convert address string to LID.
        
        Address formats:
        - Legacy: "model:stage:shard:replica@host:port"
        - PD mode: "model:stage:shard:replica:role@host:port"
        
        Where role is one of: "prefill", "decode", "none"
        """
        from connex.protocol import PDRole
        
        if "@" in addr:
            lid_part = addr.split("@")[0]
        else:
            lid_part = addr
        
        parts = lid_part.split(":")
        
        # 5-part format with PD role
        if len(parts) >= 5:
            role = parts[4] if parts[4] in (PDRole.PREFILL, PDRole.DECODE, PDRole.NONE) else PDRole.NONE
            return LogicalEndpointID(
                model_name=parts[0],
                stage_idx=int(parts[1]),
                shard_idx=int(parts[2]),
                replica_idx=int(parts[3]),
                pd_role=role,
            )
        
        # 4-part legacy format (no PD role)
        if len(parts) >= 4:
            return LogicalEndpointID(
                model_name=parts[0],
                stage_idx=int(parts[1]),
                shard_idx=int(parts[2]),
                replica_idx=int(parts[3]),
                pd_role=PDRole.NONE,
            )
        
        # Default parsing
        return LogicalEndpointID(
            model_name="default",
            stage_idx=0,
            shard_idx=0,
            replica_idx=0,
            pd_role=PDRole.NONE,
        )
    
    def _lid_to_addr(self, lid: LogicalEndpointID, host: str = "", port: int = 0) -> str:
        """Convert LID to address string.
        
        Returns format: "model:stage:shard:replica:role@host:port"
        """
        lid_str = f"{lid.model_name}:{lid.stage_idx}:{lid.shard_idx}:{lid.replica_idx}:{lid.pd_role}"
        if host and port:
            return f"{lid_str}@{host}:{port}"
        return lid_str
    
    def _enqueue_transfer(
        self,
        transfer_id: str,
        data: torch.Tensor,
        dst_lid: LogicalEndpointID,
        stream_id: StreamID,
    ) -> None:
        """Enqueue transfer data for sending."""
        import msgpack
        
        # Serialize tensor data
        if data.is_cuda:
            data_bytes = data.cpu().numpy().tobytes()
        else:
            data_bytes = data.numpy().tobytes()
        
        # Create message
        header = ProtocolHeader(
            msg_type=MessageType.DATA,
            epoch=self.route_table.epoch,
            seqno=0,  # Will be set by multiplexer
            msg_class=MessageClass.KV_CACHE,
            stream_id=stream_id,
            src_lid=self.local_lid,
            dst_lid=dst_lid,
            payload_len=len(data_bytes),
        )
        
        message = MobilityMessage(header=header, payload=data_bytes)
        
        # Enqueue
        self.scheduler.enqueue(
            peer_lid=dst_lid,
            message=message,
            stream_id=stream_id,
            priority=MessageClass.KV_CACHE,
        )
    
    def poll_completion(
        self,
        transfer_id: str,
        timeout_ms: int = 0,
    ) -> Optional[bool]:
        """Poll for transfer completion."""
        with self._transfer_lock:
            state = self._transfers.get(transfer_id)
            if not state:
                return None
            
            if state["completed"]:
                return state["success"]
        
        # Non-blocking check
        if timeout_ms == 0:
            return None
        
        # Brief wait and recheck
        time.sleep(timeout_ms / 1000.0)
        
        with self._transfer_lock:
            state = self._transfers.get(transfer_id)
            if state and state["completed"]:
                return state["success"]
        
        return None
    
    def wait_completion(
        self,
        transfer_id: str,
        timeout_ms: int = -1,
    ) -> bool:
        """Wait for transfer completion."""
        start = time.time()
        timeout_s = timeout_ms / 1000.0 if timeout_ms >= 0 else float('inf')
        
        while True:
            with self._transfer_lock:
                state = self._transfers.get(transfer_id)
                if not state:
                    return False
                
                if state["completed"]:
                    return state["success"]
            
            # Check timeout
            if time.time() - start > timeout_s:
                return False
            
            time.sleep(0.001)  # 1ms polling
    
    def mark_complete(self, transfer_id: str, success: bool) -> None:
        """Mark a transfer as complete (called by transport layer)."""
        with self._transfer_lock:
            state = self._transfers.get(transfer_id)
            if not state:
                return
            
            state["completed"] = True
            state["success"] = success
            state["completed_at"] = time.time()
        
        if success:
            self._stats["transfers_completed"] += 1
        else:
            self._stats["transfers_failed"] += 1
        
        if self.on_transfer_complete:
            self.on_transfer_complete(transfer_id, success)
        
        logger.debug(
            "Transfer %s completed: %s",
            transfer_id, "success" if success else "failed"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        with self._transfer_lock:
            active = len([t for t in self._transfers.values() if not t["completed"]])
        
        return {
            **self._stats,
            "active_transfers": active,
        }


class KVCacheTransferManager:
    """High-level manager for KV cache transfers.
    
    Provides:
    - Batch transfer operations
    - Prefetch and pipelining
    - Migration-aware transfer scheduling
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        mooncake_adapter: ConnexMooncakeAdapter,
        prefetch_depth: int = 2,
    ):
        """Initialize KV cache transfer manager.
        
        Args:
            local_lid: Local endpoint LID
            mooncake_adapter: Mooncake TE adapter
            prefetch_depth: Number of blocks to prefetch
        """
        self.local_lid = local_lid
        self.adapter = mooncake_adapter
        self.prefetch_depth = prefetch_depth
        
        # Active transfer requests
        self._requests: Dict[str, KVTransferRequest] = {}
        self._lock = threading.Lock()
        
        # Block buffer pool (pre-registered)
        self._block_buffers: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self._stats = {
            "requests_started": 0,
            "requests_completed": 0,
            "blocks_transferred": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
        }
    
    def allocate_block_buffer(
        self,
        layer_idx: int,
        head_idx: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[str, torch.Tensor]:
        """Allocate and register a block buffer.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            seq_len: Sequence length
            head_dim: Head dimension
            dtype: Data type
            
        Returns:
            (buffer_id, buffer_tensor)
        """
        buffer_id = f"kv-L{layer_idx}-H{head_idx}-{seq_len}"
        
        # Allocate K and V together (2x size)
        if torch.cuda.is_available():
            buffer = torch.empty(
                2, seq_len, head_dim,
                dtype=dtype,
                device="cuda",
            )
        else:
            buffer = torch.empty(
                2, seq_len, head_dim,
                dtype=dtype,
            )
        
        # Register with adapter
        self.adapter.register_buffer(buffer, buffer_id)
        self._block_buffers[buffer_id] = buffer
        
        return buffer_id, buffer
    
    def transfer_blocks(
        self,
        request_id: str,
        blocks: List[KVCacheBlock],
        dst_lid: LogicalEndpointID,
    ) -> str:
        """Transfer multiple KV cache blocks.
        
        Args:
            request_id: Request identifier
            blocks: Blocks to transfer
            dst_lid: Destination endpoint
            
        Returns:
            Transfer request ID
        """
        transfer_id = f"batch-{request_id}-{int(time.time()*1000)}"
        
        request = KVTransferRequest(
            request_id=request_id,
            transfer_id=transfer_id,
            src_lid=self.local_lid,
            dst_lid=dst_lid,
            blocks=blocks,
        )
        
        with self._lock:
            self._requests[transfer_id] = request
        
        # Start transfer in background
        threading.Thread(
            target=self._do_batch_transfer,
            args=(request,),
            daemon=True,
        ).start()
        
        self._stats["requests_started"] += 1
        
        return transfer_id
    
    def _do_batch_transfer(self, request: KVTransferRequest) -> None:
        """Execute batch transfer with pipelining."""
        request.status = KVTransferStatus.IN_PROGRESS
        request.started_at = time.time()
        
        active_transfers: Dict[str, KVCacheBlock] = {}
        
        try:
            block_idx = 0
            
            while block_idx < len(request.blocks) or active_transfers:
                # Start new transfers up to prefetch depth
                while (
                    block_idx < len(request.blocks) and
                    len(active_transfers) < self.prefetch_depth
                ):
                    block = request.blocks[block_idx]
                    
                    # Get or allocate buffer
                    buffer_id = f"kv-L{block.layer_idx}-H{block.head_idx}-{block.seq_len}"
                    
                    if buffer_id not in self._block_buffers:
                        # Need to register block's tensor
                        if block.key_cache is not None:
                            combined = torch.stack([block.key_cache, block.value_cache])
                            self.adapter.register_buffer(combined, buffer_id)
                            self._block_buffers[buffer_id] = combined
                    
                    # Start async send
                    dst_addr = self._lid_to_addr(request.dst_lid)
                    tid = self.adapter.send_async(
                        buffer_id,
                        dst_addr,
                        offset=0,
                        size=block.size_bytes,
                    )
                    
                    block.transfer_id = tid
                    block.status = KVTransferStatus.IN_PROGRESS
                    active_transfers[tid] = block
                    block_idx += 1
                
                # Poll for completions
                completed = []
                for tid, block in active_transfers.items():
                    result = self.adapter.poll_completion(tid, timeout_ms=1)
                    if result is not None:
                        completed.append(tid)
                        if result:
                            block.status = KVTransferStatus.COMPLETED
                            request.blocks_completed += 1
                            request.bytes_transferred += block.size_bytes
                            self._stats["blocks_transferred"] += 1
                        else:
                            block.status = KVTransferStatus.FAILED
                
                for tid in completed:
                    del active_transfers[tid]
                
                if active_transfers:
                    time.sleep(0.001)  # Brief yield
            
            request.status = KVTransferStatus.COMPLETED
            request.completed_at = time.time()
            self._stats["requests_completed"] += 1
            
            logger.info(
                "Batch transfer %s completed: %d blocks, %.2f GB/s",
                request.transfer_id,
                request.blocks_completed,
                request.throughput_gbps,
            )
            
        except Exception as e:
            request.status = KVTransferStatus.FAILED
            request.error = str(e)
            logger.error("Batch transfer %s failed: %s", request.transfer_id, e)
    
    # ==================== PD Separation Support ====================
    
    def transfer_to_decode_pool(
        self,
        request_id: str,
        blocks: List[KVCacheBlock],
        decode_selector: Optional[Callable[["List[VersionedRoute]"], "VersionedRoute"]] = None,
    ) -> str:
        """Transfer KV cache from prefill to any available decode instance.
        
        This is used in Prefill-Decode separation mode where:
        1. Prefill instance runs prompt processing
        2. KV cache is transferred to decode instance pool
        3. Decode instance continues token generation
        
        Args:
            request_id: Request identifier
            blocks: KV cache blocks to transfer
            decode_selector: Optional function to select decode target from pool.
                            If None, uses round-robin load balancing.
            
        Returns:
            Transfer request ID
            
        Raises:
            RuntimeError: If no decode instances are available
        """
        from connex.protocol import PDRole
        
        # Get all available decode endpoints
        decode_routes = self.adapter.route_table.get_decode_endpoints()
        
        if not decode_routes:
            raise RuntimeError(
                "No decode instances available for KV transfer. "
                "Ensure decode instances are registered in the route table."
            )
        
        # Select target decode instance
        if decode_selector:
            target_route = decode_selector(decode_routes)
        else:
            # Default: round-robin based on request_id hash
            idx = hash(request_id) % len(decode_routes)
            target_route = decode_routes[idx]
        
        logger.info(
            "Transferring KV cache for request %s to decode instance %s (%d blocks)",
            request_id, target_route.lid, len(blocks)
        )
        
        # Perform the transfer
        return self.transfer_blocks(request_id, blocks, target_route.lid)
    
    def receive_from_prefill(
        self,
        request_id: str,
        block_specs: List[Dict[str, Any]],
        prefill_lid: Optional[LogicalEndpointID] = None,
    ) -> str:
        """Receive KV cache blocks from a prefill instance.
        
        Used by decode instances to receive KV cache after prefill completes.
        
        Args:
            request_id: Request identifier
            block_specs: Specifications for blocks to receive
            prefill_lid: Source prefill LID. If None, uses first available.
            
        Returns:
            Transfer request ID
            
        Raises:
            RuntimeError: If no prefill instances are available
        """
        if prefill_lid is None:
            # Find available prefill instance
            prefill_routes = self.adapter.route_table.get_prefill_endpoints()
            if not prefill_routes:
                raise RuntimeError(
                    "No prefill instances available for KV receive."
                )
            prefill_lid = prefill_routes[0].lid
        
        return self.prefetch_blocks(request_id, block_specs, prefill_lid)
    
    def _lid_to_addr(self, lid: LogicalEndpointID) -> str:
        """Convert LID to address string (with PD role support)."""
        return f"{lid.model_name}:{lid.stage_idx}:{lid.shard_idx}:{lid.replica_idx}:{lid.pd_role}"
    
    def get_request_status(self, transfer_id: str) -> Optional[KVTransferRequest]:
        """Get transfer request status."""
        with self._lock:
            return self._requests.get(transfer_id)
    
    def wait_request(self, transfer_id: str, timeout_s: float = -1) -> bool:
        """Wait for transfer request to complete.
        
        Args:
            transfer_id: Transfer request ID
            timeout_s: Timeout in seconds (-1 = infinite)
            
        Returns:
            True if completed successfully
        """
        start = time.time()
        
        while True:
            with self._lock:
                request = self._requests.get(transfer_id)
                if not request:
                    return False
                
                if request.status == KVTransferStatus.COMPLETED:
                    return True
                if request.status == KVTransferStatus.FAILED:
                    return False
            
            if timeout_s >= 0 and time.time() - start > timeout_s:
                return False
            
            time.sleep(0.01)
    
    def prefetch_blocks(
        self,
        request_id: str,
        block_specs: List[Dict[str, Any]],
        src_lid: LogicalEndpointID,
    ) -> str:
        """Prefetch blocks from remote endpoint.
        
        Args:
            request_id: Request identifier
            block_specs: List of block specifications
            src_lid: Source endpoint
            
        Returns:
            Prefetch request ID
        """
        transfer_id = f"prefetch-{request_id}-{int(time.time()*1000)}"
        
        # Allocate receive buffers
        blocks = []
        for spec in block_specs:
            buffer_id, buffer = self.allocate_block_buffer(
                layer_idx=spec.get("layer_idx", 0),
                head_idx=spec.get("head_idx", 0),
                seq_len=spec.get("seq_len", 128),
                head_dim=spec.get("head_dim", 64),
                dtype=torch.float16,
            )
            
            block = KVCacheBlock(
                request_id=request_id,
                layer_idx=spec.get("layer_idx", 0),
                head_idx=spec.get("head_idx", 0),
                seq_len=spec.get("seq_len", 128),
            )
            blocks.append(block)
        
        request = KVTransferRequest(
            request_id=request_id,
            transfer_id=transfer_id,
            src_lid=src_lid,
            dst_lid=self.local_lid,
            blocks=blocks,
        )
        
        with self._lock:
            self._requests[transfer_id] = request
        
        # Start prefetch
        # (Implementation would be similar to _do_batch_transfer but with recv_async)
        
        return transfer_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            active = len([r for r in self._requests.values() 
                         if r.status == KVTransferStatus.IN_PROGRESS])
        
        return {
            **self._stats,
            "active_requests": active,
            "registered_buffers": len(self._block_buffers),
            "adapter": self.adapter.get_stats(),
        }
