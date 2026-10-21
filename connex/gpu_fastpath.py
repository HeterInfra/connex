# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
GPU-Aware Fast Path for High-Performance Tensor Transfer.

Implements GPU-optimized data transfer with mobility support:
- Capability-aware transport selection (RDMA, GPU Direct, host staging)
- Registration cache pooling for fast reconnection
- Chunked tensor transfer with compute overlap

Key insight: Mobility semantics don't mean slow.
Fast path becomes pluggable and capability-negotiation driven.

"""

import logging
import os
import threading
import time
from collections import OrderedDict
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
from connex.mobility_control import ConnectionCapability, VersionedRoute

logger = logging.getLogger(__name__)


class TransportCapability(Enum):
    """Available transport capabilities for data transfer."""
    GPU_DIRECT_RDMA = auto()    # Direct GPU-to-GPU RDMA
    GPU_DIRECT_TCP = auto()     # GPU Direct with TCP
    HOST_RDMA = auto()          # Host memory RDMA
    HOST_TCP = auto()           # Host memory TCP
    CPU_PROXY = auto()          # CPU bounce buffer
    SHARED_MEMORY = auto()      # Local shared memory


@dataclass
class TransportSelection:
    """Result of transport capability negotiation."""
    selected: TransportCapability
    fallback: Optional[TransportCapability] = None
    
    # Capabilities used
    use_registered_memory: bool = False
    use_zero_copy: bool = False
    use_pinned_buffer: bool = False
    
    # Performance hints
    max_chunk_size: int = 16 * 1024 * 1024  # 16MB
    overlap_enabled: bool = True


@dataclass
class RegistrationHandle:
    """Handle for registered memory region."""
    buffer_ptr: int               # GPU/CPU buffer pointer
    size: int                     # Buffer size
    device: str                   # Device ("cuda:X" or "cpu")
    registration: Any = None      # Transport-specific registration
    
    # Lifecycle tracking
    ref_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    # For cache management
    cache_key: Optional[str] = None
    
    def touch(self) -> None:
        """Update last used time."""
        self.last_used = time.time()
    
    def add_ref(self) -> int:
        """Add reference."""
        self.ref_count += 1
        return self.ref_count
    
    def release_ref(self) -> int:
        """Release reference."""
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count


class RegistrationCache:
    """LRU cache for memory registrations.
    
    Avoids expensive re-registration during migration/reconnection.
    Thread-safe with reference counting.
    """
    
    def __init__(
        self,
        max_entries: int = 256,
        max_memory_gb: float = 16.0,
        on_evict: Optional[Callable[[RegistrationHandle], None]] = None,
    ):
        """Initialize registration cache.
        
        Args:
            max_entries: Maximum cached registrations
            max_memory_gb: Maximum memory to track
            on_evict: Callback when entry evicted
        """
        self.max_entries = max_entries
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.on_evict = on_evict
        
        # LRU cache: key -> RegistrationHandle
        self._cache: OrderedDict[str, RegistrationHandle] = OrderedDict()
        self._lock = threading.RLock()
        
        # Memory tracking
        self._total_size = 0
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "registrations": 0,
        }
    
    def _make_key(self, buffer_ptr: int, size: int, device: str) -> str:
        """Create cache key from buffer info."""
        return f"{device}:{buffer_ptr}:{size}"
    
    def get(
        self,
        buffer_ptr: int,
        size: int,
        device: str,
    ) -> Optional[RegistrationHandle]:
        """Get cached registration.
        
        Args:
            buffer_ptr: Buffer pointer/ID
            size: Buffer size
            device: Device string
            
        Returns:
            Cached handle or None
        """
        key = self._make_key(buffer_ptr, size, device)
        
        with self._lock:
            handle = self._cache.get(key)
            if handle:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                handle.touch()
                handle.add_ref()
                self._stats["hits"] += 1
                return handle
            
            self._stats["misses"] += 1
            return None
    
    def put(
        self,
        buffer_ptr: int,
        size: int,
        device: str,
        registration: Any,
    ) -> RegistrationHandle:
        """Cache a registration.
        
        Args:
            buffer_ptr: Buffer pointer
            size: Buffer size
            device: Device string
            registration: Transport-specific registration handle
            
        Returns:
            New or existing handle
        """
        key = self._make_key(buffer_ptr, size, device)
        
        with self._lock:
            # Check if already cached
            if key in self._cache:
                handle = self._cache[key]
                self._cache.move_to_end(key)
                handle.touch()
                return handle
            
            # Evict if needed
            self._ensure_capacity(size)
            
            # Create new entry
            handle = RegistrationHandle(
                buffer_ptr=buffer_ptr,
                size=size,
                device=device,
                registration=registration,
                cache_key=key,
            )
            handle.add_ref()
            
            self._cache[key] = handle
            self._total_size += size
            self._stats["registrations"] += 1
            
            return handle
    
    def _ensure_capacity(self, needed_size: int) -> None:
        """Evict entries if needed to make room."""
        while (
            len(self._cache) >= self.max_entries or
            self._total_size + needed_size > self.max_memory_bytes
        ) and self._cache:
            # Evict oldest (first in OrderedDict)
            oldest_key, oldest_handle = self._cache.popitem(last=False)
            self._total_size -= oldest_handle.size
            self._stats["evictions"] += 1
            
            logger.debug(
                "Evicted registration %s (size=%d, age=%.1fs)",
                oldest_key, oldest_handle.size,
                time.time() - oldest_handle.created_at
            )
            
            if self.on_evict:
                try:
                    self.on_evict(oldest_handle)
                except Exception as e:
                    logger.error("Eviction callback error: %s", e)
    
    def invalidate(self, buffer_ptr: int, size: int, device: str) -> bool:
        """Invalidate a cached entry.
        
        Args:
            buffer_ptr: Buffer pointer
            size: Buffer size
            device: Device
            
        Returns:
            True if entry was found and removed
        """
        key = self._make_key(buffer_ptr, size, device)
        
        with self._lock:
            handle = self._cache.pop(key, None)
            if handle:
                self._total_size -= handle.size
                if self.on_evict:
                    self.on_evict(handle)
                return True
            return False
    
    def clear(self) -> int:
        """Clear all cached entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            
            if self.on_evict:
                for handle in self._cache.values():
                    try:
                        self.on_evict(handle)
                    except Exception:
                        pass
            
            self._cache.clear()
            self._total_size = 0
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            return {
                "entries": len(self._cache),
                "memory_mb": self._total_size / (1024 * 1024),
                "max_entries": self.max_entries,
                "max_memory_gb": self.max_memory_bytes / (1024**3),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": self._stats["hits"] / max(total, 1),
                "evictions": self._stats["evictions"],
                "registrations": self._stats["registrations"],
            }


@dataclass
class ChunkInfo:
    """Information about a tensor chunk."""
    chunk_idx: int
    total_chunks: int
    offset: int           # Byte offset in original tensor
    size: int             # Chunk size in bytes
    
    # Tensor metadata
    dtype: torch.dtype
    shape: Tuple[int, ...]
    
    # For reassembly
    is_first: bool = False
    is_last: bool = False


@dataclass
class ChunkTransferState:
    """State for chunked tensor transfer."""
    stream_id: StreamID
    request_id: str
    
    # Tensor info
    total_size: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    device: str
    
    # Chunking
    chunk_size: int
    total_chunks: int
    
    # Progress
    chunks_sent: int = 0
    chunks_received: int = 0
    bytes_transferred: int = 0
    
    # Buffers
    receive_buffer: Optional[torch.Tensor] = None
    received_chunks: Dict[int, bool] = field(default_factory=dict)
    
    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    
    # Overlap with compute
    prefetch_chunk: Optional[int] = None
    
    @property
    def is_complete(self) -> bool:
        return self.chunks_received == self.total_chunks
    
    @property
    def progress(self) -> float:
        return self.chunks_received / max(self.total_chunks, 1)


class ChunkedTensorTransfer:
    """Chunked tensor transfer with compute overlap.
    
    Features:
    - Large tensor splitting for pipelined transfer
    - Credit-based flow at chunk level
    - Compute/transfer overlap via prefetching
    """
    
    def __init__(
        self,
        default_chunk_size: int = 16 * 1024 * 1024,  # 16MB
        max_in_flight_chunks: int = 4,
        enable_overlap: bool = True,
    ):
        """Initialize chunked transfer manager.
        
        Args:
            default_chunk_size: Default chunk size in bytes
            max_in_flight_chunks: Max chunks in flight per stream
            enable_overlap: Enable compute/transfer overlap
        """
        self.default_chunk_size = default_chunk_size
        self.max_in_flight_chunks = max_in_flight_chunks
        self.enable_overlap = enable_overlap
        
        # Active transfers
        self._sends: Dict[str, ChunkTransferState] = {}
        self._recvs: Dict[str, ChunkTransferState] = {}
        self._lock = threading.RLock()
        
        # Prefetch streams for overlap
        self._prefetch_streams: Dict[str, torch.cuda.Stream] = {}
        
        # Statistics
        self._stats = {
            "transfers_started": 0,
            "transfers_completed": 0,
            "chunks_sent": 0,
            "chunks_received": 0,
            "bytes_transferred": 0,
        }
    
    def start_send(
        self,
        stream_id: StreamID,
        tensor: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> ChunkTransferState:
        """Start chunked send of a tensor.
        
        Args:
            stream_id: Stream identifier
            tensor: Tensor to send
            chunk_size: Override chunk size
            
        Returns:
            Transfer state
        """
        chunk_size = chunk_size or self.default_chunk_size
        total_size = tensor.numel() * tensor.element_size()
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        
        state = ChunkTransferState(
            stream_id=stream_id,
            request_id=stream_id.request_id,
            total_size=total_size,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            chunk_size=chunk_size,
            total_chunks=total_chunks,
        )
        
        with self._lock:
            self._sends[stream_id.to_str()] = state
            self._stats["transfers_started"] += 1
        
        logger.debug(
            "Started chunked send for %s: %d bytes in %d chunks",
            stream_id, total_size, total_chunks
        )
        
        return state
    
    def get_next_chunk(
        self,
        stream_id: StreamID,
        tensor: torch.Tensor,
    ) -> Optional[Tuple[ChunkInfo, bytes]]:
        """Get next chunk to send.
        
        Args:
            stream_id: Stream identifier
            tensor: Source tensor
            
        Returns:
            (chunk_info, chunk_data) or None if complete
        """
        key = stream_id.to_str()
        
        with self._lock:
            state = self._sends.get(key)
            if not state or state.chunks_sent >= state.total_chunks:
                return None
            
            chunk_idx = state.chunks_sent
            offset = chunk_idx * state.chunk_size
            size = min(state.chunk_size, state.total_size - offset)
            
            # Extract chunk data
            flat = tensor.view(-1)
            start_elem = offset // tensor.element_size()
            end_elem = start_elem + (size // tensor.element_size())
            
            if tensor.is_cuda:
                chunk_tensor = flat[start_elem:end_elem].cpu()
            else:
                chunk_tensor = flat[start_elem:end_elem]
            
            chunk_data = chunk_tensor.numpy().tobytes()
            
            chunk_info = ChunkInfo(
                chunk_idx=chunk_idx,
                total_chunks=state.total_chunks,
                offset=offset,
                size=size,
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                is_first=(chunk_idx == 0),
                is_last=(chunk_idx == state.total_chunks - 1),
            )
            
            state.chunks_sent += 1
            state.bytes_transferred += size
            self._stats["chunks_sent"] += 1
            
            if state.chunks_sent >= state.total_chunks:
                state.completed_at = time.time()
                self._stats["transfers_completed"] += 1
            
            return chunk_info, chunk_data
    
    def start_recv(
        self,
        stream_id: StreamID,
        total_size: int,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        device: str,
        chunk_size: int,
        total_chunks: int,
    ) -> ChunkTransferState:
        """Start receiving a chunked tensor.
        
        Args:
            stream_id: Stream identifier
            total_size: Total tensor size
            dtype: Tensor dtype
            shape: Tensor shape
            device: Target device
            chunk_size: Chunk size
            total_chunks: Total chunks expected
            
        Returns:
            Transfer state
        """
        state = ChunkTransferState(
            stream_id=stream_id,
            request_id=stream_id.request_id,
            total_size=total_size,
            dtype=dtype,
            shape=shape,
            device=device,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
        )
        
        # Pre-allocate receive buffer
        num_elements = total_size // torch.tensor([], dtype=dtype).element_size()
        if "cuda" in device:
            state.receive_buffer = torch.empty(
                num_elements, dtype=dtype, device=device
            )
        else:
            state.receive_buffer = torch.empty(
                num_elements, dtype=dtype, pin_memory=torch.cuda.is_available()
            )
        
        with self._lock:
            self._recvs[stream_id.to_str()] = state
            self._stats["transfers_started"] += 1
        
        logger.debug(
            "Started chunked recv for %s: %d bytes in %d chunks",
            stream_id, total_size, total_chunks
        )
        
        return state
    
    def receive_chunk(
        self,
        stream_id: StreamID,
        chunk_info: ChunkInfo,
        chunk_data: bytes,
    ) -> Optional[torch.Tensor]:
        """Receive a chunk and reassemble if complete.
        
        Args:
            stream_id: Stream identifier
            chunk_info: Chunk metadata
            chunk_data: Chunk bytes
            
        Returns:
            Complete tensor if finished, None otherwise
        """
        key = stream_id.to_str()
        
        with self._lock:
            state = self._recvs.get(key)
            if not state:
                # Start new receive
                state = self.start_recv(
                    stream_id=stream_id,
                    total_size=chunk_info.total_chunks * len(chunk_data),  # Estimate
                    dtype=chunk_info.dtype,
                    shape=chunk_info.shape,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    chunk_size=len(chunk_data),
                    total_chunks=chunk_info.total_chunks,
                )
            
            # Mark chunk received
            state.received_chunks[chunk_info.chunk_idx] = True
            state.chunks_received += 1
            state.bytes_transferred += len(chunk_data)
            self._stats["chunks_received"] += 1
            self._stats["bytes_transferred"] += len(chunk_data)
            
            # Copy chunk data to receive buffer
            if state.receive_buffer is not None:
                chunk_tensor = torch.frombuffer(
                    bytearray(chunk_data),
                    dtype=chunk_info.dtype
                )
                
                start_elem = chunk_info.offset // chunk_tensor.element_size()
                end_elem = start_elem + chunk_tensor.numel()
                
                if state.receive_buffer.is_cuda and self.enable_overlap:
                    # Use async copy with stream
                    prefetch_stream = self._get_prefetch_stream(key)
                    with torch.cuda.stream(prefetch_stream):
                        state.receive_buffer[start_elem:end_elem].copy_(
                            chunk_tensor.to(state.receive_buffer.device)
                        )
                else:
                    state.receive_buffer[start_elem:end_elem].copy_(chunk_tensor)
            
            # Check if complete
            if state.is_complete:
                state.completed_at = time.time()
                self._stats["transfers_completed"] += 1
                
                # Reshape and return
                if state.receive_buffer is not None:
                    result = state.receive_buffer.view(chunk_info.shape)
                    del self._recvs[key]
                    return result
        
        return None
    
    def _get_prefetch_stream(self, key: str) -> torch.cuda.Stream:
        """Get or create prefetch stream for overlap."""
        if key not in self._prefetch_streams:
            self._prefetch_streams[key] = torch.cuda.Stream()
        return self._prefetch_streams[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        with self._lock:
            return {
                "active_sends": len(self._sends),
                "active_recvs": len(self._recvs),
                **self._stats,
            }


class CapabilityNegotiator:
    """Negotiates transport capabilities between endpoints.
    
    Selects best available transport based on:
    - Local and remote capabilities
    - Current resource availability
    - Performance requirements
    """
    
    # Capability priority (higher = preferred)
    PRIORITY = {
        TransportCapability.GPU_DIRECT_RDMA: 100,
        TransportCapability.HOST_RDMA: 80,
        TransportCapability.GPU_DIRECT_TCP: 60,
        TransportCapability.HOST_TCP: 40,
        TransportCapability.SHARED_MEMORY: 90,  # High for local
        TransportCapability.CPU_PROXY: 20,
    }
    
    def __init__(
        self,
        local_caps: Optional[List[TransportCapability]] = None,
        prefer_zero_copy: bool = True,
    ):
        """Initialize capability negotiator.
        
        Args:
            local_caps: Local capabilities (auto-detect if None)
            prefer_zero_copy: Prefer zero-copy transfers
        """
        self.local_caps = local_caps or self._detect_local_caps()
        self.prefer_zero_copy = prefer_zero_copy
        
        logger.info("Local capabilities: %s", [c.name for c in self.local_caps])
    
    def _detect_local_caps(self) -> List[TransportCapability]:
        """Auto-detect local transport capabilities."""
        caps = [TransportCapability.CPU_PROXY]  # Always available
        
        # Check for CUDA
        if torch.cuda.is_available():
            caps.append(TransportCapability.HOST_TCP)
            caps.append(TransportCapability.HOST_RDMA)
            
            # Check for GPU Direct
            try:
                # This is a simplified check
                if os.environ.get("UCX_CUDA_COPY_ASYNC", "n") == "y":
                    caps.append(TransportCapability.GPU_DIRECT_RDMA)
                    caps.append(TransportCapability.GPU_DIRECT_TCP)
            except Exception:
                pass
        
        # Check for shared memory
        if os.path.exists("/dev/shm"):
            caps.append(TransportCapability.SHARED_MEMORY)
        
        return caps
    
    def negotiate(
        self,
        remote_caps: ConnectionCapability,
        is_local: bool = False,
    ) -> TransportSelection:
        """Negotiate transport with remote endpoint.
        
        Args:
            remote_caps: Remote capabilities
            is_local: Whether peer is on same node
            
        Returns:
            Selected transport configuration
        """
        # Build candidate list
        candidates = []
        
        # Check shared memory for local peers
        if is_local and TransportCapability.SHARED_MEMORY in self.local_caps:
            candidates.append(TransportCapability.SHARED_MEMORY)
        
        # Check GPU Direct RDMA
        if (
            TransportCapability.GPU_DIRECT_RDMA in self.local_caps and
            remote_caps.supports_gpu_direct and
            remote_caps.supports_rdma
        ):
            candidates.append(TransportCapability.GPU_DIRECT_RDMA)
        
        # Check Host RDMA
        if (
            TransportCapability.HOST_RDMA in self.local_caps and
            remote_caps.supports_rdma
        ):
            candidates.append(TransportCapability.HOST_RDMA)
        
        # Check GPU Direct TCP
        if (
            TransportCapability.GPU_DIRECT_TCP in self.local_caps and
            remote_caps.supports_gpu_direct
        ):
            candidates.append(TransportCapability.GPU_DIRECT_TCP)
        
        # Check Host TCP
        if TransportCapability.HOST_TCP in self.local_caps:
            candidates.append(TransportCapability.HOST_TCP)
        
        # CPU Proxy is always fallback
        candidates.append(TransportCapability.CPU_PROXY)
        
        # Sort by priority
        candidates.sort(key=lambda c: self.PRIORITY.get(c, 0), reverse=True)
        
        selected = candidates[0]
        fallback = candidates[1] if len(candidates) > 1 else None
        
        # Determine capabilities to use
        use_zero_copy = selected in (
            TransportCapability.GPU_DIRECT_RDMA,
            TransportCapability.GPU_DIRECT_TCP,
        )
        use_registered = selected in (
            TransportCapability.GPU_DIRECT_RDMA,
            TransportCapability.HOST_RDMA,
        )
        
        # Determine chunk size based on transport
        if selected in (TransportCapability.GPU_DIRECT_RDMA, TransportCapability.HOST_RDMA):
            max_chunk = 64 * 1024 * 1024  # 64MB for RDMA
        elif selected == TransportCapability.SHARED_MEMORY:
            max_chunk = 256 * 1024 * 1024  # 256MB for SHM
        else:
            max_chunk = 16 * 1024 * 1024  # 16MB for TCP
        
        result = TransportSelection(
            selected=selected,
            fallback=fallback,
            use_zero_copy=use_zero_copy,
            use_registered_memory=use_registered,
            use_pinned_buffer=not use_zero_copy,
            max_chunk_size=max_chunk,
            overlap_enabled=self.prefer_zero_copy,
        )
        
        logger.debug(
            "Negotiated transport: %s (fallback: %s)",
            selected.name, fallback.name if fallback else "none"
        )
        
        return result


class GPUAwareFastPath:
    """GPU-aware fast path for high-performance tensor transfer.
    
    Integrates:
    - Registration cache for fast reconnection
    - Chunked transfer with overlap
    - Capability negotiation
    - Warmup for migration preparation
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        cache_max_entries: int = 256,
        cache_max_memory_gb: float = 16.0,
        default_chunk_size: int = 16 * 1024 * 1024,
    ):
        """Initialize GPU-aware fast path.
        
        Args:
            local_lid: Local endpoint LID
            cache_max_entries: Max registration cache entries
            cache_max_memory_gb: Max memory for cache
            default_chunk_size: Default chunk size
        """
        self.local_lid = local_lid
        
        # Registration cache
        self.reg_cache = RegistrationCache(
            max_entries=cache_max_entries,
            max_memory_gb=cache_max_memory_gb,
            on_evict=self._on_registration_evict,
        )
        
        # Chunked transfer
        self.chunked = ChunkedTensorTransfer(
            default_chunk_size=default_chunk_size,
        )
        
        # Capability negotiator
        self.negotiator = CapabilityNegotiator()
        
        # Per-peer transport selection cache
        self._peer_transports: Dict[LogicalEndpointID, TransportSelection] = {}
        self._transport_lock = threading.Lock()
        
        # Warmup state for migration
        self._warmup_pending: Dict[LogicalEndpointID, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            "tensor_sends": 0,
            "tensor_recvs": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "cache_lookups": 0,
            "warmups_completed": 0,
        }
    
    def _on_registration_evict(self, handle: RegistrationHandle) -> None:
        """Handle registration eviction."""
        logger.debug(
            "Evicted registration: ptr=%x, size=%d, device=%s",
            handle.buffer_ptr, handle.size, handle.device
        )
        # Transport-specific cleanup would go here
    
    def get_or_negotiate_transport(
        self,
        peer_lid: LogicalEndpointID,
        peer_caps: ConnectionCapability,
        is_local: bool = False,
    ) -> TransportSelection:
        """Get or negotiate transport for peer.
        
        Args:
            peer_lid: Peer LID
            peer_caps: Peer capabilities
            is_local: Whether peer is local
            
        Returns:
            Transport selection
        """
        with self._transport_lock:
            if peer_lid in self._peer_transports:
                return self._peer_transports[peer_lid]
            
            selection = self.negotiator.negotiate(peer_caps, is_local)
            self._peer_transports[peer_lid] = selection
            return selection
    
    def prepare_send(
        self,
        tensor: torch.Tensor,
        peer_lid: LogicalEndpointID,
        stream_id: StreamID,
    ) -> Tuple[RegistrationHandle, ChunkTransferState]:
        """Prepare tensor for sending.
        
        Handles registration caching and chunking setup.
        
        Args:
            tensor: Tensor to send
            peer_lid: Destination peer
            stream_id: Stream identifier
            
        Returns:
            (registration_handle, transfer_state)
        """
        self._stats["tensor_sends"] += 1
        
        # Check registration cache
        buffer_ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        device = str(tensor.device)
        
        self._stats["cache_lookups"] += 1
        reg_handle = self.reg_cache.get(buffer_ptr, size, device)
        
        if not reg_handle:
            # Need to register
            registration = self._do_registration(tensor)
            reg_handle = self.reg_cache.put(buffer_ptr, size, device, registration)
        
        # Setup chunked transfer
        transport = self._peer_transports.get(peer_lid)
        chunk_size = transport.max_chunk_size if transport else self.chunked.default_chunk_size
        
        transfer_state = self.chunked.start_send(
            stream_id=stream_id,
            tensor=tensor,
            chunk_size=chunk_size,
        )
        
        self._stats["bytes_sent"] += size
        
        return reg_handle, transfer_state
    
    def _do_registration(self, tensor: torch.Tensor) -> Any:
        """Perform memory registration.
        
        Transport-specific implementation would go here.
        """
        # Placeholder - actual implementation depends on transport
        return {
            "ptr": tensor.data_ptr(),
            "size": tensor.numel() * tensor.element_size(),
            "device": str(tensor.device),
        }
    
    def warmup_for_migration(
        self,
        peer_lid: LogicalEndpointID,
        peer_caps: ConnectionCapability,
        buffer_sizes: List[int],
    ) -> bool:
        """Warmup path for upcoming migration.
        
        Pre-registers buffers and establishes transport state.
        
        Args:
            peer_lid: New peer after migration
            peer_caps: Peer capabilities
            buffer_sizes: Expected buffer sizes to pre-allocate
            
        Returns:
            True if warmup started
        """
        # Negotiate transport
        transport = self.get_or_negotiate_transport(peer_lid, peer_caps)
        
        # Pre-allocate and register buffers
        warmup_state = {
            "transport": transport,
            "buffers": [],
            "registrations": [],
            "started_at": time.time(),
        }
        
        for size in buffer_sizes:
            if torch.cuda.is_available():
                buf = torch.empty(size // 4, dtype=torch.float32, device="cuda")
            else:
                buf = torch.empty(size // 4, dtype=torch.float32)
            
            warmup_state["buffers"].append(buf)
            
            # Register and cache
            reg = self._do_registration(buf)
            handle = self.reg_cache.put(
                buf.data_ptr(),
                buf.numel() * buf.element_size(),
                str(buf.device),
                reg,
            )
            warmup_state["registrations"].append(handle)
        
        self._warmup_pending[peer_lid] = warmup_state
        
        logger.info(
            "Started warmup for migration to %s: %d buffers",
            peer_lid, len(buffer_sizes)
        )
        
        return True
    
    def complete_warmup(self, peer_lid: LogicalEndpointID) -> bool:
        """Complete warmup and mark ready.
        
        Args:
            peer_lid: Peer LID
            
        Returns:
            True if warmup was pending and completed
        """
        warmup_state = self._warmup_pending.pop(peer_lid, None)
        if not warmup_state:
            return False
        
        elapsed = time.time() - warmup_state["started_at"]
        self._stats["warmups_completed"] += 1
        
        logger.info(
            "Completed warmup for %s in %.2fs",
            peer_lid, elapsed
        )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fast path statistics."""
        return {
            "fast_path": self._stats.copy(),
            "registration_cache": self.reg_cache.get_stats(),
            "chunked_transfer": self.chunked.get_stats(),
            "peer_transports": {
                str(lid): sel.selected.name
                for lid, sel in self._peer_transports.items()
            },
        }
