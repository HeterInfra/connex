# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
UCX Transport Layer for High-Performance Data Transfer.

Provides UCX-based communication with:
- Tagged message and Active Message support
- Memory registration (MR) caching
- Buffer pool management
- Fallback to ZeroMQ when UCX unavailable
"""

import asyncio
import logging
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

# Configure UCX to use system library with RDMA support
# The pip-installed libucx-cu12 doesn't include InfiniBand transports,
# but the system UCX at /usr/local/ucx does have full RDMA support.
# We need to set these BEFORE importing ucp.
_UCX_SYSTEM_LIB = "/usr/local/ucx/lib"
if os.path.exists(_UCX_SYSTEM_LIB):
    # Prepend system UCX lib path so it's found first
    _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _UCX_SYSTEM_LIB not in _current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_UCX_SYSTEM_LIB}:{_current_ld_path}"
    # Also set UCX_HOME for module search
    os.environ["UCX_HOME"] = "/usr/local/ucx"
    logger.info("Configured UCX to use system library at %s for RDMA support", _UCX_SYSTEM_LIB)

# ucx-py writes defaults such as UCX_RNDV_THRESH into os.environ on import,
# so record what the user actually set before importing it.
_USER_RNDV_THRESH = os.environ.get("UCX_RNDV_THRESH")


# Try to import UCX
try:
    import ucp
    UCX_AVAILABLE = True
    logger.info("UCX-Py imported successfully, version: %s", ucp.get_ucx_version())
except ImportError:
    UCX_AVAILABLE = False
    logger.warning("UCX (ucp) not available, will use ZeroMQ fallback")

# Import ZeroMQ for fallback
import zmq
import msgpack

# v1.8.18: Module-level dtype mappings to avoid creating dicts in hot paths
DTYPE_STR_TO_TORCH = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
}

TORCH_TO_DTYPE_IDX = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
    torch.int32: 3, torch.int64: 4, torch.bool: 5,
    torch.int8: 6, torch.uint8: 7,
}

DTYPE_IDX_TO_TORCH = [
    torch.float32, torch.float16, torch.bfloat16,
    torch.int32, torch.int64, torch.bool,
    torch.int8, torch.uint8
]

# v1.8.18: Pre-computed element sizes to avoid torch.tensor([]).element_size() calls
DTYPE_ELEMENT_SIZE = {
    torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    torch.int32: 4, torch.int64: 8, torch.bool: 1,
    torch.int8: 1, torch.uint8: 1,
}


def _parse_gdir_metadata(data: Any) -> Optional[dict]:
    """Parse the GDIR header written by ``_send_ucx_cuda``.

    Format: ``b"GDIR" + struct.pack("!BBI{ndim}I", dtype_idx, ndim, numel, *shape)``.
    Returns ``{"shape", "dtype", "nbytes"}`` or None if ``data`` is not a
    well-formed header (length must match exactly and numel must equal the
    product of the shape, so ordinary payloads are not misparsed).
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 10 or data[:4] != b"GDIR":
        return None
    dtype_idx, ndim = data[4], data[5]
    if len(data) != 10 + 4 * ndim:
        return None
    numel = struct.unpack("!I", data[6:10])[0]
    shape = struct.unpack(f"!{ndim}I", data[10:10 + ndim * 4])
    prod = 1
    for d in shape:
        prod *= d
    if prod != numel or dtype_idx >= len(DTYPE_IDX_TO_TORCH):
        return None
    dtype = DTYPE_IDX_TO_TORCH[dtype_idx]
    return {"shape": shape, "dtype": dtype,
            "nbytes": numel * DTYPE_ELEMENT_SIZE.get(dtype, 4)}


_cuda_tls = threading.local()


def _bind_cuda_device(device: "int | torch.device | None") -> None:
    """Make ``device``'s primary CUDA context current in the calling thread.

    UCX's cuda_copy memory detection (cuMemGetAddressRange) needs a current
    context in the thread that posts the send/recv. The UCX event loop runs in
    its own thread, which never touched CUDA, so without this every GPU-direct
    send failed with "invalid device context" and the endpoint was torn down.
    Cached per thread, so the hot path costs one attribute lookup.
    """
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        device = torch.cuda.current_device()
    if getattr(_cuda_tls, "device", None) != device:
        torch.cuda.set_device(device)
        _cuda_tls.device = device


class UCXListenerError(RuntimeError):
    """The UCX listener could not be started (e.g. the port is already in use)."""


class TransportMode(Enum):
    """Transport mode selection."""
    UCX_RDMA = "ucx_rdma"      # UCX with RDMA (InfiniBand)
    UCX_TCP = "ucx_tcp"        # UCX with TCP
    ZEROMQ = "zeromq"          # ZeroMQ fallback


@dataclass
class TransferStats:
    """Statistics for data transfer."""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    send_latency_sum: float = 0.0
    recv_latency_sum: float = 0.0
    mr_cache_hits: int = 0
    mr_cache_misses: int = 0
    
    def avg_send_latency(self) -> float:
        return self.send_latency_sum / max(self.messages_sent, 1)
        
    def avg_recv_latency(self) -> float:
        return self.recv_latency_sum / max(self.messages_received, 1)


class MRCache:
    """Memory Registration Cache for RDMA.
    
    Caches memory registrations to avoid expensive re-registration
    on every transfer. Uses LRU eviction when cache is full.
    """
    
    def __init__(self, max_entries: int = 100):
        """Initialize MR cache.
        
        Args:
            max_entries: Maximum number of cached registrations
        """
        self.max_entries = max_entries
        self._cache: dict[int, Any] = {}  # buffer_id -> registration
        self._access_order: list[int] = []  # LRU tracking
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
    def get(self, buffer_ptr: int) -> Optional[Any]:
        """Get cached registration for buffer.
        
        Args:
            buffer_ptr: Buffer pointer/ID
            
        Returns:
            Cached registration or None
        """
        with self._lock:
            if buffer_ptr in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._access_order.remove(buffer_ptr)
                self._access_order.append(buffer_ptr)
                return self._cache[buffer_ptr]
            self._misses += 1
            return None
            
    def put(self, buffer_ptr: int, registration: Any) -> None:
        """Cache a registration.
        
        Args:
            buffer_ptr: Buffer pointer/ID
            registration: MR registration handle
        """
        with self._lock:
            if buffer_ptr in self._cache:
                return
                
            # Evict if full
            while len(self._cache) >= self.max_entries:
                oldest = self._access_order.pop(0)
                old_reg = self._cache.pop(oldest, None)
                # Close old registration if needed
                if old_reg and hasattr(old_reg, 'close'):
                    try:
                        old_reg.close()
                    except Exception:
                        pass
                        
            self._cache[buffer_ptr] = registration
            self._access_order.append(buffer_ptr)
            
    def invalidate(self, buffer_ptr: int) -> None:
        """Invalidate a cached registration."""
        with self._lock:
            if buffer_ptr in self._cache:
                reg = self._cache.pop(buffer_ptr)
                self._access_order.remove(buffer_ptr)
                if reg and hasattr(reg, 'close'):
                    try:
                        reg.close()
                    except Exception:
                        pass
                        
    def clear(self) -> None:
        """Clear all cached registrations."""
        with self._lock:
            for reg in self._cache.values():
                if reg and hasattr(reg, 'close'):
                    try:
                        reg.close()
                    except Exception:
                        pass
            self._cache.clear()
            self._access_order.clear()
            
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total, 1),
            }


class BufferPool:
    """Pre-allocated buffer pool for transfer operations.
    
    Reduces allocation overhead by reusing buffers.
    """
    
    def __init__(
        self,
        buffer_size: int = 64 * 1024 * 1024,  # 64MB default
        num_buffers: int = 4,
        device: str = "cuda",
    ):
        """Initialize buffer pool.
        
        Args:
            buffer_size: Size of each buffer in bytes
            num_buffers: Number of buffers to pre-allocate
            device: Device for buffers ("cuda" or "cpu")
        """
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        self.device = device
        
        self._buffers: list[torch.Tensor] = []
        self._available: list[int] = []
        self._lock = threading.Lock()
        
        self._allocate_buffers()
        
    def _allocate_buffers(self) -> None:
        """Pre-allocate buffers."""
        for i in range(self.num_buffers):
            if self.device == "cuda":
                # Pin memory for faster RDMA transfers
                buf = torch.empty(
                    self.buffer_size,
                    dtype=torch.uint8,
                    device="cuda",
                )
            else:
                # Pinned host memory needs an accelerator; recent CPU-only
                # torch builds raise instead of ignoring pin_memory=True.
                buf = torch.empty(
                    self.buffer_size,
                    dtype=torch.uint8,
                    pin_memory=torch.cuda.is_available(),
                )
            self._buffers.append(buf)
            self._available.append(i)
            
        logger.info(
            "BufferPool allocated %d buffers of %d MB each on %s",
            self.num_buffers, self.buffer_size // (1024*1024), self.device
        )
        
    def acquire(self, timeout: float = 5.0) -> Optional[tuple[int, torch.Tensor]]:
        """Acquire a buffer from the pool.
        
        Args:
            timeout: Max time to wait for available buffer
            
        Returns:
            (buffer_id, buffer) tuple or None if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._available:
                    idx = self._available.pop()
                    return idx, self._buffers[idx]
            time.sleep(0.001)  # Brief sleep before retry
        return None
        
    def release(self, buffer_id: int) -> None:
        """Release a buffer back to the pool.
        
        Args:
            buffer_id: Buffer ID to release
        """
        with self._lock:
            if buffer_id not in self._available:
                self._available.append(buffer_id)
                
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "total_buffers": self.num_buffers,
                "available_buffers": len(self._available),
                "buffer_size_mb": self.buffer_size // (1024*1024),
                "device": self.device,
            }


class CUDATensorCache:
    """Cache for CUDA tensors to avoid repeated allocations in GPU Direct recv.
    
    v1.8.12: Reduces GPU Direct recv overhead by reusing tensors.
    Key insight: Most tensors in pipeline have fixed shapes (hidden_states).
    """
    
    def __init__(self, max_per_key: int = 4):
        self._cache: dict[tuple[int, torch.dtype], list[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self.max_per_key = max_per_key
        self.hits = 0
        self.misses = 0
    
    def acquire(self, numel: int, dtype: torch.dtype, device: "int | str" = "cuda") -> torch.Tensor:
        """Get a cached tensor or allocate new one."""
        if isinstance(device, int):
            idx = device
        else:
            idx = torch.device(device).index
            if idx is None:
                idx = torch.cuda.current_device()
        key = (numel, dtype, idx)
        with self._lock:
            if key in self._cache and self._cache[key]:
                self.hits += 1
                return self._cache[key].pop()
        # Miss - allocate new tensor
        self.misses += 1
        return torch.empty(numel, dtype=dtype, device=f"cuda:{idx}")
    
    def release(self, tensor: torch.Tensor) -> None:
        """Return tensor to cache for reuse."""
        if not tensor.is_cuda:
            return
        key = (tensor.numel(), tensor.dtype, tensor.device.index)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = []
            if len(self._cache[key]) < self.max_per_key:
                self._cache[key].append(tensor)


class UCXTransport:
    """UCX-based transport for pipeline communication.
    
    Features:
    - UCX RDMA/TCP transport with automatic fallback
    - Memory registration caching
    - Buffer pooling
    - ZeroMQ fallback when UCX unavailable
    """
    
    def __init__(
        self,
        local_address: str,
        local_port: int,
        mode: TransportMode = TransportMode.UCX_RDMA,
        device: str = "cuda",
        buffer_size: int = 64 * 1024 * 1024,
        num_buffers: int = 4,
        port_fallback: int = 0,
    ):
        """Initialize UCX transport.
        
        Args:
            local_address: Local IP address to bind
            local_port: Local port to listen on
            port_fallback: If the UCX listener cannot bind local_port, try up
                to this many following ports; ``local_port`` is updated to the
                port actually bound. Useful when a restarted replica's old port
                is still in TIME_WAIT.
            mode: Transport mode (UCX_RDMA, UCX_TCP, or ZEROMQ)
            device: Device for buffers
            buffer_size: Size of transfer buffers
            num_buffers: Number of pre-allocated buffers
        """
        self.local_address = local_address
        self.local_port = local_port
        self.port_fallback = port_fallback
        self.device = device
        
        # Determine actual mode based on availability
        if mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP) and not UCX_AVAILABLE:
            logger.warning("UCX not available, falling back to ZeroMQ")
            mode = TransportMode.ZEROMQ
        self.mode = mode
        
        # Initialize components
        self.mr_cache = MRCache()
        self.buffer_pool = BufferPool(
            buffer_size=buffer_size,
            num_buffers=num_buffers,
            device=device,
        )
        self.stats = TransferStats()
        
        # Connection state
        self._endpoints: dict[str, Any] = {}  # peer_address -> endpoint
        self._lock = threading.RLock()
        self._running = False
        
        # UCX-specific
        self._ucp_listener: Any = None
        self._ucp_endpoints: dict[str, Any] = {}
        # Called with the peer id (as returned by recv_from_queue) when an
        # accepted connection's receive loop ends while the transport runs,
        # e.g. because the peer process died.
        self.peer_closed_callbacks: list[Callable[[str], None]] = []
        self._ucx_started = threading.Event()
        self._ucx_start_error: Optional[BaseException] = None
        self._ucp_options: dict[str, Any] | None = None
        self._ucx_tls: str | None = None
        self._ucx_net_devices: str | None = None
        self._ucx_sockaddr_tls_priority: str | None = None
        
        # UCX receive queue for incoming data
        self._recv_queue: queue.Queue = queue.Queue()
        self._recv_loop_task: Optional[asyncio.Task] = None
        # Accepted-connection receive loops, cancelled on stop.
        self._ucx_tasks: set[asyncio.Task] = set()
        self._recv_threads: list[threading.Thread] = []
        
        # v1.9.0: Per-endpoint send locks to prevent interleaving in concurrent sends
        # This ensures header + tensors are sent atomically for each request
        self._endpoint_send_locks: dict[Any, asyncio.Lock] = {}
        
        # v1.9.4: Async send tracking for fire-and-forget mode
        # Tracks pending sends and their errors for later reporting
        self._pending_sends: dict[int, asyncio.Future] = {}
        self._pending_send_errors: list[Exception] = []
        self._next_send_id = 0
        self._pending_send_lock = threading.Lock()
        
        # v1.8.12: CUDA tensor cache for GPU Direct recv
        self._cuda_tensor_cache = CUDATensorCache(max_per_key=4)
        # GPU that GPU-direct receives land on. "cuda:N" pins it; plain "cuda"
        # is resolved to the caller's current device at start().
        self._cuda_device: Optional[int] = None
        if isinstance(device, str) and device.startswith("cuda:"):
            self._cuda_device = int(device.split(":", 1)[1])
        
        # UCX event loop thread - keeps listener alive
        self._ucx_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ucx_loop_thread: Optional[threading.Thread] = None
        
        # ZeroMQ-specific
        self._zmq_ctx: Optional[zmq.Context] = None
        self._zmq_sockets: dict[str, zmq.Socket] = {}
        
        # Callbacks
        self._recv_callback: Optional[Callable[[str, bytes | bytearray], None]] = None
        
    async def start(self) -> None:
        """Start the transport layer."""
        if self._running:
            return
            
        self._running = True
        self._capture_cuda_device()
        
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            await self._start_ucx()
        else:
            self._start_zmq()
            
        logger.info(
            "UCXTransport started: mode=%s, address=%s:%d",
            self.mode.value, self.local_address, self.local_port
        )
        
    def _run_ucx_event_loop(self) -> None:
        """Run UCX event loop in background thread."""
        import asyncio
        self._ucx_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ucx_loop)
        
        try:
            # Start UCX listener
            try:
                self._ucx_loop.run_until_complete(self._start_ucx())
            except Exception as e:
                self._ucx_start_error = e
                raise
            finally:
                self._ucx_started.set()
            logger.info("UCX event loop started, running forever...")
            
            # Keep the loop running to accept connections
            self._ucx_loop.run_forever()
        except Exception as e:
            logger.error("UCX event loop error: %s", e)
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self._ucx_loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    self._ucx_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
                self._ucx_loop.run_until_complete(self._ucx_loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                self._ucx_loop.close()
            except Exception:
                pass
            logger.info("UCX event loop stopped")
    
    def start_sync(self) -> None:
        """Start the transport layer (synchronous version)."""
        if self._running:
            return
            
        self._running = True
        self._capture_cuda_device()
        
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            # Start UCX in a background thread with its own event loop
            # This keeps the listener alive to accept connections
            self._ucx_loop_thread = threading.Thread(
                target=self._run_ucx_event_loop,
                name="UCX-EventLoop",
                daemon=True,
            )
            self._ucx_loop_thread.start()

            # Wait for UCX init + listener creation. A transport without a
            # listener must not be used: peers would be told to connect to a
            # port nobody accepts on.
            self._ucx_started.wait(30)
            if self._ucp_listener is None:
                self._running = False
                err = self._ucx_start_error
                raise UCXListenerError(
                    f"UCX listener failed to start on port {self.local_port}: "
                    f"{err or 'timeout'}"
                )
        else:
            self._start_zmq()
            
        logger.info(
            "UCXTransport started (sync): mode=%s, address=%s:%d",
            self.mode.value, self.local_address, self.local_port
        )
        
    def _build_ucp_options(self) -> dict[str, str]:
        """UCX init options for this transport's mode (env overrides applied)."""
        # Configure UCX - use TLS=all to let UCX auto-select best transport
        # This allows UCX to use RDMA if available, falling back to TCP if not.
        # Explicitly specifying rc,ud,dc may fail if the pip-installed libucx
        # doesn't have IB support (system UCX does, but pip libucx-cu12 doesn't)
        ucp_options = {
            "MEMTYPE_CACHE": "n",  # Disable memory type cache for better stability
            "TLS": "all",  # Let UCX choose best transport (RDMA > TCP)
            "SOCKADDR_TLS_PRIORITY": "rdmacm,tcp",  # Prefer RDMA CM for connection
        }
        
        # TCP mode: restrict UCX to TCP-class transports. With TLS=all on a
        # host that has IB devices, every endpoint creation also probes the
        # RDMA transports, which costs ~0.7 s per connect (vs ~40 ms).
        if self.mode == TransportMode.UCX_TCP:
            ucp_options["TLS"] = "tcp,sm,self,cuda_copy"
            ucp_options["SOCKADDR_TLS_PRIORITY"] = "tcp"
            # Eager-only: when a peer dies with a rendezvous send outstanding,
            # UCX (hpcx 2.18 / UCX 1.16) can segfault in ucp_proto_rndv_rts_abort
            # (~10-25% of abrupt peer kills in benchmarks/elastic_bench.py).
            # Costs ~20-30% bandwidth for 1-8 MB messages over TCP.
            # Override with UCX_RNDV_THRESH.
            if not _USER_RNDV_THRESH:
                ucp_options["RNDV_THRESH"] = "inf"

        # RDMA mode restricted by the user to non-RDMA transports is TCP in
        # practice: give it the same crash-avoiding eager default.
        env_tls_now = {t.strip().lstrip("^") for t in os.getenv("UCX_TLS", "").split(",") if t.strip()}
        rdma_tls = {"rc", "rc_v", "rc_x", "rc_verbs", "rc_mlx5", "dc", "dc_x", "dc_mlx5",
                    "ud", "ud_v", "ud_x", "ud_verbs", "ud_mlx5", "ib", "all"}
        if (self.mode == TransportMode.UCX_RDMA and env_tls_now
                and not env_tls_now & rdma_tls
                and not _USER_RNDV_THRESH):
            ucp_options["RNDV_THRESH"] = "inf"

        # If RDMA mode requested, try to prioritize RDMA transports
        if self.mode == TransportMode.UCX_RDMA:
            # Check if we have RDMA devices available
            # Set preferences but don't restrict (use 'all' to avoid failures)
            ucp_options["NET_DEVICES"] = "all"  # Let UCX find all devices
            # Rendezvous (zero-copy) stays on in real RDMA mode, made safe by
            # requiring the UCP registration cache. Without it, a peer dying
            # with a rendezvous send in flight crashes this process in
            # ucp_request_memh_invalidate (context->rcache == NULL; see
            # docs/design/ucx-transport-testing.md). _init_ucp drops to eager if
            # the cache cannot be created.
            if ("RNDV_THRESH" not in ucp_options and not _USER_RNDV_THRESH
                    and not os.getenv("UCX_RCACHE_ENABLE")):
                ucp_options["RCACHE_ENABLE"] = "y"

        # Respect standard UCX environment overrides (and log them explicitly).
        # This is useful to ensure UCX is actually using RDMA-capable devices.
        env_tls = os.getenv("UCX_TLS")
        env_net_devices = os.getenv("UCX_NET_DEVICES")
        env_sockaddr_tls_priority = os.getenv("UCX_SOCKADDR_TLS_PRIORITY")
        if env_tls:
            ucp_options["TLS"] = env_tls
        if env_net_devices:
            ucp_options["NET_DEVICES"] = env_net_devices
        if env_sockaddr_tls_priority:
            ucp_options["SOCKADDR_TLS_PRIORITY"] = env_sockaddr_tls_priority
        return ucp_options

    # Env vars that some ucx-py builds cannot round-trip through their config
    # parser (ucx_config_to_dict raises "too many values to unpack" inside
    # ucp.init). They are passed as init options instead.
    _ENV_PASSED_AS_OPTIONS = ("UCX_SOCKADDR_TLS_PRIORITY",)

    def _init_ucp(self, ucp_options: dict[str, str]) -> None:
        """ucp.init with env/option hygiene; never silently leaves RDMA mode.

        Mutates ucp_options to what UCX was actually initialized with.
        """
        core = getattr(ucp, "core", None)
        if getattr(core, "_ctx", None) is not None:
            # ucp.init is process-global; a second transport in the same
            # process shares the first one's context (and its options).
            logger.warning("UCX already initialized in this process; reusing it, "
                           "options for this transport are ignored")
            return
        hidden = {k: os.environ.pop(k) for k in self._ENV_PASSED_AS_OPTIONS if k in os.environ}
        try:
            # Creating the registration cache needs UCM's memory hooks, whose
            # self-test fails intermittently in a process with torch/CUDA
            # loaded; a few retries usually succeed.
            attempts = (max(1, int(os.getenv("CONNEX_UCX_RCACHE_RETRIES", "5")))
                        if ucp_options.get("RCACHE_ENABLE") == "y" else 1)
            for _ in range(attempts):
                try:
                    ucp.init(dict(ucp_options))
                    return
                except Exception as e:
                    first_err = e

            # Registration cache unavailable (UCM memory hooks could not be
            # installed): rendezvous is not safe against peer death, use eager.
            if ucp_options.get("RCACHE_ENABLE") == "y":
                logger.warning(
                    "UCX init with RCACHE_ENABLE=y failed (%s, pid %d); UCP registration cache "
                    "unavailable, using eager protocol (RNDV_THRESH=inf) so peer death "
                    "cannot crash this process. Set UCX_MEM_MMAP_HOOK_MODE=reloc before "
                    "importing ucp to keep rendezvous.", first_err, os.getpid(),
                )
                ucp_options.pop("RCACHE_ENABLE")
                if not _USER_RNDV_THRESH:
                    ucp_options["RNDV_THRESH"] = "inf"
                try:
                    ucp.init(dict(ucp_options))
                    return
                except Exception as e:
                    first_err = e

            if self.mode == TransportMode.UCX_RDMA and os.getenv("CONNEX_UCX_ALLOW_TCP_FALLBACK", "0") != "1":
                raise RuntimeError(
                    f"UCX init failed in RDMA mode (TLS={ucp_options.get('TLS')} "
                    f"NET_DEVICES={ucp_options.get('NET_DEVICES')}): {first_err}. "
                    "Refusing to downgrade to TCP; set CONNEX_UCX_ALLOW_TCP_FALLBACK=1 to allow it."
                ) from first_err
            logger.error(
                "UCX init failed (%s); DOWNGRADING mode %s -> ucx_tcp with TLS=tcp,cuda_copy",
                first_err, self.mode.value,
            )
            ucp_options["TLS"] = "tcp,cuda_copy"
            ucp_options.pop("NET_DEVICES", None)
            ucp_options.pop("RCACHE_ENABLE", None)
            if not _USER_RNDV_THRESH:
                ucp_options["RNDV_THRESH"] = "inf"
            ucp.init(dict(ucp_options))
            self.mode = TransportMode.UCX_TCP
        finally:
            os.environ.update(hidden)

    async def _start_ucx(self) -> None:
        """Start UCX transport."""
        if not UCX_AVAILABLE:
            raise RuntimeError("UCX not available")
        
        # Log UCX version and available transports
        logger.info("UCX version: %s", ucp.get_ucx_version())

        ucp_options = self._build_ucp_options()
        logger.info(
            "UCX init requested: mode=%s TLS=%s NET_DEVICES=%s SOCKADDR_TLS_PRIORITY=%s RNDV_THRESH=%s RCACHE_ENABLE=%s",
            self.mode.value,
            ucp_options.get("TLS"),
            ucp_options.get("NET_DEVICES"),
            ucp_options.get("SOCKADDR_TLS_PRIORITY"),
            ucp_options.get("RNDV_THRESH"),
            ucp_options.get("RCACHE_ENABLE"),
        )
        self._init_ucp(ucp_options)
        logger.info(
            "UCX initialized: mode=%s TLS=%s NET_DEVICES=%s RNDV_THRESH=%s",
            self.mode.value,
            ucp_options.get("TLS"),
            ucp_options.get("NET_DEVICES"),
            ucp_options.get("RNDV_THRESH"),
        )

        # Persist final options for diagnostics.
        self._ucp_options = dict(ucp_options)
        self._ucx_tls = str(ucp_options.get("TLS")) if ucp_options.get("TLS") is not None else None
        self._ucx_net_devices = (
            str(ucp_options.get("NET_DEVICES")) if ucp_options.get("NET_DEVICES") is not None else None
        )
        self._ucx_sockaddr_tls_priority = (
            str(ucp_options.get("SOCKADDR_TLS_PRIORITY"))
            if ucp_options.get("SOCKADDR_TLS_PRIORITY") is not None
            else None
        )
        
        # Create listener with receive loop for each connection
        async def accept_callback(conn):
            # Generate a unique ID for this connection
            # Note: conn.get_ucp_worker() may return an int (worker handle) not an object
            import uuid
            conn_id = str(uuid.uuid4())[:8]
            peer_addr = f"ucx-peer-{conn_id}"
            
            with self._lock:
                self._ucp_endpoints[peer_addr] = conn
            logger.info("UCX: Accepted connection, assigned peer_id=%s", peer_addr)
            
            # Start a receive loop for this connection
            task = asyncio.create_task(self._ucx_recv_loop(conn, peer_addr))
            self._ucx_tasks.add(task)
            task.add_done_callback(self._ucx_tasks.discard)
            
        last_err: Optional[BaseException] = None
        for port in range(self.local_port, self.local_port + self.port_fallback + 1):
            try:
                self._ucp_listener = ucp.create_listener(accept_callback, port=port)
            except Exception as e:  # typically "Device is busy" (port in use)
                last_err = e
                continue
            if port != self.local_port:
                logger.warning("UCX port %d unavailable (%s); listening on %d instead",
                               self.local_port, last_err, port)
                self.local_port = port
            break
        else:
            raise last_err
    
    async def _ucx_recv_loop(self, endpoint: Any, peer_addr: str) -> None:
        """Receive loop for a UCX endpoint - puts received data into queue.
        
        v1.7.5: Updated to handle GPU Direct two-phase protocol:
        1. First message: metadata (contains shape, dtype, nbytes, is_cuda flag)
        2. If is_cuda: Second recv() into pre-allocated CUDA tensor
        
        Metadata is the struct-packed GDIR header only (never pickle).
        """
        
        logger.info("UCX: Starting recv loop for %s", peer_addr)
        try:
            while self._running:
                try:
                    # Use recv_obj for variable-length data transfer
                    # This receives either regular data OR GPU Direct metadata
                    data = await endpoint.recv_obj()
                    
                    if data is not None:
                        # Check if this is GPU Direct metadata
                        is_gpu_direct = False
                        gpu_direct_metadata = None
                        
                        if isinstance(data, (bytes, bytearray)):
                            # Struct-based GPU Direct metadata (see _send_ucx_cuda).
                            # Never unpickle peer bytes: that would run arbitrary
                            # code from the network, and no sender emits pickled
                            # metadata anymore.
                            gpu_direct_metadata = _parse_gdir_metadata(data)
                            is_gpu_direct = gpu_direct_metadata is not None

                        if is_gpu_direct and gpu_direct_metadata:
                            # GPU Direct: receive CUDA tensor in second phase
                            shape = gpu_direct_metadata["shape"]
                            dtype = gpu_direct_metadata["dtype"]
                            nbytes = gpu_direct_metadata["nbytes"]
                            
                            # v1.8.12: Use tensor cache instead of allocation
                            numel = 1
                            for s in shape:
                                numel *= s
                            recv_tensor = self._cuda_tensor_cache.acquire(numel, dtype, self._recv_cuda_device())
                            
                            # Receive directly into CUDA tensor
                            logger.debug("UCX: GPU Direct recv %s from %s (nbytes=%d)",
                                        shape, peer_addr, nbytes)
                            await endpoint.recv(recv_tensor)
                            
                            # Reshape and put in queue
                            result_tensor = recv_tensor.reshape(shape)
                            self._recv_queue.put((peer_addr, result_tensor))
                            self.stats.bytes_received += nbytes
                            self.stats.messages_received += 1
                        else:
                            # Standard: received bytes/bytearray or other data
                            data_size = len(data) if hasattr(data, '__len__') else 0
                            logger.debug("UCX: Received %d bytes from %s", data_size, peer_addr)
                            
                            self._recv_queue.put((peer_addr, data))
                            self.stats.bytes_received += data_size
                            self.stats.messages_received += 1
                except Exception as e:
                    if self._running:
                        logger.warning("UCX recv error from %s: %s", peer_addr, e)
                    break
        finally:
            logger.info("UCX: Recv loop ended for %s", peer_addr)
            with self._lock:
                self._ucp_endpoints.pop(peer_addr, None)
            if self._running:
                for cb in list(self.peer_closed_callbacks):
                    try:
                        cb(peer_addr)
                    except Exception:
                        logger.exception("peer_closed callback failed")
        
    def _start_zmq(self) -> None:
        """Start ZeroMQ fallback transport."""
        self._zmq_ctx = zmq.Context.instance()
        
        # Create ROUTER socket for receiving
        router = self._zmq_ctx.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(f"tcp://{self.local_address}:{self.local_port}")
        self._zmq_sockets["router"] = router
        
        logger.info("ZeroMQ: ROUTER socket bound to %s:%d", self.local_address, self.local_port)
        
    async def _shutdown_ucx(self) -> None:
        """Tear down UCX state on the loop that runs it (call on that loop).

        Order matters: listener first (no new accepts), then endpoints
        (aborted: a graceful close to a dead peer can block for seconds),
        then our receive-loop tasks, and finally ucx-py's progress task for
        this loop. If the progress task (BlockingMode._arm_worker) is left
        pending, it is destroyed at interpreter exit after ucx-py closed its
        socketpair, and asyncio prints "Task exception was never retrieved
        ... Bad file descriptor".
        """
        loop = asyncio.get_running_loop()
        if self._ucp_listener is not None:
            try:
                self._ucp_listener.close()
            except Exception:
                pass
            self._ucp_listener = None

        with self._lock:
            eps = {id(ep): ep for ep in list(self._ucp_endpoints.values()) + list(self._endpoints.values())}
            self._ucp_endpoints.clear()
            self._endpoints.clear()
        for ep in eps.values():
            try:
                if not ep.closed():
                    ep.abort()
            except Exception:
                pass

        current = asyncio.current_task()
        tasks = [t for t in self._ucx_tasks if t is not current and not t.done()]
        tasks += [f for f in self._pending_sends.values()
                  if isinstance(f, asyncio.Task) and not f.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=2.0)
        self._ucx_tasks.clear()

        # ucx-py's per-loop progress task.
        ctx = getattr(getattr(ucp, "core", None), "_ctx", None) if UCX_AVAILABLE else None
        if ctx is not None:
            for pt in [pt for pt in list(ctx.progress_tasks) if getattr(pt, "event_loop", None) is loop]:
                try:
                    if getattr(ctx, "blocking_progress_mode", False):
                        loop.remove_reader(ctx.epoll_fd)
                except Exception:
                    pass
                at = getattr(pt, "asyncio_task", None)
                if at is not None and not at.done():
                    at.cancel()
                    try:
                        await asyncio.wait([at], timeout=1.0)
                    except Exception:
                        pass
                try:
                    ctx.progress_tasks.remove(pt)
                except ValueError:
                    pass

    async def stop(self) -> None:
        """Stop the transport layer."""
        self._running = False

        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            loop = self._ucx_loop
            if loop is not None and loop.is_running() and loop is not asyncio.get_running_loop():
                self._stop_ucx_loop()
            else:
                await self._shutdown_ucx()

        # Close ZeroMQ sockets
        for sock in self._zmq_sockets.values():
            try:
                sock.close()
            except Exception:
                pass
        self._zmq_sockets.clear()
        
        # Cleanup caches
        self.mr_cache.clear()

    def _stop_ucx_loop(self, timeout: float = 5.0) -> None:
        """Shut down UCX on the background loop, then stop and join that loop."""
        loop = self._ucx_loop
        if loop is None:
            return
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._shutdown_ucx(), loop)
            try:
                fut.result(timeout)
            except Exception as e:
                logger.debug("UCX shutdown incomplete: %s", e)
            loop.call_soon_threadsafe(loop.stop)
        thread = self._ucx_loop_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._ucx_loop_thread = None

    def stop_sync(self) -> None:
        """Stop the transport layer (synchronous version)."""
        self._running = False

        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            self._stop_ucx_loop()

        # Close ZeroMQ sockets
        for sock in self._zmq_sockets.values():
            try:
                sock.close()
            except Exception:
                pass
        self._zmq_sockets.clear()
        
        # Cleanup caches
        self.mr_cache.clear()
        
    def close_endpoint(self, endpoint: Any) -> bool:
        """Close one peer connection and drop it from the endpoint caches.

        Safe to call from any thread and does not block: a UCX endpoint is
        closed asynchronously on the transport's event loop. Returns True if
        the endpoint was known.
        """
        with self._lock:
            keys = [k for k, ep in self._endpoints.items() if ep is endpoint]
            for k in keys:
                del self._endpoints[k]
            ucp_keys = [k for k, ep in self._ucp_endpoints.items() if ep is endpoint]
            for k in ucp_keys:
                del self._ucp_endpoints[k]
            zmq_keys = [k for k, s in self._zmq_sockets.items()
                        if s is endpoint and k != "router"]
            for k in zmq_keys:
                del self._zmq_sockets[k]
        known = bool(keys or ucp_keys or zmq_keys)

        if zmq_keys:
            endpoint.close()
        elif ucp_keys or (keys and self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP)):
            if self._ucx_loop is not None and self._ucx_loop.is_running():
                # Do not wait: closing an endpoint whose peer died can take
                # seconds, and callers are usually on the data path.
                future = asyncio.run_coroutine_threadsafe(endpoint.close(), self._ucx_loop)

                def _log_close_error(f):
                    if not f.cancelled() and f.exception() is not None:
                        logger.debug("UCX endpoint close failed: %s", f.exception())

                future.add_done_callback(_log_close_error)
            else:
                try:
                    endpoint.abort()
                except Exception as e:
                    logger.warning("UCX endpoint abort failed: %s", e)
        return known

    async def connect(self, peer_address: str, peer_port: int) -> Any:
        """Connect to a peer.
        
        Args:
            peer_address: Peer IP address
            peer_port: Peer port
            
        Returns:
            Connection handle
        """
        peer_key = f"{peer_address}:{peer_port}"
        
        with self._lock:
            if peer_key in self._endpoints:
                return self._endpoints[peer_key]
                
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            endpoint = await self._connect_ucx(peer_address, peer_port)
        else:
            endpoint = self._connect_zmq(peer_address, peer_port)
            
        with self._lock:
            self._endpoints[peer_key] = endpoint
            
        logger.info("Connected to peer %s", peer_key)
        return endpoint
        
    def connect_sync(self, peer_address: str, peer_port: int) -> Any:
        """Connect to a peer (synchronous version)."""
        peer_key = f"{peer_address}:{peer_port}"
        
        with self._lock:
            if peer_key in self._endpoints:
                return self._endpoints[peer_key]
        
        import asyncio
        import concurrent.futures
                
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            # Use the UCX event loop thread for connection
            if self._ucx_loop is None or not self._ucx_loop.is_running():
                # Fallback: create a new loop if UCX loop not available
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                endpoint = loop.run_until_complete(
                    self._connect_ucx(peer_address, peer_port)
                )
            else:
                # Run connection in the UCX event loop thread using run_coroutine_threadsafe
                coro = self._connect_ucx(peer_address, peer_port)
                future = asyncio.run_coroutine_threadsafe(coro, self._ucx_loop)
                try:
                    endpoint = future.result(timeout=10.0)  # 10 second timeout
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"UCX connect timeout to {peer_key}")
        else:
            endpoint = self._connect_zmq(peer_address, peer_port)
            
        with self._lock:
            self._endpoints[peer_key] = endpoint
            
        logger.info("Connected to peer %s (sync)", peer_key)
        return endpoint
        
    async def _connect_ucx(self, peer_address: str, peer_port: int) -> Any:
        """Connect via UCX with minimal handshake delay.
        
        v1.5.5: Reduced delay for faster connection establishment.
        """
        if not UCX_AVAILABLE:
            raise RuntimeError("UCX not available")
            
        endpoint = await ucp.create_endpoint(peer_address, peer_port)
        peer_key = f"{peer_address}:{peer_port}"
        self._ucp_endpoints[peer_key] = endpoint
        
        # v1.5.5: Minimal delay - UCX handshake is already complete after create_endpoint
        # Only add 1ms delay for connection stabilization
        await asyncio.sleep(0.001)
        
        return endpoint
        
    def _connect_zmq(self, peer_address: str, peer_port: int) -> zmq.Socket:
        """Connect via ZeroMQ."""
        dealer = self._zmq_ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(f"tcp://{peer_address}:{peer_port}")
        
        peer_key = f"{peer_address}:{peer_port}"
        self._zmq_sockets[peer_key] = dealer
        return dealer
        
    async def send(
        self,
        endpoint: Any,
        data: bytes | bytearray | memoryview | torch.Tensor,
        tag: int = 0,
    ) -> None:
        """Send data to endpoint.
        
        Args:
            endpoint: Connection handle
            data: Data to send (bytes or tensor)
            tag: Message tag for tagged messaging
            
        v1.7.4.8: Supports GPU Direct RDMA when UCX is configured with cuda_copy.
        CUDA tensors are sent directly without CPU transfer.
        """
        start = time.time()
        data_len = 0
        
        if isinstance(data, torch.Tensor):
            data_len = data.numel() * data.element_size()
            # v1.7.4.8: GPU Direct - send CUDA tensor directly if supported
            if data.is_cuda and self._supports_gpu_direct() and self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
                # UCX-Py can send CUDA tensors directly via __cuda_array_interface__
                # Ensure tensor is contiguous for direct transfer
                if not data.is_contiguous():
                    data = data.contiguous()
                await self._send_ucx_cuda(endpoint, data, tag)
            else:
                # Fallback to CPU bytes transfer
                data = self._tensor_to_bytes(data)
                data_len = len(data)
                if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
                    await self._send_ucx(endpoint, data, tag)
                else:
                    self._send_zmq(endpoint, data)
        else:
            data_len = len(data) if hasattr(data, '__len__') else 0
            if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
                await self._send_ucx(endpoint, data, tag)
            else:
                self._send_zmq(endpoint, data)
            
        elapsed = time.time() - start
        self.stats.bytes_sent += data_len
        self.stats.messages_sent += 1
        self.stats.send_latency_sum += elapsed
        
    def send_sync(
        self,
        endpoint: Any,
        data: bytes | bytearray | memoryview | torch.Tensor,
        tag: int = 0,
    ) -> None:
        """Send data (synchronous version with optimized UCX path).
        
        v1.5.5: Optimize UCX send path to reduce asyncio overhead.
        Use direct endpoint send when possible.
        
        v1.7.4.8: Supports GPU Direct RDMA for CUDA tensors.
        """
        import asyncio
        import concurrent.futures
        
        start = time.time()
        data_len = 0
        use_gpu_direct = False
        
        if isinstance(data, torch.Tensor):
            data_len = data.numel() * data.element_size()
            # v1.7.4.8: GPU Direct path
            if data.is_cuda and self._supports_gpu_direct() and self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
                use_gpu_direct = True
                if not data.is_contiguous():
                    data = data.contiguous()
            else:
                data = self._tensor_to_bytes(data)
                data_len = len(data)
        else:
            data_len = len(data) if hasattr(data, '__len__') else 0
            
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            # v1.5.5: Try direct send if UCX loop is running
            if self._ucx_loop is not None and self._ucx_loop.is_running():
                # Run send in the UCX event loop thread
                if use_gpu_direct:
                    coro = self._send_ucx_cuda(endpoint, data, tag)
                else:
                    coro = self._send_ucx(endpoint, data, tag)
                future = asyncio.run_coroutine_threadsafe(coro, self._ucx_loop)
                try:
                    # v1.5.5: Reduced timeout from 30s to 10s for faster failure detection
                    future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError("UCX send timeout")
            else:
                # Fallback: create a new loop if UCX loop not available
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._send_ucx(endpoint, data, tag))
        else:
            self._send_zmq(endpoint, data)
            
        elapsed = time.time() - start
        self.stats.bytes_sent += len(data)
        self.stats.messages_sent += 1
        self.stats.send_latency_sum += elapsed

    def send_async(
        self,
        endpoint: Any,
        data: bytes | bytearray | memoryview | torch.Tensor,
        tag: int = 0,
    ) -> Optional[Any]:
        """Send data asynchronously without waiting for completion.
        
        v1.5.7: Fire-and-forget async send for lower enqueue latency.
        v1.7.4.8: Support GPU Direct for CUDA tensors.
        The data is queued for transmission but send() returns immediately.
        
        Returns:
            Future object that can be used to check completion (optional)
        """
        import asyncio
        
        start = time.time()
        data_size = 0
        
        # v1.7.4.8: Check if this is a CUDA tensor for GPU Direct
        is_cuda_tensor = isinstance(data, torch.Tensor) and data.is_cuda
        
        if isinstance(data, torch.Tensor):
            data_size = data.numel() * data.element_size()
            if not is_cuda_tensor:
                # CPU tensor: convert to bytes
                data = self._tensor_to_bytes(data)
        else:
            data_size = len(data) if hasattr(data, '__len__') else 0
            
        future = None
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            if self._ucx_loop is not None and self._ucx_loop.is_running():
                # Submit to UCX event loop without waiting
                if is_cuda_tensor:
                    # v1.7.4.8: GPU Direct path for CUDA tensors
                    coro = self._send_ucx_cuda(endpoint, data, tag)
                else:
                    coro = self._send_ucx(endpoint, data, tag)
                future = asyncio.run_coroutine_threadsafe(coro, self._ucx_loop)
                # Don't wait for result - fire and forget
            else:
                # Fallback to sync if no event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                if is_cuda_tensor:
                    loop.run_until_complete(self._send_ucx_cuda(endpoint, data, tag))
                else:
                    loop.run_until_complete(self._send_ucx(endpoint, data, tag))
        else:
            if is_cuda_tensor:
                # ZeroMQ doesn't support GPU Direct, convert to bytes
                data = self._tensor_to_bytes(data.cpu())
                data_size = len(data)
            self._send_zmq(endpoint, data)
            
        elapsed = time.time() - start
        self.stats.bytes_sent += data_size
        self.stats.messages_sent += 1
        self.stats.send_latency_sum += elapsed
        
        return future
    
    def send_cuda_raw(self, endpoint: Any, tensor: torch.Tensor) -> Optional[Any]:
        """Send CUDA tensor directly without metadata (for batch GPU Direct).
        
        v1.7.5.4: Public API for raw CUDA tensor send.
        Caller must handle metadata separately (e.g., in header message).
        
        Args:
            endpoint: UCX endpoint
            tensor: CUDA tensor to send (must be contiguous)
            
        Returns:
            Future object for completion tracking (optional)
        """
        import asyncio
        
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("send_cuda_raw requires CUDA tensor")
        
        data_size = tensor.numel() * tensor.element_size()
        future = None
        
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            if self._ucx_loop is not None and self._ucx_loop.is_running():
                coro = self._send_ucx_cuda_raw(endpoint, tensor)
                future = asyncio.run_coroutine_threadsafe(coro, self._ucx_loop)
            else:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._send_ucx_cuda_raw(endpoint, tensor))
        else:
            raise RuntimeError("send_cuda_raw requires UCX transport for GPU Direct")
        
        self.stats.bytes_sent += data_size
        self.stats.messages_sent += 1
        
        return future
    
    def recv_cuda_raw(
        self, 
        endpoint: Any, 
        shape: tuple, 
        dtype: torch.dtype,
        timeout: float = 30.0
    ) -> torch.Tensor:
        """Receive CUDA tensor directly into pre-allocated GPU memory.
        
        v1.7.5.4: Public API for raw CUDA tensor receive (zero-copy GPU Direct).
        Caller provides tensor metadata (shape, dtype) from header message.
        
        Args:
            endpoint: UCX endpoint  
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            timeout: Receive timeout in seconds
            
        Returns:
            Received CUDA tensor
        """
        import asyncio
        
        if self.mode not in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            raise RuntimeError("recv_cuda_raw requires UCX transport for GPU Direct")
        
        # Pre-allocate CUDA tensor from cache
        numel = 1
        for s in shape:
            numel *= s
        recv_tensor = self._cuda_tensor_cache.acquire(numel, dtype, self._recv_cuda_device())
        
        async def _recv():
            await endpoint.recv(recv_tensor)
        
        if self._ucx_loop is not None and self._ucx_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_recv(), self._ucx_loop)
            future.result(timeout=timeout)
        else:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(_recv())
        
        data_size = numel * recv_tensor.element_size()
        self.stats.bytes_received += data_size
        self.stats.messages_received += 1
        
        return recv_tensor.reshape(shape)
    
    def recv_cuda_raw_from_queue(
        self,
        shape: tuple,
        dtype: torch.dtype,
        timeout: float = 30.0
    ) -> torch.Tensor:
        """Receive raw CUDA tensor from the queue (for N+1 protocol).
        
        v1.7.5.5: Optimized receive for raw tensor protocol.
        The sender sends raw tensors without per-tensor metadata.
        We receive from the queue and reshape using caller-provided metadata.
        
        This avoids:
        - pickle.loads() overhead on every recv
        - Extra metadata message per tensor
        
        Args:
            shape: Expected tensor shape (from header metadata)
            dtype: Expected tensor dtype (from header metadata)
            timeout: Receive timeout in seconds
            
        Returns:
            Received CUDA tensor, properly shaped
        """
        numel = 1
        for s in shape:
            numel *= s
        # v1.8.18: Use pre-computed element size
        expected_bytes = numel * DTYPE_ELEMENT_SIZE.get(dtype, 4)
        
        # Get data from queue
        result = self.recv_from_queue(timeout=timeout)
        if result is None:
            raise TimeoutError(f"recv_cuda_raw_from_queue timeout after {timeout}s")
        
        _, data = result
        
        # v1.7.5.5: Handle the raw tensor data
        if isinstance(data, torch.Tensor):
            # Already a tensor (from background recv loop that detected GPU Direct)
            if data.device.type == "cuda":
                if data.numel() == numel:
                    return data.view(dtype).reshape(shape) if data.dtype != dtype else data.reshape(shape)
                else:
                    raise ValueError(f"Tensor size mismatch: expected {numel}, got {data.numel()}")
            else:
                return data.to(dtype=dtype, device="cuda").reshape(shape)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            # Raw bytes - this shouldn't happen with raw GPU Direct protocol
            # but handle it as fallback
            data_bytes = bytes(data) if isinstance(data, memoryview) else data
            cpu_tensor = torch.frombuffer(bytearray(data_bytes), dtype=dtype)
            if cpu_tensor.numel() != numel:
                raise ValueError(f"Data size mismatch: expected {numel} elements, got {cpu_tensor.numel()}")
            return cpu_tensor.cuda().reshape(shape)
        else:
            raise TypeError(f"Unexpected data type from queue: {type(data)}")
        
    async def _send_ucx(self, endpoint: Any, data: bytes | bytearray | memoryview, tag: int) -> None:
        """Send via UCX object transfer (supports variable-length data).
        
        Note: We use send_obj/recv_obj for reliable variable-length transfers.
        These methods handle object serialization internally.
        """
        # Check MR cache
        data_ptr = id(data)
        mr = self.mr_cache.get(data_ptr)
        
        if mr:
            self.stats.mr_cache_hits += 1
        else:
            self.stats.mr_cache_misses += 1
        
        # Use send_obj which handles variable-length data
        await endpoint.send_obj(data)
    
    async def _send_ucx_cuda(self, endpoint: Any, tensor: torch.Tensor, tag: int) -> None:
        """Send CUDA tensor directly via GPU Direct RDMA.
        
        v1.7.5: Correct GPU Direct implementation using UCX send() instead of send_obj().
        v1.7.5.5: Use struct-based metadata for faster serialization.
        
        UCX-Py's send(buffer) accepts objects with __cuda_array_interface__,
        enabling true zero-copy GPU-to-GPU transfer without pickle serialization.
        
        Protocol:
        1. First send metadata (struct-packed, ~20 bytes) via send_obj
        2. Then send the raw CUDA buffer via send(tensor)
        """
        if not tensor.is_cuda:
            # Fallback to bytes transfer for non-CUDA tensors
            await self._send_ucx(endpoint, self._tensor_to_bytes(tensor), tag)
            return
            
        # Ensure tensor is contiguous for direct transfer
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # v1.7.5.5: Use struct-based metadata (much faster than pickle)
        # Format: [magic(4)][dtype(1)][ndim(1)][numel(4)][shape dims...]
        # v1.8.18: Use module-level TORCH_TO_DTYPE_IDX constant
        dtype_idx = TORCH_TO_DTYPE_IDX.get(tensor.dtype)
        if dtype_idx is None:
            # No GDIR code for this dtype: defaulting to float32 would make the
            # receiver allocate a wrong-sized buffer. Use the bytes path.
            await self._send_ucx(endpoint, self._tensor_to_bytes(tensor), tag)
            return
        ndim = len(tensor.shape)
        numel = tensor.numel()
        
        # Pack metadata: magic + dtype_idx + ndim + numel + shape
        meta_bytes = b'GDIR' + struct.pack(f'!BBI{ndim}I', dtype_idx, ndim, numel, *tensor.shape)
        
        # Send metadata with send_obj (variable length, small)
        await endpoint.send_obj(meta_bytes)
        
        _bind_cuda_device(tensor.device)
        # Send CUDA tensor directly using send()
        logger.debug("GPU Direct send: shape=%s, dtype=%s, numel=%d", 
                    tensor.shape, tensor.dtype, numel)
        await endpoint.send(tensor)
    
    async def _send_ucx_cuda_raw(self, endpoint: Any, tensor: torch.Tensor) -> None:
        """Send CUDA tensor directly via GPU Direct without metadata.
        
        v1.7.5.4: Raw CUDA tensor send for batch GPU Direct protocol.
        Caller is responsible for sending metadata separately.
        
        This is used when the application layer (connex_communicator) handles
        metadata packing in the header message.
        """
        if not tensor.is_cuda:
            raise ValueError("_send_ucx_cuda_raw requires CUDA tensor")
            
        # Ensure tensor is contiguous for direct transfer
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        _bind_cuda_device(tensor.device)
        # Send CUDA tensor directly using send()
        await endpoint.send(tensor)
    
    async def _send_batch_gpu_direct(
        self, 
        endpoint: Any, 
        header: bytes | bytearray, 
        tensors: list[torch.Tensor],
        tag: int
    ) -> None:
        """Send header + multiple CUDA tensors sequentially (no interleaving).
        
        v1.7.5.4: Fix race condition by sending all messages in one coroutine.
        v1.7.5.5: Uses struct-based metadata for faster per-tensor serialization.
        v1.9.0: Use per-endpoint lock to prevent interleaving between concurrent requests.
        v1.9.2: Fixed race condition in lock creation using setdefault.
        
        Protocol:
        - 1 header message (application-level metadata)
        - For each tensor: struct metadata (20 bytes) + raw tensor
        
        Args:
            endpoint: UCX endpoint
            header: Header bytes (application metadata)
            tensors: List of CUDA tensors to send
            tag: UCX tag
        """
        # v1.9.2: Use setdefault for thread-safe lock creation
        # This ensures only one lock per endpoint, even with concurrent calls
        endpoint_id = id(endpoint)
        lock = self._endpoint_send_locks.setdefault(endpoint_id, asyncio.Lock())
        
        async with lock:
            # Send header first (application-level metadata)
            await endpoint.send_obj(header)
            
            # Send each tensor with struct-based metadata (much faster than pickle)
            for tensor in tensors:
                if tensor.is_cuda:
                    await self._send_ucx_cuda(endpoint, tensor, tag)
                else:
                    await self._send_ucx(endpoint, self._tensor_to_bytes(tensor), tag)
    
    def send_batch_gpu_direct(
        self,
        endpoint: Any,
        header: bytes | bytearray,
        tensors: list[torch.Tensor],
        tag: int = 0,
        async_mode: bool = False,
    ) -> tuple[float, float]:
        """Send header + CUDA tensors as a batch (thread-safe, no interleaving).
        
        v1.7.5.4: Public API for batch GPU Direct send.
        This ensures metadata+tensor pairs are sent atomically.
        
        v1.9.3: CRITICAL FIX - Wait for send completion using threading.Event.
        The previous fire-and-forget mode allowed concurrent sends to interleave
        in the receive queue, causing shape mismatch errors.
        
        v1.9.4: Added async_mode parameter for fire-and-forget send.
        When async_mode=True, the function returns immediately after submitting
        the send to asyncio, without waiting for completion. Errors are tracked
        and can be checked via get_pending_send_errors().
        
        The per-endpoint lock in _send_batch_gpu_direct ensures atomic send,
        and we now wait for the coroutine to complete using a threading Event
        (which works across threads, unlike asyncio.Future.result() which can deadlock).
        
        Args:
            endpoint: UCX endpoint
            header: Header bytes containing all tensor metadata
            tensors: List of CUDA tensors to send
            tag: UCX tag
            async_mode: If True, return immediately without waiting (fire-and-forget)
            
        Returns:
            Tuple of (enqueue_time, wait_time) in seconds:
            - enqueue_time: time to submit the send to asyncio
            - wait_time: time waiting for send completion (0 if async_mode=True)
        """
        import asyncio
        import threading
        
        if self.mode not in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            raise RuntimeError("send_batch_gpu_direct requires UCX transport")
        
        # Make all tensors contiguous
        tensors = [t.contiguous() if t.is_cuda and not t.is_contiguous() else t for t in tensors]
        
        enqueue_time = 0.0
        wait_time = 0.0
        
        if self._ucx_loop is not None and self._ucx_loop.is_running():
            if async_mode:
                # v1.9.4: Fire-and-forget mode - submit and return immediately
                # The per-endpoint lock in _send_batch_gpu_direct still ensures
                # atomic send (header + all tensors together)
                with self._pending_send_lock:
                    send_id = self._next_send_id
                    self._next_send_id += 1
                
                async def send_and_track():
                    try:
                        await self._send_batch_gpu_direct(endpoint, header, tensors, tag)
                    except Exception as e:
                        # Track error for later reporting
                        with self._pending_send_lock:
                            self._pending_send_errors.append(e)
                        logger.warning("Async send %d failed: %s", send_id, e)
                    finally:
                        with self._pending_send_lock:
                            self._pending_sends.pop(send_id, None)
                
                enqueue_start = time.time()
                future = asyncio.run_coroutine_threadsafe(send_and_track(), self._ucx_loop)
                with self._pending_send_lock:
                    self._pending_sends[send_id] = future
                enqueue_time = time.time() - enqueue_start
                # wait_time = 0 for async mode
            else:
                # v1.9.3: Sync mode - wait for completion using threading.Event
                completion_event = threading.Event()
                exception_holder = [None]  # Use list to allow mutation in callback
                
                async def send_and_signal():
                    try:
                        await self._send_batch_gpu_direct(endpoint, header, tensors, tag)
                    except Exception as e:
                        exception_holder[0] = e
                    finally:
                        completion_event.set()
                
                enqueue_start = time.time()
                asyncio.run_coroutine_threadsafe(send_and_signal(), self._ucx_loop)
                enqueue_time = time.time() - enqueue_start
                
                # Wait for send to complete (with timeout)
                wait_start = time.time()
                if not completion_event.wait(timeout=60.0):
                    logger.error("send_batch_gpu_direct timed out after 60s")
                    raise RuntimeError("UCX send timeout")
                wait_time = time.time() - wait_start
                
                # Check if send raised an exception
                if exception_holder[0] is not None:
                    raise exception_holder[0]
        else:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            enqueue_start = time.time()
            loop.run_until_complete(self._send_batch_gpu_direct(endpoint, header, tensors, tag))
            enqueue_time = time.time() - enqueue_start
        
        # Update stats
        total_bytes = len(header) + sum(t.numel() * t.element_size() for t in tensors)
        self.stats.bytes_sent += total_bytes
        self.stats.messages_sent += 1 + len(tensors) * 2  # header + (meta + tensor) per tensor
        
        return enqueue_time, wait_time
    
    def get_pending_send_count(self) -> int:
        """Get the number of pending async sends.
        
        v1.9.4: Returns count of sends that have been submitted but not completed.
        """
        with self._pending_send_lock:
            return len(self._pending_sends)
    
    def get_pending_send_errors(self, clear: bool = True) -> list[Exception]:
        """Get errors from failed async sends.
        
        v1.9.4: Returns list of exceptions from async sends that failed.
        
        Args:
            clear: If True, clear the error list after returning
            
        Returns:
            List of exceptions from failed sends
        """
        with self._pending_send_lock:
            errors = list(self._pending_send_errors)
            if clear:
                self._pending_send_errors.clear()
            return errors
        
    def _send_zmq(self, endpoint: zmq.Socket, data: bytes) -> None:
        """Send via ZeroMQ."""
        endpoint.send(data)
        
    async def recv(
        self,
        endpoint: Any = None,
        tag: int = 0,
        timeout_ms: int = -1,
    ) -> Optional[bytes]:
        """Receive data.
        
        Args:
            endpoint: Connection handle (uses router if None)
            tag: Expected message tag
            timeout_ms: Receive timeout (-1 for blocking)
            
        Returns:
            Received data or None if timeout
        """
        start = time.time()
        
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            data = await self._recv_ucx(endpoint, tag)
        else:
            data = self._recv_zmq(timeout_ms)
            
        if data:
            elapsed = time.time() - start
            self.stats.bytes_received += len(data)
            self.stats.messages_received += 1
            self.stats.recv_latency_sum += elapsed
            
        return data
        
    def recv_sync(
        self,
        endpoint: Any = None,
        tag: int = 0,
        timeout_ms: int = -1,
    ) -> Optional[bytes]:
        """Receive data (synchronous version)."""
        start = time.time()
        
        if self.mode in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            data = loop.run_until_complete(self._recv_ucx(endpoint, tag))
        else:
            data = self._recv_zmq(timeout_ms)
            
        if data:
            elapsed = time.time() - start
            self.stats.bytes_received += len(data)
            self.stats.messages_received += 1
            self.stats.recv_latency_sum += elapsed
            
        return data
        
    async def _recv_ucx(self, endpoint: Any, tag: int) -> bytes:
        """Receive via UCX object transfer."""
        return await endpoint.recv_obj()
    
    async def _recv_ucx_cuda(self, endpoint: Any, tag: int) -> torch.Tensor:
        """Receive CUDA tensor directly via GPU Direct RDMA.
        
        v1.7.5: Correct GPU Direct receive implementation.
        
        Protocol (matches _send_ucx_cuda):
        1. First receive metadata via recv_obj()
        2. Pre-allocate CUDA tensor of the right size
        3. Use recv(pre_allocated_tensor) for zero-copy GPU Direct receive
        
        Returns:
            The received CUDA tensor
        """
        # Step 1: Receive the struct-packed GDIR metadata sent by _send_ucx_cuda
        meta_bytes = await endpoint.recv_obj()
        metadata = _parse_gdir_metadata(meta_bytes)
        if metadata is None:
            raise ValueError("expected GDIR metadata before GPU Direct payload")
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        nbytes = metadata["nbytes"]
        
        # Step 2: Pre-allocate CUDA tensor from cache
        # We need to allocate the exact number of elements
        numel = 1
        for s in shape:
            numel *= s
        recv_tensor = self._cuda_tensor_cache.acquire(numel, dtype, self._recv_cuda_device())
        
        # Step 3: Receive directly into CUDA tensor via GPU Direct
        # UCX-Py recv() accepts __cuda_array_interface__ objects
        logger.debug("GPU Direct recv: shape=%s, dtype=%s, nbytes=%d",
                    shape, dtype, nbytes)
        await endpoint.recv(recv_tensor)
        
        # Reshape to original shape
        return recv_tensor.reshape(shape)
        
    def _recv_zmq(self, timeout_ms: int = -1) -> Optional[bytes]:
        """Receive via ZeroMQ."""
        router = self._zmq_sockets.get("router")
        if not router:
            return None
            
        if timeout_ms >= 0:
            if not router.poll(timeout_ms):
                return None
                
        frames = router.recv_multipart()
        if len(frames) >= 2:
            return frames[-1]  # Data is last frame
        return None
        
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes for transfer.
        
        v1.7.4.8: Only used for non-GPU-Direct paths.
        When GPU Direct is enabled, tensors are sent directly.
        """
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy().tobytes()
    
    def _capture_cuda_device(self) -> None:
        """Remember the caller's CUDA device (called from start/start_sync)."""
        if self._cuda_device is None and str(self.device).startswith("cuda"):
            try:
                if torch.cuda.is_available():
                    self._cuda_device = torch.cuda.current_device()
            except Exception:
                pass

    def _recv_cuda_device(self) -> int:
        """Device for GPU-direct receives; also binds its context to this thread."""
        if self._cuda_device is None:
            self._cuda_device = torch.cuda.current_device()
        _bind_cuda_device(self._cuda_device)
        return self._cuda_device

    def _supports_gpu_direct(self) -> bool:
        """Check if GPU Direct RDMA is available.
        
        v1.7.4.8: Returns True if UCX is configured with cuda_copy transport.
        """
        if not UCX_AVAILABLE:
            return False
        # Check environment for GPU Direct support
        if os.getenv("CONNEX_GPU_DIRECT", "0") == "1":
            return True
        # Check if cuda_copy is in UCX_TLS
        ucx_tls = os.getenv("UCX_TLS", "")
        return "cuda_copy" in ucx_tls or "cuda_ipc" in ucx_tls
        
    def _bytes_to_tensor(
        self,
        data: bytes,
        dtype: torch.dtype,
        shape: tuple,
        device: str,
    ) -> torch.Tensor:
        """Convert bytes back to tensor."""
        import numpy as np
        arr = np.frombuffer(data, dtype=torch.tensor([], dtype=dtype).numpy().dtype)
        arr = arr.reshape(shape)
        tensor = torch.from_numpy(arr.copy())
        if device == "cuda":
            tensor = tensor.cuda()
        return tensor
        
    def get_stats(self) -> dict[str, Any]:
        """Get transport statistics."""
        return {
            "mode": self.mode.value,
            "ucx_tls": self._ucx_tls,
            "ucx_net_devices": self._ucx_net_devices,
            "ucx_sockaddr_tls_priority": self._ucx_sockaddr_tls_priority,
            "bytes_sent": self.stats.bytes_sent,
            "bytes_received": self.stats.bytes_received,
            "messages_sent": self.stats.messages_sent,
            "messages_received": self.stats.messages_received,
            "avg_send_latency_ms": self.stats.avg_send_latency() * 1000,
            "avg_recv_latency_ms": self.stats.avg_recv_latency() * 1000,
            "mr_cache": self.mr_cache.get_stats(),
            "buffer_pool": self.buffer_pool.get_stats(),
            "endpoints": len(self._endpoints),
        }

    def get_ucx_config(self) -> dict[str, Any]:
        """Get UCX init/config diagnostics (useful to validate RDMA vs TCP fallback)."""
        return {
            "mode": self.mode.value,
            "tls": self._ucx_tls,
            "net_devices": self._ucx_net_devices,
            "sockaddr_tls_priority": self._ucx_sockaddr_tls_priority,
            "ucp_options": dict(self._ucp_options) if self._ucp_options else None,
        }
    
    def recv_from_queue(self, timeout: float = None) -> Optional[tuple[str, bytes | bytearray]]:
        """Get received data from queue (for UCX mode).
        
        This is the main method for receiving UCX data. The actual UCX recv happens
        in the background recv loop started when connections are accepted.
        
        Args:
            timeout: Timeout in seconds. None for blocking, 0 for non-blocking.
            
        Returns:
            (peer_address, data) tuple or None if timeout
        """
        if self.mode not in (TransportMode.UCX_RDMA, TransportMode.UCX_TCP):
            return None
            
        try:
            if timeout == 0:
                return self._recv_queue.get_nowait()
            elif timeout is None:
                return self._recv_queue.get()
            else:
                return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def has_pending_data(self) -> bool:
        """Check if there's pending data in receive queue."""
        return not self._recv_queue.empty()
    
    def get_recv_queue_size(self) -> int:
        """Get number of pending messages in receive queue."""
        return self._recv_queue.qsize()
