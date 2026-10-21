# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Connex communication layer for external pipeline mode.

This implements the Connex communication backend with:
- UCX-based high-performance RDMA data transfer  
- GPU Direct RDMA for zero-copy GPU-to-GPU transfer
- ZeroMQ control plane / fallback

v1.8.22: Simplified architecture - removed unused EPC components.
"""

import os
import struct
import threading
import time
from typing import Any
from uuid import uuid4

import torch
import zmq

# Use safetensors for faster serialization
try:
    from safetensors.torch import load as safetensors_load
    from safetensors.torch import save as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Use msgpack for faster metadata serialization (fallback to pickle)
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    import pickle
    MSGPACK_AVAILABLE = False

from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_tcp_uri, split_host_port

# Import io for fallback serialization
import io
# Import pickle as fallback for msgpack
import pickle

logger = init_logger(__name__)

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

def _serialize_metadata(data: dict) -> bytes:
    """Serialize metadata using msgpack (faster) or pickle (fallback)."""
    if MSGPACK_AVAILABLE:
        return msgpack.packb(data, use_bin_type=True)
    return pickle.dumps(data)

def _deserialize_metadata(data: bytes) -> dict:
    """Deserialize metadata using msgpack or pickle."""
    if MSGPACK_AVAILABLE:
        return msgpack.unpackb(data, raw=False)
    return pickle.loads(data)

# Import metrics (optional)
try:
    from vllm.distributed.device_communicators.pipeline_comm_metrics import (
        vllm_pipeline_comm_bytes_received,
        vllm_pipeline_comm_bytes_sent,
        vllm_pipeline_comm_connection_status,
        vllm_pipeline_comm_deserialization_latency,
        vllm_pipeline_comm_errors,
        vllm_pipeline_comm_delivery_latency_local,
        vllm_pipeline_comm_messages_received,
        vllm_pipeline_comm_messages_sent,
        vllm_pipeline_comm_recv_latency,
        vllm_pipeline_comm_send_latency,
        vllm_pipeline_comm_serialization_latency,
        vllm_pipeline_comm_tensor_count,
        vllm_pipeline_comm_tensor_size,
        vllm_pipeline_comm_enqueue_latency,
        vllm_pipeline_comm_delivery_latency,
        vllm_pipeline_comm_ready_to_use_latency,
        vllm_pipeline_ucx_transport_info,
        vllm_pipeline_connection_ready_time,
        vllm_pipeline_connection_discovery_time,
        vllm_pipeline_connection_handshake_time,
        vllm_pipeline_connection_ready_timestamp,
        vllm_pipeline_connection_rdma_setup_time,
    )
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.warning("Pipeline metrics not available")

# Prefer the standalone `connex` package; fall back to a copy vendored as
# vllm.connex (layout used by the original vLLM fork).
CONNEX_AVAILABLE = False
try:
    from connex.directory import DirectoryClient
    from connex.ucx_transport import UCXTransport, TransportMode
    CONNEX_AVAILABLE = True
except ImportError:
    try:
        from vllm.connex.directory import DirectoryClient
        from vllm.connex.ucx_transport import UCXTransport, TransportMode
        CONNEX_AVAILABLE = True
    except ImportError as e:
        logger.warning("Connex EPC library not available: %s, using ZeroMQ fallback", e)


def resolve_peer_ucx_address(ucx_endpoint_str: str | None) -> tuple[str, int] | None:
    """Parse the UCX address a peer advertised in its ASSIGN reply.

    Returns (ip, port), or None when the peer advertised no usable UCX
    address. None means "talk to this peer over ZeroMQ only": the UCX port is
    never guessed from the router port, since a peer without UCX (or one that
    bound a fallback port) does not listen there and a guessed connect stalls
    or reaches the wrong process.
    """
    if not ucx_endpoint_str:
        return None
    host, sep, port_str = str(ucx_endpoint_str).strip().rpartition(":")
    if not sep or not host:
        return None
    try:
        port = int(port_str)
    except ValueError:
        return None
    if not 0 < port < 65536:
        return None
    return host.strip("[]"), port


class ConnexCommunicator:
    """Connex communicator for external pipeline mode.
    
    This communicator implements high-performance data transfer using UCX 
    with ZeroMQ as control plane/fallback.
    
    v1.8.22: Simplified - removed unused components for better performance.
    Core features:
    - UCX-based high-performance RDMA data transfer
    - GPU Direct RDMA for zero-copy GPU-to-GPU transfer
    - ZeroMQ for control plane signaling
    """

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        router_port: int,
        front_router_addr: str,
        device: str = "cuda",
        timeout_ms: int = -1,
    ):
        # Track initialization time
        init_start_time = time.time()
        
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.device = device
        self.timeout_ms = timeout_ms
        
        # Get replica_idx from environment variable (for multi-replica deployments)
        self.replica_idx = int(os.environ.get("PIPELINE_REPLICA_IDX", "0"))

        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1

        # ZeroMQ context (used for control plane and fallback)
        self.context = zmq.Context()

        # Control-plane configuration
        self.front_router_addr = front_router_addr
        self.control_socket: zmq.Socket | None = None

        # Data-plane configuration  
        self.router_port = router_port
        local_ip = self._get_net1_ip() or get_ip()
        self.router_endpoint = get_tcp_uri(local_ip, router_port)
        
        # ZeroMQ sockets (fallback data plane)
        self.router_socket: zmq.Socket | None = None
        self.forward_socket: zmq.Socket | None = None
        self.return_socket: zmq.Socket | None = None

        # EPC components (if available)
        self._epoch = 0
        self._endpoint_cache: dict[tuple[int, str], tuple[str, str, int]] = {}
        self._connection_version = 1
        
        # v1.7.1: Epoch-aware endpoint TTL management
        # Track last access time and epoch for each endpoint
        self._endpoint_last_access: dict[tuple[int, str], float] = {}  # cache_key -> timestamp
        self._endpoint_epoch: dict[tuple[int, str], int] = {}  # cache_key -> epoch when created
        self._endpoint_ttl_seconds = 300.0  # 5 minutes TTL for stale endpoints
        
        # Stream management
        self._stream_seqno: dict[str, int] = {}  # stream_id -> next_seqno
        self._pending_streams: dict[str, list] = {}  # stream_id -> pending messages
        
        # Cutover state
        self._cutover_in_progress: dict[str, bool] = {}
        
        # UCX transport state (will be set in _init_epc_components)
        self._ucx_available = False
        self._ucx_transport = None
        self._ucx_endpoints: dict[str, Any] = {}
        self._ucx_endpoint_versions: dict[str, int] = {}  # v1.8.4: Track endpoint versions
        
        # v1.8.0: True Pull model state (like Mooncake)
        # Sender allocates buffer, sends metadata via ZeroMQ
        # Receiver initiates RDMA read to pull data
        self._pending_send_buffers: dict[int, tuple[bytes, int, float]] = {}  # msg_id -> (data, size, timestamp)
        self._next_msg_id = 0
        self._pull_mode_enabled = True  # Enabled by default
        self._buffer_ttl_seconds = 0.5  # v1.8.7: Reduced from 2.0s to 0.5s for faster cleanup
        
        # v1.8.0: Async buffer cleanup (like Mooncake)
        from concurrent.futures import ThreadPoolExecutor
        self._buffer_cleaner: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        
        # v1.8.0: Default to direct UCX mode (no ZMQ signal waiting)
        # v1.8.0: Disable direct UCX recv by default - it causes issues when there are no requests
        # because the recv loop blocks forever waiting for UCX data that never arrives.
        # Instead, we use ZeroMQ signal + UCX data mode which is more robust.
        self._ucx_direct_recv = os.environ.get("CONNEX_UCX_DIRECT_RECV", "0") == "1"

        # Churn resilience: recv never blocks forever. It returns None when the
        # peer we are waiting on is known lost, or after CONNEX_RECV_TIMEOUT_MS,
        # and the caller (stage 0's pipeline retry loop) re-sends the step on a
        # refreshed route. Stage 1+ loops treat None as "nothing yet".
        self._recv_timeout_s = int(os.environ.get("CONNEX_RECV_TIMEOUT_MS", "8000")) / 1000.0
        self._peer_lost = threading.Event()
        self._waiting_recv = False
        self._last_data_peer: str | None = None  # UCX peer id of the last received message
        self._ucx_stash: dict[float, Any] = {}  # payloads that arrived before their signal
        # Request/response correlation across the pipeline. Stage 0 stamps each
        # step it sends with a fresh id; later stages echo the id they received;
        # stage 0 accepts only the return carrying the id it is waiting for, so a
        # late reply from an abandoned peer can never be taken for a retry's.
        self._corr_base = int.from_bytes(os.urandom(4), "big") << 32
        self._corr_seq = 0
        self._awaiting_corr: int | None = None
        self._corr_in: int | None = None
        self._ucx_discard: set[float] = set()  # payload timestamps of dropped signals
        self._assigned_lids: dict[int, dict | None] = {}  # target stage -> router LID
        self._registration_request: dict | None = None
        
        # v1.9.0: Receive lock to prevent concurrent recv_tensor_dict calls from
        # interleaving when taking header + tensors from the queue
        self._recv_lock = threading.Lock()
        
        # v1.9.3: Send lock to prevent concurrent send_tensor_dict calls from
        # interleaving at the UCX transport level. This ensures that header + all tensors
        # for a single batch are sent atomically before the next batch starts.
        self._send_lock = threading.Lock()
        
        # v2.0: Event subscription for route updates (churn resilience)
        self._event_socket: zmq.Socket | None = None
        self._event_stop_event = threading.Event()
        self._event_thread: threading.Thread | None = None
        
        # v1.8.7: GPU Direct RDMA enabled by default for better performance
        # When enabled, CUDA tensors are sent directly without CPU staging
        # This applies to both forward (0→1→2→3) and return (3→0) paths
        self._gpu_direct_enabled = os.environ.get("CONNEX_GPU_DIRECT", "1") == "1"
        if self._gpu_direct_enabled:
            logger.info("GPU Direct RDMA enabled for stage %d", stage_idx)
        
        # v1.9.4: Async send mode - fire-and-forget for lower send latency
        # When enabled, send_batch_gpu_direct returns immediately without waiting
        # for the UCX send to complete. Errors are tracked and logged.
        # v1.9.5: Disabled by default - async mode causes delivery latency regression
        self._async_send_enabled = os.environ.get("CONNEX_ASYNC_SEND", "0") == "1"
        if self._async_send_enabled:
            logger.info("Async send mode enabled for stage %d", stage_idx)
        
        # Initialize EPC if available
        if CONNEX_AVAILABLE:
            self._init_epc_components(local_ip)
        
        # Setup ZeroMQ (control plane + data plane fallback)
        discovery_start = time.time()
        self._setup_control_plane()
        discovery_time = time.time() - discovery_start
        
        self._setup_router_socket()
        
        # Start heartbeat thread to keep registration alive
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"connex-heartbeat-{self.stage_idx}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        
        # v2.0: Start event subscription thread for churn resilience
        self._start_event_subscription()

        # Leave the pipeline cleanly when the interpreter shuts down (SIGTERM).
        # This hook runs before non-daemon threads are joined; without it the
        # process lingers half-alive, still registered and still receiving work.
        self._shutting_down = False
        try:
            threading._register_atexit(self._leave_on_shutdown)
        except Exception as e:  # private API; absent on very old Pythons
            logger.debug("Stage %d: cannot register shutdown hook: %s", self.stage_idx, e)
        
        # Record connection ready time
        connection_ready_time = time.time() - init_start_time
        
        if METRICS_ENABLED:
            vllm_pipeline_connection_ready_time.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).observe(connection_ready_time)
            vllm_pipeline_connection_discovery_time.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).observe(discovery_time)
            vllm_pipeline_connection_ready_timestamp.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).set(time.time())
            # v1.7.5.3: Record RDMA setup time
            if hasattr(self, '_rdma_setup_time'):
                vllm_pipeline_connection_rdma_setup_time.labels(
                    backend=self.backend_name, stage_idx=str(self.stage_idx)
                ).observe(self._rdma_setup_time)
        
        logger.info(
            "Connex communicator initialized for stage %d/%d "
            "(connection version %d, ready in %.3fs, EPC=%s)",
            stage_idx, num_stages, self._connection_version,
            connection_ready_time, CONNEX_AVAILABLE
        )

    def _get_net1_ip(self) -> str | None:
        """Get net1 interface IP if available (for RDMA)."""
        try:
            import subprocess
            result = subprocess.run(
                ["ip", "-4", "addr", "show", "net1"],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'inet ' in line:
                    ip = line.strip().split()[1].split('/')[0]
                    logger.info("Using net1 IP for Connex: %s", ip)
                    return ip
        except Exception:
            pass
        return None

    def _init_epc_components(self, local_ip: str) -> None:
        """Initialize EPC components including UCX transport.
        
        v1.8.22: Removed unused components (RouteTable, StreamManager, 
        CutoverFSM, QoSScheduler) for performance. Only UCX transport 
        is actually used in the hot path.
        """
        try:
            # Initialize UCX transport for high-performance data transfer
            # v1.7.5.3: Track RDMA/UCX setup time
            rdma_setup_start = time.time()
            ucx_port = self.router_port + 1000  # UCX listens on different port
            transport_kwargs = {}
            if "port_fallback" in UCXTransport.__init__.__code__.co_varnames:
                # A restarted replica's old port may sit in TIME_WAIT for ~60 s;
                # bind a nearby port instead and advertise that one.
                transport_kwargs["port_fallback"] = int(os.environ.get("CONNEX_UCX_PORT_FALLBACK", "16"))
            self._ucx_transport = UCXTransport(
                local_address=local_ip,
                local_port=ucx_port,
                mode=TransportMode.UCX_RDMA,  # Try RDMA first
                **transport_kwargs,
            )
            self._ucx_port = ucx_port
            self._ucx_local_ip = local_ip
            self._ucx_endpoints: dict[str, Any] = {}  # peer_endpoint -> UCX endpoint
            self._ucx_endpoint_versions: dict[str, int] = {}  # v1.8.4: Track endpoint versions
            self._ucx_available = True
            
            # Start UCX transport with background event loop
            # This keeps the UCX listener alive in a separate thread
            self._ucx_transport.start_sync()
            self._ucx_port = self._ucx_transport.local_port  # may differ after fallback
            if hasattr(self._ucx_transport, "peer_closed_callbacks"):
                self._ucx_transport.peer_closed_callbacks.append(self._on_ucx_peer_closed)
            rdma_setup_time = time.time() - rdma_setup_start
            self._rdma_setup_time = rdma_setup_time  # Store for metrics in __init__

            # Log + export UCX config for validating RDMA vs TCP fallback.
            ucx_cfg = self._ucx_transport.get_ucx_config()
            logger.info(
                "Connex UCX config: mode=%s tls=%s net_devices=%s sockaddr_tls_priority=%s",
                ucx_cfg.get("mode"),
                ucx_cfg.get("tls"),
                ucx_cfg.get("net_devices"),
                ucx_cfg.get("sockaddr_tls_priority"),
            )
            if METRICS_ENABLED:
                vllm_pipeline_ucx_transport_info.labels(
                    backend=self.backend_name,
                    stage_idx=str(self.stage_idx),
                    mode=str(ucx_cfg.get("mode") or "unknown"),
                    tls=str(ucx_cfg.get("tls") or "unknown"),
                    net_devices=str(ucx_cfg.get("net_devices") or "unknown"),
                ).set(1)
            
            logger.info(
                "EPC components initialized for stage %d (UCX on %s:%d, mode=%s)",
                self.stage_idx, local_ip, ucx_port, self._ucx_transport.mode.name
            )
            
        except Exception as e:
            if type(e).__name__ == "UCXListenerError":
                # Our UCX port is taken. Joining anyway would advertise an
                # address nobody accepts on and stall every upstream peer.
                raise
            # Loud on purpose: losing UCX is a large performance downgrade.
            logger.error("Failed to initialize UCX transport: %s; DOWNGRADING stage %d to ZeroMQ",
                         e, self.stage_idx)
            self._ucx_available = False
            self._ucx_transport = None

    @property
    def backend_name(self) -> str:
        """Return 'connex' for metrics labeling."""
        return "connex"

    def _setup_control_plane(self) -> None:
        """Setup control plane connection to front router.
        
        v1.9.0: Use JSON-RPC format with REQ socket to connect to
        front-router's REP socket on port 15555.
        """
        host, port = split_host_port(self.front_router_addr)
        # Use port 15555 for JSON-RPC API (REP socket on front-router)
        api_port = 15555
        endpoint = get_tcp_uri(host, api_port)
        self.control_socket = self.context.socket(zmq.REQ)
        if self.timeout_ms > 0:
            self.control_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.control_socket.setsockopt(zmq.SNDTIMEO, 30000)  # 30s send timeout
        self.control_socket.setsockopt(zmq.LINGER, 1000)
        self.control_socket.connect(endpoint)
        logger.info(
            "Stage %d connected to Front ROUTER JSON-RPC API at %s",
            self.stage_idx, endpoint
        )

    def _setup_router_socket(self) -> None:
        """Setup ROUTER socket for receiving data."""
        import json
        
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 1000)
        if self.timeout_ms > 0:
            self.router_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.router_socket.bind(self.router_endpoint)
        
        logger.info(
            "Stage %d bound ROUTER socket on %s (data-plane forward/return)",
            self.stage_idx, self.router_endpoint
        )
        
        if METRICS_ENABLED:
            vllm_pipeline_comm_connection_status.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                socket_type="router",
            ).set(1)
        
        # Register with front router using ZeroMQ protocol
        self._register_stage()
    
    def _register_stage(self) -> None:
        """Register this stage with front router.
        
        v1.7.5: Report transport capabilities including GPU Direct support.
        v1.9.0: Use JSON-RPC format with method/params.
        """
        import json
        
        if self.control_socket is None:
            return
        
        # Build capability list based on available features
        caps = [0]  # TransportCapability.ZEROMQ = 0
        
        if self._ucx_available:
            caps.append(1)  # TransportCapability.UCX_TCP = 1
            # Check if RDMA is available
            ucx_tls = os.environ.get("UCX_TLS", "")
            if "rc" in ucx_tls or "dc" in ucx_tls or "ud" in ucx_tls:
                caps.append(2)  # TransportCapability.UCX_RDMA = 2
            # Check if GPU Direct is available
            if self._gpu_direct_enabled:
                if "cuda_copy" in ucx_tls or "cuda_ipc" in ucx_tls:
                    caps.append(3)  # TransportCapability.GPU_DIRECT = 3
                    logger.info("Stage %d: GPU Direct capability reported", self.stage_idx)
        
        # Get model name from environment
        model_name = os.environ.get("SERVED_MODEL_NAME", "default")
        
        # Build LogicalEndpointID structure
        lid = {
            "model_name": model_name,
            "stage_idx": self.stage_idx,
            "shard_idx": 0,
            "replica_idx": self.replica_idx,
        }
        
        # Build addresses list
        addresses = [self.router_endpoint]
        if self._ucx_available and hasattr(self, '_ucx_port'):
            ucx_endpoint = f"{self._ucx_local_ip}:{self._ucx_port}"
            addresses.append(ucx_endpoint)
        
        # JSON-RPC request format
        request = {
            "method": "register",
            "params": {
                "lid": lid,
                "addresses": addresses,
                "capabilities": caps,
            }
        }
        
        self._registration_request = request  # re-sent by heartbeat if evicted
        self.control_socket.send_json(request)
        
        try:
            reply = self.control_socket.recv_json()
            if reply.get("success"):
                server_epoch = reply.get("epoch", self._epoch)
                is_reconnect = reply.get("is_reconnect", False)
                route_table = reply.get("route_table", [])
                logger.info(
                    "Stage %d registered with Front ROUTER (epoch=%d, reconnect=%s, routes=%d, caps=%s)",
                    self.stage_idx, server_epoch, is_reconnect, len(route_table), caps
                )
                # Update local epoch
                self._epoch = server_epoch
            else:
                error = reply.get("error", "unknown error")
                logger.warning("Stage %d registration failed: %s", self.stage_idx, error)
        except Exception as exc:
            logger.warning("Stage %d registration ack failed: %s", self.stage_idx, exc)

    def _heartbeat_loop(self) -> None:
        """Background thread to send periodic heartbeats to FrontRouter.
        
        This keeps the stage registration alive. FrontRouter has a TTL-based
        cleanup mechanism that removes stale endpoints if no heartbeat is
        received within the timeout period (default 12s).
        
        v1.9.0: Use JSON-RPC format with REQ socket.
        """
        import json
        
        # Create dedicated REQ socket for heartbeat to avoid blocking main thread
        heartbeat_socket = self.context.socket(zmq.REQ)
        heartbeat_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout
        heartbeat_socket.setsockopt(zmq.SNDTIMEO, 2000)  # 2s send timeout
        heartbeat_socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        
        host, port = split_host_port(self.front_router_addr)
        # Use port 15555 for JSON-RPC API
        api_port = 15555
        endpoint = get_tcp_uri(host, api_port)
        heartbeat_socket.connect(endpoint)
        
        heartbeat_interval = 5.0  # Send heartbeat every 5 seconds
        logger.info("Stage %d: Heartbeat thread started (interval=%.1fs, endpoint=%s)",
                   self.stage_idx, heartbeat_interval, endpoint)
        
        # Get model name from environment
        model_name = os.environ.get("SERVED_MODEL_NAME", "default")
        
        # Build LogicalEndpointID structure
        lid = {
            "model_name": model_name,
            "stage_idx": self.stage_idx,
            "shard_idx": 0,
            "replica_idx": self.replica_idx,
        }
        
        hb_count = 0
        while not self._heartbeat_stop_event.is_set():
            try:
                # JSON-RPC heartbeat request
                request = {
                    "method": "heartbeat",
                    "params": {
                        "lid": lid,
                    }
                }
                heartbeat_socket.send_json(request)
                hb_count += 1
                
                # Try to receive ack (with timeout)
                try:
                    reply = heartbeat_socket.recv_json()
                    if not reply.get("success", False) and self._registration_request:
                        # Evicted (e.g. a peer reported us failed, or the lease
                        # expired during a stall) while still alive: rejoin.
                        logger.warning("Stage %d: not registered at front router; re-registering",
                                       self.stage_idx)
                        heartbeat_socket.send_json(self._registration_request)
                        heartbeat_socket.recv_json()
                    if hb_count <= 5 or hb_count % 12 == 0:  # Log first 5 and every minute
                        success = reply.get("success", False)
                        epoch = reply.get("epoch", -1)
                        logger.info("Stage %d: Heartbeat #%d ack received (success=%s, epoch=%d)", 
                                   self.stage_idx, hb_count, success, epoch)
                except zmq.Again:
                    if hb_count <= 5 or hb_count % 12 == 0:
                        logger.info("Stage %d: Heartbeat #%d sent, no ack", 
                                   self.stage_idx, hb_count)
                    
            except zmq.ZMQError as e:
                if not self._heartbeat_stop_event.is_set():
                    logger.warning("Stage %d: Heartbeat send failed: %s", 
                                 self.stage_idx, e)
            except Exception as e:
                if not self._heartbeat_stop_event.is_set():
                    logger.warning("Stage %d: Heartbeat error: %s", 
                                 self.stage_idx, e)
            
            # Wait for next heartbeat interval or stop signal
            self._heartbeat_stop_event.wait(heartbeat_interval)
        
        # Cleanup
        try:
            heartbeat_socket.close()
        except Exception:
            pass
        logger.info("Stage %d: Heartbeat thread stopped", self.stage_idx)

    def _start_event_subscription(self) -> None:
        """Start event subscription thread for churn resilience.
        
        v2.0: Subscribe to Front Router events (ENDPOINT_DEREGISTERED, etc.)
        to invalidate cache and trigger route refresh when replicas fail.
        """
        host, port = split_host_port(self.front_router_addr)
        # Event PUB socket on front-router is on port 15560
        event_port = 15560
        event_endpoint = get_tcp_uri(host, event_port)
        
        self._event_socket = self.context.socket(zmq.SUB)
        self._event_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for poll
        self._event_socket.setsockopt(zmq.LINGER, 0)
        self._event_socket.connect(event_endpoint)
        # Subscribe to all connex events
        self._event_socket.setsockopt_string(zmq.SUBSCRIBE, "connex.")
        
        self._event_stop_event.clear()
        self._event_thread = threading.Thread(
            target=self._event_subscription_loop,
            name=f"connex-events-{self.stage_idx}",
            daemon=True,
        )
        self._event_thread.start()
        
        logger.info(
            "Stage %d: Event subscription started (endpoint=%s)",
            self.stage_idx, event_endpoint
        )

    def _event_subscription_loop(self) -> None:
        """Background thread to process events from Front Router.
        
        v2.0: Handles ENDPOINT_DEREGISTERED to invalidate cache and
        trigger route refresh for resilience during churn.
        v2.1: Added ROUTE_UPDATED handling for pod IP recovery.
        """
        import json
        
        # Event type constants (from front_router.py)
        ENDPOINT_REGISTERED = 0
        ENDPOINT_DEREGISTERED = 1
        ENDPOINT_UNHEALTHY = 2
        ROUTE_UPDATED = 9
        
        model_name = os.environ.get("SERVED_MODEL_NAME", "default")
        
        while not self._event_stop_event.is_set():
            try:
                # Poll with timeout to allow checking stop event
                if self._event_socket is None:
                    break
                    
                if self._event_socket.poll(timeout=1000):
                    topic, data = self._event_socket.recv_multipart()
                    
                    try:
                        event_dict = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    
                    event_type = event_dict.get("type", 0)
                    payload = event_dict.get("payload", {})
                    event_epoch = event_dict.get("epoch", 0)
                    
                    # Handle ENDPOINT_REGISTERED - check for address change (pod restart)
                    if event_type == ENDPOINT_REGISTERED:
                        lid_dict = payload.get("lid", {})
                        affected_model = lid_dict.get("model_name", "")
                        affected_stage = lid_dict.get("stage_idx", -1)
                        address_changed = payload.get("address_changed", False)
                        
                        if affected_model == model_name and address_changed:
                            # IP changed, need to refresh connections
                            is_upstream = affected_stage == self.stage_idx - 1
                            is_downstream = affected_stage == self.stage_idx + 1
                            
                            if is_upstream or is_downstream:
                                old_addrs = payload.get("old_addresses", [])
                                new_addrs = payload.get("addresses", [])
                                logger.warning(
                                    "Stage %d: Received ENDPOINT_REGISTERED with address change "
                                    "for stage %d: %s -> %s (epoch=%d)",
                                    self.stage_idx, affected_stage, old_addrs, new_addrs, event_epoch
                                )
                                self._handle_route_change(affected_stage, is_downstream)
                    
                    # Handle ROUTE_UPDATED - pod IP changed after recovery
                    elif event_type == ROUTE_UPDATED:
                        lid_dict = payload.get("lid", {})
                        affected_model = lid_dict.get("model_name", "")
                        affected_stage = lid_dict.get("stage_idx", -1)
                        
                        if affected_model == model_name:
                            old_addrs = payload.get("old_addresses", [])
                            new_addrs = payload.get("new_addresses", [])
                            
                            is_upstream = affected_stage == self.stage_idx - 1
                            is_downstream = affected_stage == self.stage_idx + 1
                            
                            if is_upstream or is_downstream:
                                logger.warning(
                                    "Stage %d: Received ROUTE_UPDATED for stage %d: %s -> %s (epoch=%d)",
                                    self.stage_idx, affected_stage, old_addrs, new_addrs, event_epoch
                                )
                                self._handle_route_change(affected_stage, is_downstream)
                    
                    # Handle ENDPOINT_DEREGISTERED - invalidate cache for affected stage
                    elif event_type == ENDPOINT_DEREGISTERED:
                        lid_dict = payload.get("lid", {})
                        affected_model = lid_dict.get("model_name", "")
                        affected_stage = lid_dict.get("stage_idx", -1)
                        affected_replica = lid_dict.get("replica_idx", 0)
                        
                        # Only process events for our model
                        if affected_model == model_name:
                            logger.warning(
                                "Stage %d: Received ENDPOINT_DEREGISTERED for stage %d replica %d (epoch=%d)",
                                self.stage_idx, affected_stage, affected_replica, event_epoch
                            )
                            
                            # Invalidate cache for the affected stage
                            self._invalidate_route_cache(affected_stage)
                            
                            # Bump connection version to force reconnection
                            self._connection_version += 1
                            
                            # Close existing socket to the affected stage if any
                            if affected_stage == self.stage_idx + 1 and self.forward_socket is not None:
                                try:
                                    self.forward_socket.close()
                                except Exception:
                                    pass
                                self.forward_socket = None
                                logger.info(
                                    "Stage %d: Forward socket closed, will reconnect with fallback",
                                    self.stage_idx
                                )
                            
                            # Close UCX endpoint to the affected stage
                            self._close_ucx_endpoint_for_stage(affected_stage)
                    
                    # Handle ENDPOINT_UNHEALTHY similarly
                    elif event_type == ENDPOINT_UNHEALTHY:
                        lid_dict = payload.get("lid", {})
                        affected_model = lid_dict.get("model_name", "")
                        affected_stage = lid_dict.get("stage_idx", -1)
                        
                        if affected_model == model_name:
                            logger.warning(
                                "Stage %d: Received ENDPOINT_UNHEALTHY for stage %d (epoch=%d)",
                                self.stage_idx, affected_stage, event_epoch
                            )
                            # Invalidate cache - next request will get fallback route
                            self._invalidate_route_cache(affected_stage)
                    
            except zmq.Again:
                # Timeout, continue polling
                continue
            except Exception as e:
                if not self._event_stop_event.is_set():
                    logger.warning("Stage %d: Event subscription error: %s", self.stage_idx, e)
        
        # Cleanup
        if self._event_socket:
            try:
                self._event_socket.close()
            except Exception:
                pass
            self._event_socket = None
        logger.info("Stage %d: Event subscription thread stopped", self.stage_idx)

    def _handle_route_change(self, affected_stage: int, is_downstream: bool) -> None:
        """Handle route change for a peer stage (IP changed after pod restart).
        
        Args:
            affected_stage: The stage whose IP changed
            is_downstream: True if affected stage is downstream (forward path)
        """
        # Invalidate cached routes
        self._invalidate_route_cache(affected_stage)
        
        # Bump connection version to force reconnection
        self._connection_version += 1
        
        # Close existing sockets
        if is_downstream and self.forward_socket is not None:
            try:
                self.forward_socket.close()
            except Exception:
                pass
            self.forward_socket = None
            logger.info(
                "Stage %d: Forward socket to stage %d closed due to IP change",
                self.stage_idx, affected_stage
            )
        elif not is_downstream and self.is_last and self.return_socket is not None:
            # Upstream change affects return path for last stage
            try:
                self.return_socket.close()
            except Exception:
                pass
            self.return_socket = None
            logger.info(
                "Stage %d: Return socket to stage %d closed due to IP change",
                self.stage_idx, affected_stage
            )
        
        # Close UCX endpoint to the affected stage
        self._close_ucx_endpoint_for_stage(affected_stage)

    def _invalidate_route_cache(self, stage_idx: int) -> None:
        """Invalidate cached routes for a specific stage.
        
        v2.0: Called when a stage is deregistered to force route refresh.
        """
        keys_to_remove = [
            key for key in self._endpoint_cache.keys()
            if key[0] == stage_idx
        ]
        
        for key in keys_to_remove:
            self._endpoint_cache.pop(key, None)
            self._endpoint_last_access.pop(key, None)
            self._endpoint_epoch.pop(key, None)
        
        if keys_to_remove:
            logger.info(
                "Stage %d: Invalidated %d cached routes for stage %d",
                self.stage_idx, len(keys_to_remove), stage_idx
            )

        if self.is_first and stage_idx > self.stage_idx and self._waiting_recv:
            logger.warning(
                "Stage %d: downstream stage %d lost while waiting for a return; "
                "abandoning this recv so the step is retried on a new route",
                self.stage_idx, stage_idx,
            )
            self._peer_lost.set()

    def _on_ucx_peer_closed(self, peer: str) -> None:
        """Transport callback: an accepted UCX connection's receive loop ended."""
        if self.is_first and self._waiting_recv and peer == self._last_data_peer:
            logger.warning(
                "Stage %d: UCX connection %s that delivered our last return closed; "
                "treating downstream as lost", self.stage_idx, peer,
            )
            self._peer_lost.set()
            self._report_peer_failure(self.stage_idx + 1)

    def _leave_on_shutdown(self) -> None:
        """Deregister this stage from the front router at interpreter shutdown."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._registration_request = None  # never re-register from here on
        self._heartbeat_stop_event.set()
        lid = {
            "model_name": os.environ.get("SERVED_MODEL_NAME", "default"),
            "stage_idx": self.stage_idx,
            "shard_idx": 0,
            "replica_idx": self.replica_idx,
        }
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            host, _ = split_host_port(self.front_router_addr)
            sock.connect(get_tcp_uri(host, 15555))
            sock.send_json({"method": "deregister", "params": {"lid": lid}})
            logger.warning("Stage %d: shutting down, deregistered: %s",
                           self.stage_idx, sock.recv_json())
        except Exception as e:
            logger.warning("Stage %d: deregister on shutdown failed: %s", self.stage_idx, e)
        finally:
            sock.close()

    def _report_peer_failure(self, stage_idx: int) -> None:
        """Deregister the downstream replica we were assigned, so the front
        router stops routing to it now instead of after its heartbeat timeout.
        A replica that is actually alive re-registers on its next heartbeat."""
        lid = self._assigned_lids.get(stage_idx)
        if not lid:
            return

        def run():
            sock = self.context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.setsockopt(zmq.SNDTIMEO, 2000)
            try:
                host, _ = split_host_port(self.front_router_addr)
                sock.connect(get_tcp_uri(host, 15555))
                sock.send_json({"method": "deregister", "params": {"lid": lid}})
                reply = sock.recv_json()
                logger.warning("Stage %d: reported failed peer %s to front router: %s",
                               self.stage_idx, lid, reply)
            except Exception as e:
                logger.warning("Stage %d: failed to report peer failure: %s", self.stage_idx, e)
            finally:
                sock.close()

        threading.Thread(target=run, daemon=True, name="connex-report-failure").start()

    def _close_ucx_endpoint_for_stage(self, stage_idx: int) -> None:
        """Close UCX endpoints for a specific stage.
        
        v2.0: Called when a stage is deregistered to clean up connections.
        """
        if not self._ucx_available or not self._ucx_transport:
            return
        
        # Find and close UCX endpoints that were connected to the stage
        endpoints_to_remove = []
        for endpoint_key in list(self._ucx_endpoints.keys()):
            # UCX endpoint keys are typically "host:port"
            # We need to check if this corresponds to the affected stage
            # For now, we'll clear all UCX endpoints and let them be re-established
            # on next send (this is conservative but safe)
            endpoints_to_remove.append(endpoint_key)
        
        for endpoint_key in endpoints_to_remove:
            ep = self._ucx_endpoints.pop(endpoint_key, None)
            if ep:
                try:
                    self._ucx_transport.close_endpoint(ep)
                except Exception:
                    pass
        
        if endpoints_to_remove:
            logger.info(
                "Stage %d: Closed %d UCX endpoints for stage %d",
                self.stage_idx, len(endpoints_to_remove), stage_idx
            )

    def _request_assignment(
        self, target_stage_idx: int, direction: str
    ) -> tuple[str, str, str | None]:
        """Request routing assignment with epoch-based caching.
        
        v1.8.3: Returns (endpoint, routing_key, ucx_endpoint).
        v1.9.0: Use JSON-RPC format with get_route method.
        """
        import json
        
        cache_key = (target_stage_idx, direction)
        
        # Check cache first
        if cache_key in self._endpoint_cache:
            cached = self._endpoint_cache[cache_key]
            # v1.8.3: Handle both old (3-tuple) and new (4-tuple) cache format
            if len(cached) >= 4:
                cached_endpoint, cached_routing_key, cached_version, cached_ucx_endpoint = cached
            else:
                cached_endpoint, cached_routing_key, cached_version = cached
                cached_ucx_endpoint = None
            if cached_version == self._connection_version:
                # v1.7.1: Update last access time for TTL tracking
                self._endpoint_last_access[cache_key] = time.time()
                return cached_endpoint, cached_routing_key, cached_ucx_endpoint
        
        if self.control_socket is None:
            raise RuntimeError("Control socket not initialized")
        
        # v1.7.1: Periodically cleanup stale endpoints
        self._cleanup_stale_endpoints()
        
        # Get model name from environment
        model_name = os.environ.get("SERVED_MODEL_NAME", "default")
        
        # Build target LogicalEndpointID
        # Use same replica_idx by default - route is fixed until worker detects
        # failure or high latency and invalidates cache (then re-request triggers
        # front-router fallback to another replica).
        target_lid = {
            "model_name": model_name,
            "stage_idx": target_stage_idx,
            "shard_idx": 0,
            "replica_idx": self.replica_idx,
        }
        
        # Request from front router using JSON-RPC with retry
        # v1.7.5.6: Add retry logic for no-target errors (target stage not ready yet)
        # v1.8.7: Reduced retry delay from 1.0s to 0.2s for faster topology convergence
        max_retries = 150  # Up to 30 seconds of retries (150 * 0.2s)
        retry_delay = 0.2  # 0.2 second between retries (faster convergence)
        
        for attempt in range(max_retries):
            # JSON-RPC request for get_route
            request = {
                "method": "get_route",
                "params": {
                    "lid": target_lid,
                }
            }
            if attempt == 0:
                logger.info(
                    "Stage %d: Sending get_route request to Front ROUTER for stage %d (%s)",
                    self.stage_idx, target_stage_idx, direction
                )
            self.control_socket.send_json(request)
            
            reply = self.control_socket.recv_json()
            
            # Check for error response (e.g., target not ready)
            if "error" in reply:
                error_msg = reply["error"]
                if "not found" in error_msg.lower() and attempt < max_retries - 1:
                    logger.warning(
                        "Stage %d: Target stage %d not ready yet (attempt %d/%d), retrying in %.1fs...",
                        self.stage_idx, target_stage_idx, attempt + 1, max_retries, retry_delay
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"get_route failed: {error_msg} (after {attempt + 1} attempts)")
            
            if attempt > 0:
                logger.info("Stage %d: get_route succeeded after %d retries", self.stage_idx, attempt)
            logger.info("Stage %d: Received get_route reply: %s", self.stage_idx, reply)
            
            # v2.0: Check if this is a fallback route (original replica not available)
            is_fallback = reply.get("is_fallback", False)
            if is_fallback:
                original_replica = reply.get("original_replica", self.replica_idx)
                fallback_replica = reply.get("fallback_replica", -1)
                logger.warning(
                    "Stage %d: Using FALLBACK route for stage %d (replica %d -> %d)",
                    self.stage_idx, target_stage_idx, original_replica, fallback_replica
                )
            
            # Parse route from reply
            route = reply.get("route")
            if not route:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Stage %d: No route returned for stage %d (attempt %d/%d), retrying...",
                        self.stage_idx, target_stage_idx, attempt + 1, max_retries
                    )
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError(f"No route found for stage {target_stage_idx}")
            
            # Extract endpoint from route addresses
            addresses = route.get("addresses", [])
            if not addresses:
                raise RuntimeError(f"Route for stage {target_stage_idx} has no addresses")
            
            endpoint = addresses[0]  # Primary endpoint (ZeroMQ)
            self._assigned_lids[target_stage_idx] = route.get("lid")
            routing_key = f"rk-{target_stage_idx}-{uuid4().hex[:8]}"
            
            # Check for UCX endpoint (second address if available)
            ucx_endpoint = addresses[1] if len(addresses) > 1 else None
            break  # Success
        else:
            raise RuntimeError(f"get_route failed after {max_retries} retries: target stage {target_stage_idx} not ready")
        
        # Cache with version
        self._endpoint_cache[cache_key] = (
            endpoint, routing_key, self._connection_version, ucx_endpoint  # v1.8.3: Include UCX endpoint
        )
        # v1.7.1: Track creation time and epoch for TTL management
        self._endpoint_last_access[cache_key] = time.time()
        self._endpoint_epoch[cache_key] = self._epoch
        
        logger.info(
            "Stage %d assignment -> stage %d (%s): endpoint=%s routing_key=%s ucx=%s (epoch=%d)",
            self.stage_idx, target_stage_idx, direction, endpoint, routing_key, ucx_endpoint, self._epoch
        )
        
        return endpoint, routing_key, ucx_endpoint

    def _connect_dealer(self, endpoint: str, routing_key: str) -> zmq.Socket:
        """Create DEALER socket connected to endpoint."""
        socket = self.context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 1000)
        socket.setsockopt_string(zmq.IDENTITY, routing_key)
        socket.connect(endpoint)
        return socket

    def _get_forward_socket(self) -> zmq.Socket:
        """Get forward socket with version-aware reconnection."""
        if self.is_last:
            raise RuntimeError("Last stage has no forward socket")
        
        # Check if reconnection needed due to version change
        cache_key = (self.stage_idx + 1, "forward")
        need_reconnect = False
        
        if self.forward_socket is None:
            need_reconnect = True
        elif cache_key in self._endpoint_cache:
            cached = self._endpoint_cache[cache_key]
            cached_version = cached[2]  # v1.8.4: Cache now has 4 elements
            if cached_version < self._connection_version:
                need_reconnect = True
                try:
                    self.forward_socket.close()
                except Exception:
                    pass
                self.forward_socket = None
        
        if need_reconnect:
            endpoint, routing_key, _ = self._request_assignment(
                self.stage_idx + 1, "forward"
            )
            self.forward_socket = self._connect_dealer(endpoint, routing_key)
            
            if METRICS_ENABLED:
                vllm_pipeline_comm_connection_status.labels(
                    backend=self.backend_name,
                    stage_idx=str(self.stage_idx),
                    socket_type="forward",
                ).set(1)
            
            logger.debug(
                "Stage %d created forward socket (version=%d)",
                self.stage_idx, self._connection_version
            )
        
        return self.forward_socket

    def _get_return_socket(self) -> zmq.Socket:
        """Get return socket with version-aware reconnection."""
        if not self.is_last:
            raise RuntimeError("Return socket only on last stage")
        
        cache_key = (0, "return")
        need_reconnect = False
        
        if self.return_socket is None:
            need_reconnect = True
        elif cache_key in self._endpoint_cache:
            cached = self._endpoint_cache[cache_key]
            cached_version = cached[2]  # v1.8.4: Cache now has 4 elements
            if cached_version < self._connection_version:
                need_reconnect = True
                try:
                    self.return_socket.close()
                except Exception:
                    pass
                self.return_socket = None
        
        if need_reconnect:
            endpoint, routing_key, _ = self._request_assignment(0, "return")
            self.return_socket = self._connect_dealer(endpoint, routing_key)
            
            if METRICS_ENABLED:
                vllm_pipeline_comm_connection_status.labels(
                    backend=self.backend_name,
                    stage_idx=str(self.stage_idx),
                    socket_type="return",
                ).set(1)
            
            logger.debug(
                "Stage %d created return socket (version=%d)",
                self.stage_idx, self._connection_version
            )
        
        return self.return_socket

    def invalidate_endpoint_cache(self, target_stage_idx: int | None = None) -> None:
        """Invalidate endpoint cache for topology changes."""
        if target_stage_idx is None:
            self._endpoint_cache.clear()
            # v1.7.1: Also clear TTL tracking
            self._endpoint_last_access.clear()
            self._endpoint_epoch.clear()
            self._connection_version += 1
            self._epoch += 1
            logger.info(
                "Stage %d invalidated all cache (version=%d, epoch=%d)",
                self.stage_idx, self._connection_version, self._epoch
            )
        else:
            keys_to_remove = [
                key for key in self._endpoint_cache.keys()
                if key[0] == target_stage_idx
            ]
            for key in keys_to_remove:
                del self._endpoint_cache[key]
                # v1.7.1: Also remove from TTL tracking
                self._endpoint_last_access.pop(key, None)
                self._endpoint_epoch.pop(key, None)
            if keys_to_remove:
                self._connection_version += 1
                logger.info(
                    "Stage %d invalidated cache for stage %d (version=%d)",
                    self.stage_idx, target_stage_idx, self._connection_version
                )
    
    def _invalidate_ucx_endpoints(self, target_stage_idx: int | None = None) -> None:
        """v1.8.4: Invalidate UCX endpoint cache for topology changes."""
        if target_stage_idx is None:
            # Clear all UCX endpoints
            old_count = len(self._ucx_endpoints)
            self._ucx_endpoints.clear()
            self._ucx_endpoint_versions.clear()
            if old_count > 0:
                logger.info(
                    "Stage %d: Invalidated all UCX endpoint cache (%d endpoints)",
                    self.stage_idx, old_count
                )
        else:
            # Clear UCX endpoints for specific target stage
            keys_to_remove = [
                key for key in self._ucx_endpoints.keys()
                if key.startswith(f"{target_stage_idx}_")
            ]
            for key in keys_to_remove:
                self._ucx_endpoints.pop(key, None)
                self._ucx_endpoint_versions.pop(key, None)
            if keys_to_remove:
                logger.info(
                    "Stage %d: Invalidated UCX endpoint cache for stage %d (%d endpoints)",
                    self.stage_idx, target_stage_idx, len(keys_to_remove)
                )
    
    def _cleanup_stale_endpoints(self) -> int:
        """v1.7.1: Clean up stale endpoints based on TTL and epoch.
        
        Returns number of endpoints cleaned up.
        """
        current_time = time.time()
        current_epoch = self._epoch
        cleaned = 0
        
        keys_to_remove = []
        for key in list(self._endpoint_cache.keys()):
            last_access = self._endpoint_last_access.get(key, 0)
            endpoint_epoch = self._endpoint_epoch.get(key, 0)
            
            # Remove if TTL expired OR if endpoint was created in old epoch
            age = current_time - last_access
            epoch_stale = endpoint_epoch < current_epoch
            
            if age > self._endpoint_ttl_seconds or epoch_stale:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._endpoint_cache.pop(key, None)
            self._endpoint_last_access.pop(key, None)
            self._endpoint_epoch.pop(key, None)
            cleaned += 1
            logger.debug(
                "Stage %d: Cleaned stale endpoint %s (epoch=%d, age=%.1fs)",
                self.stage_idx, key, self._endpoint_epoch.get(key, -1),
                current_time - self._endpoint_last_access.get(key, 0)
            )
        
        if cleaned > 0:
            logger.info(
                "Stage %d: Cleaned %d stale endpoints (current epoch=%d)",
                self.stage_idx, cleaned, current_epoch
            )
        
        return cleaned

    def _get_ucx_endpoint(self, target_stage: int, direction: str, retry_on_failure: bool = True) -> Any:
        """Get or create UCX endpoint for target stage.
        
        v1.8.4: Added version tracking and automatic retry on connection failure.
        """
        cache_key = f"{target_stage}_{direction}"
        if not hasattr(self, "_no_ucx_peers"):
            # cache_key -> (connection_version, time): peers seen without a
            # UCX address. Re-checked after 5 s or a version bump, so a
            # replacement replica with UCX is picked up.
            self._no_ucx_peers: dict[str, tuple[int, float]] = {}
        no_ucx = self._no_ucx_peers.get(cache_key)
        if (no_ucx is not None and no_ucx[0] == self._connection_version
                and time.time() - no_ucx[1] < 5.0):
            return None  # peer has no UCX address; ZeroMQ path

        # Check if cached endpoint is still valid (same connection version)
        if cache_key in self._ucx_endpoints:
            cached_version = self._ucx_endpoint_versions.get(cache_key, 0)
            if cached_version >= self._connection_version:
                return self._ucx_endpoints[cache_key]
            else:
                # Version changed, invalidate cached endpoint
                logger.info(
                    "Stage %d: UCX endpoint cache stale for stage %d (cached_v=%d, current_v=%d), reconnecting",
                    self.stage_idx, target_stage, cached_version, self._connection_version
                )
                old_ep = self._ucx_endpoints.pop(cache_key, None)
                self._ucx_endpoint_versions.pop(cache_key, None)
                # Try to close old endpoint gracefully
                if old_ep is not None:
                    try:
                        self._ucx_transport.close_endpoint(old_ep)
                    except Exception:
                        pass
        
        # Get peer's UCX address from front router (forces fresh lookup)
        endpoint, _, ucx_endpoint_str = self._request_assignment(target_stage, direction)

        # v1.8.3: Use UCX endpoint from ASSIGN reply. A peer that advertises
        # no UCX address gets the ZeroMQ path (None), never a guessed port.
        resolved = resolve_peer_ucx_address(ucx_endpoint_str)
        if resolved is None:
            if no_ucx is None:
                logger.warning(
                    "Stage %d: stage %d (%s) advertises no UCX address; using ZeroMQ for it",
                    self.stage_idx, target_stage, endpoint,
                )
            self._no_ucx_peers[cache_key] = (self._connection_version, time.time())
            return None
        self._no_ucx_peers.pop(cache_key, None)
        peer_ip, peer_ucx_port = resolved

        logger.info(
            "Stage %d: Creating UCX endpoint to stage %d at %s:%d",
            self.stage_idx, target_stage, peer_ip, peer_ucx_port
        )
        
        # Connect via UCX using synchronous method
        try:
            ucx_ep = self._ucx_transport.connect_sync(peer_ip, peer_ucx_port)
            
            if ucx_ep:
                self._ucx_endpoints[cache_key] = ucx_ep
                self._ucx_endpoint_versions[cache_key] = self._connection_version  # v1.8.4: Track version
                logger.info(
                    "Stage %d: UCX connection established to stage %d (%s:%d, version=%d)",
                    self.stage_idx, target_stage, peer_ip, peer_ucx_port, self._connection_version
                )
            else:
                logger.warning(
                    "Stage %d: UCX connect returned None for stage %d (%s:%d)",
                    self.stage_idx, target_stage, peer_ip, peer_ucx_port
                )
            
            return ucx_ep
        except Exception as e:
            logger.error(
                "Stage %d: UCX connect failed to stage %d (%s:%d): %s",
                self.stage_idx, target_stage, peer_ip, peer_ucx_port, e
            )
            # v1.8.4: On connection failure, invalidate endpoint cache and retry once
            if retry_on_failure:
                logger.info(
                    "Stage %d: Invalidating endpoint cache and retrying UCX connection to stage %d",
                    self.stage_idx, target_stage
                )
                # Invalidate the specific endpoint cache entry
                assign_cache_key = (target_stage, direction)
                self._endpoint_cache.pop(assign_cache_key, None)
                self._endpoint_last_access.pop(assign_cache_key, None)
                self._endpoint_epoch.pop(assign_cache_key, None)
                self._connection_version += 1
                # Retry without further retries to avoid infinite loop
                return self._get_ucx_endpoint(target_stage, direction, retry_on_failure=False)
            raise

    def handle_topology_change(
        self,
        changed_stage_idx: int | None = None,
        new_num_stages: int | None = None,
    ) -> None:
        """Handle topology change for endpoint mobility."""
        if new_num_stages is not None:
            old_stages = self.num_stages
            self.num_stages = new_num_stages
            self.is_last = self.stage_idx == self.num_stages - 1
            logger.info(
                "Stage %d: Topology change %d -> %d stages",
                self.stage_idx, old_stages, new_num_stages
            )
        
        self.invalidate_endpoint_cache(changed_stage_idx)
        
        # v1.8.4: Also invalidate UCX endpoint cache
        self._invalidate_ucx_endpoints(changed_stage_idx)
        
        # Close affected sockets
        if changed_stage_idx is None or changed_stage_idx == self.stage_idx + 1:
            if self.forward_socket:
                try:
                    self.forward_socket.close()
                except Exception:
                    pass
                self.forward_socket = None
        
        if changed_stage_idx is None or changed_stage_idx == 0:
            if self.return_socket:
                try:
                    self.return_socket.close()
                except Exception:
                    pass
                self.return_socket = None

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        """Send tensor dictionary with Pull model + true async (like Mooncake).
        
        v1.8.0 Optimizations:
        - Pull model: sender stores data in UCX buffer, sends metadata via ZeroMQ
        - Receiver initiates UCX read (pull) to fetch data
        - True async: send() returns immediately after metadata sent
        - Compact header: 17 bytes (vs 41 bytes before)
        - Buffer cleanup in background thread
        
        v1.9.3: Added send lock to prevent concurrent send operations from
        interleaving at the UCX transport level.
        
        Header format (17 bytes):
        [flags(1)][tensor_len(4)][meta_len(4)][timestamp(8)]
        """
        # v1.9.3: Acquire send lock to prevent interleaving of concurrent sends
        # This ensures header + all tensors are sent atomically
        with self._send_lock:
            try:
                self._send_tensor_dict_impl(tensor_dict, dst)
            except Exception:
                # On send failure, invalidate route cache so next request re-asks
                # front-router (and gets fallback to another replica if needed)
                target_stage = 0 if self.is_last else self.stage_idx + 1
                self._invalidate_route_cache(target_stage)
                raise
    
    def _send_tensor_dict_impl(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        """Internal implementation of send_tensor_dict (called with lock held)."""
        send_start = time.time()

        if self.is_last:
            direction = "return"
            target_stage = 0
        else:
            direction = "forward"
            target_stage = self.stage_idx + 1

        if METRICS_ENABLED:
            vllm_pipeline_comm_tensor_count.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(len([v for v in tensor_dict.values() if isinstance(v, torch.Tensor)]))

        # v1.7.4.8: GPU Direct RDMA path - send CUDA tensors directly
        if (self._gpu_direct_enabled and self._ucx_available
                and self._get_ucx_endpoint(target_stage, direction) is not None):
            self._send_tensor_dict_gpu_direct(
                tensor_dict, direction, send_start, target_stage
            )
            return
        
        # Standard path: CPU transfer + safetensors serialization
        tensors = {}
        non_tensors = {}
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                if v.device.type == "cuda":
                    tensors[k] = v.cpu()
                else:
                    tensors[k] = v
            else:
                non_tensors[k] = v

        # Serialize tensors with safetensors (fast) or torch.save (fallback)
        if SAFETENSORS_AVAILABLE and tensors:
            tensor_bytes = safetensors_save(tensors)
            use_safetensors = True
        elif tensors:
            buffer = io.BytesIO()
            torch.save(tensors, buffer)
            tensor_bytes = buffer.getvalue()
            use_safetensors = False
        else:
            tensor_bytes = b""
            use_safetensors = SAFETENSORS_AVAILABLE

        # Serialize non-tensors with msgpack (faster than pickle)
        non_tensor_bytes = _serialize_metadata(non_tensors)
        
        total_tensor_bytes = len(tensor_bytes)
        
        if METRICS_ENABLED and total_tensor_bytes > 0:
            vllm_pipeline_comm_tensor_size.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(total_tensor_bytes)

        serialization_time = time.time() - send_start
        if METRICS_ENABLED:
            vllm_pipeline_comm_serialization_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).observe(serialization_time)

        send_timestamp_value = time.time()
        
        # v1.8.0: Compact header format (17 bytes)
        # [flags(1)][tensor_len(4)][meta_len(4)][timestamp(8)]
        # flags: bit0 = use_safetensors, bit1 = use_msgpack
        flags = (1 if use_safetensors else 0) | (2 if MSGPACK_AVAILABLE else 0)
        
        header_size = 17  # 1+4+4+8
        total_size = header_size + len(tensor_bytes) + len(non_tensor_bytes)
        
        # Allocate once, fill with memoryview for zero-copy
        combined_data = bytearray(total_size)
        
        # Pack compact header directly into bytearray
        struct.pack_into(
            "!BIId",  # flags, tensor_len, meta_len, timestamp
            combined_data,
            0,
            flags,
            len(tensor_bytes),
            len(non_tensor_bytes),
            send_timestamp_value
        )
        
        # Copy payloads directly into bytearray
        offset = header_size
        combined_data[offset:offset + len(tensor_bytes)] = tensor_bytes
        offset += len(tensor_bytes)
        combined_data[offset:offset + len(non_tensor_bytes)] = non_tensor_bytes
        
        # v1.8.0: Log header info for debugging
        logger.debug("Stage %d: Send header - flags=0x%02x, tensor_len=%d, meta_len=%d, total=%d",
                    self.stage_idx, flags, len(tensor_bytes), len(non_tensor_bytes), total_size)
        
        # Get the ZeroMQ socket
        socket = self._get_return_socket() if self.is_last else self._get_forward_socket()
        
        if self.is_first:
            self._corr_seq += 1
            corr = self._corr_base + self._corr_seq
            self._awaiting_corr = corr
        else:
            corr = self._corr_in or 0
        corr_frame = struct.pack("!Q", corr)

        # Decide transport: Pull model via UCX vs ZeroMQ fallback
        use_pull_model = (
            self._ucx_available 
            and self._ucx_transport is not None
            and self._pull_mode_enabled
        )
        
        # v1.8.2: Measure enqueue time only for the actual network send
        # (like Mooncake: only measures ZeroMQ send, not UCX setup)
        enqueue_start = time.time()
        
        if use_pull_model and self._get_ucx_endpoint(target_stage, direction) is None:
            use_pull_model = False  # peer has no UCX address

        if use_pull_model:
            # v1.8.0: Pull model with ZeroMQ signal + UCX data
            # 1. Send data via UCX (fire-and-forget)
            # 2. Send ZeroMQ signal to notify receiver
            # 3. Async cleanup after TTL
            try:
                msg_id = self._next_msg_id
                self._next_msg_id += 1
                
                # Store data for pull (receiver will fetch via UCX)
                self._pending_send_buffers[msg_id] = (combined_data, total_size, send_timestamp_value)
                
                # Queue data to UCX for receiver to pull (fire-and-forget)
                self._send_via_ucx_async(target_stage, direction, combined_data, msg_id)
                
                # Send ZeroMQ signal to notify receiver that UCX data is available
                # This allows receiver to use ZeroMQ signal + UCX data mode (more robust)
                socket.send_multipart([
                    b"UCX",  # Signal that data is available via UCX
                    # Send timestamp from the header: the receiver uses it to
                    # pair this signal with its UCX payload and drop stale ones.
                    struct.pack("!d", send_timestamp_value),
                    corr_frame,
                ])
                
                # Async cleanup after buffer TTL
                if self._buffer_cleaner is not None:
                    self._buffer_cleaner.submit(
                        self._cleanup_send_buffer, msg_id, self._buffer_ttl_seconds
                    )
                
            except Exception as e:
                logger.warning("Stage %d: Pull model UCX failed: %s, fallback to ZeroMQ", self.stage_idx, e)
                use_pull_model = False
        
        if not use_pull_model:
            # ZeroMQ-only mode: send everything via ZeroMQ
            socket.send_multipart([
                b"ZMQ",
                combined_data,
                corr_frame,
            ])

        enqueue_time = time.time() - enqueue_start
        total_send_time = time.time() - send_start

        if METRICS_ENABLED:
            vllm_pipeline_comm_bytes_sent.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).inc(total_size)
            vllm_pipeline_comm_messages_sent.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).inc()
            vllm_pipeline_comm_send_latency.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(total_send_time)
            vllm_pipeline_comm_enqueue_latency.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(enqueue_time)
        
        logger.debug(
            "Stage %d: send completed (%.3fs, enqueue=%.4fs, pull=%s, size=%d)",
            self.stage_idx, total_send_time, enqueue_time, use_pull_model, total_size
        )
    
    def _send_via_ucx_async(
        self,
        target_stage: int,
        direction: str,
        data: bytes | bytearray,
        msg_id: int,
    ) -> None:
        """Send data via UCX asynchronously (fire-and-forget).
        
        v1.8.0: True async send - returns immediately without waiting.
        Data is queued to UCX event loop for transmission.
        """
        ucx_ep = self._get_ucx_endpoint(target_stage, direction)
        if ucx_ep is None:
            raise RuntimeError(f"Failed to get UCX endpoint for stage {target_stage}")
        
        tag = self.stage_idx * 100 + (1 if direction == "forward" else 2)
        
        # Fire-and-forget async send
        self._ucx_transport.send_async(ucx_ep, data, tag=tag)
    
    def _cleanup_send_buffer(self, msg_id: int, delay: float) -> None:
        """Cleanup send buffer after delay (runs in background thread).
        
        v1.8.0: Like Mooncake's async buffer cleanup.
        """
        time.sleep(delay)
        self._pending_send_buffers.pop(msg_id, None)
    
    def _send_tensor_dict_gpu_direct(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        direction: str,
        send_start: float,
        target_stage: int,
    ) -> None:
        """Send tensor dictionary using GPU Direct RDMA.
        
        v1.7.4.8: Zero-copy GPU-to-GPU transfer path.
        
        Instead of serializing tensors to CPU bytes, we:
        1. Send tensor metadata (shapes, dtypes) via standard UCX
        2. Send each CUDA tensor directly via GPU Direct RDMA
        3. Send non-tensor data via standard UCX
        
        This eliminates GPU→CPU→GPU copies for tensor data.
        """
        # Separate tensors and non-tensors
        cuda_tensors = {}
        cpu_tensors = {}
        non_tensors = {}
        
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                if v.device.type == "cuda":
                    # Ensure contiguous for direct transfer
                    cuda_tensors[k] = v.contiguous() if not v.is_contiguous() else v
                else:
                    cpu_tensors[k] = v
            else:
                non_tensors[k] = v
        
        # Build tensor metadata (shapes, dtypes for reconstruction)
        tensor_meta = {}
        for k, v in cuda_tensors.items():
            tensor_meta[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": "cuda",
                "nbytes": v.numel() * v.element_size(),
            }
        for k, v in cpu_tensors.items():
            tensor_meta[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": "cpu",
                "nbytes": v.numel() * v.element_size(),
            }
        
        # EPC metadata
        stream_id = f"{direction}_{self.stage_idx}"
        seqno = self._stream_seqno.get(stream_id, 0)
        self._stream_seqno[stream_id] = seqno + 1
        
        non_tensors["__epoch__"] = self._epoch
        non_tensors["__seqno__"] = seqno
        non_tensors["__tensor_meta__"] = tensor_meta
        non_tensors["__gpu_direct__"] = True
        non_tensors["__num_cuda_tensors__"] = len(cuda_tensors)
        
        # Serialize metadata with msgpack
        meta_bytes = _serialize_metadata(non_tensors)
        
        # Serialize CPU tensors with safetensors (if any)
        if cpu_tensors and SAFETENSORS_AVAILABLE:
            cpu_tensor_bytes = safetensors_save(cpu_tensors)
        elif cpu_tensors:
            buffer = io.BytesIO()
            torch.save(cpu_tensors, buffer)
            cpu_tensor_bytes = buffer.getvalue()
        else:
            cpu_tensor_bytes = b""
        
        send_timestamp = time.time()
        
        # Build header for GPU Direct mode
        # v1.8.0: Use compact 25-byte header matching receiver
        # Header: [flags(1)][meta_len(8)][cpu_tensor_len(8)][timestamp(8)]
        # flags bit2 = GPU Direct mode
        flags = 4 | (1 if SAFETENSORS_AVAILABLE else 0) | (2 if MSGPACK_AVAILABLE else 0)
        
        header_size = 25
        header_data = struct.pack(
            "!BQQd",
            flags,
            len(meta_bytes),
            len(cpu_tensor_bytes),
            send_timestamp,
        )
        
        # First message: header + metadata + CPU tensors
        combined_data = bytearray(header_size + len(meta_bytes) + len(cpu_tensor_bytes))
        combined_data[:header_size] = header_data
        combined_data[header_size:header_size + len(meta_bytes)] = meta_bytes
        combined_data[header_size + len(meta_bytes):] = cpu_tensor_bytes
        
        # Get UCX endpoint
        ucx_ep = self._get_ucx_endpoint(target_stage, direction)
        if ucx_ep is None:
            raise RuntimeError(f"Failed to get UCX endpoint for stage {target_stage}")
        
        tag = self.stage_idx * 100 + (1 if direction == "forward" else 2)
        
        # v1.7.5.4: Use batch send to prevent interleaving race condition
        # Each tensor's metadata+data pair must be sent atomically
        # v1.7.5.8: Get separate enqueue and wait times for accurate metrics
        # v1.9.4: Use async_mode for fire-and-forget send (lower latency)
        tensor_list = [cuda_tensors[name] for name in sorted(cuda_tensors.keys())]
        ucx_enqueue_time, ucx_wait_time = self._ucx_transport.send_batch_gpu_direct(
            ucx_ep, combined_data, tensor_list, tag=tag,
            async_mode=self._async_send_enabled
        )
        
        # Send ZeroMQ signal (GPU Direct mode indicator: "G"). Same frame
        # layout as the "UCX" signal: the header's send timestamp lets the
        # receiver pair the signal with its UCX header, and the correlation
        # frame keeps stage 0's abandoned-step filtering working.
        if self.is_first:
            self._corr_seq += 1
            corr = self._corr_base + self._corr_seq
            self._awaiting_corr = corr
        else:
            corr = self._corr_in or 0
        socket = self._get_return_socket() if self.is_last else self._get_forward_socket()
        if socket:
            socket.send_multipart([b"G", struct.pack("!d", send_timestamp), struct.pack("!Q", corr)])
        
        total_bytes = len(combined_data) + sum(
            t.numel() * t.element_size() for t in cuda_tensors.values()
        )
        total_send_time = time.time() - send_start
        
        if METRICS_ENABLED:
            vllm_pipeline_comm_bytes_sent.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).inc(total_bytes)
            vllm_pipeline_comm_messages_sent.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).inc()
            vllm_pipeline_comm_send_latency.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(total_send_time)
            # v1.7.5.8: Record only the enqueue time (submission to asyncio), not the wait
            # This makes it comparable with async backends like Mooncake
            vllm_pipeline_comm_enqueue_latency.labels(
                backend=self.backend_name,
                stage_idx=str(self.stage_idx),
                direction=direction,
            ).observe(ucx_enqueue_time)
        
        logger.debug(
            "Stage %d: GPU Direct send completed (%.3fs, enqueue=%.4fs, wait=%.4fs, size=%d, cuda_tensors=%d)",
            self.stage_idx, total_send_time, ucx_enqueue_time, ucx_wait_time, total_bytes, len(cuda_tensors)
        )
    
    def _send_via_ucx_pure(
        self,
        target_stage: int,
        direction: str,
        data: bytes | bytearray,
    ) -> None:
        """Send data via pure UCX with async mode for lower latency.
        
        v1.5.7: Use async send (fire-and-forget) to reduce enqueue latency.
        The ZeroMQ signal is sent AFTER UCX data is queued, so receiver
        knows data is in-flight when signal arrives.
        
        v1.7.1: Zero-copy optimization. Data already contains unified header:
        [stage_idx(8)][epoch(8)][flags(1)][tensor_len(8)][meta_len(8)][timestamp(8)][payload]
        No additional packing needed - just send directly.
        
        v1.7.2.2: Removed bytes(data) copy - pass bytearray/memoryview directly to UCX.
        UCX-Py's send_obj accepts buffer-protocol objects, avoiding unnecessary copy.
        """
        ucx_ep = self._get_ucx_endpoint(target_stage, direction)
        if ucx_ep is None:
            raise RuntimeError(f"Failed to get UCX endpoint for stage {target_stage}")
        
        tag = self.stage_idx * 100 + (1 if direction == "forward" else 2)
        
        # v1.7.1: Data already has unified header, send directly without extra copy
        # Avoid bytes(data) copy; UCX-Py send_obj accepts buffer-protocol objects.
        self._ucx_transport.send_async(ucx_ep, data, tag=tag)

    def recv_tensor_dict(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        """Receive tensor dictionary with Pull model + direct UCX.
        
        v1.8.0 Optimizations:
        - Pull model: receiver directly reads from UCX queue (no ZMQ signal)
        - Compact header: 17 bytes (vs 41 bytes before)
        - Default to direct UCX mode for minimum latency
        
        v1.9.0: Use recv lock to prevent concurrent calls from interleaving
        
        Header format (17 bytes):
        [flags(1)][tensor_len(4)][meta_len(4)][timestamp(8)]
        """
        # v1.9.0: Acquire lock to prevent concurrent recv interleaving
        # This ensures header + all tensors are received atomically
        with self._recv_lock:
            try:
                return self._recv_tensor_dict_impl(src)
            except Exception:
                # On recv failure, invalidate route cache so next request re-asks
                # front-router (and gets fallback to another replica if needed)
                affected_stage = 0 if self.is_first else self.stage_idx + 1
                self._invalidate_route_cache(affected_stage)
                raise
    
    def _recv_tensor_dict_impl(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        """Internal implementation of recv_tensor_dict (called with lock held)."""
        recv_start = time.time()
        direction = "return" if self.is_first else "forward"
        delivery_local_start_time = recv_start

        # v1.8.0: Default to direct UCX mode (pull model)
        # Receiver directly blocks on UCX queue without ZMQ signal
        if self._ucx_direct_recv and self._ucx_available:
            combined_data, _, _ = self._recv_via_ucx_pure(direction, timeout=30.0)
            deser_start = time.time()
            
            # Check if this is GPU Direct format by looking at flags
            if combined_data is not None and len(combined_data) >= 17:
                combined_mv = combined_data if isinstance(combined_data, memoryview) else memoryview(combined_data)
                flags = combined_mv[0]
                is_gpu_direct = bool(flags & 4)  # bit 2 = GPU Direct mode
                if is_gpu_direct:
                    logger.debug("Stage %d: Detected GPU Direct mode (flags=0x%02x)", 
                                self.stage_idx, flags)
                    return self._recv_tensor_dict_gpu_direct(
                        direction, recv_start, recv_start, pre_received_header=combined_data
                    )
        else:
            # Fallback mode: Wait for signal via ZeroMQ first
            if self.router_socket is None:
                raise RuntimeError("Router socket not initialized")

            while True:
                frames = self._wait_signal()
                if frames is None:
                    return None
                corr = (struct.unpack("!Q", frames[3])[0]
                        if len(frames) > 3 and len(frames[3]) == 8 else None)
                if not self.is_first:
                    self._corr_in = corr
                    break
                if corr is None or self._awaiting_corr is None or corr == self._awaiting_corr:
                    break
                logger.warning("Stage %d: dropping return for an abandoned step (corr %x, waiting for %x)",
                               self.stage_idx, corr, self._awaiting_corr)
                if frames[1] in (b"U", b"UCX", b"G") and len(frames) > 2 and len(frames[2]) == 8:
                    dropped_ts = struct.unpack("!d", frames[2])[0]
                    if self._ucx_stash.pop(dropped_ts, None) is None and len(self._ucx_discard) < 256:
                        self._ucx_discard.add(dropped_ts)
            identity = frames[0]
            mode_data = frames[1] if len(frames) > 1 else b"ZMQ"
            signal_received_time = time.time()
            
            deser_start = time.time()
            combined_data = None
            send_timestamp = 0
            
            # Mode "G" = GPU Direct RDMA mode
            # Mode "M" = Metadata-first (legacy)
            # Mode "ZMQ" = ZeroMQ-only mode (fallback)
            if mode_data == b"G":
                want_ts = (struct.unpack("!d", frames[2])[0]
                           if len(frames) > 2 and len(frames[2]) == 8 else None)
                header = self._recv_ucx_matching(direction, want_ts)
                if header is None:
                    return None
                return self._recv_tensor_dict_gpu_direct(
                    direction, signal_received_time, recv_start, pre_received_header=header)
            elif mode_data.startswith(b"M"):
                # Legacy pull model
                combined_data, _, _ = self._recv_via_ucx_pure(direction)
                delivery_local_start_time = signal_received_time
            elif mode_data in (b"U", b"UCX"):
                want_ts = (struct.unpack("!d", frames[2])[0]
                           if len(frames) > 2 and len(frames[2]) == 8 else None)
                combined_data = self._recv_ucx_matching(direction, want_ts)
                if combined_data is None:
                    return None
                delivery_local_start_time = signal_received_time
            elif mode_data == b"ZMQ" and len(frames) > 2:
                combined_data = frames[2]
                delivery_local_start_time = recv_start
            else:
                raise RuntimeError(f"Invalid message mode: {mode_data}")
        
        if combined_data is None or len(combined_data) == 0:
            return {}
        
        data_received_time = time.time()
        
        # v1.8.0: Parse compact header format (17 bytes)
        # [flags(1)][tensor_len(4)][meta_len(4)][timestamp(8)]
        combined_mv = combined_data if isinstance(combined_data, memoryview) else memoryview(combined_data)

        flags = combined_mv[0]
        use_safetensors = bool(flags & 1)
        use_msgpack = bool(flags & 2)
        
        # Detect header format based on data length and content
        # v1.8.0 compact header: 17 bytes (1+4+4+8)
        # v1.7.x legacy header: 25 bytes (1+8+8+8) after stripping stage_idx+epoch
        header_valid = False
        if len(combined_mv) >= 17:
            # Try compact header first (17 bytes)
            tensor_len, non_tensor_len, send_timestamp = struct.unpack("!IId", combined_mv[1:17])
            header_size = 17
            
            # Validate: if tensor_len + non_tensor_len + header_size != total, try legacy
            if tensor_len + non_tensor_len + header_size == len(combined_mv):
                header_valid = True
                logger.debug("Stage %d: Using compact 17-byte header (tensor=%d, meta=%d)", 
                            self.stage_idx, tensor_len, non_tensor_len)
            elif len(combined_mv) >= 25:
                # Fallback to legacy 25-byte header
                tensor_len_q, non_tensor_len_q, send_timestamp = struct.unpack("!QQd", combined_mv[1:25])
                tensor_len, non_tensor_len = int(tensor_len_q), int(non_tensor_len_q)
                header_size = 25
                if tensor_len + non_tensor_len + header_size == len(combined_mv):
                    header_valid = True
                    logger.debug("Stage %d: Using legacy 25-byte header (tensor=%d, meta=%d)",
                                self.stage_idx, tensor_len, non_tensor_len)
        
        if not header_valid:
            # Log detailed diagnostic info
            logger.error("Stage %d: Header validation failed! Data length=%d, flags=0x%02x",
                        self.stage_idx, len(combined_mv), flags)
            if len(combined_mv) >= 17:
                t17, m17, _ = struct.unpack("!IId", combined_mv[1:17])
                logger.error("  17-byte parse: tensor=%d, meta=%d, sum=%d", t17, m17, t17+m17+17)
            if len(combined_mv) >= 25:
                t25, m25, _ = struct.unpack("!QQd", combined_mv[1:25])
                logger.error("  25-byte parse: tensor=%d, meta=%d, sum=%d", t25, m25, t25+m25+25)
            logger.error("  First 50 bytes hex: %s", combined_mv[:min(50, len(combined_mv))].hex())
            raise RuntimeError(f"Stage {self.stage_idx}: Invalid header format, data length={len(combined_mv)}")
        
        tensor_bytes_mv = combined_mv[header_size:header_size + tensor_len]
        non_tensor_bytes_mv = combined_mv[header_size + tensor_len:header_size + tensor_len + non_tensor_len]
        
        # Deserialize tensors
        tensor_dict: dict[str, Any] = {}
        
        if tensor_bytes_mv:
            # safetensors requires PyBytes (copy)
            tensor_bytes = bytes(tensor_bytes_mv)
            if use_safetensors and SAFETENSORS_AVAILABLE:
                tensor_dict = dict(safetensors_load(tensor_bytes))
            else:
                buffer = io.BytesIO(tensor_bytes)
                try:
                    tensor_dict = torch.load(buffer, weights_only=True, map_location="cpu")
                except Exception:
                    buffer.seek(0)
                    tensor_dict = torch.load(buffer, weights_only=False, map_location="cpu")
        
        # Deserialize non-tensors
        if non_tensor_bytes_mv:
            non_tensors = (
                _deserialize_metadata(non_tensor_bytes_mv)
                if use_msgpack
                else pickle.loads(non_tensor_bytes_mv)
            )
            # Note: send_timestamp now comes from binary header, not metadata
            non_tensors.pop("__send_timestamp__", None)  # Remove if present (backwards compat)
            non_tensors.pop("__epoch__", None)
            non_tensors.pop("__seqno__", None)
            tensor_dict.update(non_tensors)
        
        deserialization_time = time.time() - deser_start
        
        # v1.7.4: Measure GPU transfer time separately, use non_blocking for overlap
        gpu_transfer_start = time.time()
        # Move tensors to GPU if needed
        if self.device == "cuda":
            for key, value in tensor_dict.items():
                if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                    tensor_dict[key] = value.cuda(non_blocking=True)
            # Sync to ensure all transfers complete
            torch.cuda.current_stream().synchronize()
        gpu_transfer_time = time.time() - gpu_transfer_start
        
        recv_time = time.time() - recv_start

        if METRICS_ENABLED:
            vllm_pipeline_comm_bytes_received.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).inc(len(combined_data))
            vllm_pipeline_comm_messages_received.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).inc()
            vllm_pipeline_comm_recv_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(recv_time)
            vllm_pipeline_comm_deserialization_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).observe(deserialization_time)
            # v1.5.6: Delivery latency = time from send start to data fully received
            # This excludes deserialization and GPU transfer on receiver side
            vllm_pipeline_comm_delivery_latency_local.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(max(0.0, data_received_time - delivery_local_start_time))
            if send_timestamp > 0:
                delivery_latency = data_received_time - send_timestamp
                if delivery_latency >= 0:
                    vllm_pipeline_comm_delivery_latency.labels(
                        backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
                    ).observe(delivery_latency)
            # v1.5.6: Ready-to-use latency = GPU transfer time (tensor ready on GPU)
            vllm_pipeline_comm_ready_to_use_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(gpu_transfer_time)
        
        return tensor_dict
    
    def _recv_tensor_dict_gpu_direct(
        self,
        direction: str,
        signal_received_time: float,
        recv_start: float,
        pre_received_header: memoryview | bytes | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        """Receive tensor dictionary with GPU Direct RDMA.
        
        v1.7.4.8: Zero-copy GPU-to-GPU receive path.
        
        Message format:
        1. Header + metadata + CPU tensors (standard UCX)
        2. Each CUDA tensor directly via GPU Direct RDMA
        
        The receiver gets tensor metadata first, then receives each GPU tensor
        directly into GPU memory without CPU staging.
        
        v1.7.5.6: Added pre_received_header parameter for cases where header
        was already received (e.g., in direct UCX recv mode).
        v1.7.5.7: Added detailed timing for performance analysis.
        """
        t0 = time.time()
        
        # Receive header message via UCX (unless already received)
        if pre_received_header is not None:
            combined_data = pre_received_header
            t_header = t0  # Already received
        else:
            combined_data, _, _ = self._recv_via_ucx_pure(direction)
            t_header = time.time()
        
        if combined_data is None or len(combined_data) == 0:
            return {}
        
        deser_start = time.time()
        data_received_time = time.time()
        combined_mv = combined_data if isinstance(combined_data, memoryview) else memoryview(combined_data)
        
        # Parse header: [flags(1)][meta_len(8)][cpu_tensor_len(8)][timestamp(8)]
        flags = combined_mv[0]
        is_gpu_direct = bool(flags & 4)
        use_safetensors = bool(flags & 1)
        use_msgpack = bool(flags & 2)
        
        meta_len, cpu_tensor_len, send_timestamp = struct.unpack("!QQd", combined_mv[1:25])
        meta_bytes = combined_mv[25:25 + meta_len]
        cpu_tensor_bytes = combined_mv[25 + meta_len:25 + meta_len + cpu_tensor_len]
        
        t_parse = time.time()
        
        # Deserialize metadata
        metadata = _deserialize_metadata(meta_bytes) if use_msgpack else pickle.loads(bytes(meta_bytes))
        tensor_meta = metadata.pop("__tensor_meta__", {})
        num_cuda_tensors = metadata.pop("__num_cuda_tensors__", 0)
        metadata.pop("__gpu_direct__", None)
        metadata.pop("__epoch__", None)
        metadata.pop("__seqno__", None)
        
        tensor_dict: dict[str, Any] = dict(metadata)
        
        # Deserialize CPU tensors
        if cpu_tensor_bytes:
            cpu_tensor_bytes_copy = bytes(cpu_tensor_bytes)
            if use_safetensors and SAFETENSORS_AVAILABLE:
                cpu_tensors = dict(safetensors_load(cpu_tensor_bytes_copy))
            else:
                buffer = io.BytesIO(cpu_tensor_bytes_copy)
                try:
                    cpu_tensors = torch.load(buffer, weights_only=True, map_location="cpu")
                except Exception:
                    buffer.seek(0)
                    cpu_tensors = torch.load(buffer, weights_only=False, map_location="cpu")
            
            # Move CPU tensors to GPU
            for k, v in cpu_tensors.items():
                if isinstance(v, torch.Tensor) and self.device == "cuda":
                    tensor_dict[k] = v.cuda(non_blocking=True)
                else:
                    tensor_dict[k] = v
        
        # Receive CUDA tensors via GPU Direct
        # v1.7.5.4: Receive tensors via _recv_via_ucx_pure which handles
        # the metadata+tensor protocol from _send_ucx_cuda automatically
        cuda_tensor_names = sorted([
            k for k, m in tensor_meta.items() if m.get("device") == "cuda"
        ])
        
        # v1.8.10: Batch receive all CUDA tensors with minimal blocking
        # Since send_batch_gpu_direct sends all tensors atomically, they should
        # all be in the queue by the time we receive the header.
        for i, tensor_name in enumerate(cuda_tensor_names):
            meta = tensor_meta[tensor_name]
            shape = tuple(meta["shape"])
            dtype_str = meta["dtype"]
            
            # v1.8.18: Use module-level DTYPE_STR_TO_TORCH constant
            dtype = DTYPE_STR_TO_TORCH.get(dtype_str, torch.float32)
            
            # v1.8.10: Use short timeout for subsequent tensors (they should already be queued)
            # First tensor may need longer wait, subsequent tensors are already in queue
            timeout = 30.0 if i == 0 else 0.5
            recv_data, _, _ = self._recv_via_ucx_pure(direction, timeout=timeout)
            
            if isinstance(recv_data, torch.Tensor):
                # GPU Direct: received tensor directly from recv loop
                if recv_data.device.type == "cuda":
                    if recv_data.shape == shape and recv_data.dtype == dtype:
                        tensor_dict[tensor_name] = recv_data
                    elif recv_data.numel() == meta.get("nbytes", 0) // recv_data.element_size():
                        tensor_dict[tensor_name] = recv_data.reshape(shape)
                    else:
                        logger.warning(
                            "Stage %d: GPU Direct tensor %s shape mismatch: expected %s, got %s",
                            self.stage_idx, tensor_name, shape, recv_data.shape
                        )
                        tensor_dict[tensor_name] = recv_data.reshape(shape)
                else:
                    tensor_dict[tensor_name] = recv_data.to(dtype=dtype, device="cuda").reshape(shape)
            else:
                # Fallback: received as bytes
                recv_bytes = bytes(recv_data) if isinstance(recv_data, memoryview) else recv_data
                cpu_tensor = torch.frombuffer(bytearray(recv_bytes), dtype=dtype).reshape(shape)
                tensor_dict[tensor_name] = cpu_tensor.cuda()
        
        # v1.8.11: Skip explicit CUDA sync for GPU Direct mode
        # The tensor was received directly into GPU memory via RDMA.
        # UCX's recv() is synchronous - it waits for the transfer to complete.
        # Any subsequent CUDA operation on the tensor will implicitly sync.
        # This avoids an extra sync barrier that adds latency.
        gpu_transfer_start = time.time()
        # Note: Not calling synchronize() here - let the model forward do implicit sync
        gpu_transfer_time = 0.0
        
        deserialization_time = time.time() - deser_start
        recv_time = time.time() - recv_start
        
        if METRICS_ENABLED:
            total_bytes = len(combined_data) + sum(
                m.get("nbytes", 0) for m in tensor_meta.values() if m.get("device") == "cuda"
            )
            vllm_pipeline_comm_bytes_received.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).inc(total_bytes)
            vllm_pipeline_comm_messages_received.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).inc()
            vllm_pipeline_comm_recv_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(recv_time)
            vllm_pipeline_comm_deserialization_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx)
            ).observe(deserialization_time)
            # v1.7.5: Add delivery_latency metrics for GPU Direct mode
            # Use signal_received_time as local start since that's when we knew data was coming
            vllm_pipeline_comm_delivery_latency_local.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(max(0.0, data_received_time - signal_received_time))
            if send_timestamp > 0:
                delivery_latency = data_received_time - send_timestamp
                if delivery_latency >= 0:
                    vllm_pipeline_comm_delivery_latency.labels(
                        backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
                    ).observe(delivery_latency)
            # v1.7.5.3: ready_to_use latency = actual GPU sync time (not recv - deser)
            vllm_pipeline_comm_ready_to_use_latency.labels(
                backend=self.backend_name, stage_idx=str(self.stage_idx), direction=direction
            ).observe(gpu_transfer_time)
        
        logger.debug(
            "Stage %d: GPU Direct recv completed (%.3fs, cuda_tensors=%d)",
            self.stage_idx, recv_time, len(cuda_tensor_names)
        )
        
        return tensor_dict
    
    def _wait_signal(self) -> list[bytes] | None:
        """Wait for the next ZMQ signal; None on peer loss (stage 0) or timeout."""
        self._peer_lost.clear()
        self._waiting_recv = True
        deadline = time.time() + self._recv_timeout_s
        try:
            while True:
                if self._peer_lost.is_set():
                    return None
                if self.router_socket.poll(50):
                    return self.router_socket.recv_multipart()
                if time.time() >= deadline:
                    if self.is_first:
                        logger.warning(
                            "Stage %d: no return within %.1fs; peer may have failed",
                            self.stage_idx, self._recv_timeout_s,
                        )
                    return None
        finally:
            self._waiting_recv = False

    @staticmethod
    def _header_timestamp(mv: memoryview) -> float | None:
        """Send timestamp from a 17- or 25-byte compact header, if parseable."""
        n = len(mv)
        if n >= 17:
            t, m, ts = struct.unpack("!IId", mv[1:17])
            if t + m + 17 == n:
                return ts
        if n >= 25:
            t, m, ts = struct.unpack("!QQd", mv[1:25])
            if t + m + 25 == n:
                return ts
        return None

    def _recv_ucx_matching(self, direction: str, want_ts: float | None):
        """Receive the UCX payload announced by a signal.

        Payloads are matched by header send timestamp: older ones (left over
        from a peer that was abandoned) are dropped, newer ones are kept for
        their own signal. Returns None on timeout or peer loss.
        """
        if want_ts is not None:
            stashed = self._ucx_stash.pop(want_ts, None)
            if stashed is not None:
                return stashed
        deadline = time.time() + self._recv_timeout_s
        self._waiting_recv = True
        try:
            while True:
                if self._peer_lost.is_set():
                    return None
                left = deadline - time.time()
                if left <= 0:
                    logger.warning("Stage %d: UCX payload for signal not received in %.1fs",
                                   self.stage_idx, self._recv_timeout_s)
                    return None
                result = self._ucx_transport.recv_from_queue(timeout=min(left, 0.05))
                if result is None:
                    continue
                peer, data = result
                self._last_data_peer = peer
                if isinstance(data, torch.Tensor) or want_ts is None:
                    return data if isinstance(data, torch.Tensor) else memoryview(data)
                mv = memoryview(data)
                ts = self._header_timestamp(mv)
                if ts is not None and ts in self._ucx_discard:
                    self._ucx_discard.discard(ts)
                    continue
                if ts is None or ts == want_ts:
                    return mv
                if ts < want_ts:
                    logger.warning("Stage %d: dropping stale UCX payload (%.6f < %.6f)",
                                   self.stage_idx, ts, want_ts)
                    continue
                if len(self._ucx_stash) < 64:
                    self._ucx_stash[ts] = mv
        finally:
            self._waiting_recv = False

    def _recv_via_ucx_pure(self, direction: str, timeout: float = 30.0) -> tuple[memoryview | torch.Tensor, int, int]:
        """Receive data via pure UCX mode with blocking wait.
        
        v1.8.0: Simplified for Pull model - data is returned directly without
        stripping stage_idx/epoch prefix (compact header doesn't have these).
        
        v1.8.11: Optimized - use single queue.get() call with timeout.
        The previous "try non-blocking first" pattern added overhead.
        
        Returns (data, src_stage, epoch) where:
        - data: raw message data (memoryview or torch.Tensor for GPU Direct)
        - src_stage: always 0 for v1.8.0 compact header
        - epoch: current epoch
        """
        # v1.8.11: Single blocking get with timeout
        # This is more efficient than try-non-blocking-then-blocking pattern
        result = self._ucx_transport.recv_from_queue(timeout=timeout)
        if result is not None:
            self._last_data_peer, data = result
            # GPU Direct may return CUDA tensor directly
            if isinstance(data, torch.Tensor):
                return data, 0, self._epoch
            # v1.8.0: Return data directly (no stage_idx/epoch prefix to strip)
            mv = memoryview(data)
            return mv, 0, self._epoch
        
        raise RuntimeError(f"Stage {self.stage_idx}: UCX recv timeout after {timeout}s")

    def get_connex_metrics(self) -> dict[str, Any]:
        """Get Connex-specific metrics."""
        return {
            "connection_version": self._connection_version,
            "epoch": self._epoch,
            "cached_endpoints": len(self._endpoint_cache),
            "epc_available": CONNEX_AVAILABLE,
            "stream_seqnos": dict(self._stream_seqno),
            "pending_buffers": len(self._pending_send_buffers),
        }

    def close(self) -> None:
        """Close communicator and cleanup."""
        # Stop heartbeat thread first
        if hasattr(self, '_heartbeat_stop_event'):
            self._heartbeat_stop_event.set()
        if hasattr(self, '_heartbeat_thread') and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        
        # v2.0: Stop event subscription thread
        if hasattr(self, '_event_stop_event'):
            self._event_stop_event.set()
        if hasattr(self, '_event_thread') and self._event_thread is not None and self._event_thread.is_alive():
            self._event_thread.join(timeout=2.0)
        
        # v1.8.0: Cleanup buffer cleaner
        if self._buffer_cleaner is not None:
            self._buffer_cleaner.shutdown(wait=False)
            self._buffer_cleaner = None
        self._pending_send_buffers.clear()
        
        self._endpoint_cache.clear()
        
        for socket in [
            self.forward_socket, 
            self.return_socket, 
            self.router_socket, 
            self.control_socket,
            getattr(self, '_event_socket', None),  # v2.0: Include event socket
        ]:
            if socket:
                try:
                    socket.close()
                except Exception:
                    pass
        
        if self.context:
            try:
                self.context.term()
            except Exception:
                pass
