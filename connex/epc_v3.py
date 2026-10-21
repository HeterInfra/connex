# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
EPC V3.0: Mobility-Aware Elastic Pipeline Communications.

Enhanced EPC API integrating all V3.0 features:
- Unified protocol headers with epoch-based versioning
- Versioned routing with shadow connection warmup  
- Multi-queue scheduling with credit-based backpressure
- GPU-aware fast path with registration caching
- Mooncake TE integration for KV cache transfer

"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import msgpack

from connex.directory import (
    DirectoryClient,
    EndpointInfo,
    MembershipEvent,
    TransportCapability as DirTransportCapability,
)
from connex.route_table import RouteEntry, RouteTable
from connex.cutover_fsm import CutoverContext, CutoverFSM, CutoverState
from connex.stream_manager import StreamManager, StreamMessage
from connex.ucx_transport import TransportMode, UCXTransport
from connex.qos_scheduler import QoSScheduler, TrafficClass

# V3.0 Mobility-Aware Transport Runtime
from connex.protocol import (
    LogicalEndpointID,
    StreamID,
    MessageClass,
    MessageType,
    ProtocolHeader,
    MobilityMessage,
    CreditInfo,
    create_data_message,
    create_credit_update,
    create_stale_epoch_feedback,
    create_heartbeat,
)
from connex.mobility_control import (
    VersionedRoute,
    VersionedRouteTable,
    ConnectionCapability,
    ConnectionState,
    RouteState,
    ShadowConnectionManager,
    DualPathCutoverManager,
    DualPathState,
)
from connex.mobility_dataplane import (
    MultiQueueScheduler,
    StreamMultiplexer,
    CongestionController,
    PeerQueueState,
    StreamReorderState,
)

# Prometheus metrics for dynamic scaling
try:
    from vllm.distributed.device_communicators.pipeline_comm_metrics import (
        vllm_pipeline_reconfiguration_latency,
        vllm_pipeline_epoch_transitions,
        vllm_pipeline_stale_epoch_events,
        vllm_pipeline_scale_events,
        vllm_pipeline_recovery_time,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

from connex.gpu_fastpath import (
    GPUAwareFastPath,
    RegistrationCache,
    ChunkedTensorTransfer,
    CapabilityNegotiator,
    TransportCapability,
    TransportSelection,
)
from connex.mooncake_integration import (
    ConnexMooncakeAdapter,
    KVCacheTransferManager,
    KVCacheBlock,
    KVTransferRequest,
    KVTransferStatus,
)

logger = logging.getLogger(__name__)


# ========== Dynamic Scaling Manager ==========

@dataclass
class ScalingEvent:
    """Represents a scale-up or scale-down event."""
    event_type: str  # "scale_up", "scale_down", "migrate"
    target_stage: int
    new_replica_idx: int = 0
    new_address: str = ""
    old_address: str = ""
    new_epoch: int = 0
    timestamp: float = field(default_factory=time.time)
    completed: bool = False
    
    
class DynamicScalingManager:
    """Manages dynamic scaling operations for EPC V3.0.
    
    Default mode for production deployment. Handles:
    - Scale-up: Adding new pipeline stage replicas
    - Scale-down: Removing pipeline stage replicas
    - Migration: Moving replicas to different nodes
    
    Uses shadow connection warmup for zero-downtime reconfiguration.
    """
    
    # Default scaling parameters
    WARMUP_TIMEOUT_S = 10.0
    DRAIN_TIMEOUT_S = 5.0
    CUTOVER_RETRY_MAX = 3
    
    def __init__(
        self,
        endpoint: "EPCV3Endpoint",
        on_scale_complete: Optional[Callable[[ScalingEvent], None]] = None,
    ):
        """Initialize dynamic scaling manager.
        
        Args:
            endpoint: Parent EPC endpoint
            on_scale_complete: Callback when scaling completes
        """
        self.endpoint = endpoint
        self.on_scale_complete = on_scale_complete
        
        # Scaling state
        self._pending_events: Dict[int, ScalingEvent] = {}  # epoch -> event
        self._lock = threading.Lock()
        self._scaling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self._stats = {
            "scale_up_count": 0,
            "scale_down_count": 0,
            "migration_count": 0,
            "avg_reconfiguration_latency_ms": 0.0,
            "reconfiguration_latencies": [],
            "failed_events": 0,
        }
        
    def start(self) -> None:
        """Start the scaling manager."""
        self._stop_event.clear()
        self._scaling_thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="scaling-manager",
        )
        self._scaling_thread.start()
        logger.info("Dynamic scaling manager started")
        
    def stop(self) -> None:
        """Stop the scaling manager."""
        self._stop_event.set()
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
            self._scaling_thread = None
        logger.info("Dynamic scaling manager stopped")
        
    def _event_loop(self) -> None:
        """Main event loop for processing membership changes."""
        while not self._stop_event.wait(timeout=0.1):
            # Check for membership events from directory
            if hasattr(self.endpoint.directory, 'poll_events'):
                events = self.endpoint.directory.poll_events()
                for event in events:
                    self._handle_membership_event(event)
                    
    def _handle_membership_event(self, event: MembershipEvent) -> None:
        """Handle membership event from directory service."""
        logger.info("Received membership event: %s", event)
        
        if event.event_type == "replica_added":
            self.initiate_scale_up(
                target_stage=event.stage_idx,
                new_address=event.new_address,
                new_replica_idx=event.replica_idx,
            )
        elif event.event_type == "replica_removed":
            self.initiate_scale_down(
                target_stage=event.stage_idx,
                replica_idx=event.replica_idx,
            )
        elif event.event_type == "replica_migrating":
            self.initiate_migration(
                target_stage=event.stage_idx,
                old_address=event.old_address,
                new_address=event.new_address,
            )
            
    def initiate_scale_up(
        self,
        target_stage: int,
        new_address: str,
        new_replica_idx: int = 0,
    ) -> bool:
        """Initiate scale-up operation for a stage.
        
        Adds a new replica with shadow connection warmup.
        
        Args:
            target_stage: Target pipeline stage
            new_address: Address of new replica
            new_replica_idx: Replica index
            
        Returns:
            True if scale-up initiated successfully
        """
        new_epoch = self.endpoint.bump_epoch()
        
        event = ScalingEvent(
            event_type="scale_up",
            target_stage=target_stage,
            new_replica_idx=new_replica_idx,
            new_address=new_address,
            new_epoch=new_epoch,
        )
        
        with self._lock:
            self._pending_events[new_epoch] = event
            
        # Start async scale-up process
        thread = threading.Thread(
            target=self._execute_scale_up,
            args=(event,),
            daemon=True,
        )
        thread.start()
        
        self._stats["scale_up_count"] += 1
        return True
        
    def _execute_scale_up(self, event: ScalingEvent) -> None:
        """Execute scale-up with shadow warmup.
        
        Steps:
        1. Create shadow connection to new replica
        2. Warmup connection (pre-register MRs, exchange caps)
        3. Add to route table with new epoch
        4. Complete cutover
        """
        start_time = time.time()
        
        try:
            # Step 1: Create target LID
            target_lid = LogicalEndpointID(
                model_name=self.endpoint.model_name,
                stage_idx=event.target_stage,
                shard_idx=0,
                replica_idx=event.new_replica_idx,
            )
            
            # Step 2: Create shadow connection if enabled
            if self.endpoint.config.enable_shadow_warmup:
                shadow_route = self.endpoint.shadow_manager.create_shadow(
                    lid=target_lid,
                    new_address=event.new_address,
                    new_epoch=event.new_epoch,
                )
                
                # Wait for warmup to complete
                warmup_success = self._wait_for_warmup(shadow_route)
                if not warmup_success:
                    logger.error("Shadow warmup failed for scale-up")
                    self._stats["failed_events"] += 1
                    return
                    
            # Step 3: Establish actual connection
            if ":" in event.new_address:
                host, port = event.new_address.rsplit(":", 1)
                conn = self.endpoint.transport.connect_sync(host, int(port))
                
                # Add to connections and route table
                with self.endpoint._lock:
                    if event.target_stage not in self.endpoint._connections:
                        self.endpoint._connections[event.target_stage] = conn
                        
                # Create versioned route
                peer_caps = self.endpoint._negotiate_capabilities(conn)
                self.endpoint.versioned_route_table.add_route(
                    lid=target_lid,
                    addresses=[event.new_address],
                    capabilities=peer_caps,
                    epoch=event.new_epoch,
                )
                
            # Step 4: Mark complete
            event.completed = True
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(latency_ms, event_type="scale_up")
            self._observe_scale_event("scale_up", "success")
            
            logger.info(
                "Scale-up complete for stage %d replica %d (latency: %.2fms, epoch: %d)",
                event.target_stage, event.new_replica_idx, latency_ms, event.new_epoch
            )
            
            if self.on_scale_complete:
                self.on_scale_complete(event)
                
        except Exception as e:
            logger.error("Scale-up failed: %s", e)
            self._stats["failed_events"] += 1
            self._observe_scale_event("scale_up", "failed")
            
    def initiate_scale_down(
        self,
        target_stage: int,
        replica_idx: int = 0,
    ) -> bool:
        """Initiate scale-down operation.
        
        Removes a replica with graceful drain.
        
        Args:
            target_stage: Target pipeline stage
            replica_idx: Replica to remove
            
        Returns:
            True if scale-down initiated
        """
        new_epoch = self.endpoint.bump_epoch()
        
        event = ScalingEvent(
            event_type="scale_down",
            target_stage=target_stage,
            new_replica_idx=replica_idx,
            new_epoch=new_epoch,
        )
        
        with self._lock:
            self._pending_events[new_epoch] = event
            
        # Start async scale-down
        thread = threading.Thread(
            target=self._execute_scale_down,
            args=(event,),
            daemon=True,
        )
        thread.start()
        
        self._stats["scale_down_count"] += 1
        return True
        
    def _execute_scale_down(self, event: ScalingEvent) -> None:
        """Execute scale-down with graceful drain.
        
        Steps:
        1. Mark replica as draining
        2. Wait for in-flight messages to complete
        3. Remove from route table
        4. Close connection
        """
        start_time = time.time()
        
        try:
            target_lid = LogicalEndpointID(
                model_name=self.endpoint.model_name,
                stage_idx=event.target_stage,
                shard_idx=0,
                replica_idx=event.new_replica_idx,
            )
            
            # Step 1: Mark as draining (stop sending new traffic)
            self.endpoint.versioned_route_table.mark_draining(target_lid)
            
            # Step 2: Wait for drain
            drain_success = self._wait_for_drain(target_lid)
            if not drain_success:
                logger.warning("Drain timeout, proceeding with removal")
                
            # Step 3: Remove from route table
            self.endpoint.versioned_route_table.remove_route(target_lid)
            
            # Step 4: Close connection
            with self.endpoint._lock:
                if event.target_stage in self.endpoint._connections:
                    conn = self.endpoint._connections.pop(event.target_stage)
                    # Close connection if transport supports it
                    if hasattr(self.endpoint.transport, 'close_connection'):
                        self.endpoint.transport.close_connection(conn)
                        
            event.completed = True
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(latency_ms, event_type="scale_down")
            self._observe_scale_event("scale_down", "success")
            
            logger.info(
                "Scale-down complete for stage %d replica %d (latency: %.2fms)",
                event.target_stage, event.new_replica_idx, latency_ms
            )
            
            if self.on_scale_complete:
                self.on_scale_complete(event)
                
        except Exception as e:
            logger.error("Scale-down failed: %s", e)
            self._stats["failed_events"] += 1
            self._observe_scale_event("scale_down", "failed")
            
    def initiate_migration(
        self,
        target_stage: int,
        old_address: str,
        new_address: str,
    ) -> bool:
        """Initiate replica migration.
        
        Uses dual-path cutover for zero-downtime migration.
        
        Args:
            target_stage: Target stage
            old_address: Current address
            new_address: New address after migration
            
        Returns:
            True if migration initiated
        """
        new_epoch = self.endpoint.bump_epoch()
        
        event = ScalingEvent(
            event_type="migrate",
            target_stage=target_stage,
            old_address=old_address,
            new_address=new_address,
            new_epoch=new_epoch,
        )
        
        with self._lock:
            self._pending_events[new_epoch] = event
            
        # Start async migration
        thread = threading.Thread(
            target=self._execute_migration,
            args=(event,),
            daemon=True,
        )
        thread.start()
        
        self._stats["migration_count"] += 1
        return True
        
    def _execute_migration(self, event: ScalingEvent) -> None:
        """Execute migration with dual-path cutover.
        
        Steps:
        1. Create shadow connection to new location
        2. Warmup shadow connection
        3. Start dual-path mode (send to both)
        4. Wait for old path to drain
        5. Switch to new path only
        """
        start_time = time.time()
        
        try:
            target_lid = LogicalEndpointID(
                model_name=self.endpoint.model_name,
                stage_idx=event.target_stage,
            )
            
            # Step 1 & 2: Shadow warmup
            if self.endpoint.config.enable_shadow_warmup:
                shadow_route = self.endpoint.shadow_manager.create_shadow(
                    lid=target_lid,
                    new_address=event.new_address,
                    new_epoch=event.new_epoch,
                )
                
                warmup_success = self._wait_for_warmup(shadow_route)
                if not warmup_success:
                    logger.error("Shadow warmup failed for migration")
                    self._stats["failed_events"] += 1
                    return
                    
            # Step 3: Start dual-path mode
            stream_id = StreamID(request_id="migration", direction="forward", token_step=0)
            self.endpoint.cutover_manager.start_dual_path(
                stream_id=stream_id,
                old_address=event.old_address,
                new_address=event.new_address,
            )
            
            # Step 4: Wait for drain
            drain_success = self._wait_for_drain(target_lid)
            
            # Step 5: Complete cutover
            self.endpoint.cutover_manager.complete_cutover(stream_id)
            
            # Update connection
            if ":" in event.new_address:
                host, port = event.new_address.rsplit(":", 1)
                conn = self.endpoint.transport.connect_sync(host, int(port))
                
                with self.endpoint._lock:
                    self.endpoint._connections[event.target_stage] = conn
                    
            # Update route
            peer_caps = ConnectionCapability()
            self.endpoint.versioned_route_table.update_route(
                lid=target_lid,
                new_addresses=[event.new_address],
                new_epoch=event.new_epoch,
            )
            
            event.completed = True
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(latency_ms, event_type="migration")
            self._observe_scale_event("migration", "success")
            
            logger.info(
                "Migration complete for stage %d: %s -> %s (latency: %.2fms)",
                event.target_stage, event.old_address, event.new_address, latency_ms
            )
            
            if self.on_scale_complete:
                self.on_scale_complete(event)
                
        except Exception as e:
            logger.error("Migration failed: %s", e)
            self._stats["failed_events"] += 1
            self._observe_scale_event("migration", "failed")
            
    def _wait_for_warmup(
        self,
        shadow_route: VersionedRoute,
        timeout_s: float = None,
    ) -> bool:
        """Wait for shadow connection warmup to complete."""
        timeout = timeout_s or self.WARMUP_TIMEOUT_S
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if shadow_route.warmup_complete:
                return True
            time.sleep(0.05)
            
        return False
        
    def _wait_for_drain(
        self,
        lid: LogicalEndpointID,
        timeout_s: float = None,
    ) -> bool:
        """Wait for in-flight messages to drain."""
        timeout = timeout_s or self.DRAIN_TIMEOUT_S
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            # Check if queue is empty for this peer
            peer_state = self.endpoint.mq_scheduler.get_peer_state(lid)
            if peer_state and peer_state.is_drained():
                return True
            time.sleep(0.05)
            
        return False
        
    def _record_latency(self, latency_ms: float, event_type: str = "unknown") -> None:
        """Record reconfiguration latency for statistics and Prometheus."""
        self._stats["reconfiguration_latencies"].append(latency_ms)
        
        # Update running average
        latencies = self._stats["reconfiguration_latencies"]
        self._stats["avg_reconfiguration_latency_ms"] = sum(latencies) / len(latencies)
        
        # Observe Prometheus metric
        if _METRICS_AVAILABLE:
            try:
                stage_idx = str(getattr(self.endpoint, 'stage_idx', 0))
                vllm_pipeline_reconfiguration_latency.labels(
                    backend="connex-v3",
                    stage_idx=stage_idx,
                    event_type=event_type,
                ).observe(latency_ms / 1000.0)  # Convert to seconds
            except Exception:
                pass  # Ignore metric errors

    def _observe_scale_event(self, event_type: str, status: str) -> None:
        """Observe scale event in Prometheus metrics."""
        if _METRICS_AVAILABLE:
            try:
                stage_idx = str(getattr(self.endpoint, 'stage_idx', 0))
                vllm_pipeline_scale_events.labels(
                    backend="connex-v3",
                    stage_idx=stage_idx,
                    event_type=event_type,
                    status=status,
                ).inc()
            except Exception:
                pass  # Ignore metric errors

        
    def get_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self._lock:
            pending_count = len([e for e in self._pending_events.values() if not e.completed])
            
        return {
            "scale_up_count": self._stats["scale_up_count"],
            "scale_down_count": self._stats["scale_down_count"],
            "migration_count": self._stats["migration_count"],
            "failed_events": self._stats["failed_events"],
            "pending_events": pending_count,
            "avg_reconfiguration_latency_ms": self._stats["avg_reconfiguration_latency_ms"],
            "latency_p50_ms": self._compute_percentile(50),
            "latency_p99_ms": self._compute_percentile(99),
        }
        
    def _compute_percentile(self, p: int) -> float:
        """Compute latency percentile."""
        latencies = self._stats["reconfiguration_latencies"]
        if not latencies:
            return 0.0
            
        sorted_lat = sorted(latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


@dataclass
class EPCV3Config:
    """Configuration for EPC V3.0 endpoint."""
    model_name: str
    stage_idx: int
    num_stages: int
    
    # Network
    local_address: str = ""
    local_port: int = 0
    front_router_addr: str = ""  # Optional - leave empty for direct connect mode
    
    # Direct connect mode (no front router needed)
    # Map stage_idx -> "host:port" for peer-to-peer setup
    peer_addresses: Optional[Dict[int, str]] = None
    
    # Transport
    transport_mode: TransportMode = TransportMode.ZEROMQ
    device: str = "cuda"
    
    # Buffers
    buffer_size: int = 64 * 1024 * 1024  # 64MB
    num_buffers: int = 4
    
    # QoS & Flow Control
    migration_pacing_mbps: float = 1000.0
    credits_per_class: int = 8
    credit_refill_batch: int = 4
    
    # GPU Fast Path
    enable_gpu_fastpath: bool = True
    mr_cache_size: int = 256
    chunk_size: int = 16 * 1024 * 1024  # 16MB
    overlap_transfer_compute: bool = True
    
    # Shadow connections
    enable_shadow_warmup: bool = True
    shadow_warmup_timeout: float = 5.0
    
    # Congestion control
    enable_congestion_control: bool = True
    target_rtt_us: float = 100.0
    
    # KV Cache (Mooncake TE)
    enable_kv_cache: bool = True
    kv_cache_size_gb: float = 8.0
    
    # Timeouts
    connect_timeout_ms: int = 30000
    recv_timeout_ms: int = -1  # Blocking
    heartbeat_interval_s: float = 1.0
    
    # Dynamic Scaling (DEFAULT MODE)
    enable_dynamic_scaling: bool = True  # Default enabled for production
    scaling_warmup_timeout_s: float = 10.0
    scaling_drain_timeout_s: float = 5.0
    scaling_poll_interval_s: float = 0.1


class EPCV3Endpoint:
    """V3.0 EPC Endpoint with Mobility-Aware Transport Runtime.
    
    Integrates all V3.0 components:
    - Protocol layer with epoch-based versioning
    - Mobility control with shadow connections
    - Multi-queue data plane with credit flow
    - GPU-aware fast path with MR caching
    - Mooncake TE adapter for KV cache
    """
    
    def __init__(self, config: EPCV3Config):
        """Initialize EPC V3.0 endpoint.
        
        Args:
            config: Endpoint configuration
        """
        self.config = config
        self.model_name = config.model_name
        self.stage_idx = config.stage_idx
        self.num_stages = config.num_stages
        self.device = config.device
        
        # Create Logical Endpoint ID
        self.lid = LogicalEndpointID(
            model_name=config.model_name,
            stage_idx=config.stage_idx,
            shard_idx=0,
            replica_idx=0,
        )
        
        # Initialize epoch counter
        self._epoch = 0
        self._epoch_lock = threading.Lock()
        
        # === Layer A: Control Plane ===
        self.directory = DirectoryClient(
            front_router_addr=config.front_router_addr,
            model_name=config.model_name,
            stage_idx=config.stage_idx,
            local_addresses=[f"{config.local_address}:{config.local_port}"],
            transport_caps=[
                DirTransportCapability.UCX_RDMA,
                DirTransportCapability.UCX_TCP,
                DirTransportCapability.ZEROMQ,
            ],
        )
        
        self.legacy_route_table = RouteTable(
            local_lid=self.lid.to_tuple(),
            local_caps=[
                DirTransportCapability.UCX_RDMA,
                DirTransportCapability.UCX_TCP,
                DirTransportCapability.ZEROMQ,
            ],
        )
        
        # === Layer B: Mobility Layer (V3.0) ===
        local_caps = ConnectionCapability(
            supports_rdma=config.transport_mode == TransportMode.UCX_RDMA,
            supports_gpu_direct=config.enable_gpu_fastpath and config.device == "cuda",
            supports_host_staging=True,
            max_message_size=config.buffer_size,
        )
        
        self.versioned_route_table = VersionedRouteTable(
            local_lid=self.lid,
        )
        
        self.shadow_manager = ShadowConnectionManager(
            route_table=self.versioned_route_table,
            on_warmup_complete=self._on_shadow_ready,
            warmup_timeout_s=config.shadow_warmup_timeout,
        )
        
        self.cutover_manager = DualPathCutoverManager(
            route_table=self.versioned_route_table,
            on_cutover_complete=self._on_cutover_complete,
        )
        
        # === Layer C: Data Plane (V3.0) ===
        self.mq_scheduler = MultiQueueScheduler(
            local_lid=self.lid,
            route_table=self.versioned_route_table,
            on_send=self._do_send_v3,
        )
        
        self.stream_multiplexer = StreamMultiplexer(
            local_lid=self.lid,
        )
        
        self.congestion_controller = CongestionController()
        
        # === Layer D: Transport Layer ===
        self.transport = UCXTransport(
            local_address=config.local_address,
            local_port=config.local_port,
            mode=config.transport_mode,
            device=config.device,
            buffer_size=config.buffer_size,
            num_buffers=config.num_buffers,
        )
        
        # GPU Fast Path (V3.0)
        if config.enable_gpu_fastpath:
            self.gpu_fastpath = GPUAwareFastPath(
                local_lid=self.lid,
                cache_max_entries=config.mr_cache_size,
                default_chunk_size=config.chunk_size,
            )
        else:
            self.gpu_fastpath = None
            
        # Capability negotiator (uses auto-detect for local capabilities)
        self.cap_negotiator = CapabilityNegotiator()
        
        # Store local capabilities for reference
        self._local_caps = local_caps
        
        # === KV Cache Integration (Mooncake TE) ===
        if config.enable_kv_cache:
            # Initialize adapter first with required components
            self.mooncake_adapter = ConnexMooncakeAdapter(
                local_lid=self.lid,
                route_table=self.versioned_route_table,
                scheduler=self.mq_scheduler,
                fast_path=self.gpu_fastpath if self.gpu_fastpath else GPUAwareFastPath(self.lid),
            )
            self.kv_manager = KVCacheTransferManager(
                local_lid=self.lid,
                mooncake_adapter=self.mooncake_adapter,
            )
        else:
            self.kv_manager = None
            self.mooncake_adapter = None
        
        # === Dynamic Scaling Manager (V3.0 DEFAULT MODE) ===
        if config.enable_dynamic_scaling:
            self.scaling_manager = DynamicScalingManager(
                endpoint=self,
                on_scale_complete=self._on_scaling_complete,
            )
            # Update warmup timeout from config
            self.scaling_manager.WARMUP_TIMEOUT_S = config.scaling_warmup_timeout_s
            self.scaling_manager.DRAIN_TIMEOUT_S = config.scaling_drain_timeout_s
        else:
            self.scaling_manager = None
            
        # === Legacy Components (for backward compatibility) ===
        self.legacy_cutover_fsm = CutoverFSM(
            stage_idx=config.stage_idx,
            on_state_change=self._on_legacy_cutover_state_change,
        )
        
        self.legacy_qos = QoSScheduler(
            on_send=self._do_send_legacy,
            migration_pacing_mbps=config.migration_pacing_mbps,
        )
        
        self.stream_manager = StreamManager(
            stage_idx=config.stage_idx,
        )
        
        # Connection state
        self._connections: Dict[int, Any] = {}  # stage_idx -> connection
        self._peer_routes: Dict[int, VersionedRoute] = {}  # stage_idx -> route
        self._lock = threading.RLock()
        self._running = False
        
        # Heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "epoch_bumps": 0,
            "stale_epoch_events": 0,
            "cutover_count": 0,
            "credit_updates": 0,
        }
        
        # Timing
        self._init_start_time = time.time()
        self._connection_ready_time = 0.0
        
    @property
    def epoch(self) -> int:
        """Current routing epoch."""
        with self._epoch_lock:
            return self._epoch
    
    def bump_epoch(self) -> int:
        """Increment epoch and return new value."""
        with self._epoch_lock:
            self._epoch += 1
            self._stats["epoch_bumps"] += 1
            # Observe epoch transition in Prometheus
            if _METRICS_AVAILABLE:
                try:
                    vllm_pipeline_epoch_transitions.labels(
                        backend="connex-v3",
                        stage_idx=str(self.stage_idx),
                    ).inc()
                except Exception:
                    pass
            return self._epoch
        
    @property
    def is_first(self) -> bool:
        return self.stage_idx == 0
        
    @property
    def is_last(self) -> bool:
        return self.stage_idx == self.num_stages - 1
        
    @property
    def backend_name(self) -> str:
        return "connex-v3"
        
    def start(self) -> float:
        """Start the EPC V3.0 endpoint.
        
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
        
        # Start schedulers
        self.mq_scheduler.start()
        self.legacy_qos.start()
        
        # Start GPU fast path
        if self.gpu_fastpath:
            self.gpu_fastpath.start()
        
        # Start KV manager
        if self.kv_manager:
            self.kv_manager.start()
        
        # Start dynamic scaling manager (default mode for production)
        if self.scaling_manager:
            self.scaling_manager.start()
            logger.info("Dynamic scaling enabled (default mode)")
        
        # Register with directory
        self.directory.register()
        
        # Start heartbeat
        self._start_heartbeat()
        
        self._connection_ready_time = time.time() - start_time
        
        logger.info(
            "EPC V3.0 endpoint started for stage %d/%d (ready in %.3fs, epoch=%d)",
            self.stage_idx, self.num_stages, self._connection_ready_time, self.epoch
        )
        
        return self._connection_ready_time
        
    def stop(self) -> None:
        """Stop the EPC V3.0 endpoint."""
        self._running = False
        
        # Stop heartbeat
        self._stop_heartbeat()
        
        # Stop schedulers
        self.mq_scheduler.stop()
        self.legacy_qos.stop()
        
        # Stop GPU fast path
        if self.gpu_fastpath:
            self.gpu_fastpath.stop()
            
        # Stop KV manager
        if self.kv_manager:
            self.kv_manager.stop()
        
        # Stop dynamic scaling manager
        if self.scaling_manager:
            self.scaling_manager.stop()
        
        # Stop directory
        self.directory.stop()
        
        # Stop transport
        self.transport.stop_sync()
        
        # Close connections
        with self._lock:
            self._connections.clear()
            self._peer_routes.clear()
            
    def connect(self, target_stage: int) -> bool:
        """Connect to a peer stage with V3.0 protocol.
        
        Supports two modes:
        1. Direct connect mode: Use peer_addresses from config (no front router needed)
        2. Directory mode: Use front router for endpoint discovery
        
        Args:
            target_stage: Target stage index
            
        Returns:
            True if connection successful
        """
        if target_stage in self._connections:
            return True
            
        target_lid = LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=target_stage,
            shard_idx=0,
            replica_idx=0,
        )
        
        # Try direct connect mode first (no front router needed)
        addr = None
        if self.config.peer_addresses and target_stage in self.config.peer_addresses:
            addr = self.config.peer_addresses[target_stage]
            logger.info("Using direct connect mode for stage %d: %s", target_stage, addr)
        else:
            # Fall back to directory lookup
            endpoint = self.directory.lookup(target_lid.to_tuple()) if self.config.front_router_addr else None
        
            if not endpoint and self.config.front_router_addr:
                # Try requesting assignment from front router
                result = self.directory.request_assignment(target_stage, "forward")
                if result:
                    addr, _ = result
            elif endpoint:
                addr = endpoint.primary_address
                    
        if not addr:
            logger.error("Could not find endpoint for stage %d (no peer_addresses and no front_router)", target_stage)
            return False
        
        if ":" not in addr:
            logger.error("Invalid address format for stage %d: %s", target_stage, addr)
            return False
            
        try:
            host, port = addr.rsplit(":", 1)
            conn = self.transport.connect_sync(host, int(port))
            
            # Negotiate capabilities
            peer_caps = self._negotiate_capabilities(conn)
            
            # Create versioned route
            route = VersionedRoute(
                lid=target_lid,
                addresses=[addr],
                capabilities=peer_caps,
                epoch=self.epoch,
                connection=conn,
                connection_state=ConnectionState.CONNECTED,
            )
            
            with self._lock:
                self._connections[target_stage] = conn
                self._peer_routes[target_stage] = route
                
            # Add to versioned route table
            self.versioned_route_table.add_route(
                lid=target_lid,
                addresses=[addr],
                capabilities=peer_caps,
                epoch=self.epoch,
            )
            
            # Initialize credits for peer
            peer_state = self.mq_scheduler.get_or_create_peer(target_lid)
            for mc in MessageClass:
                peer_state.credits.replenish(mc, self.config.credits_per_class)
            
            logger.info(
                "Connected to stage %d at %s (epoch=%d, caps=%s)",
                target_stage, addr, self.epoch, peer_caps.to_dict()
            )
            return True
            
        except Exception as e:
            logger.error("Failed to connect to stage %d: %s", target_stage, e)
            return False
            
    def _negotiate_capabilities(self, conn: Any) -> ConnectionCapability:
        """Negotiate transport capabilities with peer.
        
        Args:
            conn: Connection handle
            
        Returns:
            Negotiated capabilities
        """
        # TODO: Implement actual capability exchange over connection
        # For now, return default capabilities based on local config
        return ConnectionCapability(
            supports_rdma=self.config.transport_mode == TransportMode.UCX_RDMA,
            supports_gpu_direct=self.config.enable_gpu_fastpath and self.config.device == "cuda",
            supports_host_staging=True,
            max_message_size=self.config.buffer_size,
        )
    
    # ========== V3.0 Send/Recv API ==========
    
    def send_tensor_dict_v3(
        self,
        tensor_dict: Dict[str, Any],
        dst: Optional[int] = None,
        stream_id: Optional[StreamID] = None,
        msg_class: MessageClass = MessageClass.TOKEN_CRITICAL,
    ) -> bool:
        """Send tensor dictionary using V3.0 protocol.
        
        Uses:
        - Protocol headers with epoch versioning
        - Multi-queue scheduler with credit backpressure
        - GPU fast path for tensor transfer
        
        Args:
            tensor_dict: Dictionary of tensors and metadata
            dst: Destination stage (default: next stage)
            stream_id: Stream identifier
            msg_class: Message priority class
            
        Returns:
            True if enqueued successfully
        """
        if dst is None:
            dst = self.stage_idx + 1
            
        if dst >= self.num_stages:
            logger.warning("Cannot send beyond last stage")
            return False
            
        # Ensure connected
        if dst not in self._connections:
            if not self.connect(dst):
                return False
                
        # Get route and validate epoch
        route = self._peer_routes.get(dst)
        if route and route.epoch < self.epoch:
            # Route is stale, need refresh
            logger.warning("Stale route for stage %d (epoch %d < %d)", 
                          dst, route.epoch, self.epoch)
            self._stats["stale_epoch_events"] += 1
            if _METRICS_AVAILABLE:
                try:
                    vllm_pipeline_stale_epoch_events.labels(
                        backend="connex-v3",
                        stage_idx=str(self.stage_idx),
                    ).inc()
                except Exception:
                    pass
            
        # Create stream ID if not provided
        if stream_id is None:
            request_id = str(tensor_dict.get("request_id", 0))
            stream_id = StreamID(request_id=request_id, direction="forward", token_step=0)
        
        # Get next sequence number
        seqno = self.stream_multiplexer.allocate_seqno(stream_id)
        
        # Serialize tensor dict
        serialized_data, tensor_meta = self._serialize_tensors(tensor_dict)
        
        # Create protocol header
        target_lid = LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=dst,
        )
        
        header = ProtocolHeader(
            src_lid=self.lid,
            dst_lid=target_lid,
            epoch=self.epoch,
            stream_id=stream_id,
            seqno=seqno,
            msg_class=msg_class,
            msg_type=MessageType.DATA,
        )
        
        # Create mobility message
        msg = MobilityMessage(
            header=header,
            payload=serialized_data,
            tensor_meta=tensor_meta,
        )
        
        # Use GPU fast path for tensor transfer if available
        if self.gpu_fastpath and any(
            isinstance(v, torch.Tensor) and v.is_cuda 
            for v in tensor_dict.values()
        ):
            return self._send_via_fastpath(msg, dst, route)
        else:
            return self._send_via_scheduler(msg, dst, msg_class)
    
    def _serialize_tensors(
        self, 
        tensor_dict: Dict[str, Any]
    ) -> Tuple[bytes, Dict[str, dict]]:
        """Serialize tensor dictionary.
        
        Args:
            tensor_dict: Input tensors and metadata
            
        Returns:
            (serialized_data, tensor_metadata)
        """
        serialized = {}
        tensor_meta = {}
        
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_meta[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "device": str(value.device),
                    "is_cuda": value.is_cuda,
                }
                # For non-GPU-direct transfers, copy to CPU
                if value.is_cuda:
                    value = value.cpu()
                serialized[key] = value.numpy().tobytes()
            else:
                serialized[key] = value
                
        data = msgpack.packb({
            "data": serialized,
            "meta": tensor_meta,
            "src_stage": self.stage_idx,
            "epoch": self.epoch,
            "timestamp": time.time(),
        }, use_bin_type=True)
        
        return data, tensor_meta
    
    def _send_via_fastpath(
        self,
        msg: MobilityMessage,
        dst: int,
        route: Optional[VersionedRoute],
    ) -> bool:
        """Send via GPU fast path.
        
        Uses prepare_send for registration caching, then falls back to scheduler.
        
        Args:
            msg: Message to send
            dst: Destination stage
            route: Route entry
            
        Returns:
            True if sent successfully
        """
        if not self.gpu_fastpath or not route:
            return self._send_via_scheduler(
                msg, dst, msg.header.msg_class
            )
            
        try:
            # For GPU tensors, prepare registration (caching benefit)
            # Then still send via scheduler for now
            # Future: implement direct GPU-to-GPU transfer
            
            self._stats["messages_sent"] += 1
            self._stats["bytes_sent"] += len(msg.payload) if msg.payload else 0
            
            # Currently fall back to scheduler-based sending
            # GPU fast path prepares registration cache for efficiency
            return self._send_via_scheduler(
                msg, dst, msg.header.msg_class
            )
            
        except Exception as e:
            logger.error("GPU fast path send failed: %s", e)
            # Fallback to scheduler
            return self._send_via_scheduler(
                msg, dst, msg.header.msg_class
            )
    
    def _send_via_scheduler(
        self,
        msg: MobilityMessage,
        dst: int,
        msg_class: MessageClass,
    ) -> bool:
        """Send via multi-queue scheduler.
        
        Args:
            msg: Message to send
            dst: Destination stage
            msg_class: Priority class
            
        Returns:
            True if enqueued
        """
        target_lid = LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=dst,
        )
        
        # Create StreamID from message header if available
        stream_id = msg.header.stream_id if hasattr(msg.header, 'stream_id') and msg.header.stream_id else StreamID(request_id="0", direction="forward", token_step=0)
        
        return self.mq_scheduler.enqueue(
            peer_lid=target_lid,
            message=msg,
            stream_id=stream_id,
            priority=msg_class,
        )
    
    def _do_send_v3(self, queued_msg: Any, route: Any = None) -> bool:
        """Internal send callback for V3.0 scheduler.
        
        Args:
            queued_msg: QueuedMessage from scheduler
            route: Optional route (from scheduler)
            
        Returns:
            True if sent successfully
        """
        try:
            msg = queued_msg.message
            dst_lid = msg.header.dst_lid
            dst = dst_lid.stage_idx
            conn = self._connections.get(dst)
            
            if not conn:
                return False
                
            # Serialize full message
            data = msg.serialize()
            self.transport.send_sync(conn, data)
            
            self._stats["messages_sent"] += 1
            self._stats["bytes_sent"] += len(data)
            
            return True
            
        except Exception as e:
            logger.error("V3 send failed: %s", e)
            return False
    
    def recv_tensor_dict_v3(
        self,
        src: Optional[int] = None,
        timeout_ms: int = -1,
    ) -> Optional[Tuple[Dict[str, Any], ProtocolHeader]]:
        """Receive tensor dictionary using V3.0 protocol.
        
        Returns message with protocol header for epoch validation.
        
        Args:
            src: Source stage (default: previous stage)
            timeout_ms: Receive timeout
            
        Returns:
            (tensor_dict, header) or None if timeout
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
            
        # Deserialize mobility message
        try:
            msg = MobilityMessage.deserialize(data)
        except Exception as e:
            # Fallback to legacy format
            logger.debug("Failed to parse V3 message, trying legacy: %s", e)
            return self._recv_legacy(data)
            
        header = msg.header
        
        # Validate epoch
        if header.epoch < self.epoch:
            # Stale epoch - send feedback
            logger.warning(
                "Received stale epoch message (got %d, current %d)",
                header.epoch, self.epoch
            )
            self._stats["stale_epoch_events"] += 1
            if _METRICS_AVAILABLE:
                try:
                    vllm_pipeline_stale_epoch_events.labels(
                        backend="connex-v3",
                        stage_idx=str(self.stage_idx),
                    ).inc()
                except Exception:
                    pass
            self._send_stale_epoch_feedback(header.src_lid, header.epoch)
        
        # Check for duplicates via stream multiplexer
        if self.stream_multiplexer.is_duplicate(header.stream_id, header.seqno):
            logger.debug("Dropping duplicate seqno %d for stream %s", 
                        header.seqno, header.stream_id)
            return self.recv_tensor_dict_v3(src, timeout_ms)  # Try next message
            
        # Mark as received
        self.stream_multiplexer.mark_received(header.stream_id, header.seqno)
        
        # Reconstruct tensors
        tensor_dict = self._deserialize_tensors(msg.payload, msg.tensor_meta)
        
        self._stats["messages_received"] += 1
        self._stats["bytes_received"] += len(data)
        
        return tensor_dict, header
    
    def _deserialize_tensors(
        self,
        data: bytes,
        tensor_meta: Dict[str, dict],
    ) -> Dict[str, Any]:
        """Deserialize tensor dictionary.
        
        Args:
            data: Serialized data
            tensor_meta: Tensor metadata
            
        Returns:
            Reconstructed tensor dictionary
        """
        import numpy as np
        
        unpacked = msgpack.unpackb(data, raw=False)
        result = {}
        
        for key, value in unpacked["data"].items():
            if key in tensor_meta:
                meta = tensor_meta[key]
                dtype_str = meta["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                shape = tuple(meta["shape"])
                
                arr = np.frombuffer(value, dtype=torch.tensor([], dtype=dtype).numpy().dtype)
                arr = arr.reshape(shape)
                tensor = torch.from_numpy(arr.copy())
                
                if self.device == "cuda":
                    tensor = tensor.cuda()
                    
                result[key] = tensor
            else:
                result[key] = value
                
        return result
    
    def _recv_legacy(self, data: bytes) -> Optional[Tuple[Dict[str, Any], None]]:
        """Fallback to legacy message format.
        
        Args:
            data: Raw received data
            
        Returns:
            (tensor_dict, None) for legacy messages
        """
        try:
            message = msgpack.unpackb(data, raw=False)
            
            if message.get("type") != "tensor_dict":
                return None
                
            import numpy as np
            result = {}
            tensor_meta = message.get("tensor_meta", {})
            
            for key, value in message["data"].items():
                if key in tensor_meta:
                    meta = tensor_meta[key]
                    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
                    shape = tuple(meta["shape"])
                    
                    arr = np.frombuffer(value, dtype=torch.tensor([], dtype=dtype).numpy().dtype)
                    arr = arr.reshape(shape)
                    tensor = torch.from_numpy(arr.copy())
                    
                    if self.device == "cuda":
                        tensor = tensor.cuda()
                        
                    result[key] = tensor
                else:
                    result[key] = value
                    
            return result, None
            
        except Exception as e:
            logger.error("Legacy recv failed: %s", e)
            return None
    
    # ========== Credit-Based Backpressure ==========
    
    def update_credits(
        self,
        peer_stage: int,
        msg_class: MessageClass,
        credits: int,
    ) -> None:
        """Update credits for a peer.
        
        Args:
            peer_stage: Peer stage index
            msg_class: Message class
            credits: Credits to add
        """
        peer_lid = LogicalEndpointID(
            model_name=self.model_name,
            stage_idx=peer_stage,
        )
        
        self.mq_scheduler.process_credits(
            peer_id=peer_lid,
            credits_update={msg_class: credits},
        )
        
        self._stats["credit_updates"] += 1
        
    def send_credit_update(
        self,
        dst: int,
        credits: Dict[MessageClass, int],
    ) -> bool:
        """Send credit update to peer.
        
        Args:
            dst: Destination stage
            credits: Credits per class
            
        Returns:
            True if sent
        """
        credit_info = CreditInfo(credits_available=credits)
        msg = create_credit_update(
            src_lid=self.lid,
            dst_lid=LogicalEndpointID(self.model_name, dst),
            credit_info=credit_info,
        )
        
        return self._send_control_message(dst, msg)
    
    # ========== Epoch & Stale Route Feedback ==========
    
    def _send_stale_epoch_feedback(
        self,
        peer_lid: LogicalEndpointID,
        stale_epoch: int,
    ) -> None:
        """Send stale epoch feedback for fast route refresh.
        
        Args:
            peer_lid: Peer LID
            stale_epoch: The stale epoch observed
        """
        msg = create_stale_epoch_feedback(
            src_lid=self.lid,
            dst_lid=peer_lid,
            stale_epoch=stale_epoch,
            current_epoch=self.epoch,
        )
        
        self._send_control_message(peer_lid.stage_idx, msg)
        
    def handle_stale_epoch_feedback(
        self,
        feedback: MobilityMessage,
    ) -> None:
        """Handle stale epoch feedback from peer.
        
        Triggers route refresh or epoch bump.
        
        Args:
            feedback: Stale epoch feedback message
        """
        peer_epoch = feedback.header.epoch
        
        if peer_epoch > self.epoch:
            # We're behind - update our epoch
            with self._epoch_lock:
                old_epoch = self._epoch
                self._epoch = peer_epoch
                
            logger.info(
                "Updated epoch from %d to %d based on peer feedback",
                old_epoch, peer_epoch
            )
            
            # Refresh routes
            self._refresh_routes()
    
    def _refresh_routes(self) -> None:
        """Refresh routes after epoch change."""
        with self._lock:
            for stage, route in self._peer_routes.items():
                route.epoch = self.epoch
                self.versioned_route_table.update_route_epoch(route.lid, self.epoch)
    
    # ========== Migration & Cutover ==========
    
    def prepare_migration(
        self,
        stream_id: StreamID,
        new_address: str,
        new_epoch: int,
    ) -> bool:
        """Prepare for endpoint migration with V3.0 protocol.
        
        Uses shadow connection warmup before cutover.
        
        Args:
            stream_id: Stream to migrate
            new_address: New endpoint address  
            new_epoch: Target epoch
            
        Returns:
            True if preparation started
        """
        old_addr = f"{self.config.local_address}:{self.config.local_port}"
        
        # Start shadow connection warmup
        if self.config.enable_shadow_warmup:
            shadow_route = self.shadow_manager.create_shadow(
                lid=self.lid,
                new_address=new_address,
                new_epoch=new_epoch,
            )
            
            # Begin warmup
            self.shadow_manager.start_warmup(shadow_route)
            
        # Notify directory
        success = self.directory.notify_migration(
            new_addresses=[new_address],
            target_epoch=new_epoch,
        )
        
        return success
        
    def execute_cutover(self, stream_id: StreamID) -> bool:
        """Execute traffic cutover with dual-path support.
        
        Args:
            stream_id: Stream to cutover
            
        Returns:
            True if cutover successful
        """
        # Get current sequence number for drain tracking
        last_seqno = self.stream_multiplexer.get_current_seqno(stream_id)
        
        # Start dual-path cutover
        result = self.cutover_manager.start_cutover(
            stream_id=stream_id,
            last_old_seqno=last_seqno,
        )
        
        if result:
            # Bump epoch
            new_epoch = self.bump_epoch()
            
            # Update routes
            self._refresh_routes()
            
            self._stats["cutover_count"] += 1
            
        return result
    
    def _on_shadow_ready(self, route: VersionedRoute) -> None:
        """Callback when shadow connection warmup completes."""
        logger.info(
            "Shadow connection ready for %s (epoch=%d)",
            route.lid, route.epoch
        )
        route.warmup_complete = True
        route.connection_state = ConnectionState.READY
        
    def _on_cutover_complete(self, stream_id: StreamID) -> None:
        """Callback when cutover completes."""
        logger.info("Cutover complete for stream %s", stream_id)
        
    def _on_scaling_complete(self, event: ScalingEvent) -> None:
        """Callback when dynamic scaling operation completes."""
        logger.info(
            "Scaling event complete: %s for stage %d (epoch=%d, latency=%.2fms)",
            event.event_type, event.target_stage, event.new_epoch,
            (time.time() - event.timestamp) * 1000
        )
    
    # ========== Dynamic Scaling Public API ==========
    
    def trigger_scale_up(
        self,
        target_stage: int,
        new_address: str,
        replica_idx: int = 0,
    ) -> bool:
        """Manually trigger scale-up operation.
        
        Use this for testing or external orchestrator integration.
        In production, scale events come from directory service.
        
        Args:
            target_stage: Stage to add replica
            new_address: Address of new replica
            replica_idx: Replica index
            
        Returns:
            True if scale-up initiated
        """
        if not self.scaling_manager:
            logger.warning("Dynamic scaling not enabled")
            return False
            
        return self.scaling_manager.initiate_scale_up(
            target_stage=target_stage,
            new_address=new_address,
            new_replica_idx=replica_idx,
        )
        
    def trigger_scale_down(
        self,
        target_stage: int,
        replica_idx: int = 0,
    ) -> bool:
        """Manually trigger scale-down operation.
        
        Args:
            target_stage: Stage to remove replica from
            replica_idx: Replica index to remove
            
        Returns:
            True if scale-down initiated
        """
        if not self.scaling_manager:
            logger.warning("Dynamic scaling not enabled")
            return False
            
        return self.scaling_manager.initiate_scale_down(
            target_stage=target_stage,
            replica_idx=replica_idx,
        )
        
    def trigger_migration(
        self,
        target_stage: int,
        old_address: str,
        new_address: str,
    ) -> bool:
        """Manually trigger migration operation.
        
        Args:
            target_stage: Stage to migrate
            old_address: Current address
            new_address: New address
            
        Returns:
            True if migration initiated
        """
        if not self.scaling_manager:
            logger.warning("Dynamic scaling not enabled")
            return False
            
        return self.scaling_manager.initiate_migration(
            target_stage=target_stage,
            old_address=old_address,
            new_address=new_address,
        )
        
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get dynamic scaling statistics.
        
        Returns:
            Scaling statistics including reconfiguration latencies
        """
        if not self.scaling_manager:
            return {"enabled": False}
            
        stats = self.scaling_manager.get_stats()
        stats["enabled"] = True
        return stats
    
    # ========== KV Cache Transfer (Mooncake TE) ==========
    
    def register_kv_buffer(
        self,
        buffer: torch.Tensor,
        buffer_id: str,
    ) -> bool:
        """Register KV cache buffer for transfer.
        
        Args:
            buffer: GPU tensor for KV cache
            buffer_id: Unique buffer identifier
            
        Returns:
            True if registered
        """
        if not self.kv_manager:
            return False
            
        return self.kv_manager.register_buffer(buffer, buffer_id)
    
    def send_kv_cache(
        self,
        request_id: int,
        layer_idx: int,
        block: KVCacheBlock,
        dst: int,
    ) -> KVTransferRequest:
        """Send KV cache block via Mooncake TE interface.
        
        Args:
            request_id: Request identifier
            layer_idx: Transformer layer index
            block: KV cache block
            dst: Destination stage
            
        Returns:
            Transfer request handle
        """
        if not self.mooncake_adapter:
            raise RuntimeError("KV cache not enabled")
            
        return self.mooncake_adapter.send_kv_cache(
            request_id=request_id,
            layer_idx=layer_idx,
            block=block,
            dst_lid=LogicalEndpointID(self.model_name, dst),
        )
    
    def recv_kv_cache(
        self,
        request_id: int,
        layer_idx: int,
        timeout_ms: int = -1,
    ) -> Optional[KVCacheBlock]:
        """Receive KV cache block.
        
        Args:
            request_id: Request identifier
            layer_idx: Layer index
            timeout_ms: Timeout
            
        Returns:
            KV cache block or None
        """
        if not self.mooncake_adapter:
            raise RuntimeError("KV cache not enabled")
            
        return self.mooncake_adapter.recv_kv_cache(
            request_id=request_id,
            layer_idx=layer_idx,
            timeout_ms=timeout_ms,
        )
    
    # ========== Heartbeat ==========
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat thread."""
        if self._heartbeat_thread:
            return
            
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()
        
    def _stop_heartbeat(self) -> None:
        """Stop heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
            
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop."""
        interval = self.config.heartbeat_interval_s
        
        while not self._heartbeat_stop.wait(timeout=interval):
            with self._lock:
                for stage in list(self._connections.keys()):
                    try:
                        self._send_heartbeat(stage)
                    except Exception as e:
                        logger.warning("Heartbeat to stage %d failed: %s", stage, e)
                        
    def _send_heartbeat(self, dst: int) -> None:
        """Send heartbeat to peer."""
        msg = create_heartbeat(
            src_lid=self.lid,
            dst_lid=LogicalEndpointID(self.model_name, dst),
            epoch=self.epoch,
        )
        self._send_control_message(dst, msg)
        
    def _send_control_message(self, dst: int, msg: MobilityMessage) -> bool:
        """Send control message."""
        try:
            conn = self._connections.get(dst)
            if conn:
                data = msg.serialize()
                self.transport.send_sync(conn, data)
                return True
        except Exception as e:
            logger.error("Control message to %d failed: %s", dst, e)
        return False
    
    # ========== Legacy API (Backward Compatibility) ==========
    
    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
        dst: Optional[int] = None,
    ) -> None:
        """Legacy send API - delegates to V3.0.
        
        Args:
            tensor_dict: Tensor dictionary
            dst: Destination stage
        """
        self.send_tensor_dict_v3(
            tensor_dict=tensor_dict,
            dst=dst,
            msg_class=MessageClass.TOKEN_CRITICAL,
        )
        
    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        timeout_ms: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """Legacy recv API - delegates to V3.0.
        
        Args:
            src: Source stage
            timeout_ms: Timeout
            
        Returns:
            Tensor dictionary or None
        """
        result = self.recv_tensor_dict_v3(src, timeout_ms)
        
        if result is None:
            return None
            
        tensor_dict, _ = result
        return tensor_dict
    
    def _do_send_legacy(self, msg: Any) -> bool:
        """Legacy send callback."""
        return self._do_send_v3(msg)
        
    def _on_legacy_cutover_state_change(
        self,
        state: CutoverState,
        ctx: CutoverContext,
    ) -> None:
        """Handle legacy cutover state changes."""
        logger.debug("Legacy cutover state: %s", state.name)
        
    def _on_send_complete(self, dst: int) -> None:
        """Callback when send completes."""
        # Refill credits
        self.send_credit_update(
            dst=dst,
            credits={MessageClass.TOKEN_CRITICAL: self.config.credit_refill_batch},
        )
    
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive endpoint statistics."""
        stats = {
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "backend": self.backend_name,
            "version": "3.0.0",
            "epoch": self.epoch,
            "connection_ready_time": self._connection_ready_time,
            "connections": len(self._connections),
            "counters": dict(self._stats),
        }
        
        # Component stats (use safe accessor pattern)
        if hasattr(self.transport, 'get_stats'):
            stats["transport"] = self.transport.get_stats()
        
        if hasattr(self.mq_scheduler, 'get_queue_stats'):
            stats["mq_scheduler"] = self.mq_scheduler.get_queue_stats()
        elif hasattr(self.mq_scheduler, 'get_stats'):
            stats["mq_scheduler"] = self.mq_scheduler.get_stats()
        
        if hasattr(self.versioned_route_table, 'get_stats'):
            stats["versioned_routes"] = self.versioned_route_table.get_stats()
        else:
            # Fallback - provide basic route info
            with self._lock:
                stats["versioned_routes"] = {
                    "route_count": len(self._peer_routes),
                    "epoch": self.epoch,
                }
        
        if hasattr(self.stream_multiplexer, 'get_stats'):
            stats["stream_multiplexer"] = self.stream_multiplexer.get_stats()
        else:
            stats["stream_multiplexer"] = self.stream_multiplexer._stats.copy()
        
        # CongestionController doesn't have get_stats, use default
        stats["congestion"] = {
            "enabled": True,
            "target_delay_us": CongestionController.TARGET_DELAY_US,
        }
        
        if self.gpu_fastpath and hasattr(self.gpu_fastpath, 'get_stats'):
            stats["gpu_fastpath"] = self.gpu_fastpath.get_stats()
        elif self.gpu_fastpath:
            stats["gpu_fastpath"] = self.gpu_fastpath._stats.copy()
            
        if self.kv_manager and hasattr(self.kv_manager, 'get_stats'):
            stats["kv_cache"] = self.kv_manager.get_stats()
        elif self.kv_manager:
            stats["kv_cache"] = self.kv_manager._stats.copy()
        
        # Dynamic scaling stats (V3.0)
        if self.scaling_manager:
            stats["dynamic_scaling"] = self.scaling_manager.get_stats()
        else:
            stats["dynamic_scaling"] = {"enabled": False}
            
        return stats


# ========== Convenience Factory Functions ==========

def create_v3_endpoint(
    model_name: str,
    stage_idx: int,
    num_stages: int,
    front_router_addr: str,
    local_address: str = "",
    local_port: int = 0,
    device: str = "cuda",
    **kwargs,
) -> EPCV3Endpoint:
    """Create an EPC V3.0 endpoint with default configuration.
    
    Args:
        model_name: Model name
        stage_idx: Pipeline stage index
        num_stages: Total pipeline stages
        front_router_addr: Front router address
        local_address: Local bind address
        local_port: Local bind port  
        device: Device for tensors
        **kwargs: Additional config options
        
    Returns:
        Configured EPCV3Endpoint
    """
    config = EPCV3Config(
        model_name=model_name,
        stage_idx=stage_idx,
        num_stages=num_stages,
        local_address=local_address or "0.0.0.0",
        local_port=local_port or (15550 + stage_idx),
        front_router_addr=front_router_addr,
        device=device,
        **kwargs,
    )
    
    return EPCV3Endpoint(config)


def create_direct_connect_endpoint(
    model_name: str,
    stage_idx: int,
    num_stages: int,
    peer_addresses: Dict[int, str],
    local_address: str = "0.0.0.0",
    local_port: int = 0,
    device: str = "cuda",
    **kwargs,
) -> EPCV3Endpoint:
    """Create endpoint with direct connect mode (no front router needed).
    
    Ideal for debugging and simple setups.
    
    Args:
        model_name: Model name
        stage_idx: Pipeline stage index
        num_stages: Total pipeline stages
        peer_addresses: Map of stage_idx -> "host:port"
        local_address: Local bind address
        local_port: Local bind port
        device: Device for tensors
        **kwargs: Additional config options
        
    Returns:
        Configured EPCV3Endpoint
        
    Example:
        # 2-stage pipeline setup
        # Stage 0:
        ep0 = create_direct_connect_endpoint(
            model_name="llama",
            stage_idx=0,
            num_stages=2,
            peer_addresses={1: "10.0.0.2:15551"},
            local_port=15550,
        )
        
        # Stage 1:
        ep1 = create_direct_connect_endpoint(
            model_name="llama",
            stage_idx=1,
            num_stages=2,
            peer_addresses={0: "10.0.0.1:15550"},
            local_port=15551,
        )
    """
    config = EPCV3Config(
        model_name=model_name,
        stage_idx=stage_idx,
        num_stages=num_stages,
        local_address=local_address,
        local_port=local_port or (15550 + stage_idx),
        peer_addresses=peer_addresses,
        front_router_addr="",  # Not needed in direct connect mode
        device=device,
        **kwargs,
    )
    
    return EPCV3Endpoint(config)


def create_endpoint_with_mooncake(
    model_name: str,
    stage_idx: int,
    num_stages: int,
    front_router_addr: str,
    kv_cache_size_gb: float = 8.0,
    **kwargs,
) -> EPCV3Endpoint:
    """Create endpoint with Mooncake TE KV cache enabled.
    
    Args:
        model_name: Model name
        stage_idx: Stage index
        num_stages: Total stages
        front_router_addr: Front router address
        kv_cache_size_gb: KV cache size in GB
        **kwargs: Additional options
        
    Returns:
        Configured endpoint with KV cache
    """
    config = EPCV3Config(
        model_name=model_name,
        stage_idx=stage_idx,
        num_stages=num_stages,
        front_router_addr=front_router_addr,
        enable_kv_cache=True,
        kv_cache_size_gb=kv_cache_size_gb,
        **kwargs,
    )
    
    return EPCV3Endpoint(config)
