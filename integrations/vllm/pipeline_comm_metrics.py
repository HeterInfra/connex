# SPDX-License-Identifier: Apache-2.0
"""
Prometheus metrics for pipeline parallelism communication backends.

This module provides detailed metrics for monitoring communication performance
across all pipeline parallelism backends (ZeroMQ, NCCL-Reconfig, Connex, NIXL, Mooncake).
All metrics include a 'backend' label to distinguish between different communication backends.
"""

import os

from prometheus_client import Counter, Gauge, Histogram, REGISTRY

# Check if multiprocess mode is enabled
_multiprocess_mode = os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None

# ============================================================================
# Generic Pipeline Communication Metrics (with backend label)
# ============================================================================

# Data volume metrics (bytes)
vllm_pipeline_comm_bytes_sent = Counter(
    "vllm:pipeline_comm_bytes_sent_total",
    "Total bytes sent via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],  # direction: forward/return
    registry=REGISTRY
)

vllm_pipeline_comm_bytes_received = Counter(
    "vllm:pipeline_comm_bytes_received_total",
    "Total bytes received via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],  # direction: forward/return
    registry=REGISTRY
)

# Message count metrics
vllm_pipeline_comm_messages_sent = Counter(
    "vllm:pipeline_comm_messages_sent_total",
    "Total messages sent via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],
    registry=REGISTRY
)

vllm_pipeline_comm_messages_received = Counter(
    "vllm:pipeline_comm_messages_received_total",
    "Total messages received via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],
    registry=REGISTRY
)

# Latency metrics (seconds)
# Total send latency (serialization + enqueue/transfer + sync)
vllm_pipeline_comm_send_latency = Histogram(
    "vllm:pipeline_comm_send_latency_seconds",
    "Total time spent sending data via pipeline communication backend (includes serialization + transfer)",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

vllm_pipeline_comm_recv_latency = Histogram(
    "vllm:pipeline_comm_recv_latency_seconds",
    "Total time spent receiving data via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# ============================================================================
# Three-Level Latency Breakdown (for understanding async vs sync communication)
# ============================================================================

# Level 1: Enqueue Latency - how fast send() returns (critical for scheduling)
# This measures: buffer allocation + enqueue to transfer engine + network send
# For async backends (Mooncake/NIXL): very fast (just enqueue metadata)
# For sync backends (NCCL with sync): includes full transfer time
# v1.5.6: Does NOT include serialization time
vllm_pipeline_comm_enqueue_latency = Histogram(
    "vllm:pipeline_comm_enqueue_latency_seconds",
    "Time for send() call to return after serialization (enqueue latency). Fast for async backends, slow for sync backends.",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

# Level 2: Delivery Latency - when receiver can recv() the message (E2E critical)
# v1.5.6: Measured from send_timestamp (just before network send) to data_received_time
# This is the ACTUAL network transfer latency, excluding serialization on sender
# and deserialization/GPU transfer on receiver
vllm_pipeline_comm_delivery_latency = Histogram(
    "vllm:pipeline_comm_delivery_latency_seconds",
    "Network delivery latency: time from send start to data fully received (excludes serialization/deserialization).",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Receiver-local delivery latency (robust to cross-node clock skew).
# This measures: time from receiver getting metadata/signal to data fully received
# (before deserialization/GPU transfer). Useful when sender/receiver clocks are not
# tightly synchronized (NTP/PTP drift can make the sender-timestamp metric negative).
vllm_pipeline_comm_delivery_latency_local = Histogram(
    "vllm:pipeline_comm_delivery_latency_local_seconds",
    "Receiver-local delivery latency: time from receiver signal/metadata receipt to data fully received (excludes deserialization/GPU transfer). Robust to clock skew.",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Level 3: Ready-to-Use Latency - when GPU tensor is ready for next kernel (pipeline critical)
# v1.5.6: Measures ONLY the CPU->GPU tensor transfer time (after deserialization)
# Previous versions incorrectly included deserialization time
vllm_pipeline_comm_ready_to_use_latency = Histogram(
    "vllm:pipeline_comm_ready_to_use_latency_seconds",
    "GPU transfer time: time for deserialized tensor to be moved to GPU (ready for computation).",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY
)

# Transfer engine specific metrics
vllm_pipeline_comm_nccl_sync_latency = Histogram(
    "vllm:pipeline_comm_nccl_sync_latency_seconds",
    "Time spent in NCCL stream.synchronize() - measures actual GPU transfer time",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY
)

vllm_pipeline_comm_buffer_copy_latency = Histogram(
    "vllm:pipeline_comm_buffer_copy_latency_seconds",
    "Time spent copying data to/from transfer buffers (CPU<->GPU, buffer allocation)",
    labelnames=["backend", "stage_idx", "operation"],  # operation: alloc/copy_to/copy_from
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=REGISTRY
)

# Tensor size metrics
vllm_pipeline_comm_tensor_size = Histogram(
    "vllm:pipeline_comm_tensor_size_bytes",
    "Size of individual tensors sent via pipeline communication backend",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[
        1024,           # 1 KB
        10240,          # 10 KB
        102400,         # 100 KB
        1048576,        # 1 MB
        10485760,       # 10 MB
        104857600,      # 100 MB
        1073741824,     # 1 GB
    ],
    registry=REGISTRY
)

vllm_pipeline_comm_tensor_count = Histogram(
    "vllm:pipeline_comm_tensor_count",
    "Number of tensors per message (pipeline parallelism)",
    labelnames=["backend", "stage_idx", "direction"],
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
    registry=REGISTRY
)

# Serialization/deserialization time
vllm_pipeline_comm_serialization_latency = Histogram(
    "vllm:pipeline_comm_serialization_latency_seconds",
    "Time spent serializing tensors (pipeline parallelism)",
    labelnames=["backend", "stage_idx"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

vllm_pipeline_comm_deserialization_latency = Histogram(
    "vllm:pipeline_comm_deserialization_latency_seconds",
    "Time spent deserializing tensors (pipeline parallelism)",
    labelnames=["backend", "stage_idx"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

# Error metrics
vllm_pipeline_comm_errors = Counter(
    "vllm:pipeline_comm_errors_total",
    "Total pipeline communication errors",
    labelnames=["backend", "stage_idx", "error_type"],  # error_type: timeout/serialization/network
    registry=REGISTRY
)

# Current state metrics
vllm_pipeline_comm_connection_status = Gauge(
    "vllm:pipeline_comm_connection_status",
    "Pipeline communication connection status (1=connected, 0=disconnected)",
    labelnames=["backend", "stage_idx", "socket_type"],  # socket_type: push/pull/return_push/return_pull
    registry=REGISTRY,
    multiprocess_mode='livesum'  # Gauge in multiprocess mode needs a mode
)

# Throughput metrics (derived from counters, but useful for dashboards)
vllm_pipeline_comm_throughput_mbps = Gauge(
    "vllm:pipeline_comm_throughput_mbps",
    "Estimated pipeline communication throughput in MB/s",
    labelnames=["backend", "stage_idx", "direction"],
    registry=REGISTRY,
    multiprocess_mode='livesum'  # Gauge in multiprocess mode needs a mode
)

# ============================================================================
# Connex UCX transport diagnostics (helps validate RDMA vs TCP fallback)
# ============================================================================

vllm_pipeline_ucx_transport_info = Gauge(
    "vllm:pipeline_ucx_transport_info",
    "Connex UCX transport configuration (mode/TLS/NET_DEVICES). Value is always 1 for current config.",
    labelnames=["backend", "stage_idx", "mode", "tls", "net_devices"],
    registry=REGISTRY,
    multiprocess_mode="livesum",
)

# ============================================================================
# Pipeline Mode Request-Level Latency Metrics (TTFT/TPOT for external pipeline)
# ============================================================================
# These metrics are recorded at stage-0 from request-level timing stats.
# They are intended to be comparable across pipeline communication backends
# via the shared `backend` label.

# Time to First Token (TTFT) - from request arrival to first token generated
vllm_pipeline_ttft = Histogram(
    "vllm:pipeline_time_to_first_token_seconds",
    "Time from request arrival to first token (external pipeline mode). "
    "Recorded at stage-0 from request-level timing stats.",
    labelnames=["backend", "model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    registry=REGISTRY
)

# Time Per Output Token (TPOT) - mean time per output token (decode-only).
# Excludes the token generated by the prefill phase, matching vLLM's
# `mean_time_per_output_token` definition.
vllm_pipeline_tpot = Histogram(
    "vllm:pipeline_time_per_output_token_seconds",
    "Mean time per output token in external pipeline mode (decode-only). "
    "Recorded at stage-0 from finished request stats.",
    labelnames=["backend", "model_name"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    registry=REGISTRY
)

# E2E Request Latency - total time from arrival to completion
vllm_pipeline_e2e_latency = Histogram(
    "vllm:pipeline_e2e_request_latency_seconds",
    "End-to-end request latency in external pipeline mode. "
    "Recorded at stage-0 from request arrival to completion.",
    labelnames=["backend", "model_name"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0],
    registry=REGISTRY
)
# ============================================================================
# Connection Establishment / Ready Time Metrics
# ============================================================================
# These metrics measure how long it takes for a communication backend to be
# fully ready after initialization. This is critical for endpoint mobility
# scenarios where connection setup overhead impacts overall latency.

# Connection Ready Time - total time from communicator init to fully connected
vllm_pipeline_connection_ready_time = Histogram(
    "vllm:pipeline_connection_ready_time_seconds",
    "Time from communicator initialization start to fully connected and ready. "
    "Measures total connection establishment overhead including discovery, handshake, etc.",
    labelnames=["backend", "stage_idx"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY
)

# Component-level connection timing breakdown
vllm_pipeline_connection_discovery_time = Histogram(
    "vllm:pipeline_connection_discovery_time_seconds",
    "Time spent in service discovery / front router registration.",
    labelnames=["backend", "stage_idx"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY
)

vllm_pipeline_connection_handshake_time = Histogram(
    "vllm:pipeline_connection_handshake_time_seconds",
    "Time spent in connection handshake (e.g., NCCL unique_id exchange, RDMA connection).",
    labelnames=["backend", "stage_idx"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

vllm_pipeline_connection_nccl_init_time = Histogram(
    "vllm:pipeline_connection_nccl_init_time_seconds",
    "Time spent initializing NCCL communicator (ncclCommInitRank).",
    labelnames=["backend", "stage_idx", "peer_stage"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

vllm_pipeline_connection_rdma_setup_time = Histogram(
    "vllm:pipeline_connection_rdma_setup_time_seconds",
    "Time spent setting up RDMA connections (Mooncake/NIXL buffer registration, etc.).",
    labelnames=["backend", "stage_idx"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

# Connection state gauge
vllm_pipeline_connection_ready_timestamp = Gauge(
    "vllm:pipeline_connection_ready_timestamp_seconds",
    "Unix timestamp when the connection became ready (for correlation)",
    labelnames=["backend", "stage_idx"],
    registry=REGISTRY,
    multiprocess_mode='livesum'
)
# ============================================================================
# Dynamic Scaling / Reconfiguration Metrics
# ============================================================================
# These metrics track reconfiguration events during dynamic scaling (scale-up,
# scale-down, migration). Critical for measuring endpoint mobility overhead.

# Reconfiguration latency - time to complete a scaling event
vllm_pipeline_reconfiguration_latency = Histogram(
    "vllm:pipeline_reconfiguration_latency_seconds",
    "Time to complete a reconfiguration event (scale-up, scale-down, migration).",
    labelnames=["backend", "stage_idx", "event_type"],  # event_type: scale_up/scale_down/migration
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

# Epoch transition counter - number of epoch bumps
vllm_pipeline_epoch_transitions = Counter(
    "vllm:pipeline_epoch_transitions_total",
    "Total number of epoch transitions (route updates).",
    labelnames=["backend", "stage_idx"],
    registry=REGISTRY
)

# Stale epoch events - requests received with old epoch
vllm_pipeline_stale_epoch_events = Counter(
    "vllm:pipeline_stale_epoch_events_total",
    "Total number of stale epoch events (messages received with outdated epoch).",
    labelnames=["backend", "stage_idx"],
    registry=REGISTRY
)

# Connection warmup latency - time for new connection to become warm
vllm_pipeline_connection_warmup_latency = Histogram(
    "vllm:pipeline_connection_warmup_latency_seconds",
    "Time for a newly established connection to become warm (first successful data transfer).",
    labelnames=["backend", "stage_idx"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

# Recovery time - from scale completion to connection ready
vllm_pipeline_recovery_time = Histogram(
    "vllm:pipeline_recovery_time_seconds",
    "Time from reconfiguration completion to connection ready (recovery time).",
    labelnames=["backend", "stage_idx", "event_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY
)

# Scaling event counters
vllm_pipeline_scale_events = Counter(
    "vllm:pipeline_scale_events_total",
    "Total number of scaling events.",
    labelnames=["backend", "stage_idx", "event_type", "status"],  # status: success/failed
    registry=REGISTRY
)