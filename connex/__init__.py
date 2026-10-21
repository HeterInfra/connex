# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Connex: a communication library for distributed AI infrastructure built on
endpoint mobility primitives.

It targets distributed and heterogeneous serving with many replicas of a
distributed model (pipeline chains, prefill/decode pools, attention/FFN
workers). Endpoints are addressed by stable logical endpoint IDs instead of
fixed ranks, so they can join, leave, fail and migrate while traffic is in
flight, and in-flight work hands over to another replica on failure or
scale-down. Multi-replica pipeline parallelism is the first fully integrated
scenario (``connex.pp``); prefill-decode has primitives, attention-FFN is on
the roadmap.

Layers:
- Control plane: directory / front-router (routes, leases, membership events)
- Mobility layer: versioned routing, shadow connections, cutover FSM
- Data plane: multi-queue scheduler, stream multiplexer, credit flow
- Transport: UCX / ZeroMQ, Mooncake Transfer Engine (ConnexTE)

The pipeline-parallel core used with inference engines is ``connex.pp``
(engine adapters in ``connex.adapters``); the optional request retry/resume
proxy is ``connex.serving.proxy``.
"""

__version__ = "0.1.0"

# Original components
from connex.directory import DirectoryClient, EndpointInfo
from connex.route_table import RouteTable, RouteEntry
from connex.cutover_fsm import CutoverFSM, CutoverState
from connex.stream_manager import StreamManager, StreamTable
from connex.ucx_transport import UCXTransport
from connex.qos_scheduler import QoSScheduler, TrafficClass
from connex.epc_api import EPCEndpoint

# Mobility-aware transport runtime
from connex.protocol import (
    LogicalEndpointID,
    PDRole,
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
    MooncakeTEInterface,
)

# V3.0 Integrated EPC API
from connex.epc_v3 import (
    EPCV3Config,
    EPCV3Endpoint,
    create_v3_endpoint,
    create_direct_connect_endpoint,  # No front router needed
    create_endpoint_with_mooncake,
)

# V2.0 EPC Client (for ConnexTE with Front Router)
from connex.epc_client import (
    EPCV2Client,
    EPCV2ClientConfig,
    EventType,
    TransportCapability as EPCV2TransportCapability,
    RouteState as EPCV2RouteState,
    LogicalEndpointID as EPCV2LogicalEndpointID,
    RouteEntry as EPCV2RouteEntry,
    Event as EPCV2Event,
    RouteTableCache,
    DataPlaneManager,
    ShadowConnectionManager as EPCV2ShadowConnectionManager,
    create_epc_client,
)

__all__ = [
    # Version
    "__version__",
    
    # Original EPC components (Legacy)
    "DirectoryClient",
    "EndpointInfo",
    "RouteTable",
    "RouteEntry",
    "CutoverFSM",
    "CutoverState",
    "StreamManager",
    "StreamTable",
    "UCXTransport",
    "QoSScheduler",
    "TrafficClass",
    "EPCEndpoint",
    
    # V3.0 Integrated EPC API (Recommended)
    "EPCV3Config",
    "EPCV3Endpoint",
    "create_v3_endpoint",
    "create_direct_connect_endpoint",  # No front router needed
    "create_endpoint_with_mooncake",
    
    # V2.0 EPC Client (for ConnexTE with Front Router)
    "EPCV2Client",
    "EPCV2ClientConfig",
    "EventType",
    "EPCV2TransportCapability",
    "EPCV2RouteState",
    "EPCV2LogicalEndpointID",
    "EPCV2RouteEntry",
    "EPCV2Event",
    "RouteTableCache",
    "DataPlaneManager",
    "EPCV2ShadowConnectionManager",
    "create_epc_client",
    
    # V3.0.0 Protocol
    "LogicalEndpointID",
    "PDRole",
    "StreamID",
    "MessageClass",
    "MessageType",
    "ProtocolHeader",
    "MobilityMessage",
    "CreditInfo",
    "create_data_message",
    "create_credit_update",
    "create_stale_epoch_feedback",
    "create_heartbeat",
    
    # V3.0.0 Mobility Control Plane
    "VersionedRoute",
    "VersionedRouteTable",
    "ConnectionCapability",
    "ConnectionState",
    "RouteState",
    "ShadowConnectionManager",
    "DualPathCutoverManager",
    "DualPathState",
    
    # V3.0.0 Mobility Data Plane
    "MultiQueueScheduler",
    "StreamMultiplexer",
    "CongestionController",
    "PeerQueueState",
    "StreamReorderState",
    
    # V3.0.0 GPU Fast Path
    "GPUAwareFastPath",
    "RegistrationCache",
    "ChunkedTensorTransfer",
    "CapabilityNegotiator",
    "TransportCapability",
    "TransportSelection",
    
    # V3.0.0 Mooncake Integration
    "ConnexMooncakeAdapter",
    "KVCacheTransferManager",
    "KVCacheBlock",
    "KVTransferRequest",
    "KVTransferStatus",
    "MooncakeTEInterface",
]
