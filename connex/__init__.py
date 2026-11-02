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

__version__ = "0.2.0"

# Public names are resolved lazily (PEP 562) so that ``import connex`` and
# ``import connex.<subpackage>`` do not pull in torch/vLLM/UCX up front.
_LAZY = {
    'DirectoryClient': ('connex.directory', 'DirectoryClient'),
    'EndpointInfo': ('connex.directory', 'EndpointInfo'),
    'RouteTable': ('connex.route_table', 'RouteTable'),
    'RouteEntry': ('connex.route_table', 'RouteEntry'),
    'CutoverFSM': ('connex.cutover_fsm', 'CutoverFSM'),
    'CutoverState': ('connex.cutover_fsm', 'CutoverState'),
    'StreamManager': ('connex.stream_manager', 'StreamManager'),
    'StreamTable': ('connex.stream_manager', 'StreamTable'),
    'UCXTransport': ('connex.ucx_transport', 'UCXTransport'),
    'QoSScheduler': ('connex.qos_scheduler', 'QoSScheduler'),
    'TrafficClass': ('connex.qos_scheduler', 'TrafficClass'),
    'EPCEndpoint': ('connex.epc_api', 'EPCEndpoint'),
    'LogicalEndpointID': ('connex.protocol', 'LogicalEndpointID'),
    'PDRole': ('connex.protocol', 'PDRole'),
    'StreamID': ('connex.protocol', 'StreamID'),
    'MessageClass': ('connex.protocol', 'MessageClass'),
    'MessageType': ('connex.protocol', 'MessageType'),
    'ProtocolHeader': ('connex.protocol', 'ProtocolHeader'),
    'MobilityMessage': ('connex.protocol', 'MobilityMessage'),
    'CreditInfo': ('connex.protocol', 'CreditInfo'),
    'create_data_message': ('connex.protocol', 'create_data_message'),
    'create_credit_update': ('connex.protocol', 'create_credit_update'),
    'create_stale_epoch_feedback': ('connex.protocol', 'create_stale_epoch_feedback'),
    'create_heartbeat': ('connex.protocol', 'create_heartbeat'),
    'VersionedRoute': ('connex.mobility_control', 'VersionedRoute'),
    'VersionedRouteTable': ('connex.mobility_control', 'VersionedRouteTable'),
    'ConnectionCapability': ('connex.mobility_control', 'ConnectionCapability'),
    'ConnectionState': ('connex.mobility_control', 'ConnectionState'),
    'RouteState': ('connex.mobility_control', 'RouteState'),
    'ShadowConnectionManager': ('connex.mobility_control', 'ShadowConnectionManager'),
    'DualPathCutoverManager': ('connex.mobility_control', 'DualPathCutoverManager'),
    'DualPathState': ('connex.mobility_control', 'DualPathState'),
    'MultiQueueScheduler': ('connex.mobility_dataplane', 'MultiQueueScheduler'),
    'StreamMultiplexer': ('connex.mobility_dataplane', 'StreamMultiplexer'),
    'CongestionController': ('connex.mobility_dataplane', 'CongestionController'),
    'PeerQueueState': ('connex.mobility_dataplane', 'PeerQueueState'),
    'StreamReorderState': ('connex.mobility_dataplane', 'StreamReorderState'),
    'GPUAwareFastPath': ('connex.gpu_fastpath', 'GPUAwareFastPath'),
    'RegistrationCache': ('connex.gpu_fastpath', 'RegistrationCache'),
    'ChunkedTensorTransfer': ('connex.gpu_fastpath', 'ChunkedTensorTransfer'),
    'CapabilityNegotiator': ('connex.gpu_fastpath', 'CapabilityNegotiator'),
    'TransportCapability': ('connex.gpu_fastpath', 'TransportCapability'),
    'TransportSelection': ('connex.gpu_fastpath', 'TransportSelection'),
    'ConnexMooncakeAdapter': ('connex.mooncake_integration', 'ConnexMooncakeAdapter'),
    'KVCacheTransferManager': ('connex.mooncake_integration', 'KVCacheTransferManager'),
    'KVCacheBlock': ('connex.mooncake_integration', 'KVCacheBlock'),
    'KVTransferRequest': ('connex.mooncake_integration', 'KVTransferRequest'),
    'KVTransferStatus': ('connex.mooncake_integration', 'KVTransferStatus'),
    'MooncakeTEInterface': ('connex.mooncake_integration', 'MooncakeTEInterface'),
    'EPCV3Config': ('connex.epc_v3', 'EPCV3Config'),
    'EPCV3Endpoint': ('connex.epc_v3', 'EPCV3Endpoint'),
    'create_v3_endpoint': ('connex.epc_v3', 'create_v3_endpoint'),
    'create_direct_connect_endpoint': ('connex.epc_v3', 'create_direct_connect_endpoint'),
    'create_endpoint_with_mooncake': ('connex.epc_v3', 'create_endpoint_with_mooncake'),
    'EPCV2Client': ('connex.epc_client', 'EPCV2Client'),
    'EPCV2ClientConfig': ('connex.epc_client', 'EPCV2ClientConfig'),
    'EventType': ('connex.epc_client', 'EventType'),
    'EPCV2TransportCapability': ('connex.epc_client', 'TransportCapability'),
    'EPCV2RouteState': ('connex.epc_client', 'RouteState'),
    'EPCV2LogicalEndpointID': ('connex.epc_client', 'LogicalEndpointID'),
    'EPCV2RouteEntry': ('connex.epc_client', 'RouteEntry'),
    'EPCV2Event': ('connex.epc_client', 'Event'),
    'RouteTableCache': ('connex.epc_client', 'RouteTableCache'),
    'DataPlaneManager': ('connex.epc_client', 'DataPlaneManager'),
    'EPCV2ShadowConnectionManager': ('connex.epc_client', 'ShadowConnectionManager'),
    'create_epc_client': ('connex.epc_client', 'create_epc_client'),
}


def __getattr__(name):
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'connex' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


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
