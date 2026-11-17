# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Connex: a communication library for distributed LLM serving in which the set
of peers keeps changing.

Endpoints are addressed by stable logical endpoint IDs instead of fixed
ranks, so they can join, leave, fail and migrate while traffic is in flight,
and in-flight work hands over to another replica on failure or scale-down.

Where things live:
- Control plane: ``connex.front_router`` (routes, leases, membership events;
  client ``connex.epc_client``) and the lightweight ``connex.directory``.
- Engine integrations: ``connex.adapters.vllm`` (PP, PD connector),
  ``connex.adapters.sglang``, ``connex.pd``, ``connex.ep``, ``connex.rl``,
  ``connex.agents``.
- Data plane: connex-rs (``connex_rs``, Rust over UCX), Mooncake Transfer
  Engine, or ZeroMQ.
- Serving: ``connex.serving.pd_router`` and the retry/resume proxy
  ``connex.serving.proxy``.
"""

__version__ = "0.2.1"

# Public names are resolved lazily (PEP 562) so that ``import connex`` and
# ``import connex.<subpackage>`` do not pull in torch/vLLM/UCX up front.
_LAZY = {
    'DirectoryClient': ('connex.directory', 'DirectoryClient'),
    'EndpointInfo': ('connex.directory', 'EndpointInfo'),
    'ElasticPeerSet': ('connex.elastic', 'ElasticPeerSet'),
    'LogicalEndpointID': ('connex.protocol', 'LogicalEndpointID'),
    'PDRole': ('connex.protocol', 'PDRole'),
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


__all__ = ["__version__", *_LAZY]
