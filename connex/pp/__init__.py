# SPDX-License-Identifier: Apache-2.0
"""Connex PP: engine-independent communication for elastic pipeline
parallelism with in-flight handover.

Public API (see docs/design/engine-contract.md):

    PipelineEndpoint (= ConnexTEEndpoint)  one stage's data/step endpoint
    Stage0Link, StageWorker, StepFailed    step drivers built on an endpoint
    EngineHooks, StepTransport, MetricsSink  protocols an engine/adapter fills
    StaticMembership, select_peer          membership / routing
    KVLedger, find_missing, merge_missing  handover helpers; CX_CTX, CX_MISSING
    wire                                   the wire protocol (pure stdlib)

Importing this package needs neither an inference engine nor torch; the
endpoint (torch, pyzmq, safetensors; Mooncake optional) loads on first use.
"""

from connex.pp import wire  # noqa: F401
from connex.pp.handover import (  # noqa: F401
    CX_CTX, CX_MISSING, FINISHED, REQUEST_IDS, FinishedOutbox, KVLedger, RewindBudget,
    blank_rows, find_missing, merge_missing,
)
from connex.pp.hooks import (  # noqa: F401
    EngineHooks, MetricsSink, NetUtils, NullMetrics, StepTransport,
)
from connex.pp.link import Stage0Link, StageWorker, StepFailed  # noqa: F401
from connex.pp.ring import RegionTable  # noqa: F401
from connex.pp.routing import Lid, Route, StaticMembership, select_peer  # noqa: F401
from connex.pp.steps import CorrFilter, ReplyCache, StepIds  # noqa: F401

__all__ = [
    "wire", "CX_CTX", "CX_MISSING", "FINISHED", "REQUEST_IDS", "FinishedOutbox", "KVLedger",
    "RewindBudget", "blank_rows", "find_missing", "merge_missing", "EngineHooks", "MetricsSink",
    "NetUtils", "NullMetrics", "StepTransport", "Stage0Link", "StageWorker", "StepFailed",
    "RegionTable", "Lid", "Route", "StaticMembership", "select_peer", "CorrFilter", "ReplyCache",
    "StepIds", "ConnexTEEndpoint", "PipelineEndpoint", "ZmqStageBase", "TensorDictCodec",
]

_LAZY = {
    "ConnexTEEndpoint": "connex.pp.endpoint",
    "PipelineEndpoint": "connex.pp.endpoint",
    "ZmqStageBase": "connex.pp.zmq_base",
    "TensorDictCodec": "connex.pp.zmq_base",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib
    return getattr(importlib.import_module(mod), name)
