# SPDX-License-Identifier: Apache-2.0
"""ConnexTE communicator, as loaded by the vLLM fork.

Copy this file to vllm/distributed/device_communicators/connexte_communicator.py
(docker/Dockerfile does) and put the Connex repo on PYTHONPATH.
The implementation lives in the engine-neutral core (connex.pp.endpoint) and
the vLLM adapter (connex.adapters.vllm.connexte); this shim only re-exports
it under the module path and class name the fork imports. Wire protocol and
behaviour are unchanged from the self-contained version of this file (271a603).
"""

from connex.adapters.vllm.connexte import (  # noqa: F401
    METRICS_ENABLED,
    ConnexTECommunicator,
    _ACK,
    _ACK_MARK,
    _CorrFilter,
    _G2,
    _LayoutCodec,
    _align,
    _Lease,
    _RegionTable,
    _decode_g2,
    _decode_layout,
    _dealer_identity,
    _derive_front_router_events,
    _encode_g2,
    _encode_layout,
    _probe_tcp,
    _ChainFailed,
)
