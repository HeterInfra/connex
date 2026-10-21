# SPDX-License-Identifier: Apache-2.0
"""ConnexTE communicator for the vLLM fork's external pipeline mode.

A thin layer over :class:`connex.pp.endpoint.ConnexTEEndpoint`: it binds
vLLM's logger, network helpers, tensor-metadata codec (so the ZeroMQ-only
fallback frames are byte-identical to the fork's ZeroMQPPCommunicator) and
Prometheus collectors (``vllm_pipeline_comm_*``), and keeps the class name,
constructor and methods the fork calls:

    send_tensor_dict / recv_tensor_dict           (K = 1)
    new_step_id / send_step[_async] / recv_step / abandon_step   (stage 0, K > 1)
    recv_forward / send_reply[_async]             (stages 1.., K > 1)
    inflight, downstream_healthy, close

Everything else (wire format, retries, failover) is the core's.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_tcp_uri, join_host_port, split_host_port

from connex.pp import endpoint as _core
from connex.pp.endpoint import ConnexTEEndpoint
from connex.pp.hooks import PrefixedPrometheusSink
from connex.pp.zmq_base import DEFAULT_CODEC, TensorDictCodec

logger = init_logger(__name__)

try:
    from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict
    _CODEC = TensorDictCodec(TensorMetadata, _split_tensor_dict)
except ImportError:  # unit tests with stubbed vllm modules
    _CODEC = DEFAULT_CODEC

try:
    import vllm.distributed.device_communicators.pipeline_comm_metrics as _metrics_mod
    METRICS_ENABLED = True
except ImportError:
    _metrics_mod = None
    METRICS_ENABLED = False


class _VllmNet:
    get_ip = staticmethod(get_ip)
    get_tcp_uri = staticmethod(get_tcp_uri)
    join_host_port = staticmethod(join_host_port)
    split_host_port = staticmethod(split_host_port)


class ConnexTECommunicator(ConnexTEEndpoint):
    """ConnexTE: Mooncake TE data plane + Mobility V2 control plane (Front Router)."""

    log = logger
    net = _VllmNet
    codec = _CODEC

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        router_port: int,
        front_router_addr: str,
        device: str = "cuda",
        timeout_ms: int = -1,
        **kw,
    ):
        if METRICS_ENABLED and "metrics" not in kw:
            kw["metrics"] = PrefixedPrometheusSink(
                _metrics_mod, "vllm_pipeline_", "connexte", stage_idx)
        super().__init__(stage_idx, num_stages, router_port, front_router_addr,
                         device=device, timeout_ms=timeout_ms, **kw)


# Internals re-exported for tests and tools written against the old module.
_RegionTable = _core._RegionTable
_Lease = _core._Lease
_CorrFilter = _core._CorrFilter
_LayoutCodec = _core._LayoutCodec
_align = _core._align
_encode_layout = _core._encode_layout
_decode_layout = _core._decode_layout
_encode_g2 = _core._encode_g2
_decode_g2 = _core._decode_g2
_ACK_MARK = _core._ACK_MARK
_ACK = _core._ACK
_G2 = _core._G2
_dealer_identity = _core._dealer_identity
_derive_front_router_events = _core._derive_front_router_events
_probe_tcp = _core._probe_tcp
_ChainFailed = _core._ChainFailed
