# SPDX-License-Identifier: Apache-2.0
"""Pluggable pieces of a pipeline endpoint: network helpers, metrics sink,
and the callbacks an inference engine implements (:class:`EngineHooks`).

Nothing here imports an inference engine. Adapters (connex.adapters.*)
replace the defaults with engine-specific ones (e.g. vLLM's logger, its
network helpers and its Prometheus collectors).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable


# ---- network helpers -----------------------------------------------------------
# Same semantics as vllm.utils.network_utils (so endpoints and URLs are spelled
# identically on the wire), without the dependency.

def _is_ipv6(addr: str) -> bool:
    try:
        ipaddress.IPv6Address(addr.split("%")[0])
        return True
    except ValueError:
        return False


def get_ip() -> str:
    """CONNEX_HOST_IP, else VLLM_HOST_IP, else the address of the default
    route's interface."""
    host_ip = os.getenv("CONNEX_HOST_IP") or os.getenv("VLLM_HOST_IP")
    if host_ip:
        return host_ip
    for fam, probe in ((socket.AF_INET, ("8.8.8.8", 80)),  # release-scan: allow (no packet is sent)
                       (socket.AF_INET6, ("2001:4860:4860::8888", 80))):
        try:
            with socket.socket(fam, socket.SOCK_DGRAM) as s:
                s.connect(probe)  # UDP: no packet is sent
                return s.getsockname()[0]
        except Exception:  # noqa: BLE001
            pass
    logging.getLogger(__name__).warning(
        "Failed to get the IP address, using 0.0.0.0 (set CONNEX_HOST_IP)")
    return "0.0.0.0"


def get_tcp_uri(ip: str, port: int) -> str:
    return f"tcp://[{ip}]:{port}" if _is_ipv6(ip) else f"tcp://{ip}:{port}"


def join_host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if _is_ipv6(host) else f"{host}:{port}"


def split_host_port(host_port: str) -> tuple[str, int]:
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        return host[1:], int(port.split(":")[1])
    host, port = host_port.split(":")
    return host, int(port)


class NetUtils:
    """Namespace of the four helpers; endpoints call ``self.net.<fn>``."""
    get_ip = staticmethod(get_ip)
    get_tcp_uri = staticmethod(get_tcp_uri)
    join_host_port = staticmethod(join_host_port)
    split_host_port = staticmethod(split_host_port)


# ---- metrics ---------------------------------------------------------------------

METRIC_NAMES = (
    # per message (label: direction = "forward" | "return")
    "bytes_sent", "bytes_received", "messages_sent", "messages_received",
    "send_latency", "recv_latency", "enqueue_latency", "delivery_latency",
    "delivery_latency_local", "ready_to_use_latency", "tensor_count", "tensor_size",
    # no direction
    "serialization_latency", "deserialization_latency",
    "buffer_copy_latency",   # label: operation
    "errors",                # label: error_type
    "connection_status",     # label: socket_type
    # connection set-up
    "connection_ready_time", "connection_rdma_setup_time",
    "connection_ready_timestamp", "connection_discovery_time",
)


class MetricsSink(Protocol):
    """Optional metrics hook. ``name`` is one of :data:`METRIC_NAMES`;
    ``labels`` carry direction / operation / error_type / socket_type.
    The sink binds backend and stage itself."""

    enabled: bool

    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def inc(self, name: str, value: float = 1.0, **labels: str) -> None: ...
    def set(self, name: str, value: float, **labels: str) -> None: ...


class NullMetrics:
    enabled = False

    def observe(self, name, value, **labels):
        pass

    def inc(self, name, value=1.0, **labels):
        pass

    def set(self, name, value, **labels):
        pass


NULL_METRICS = NullMetrics()


class PrefixedPrometheusSink:
    """Maps metric names onto prometheus collectors found in ``module`` as
    ``<prefix>comm_<name>`` or ``<prefix><name>`` (connection metrics), each
    with labels backend, stage_idx and the call's labels. Children are bound
    once per label set (no labels() lookup on the hot path)."""

    enabled = True

    def __init__(self, module: Any, prefix: str, backend: str, stage_idx: int):
        self.module, self.prefix = module, prefix
        self.base = {"backend": backend, "stage_idx": stage_idx}
        self._children: dict = {}

    def _child(self, name: str, labels: dict):
        key = (name, tuple(sorted(labels.items())))
        ch = self._children.get(key)
        if ch is None:
            m = (getattr(self.module, f"{self.prefix}comm_{name}", None)
                 or getattr(self.module, f"{self.prefix}{name}", None))
            if m is None:
                ch = False
            else:
                ch = m.labels(**self.base, **labels)
            self._children[key] = ch
        return ch

    def observe(self, name, value, **labels):
        ch = self._child(name, labels)
        if ch:
            ch.observe(value)

    def inc(self, name, value=1.0, **labels):
        ch = self._child(name, labels)
        if ch:
            ch.inc(value)

    def set(self, name, value, **labels):
        ch = self._child(name, labels)
        if ch:
            ch.set(value)


# ---- engine hooks ----------------------------------------------------------------

RequestId = Any  # engines use str (vLLM) or str/int (SGLang); must be hashable + picklable


@runtime_checkable
class EngineHooks(Protocol):
    """What an inference engine provides to Connex for elastic pipeline
    parallelism with in-flight handover (docs/design/engine-contract.md).

    Stage 0 (the scheduling stage) implements :meth:`reprefill`; stages 1+
    implement :meth:`kv_held` and :meth:`release`. All stages may implement
    :meth:`release` for finished requests.
    """

    def kv_held(self, request_id: RequestId) -> int:
        """Stages 1+: number of leading tokens of ``request_id`` whose KV this
        stage computed itself and still holds (0 if none). Must not count
        KV it did not compute (a fresh replica after failover holds 0)."""
        ...

    def release(self, request_ids: Iterable[RequestId]) -> None:
        """Drop all per-request state (KV blocks) of these requests. Called
        for requests reported missing (their next appearance must be a fresh
        prefill) and for finished requests."""
        ...

    def reprefill(self, request_ids: Sequence[RequestId]) -> None:
        """Stage 0: these in-flight requests lack KV somewhere downstream.
        Discard the token sampled for them in this step and recompute them
        from scratch (prompt + already generated tokens) on the current chain,
        e.g. by rewinding num_computed_tokens to 0."""
        ...

    def abort(self, request_ids: Sequence[RequestId]) -> None:
        """Stage 0: finish these requests with an error (they were rewound
        more than the re-prefill budget allows)."""
        ...


class StepTransport(Protocol):
    """The step-level API of a pipeline endpoint (implemented by
    :class:`connex.pp.endpoint.ConnexTEEndpoint`; drivers in
    :mod:`connex.pp.link` only need this)."""

    stage_idx: int
    num_stages: int
    is_first: bool
    is_last: bool

    @property
    def inflight(self) -> int: ...
    def new_step_id(self) -> int: ...
    def send_step(self, tensor_dict: dict, corr: int, dst: Optional[int] = None) -> None: ...
    def recv_step(self, corr: int, timeout_s: Optional[float] = None) -> Optional[dict]: ...
    def abandon_step(self, corr: int) -> None: ...
    def recv_forward(self) -> tuple[Optional[dict], Optional[tuple]]: ...
    def send_reply(self, tensor_dict: dict, ctx: Optional[tuple]) -> None: ...
    def downstream_healthy(self) -> bool: ...
    # Optional (fast handover): True if step ``corr`` was sent on a chain that
    # has failed over since; Stage0Link then re-sends it without waiting.
    # def step_stale(self, corr: int) -> bool: ...
