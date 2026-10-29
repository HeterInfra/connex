# SPDX-License-Identifier: Apache-2.0
"""ConnexTE pipeline endpoint: the engine-neutral core of Connex PP.

ConnexTE = Mooncake Transfer Engine data plane (GPUDirect send ring with
ack-based reuse, or host buffers) + ZeroMQ signals + Front Router control
plane (EPCV2Client: registration, leases, membership events).

One :class:`ConnexTEEndpoint` per pipeline stage process. Stage 0 drives
steps (``new_step_id`` / ``send_step`` / ``recv_step`` / ``abandon_step``);
stages 1.. answer them (``recv_forward`` / ``send_reply``). The plain
``send_tensor_dict`` / ``recv_tensor_dict`` pair is the K=1 form of the same
protocol. Wire format: :mod:`connex.pp.wire`; engine contract:
docs/design/engine-contract.md.

This module imports no inference engine. Engine adapters subclass the
endpoint and set ``log``, ``net``, ``codec`` and ``metrics``
(connex/adapters/vllm). This code was moved here unchanged in behaviour from
integrations/vllm/connexte_communicator.py (fork commit 271a603).
"""

import logging
import os
import struct
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import zmq
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from connex.pp import transports as _transports
from connex.pp.hooks import split_host_port, join_host_port
from connex.pp.ring import Lease as _Lease, RegionTable as _RegionTable
from connex.pp.routing import select_peer
from connex.pp.steps import CorrFilter as _CorrFilter
from connex.pp.wire import (
    ACK as _ACK,
    ACK_MARK as _ACK_MARK,
    DT_CODE as _DT_CODE,
    DT_NAMES as _DT_NAMES,
    G2 as _G2,
    LayoutCodec as _LayoutCodec,
    align as _align,
    decode_g2 as _decode_g2,
    decode_layout as _decode_layout,
    encode_g2 as _encode_g2,
    encode_layout as _encode_layout,
    signal_corr as _signal_corr,
)
from connex.pp.zmq_base import ZmqStageBase

logger = logging.getLogger(__name__)

MOONCAKE_AVAILABLE = _transports.MOONCAKE_AVAILABLE
TransferEngineClass = _transports._MooncakeTE

EPCV2_CLIENT_AVAILABLE = False
try:
    from connex.epc_client import (
        EPCV2Client,
        EPCV2ClientConfig,
        TransportCapability,
        EventType,
    )
    EPCV2_CLIENT_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    logger.warning("EPCV2Client not available: %s", e)

# Churn resilience settings (configurable via environment variables)
_CONNEXTE_SEND_TIMEOUT_MS = int(os.getenv("CONNEXTE_SEND_TIMEOUT_MS", "5000"))  # 5s ZMQ send timeout
_CONNEXTE_RECV_TIMEOUT_MS = int(os.getenv("CONNEXTE_RECV_TIMEOUT_MS", "8000"))  # 8s ZMQ recv timeout for retries


_MIN_WAIT_S = 0.005  # recv_step: below this remaining time, it is a timeout


def _dealer_identity(tag: str) -> bytes:
    """ZMQ identity of a data-plane DEALER: unique per socket.

    A ROUTER that already has a connection with identity X treats a second
    connection announcing X as anonymous and silently drops everything it
    sends (ROUTER_HANDOVER=0), for as long as that connection lives. With
    fixed identities ("rk-1", "rk-return-3") a DEALER re-created while the
    old one's TCP connection was still draining (LINGER, in-flight K>1
    sends), or two upstream replicas sharing one downstream after failover,
    lose every message -- the "re-sent step never answered" failure under
    churn with CONNEX_PP_INFLIGHT>1."""
    return f"{tag}-{os.urandom(4).hex()}".encode()

# Opt-in per-phase timing (CONNEXTE_PROFILE=1): logs mean ms per phase every
# CONNEXTE_PROFILE_EVERY receives. Used to find where per-hop latency goes.
_PROFILE = os.getenv("CONNEXTE_PROFILE", "0") == "1"
# K>1 only: send_step_async / send_reply_async hand the "wait for the device
# copies, then signal" part of a GPUDirect send to one sender thread
# (CONNEXTE_ASYNC_SEND=0 makes them synchronous again).
_ASYNC_SEND = os.getenv("CONNEXTE_ASYNC_SEND", "1") == "1"
_SWITCH_INTERVAL_S = float(os.getenv("CONNEXTE_SWITCH_INTERVAL_S", "0.0002"))
# CONNEXTE_TRACE=/path/prefix: append "time.time() stage event corr" lines to
# <prefix>.stage<N>.txt (send called, signal sent, message arrived, recv done)
# to rebuild the cross-stage timeline of each step. Off by default.
_TRACE = os.getenv("CONNEXTE_TRACE", "")


class _Trace:
    def __init__(self, stage_idx: int):
        self.f = open(f"{_TRACE}.stage{stage_idx}.txt", "a", buffering=1 << 16)
        self.stage = stage_idx

    def __call__(self, ev: str, corr) -> None:
        self.f.write(f"{time.time():.6f} {self.stage} {ev} {corr}\n")


_trace = None


def _tr(stage_idx: int, ev: str, corr) -> None:
    global _trace
    if _TRACE:
        if _trace is None:
            _trace = _Trace(stage_idx)
        _trace(ev, corr)
_PROFILE_EVERY = int(os.getenv("CONNEXTE_PROFILE_EVERY", "64"))


# GPUDirect path (CONNEXTE_GDR=1): tensors go GPU -> registered GPU staging
# ring -> RDMA read into the receiver's registered GPU buffer. No host copies,
# no safetensors. The small header (layout + non-tensor fields) rides in the
# ZMQ signal. Falls back to the original host-buffer path when disabled.
_GDR = os.getenv("CONNEXTE_GDR", "1") == "1"
# Staging buffers of the "GPUDirect" path in pinned host memory instead of
# GPU memory: "1", "0" (default), or "auto" = host when K>1 (CONNEX_PP_INFLIGHT)
# and MOONCAKE_PROTOCOL=tcp. Only useful together with CONNEX_PP_OVERLAP=1; with
# synchronous sends it measured slower (K=2: 795 vs 822 tok/s). Why: Mooncake's tcp transport moves GPU memory
# with a synchronous cudaMemcpy, which waits for every kernel queued on the
# legacy default stream, so with K>1 the receive of one microbatch's reply
# blocked until the forward of the other microbatch finished (measured 2 ms
# r_gdr, and no overlap possible). With host staging the device<->host copies
# are cudaMemcpyAsync on the caller's stream and Mooncake never touches CUDA.
_GDR_HOST = os.getenv("CONNEXTE_GDR_HOST", "0")
_GDR_SEND_MB = int(os.getenv("CONNEXTE_GDR_SEND_MB", "512"))
_GDR_RECV_MB = int(os.getenv("CONNEXTE_GDR_RECV_MB", "256"))
_GDR_HOLD_S = float(os.getenv("CONNEXTE_GDR_HOLD_S", "1.0"))  # time-based hold (legacy / no-ack mode)
# Ack-based ring reuse (CONNEXTE_GDR_ACK=1): the receiver sends a tiny
# b"CXACK"+token message to the sender's ROUTER once its RDMA read completed;
# the sender frees the ring region only then (or after the long safety
# timeout below, which only matters if the receiver died). Requires wire v2.
_GDR_ACK = os.getenv("CONNEXTE_GDR_ACK", "1") == "1"
_GDR_ACK_TIMEOUT_S = float(os.getenv("CONNEXTE_GDR_ACK_TIMEOUT_S", "10.0"))
# Signal header format sent by the GPUDirect path: 2 = fixed struct + cached
# layout bytes (no pickle unless non-tensor fields exist), 1 = legacy pickle
# ("G1"), for mixed deployments with receivers that predate v2. Receivers
# always accept both.
_GDR_WIRE = int(os.getenv("CONNEXTE_GDR_WIRE", "2"))
# Receive slots: 0/1 = copy (clone) out of the single registered recv buffer
# (safe default); N>1 = rotate N slots and hand vLLM views (no clone). Only
# safe if the consumer drops the received tensors before N more receives.
_GDR_RECV_SLOTS = int(os.getenv("CONNEXTE_GDR_RECV_SLOTS", "0"))
# Drop messages whose correlation id is older than the newest one already
# seen from the same stage-0 incarnation (non-first stages only).
_STALE_DROP = os.getenv("CONNEXTE_STALE_DROP", "1") == "1"
# Pipelining (CONNEX_PP_INFLIGHT=K): number of pipeline steps (microbatches)
# stage 0 may keep in flight. 1 = today's strictly serial protocol, unchanged.
# With K > 1 the communicator additionally offers a per-step API
# (new_step_id/send_step/recv_step on stage 0, recv_forward/send_reply on
# later stages): each step keeps its own correlation id, replies are matched
# by id (any order), a retry re-sends one step under the same id, later stages
# remember their last K replies for dedupe and accept ids up to K-1 behind the
# newest one. The wire format is unchanged (the id was already in every
# signal), so K only changes local matching/caching rules.
_PP_INFLIGHT = max(1, int(os.getenv("CONNEX_PP_INFLIGHT", "1") or "1"))
# 1 = old behaviour: full route-table RPC on every peer lookup.
_ROUTE_REFRESH_ALWAYS = os.getenv("CONNEXTE_ROUTE_REFRESH_ALWAYS", "0") == "1"

# Fast handover (CONNEXTE_FAST_HANDOVER, default on). A missed lease or a
# data-path timeout is only a suspicion (a busy engine starves its heartbeat
# thread; a long step looks like a lost one). It is confirmed by an active
# probe: a TCP connect to the peer's ZMQ port. The listening socket is served
# by the kernel, so a live but GIL-starved engine still accepts, while a dead
# process (host network) refuses at once. A confirmed death bumps a "chain
# epoch"; stage 0 then fails every in-flight step sent before it at once
# (instead of each waiting out its own recv timeout) and re-sends them.
_FAST_HANDOVER = os.getenv("CONNEXTE_FAST_HANDOVER", "1") != "0"
_PROBE_TIMEOUT_S = float(os.getenv("CONNEXTE_PROBE_TIMEOUT_S", "0.5") or "0.5")
# How often a waiting stage-0 recv checks the chain epoch.
_EPOCH_POLL_MS = 50
# Grace before stage 0 re-sends after a death farther down the chain, so the
# stage just upstream of the dead one (which probes it too) has failed over.
_REMOTE_FAILOVER_DELAY_S = float(os.getenv("CONNEXTE_REMOTE_FAILOVER_DELAY_S", "0.2") or "0.2")
# Same grace when the death came from a transport event (connex-rs): the
# upstream stage gets the same event at the same moment, so it can be short.
_TRANSPORT_FAILOVER_DELAY_S = float(os.getenv("CONNEX_RS_REMOTE_FAILOVER_DELAY_S", "0.05") or "0")


def _probe_tcp(addr: str | None, timeout_s: float = _PROBE_TIMEOUT_S) -> bool | None:
    """Is anything listening at ``addr`` (tcp://host:port)? True = alive,
    False = refused/unreachable (two attempts), None = cannot tell."""
    import socket as _socket

    if not addr:
        return None
    try:
        hp = addr.split("://", 1)[-1]
        host, port = hp.rsplit(":", 1)
        host, port = host.strip("[]"), int(port)
    except ValueError:
        return None
    if host in ("*", "0.0.0.0", ""):
        return None
    for _ in range(2):
        try:
            _socket.create_connection((host, port), timeout=timeout_s).close()
            return True
        except ConnectionRefusedError:
            return False
        except OSError:
            continue  # timeout / unreachable: try once more
    return False

# Design note (not implemented): push model. Today each hop is
# signal -> receiver wakes -> RDMA READ -> ack, i.e. one ZMQ RTT plus one
# RDMA RTT on the critical path. With push, every receiver advertises K
# registered recv slots (ptr, size, rkey via Mooncake segment) to its
# upstream once, plus a credit per slot it has finished consuming (the ack
# above, flowing the other way). The sender then transfer_sync_write()s the
# D2D-staged bytes straight into a credited slot and sends the small header
# afterwards; the receiver only polls ZMQ and never issues a READ, saving one
# RDMA RTT (~10 us at 16 KB) and the receiver-side transfer submission cost,
# and letting the write overlap the sender's next compute. Open issues:
# credits must be invalidated on receiver restart (epoch in the advert),
# large prefill messages need a pull fallback, and slot reuse needs the same
# consumer-done event rule as CONNEXTE_GDR_RECV_SLOTS.



class _PhaseTimer:
    def __init__(self, stage_idx: int, log: logging.Logger = logger):
        self.stage_idx = stage_idx
        self.log = log
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.n = 0
        self.last_recv_done: float | None = None

    def add(self, phase: str, dt: float) -> None:
        self.sums[phase] += dt
        self.counts[phase] += 1

    def tick(self) -> None:
        self.n += 1
        if self.n % _PROFILE_EVERY == 0:
            parts = " ".join(f"{k}={1000 * self.sums[k] / self.counts[k]:.2f}"
                             for k in sorted(self.sums))
            self.log.info("Stage %s CONNEXTE_PROFILE (ms, n=%d): %s", self.stage_idx, self.n, parts)
            self.sums.clear()
            self.counts.clear()
_CONNEXTE_SEND_RETRIES = int(os.getenv("CONNEXTE_SEND_RETRIES", "3"))  # Max send retries on failure
_CONNEXTE_RECV_RETRIES = int(os.getenv("CONNEXTE_RECV_RETRIES", "3"))  # Max recv retries (re-send + re-recv)


def _derive_front_router_events(api_addr: str) -> str:
    """Derive events address from API address (replace port 15555 -> 15560)."""
    api_addr = api_addr.replace("tcp://", "").strip()
    host, _ = split_host_port(api_addr)
    return f"tcp://{join_host_port(host, 15560)}"


class _ChainFailed(Exception):
    """Stage 0 recv: the step being waited for was lost to a failover."""


class ConnexTEEndpoint(ZmqStageBase):
    """ConnexTE: Mooncake TE data plane + Mobility V2 control plane (Front Router).

    - Control plane: EPCV2Client -> Front Router (register, epoch, migration events)
    - Data plane: Mooncake Transfer Engine for RDMA
    - Metadata: ZeroMQ ROUTER/DEALER for buffer pointer/size coordination
    """

    # Pipelining defaults (CONNEX_PP_INFLIGHT); class level so that instances
    # built without __init__ (unit tests) behave like K=1.
    _inflight = 1
    _mb_outstanding: "set[int] | frozenset" = frozenset()
    # GPUDirect staging buffers in GPU memory unless CONNEXTE_GDR_HOST picks host
    _gdr_host = False
    _gdr_kw: dict = {"device": "cuda"}
    # Fast-handover state (see _init_handover); class level for the same reason.
    _chain_epoch = 0     # bumped when our direct downstream peer is failed over
    _remote_epoch = 0    # bumped when a replica farther down is confirmed dead
    _remote_fast_epoch = -1  # remote epoch last bumped by a transport event
    _k1_sent = None      # K=1: (chain_epoch, remote_epoch, t) of the last send
    _sent_at = None      # K>1: corr -> (chain_epoch, remote_epoch, t)
    _probing = None      # (stage, replica) probes running in the background
    _last_reply_corr = None  # K=1 stage 0: id of the last step answered

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        router_port: int,
        front_router_addr: str,
        device: str = "cuda",
        timeout_ms: int = -1,
        *,
        membership: Any = None,
        transfer_engine: Any = None,
        local_url: Optional[str] = None,
        metrics: Any = None,
        inflight: Optional[int] = None,
    ):
        """Engine-neutral extras (keyword only, all optional):

        membership: an EPC-client-like membership source (see
            connex.pp.routing); default: EPCV2Client to the Front Router.
        transfer_engine, local_url: an already initialised TransferEngine
            (connex.pp.transports) and the URL peers read from; default:
            Mooncake from the MOONCAKE_* environment.
        metrics: a connex.pp.hooks.MetricsSink; default: no metrics.
        inflight: steps in flight K; default: CONNEX_PP_INFLIGHT (1).
        """
        connexte_init_start = time.time()
        self._membership = membership
        self._injected_engine = (transfer_engine, local_url)
        if metrics is not None:
            self.metrics = metrics

        # Resolve Front Router API/Events for Mobility V2
        self._front_router_api = os.getenv("FRONT_ROUTER_API") or os.getenv(
            "CONNEXTE_FRONT_ROUTER_API"
        )
        self._front_router_events = os.getenv("FRONT_ROUTER_EVENTS") or os.getenv(
            "CONNEXTE_FRONT_ROUTER_EVENTS"
        )
        if not self._front_router_api:
            base = front_router_addr.strip()
            if not base.startswith("tcp://"):
                base = f"tcp://{base}"
            self._front_router_api = base
        if not self._front_router_events:
            self._front_router_events = _derive_front_router_events(
                self._front_router_api.replace("tcp://", "")
            )

        self._model_name = os.getenv("SERVED_MODEL_NAME", "default")
        self._epc_client: Optional[Any] = None
        self._peer_dealer_cache: dict[int, Any] = {}
        self._mcache: dict = {}  # bound prometheus children, see _m()
        self._sticky_peer: dict[int, int] = {}
        self._route_dirty: set[int] = set()  # stages whose cached route failed
        self._sndtimeo_ok: dict[int, Any] = {}  # id(socket) -> socket with SNDTIMEO set
        self._ack_socks: dict[str, Any] = {}
        self._stash: list = []  # non-ack frames read while draining acks
        self._gdr = None

        super().__init__(
            stage_idx=stage_idx,
            num_stages=num_stages,
            router_port=router_port,
            front_router_addr=front_router_addr,
            device=device,
            timeout_ms=timeout_ms,
        )

        # Mooncake-specific state
        self._mooncake_engine: Any = None
        self._mooncake_available = False
        self._local_url: str | None = None
        self._remote_urls: dict[int, str] = {}
        self._buffer_cleaner: ThreadPoolExecutor | None = None

        # Multi-replica routing: track originator (stage 0) endpoint for return path
        self._current_originator_endpoint: str | None = None

        # Churn resilience: track recently deregistered replicas to avoid routing to them
        import threading as _threading
        self._deregistered_replicas: set[tuple[int, int]] = set()  # (stage_idx, replica_idx)
        # Missed-lease (UNHEALTHY) peers: only a suspicion. A busy engine can
        # starve its heartbeat thread, so routing changes only once the data
        # path confirms it (a recv timeout while the suspect is our peer).
        self._suspect_replicas: set[tuple[int, int]] = set()
        self._init_handover()
        self._prof = _PhaseTimer(stage_idx, self.log) if _PROFILE else None
        # Request/response correlation (see _corr_for_send / recv_tensor_dict):
        # stage 0 stamps each pipeline step with an id and reuses it on retry;
        # later stages echo it; stage 0 drops replies for other ids and a
        # stage that sees an id it already answered re-sends the cached reply
        # instead of recomputing (retries become idempotent).
        self._corr_base = int.from_bytes(os.urandom(4), "big") << 32
        self._corr_seq = 0
        self._pending_corr: int | None = None
        self._corr_in: int | None = None
        self._last_done: tuple[int, bytes, int | None] | None = None
        self._recent_step_s: list[float] = []
        self._corr_filter = _CorrFilter()
        self._last_token: int | None = None  # ring lease pinned for _resend_last
        k = _PP_INFLIGHT if inflight is None else max(1, int(inflight))
        if k > 1:
            self._init_inflight(k)
        self._layouts = _LayoutCodec()
        self._peer_lock = _threading.Lock()
        # Track the last known-good peer per stage for fast fallback
        self._last_failed_peer: dict[int, str] = {}  # stage_idx -> failed endpoint

        # Data-plane backend (CONNEX_PP_BACKEND). connex-rs starts before the
        # control plane: its URL is advertised as a second address so peers
        # can watch each other's liveness over it.
        self._backend = "injected" if transfer_engine is not None else _transports.backend_from_env()
        self._crs = None  # (engine, url) of connex-rs
        self._crs_alive: set = set()   # peers (stage, replica) with a live connex-rs link
        self._crs_dead: set = set()    # confirmed dead; not re-watched until they register again
        self._crs_watch_stop = threading.Event()
        if self._backend == "connex-rs":
            try:
                eng, url = _transports.create_connexrs_engine(
                    stage_idx, self._replica_idx(), self._model_name, front_router_addr,
                    self.net, self.log)
                if eng is not None:
                    self._crs = (eng, url)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("Stage %s: connex-rs engine failed to start (%s); falling back "
                                 "to the ZeroMQ data path", stage_idx, exc)

        # ConnexTE: Start EPCV2Client for Mobility V2 control plane
        if self._membership is not None:
            self._attach_membership(self._membership)
        elif EPCV2_CLIENT_AVAILABLE:
            self._setup_epc_client()

        rdma_setup_start = time.time()
        self._setup_mooncake_engine()
        rdma_setup_time = time.time() - rdma_setup_start

        connexte_init_time = time.time() - connexte_init_start

        if self.metrics.enabled:
            self.metrics.observe("connection_ready_time", connexte_init_time)
            self.metrics.observe("connection_rdma_setup_time", rdma_setup_time)
            self.metrics.set("connection_ready_timestamp", time.time())

        self.log.info(
            "ConnexTE communicator initialized for stage %s (Mooncake: %s, EPCV2: %s, init: %.3fs)",
            stage_idx,
            self._mooncake_available,
            self._epc_client is not None,
            connexte_init_time,
        )

    @property
    def backend_name(self) -> str:
        return "connexte"

    def _setup_control_plane(self) -> None:
        """ConnexTE uses EPCV2Client when available; else fallback to base zeromq control."""
        if not EPCV2_CLIENT_AVAILABLE and getattr(self, "_membership", None) is None:
            super()._setup_control_plane()

    def _register_stage(self) -> None:
        """ConnexTE registers via EPCV2Client when available."""
        if not EPCV2_CLIENT_AVAILABLE and getattr(self, "_membership", None) is None:
            super()._register_stage()

    def _setup_epc_client(self) -> None:
        """Initialize EPCV2Client and register with Front Router (Mobility V2)."""
        if not EPCV2_CLIENT_AVAILABLE or self._epc_client is not None:
            return

        try:
            # Generate unique replica_idx based on pod IP to distinguish replicas
            # Use last octet of IP as a simple distinguisher
            replica_idx = self._replica_idx()
            addresses = [self.router_endpoint]
            if getattr(self, "_crs", None) is not None:
                addresses.append(self._crs[1])  # connex-rs URL (liveness watch)

            config = EPCV2ClientConfig(
                front_router_api=self._front_router_api,
                front_router_events=self._front_router_events,
                model_name=self._model_name,
                stage_idx=self.stage_idx,
                replica_idx=replica_idx,
                local_addresses=addresses,
                capabilities=[TransportCapability.ZEROMQ, TransportCapability.MOONCAKE_TE],
                mooncake_enabled=True,
                # a tensor-parallel stage is one endpoint: TP rank 0 (this
                # endpoint) is shard 0 of a group whose other ranks register
                # for liveness only (connex.pp.group.TPShardMember)
                group_size=max(1, int(os.getenv("CONNEX_PP_TP_SIZE", "1") or "1")),
            )
            self._epc_client = EPCV2Client(config)
            self._register_event_handlers(self._epc_client)

            if self._epc_client.start():
                self.log.info(
                    "ConnexTE stage %s (replica=%s): EPCV2Client registered with Front Router",
                    self.stage_idx, replica_idx
                )
            else:
                self.log.warning("ConnexTE stage %s: EPCV2Client registration failed", self.stage_idx)
                self._epc_client.stop()
                self._epc_client = None
        except Exception as exc:  # noqa: BLE001
            self.log.warning("ConnexTE stage %s: EPCV2Client setup failed: %s", self.stage_idx, exc)
            self._epc_client = None
            self._epc_client = None

    def _replica_idx(self) -> int:
        # Prefer the deployment's replica index: the IP octet makes the
        # rank-based peer wiring cross replicas (r0's stage 2 -> r1's
        # stage 3), so killing one replica also breaks the other.
        env_replica = os.getenv("PIPELINE_REPLICA_IDX")
        if env_replica not in (None, ""):
            return int(env_replica)
        return int(self.net.get_ip().split(".")[-1]) % 256

    def _register_event_handlers(self, client: Any) -> None:
        """Membership events -> routing state (pod IP recovery, joins,
        departures). An UNHEALTHY peer (missed lease) only becomes a suspect,
        see _on_endpoint_unhealthy."""
        on = getattr(client, "on_event", None)
        if on is None:
            return
        if EPCV2_CLIENT_AVAILABLE:
            ev = EventType
            names = (ev.ROUTE_UPDATED, ev.ENDPOINT_REGISTERED, ev.ENDPOINT_DEREGISTERED,
                     ev.ENDPOINT_UNHEALTHY)
        else:  # pragma: no cover
            names = ("route_updated", "endpoint_registered", "endpoint_deregistered",
                     "endpoint_unhealthy")
        for name, fn in zip(names, (self._on_route_updated, self._on_endpoint_registered,
                                    self._on_endpoint_deregistered, self._on_endpoint_unhealthy)):
            on(name, fn)

    def _attach_membership(self, client: Any) -> None:
        """Use an injected membership source instead of EPCV2Client."""
        self._epc_client = client
        self._register_event_handlers(client)
        start = getattr(client, "start", None)
        if start is not None and start() is False:
            self.log.warning("Stage %s: membership source failed to start", self.stage_idx)

    def _on_route_updated(self, event: Any) -> None:
        """Handle ROUTE_UPDATED event from Front Router.

        When a peer's IP changes (e.g., pod crash recovery), we need to:
        1. Invalidate cached connections to that peer
        2. Force refresh routes to get new addresses
        """
        try:
            lid_dict = event.payload.get("lid", {})
            changed_stage = lid_dict.get("stage_idx", -1)
            old_addrs = event.payload.get("old_addresses", [])
            new_addrs = event.payload.get("new_addresses", [])

            # Check if this affects our upstream or downstream peer
            is_upstream = changed_stage == self.stage_idx - 1
            is_downstream = changed_stage == self.stage_idx + 1

            if is_upstream or is_downstream:
                self.log.info(
                    "ConnexTE stage %s: Route updated for stage %s (%s -> %s), "
                    "invalidating peer cache",
                    self.stage_idx, changed_stage, old_addrs, new_addrs
                )

                # Invalidate specific peer's socket cache
                if changed_stage in self._peer_dealer_cache:
                    sock, _ = self._peer_dealer_cache.pop(changed_stage)
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass

                # Clear forward/return socket if affected
                if is_downstream:
                    if self.forward_socket is not None:
                        try:
                            self.forward_socket.close()
                        except Exception:
                            pass
                        self.forward_socket = None
                elif is_upstream and self.is_last:
                    if self.return_socket is not None:
                        try:
                            self.return_socket.close()
                        except Exception:
                            pass
                        self.return_socket = None

                self._route_dirty.add(changed_stage)
                # Update Mooncake remote URL cache if applicable
                if changed_stage in self._remote_urls:
                    del self._remote_urls[changed_stage]

        except Exception as exc:
            self.log.warning(
                "ConnexTE stage %s: Error handling ROUTE_UPDATED: %s",
                self.stage_idx, exc
            )

    def _on_endpoint_registered(self, event: Any) -> None:
        """Handle ENDPOINT_REGISTERED event from Front Router.

        When a peer registers (possibly with new IP after restart),
        refresh connections if address changed.
        """
        try:
            lid_dict = event.payload.get("lid", {})
            changed_stage = lid_dict.get("stage_idx", -1)
            changed_replica = lid_dict.get("replica_idx", -1)
            address_changed = event.payload.get("address_changed", False)

            # Multi-replica note:
            # Even when address does NOT change, a new replica registration changes
            # membership/topology. Our peer selection logic (_get_peer_endpoint)
            # depends on the sorted list of replicas at *our* stage (to compute rank)
            # and at the target stage. If we cache the downstream DEALER socket before
            # all replicas are visible, we can get stuck routing multiple replicas to
            # the same downstream replica, causing overload/timeouts.
            self.log.info(
                "ConnexTE stage %s: Received ENDPOINT_REGISTERED for stage %s replica %s (address_changed=%s)",
                self.stage_idx,
                changed_stage,
                changed_replica,
                address_changed,
            )

            # Clear deregistered tracking for this replica (it's back)
            self._deregistered_replicas.discard((changed_stage, changed_replica))
            getattr(self, "_crs_dead", set()).discard((changed_stage, changed_replica))
            self._suspect_replicas.discard((changed_stage, changed_replica))
            # Sticky routing: a replica (re)joining is not a reason to move
            # established chains -- remapping mid-flight sends in-flight
            # requests to stages that do not hold their KV cache. Only act if
            # the replica we route to changed address.
            sticky = getattr(self, "_sticky_peer", {})
            if not address_changed:
                return
            if changed_stage in sticky and sticky[changed_stage] != changed_replica:
                return

            # If address changed, reuse route update logic (it also clears Mooncake cache).
            if address_changed:
                if changed_stage in (self.stage_idx - 1, self.stage_idx + 1):
                    self._on_route_updated(event)
                return

            # Address unchanged: invalidate cached peer mapping when membership changes.
            targets_to_invalidate: set[int] = set()

            if changed_stage == self.stage_idx:
                # Our own stage got a new replica: our rank mapping may change, so our
                # chosen upstream/downstream peer may need to change.
                if self.stage_idx - 1 >= 0:
                    targets_to_invalidate.add(self.stage_idx - 1)
                if self.stage_idx + 1 < self.num_stages:
                    targets_to_invalidate.add(self.stage_idx + 1)
            elif changed_stage in (self.stage_idx - 1, self.stage_idx + 1):
                targets_to_invalidate.add(changed_stage)

            for target_stage in targets_to_invalidate:
                if target_stage in self._peer_dealer_cache:
                    sock, _ = self._peer_dealer_cache.pop(target_stage)
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass
                    self.log.info(
                        "ConnexTE stage %s: Invalidated cached DEALER for peer stage %s due to membership change",
                        self.stage_idx,
                        target_stage,
                    )

                # Reset forward/return sockets so next request reconnects.
                if target_stage == self.stage_idx + 1:
                    self.forward_socket = None
                elif target_stage == 0 and self.is_last:
                    self.return_socket = None
        except Exception as exc:
            self.log.warning(
                "ConnexTE stage %s: Error handling ENDPOINT_REGISTERED: %s",
                self.stage_idx, exc
            )

    def _on_endpoint_unhealthy(self, event: Any) -> None:
        """UNHEALTHY = the peer missed its lease. Under load that is often a
        starved heartbeat of a live engine, so it is not a departure: remember
        it as a suspect if it is the downstream peer we route to and probe it
        (fast handover): a refused connect confirms the death at once. Stage 0
        also probes replicas farther down the chain, whose death loses its
        in-flight steps although its own peer is fine."""
        lid_dict = event.payload.get("lid", {}) if hasattr(event, "payload") else {}
        st, rep = lid_dict.get("stage_idx", -1), lid_dict.get("replica_idx", -1)
        if st == self.stage_idx + 1 and getattr(self, "_sticky_peer", {}).get(st) == rep:
            self._suspect_replicas.add((st, rep))
            self.log.warning("ConnexTE stage %s: downstream peer stage %s replica %s missed its "
                           "lease (suspect; probing)", self.stage_idx, st, rep)
            self._probe_async(st, rep)
        elif self.is_first and st > 1 and self.num_stages > 2:
            self._probe_async(st, rep)

    # ---- fast handover ---------------------------------------------------------
    def _init_handover(self) -> None:
        self._chain_epoch = 0
        self._remote_epoch = 0
        self._k1_sent = None
        self._sent_at = OrderedDict()
        self._probing = set()

    def _peer_addr(self, stage: int, replica: int) -> str | None:
        """Address of (stage, replica) from the local route cache (no RPC)."""
        epc = getattr(self, "_epc_client", None)
        if epc is None:
            return None
        try:
            for r in epc.get_pipeline_peers():
                if r.lid.stage_idx == stage and r.lid.replica_idx == replica and r.addresses:
                    return r.addresses[0]
        except Exception:  # noqa: BLE001
            return None
        return None

    def _probe_replica(self, stage: int, replica: int) -> bool | None:
        """True = alive, False = dead, None = unknown (no address / disabled)."""
        if not _FAST_HANDOVER:
            return None
        return _probe_tcp(self._peer_addr(stage, replica))

    def _probe_async(self, stage: int, replica: int) -> None:
        if not _FAST_HANDOVER:
            return
        if self._probing is None:
            self._probing = set()
        key = (stage, replica)
        if key in self._probing or self._peer_addr(stage, replica) is None:
            return
        self._probing.add(key)
        threading.Thread(target=self._probe_and_act, args=key, daemon=True,
                         name=f"connexte-probe-{stage}-{replica}").start()

    def _probe_and_act(self, stage: int, replica: int) -> None:
        try:
            alive = self._probe_replica(stage, replica)
            if alive is not False:
                if alive:
                    self.log.info("ConnexTE stage %s: stage %s replica %s missed its lease but "
                                "accepts connections (live; keeping it)",
                                self.stage_idx, stage, replica)
                return
            self.on_peer_dead(stage, replica)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("ConnexTE stage %s: probe of stage %s replica %s failed: %s",
                           self.stage_idx, stage, replica, exc)
        finally:
            self._probing.discard((stage, replica))

    def on_peer_dead(self, stage: int, replica: int, delay: bool = True,
                     source: str = "probe") -> None:
        """(stage, replica) is confirmed dead. Our direct peer: fail over now
        (bumps the chain epoch). Farther down (stage 0 only): the stage just
        upstream of it fails over itself; stage 0 bumps the remote epoch so
        its overdue in-flight steps are re-sent at once. source="transport"
        (a connex-rs peer-failure event): every stage learns at once, so the
        grace is short and, at K=1, a step counts as overdue sooner."""
        if stage == self.stage_idx + 1:
            if getattr(self, "_sticky_peer", {}).get(stage) != replica:
                return
            self._suspect_replicas.discard((stage, replica))
            self.log.warning("ConnexTE stage %s: downstream peer stage %s replica %s is dead "
                           "(probe refused); failing over", self.stage_idx, stage, replica)

            class _E:
                payload = {"lid": {"stage_idx": stage, "replica_idx": replica}}
            self._on_endpoint_deregistered(_E())
        elif self.is_first and stage > 1:
            grace = _TRANSPORT_FAILOVER_DELAY_S if source == "transport" else _REMOTE_FAILOVER_DELAY_S
            if delay and grace > 0:
                time.sleep(grace)
            self._remote_epoch += 1
            if source == "transport":
                self._remote_fast_epoch = self._remote_epoch
            self.log.warning("ConnexTE stage 0: stage %s replica %s is dead (probe refused); "
                           "re-sending overdue in-flight steps", stage, replica)

    def _mark_sent(self, corr: int | None = None) -> None:
        """Stage 0: remember the chain epoch a step was (re)sent under."""
        rec = (self._chain_epoch, self._remote_epoch, time.monotonic())
        if self._inflight > 1 and corr is not None:
            if self._sent_at is None:
                self._sent_at = OrderedDict()
            self._sent_at[corr] = rec
            self._sent_at.move_to_end(corr)
            while len(self._sent_at) > 8 * self._inflight + 16:
                self._sent_at.popitem(last=False)
        else:
            self._k1_sent = rec

    def _overdue_s(self) -> float:
        """A step older than this is overdue: with K steps in flight a reply
        normally comes within ~K recent step times."""
        r = sorted(self._recent_step_s[-16:])
        med = r[len(r) // 2] if r else 0.5
        return max(0.5, 2.0 * med * max(1, self._inflight))

    def _send_is_stale(self, rec) -> bool:
        """The step sent under ``rec`` was lost to a failover: re-send it now."""
        if rec is None or not _FAST_HANDOVER:
            return False
        if rec[0] != self._chain_epoch:
            return True
        if rec[1] == self._remote_epoch:
            return False
        overdue = self._overdue_s()
        if self._inflight == 1 and self._remote_fast_epoch == self._remote_epoch:
            # K=1 after a transport-confirmed death: a downstream stage
            # handles one step at a time, so a re-sent step that was not
            # lost arrives after the original and is answered from its reply
            # cache -- an early re-send costs a message, never a recompute.
            r = sorted(self._recent_step_s[-16:])
            med = r[len(r) // 2] if r else 0.05
            overdue = max(0.1, 2.0 * med)
        return time.monotonic() - rec[2] >= overdue

    def step_stale(self, corr: int) -> bool:
        """K>1: step ``corr`` was sent before a failover of its chain."""
        return self._send_is_stale((self._sent_at or {}).get(corr))

    def _wait_rec(self):
        if self._inflight > 1:
            corr = getattr(self._tls, "wait_corr", None)
            return (self._sent_at or {}).get(corr) if corr is not None else None
        return self._k1_sent

    def downstream_suspect(self) -> bool:
        st = self.stage_idx + 1
        cur = getattr(self, "_sticky_peer", {}).get(st)
        return cur is not None and (st, cur) in self._suspect_replicas

    def confirm_peer_failure(self) -> bool:
        """Data-path suspicion: a step to the downstream peer timed out.
        Probe the peer; a refused connect means dead -> fail over now. If it
        cannot be probed (no address), fall back to the lease: a suspect peer
        (missed lease) plus a timeout is taken as dead. A peer that accepts
        connections is kept (slow, not dead). Returns True on failover."""
        st = self.stage_idx + 1
        cur = getattr(self, "_sticky_peer", {}).get(st)
        if cur is None:
            return False
        alive = self._probe_replica(st, cur)
        if alive:
            return False
        if alive is None and (st, cur) not in self._suspect_replicas:
            return False
        self._suspect_replicas.discard((st, cur))
        self.log.warning("ConnexTE stage %s: peer stage %s replica %s confirmed dead by a "
                       "data-path timeout (%s); failing over", self.stage_idx, st, cur,
                       "probe refused" if alive is False else "suspect")

        class _E:
            payload = {"lid": {"stage_idx": st, "replica_idx": cur}}
        self._on_endpoint_deregistered(_E())
        return True

    def _io_lock(self):
        """Lock serialising use/close of the DEALER sockets (K>1: sends come
        from the signal thread; events and recv timeouts from others)."""
        lk = getattr(self, "_send_lock", None)
        if lk is None:
            import contextlib
            return contextlib.nullcontext()
        return lk

    def _on_endpoint_deregistered(self, event: Any) -> None:
        """Handle ENDPOINT_DEREGISTERED event from Front Router.

        When a peer deregisters, we must:
        1. Track it so future peer selection avoids this replica
        2. Immediately invalidate ALL cached connections to the affected stage
        3. Reset forward/return sockets to force reconnection to surviving replicas
        """
        try:
            lid_dict = event.payload.get("lid", {})
            changed_stage = lid_dict.get("stage_idx", -1)
            changed_replica = lid_dict.get("replica_idx", -1)

            self.log.warning(
                "ConnexTE stage %s: ENDPOINT_DEREGISTERED for stage %s replica %s "
                "— marking as unavailable and invalidating connections",
                self.stage_idx, changed_stage, changed_replica
            )

            # Track deregistered replica to avoid routing to it
            self._deregistered_replicas.add((changed_stage, changed_replica))
            if self.is_first and changed_stage > 1:
                self._remote_epoch += 1  # may be on our chain: re-send overdue steps
            sticky = getattr(self, "_sticky_peer", {})
            if (changed_stage in sticky and sticky[changed_stage] != changed_replica
                    and not (changed_stage == 0 and self.is_last)):
                return  # not our peer: keep our chain as it is
            sticky.pop(changed_stage, None)
            if changed_stage == self.stage_idx + 1:
                # every step already sent to the old peer is lost
                self._chain_epoch += 1

            with self._io_lock(), self._peer_lock:
                # Close cached DEALER socket for the affected stage
                if changed_stage in self._peer_dealer_cache:
                    sock, old_endpoint = self._peer_dealer_cache.pop(changed_stage)
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass
                    self._last_failed_peer[changed_stage] = old_endpoint
                    self.log.info(
                        "ConnexTE stage %s: Closed stale DEALER to stage %s (endpoint=%s)",
                        self.stage_idx, changed_stage, old_endpoint
                    )

                # Reset forward/return socket so next request reconnects to surviving replica
                if changed_stage == self.stage_idx + 1:
                    if self.forward_socket is not None:
                        try:
                            self.forward_socket.close()
                        except Exception:
                            pass
                    self.forward_socket = None
                    self.log.info(
                        "ConnexTE stage %s: Reset forward_socket — next send will route to surviving replica",
                        self.stage_idx
                    )
                elif changed_stage == 0 and self.is_last:
                    if self.return_socket is not None:
                        try:
                            self.return_socket.close()
                        except Exception:
                            pass
                    self.return_socket = None
                    self.log.info(
                        "ConnexTE stage %s: Reset return_socket — next return will route to surviving replica",
                        self.stage_idx
                    )

                # Return sockets keyed by originator endpoint ("return_<ep>")
                # are left alone: every reply names its own stage-0 replica,
                # and closing the socket to a *surviving* stage 0 here (the
                # old behaviour) dropped its in-flight replies.
        except Exception as exc:
            self.log.warning(
                "ConnexTE stage %s: Error handling ENDPOINT_DEREGISTERED: %s",
                self.stage_idx, exc
            )

    def _setup_mooncake_engine(self) -> None:
        """Initialise the transfer engine (Mooncake from the environment, or
        the injected one) and, on CUDA devices, the GPUDirect buffers."""
        injected, injected_url = getattr(self, "_injected_engine", (None, None))
        backend = getattr(self, "_backend", "mooncake")
        if injected is None and getattr(self, "_crs", None) is not None:
            injected, injected_url = self._crs
        elif injected is None and backend == "zmq":
            self.log.info("Stage %s: data plane backend zmq: using the ZeroMQ path", self.stage_idx)
            return
        elif injected is None and backend == "connex-rs":
            self.log.warning("Stage %s: connex-rs unavailable; trying Mooncake", self.stage_idx)
        if injected is None and (not MOONCAKE_AVAILABLE or TransferEngineClass is None):
            self.log.info(
                "Stage %s: Mooncake not available, using ZeroMQ fallback",
                self.stage_idx,
            )
            return

        try:
            if injected is not None:
                self._mooncake_engine, self._local_url = injected, injected_url
            else:
                self._mooncake_engine, self._local_url = _transports.create_mooncake_engine(
                    self.stage_idx, self.front_router_addr, self.net, self.log,
                    TransferEngineClass)
                if self._mooncake_engine is None:
                    return

            err = _transports.verify_engine(self._mooncake_engine)
            if err is not None:
                self.log.warning(
                    "Stage %s: Mooncake verification failed: %s. Falling back to ZeroMQ.",
                    self.stage_idx,
                    err,
                )
                self._mooncake_available = False
                self._mooncake_engine = None
                return

            self._buffer_cleaner = ThreadPoolExecutor(max_workers=1)
            self._mooncake_available = True
            self._start_peer_watch()
            self._gdr = None
            if _GDR and str(getattr(self, "device", "cpu")).startswith("cuda"):
                try:
                    if _GDR_HOST == "auto":
                        self._gdr_host = (self._inflight > 1 and
                                          os.getenv("MOONCAKE_PROTOCOL", "tcp") == "tcp")
                    else:
                        self._gdr_host = _GDR_HOST == "1"
                    self._gdr_kw = (dict(device="cpu", pin_memory=True) if self._gdr_host
                                    else dict(device="cuda"))
                    send_buf = torch.empty(_GDR_SEND_MB << 20, dtype=torch.uint8, **self._gdr_kw)
                    if self._mooncake_engine.register_memory(send_buf.data_ptr(),
                                                             send_buf.numel()) != 0:
                        raise RuntimeError("register_memory failed for GPU send ring")
                    self._gdr_recv = torch.empty(_GDR_RECV_MB << 20, dtype=torch.uint8, **self._gdr_kw)
                    if self._mooncake_engine.register_memory(self._gdr_recv.data_ptr(),
                                                             self._gdr_recv.numel()) != 0:
                        raise RuntimeError("register_memory failed for GPU recv buffer")
                    self._gdr_buf = send_buf
                    self._gdr_send_evt = torch.cuda.Event()
                    self._gdr_ack_on = _GDR_ACK and _GDR_WIRE >= 2
                    nslots = max(1, _GDR_RECV_SLOTS)
                    self._gdr_slot_size = (self._gdr_recv.numel() // nslots) // 256 * 256
                    self._gdr_slot_evts = [torch.cuda.Event() for _ in range(nslots)]
                    self._gdr_slot_used = [False] * nslots
                    self._gdr_slot_cur: int | None = None
                    self._gdr = _RegionTable(send_buf.numel())
                    self.log.info("Stage %s: ConnexTE GPUDirect path on (send ring %d MB, recv %d MB, "
                                "staging in %s memory)", self.stage_idx, _GDR_SEND_MB, _GDR_RECV_MB,
                                "pinned host" if self._gdr_host else "GPU")
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("Stage %s: GPUDirect path unavailable (%s); using host buffers",
                                   self.stage_idx, exc)
                    self._gdr = None

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "Stage %s: Failed to initialize Mooncake: %s. Falling back to ZeroMQ.",
                self.stage_idx,
                exc,
            )
            self._mooncake_available = False
            self._mooncake_engine = None

    # ---- transport peer events (connex-rs) ---------------------------------
    # The engine reports a peer's death within milliseconds (UCX endpoint
    # error: the dead process's kernel tears its connections down). That is
    # taken as a confirmed death and drives the same failover as a refused
    # probe (on_peer_dead). A heartbeat or operation timeout only means hung
    # or unreachable: it marks a suspect and the TCP probe decides, like a
    # missed lease. Every stage keeps a connex-rs link to every peer
    # advertising a connex-rs URL, so stage 0 also hears about stages it does
    # not talk to directly.

    def _start_peer_watch(self) -> None:
        eng = self._mooncake_engine
        if not hasattr(eng, "wait_event"):
            return
        threading.Thread(target=self._peer_watch, args=(eng,), daemon=True,
                         name=f"connex-rs-watch-{self.stage_idx}").start()

    def _peer_watch(self, eng) -> None:
        last_scan = 0.0
        stats_s = float(os.getenv("CONNEX_RS_STATS_S", "60") or "0")
        last_stats = time.monotonic()
        stop = self._crs_watch_stop
        while not stop.is_set():
            try:
                ev = eng.wait_event(100)
            except Exception:  # noqa: BLE001  (engine closed)
                return
            if ev is not None:
                try:
                    self._on_transport_event(ev)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("Stage %s: transport event %s: %s", self.stage_idx, ev, exc)
            now = time.monotonic()
            if now - last_scan >= 1.0:
                last_scan = now
                self._crs_link_peers(eng)
            if stats_s > 0 and now - last_stats >= stats_s and hasattr(eng, "stats"):
                last_stats = now
                try:  # leak checks: endpoints, regions, fds, RSS over time
                    st = eng.stats()
                    fds = len(os.listdir("/proc/self/fd"))
                    with open("/proc/self/statm") as f:
                        rss_mb = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") >> 20
                    self.log.info("Stage %s: connex-rs stats eps %d/%d live %d regions %d (%d MB) "
                                  "rkeys %d inflight %d rma %d/%d failures %d fds %d rss %d MB",
                                  self.stage_idx, st["eps_created"], st["eps_closed"],
                                  st["live_peers"], st["regions"], st["region_bytes"] >> 20,
                                  st["remote_rkeys"], st["inflight"], st["rma_ok"], st["rma_err"],
                                  st["failures"], fds, rss_mb)
                except Exception:  # noqa: BLE001
                    pass

    def _crs_initiates(self, key: tuple) -> bool:
        """Whether this side opens the connex-rs link to peer `key`. Exactly
        one side of every pair does, and it is the side the data path
        connects from: a receiver reads from its sender, so a stage connects
        to the stage before it and stage 0 to the last stage (the return
        path). Were the watcher to link the other way, its connect could
        cross the data path's first read, and two simultaneous connects
        replace each other's endpoint on both sides: in-flight reads fail,
        a live peer is reported reset, and neither side reports the link up.
        Pairs with no data path between them link from the lower key."""
        s, n = self.stage_idx, self.num_stages
        t = key[0]
        if t != s and n > 1:
            if t == s - 1 or (s == 0 and t == n - 1):
                return True   # we read from it
            if t == s + 1 or (t == 0 and s == n - 1):
                return False  # it reads from us
        return key > (s, self._replica_idx())

    def _crs_link_peers(self, eng) -> None:
        """Connect to every peer replica's connex-rs URL we have no live
        link to (and that is not known dead), for the pairs this side opens
        (_crs_initiates): the other side links to us, so no pair connects
        both ways at once."""
        epc = getattr(self, "_epc_client", None)
        if epc is None:
            return
        try:
            peers = epc.get_pipeline_peers()
        except Exception:  # noqa: BLE001
            return
        for r in peers:
            key = (r.lid.stage_idx, r.lid.replica_idx)
            if key == (self.stage_idx, self._replica_idx()) or not self._crs_initiates(key) \
                    or key in self._crs_alive or key in self._crs_dead:
                continue
            url = next((a for a in r.addresses[1:] if "#" in a), None)
            if url is None:
                continue
            try:
                eng.connect(url)
            except Exception as exc:  # noqa: BLE001
                self.log.debug("Stage %s: connex-rs link to %s: %s", self.stage_idx, url, exc)

    def _on_transport_event(self, ev: dict) -> None:
        peer = ev.get("peer")
        if not peer:
            return
        key = (int(peer[1]), int(peer[3]))
        if ev["type"] == "up":  # link proven (round trip / peer hello), not just created
            if key not in self._crs_alive:
                self._crs_alive.add(key)
                self.log.info("Stage %s: connex-rs link to stage %s replica %s up",
                              self.stage_idx, *key)
            return
        if ev["type"] != "failed" or key not in self._crs_alive:
            return  # not a link we had (a reconnect attempt to a dead peer)
        self._crs_alive.discard(key)
        reason = str(ev.get("reason", ""))
        stage, replica = key
        if "timeout" in reason:
            self.log.warning("Stage %s: stage %s replica %s unresponsive (%s; suspect, probing)",
                             self.stage_idx, stage, replica, reason)
            self._suspect_replicas.add(key)
            self._probe_async(stage, replica)
            return
        self._crs_dead.add(key)
        self._deregistered_replicas.add(key)  # route around it until it registers again
        self.log.warning("Stage %s: stage %s replica %s failed (connex-rs: %s)",
                         self.stage_idx, stage, replica, reason)
        if self._m_enabled():
            self.metrics.inc("transport_peer_failures")
        # on_peer_dead may sleep (stage 0, remote failover grace): not here.
        threading.Thread(target=self.on_peer_dead, args=(stage, replica),
                         kwargs={"source": "transport"}, daemon=True,
                         name=f"connex-rs-dead-{stage}-{replica}").start()

    def _m_enabled(self) -> bool:
        m = getattr(self, "metrics", None)
        return bool(m is not None and getattr(m, "enabled", False))

    def _allocate_buffer(self, size: int) -> int:
        if not self._mooncake_available or self._mooncake_engine is None:
            raise RuntimeError("Mooncake engine not available")
        buffer_ptr = self._mooncake_engine.allocate_managed_buffer(size)
        if buffer_ptr <= 0:
            raise RuntimeError(f"Failed to allocate Mooncake buffer of size {size}")
        return buffer_ptr

    def _free_buffer(self, buffer_ptr: int, size: int) -> None:
        if not self._mooncake_available or self._mooncake_engine is None:
            return
        try:
            self._mooncake_engine.free_managed_buffer(buffer_ptr, size)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Stage %s: Failed to free buffer: %s", self.stage_idx, exc)

    def _write_to_buffer(self, buffer_ptr: int, data: bytes, length: int) -> None:
        if self._mooncake_engine is not None:
            self._mooncake_engine.write_bytes_to_buffer(buffer_ptr, data, length)

    def _read_from_buffer(self, buffer_ptr: int, length: int) -> bytes:
        if self._mooncake_engine is not None:
            return self._mooncake_engine.read_bytes_from_buffer(buffer_ptr, length)
        return b""

    def _transfer_sync(
        self, remote_url: str, dst_ptr: int, src_ptr: int, length: int
    ) -> int:
        if self._mooncake_engine is None:
            raise RuntimeError("Mooncake engine not available")
        ret = self._mooncake_engine.transfer_sync_read(
            remote_url, dst_ptr, src_ptr, length
        )
        if ret < 0:
            raise RuntimeError(f"Mooncake transfer failed with code {ret}")
        return ret

    # ---- small hot-path helpers -------------------------------------------
    def _ensure_sndtimeo(self, sock: Any) -> None:
        if self._sndtimeo_ok.get(id(sock)) is not sock:
            sock.setsockopt(zmq.SNDTIMEO, _CONNEXTE_SEND_TIMEOUT_MS)
            if len(self._sndtimeo_ok) > 64:
                self._sndtimeo_ok.clear()
            self._sndtimeo_ok[id(sock)] = sock

    def _handle_ack(self, data: bytes) -> None:
        if self._gdr is not None and len(data) >= len(_ACK_MARK) + _ACK.size:
            self._gdr.ack(_ACK.unpack_from(data, len(_ACK_MARK))[0])

    def _drain_acks(self) -> None:
        """Consume queued acks without blocking. Data messages read on the way
        are stashed for recv_tensor_dict (single-threaded use of the ROUTER)."""
        sock = self.router_socket
        while True:
            try:
                frames = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            if frames and frames[-1].startswith(_ACK_MARK):
                self._handle_ack(frames[-1])
            else:
                self._stash.append(frames)

    def _send_ack(self, ack_ep: str, token: int) -> None:
        """Tell the sender its ring region is free (RDMA read completed)."""
        sock = self._ack_socks.get(ack_ep)
        try:
            if sock is None:
                ep = ack_ep
                if not ep.startswith("tcp://"):
                    host, port = self.net.split_host_port(ep)
                    ep = self.net.get_tcp_uri(host, port)
                sock = self.context.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDHWM, 1024)
                sock.connect(ep)
                if len(self._ack_socks) > 32:
                    for old in self._ack_socks.values():
                        old.close()
                    self._ack_socks.clear()
                self._ack_socks[ack_ep] = sock
            sock.send(_ACK_MARK + _ACK.pack(token), zmq.NOBLOCK)
        except zmq.ZMQError as exc:  # sender falls back to its safety timeout
            self.log.debug("Stage %s: ack to %s not sent: %s", self.stage_idx, ack_ep, exc)

    def _gdr_region(self, total: int, want_ack: bool) -> tuple[torch.Tensor, int]:
        """Registered GPU region for one message: a ring slice, or a one-off
        registered buffer when every candidate ring slice is still unacked."""
        table = self._gdr
        hold = _GDR_ACK_TIMEOUT_S if want_ack else _GDR_HOLD_S
        if table.awaiting_ack() and self._inflight == 1:
            # K>1: a separate thread owns the ROUTER (recv/prefetch); acks
            # are consumed there by recv_tensor_dict instead.
            self._drain_acks()
        now = time.monotonic()
        for buf in table.collect(now):
            self._mooncake_engine.unregister_memory(buf.data_ptr())
        got = table.alloc(total, now, hold, want_ack)
        if got is not None:
            off, tok = got
            return self._gdr_buf[off:off + total], tok
        region = torch.empty(total, dtype=torch.uint8, **self._gdr_kw)
        rc = self._mooncake_engine.register_memory(region.data_ptr(), total)
        if rc != 0:
            raise RuntimeError(f"register_memory failed ({rc}) for {total} B temp send buffer")
        self.log.debug("Stage %s: send ring busy, temp buffer of %d B", self.stage_idx, total)
        return region, table.add_temp(region, now, hold, want_ack)

    def _send_gdr(self, tensor_dict: dict, dst: int | None, send_start: float,
                  defer: bool = False):
        """GPUDirect send. With defer=True (K>1 async send) the device copies
        are only enqueued on the caller's stream; returns (event, sig) and the
        caller's sender thread waits for the event and signals the peer."""
        prof = self._prof
        evt = None
        p0 = time.perf_counter()
        tensors, non_tensors = [], {}
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                tensors.append((k, v))
            else:
                non_tensors[k] = v
        origin = non_tensors.pop("__stage0_endpoint__", None)
        if self.is_first and origin is None:
            origin = self.router_endpoint
        elif not self.is_first and not self.is_last and self._origin_ep():
            origin = self._origin_ep()

        key = tuple((k, str(t.dtype).rsplit(".", 1)[-1], tuple(t.shape)) for k, t in tensors)
        layout, total, layout_b = self._layouts.layout_for(
            key, [t.numel() * t.element_size() for _, t in tensors])
        v2 = _GDR_WIRE >= 2 and layout_b is not None
        want_ack = v2 and self._gdr_ack_on
        ptr, tok = 0, 0
        if total:
            region, tok = self._gdr_region(total, want_ack)
            ptr = region.data_ptr()
            for (k, t), (_, _, _, o, nb) in zip(tensors, layout):
                if t.device.type != "cuda" and not self._gdr_host:
                    t = t.to("cuda", non_blocking=True)
                region[o:o + nb].copy_(t.contiguous().view(-1).view(torch.uint8), non_blocking=True)
            # Wait for the copies only (an event right behind them), not for
            # anything queued on the stream later.
            if defer:
                evt = torch.cuda.Event()
                evt.record()
            else:
                self._gdr_send_evt.record()
                self._gdr_send_evt.synchronize()
        p1 = time.perf_counter()

        url = self._local_url.encode("utf-8")
        corr = self._corr_for_send()
        if _TRACE:
            _tr(self.stage_idx, "send_called" if defer else "send_sync", corr)
        if v2:
            import pickle

            nt_b = pickle.dumps(non_tensors, protocol=pickle.HIGHEST_PROTOCOL) if non_tensors else b""
            ack_ep = self.router_endpoint.encode("utf-8") if want_ack and tok else b""
            tail = _encode_g2(tok if ack_ep else 0, layout_b, ack_ep,
                              (origin or "").encode("utf-8"), nt_b)
        else:
            import pickle

            nt = dict(non_tensors)
            if origin:
                nt["__stage0_endpoint__"] = origin
            tail = b"G1" + pickle.dumps({"layout": layout, "nt": nt},
                                        protocol=pickle.HIGHEST_PROTOCOL)
        sig = (b"CONNEXTE" + struct.pack("!QQI", ptr, total, len(url)) + url
               + struct.pack("!Q", corr) + tail)
        if not defer:
            self._send_via_zeromq(sig, dst)
            if _TRACE:
                _tr(self.stage_idx, "signaled", corr)
        if not self.is_first and self._inflight > 1:
            self._remember_done(corr, ("sig", sig), dst, tok)
        elif not self.is_first:
            # Keep the exact region alive for a duplicate request (_resend_last
            # re-signals it); vLLM may reuse the source tensors right away.
            if self._last_token is not None:
                self._gdr.unpin(self._last_token)
            self._last_token = tok or None
            if tok:
                self._gdr.pin(tok)
            self._last_done = (corr, ("sig", sig), dst)
        if prof is not None:
            prof.add("s_gdr_copy", p1 - p0)
            prof.add("s_signal", time.perf_counter() - p1)
        m = self.metrics
        if m.enabled:
            direction = "return" if self.is_last else "forward"
            m.observe("serialization_latency", p1 - p0)
            m.observe("send_latency", time.time() - send_start, direction=direction)
            m.inc("bytes_sent", total, direction=direction)
            m.inc("messages_sent", direction=direction)
        if defer:
            return evt, sig

    def _gdr_dst(self, total: int) -> tuple[torch.Tensor, bool]:
        """Destination inside the registered recv buffer. Returns (dst, views_ok).
        Waits only for the GPU work that last read this slot."""
        n = len(self._gdr_slot_evts)
        if n <= 1:
            if self._gdr_slot_used[0]:
                self._gdr_slot_evts[0].synchronize()  # previous clones done
            return self._gdr_recv, False
        cur = self._gdr_slot_cur
        if cur is not None:
            # By now the consumer of the previous slot has queued its kernels.
            self._gdr_slot_evts[cur].record()
            self._gdr_slot_used[cur] = True
        j = 0 if cur is None else (cur + 1) % n
        if self._gdr_slot_used[j]:
            self._gdr_slot_evts[j].synchronize()
        self._gdr_slot_cur = j
        base = j * self._gdr_slot_size
        return self._gdr_recv[base:base + self._gdr_slot_size], True

    def _recv_gdr_payload(self, layout: list, nt: dict, remote_url: str, src_ptr: int,
                          total: int, ack: tuple[str, int] | None = None) -> dict:
        result: dict[str, Any] = {}
        if total:
            temp = None
            if total <= self._gdr_slot_size:
                dst, views = self._gdr_dst(total)
                views = views and not self._gdr_host
            else:
                dst = temp = torch.empty(total, dtype=torch.uint8, **self._gdr_kw)
                views = False
                rc = self._mooncake_engine.register_memory(temp.data_ptr(), total)
                if rc != 0:
                    raise RuntimeError(f"register_memory failed ({rc}) for {total} B recv buffer")
            try:
                self._transfer_sync(remote_url, dst.data_ptr(), src_ptr, total)
                if ack is not None and ack[1]:
                    self._send_ack(*ack)  # read done: sender may reuse the region
                for k, dtype, shape, o, nb in layout:
                    v = dst[o:o + nb].view(getattr(torch, dtype)).view(shape)
                    if self._gdr_host:
                        # pinned host -> device, async on the caller's stream
                        result[k] = v.to(self.device, non_blocking=True)
                    else:
                        result[k] = v if views else v.clone()
                if temp is not None:
                    torch.cuda.current_stream().synchronize()
                elif not views:
                    # Clones are ordered before any later consumer on this
                    # stream; only the next RDMA into the buffer must wait.
                    self._gdr_slot_evts[0].record()
                    self._gdr_slot_used[0] = True
            finally:
                if temp is not None:
                    self._mooncake_engine.unregister_memory(temp.data_ptr())
        nt.pop("__send_timestamp__", None)
        if "__stage0_endpoint__" in nt:
            self._current_originator_endpoint = nt["__stage0_endpoint__"]
        result.update(nt)
        return result

    def _recv_gdr(self, header: bytes, remote_url: str, src_ptr: int, total: int) -> dict:
        """Legacy "G1" (pickled header)."""
        import pickle

        info = pickle.loads(header)
        return self._recv_gdr_payload(info["layout"], info["nt"], remote_url, src_ptr, total)

    def _recv_gdr2(self, body: bytes, remote_url: str, src_ptr: int, total: int) -> dict:
        token, layout_b, ack_ep, origin, nt_b = _decode_g2(body)
        if nt_b:
            import pickle

            nt = pickle.loads(nt_b)
        else:
            nt = {}
        if origin:
            nt["__stage0_endpoint__"] = origin
        layout = self._layouts.parse(layout_b) if layout_b else []
        return self._recv_gdr_payload(layout, nt, remote_url, src_ptr, total,
                                      (ack_ep, token) if ack_ep else None)

    def _corr_for_send(self) -> int:
        if self._inflight > 1:
            c = getattr(self._tls, "corr", None)
            if c is not None:
                return c
        if not self.is_first:
            return self._corr_in or 0
        if self._pending_corr is None:  # new step; a retry keeps its id
            self._corr_seq += 1
            self._pending_corr = self._corr_base + self._corr_seq
        return self._pending_corr

    def _resend_last(self) -> None:
        """Answer a duplicate step with the reply we already produced."""
        corr, data, dst = self._last_done
        if isinstance(data, tuple) and data[0] == "sig":
            # GPUDirect: the region is pinned until the next reply, so the
            # original signal still points at intact bytes.
            self._send_via_zeromq(data[1], dst)
            return
        src_ptr = self._allocate_buffer(len(data))
        self._write_to_buffer(src_ptr, data, len(data))
        url = self._local_url.encode("utf-8")
        meta = struct.pack("!QQI", src_ptr, len(data), len(url)) + url + struct.pack("!Q", corr)
        self._send_via_zeromq(b"CONNEXTE" + meta, dst)
        if self._buffer_cleaner is not None:
            self._buffer_cleaner.submit(self._wait_for_ack_and_free, src_ptr, len(data))

    # ---- pipelining: K steps in flight (CONNEX_PP_INFLIGHT=K > 1) ----------
    #
    # Threading model with K > 1: one thread owns the ROUTER socket (stage 0:
    # whoever calls recv_step, serialised by _mb_recv_lock; later stages: the
    # prefetch thread calling recv_forward). Sends may come from another
    # thread; they are serialised by _send_lock (send_reply and duplicate
    # re-sends issued from the receiving thread share the DEALER sockets).
    # Per-step context (correlation id, originator endpoint, recv timeout)
    # travels in thread-local overrides instead of the shared _corr_in /
    # _current_originator_endpoint fields, which a prefetching receiver would
    # otherwise overwrite before the current step's reply is sent.

    def _init_inflight(self, k: int) -> None:
        self._inflight = max(1, int(k))
        if self._inflight > 1 and _ASYNC_SEND and _SWITCH_INTERVAL_S > 0:
            # The sender thread must get the GIL as soon as its copies are
            # done; with the default 5 ms switch interval a busy engine
            # thread delays every signal by up to 5 ms (measured: K=2 got
            # slower than synchronous sends).
            sys.setswitchinterval(_SWITCH_INTERVAL_S)
        self._tls = threading.local()
        self._mb_lock = threading.Lock()
        self._mb_recv_lock = threading.Lock()
        self._send_lock = threading.RLock()
        self._mb_outstanding = set()
        self._mb_done: "OrderedDict[int, dict]" = OrderedDict()
        self._step_timeouts: "OrderedDict[int, int]" = OrderedDict()
        # later stages: corr -> (data, dst, ring token, origin) of the last K replies
        self._done_ring: "OrderedDict[int, tuple]" = OrderedDict()
        self._corr_filter = _CorrFilter(window=self._inflight)

    @property
    def inflight(self) -> int:
        return self._inflight

    def _origin_ep(self) -> str | None:
        if self._inflight > 1:
            o = getattr(self._tls, "origin", None)
            if o:
                return o
        return self._current_originator_endpoint

    def _remember_done(self, corr: int, data, dst, tok: int | None) -> None:
        """K>1 replacement for _last_done/_last_token: keep the last K replies
        (and pin their GPU ring regions) so a retry of any in-flight step is
        answered from cache instead of being recomputed."""
        ring = self._done_ring
        old = ring.pop(corr, None)
        if old is not None and old[2] and self._gdr is not None:
            self._gdr.unpin(old[2])
        if tok and self._gdr is not None:
            self._gdr.pin(tok)
        ring[corr] = (data, dst, tok or None, self._origin_ep())
        while len(ring) > self._inflight:
            _, (_, _, t, _) = ring.popitem(last=False)
            if t and self._gdr is not None:
                self._gdr.unpin(t)
        self._last_done = (corr, data, dst)  # keeps K=1 introspection meaningful

    def _resend_done(self, corr: int) -> None:
        data, dst, _, origin = self._done_ring[corr]
        prev = getattr(self._tls, "origin", None)
        self._tls.origin = origin
        try:
            with self._send_lock:
                saved, self._last_done = self._last_done, (corr, data, dst)
                try:
                    self._resend_last()
                finally:
                    self._last_done = saved
        finally:
            self._tls.origin = prev

    # stage 0 ---------------------------------------------------------------
    def new_step_id(self) -> int:
        """Allocate the correlation id of a new pipeline step and mark it
        outstanding. A retry of that step must reuse the same id."""
        if self._inflight == 1:
            self._corr_seq += 1
            self._pending_corr = self._corr_base + self._corr_seq
            return self._pending_corr
        with self._mb_lock:
            self._corr_seq += 1
            corr = self._corr_base + self._corr_seq
            self._mb_outstanding.add(corr)
        return corr

    def send_step(self, tensor_dict: dict, corr: int, dst: int | None = None) -> None:
        """Send (or re-send, on retry) one pipeline step under id ``corr``."""
        if self._inflight == 1:
            self._pending_corr = corr
            return self.send_tensor_dict(tensor_dict, dst)
        prev = getattr(self._tls, "corr", None)
        self._tls.corr = corr
        try:
            with self._send_lock:
                self._mark_sent(corr)
                self.send_tensor_dict(tensor_dict, dst)
        finally:
            self._tls.corr = prev

    def send_step_async(self, tensor_dict: dict, corr: int, dst: int | None = None):
        """send_step that returns a Future instead of waiting for the GPU.

        K>1 + GPUDirect path only (else it sends synchronously and returns a
        completed Future): the copies into the registered send region are
        enqueued on the caller's CUDA stream right away (stream order makes
        them read the tensors before any later kernel overwrites them), and a
        single sender thread waits for those copies and signals the peer.
        Stage 0 can then go on with the next engine step while its forward is
        still running, instead of blocking in the send until it finishes."""
        if not self._async_send_ok():
            return self._done_future(self.send_step, tensor_dict, corr, dst)
        prev = getattr(self._tls, "corr", None)
        self._tls.corr = corr
        try:
            with self._send_lock:
                self._mark_sent(corr)
                evt, sig = self._send_gdr(tensor_dict, dst, time.time(), defer=True)
        finally:
            self._tls.corr = prev
        return self._submit_signal(evt, sig, dst, None)

    def _async_send_ok(self) -> bool:
        return (_ASYNC_SEND and self._inflight > 1 and self._mooncake_available
                and getattr(self, "_gdr", None) is not None)

    @staticmethod
    def _done_future(fn, *args):
        from concurrent.futures import Future

        f = Future()
        try:
            f.set_result(fn(*args))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f

    def _submit_signal(self, evt, sig: bytes, dst, origin):
        ex = getattr(self, "_signal_ex", None)
        if ex is None:
            ex = self._signal_ex = ThreadPoolExecutor(1, thread_name_prefix="connexte_signal")
        return ex.submit(self._signal_when_ready, evt, sig, dst, origin)

    def _signal_when_ready(self, evt, sig: bytes, dst, origin) -> None:
        if evt is not None:
            evt.synchronize()
        prev = getattr(self._tls, "origin", None)
        self._tls.origin = origin  # return-socket choice on the last stage
        try:
            with self._send_lock:
                self._send_via_zeromq(sig, dst)
            if _TRACE:
                _tr(self.stage_idx, "signaled", _signal_corr(sig))
        finally:
            self._tls.origin = prev

    def recv_step(self, corr: int, timeout_s: float | None = None) -> dict | None:
        """Wait for the reply of step ``corr``. Replies of other outstanding
        steps that arrive first are buffered for their own recv_step; replies
        of abandoned steps are dropped. Returns None on timeout (the caller
        re-sends this step with send_step and calls recv_step again)."""
        if self._inflight == 1:
            return self.recv_tensor_dict()
        if timeout_s is None:
            timeout_s = self._recv_timeout_ms() / 1000.0
        deadline = time.monotonic() + timeout_s
        while True:
            with self._mb_recv_lock:
                with self._mb_lock:
                    if corr in self._mb_done:
                        self._mb_outstanding.discard(corr)
                        return self._mb_done.pop(corr)
                if self.step_stale(corr):
                    # sent before a failover of its chain: lost, re-send now
                    self.log.warning("Stage %s: step %x was in flight on a failed chain; "
                                   "re-sending now", self.stage_idx, corr)
                    return None
                remaining = deadline - time.monotonic()
                if remaining < _MIN_WAIT_S:
                    return self._step_timed_out(corr, timeout_s)
                self._tls.timeout_ms = max(1, int(remaining * 1000))
                self._tls.rx_corr = None
                self._tls.in_step = True
                self._tls.wait_corr = corr
                try:
                    res = self.recv_tensor_dict()
                finally:
                    self._tls.timeout_ms = None
                    self._tls.in_step = False
                    self._tls.wait_corr = None
                got = self._tls.rx_corr
            if res is None:
                # recv_tensor_dict times out on its own (wall-clock) deadline;
                # never spin on the sliver left on ours (the old "recv timed
                # out after 0 ms" bursts, each closing the forward socket).
                if got is None and deadline - time.monotonic() < _MIN_WAIT_S:
                    return self._step_timed_out(corr, timeout_s)
                continue
            with self._mb_lock:
                if got == corr:
                    self._mb_outstanding.discard(corr)
                    return res
                if got in self._mb_outstanding:
                    self._mb_done[got] = res

    def _step_timed_out(self, corr: int, timeout_s: float) -> None:
        """One recv_step attempt for ``corr`` expired (K>1). Logged once per
        attempt. The forward connection is kept unless the data path says the
        peer is gone: the downstream peer is suspect (missed lease), or this
        step already timed out twice (the reconnect is harmless with unique
        DEALER identities and recovers from a broken connection)."""
        n = self._step_timeouts.get(corr, 0) + 1
        self._step_timeouts[corr] = n
        while len(self._step_timeouts) > 4 * self._inflight + 8:
            self._step_timeouts.pop(next(iter(self._step_timeouts)))
        self.log.warning("Stage %s: no reply for step %x within %.0f ms (attempt %d)",
                       self.stage_idx, corr, timeout_s * 1000, n)
        if not self.confirm_peer_failure() and n >= 2:
            downstream = self.stage_idx + 1
            if downstream < self.num_stages:
                self._invalidate_stage_connection(downstream)
        return None

    def downstream_healthy(self) -> bool:
        """The downstream peer we route to is known and not departed/suspect."""
        st = self.stage_idx + 1
        if st >= self.num_stages:
            return True
        cur = getattr(self, "_sticky_peer", {}).get(st)
        return (cur is not None and (st, cur) not in self._deregistered_replicas
                and (st, cur) not in self._suspect_replicas)

    def abandon_step(self, corr: int) -> None:
        """Give up on a step: its late reply (if any) will be dropped."""
        with self._mb_lock:
            self._mb_outstanding.discard(corr)
            self._mb_done.pop(corr, None)
            self._step_timeouts.pop(corr, None)
        if self._pending_corr == corr:
            self._pending_corr = None

    # stages 1.. --------------------------------------------------------------
    def recv_forward(self) -> tuple[dict | None, tuple | None]:
        """Receive the next step from upstream. Returns (tensor_dict, ctx);
        pass ctx to send_reply for that step's output. Safe to call from a
        prefetch thread while another thread computes/sends the previous step."""
        if self._inflight > 1:
            self._tls.rx_corr = None
        td = self.recv_tensor_dict()
        if td is None:
            return None, None
        corr = getattr(self._tls, "rx_corr", None) if self._inflight > 1 else self._corr_in
        origin = td.get("__stage0_endpoint__") if isinstance(td, dict) else None
        return td, (corr, origin or self._current_originator_endpoint)

    def send_reply(self, tensor_dict: dict, ctx: tuple | None) -> None:
        """Send the output of the step received with ``ctx`` downstream (or
        back to its stage 0 when this is the last stage)."""
        if self._inflight == 1 or ctx is None:
            return self.send_tensor_dict(tensor_dict)
        corr, origin = ctx
        pc, po = getattr(self._tls, "corr", None), getattr(self._tls, "origin", None)
        self._tls.corr, self._tls.origin = corr, origin
        try:
            with self._send_lock:
                self.send_tensor_dict(tensor_dict)
        finally:
            self._tls.corr, self._tls.origin = pc, po

    def send_reply_async(self, tensor_dict: dict, ctx: tuple | None):
        """send_reply that returns a Future (see send_step_async): the copies
        are enqueued now on the caller's stream, the sender thread waits for
        them and signals, so the compute loop can launch the next step."""
        if not self._async_send_ok() or ctx is None:
            return self._done_future(self.send_reply, tensor_dict, ctx)
        corr, origin = ctx
        pc, po = getattr(self._tls, "corr", None), getattr(self._tls, "origin", None)
        self._tls.corr, self._tls.origin = corr, origin
        try:
            with self._send_lock:
                evt, sig = self._send_gdr(tensor_dict, None, time.time(), defer=True)
        finally:
            self._tls.corr, self._tls.origin = pc, po
        return self._submit_signal(evt, sig, None, origin)

    def _recv_timeout_ms(self) -> int:
        """Base timeout, stretched to 3x the slowest recent step so that a
        slow-but-healthy step (cold prefill, long batch) is not taken for a
        dead peer."""
        if self._inflight > 1:
            t = getattr(self._tls, "timeout_ms", None)
            if t is not None:
                return t
        base = _CONNEXTE_RECV_TIMEOUT_MS if _CONNEXTE_RECV_TIMEOUT_MS > 0 else 8000
        if self.is_first and self._recent_step_s:
            return int(max(base, 3000 * max(self._recent_step_s)))
        return base

    def _wait_for_ack_and_free(self, src_ptr: int, length: int) -> None:
        """Wait for receiver to complete RDMA transfer, then free buffer.

        Uses the same approach as mooncake: a conservative sleep to ensure
        the receiver has completed the RDMA READ before we free the buffer.
        """
        time.sleep(1.0)  # 1 second - same as mooncake, safe for RDMA
        self._free_buffer(src_ptr, length)

    def _get_peer_endpoint(self, target_stage_idx: int, exclude_replica: Optional[int] = None) -> Optional[str]:
        """Get peer endpoint from EPCV2Client route table.

        In multi-replica scenarios, we need to select a consistent peer
        based on our own rank among replicas at our stage, not the raw
        replica_idx value. This ensures proper 1:1 mapping between stages.

        With fault tolerance: if the preferred peer is unavailable (excluded),
        fall back to any other available peer at the target stage.

        Refreshes routes from the Front Router only when the sticky peer is
        unknown, failed, or deregistered (CONNEXTE_ROUTE_REFRESH_ALWAYS=1
        restores refresh-on-every-call). During deployment transitions, the EPC client's local cache may contain
        stale entries from previous deployments (race between registration and
        heartbeat-based cleanup).

        Example: If Stage 0 has replicas [2, 22] and Stage 1 has [24, 26]:
        - Stage 0 rank 0 (replica 2) → Stage 1 rank 0 (replica 24)
        - Stage 0 rank 1 (replica 22) → Stage 1 rank 1 (replica 26)

        Args:
            target_stage_idx: The stage to find a peer for
            exclude_replica: Optional replica_idx to exclude (for failover)
        """
        if self._epc_client is None:
            return None

        # Fast path: the sticky peer is still in the local route cache and
        # nothing told us it failed -> no route-table RPC (a full refresh is
        # ~20 ms at 2000 routes). Failures (_invalidate_stage_connection),
        # ROUTE_UPDATED and deregistration events force the slow path.
        current = self._sticky_peer.get(target_stage_idx)
        if (not _ROUTE_REFRESH_ALWAYS and current is not None
                and target_stage_idx not in self._route_dirty
                and current != exclude_replica
                and (target_stage_idx, current) not in self._deregistered_replicas):
            for r in self._epc_client.get_pipeline_peers():
                if (r.lid.stage_idx == target_stage_idx and r.lid.replica_idx == current
                        and r.addresses):
                    return r.addresses[0]

        # Slow path: refresh from the Front Router to avoid stale cache entries
        # (registration snapshots and missed PUB/SUB deregistrations). Use the
        # filtered per-stage API when the client has it.
        stage_fn = getattr(self._epc_client, "get_stage_peers", None)
        fresh_target = None
        try:
            if stage_fn is not None:
                try:  # slow path = sticky peer missing/failed: ask the router
                    fresh_target = stage_fn(target_stage_idx, refresh="always")
                except TypeError:
                    fresh_target = stage_fn(target_stage_idx)
                if not isinstance(fresh_target, list):
                    fresh_target = None
            if fresh_target is None:
                self._epc_client.refresh_routes()
        except Exception as e:
            self.log.warning("Stage %s: Failed to refresh routes: %s, using cache", self.stage_idx, e)
        self._route_dirty.discard(target_stage_idx)

        # Get all peers at target stage (exclude failed replica if specified)
        all_peers = self._epc_client.get_pipeline_peers()
        if fresh_target is not None:
            all_peers = [r for r in all_peers if r.lid.stage_idx != target_stage_idx] + fresh_target

        if self.log.isEnabledFor(10):  # DEBUG: the dump is large at scale
            self.log.debug(
                "Stage %s: Full route table (%d entries): %s",
                self.stage_idx, len(all_peers),
                [(r.lid.stage_idx, r.lid.replica_idx, r.addresses[0] if r.addresses else "N/A")
                 for r in all_peers],
            )

        # Selection rules (stale-address filter, departed replicas skipped,
        # sticky, else rank based): connex.pp.routing.select_peer.
        lid = getattr(self._epc_client, "lid", None)
        addr, rep, new_choice = select_peer(
            all_peers, target_stage_idx, self.stage_idx,
            getattr(lid, "replica_idx", None), current, self._deregistered_replicas,
            exclude_replica, self.log)
        if new_choice:
            self._sticky_peer[target_stage_idx] = rep
        return addr

    def _get_dealer_for_stage(self, target_stage_idx: int, force_refresh: bool = False) -> Any:
        """Get or create DEALER socket for target stage. Use EPCV2Client if available.

        Args:
            target_stage_idx: Target stage index to connect to
            force_refresh: If True, ignore cache and get fresh endpoint
        """
        # Use cached socket if available and not force refreshing
        if target_stage_idx in self._peer_dealer_cache and not force_refresh:
            sock, cached_endpoint = self._peer_dealer_cache[target_stage_idx]
            if sock is not None:
                return sock

        endpoint = self._get_peer_endpoint(target_stage_idx)
        if endpoint:
            if not endpoint.startswith("tcp://"):
                host, port = self.net.split_host_port(endpoint)
                endpoint = self.net.get_tcp_uri(host, port)
            dealer = self.context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 1000)
            dealer.setsockopt(zmq.IDENTITY, _dealer_identity(f"rk-{target_stage_idx}"))
            if self.timeout_ms > 0:
                dealer.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            dealer.connect(endpoint)
            self._peer_dealer_cache[target_stage_idx] = (dealer, endpoint)
            if self.metrics.enabled:
                self.metrics.set("connection_status", 1, socket_type="dealer")
            return dealer

        # Fallback: wait for endpoint to become available
        # _get_peer_endpoint already refreshes routes, so no need to call
        # refresh_routes() separately here.
        if self._epc_client:
            for _ in range(30):
                endpoint = self._get_peer_endpoint(target_stage_idx)
                if endpoint:
                    break
                time.sleep(1.0)
        if not endpoint:
            if self.control_socket is not None:
                direction = "forward" if target_stage_idx > self.stage_idx else "return"
                endpoint, routing_key = self._request_assignment(
                    target_stage_idx, direction
                )
                return self._connect_dealer(endpoint, routing_key)
            raise RuntimeError(
                f"ConnexTE stage {self.stage_idx}: Peer stage {target_stage_idx} not found. "
                "Ensure Connex Front Router is running and all stages have registered."
            )
        if not endpoint.startswith("tcp://"):
            host, port = self.net.split_host_port(endpoint)
            endpoint = self.net.get_tcp_uri(host, port)
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 1000)
        dealer.setsockopt(zmq.IDENTITY, _dealer_identity(f"rk-{target_stage_idx}"))
        if self.timeout_ms > 0:
            dealer.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        dealer.connect(endpoint)
        self._peer_dealer_cache[target_stage_idx] = (dealer, endpoint)
        return dealer

    def _send_via_zeromq(self, data: bytes, dst: int | None) -> None:
        """Send data via ZeroMQ with retry and failover on failure.

        If the first send attempt fails (timeout or error), invalidate the
        cached connection, refresh routes from Front Router, and retry with
        a potentially different peer (surviving replica).
        """
        if dst is None:
            dst = (self.stage_idx + 1) % self.num_stages

        max_retries = _CONNEXTE_SEND_RETRIES
        last_error = None

        for attempt in range(max_retries):
            try:
                if dst == 0 and self.is_last:
                    socket = self._get_return_socket()
                elif dst == (self.stage_idx + 1) % self.num_stages and not self.is_last:
                    socket = self._get_forward_socket()
                else:
                    raise RuntimeError(
                        f"Unsupported destination stage {dst} from stage {self.stage_idx}"
                    )

                # Set a finite send timeout to detect dead peers
                self._ensure_sndtimeo(socket)
                socket.send(data)
                return  # Success

            except zmq.Again as e:
                # ZMQ send timed out — peer is likely dead
                last_error = e
                self.log.warning(
                    "ConnexTE stage %s: ZMQ send to stage %s timed out (attempt %d/%d), "
                    "invalidating connection and retrying with failover",
                    self.stage_idx, dst, attempt + 1, max_retries
                )
                self._invalidate_stage_connection(dst)

            except zmq.ZMQError as e:
                last_error = e
                self.log.warning(
                    "ConnexTE stage %s: ZMQ send to stage %s failed: %s (attempt %d/%d), "
                    "invalidating connection and retrying",
                    self.stage_idx, dst, e, attempt + 1, max_retries
                )
                self._invalidate_stage_connection(dst)

        raise RuntimeError(
            f"ConnexTE stage {self.stage_idx}: Failed to send to stage {dst} "
            f"after {max_retries} attempts. Last error: {last_error}"
        )

    def _invalidate_stage_connection(self, target_stage: int) -> None:
        """Invalidate all cached connections to a specific stage and force reconnection."""
        self._route_dirty.add(target_stage)  # next lookup refreshes routes
        with self._io_lock(), self._peer_lock:
            if target_stage in self._peer_dealer_cache:
                sock, old_endpoint = self._peer_dealer_cache.pop(target_stage)
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                self._last_failed_peer[target_stage] = old_endpoint
                self.log.info(
                    "ConnexTE stage %s: Invalidated connection to stage %s (was %s)",
                    self.stage_idx, target_stage, old_endpoint
                )

            if target_stage == self.stage_idx + 1:
                if self.forward_socket is not None:
                    try:
                        self.forward_socket.close()
                    except Exception:
                        pass
                self.forward_socket = None
            elif target_stage == 0 and self.is_last:
                if self.return_socket is not None:
                    try:
                        self.return_socket.close()
                    except Exception:
                        pass
                self.return_socket = None

    def _invalidate_peer_cache(self) -> None:
        """Invalidate peer DEALER cache on migration events."""
        for stage_idx, (sock, _) in list(self._peer_dealer_cache.items()):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        self._peer_dealer_cache.clear()
        self.forward_socket = None
        self.return_socket = None

    def _get_forward_socket(self) -> Any:
        if self.is_last:
            raise RuntimeError("Last stage has no forward socket")
        if self.forward_socket is None:
            self.forward_socket = self._get_dealer_for_stage(self.stage_idx + 1)
            if self.forward_socket is None:
                endpoint, routing_key = self._request_assignment(
                    self.stage_idx + 1, "forward"
                )
                self.forward_socket = self._connect_dealer(endpoint, routing_key)
        return self.forward_socket

    def _get_return_socket(self) -> Any:
        """Get return socket for sending results back to stage 0.

        In multi-replica scenarios, uses the originator endpoint saved during
        recv_tensor_dict to ensure we return to the correct stage 0 replica.
        """
        if not self.is_last:
            raise RuntimeError("Return socket only on last stage")

        # Multi-replica: use originator endpoint if available
        origin_ep = self._origin_ep()
        if origin_ep:
            # Check if we already have a socket to this endpoint
            cache_key = f"return_{origin_ep}"
            if cache_key in self._peer_dealer_cache:
                sock, _ = self._peer_dealer_cache[cache_key]
                if sock is not None:
                    return sock

            # Create new DEALER to originator
            endpoint = origin_ep
            if not endpoint.startswith("tcp://"):
                host, port = self.net.split_host_port(endpoint)
                endpoint = self.net.get_tcp_uri(host, port)

            dealer = self.context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 1000)
            dealer.setsockopt(zmq.IDENTITY, _dealer_identity(f"rk-return-{self.stage_idx}"))
            if self.timeout_ms > 0:
                dealer.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            dealer.connect(endpoint)
            self._peer_dealer_cache[cache_key] = (dealer, endpoint)

            self.log.debug(
                "Stage %s: Created return socket to originator %s",
                self.stage_idx, endpoint
            )
            return dealer

        # Fallback: single-replica mode or no originator info
        if self.return_socket is None:
            self.return_socket = self._get_dealer_for_stage(0)
            if self.return_socket is None:
                endpoint, routing_key = self._request_assignment(0, "return")
                self.return_socket = self._connect_dealer(endpoint, routing_key)
        return self.return_socket

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        """Send via Mooncake TE if available, else fallback to ZeroMQ."""
        if self.is_first and self._inflight == 1:
            self._mark_sent()
        if not self._mooncake_available:
            return super().send_tensor_dict(tensor_dict, dst)

        import pickle

        send_start = time.time()
        prof = self._prof
        if prof is not None:
            p0 = time.perf_counter()
            if prof.last_recv_done is not None:
                prof.add("compute_gap", p0 - prof.last_recv_done)
        if getattr(self, "_gdr", None) is not None:
            return self._send_gdr(tensor_dict, dst, send_start)
        try:
            tensors = {}
            non_tensors = {}
            for k, v in tensor_dict.items():
                if isinstance(v, torch.Tensor):
                    # Use simple .cpu() like mooncake — pin_memory() has high
                    # per-call overhead that compounds across decode steps.
                    tensors[k] = v.cpu() if v.device.type != "cpu" else v
                else:
                    non_tensors[k] = v

            # Multi-replica support:
            # - stage 0 adds its endpoint for return routing
            # - middle stages pass through saved originator endpoint
            if self.is_first and "__stage0_endpoint__" not in non_tensors:
                non_tensors["__stage0_endpoint__"] = self.router_endpoint
            elif not self.is_first and not self.is_last and self._origin_ep():
                # Middle stage: pass through originator endpoint
                non_tensors["__stage0_endpoint__"] = self._origin_ep()

            if prof is not None:
                p1 = time.perf_counter(); prof.add("s_tocpu", p1 - p0)
            tensor_bytes = safetensors_save(tensors) if tensors else b""
            non_tensor_bytes = pickle.dumps(non_tensors)

            send_timestamp_value = time.time()
            tensor_len = len(tensor_bytes)
            combined_data = (
                struct.pack("!Qd", tensor_len, send_timestamp_value)
                + tensor_bytes
                + non_tensor_bytes
            )
            total_length = len(combined_data)
            direction = "return" if self.is_last else "forward"

            m = self.metrics
            if m.enabled:
                m.observe("serialization_latency", time.time() - send_start)
                m.observe("tensor_count", len(tensors), direction=direction)
                m.observe("tensor_size", total_length, direction=direction)

            if prof is not None:
                p2 = time.perf_counter(); prof.add("s_serialize", p2 - p1)
            src_ptr = self._allocate_buffer(total_length)
            if prof is not None:
                p3 = time.perf_counter(); prof.add("s_alloc", p3 - p2)
            self._write_to_buffer(src_ptr, combined_data, total_length)
            if prof is not None:
                p4 = time.perf_counter(); prof.add("s_copyin", p4 - p3)

            sender_url_bytes = self._local_url.encode("utf-8")
            corr = self._corr_for_send()
            metadata = (
                struct.pack("!QQI", src_ptr, total_length, len(sender_url_bytes))
                + sender_url_bytes
                + struct.pack("!Q", corr)
            )
            marker = b"CONNEXTE"

            enqueue_start = time.time()
            self._send_via_zeromq(marker + metadata, dst)
            enqueue_time = time.time() - enqueue_start
            if prof is not None:
                prof.add("s_signal", time.perf_counter() - p4)

            if self._buffer_cleaner is not None:
                self._buffer_cleaner.submit(
                    self._wait_for_ack_and_free, src_ptr, total_length
                )
            if not self.is_first and self._inflight > 1:
                self._remember_done(corr, combined_data, dst, None)
            elif not self.is_first:
                self._last_done = (corr, combined_data, dst)
            send_time = time.time() - send_start
            if m.enabled:
                m.inc("bytes_sent", total_length, direction=direction)
                m.inc("messages_sent", direction=direction)
                m.observe("send_latency", send_time, direction=direction)
                m.observe("enqueue_latency", enqueue_time, direction=direction)

        except Exception as exc:
            self.log.error("Stage %s: ConnexTE send failed: %s", self.stage_idx, exc)
            raise RuntimeError(f"Stage {self.stage_idx}: ConnexTE send failed: {exc}") from exc

    def recv_tensor_dict(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Receive via Mooncake TE if available, else fallback to ZeroMQ.

        Churn resilience: Uses a finite recv timeout (_CONNEXTE_RECV_TIMEOUT_MS)
        so that if a downstream peer dies mid-pipeline, recv doesn't block forever.
        On timeout, returns None so the caller can retry the full send+recv cycle.
        """
        if not self._mooncake_available:
            return super().recv_tensor_dict(src)

        import pickle

        recv_start = time.time()
        try:
            # Always set a finite recv timeout for churn resilience
            deadline = time.time() + self._recv_timeout_ms() / 1000.0
            r0 = time.perf_counter()
            # K=1: the fork runs send and recv on different threads, so the
            # recv may start before the send of its step is marked; judge
            # staleness by a failover *during* this wait instead.
            k1_rec = ((self._chain_epoch, self._remote_epoch, time.monotonic())
                      if self.is_first and self._inflight == 1 else None)
            while True:
                remaining_ms = int((deadline - time.time()) * 1000)
                if remaining_ms <= 0:
                    raise zmq.Again()
                if self._stash:
                    frames = self._stash.pop(0)
                else:
                    # poll instead of a per-call RCVTIMEO setsockopt; stage 0
                    # polls in slices to notice a failover of its chain
                    rec = None
                    if self.is_first and _FAST_HANDOVER:
                        rec = k1_rec if k1_rec is not None else self._wait_rec()
                    if rec is not None:
                        if self._send_is_stale(rec):
                            raise _ChainFailed()
                        if not self.router_socket.poll(min(remaining_ms, _EPOCH_POLL_MS), zmq.POLLIN):
                            continue
                    elif not self.router_socket.poll(remaining_ms, zmq.POLLIN):
                        raise zmq.Again()
                    try:
                        frames = self.router_socket.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        # another thread (_drain_acks on the send path) took
                        # the message we polled: it is in the stash. Not a
                        # timeout (that was a spurious "timed out after 0 ms",
                        # a needless re-send).
                        continue
                meta_received_time = time.time()
                r1 = time.perf_counter()
                if len(frames) < 2:
                    return None
                data = frames[-1]
                if data.startswith(_ACK_MARK):
                    self._handle_ack(data)
                    continue
                marker = b"CONNEXTE"
                if not data.startswith(marker):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Expected CONNEXTE message, got different format"
                    )
                metadata = data[len(marker) :]
                if len(metadata) < 20:
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Invalid metadata length"
                    )
                src_ptr, total_length, sender_url_len = struct.unpack(
                    "!QQI", metadata[:20]
                )
                if len(metadata) < 20 + sender_url_len:
                    raise RuntimeError(f"Stage {self.stage_idx}: Incomplete sender URL")
                remote_url = metadata[20 : 20 + sender_url_len].decode("utf-8")
                tail = metadata[20 + sender_url_len :]
                corr = struct.unpack("!Q", tail[:8])[0] if len(tail) >= 8 else None
                if corr is not None and self.is_first and self._mb_outstanding:
                    # K>1 per-step API in use: accept any outstanding step.
                    if corr not in self._mb_outstanding:
                        self.log.warning("Stage %s: dropping reply for abandoned step %x",
                                       self.stage_idx, corr)
                        continue
                elif corr is not None and self.is_first and self._inflight == 1 \
                        and self._last_reply_corr is not None \
                        and (corr >> 32) == (self._last_reply_corr >> 32) \
                        and corr <= self._last_reply_corr:
                    # K=1: a late duplicate of a step already answered (the
                    # step was re-sent). Never take it for the current step,
                    # even before the send thread has set _pending_corr.
                    self.log.warning("Stage %s: dropping duplicate reply for answered step %x",
                                   self.stage_idx, corr)
                    continue
                elif corr is not None and self.is_first and self._pending_corr is not None \
                        and corr != self._pending_corr:
                    self.log.warning("Stage %s: dropping reply for abandoned step %x (waiting for %x)",
                                   self.stage_idx, corr, self._pending_corr)
                    continue
                if corr is not None and not self.is_first and self._inflight > 1 \
                        and corr in self._done_ring:
                    self.log.warning("Stage %s: duplicate step %x (sender retry); re-sending cached reply",
                                   self.stage_idx, corr)
                    self._resend_done(corr)
                    continue
                if corr is not None and not self.is_first and self._last_done is not None \
                        and corr == self._last_done[0]:
                    self.log.warning("Stage %s: duplicate step %x (sender retry); re-sending cached reply",
                                   self.stage_idx, corr)
                    self._resend_last()
                    continue
                if _STALE_DROP and not self.is_first and not self._corr_filter.accept(corr):
                    self.log.warning("Stage %s: dropping stale step %x (newer step already seen)",
                                   self.stage_idx, corr)
                    continue
                break

            if _TRACE:
                _tr(self.stage_idx, "arrived", corr)
            gtag = tail[8:10]
            if gtag == b"G2" or gtag == b"G1":
                if gtag == b"G2":
                    result = self._recv_gdr2(tail[10:], remote_url, src_ptr, total_length)
                else:
                    result = self._recv_gdr(tail[10:], remote_url, src_ptr, total_length)
                if self.is_first:
                    self._pending_corr = None
                    self._last_reply_corr = corr
                    self._recent_step_s = (self._recent_step_s + [time.time() - recv_start])[-32:]
                else:
                    self._corr_in = corr
                if self._inflight > 1:
                    self._tls.rx_corr = corr
                if _TRACE:
                    _tr(self.stage_idx, "recv_done", corr)
                if self._prof is not None:
                    r6 = time.perf_counter()
                    self._prof.add("r_wait", r1 - r0)
                    self._prof.add("r_gdr", r6 - r1)
                    self._prof.last_recv_done = r6
                    self._prof.tick()
                m = self.metrics
                if m.enabled:
                    direction = "return" if self.is_first else "forward"
                    m.inc("bytes_received", total_length, direction=direction)
                    m.inc("messages_received", direction=direction)
                    m.observe("recv_latency", time.time() - recv_start, direction=direction)
                return result
            dst_ptr = self._allocate_buffer(total_length)
            r2 = time.perf_counter()
            direction = "return" if self.is_first else "forward"

            try:
                self._transfer_sync(remote_url, dst_ptr, src_ptr, total_length)
                r3 = time.perf_counter()
                combined_data = self._read_from_buffer(dst_ptr, total_length)
            finally:
                self._free_buffer(dst_ptr, total_length)
            r4 = time.perf_counter()
            if self.is_first:
                self._pending_corr = None
                self._last_reply_corr = corr
                self._recent_step_s = (self._recent_step_s + [time.time() - recv_start])[-32:]
            else:
                self._corr_in = corr
            if self._inflight > 1:
                self._tls.rx_corr = corr

            data_received_time = time.time()
            tensor_len, send_timestamp = struct.unpack("!Qd", combined_data[:16])
            tensor_bytes = combined_data[16 : 16 + tensor_len]
            non_tensor_bytes = combined_data[16 + tensor_len :]

            result = {}
            if tensor_bytes:
                tensors = safetensors_load(tensor_bytes)
                for k, v in tensors.items():
                    result[k] = v
            if non_tensor_bytes:
                non_tensors = pickle.loads(non_tensor_bytes)
                non_tensors.pop("__send_timestamp__", None)

                # Multi-replica support: save originator endpoint for return path
                # For last stage: use it for return socket
                # For middle stages: pass it through to next stage
                if "__stage0_endpoint__" in non_tensors:
                    self._current_originator_endpoint = non_tensors["__stage0_endpoint__"]
                    if self.is_last:
                        self.log.debug(
                            "Stage %s: Saved originator endpoint %s for return",
                            self.stage_idx, self._current_originator_endpoint
                        )

                result.update(non_tensors)

            r5 = time.perf_counter()
            if tensor_bytes and self.device.startswith("cuda"):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.to(self.device, non_blocking=True)
                torch.cuda.current_stream().synchronize()
            if self._prof is not None:
                r6 = time.perf_counter()
                for name, dt in (("r_wait", r1 - r0), ("r_alloc", r2 - r1), ("r_rdma", r3 - r2),
                                 ("r_copyout_free", r4 - r3), ("r_deser", r5 - r4), ("r_tocuda", r6 - r5)):
                    self._prof.add(name, dt)
                self._prof.last_recv_done = r6
                self._prof.tick()

            recv_time = time.time() - recv_start
            m = self.metrics
            if m.enabled:
                m.inc("bytes_received", total_length, direction=direction)
                m.inc("messages_received", direction=direction)
                m.observe("recv_latency", recv_time, direction=direction)
                m.observe("delivery_latency_local",
                          max(0.0, data_received_time - meta_received_time), direction=direction)
                if send_timestamp > 0:
                    dl = data_received_time - send_timestamp
                    if dl >= 0:
                        m.observe("delivery_latency", dl, direction=direction)

            return result

        except _ChainFailed:
            # The chain failed over while we waited: the step is lost, the
            # caller re-sends it now (on the new chain) instead of waiting out
            # the timeout. Sockets were already reset by the failover.
            if not (self._inflight > 1 and getattr(self._tls, "in_step", False)):
                self.log.warning("Stage %s: downstream chain failed over while waiting; "
                               "returning None so the step is re-sent now", self.stage_idx)
            return None

        except zmq.Again:
            if self._inflight > 1 and getattr(self._tls, "in_step", False):
                return None  # recv_step decides (logs once, keeps the socket)
            # Recv timed out — peer may be dead (churn scenario).
            # Return None so the caller (gpu_model_runner stage-0 loop) can
            # invalidate connections and retry the full send+recv cycle.
            elapsed_ms = (time.time() - recv_start) * 1000
            self.log.warning(
                "Stage %s: ConnexTE recv timed out after %.0f ms "
                "(CONNEXTE_RECV_TIMEOUT_MS=%d). Peer may have failed during churn. "
                "Returning None for retry.",
                self.stage_idx, elapsed_ms, _CONNEXTE_RECV_TIMEOUT_MS
            )
            # Probe the peer: a dead one is failed over now (not after the
            # lease + grace); otherwise invalidate the downstream connection.
            if self.is_first and self.confirm_peer_failure():
                return None
            downstream = self.stage_idx + 1
            if downstream < self.num_stages:
                self._invalidate_stage_connection(downstream)
            return None

        except Exception as exc:
            if self.is_first and "Mooncake transfer failed" in str(exc):
                # The last stage announced a reply and died before we pulled
                # the payload. Treat it like a timeout so the step is re-sent
                # (idempotent) and fails over, instead of killing the engine.
                self.log.warning(
                    "Stage %s: reply payload pull failed (%s); peer gone, "
                    "treating as timeout", self.stage_idx, exc)
                if self._inflight > 1 and getattr(self._tls, "in_step", False):
                    return None  # recv_step decides
                downstream = self.stage_idx + 1
                if downstream < self.num_stages:
                    self._invalidate_stage_connection(downstream)
                return None
            self.log.error(
                "Stage %s: ConnexTE recv failed: %s", self.stage_idx, exc
            )
            raise RuntimeError(
                f"Stage {self.stage_idx}: ConnexTE recv failed: {exc}"
            ) from exc

    def get_connexte_metrics(self) -> dict[str, Any]:
        """Get ConnexTE-specific metrics."""
        return {
            "mooncake_available": self._mooncake_available,
            "epc_client_available": self._epc_client is not None,
            "local_url": self._local_url,
            "remote_urls": self._remote_urls,
        }

    def close(self) -> None:
        stop = getattr(self, "_crs_watch_stop", None)
        if stop is not None:
            stop.set()
        crs = getattr(self, "_crs", None)
        if self._epc_client is not None:
            self._epc_client.stop()
            self._epc_client = None
        # Sender thread (K>1 async sends) first: it uses the DEALER sockets.
        ex = getattr(self, "_signal_ex", None)
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
            self._signal_ex = None
        self._invalidate_peer_cache()
        # Ack DEALERs (GPUDirect path): an unclosed socket makes
        # Context.term() block forever.
        for sock in getattr(self, "_ack_socks", {}).values():
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass
        self._ack_socks = {}
        if self._buffer_cleaner is not None:
            self._buffer_cleaner.shutdown(wait=False)
            self._buffer_cleaner = None
        if crs is not None:
            try:
                crs[0].close()
            except Exception:  # noqa: BLE001
                pass
            self._crs = None
        return super().close()


# Names kept importable from here for code written against the old module.
_REEXPORTS = (_Lease, _DT_CODE, _DT_NAMES, _G2, _align, _decode_layout, _encode_layout)

# Public alias: the stage endpoint of the core API.
PipelineEndpoint = ConnexTEEndpoint
