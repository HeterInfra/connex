# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Connex transfer backend for SGLang prefill-decode disaggregation.

    pip install "connex-ai[sglang,rs]"
    python -m sglang.launch_server ... --disaggregation-mode prefill \\
        --disaggregation-transfer-backend connex

SGLang's PD protocol (bootstrap server, decode pre-allocation, chunked sends
from prefill to decode, aux data with the first token, abort handling) is
kept exactly as in its Mooncake backend: ``ConnexKVManager`` *is* the
Mooncake KV manager with the transfer engine replaced by connex-rs. What
changes is the data plane underneath:

* connex-rs (Rust over UCX, GPUDirect RDMA) moves the KV pages with
  one-sided writes posted all at once and waited for together;
* peers are logical endpoints: a dead decode instance fails the writes to it
  within milliseconds (UCX endpoint error) instead of a transfer timeout, and
  an instance restarted under the same identity is reconnected without
  restarting its peers;
* instances can register in the Connex front-router PD pools
  (``CONNEX_PD_FRONT_ROUTER``, ``CONNEX_PD_HTTP_URL``, ``CONNEX_PD_REPLICA``),
  so ``connex.serving.pd_router --engine sglang`` follows scale up and down.

Activation needs no SGLang change: the ``sglang.srt.plugins`` entry point
``connex`` (installed with connex-ai) calls :func:`register`, which adds
``connex`` to ``--disaggregation-transfer-backend``, adds a ``CONNEX`` member
to ``TransferBackend`` and routes ``get_kv_class`` to the classes here. The
upstream hook this stands in for is a registry for out-of-tree transfer
backends (``TransferBackend`` is a closed enum and ``get_kv_class`` an
if-chain); see docs/pd-disaggregation.md.
"""
from __future__ import annotations

import logging
import os
import threading
import zlib
from typing import List

log = logging.getLogger("connex.pd.sglang")

BACKEND = "connex"


class ConnexRsTransferEngine:
    """The ``MooncakeTransferEngine`` surface SGLang's Mooncake backend uses,
    implemented with connex-rs. Session id = connex-rs URL
    (``ip:port#model|stage|shard|replica``)."""

    def __init__(self, hostname: str, role: str, gpu_id: int = 0, engine_cls=None,
                 timeout_ms: int = 30000):
        if engine_cls is None:
            import connex_rs  # type: ignore

            engine_cls = connex_rs.Engine
        from connex.pd.membership import PD_STAGE
        from connex.pp.transports import ucx_net_devices_for_ip

        from connex.pd.dataplane import ZCOPY_THRESH

        os.environ.setdefault("UCX_TCP_CM_REUSEADDR", "y")
        # see connex.pd.dataplane.ConnexRsPlane: no bcopy on GPU memory
        os.environ.setdefault("UCX_ZCOPY_THRESH", str(ZCOPY_THRESH))
        if not os.getenv("UCX_NET_DEVICES"):
            nd = os.getenv("CONNEX_RS_NET_DEVICES", "auto")
            nd = ucx_net_devices_for_ip(hostname) if nd == "auto" else nd
            if nd:
                os.environ["UCX_NET_DEVICES"] = nd
        replica = int(os.getenv("CONNEX_PD_REPLICA",
                                zlib.crc32(f"{hostname}:{os.getpid()}".encode()) & 0x7FFFFFF))
        port = int(os.getenv("CONNEX_RS_PORT", "0"))
        self.engine = engine_cls("sglang-pd", PD_STAGE.get(role, 1001), gpu_id, replica,
                                 listen_ip=hostname, port=port + gpu_id if port else 0,
                                 heartbeat_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_MS", "1000")),
                                 heartbeat_timeout_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_TIMEOUT_MS",
                                                                    "3000")))
        self.session_id = self.engine.local_url
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.timeout_ms = timeout_ms
        self.min_len = int(os.environ["UCX_ZCOPY_THRESH"]) if os.environ[
            "UCX_ZCOPY_THRESH"].isdigit() else 0
        self._lock = threading.Lock()
        log.info("connex-rs transfer engine at %s", self.session_id)

    # -- memory ------------------------------------------------------------------
    def register(self, ptr, length) -> None:
        rc = self.engine.register_memory(ptr, length)
        if rc != 0:
            raise RuntimeError(f"connex-rs memory registration failed (ret={rc})")

    def deregister(self, ptr) -> None:
        self.engine.unregister_memory(ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        rc = 0
        for p, n in zip(ptrs, lengths):
            rc = self.engine.register_memory(p, n) or rc
        return rc

    def batch_deregister(self, ptrs: List[int]) -> int:
        rc = 0
        for p in ptrs:
            rc = self.engine.unregister_memory(p) or rc
        return rc

    # -- transfers (prefill writes into decode memory) ----------------------------
    def transfer_sync(self, session_id: str, buffer: int, peer_buffer_address: int,
                      length: int) -> int:
        return self.engine.transfer_sync_write(session_id, buffer, peer_buffer_address, length,
                                               self.timeout_ms)

    def batch_transfer_sync(self, session_id: str, buffers: List[int],
                            peer_buffer_addresses: List[int], lengths: List[int]) -> int:
        short = [n for n in lengths if n < self.min_len]
        if short:
            # below UCX's zero-copy threshold the write is a CPU copy from GPU
            # memory (segfault); use --page-size >= 16 so pages are large enough
            log.error("connex-rs: %d KV chunks shorter than %d B (smallest %d B); "
                      "increase --page-size", len(short), self.min_len, min(short))
            return -4
        put = getattr(self.engine, "put_async", None)
        if put is None:  # connex-rs without put_async: one write at a time
            for b, p, n in zip(buffers, peer_buffer_addresses, lengths):
                rc = self.transfer_sync(session_id, b, p, n)
                if rc < 0:
                    return rc
            return 0
        handles, rc = [], 0
        try:
            for b, p, n in zip(buffers, peer_buffer_addresses, lengths):
                handles.append(put(session_id, b, p, n, self.timeout_ms))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-rs: posting writes to %s failed: %s", session_id, e)
            rc = -2
        for h in handles:  # drain everything posted before reporting
            try:
                if not h.wait(self.timeout_ms + 1000):
                    rc = rc or -3
            except Exception:  # noqa: BLE001
                rc = rc or -2
        return rc

    # -- misc --------------------------------------------------------------------
    def get_session_id(self):
        return self.session_id

    def send_probe(self, peer_session_id: str) -> int:
        try:
            self.engine.connect(peer_session_id)
            return 0
        except Exception:  # noqa: BLE001
            return -1

    def get_engine(self):
        return self.engine

    def get_ib_device(self):
        return os.getenv("UCX_NET_DEVICES", "")


_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_MEMBER = None


def get_engine(hostname: str, role: str, gpu_id: int) -> ConnexRsTransferEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = ConnexRsTransferEngine(hostname, role, gpu_id)
        return _ENGINE


def _maybe_register_member(role: str, rank: int) -> None:
    """Register this instance in the front-router PD pools (rank 0 only)."""
    global _MEMBER
    fr = os.getenv("CONNEX_PD_FRONT_ROUTER")
    url = os.getenv("CONNEX_PD_HTTP_URL")
    if not fr or not url or rank != 0 or _MEMBER is not None:
        return
    from connex.pd.membership import PDMember

    bootstrap = os.getenv("CONNEX_PD_BOOTSTRAP")   # host:port, advertised for prefill
    _MEMBER = PDMember(fr, os.getenv("CONNEX_PD_MODEL", "default"), role,
                       int(os.getenv("CONNEX_PD_REPLICA", "0")), url,
                       kv_addr=bootstrap if role == "prefill" else None)
    _MEMBER.start()


def _classes():
    """The backend classes; imports SGLang lazily."""
    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVBootstrapServer,
        MooncakeKVManager,
        MooncakeKVReceiver,
        MooncakeKVSender,
    )
    from sglang.srt.disaggregation.utils import DisaggregationMode

    class ConnexKVManager(MooncakeKVManager):
        def init_engine(self):
            role = ("prefill" if self.disaggregation_mode == DisaggregationMode.PREFILL
                    else "decode")
            self.engine = get_engine(self.local_ip, role, int(self.kv_args.gpu_id))
            try:
                _maybe_register_member(role, int(self.kv_args.engine_rank))
            except Exception as e:  # noqa: BLE001  membership is optional
                log.warning("front-router registration failed: %s", e)

    class ConnexKVSender(MooncakeKVSender):
        pass

    class ConnexKVReceiver(MooncakeKVReceiver):
        pass

    class ConnexKVBootstrapServer(MooncakeKVBootstrapServer):
        pass

    return ConnexKVManager, ConnexKVSender, ConnexKVReceiver, ConnexKVBootstrapServer


def _extend_enum(enum_cls, name: str, value: str):
    """Add a member to a closed Enum (what aenum.extend_enum does)."""
    if value in enum_cls._value2member_map_:
        return enum_cls._value2member_map_[value]
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    if hasattr(member, "_sort_order_"):
        member._sort_order_ = len(enum_cls._member_names_)
    enum_cls._member_map_[name] = member
    enum_cls._value2member_map_[value] = member
    enum_cls._member_names_.append(name)
    try:
        type.__setattr__(enum_cls, name, member)
    except AttributeError:
        pass
    return member


def register() -> None:
    """``sglang.srt.plugins`` entry point: make ``connex`` a transfer backend."""
    import sglang.srt.disaggregation.utils as du

    try:
        from sglang.srt.arg_groups.choices import (
            DISAGG_TRANSFER_BACKEND_CHOICES,
            add_disagg_transfer_backend_choices,
        )

        if BACKEND not in DISAGG_TRANSFER_BACKEND_CHOICES:
            add_disagg_transfer_backend_choices([BACKEND])
    except ImportError:  # older layout: choices live elsewhere
        log.warning("could not extend --disaggregation-transfer-backend choices")
    member = _extend_enum(du.TransferBackend, "CONNEX", BACKEND)
    if getattr(du.get_kv_class, "_connex", False):
        return
    orig = du.get_kv_class

    def get_kv_class(transfer_backend, class_type):
        if transfer_backend is member or getattr(transfer_backend, "value", None) == BACKEND:
            if class_type == du.KVClassType.KVARGS:
                from sglang.srt.disaggregation.base import KVArgs

                return KVArgs
            mgr, snd, rcv, boot = _classes()
            return {du.KVClassType.MANAGER: mgr, du.KVClassType.SENDER: snd,
                    du.KVClassType.RECEIVER: rcv,
                    du.KVClassType.BOOTSTRAP_SERVER: boot}.get(class_type)
        return orig(transfer_backend, class_type)

    get_kv_class._connex = True
    du.get_kv_class = get_kv_class
    # modules that imported get_kv_class by name before the plugin ran
    import sys

    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "get_kv_class", None) is orig:
            mod.get_kv_class = get_kv_class
    log.info("SGLang transfer backend %r registered (connex-rs data plane)", BACKEND)
