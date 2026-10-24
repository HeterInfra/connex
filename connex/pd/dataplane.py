# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Data planes for KV block transfer: register memory, batched one-sided reads.

The PD transfer needs three things from a data plane: register the KV cache,
tell peers how to reach it (``local_url``), and read a batch of
``(dst, src, length)`` ranges from a peer, returning 0 or a negative code.

* ``ConnexRsPlane``: connex-rs (Rust over UCX; RDMA/TCP, GPUDirect). All
  reads of a batch are posted at once (``get_async``) and then waited for;
  a peer that dies fails the batch within milliseconds (code -2).
* ``MooncakePlane``: Mooncake Transfer Engine, ``batch_transfer_sync_read``.
* ``LoopbackPlane``: same-process ``memmove`` for host memory; tests only.

``make_plane()`` picks one from ``CONNEX_PD_BACKEND`` (``auto`` | ``connex-rs``
| ``mooncake`` | ``loopback``); ``auto`` follows the same rule as connex.pp:
connex-rs when installed and an RDMA device is present, else Mooncake.
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Optional, Protocol, Sequence

log = logging.getLogger("connex.pd")

ZCOPY_THRESH = 4096   # bytes; reads into GPU memory must be at least this long
PEER_FAILED = -2
TIMEOUT = -3
INVALID = -4


class KVDataPlane(Protocol):
    name: str
    local_url: str

    def register(self, ptr: int, length: int) -> int: ...
    def unregister(self, ptr: int) -> int: ...
    def read(self, peer_url: str, ops: Sequence[tuple[int, int, int]]) -> int: ...
    def close(self) -> None: ...


def _make_cuda_ctx_current() -> None:
    """Make this thread's CUDA context current before connex-rs captures it.

    connex-rs names GPU memory to UCX only if the registering thread has a
    current CUDA context (cuPointerGetAttribute). An engine may call
    register_kv_caches from a thread where the runtime never made its
    primary context current; UCX then treats the KV cache as host memory and
    copies into it with the CPU (segfault). cudaSetDevice fixes that."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.set_device(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        pass


class ConnexRsPlane:
    name = "connex-rs"

    def __init__(self, model: str, role_idx: int, replica: int, rank: int, ip: str, port: int = 0,
                 timeout_ms: int = 10000, engine_cls=None):
        if engine_cls is None:
            import connex_rs  # type: ignore

            engine_cls = connex_rs.Engine
        os.environ.setdefault("UCX_TCP_CM_REUSEADDR", "y")
        # UCX v1 RMA picks bcopy (a CPU memcpy into the destination) for reads
        # below its zero-copy threshold, whatever the memory type. Without a
        # registration cache that threshold grows past a KV block, and the
        # cache is unavailable in processes where UCX cannot install its
        # memory hooks (any process that imported OpenCV, e.g. vLLM): reads
        # into GPU memory then segfault. Pin the threshold below block size;
        # control messages (< 4 KB) stay bcopy.
        os.environ.setdefault("UCX_ZCOPY_THRESH", str(ZCOPY_THRESH))
        if not os.getenv("UCX_NET_DEVICES"):
            from connex.pp.transports import ucx_net_devices_for_ip

            nd = os.getenv("CONNEX_RS_NET_DEVICES", "auto")
            nd = ucx_net_devices_for_ip(ip) if nd == "auto" else nd
            if nd:
                os.environ["UCX_NET_DEVICES"] = nd
        # identity: (model, role, rank, replica); peer-death events and
        # same-identity rejoin come with it
        _make_cuda_ctx_current()
        self.e = engine_cls(model, role_idx, rank, replica, listen_ip=ip, port=port,
                            heartbeat_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_MS", "1000")),
                            heartbeat_timeout_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_TIMEOUT_MS", "3000")))
        self.local_url = self.e.local_url
        self.timeout_ms = timeout_ms

    def register(self, ptr: int, length: int) -> int:
        _make_cuda_ctx_current()
        return self.e.register_memory(ptr, length)

    def unregister(self, ptr: int) -> int:
        return self.e.unregister_memory(ptr)

    def read(self, peer_url: str, ops) -> int:
        handles = []
        try:
            for dst, src, n in ops:
                handles.append(self.e.get_async(peer_url, dst, src, n, self.timeout_ms))
        except Exception as e:  # noqa: BLE001  (peer unknown / failed while posting)
            log.warning("connex-rs: posting reads to %s failed: %s", peer_url, e)
            rc = PEER_FAILED
        else:
            rc = 0
        for h in handles:  # always drain: no late DMA after we return
            try:
                if not h.wait(self.timeout_ms + 1000):
                    rc = rc or TIMEOUT
            except Exception as e:  # noqa: BLE001
                if rc == 0:
                    log.warning("connex-rs: read from %s failed: %s", peer_url, e)
                rc = rc or PEER_FAILED
        return rc

    def close(self) -> None:
        try:
            self.e.close()
        except Exception:  # noqa: BLE001
            pass


class MooncakePlane:
    name = "mooncake"

    def __init__(self, ip: str, protocol: Optional[str] = None, device: Optional[str] = None,
                 engine_cls=None):
        if engine_cls is None:
            from mooncake.engine import TransferEngine  # type: ignore

            engine_cls = TransferEngine
        self.e = engine_cls()
        protocol = protocol or os.getenv("MOONCAKE_PROTOCOL", "rdma")
        device = device if device is not None else os.getenv("MOONCAKE_DEVICE_NAME", "")
        rc = self.e.initialize(ip, os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"),
                               protocol, device)
        if rc != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed: {rc}")
        self.local_url = f"{ip}:{self.e.get_rpc_port()}"

    def register(self, ptr: int, length: int) -> int:
        return self.e.register_memory(ptr, length)

    def unregister(self, ptr: int) -> int:
        return self.e.unregister_memory(ptr)

    def read(self, peer_url: str, ops) -> int:
        if not ops:
            return 0
        dst, src, n = (list(x) for x in zip(*ops))
        return self.e.batch_transfer_sync_read(peer_url, dst, src, n)

    def close(self) -> None:
        pass


class LoopbackPlane:
    """In-process data plane over host memory (tests, CPU demos). Peers are
    other LoopbackPlane objects in the same process, found by URL."""

    name = "loopback"
    _peers: dict[str, "LoopbackPlane"] = {}
    _lock = threading.Lock()
    _n = 0

    def __init__(self, *_, **__):
        with LoopbackPlane._lock:
            LoopbackPlane._n += 1
            self.local_url = f"loopback:{LoopbackPlane._n}"
            LoopbackPlane._peers[self.local_url] = self
        self.regions: list[tuple[int, int]] = []
        self.reads = 0

    def register(self, ptr: int, length: int) -> int:
        self.regions.append((ptr, length))
        return 0

    def unregister(self, ptr: int) -> int:
        self.regions = [r for r in self.regions if r[0] != ptr]
        return 0

    def read(self, peer_url: str, ops) -> int:
        peer = LoopbackPlane._peers.get(peer_url)
        if peer is None:
            return PEER_FAILED
        for dst, src, n in ops:
            if not any(s <= src and src + n <= s + ln for s, ln in peer.regions):
                return INVALID
            ctypes.memmove(dst, src, n)
            self.reads += 1
        return 0

    def close(self) -> None:
        with LoopbackPlane._lock:
            LoopbackPlane._peers.pop(self.local_url, None)


def backend_from_env() -> str:
    b = (os.getenv("CONNEX_PD_BACKEND") or "auto").strip().lower()
    b = {"connexrs": "connex-rs", "connex_rs": "connex-rs", "rs": "connex-rs"}.get(b, b)
    if b == "auto":
        import importlib.util

        from connex.pp.transports import rdma_present

        if importlib.util.find_spec("connex_rs") is not None and rdma_present():
            return "connex-rs"
        if importlib.util.find_spec("mooncake") is not None:
            return "mooncake"
        if importlib.util.find_spec("connex_rs") is not None:
            return "connex-rs"
        raise RuntimeError("no KV data plane: install connex-rs (connex-ai[rs]) or Mooncake "
                           "(connex-ai[mooncake]), or set CONNEX_PD_BACKEND=loopback for tests")
    if b not in ("connex-rs", "mooncake", "loopback"):
        raise ValueError(f"CONNEX_PD_BACKEND must be auto, connex-rs, mooncake or loopback, got {b!r}")
    return b


def make_plane(model: str, role_idx: int, replica: int, rank: int, ip: str,
               backend: Optional[str] = None, port: int = 0) -> KVDataPlane:
    backend = backend or backend_from_env()
    if backend == "connex-rs":
        return ConnexRsPlane(model, role_idx, replica, rank, ip, port)
    if backend == "mooncake":
        return MooncakePlane(ip)
    return LoopbackPlane()
