# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Host-staged KV transfer for hosts without GPUDirect RDMA.

A GPU whose host has no ``nvidia_peermem`` (or another peer-memory driver)
cannot expose device memory to the NIC: registering the KV cache with the
data plane fails. Without staging, the only remaining path is UCX over TCP,
which copies through the host on every read and is an order of magnitude
slower (and, with HPC-X UCX 1.16, crashed the owner under load).

``HostStaging`` keeps a pinned host mirror of every registered KV span and
registers the mirror instead. Addresses map by a constant offset per span:

* producer: after a prefill finishes, the blocks of the held request are
  copied device -> mirror on the engine's stream (``stage_out``) and peers
  are told the mirror's addresses (``translate``); an acquire waits for the
  copy before it returns the blocks;
* consumer: reads land in the local mirror (``mirror_ops``) and are copied
  mirror -> device on a side stream (``stage_in``) before the pull counts as
  done.

Both sides stay one-sided RDMA reads between host buffers; the extra cost
is one PCIe copy on each side (about 10 ms for the 128 MB of KV of a
1024-token Llama-3-8B prompt).
"""
from __future__ import annotations

import bisect
import logging
import os
from typing import Optional, Sequence

log = logging.getLogger("connex.pd")


def gpudirect_available() -> bool:
    """A peer-memory driver that lets the NIC reach GPU memory is loaded
    (``nvidia_peermem``, or the older ``nv_peer_mem``). Without it UCX may
    still accept a registration of device memory (through its CUDA memory
    domain) while every RDMA read of it fails, so the registration result
    alone does not tell."""
    return any(os.path.exists(p) for p in ("/sys/module/nvidia_peermem",
                                           "/sys/module/nv_peer_mem",
                                           "/sys/kernel/mm/memory_peers/nv_mem"))


def staging_mode() -> str:
    """``CONNEX_PD_HOST_STAGING``: ``auto`` (default: stage when the plane
    cannot register device memory), ``1`` (always) or ``0`` (never)."""
    m = (os.getenv("CONNEX_PD_HOST_STAGING") or "auto").strip().lower()
    return {"true": "1", "yes": "1", "on": "1", "false": "0", "no": "0", "off": "0"}.get(m, m)


class _DevView:
    """``__cuda_array_interface__`` over a raw device range (no copy)."""

    def __init__(self, ptr: int, n: int):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


class HostStaging:
    def __init__(self, spans: Sequence[tuple[int, int]], device: Optional[int] = None):
        import torch

        self.torch = torch
        self.device = torch.cuda.current_device() if device is None else device
        self._starts: list[int] = []
        self._spans: list[tuple[int, int, object, object]] = []   # (start, n, dev view, mirror)
        for start, n in sorted(spans):
            dev = torch.as_tensor(_DevView(start, n), device=f"cuda:{self.device}")
            mirror = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            self._starts.append(start)
            self._spans.append((start, n, dev, mirror))
        self._stream = torch.cuda.Stream(device=self.device)
        log.info("connex PD: host staging of %d KV spans (%.1f GB pinned)", len(self._spans),
                 sum(n for _, n, _, _ in self._spans) / 1e9)

    # -- addresses ---------------------------------------------------------------
    def _find(self, addr: int):
        i = bisect.bisect_right(self._starts, addr) - 1
        if i < 0:
            raise ValueError(f"{addr:#x} is not in a staged KV span")
        start, n, dev, mirror = self._spans[i]
        if addr >= start + n:
            raise ValueError(f"{addr:#x} is not in a staged KV span")
        return start, dev, mirror

    def mirror(self, addr: int) -> int:
        start, _, m = self._find(addr)
        return m.data_ptr() + (addr - start)

    def mirror_spans(self) -> list[tuple[int, int]]:
        return [(m.data_ptr(), n) for _, n, _, m in self._spans]

    def translate(self, layout):
        """The layout with its base moved into the mirror (what peers read)."""
        from dataclasses import replace

        return replace(layout, base=self.mirror(layout.base))

    def mirror_ops(self, ops):
        return [(self.mirror(d), s, n) for d, s, n in ops]

    # -- copies ----------------------------------------------------------------
    def _copy(self, ranges, to_host: bool) -> None:
        for addr, n in _merge(ranges):
            start, dev, m = self._find(addr)
            o = addr - start
            if to_host:
                m[o:o + n].copy_(dev[o:o + n], non_blocking=True)
            else:
                dev[o:o + n].copy_(m[o:o + n], non_blocking=True)

    def stage_out(self, ranges) -> object:
        """Copy device ranges into the mirror on the current (engine) stream,
        after the work that wrote them; returns an event to wait on."""
        torch = self.torch
        with torch.cuda.device(self.device):
            self._copy(ranges, to_host=True)
            ev = torch.cuda.Event()
            ev.record()
        return ev

    def stage_in(self, ranges) -> None:
        """Copy mirror ranges to the device and wait (side stream)."""
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            self._copy(ranges, to_host=False)
        self._stream.synchronize()


def _merge(ranges) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, n in sorted(ranges):
        if out and a <= out[-1][0] + out[-1][1]:
            pa, pn = out[-1]
            out[-1] = (pa, max(pa + pn, a + n) - pa)
        else:
            out.append((a, n))
    return out


def block_ranges(layouts, blocks: Sequence[int]) -> list[tuple[int, int]]:
    """Byte ranges of ``blocks`` in every layout."""
    out = []
    for lo in layouts:
        for o in range(lo.outer):
            b0 = lo.base + o * lo.outer_stride
            for b in blocks:
                out.append((b0 + b * lo.block_stride, lo.seg_bytes))
    return out


def register_with_staging(plane, spans, what: str) -> Optional[HostStaging]:
    """Register ``spans`` with ``plane``; fall back to (or force) host
    staging when device memory cannot be registered. Returns the staging
    object (also kept as ``plane._connex_staging``) or None."""
    from .dataplane import is_device_ptr

    mode = staging_mode()
    staging = None
    if mode == "auto" and spans and is_device_ptr(spans[0][0]) and not gpudirect_available():
        log.warning("%s: no GPUDirect RDMA on this host (no peer-memory driver loaded); "
                    "staging KV through pinned host memory", what)
        mode = "1"
    if mode != "1":
        done = []
        for start, n in spans:
            rc = plane.register(start, n)
            if rc != 0:
                for s in done:
                    plane.unregister(s)
                if mode == "0" or not is_device_ptr(start):
                    raise RuntimeError(f"{plane.name}: register_memory({start:#x}, {n}) = {rc}")
                log.warning("%s: %s cannot register GPU memory (rc %d; no GPUDirect RDMA on "
                            "this host?); staging KV through pinned host memory",
                            what, plane.name, rc)
                break
            done.append(start)
        else:
            return None
    staging = HostStaging(spans)
    for start, n in staging.mirror_spans():
        rc = plane.register(start, n)
        if rc != 0:
            raise RuntimeError(f"{plane.name}: register_memory(host {start:#x}, {n}) = {rc}")
    plane._connex_staging = staging
    return staging
