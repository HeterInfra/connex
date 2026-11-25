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


def vector_enabled() -> bool:
    """``CONNEX_PD_STAGING_VECTOR=0`` restores one copy per byte range."""
    return (os.getenv("CONNEX_PD_STAGING_VECTOR") or "1").strip().lower() not in (
        "0", "false", "no", "off")


def _mapped_device_ptr(ptr: int) -> Optional[int]:
    """Device address of pinned host memory (UVA), or None when the memory is
    not mapped into the device address space or the query is unavailable. A
    kernel touching unmapped host memory faults and poisons the context, so
    the vectorised path is only used when this says the mirror is mapped."""
    try:
        try:
            from cuda.bindings import runtime as rt
        except ImportError:
            from cuda import cudart as rt   # older cuda-python
        err, attr = rt.cudaPointerGetAttributes(ptr)
        if int(err) != 0 or attr.type != rt.cudaMemoryType.cudaMemoryTypeHost:
            return None
        return int(attr.devicePointer) or None
    except Exception:  # noqa: BLE001 - fall through to the ctypes route
        pass
    import ctypes

    class _Attr(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int),
                    ("devicePointer", ctypes.c_void_p), ("hostPointer", ctypes.c_void_p)]

    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        a = _Attr()
        if lib.cudaPointerGetAttributes(ctypes.byref(a), ctypes.c_void_p(ptr)) != 0 or a.type != 1:
            return None
        return int(a.devicePointer or 0) or None
    return None


def _merge_in_spans(ranges, starts, lens):
    """Sorted, merged byte ranges as numpy arrays ``(span, addr, n)``, or
    None when a range is not inside one span. Adjacent ranges merge only
    inside one span: neighbouring KV tensors are distinct spans with
    unrelated mirror addresses."""
    import numpy as np

    r = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    a, n = np.ascontiguousarray(r[:, 0]), np.ascontiguousarray(r[:, 1])
    if len(n) and n.min() <= 0:
        keep = n > 0
        a, n = a[keep], n[keep]
    if len(a) > 1 and not (a[1:] >= a[:-1]).all():
        order = np.argsort(a, kind="stable")
        a, n = a[order], n[order]
    sp = np.searchsorted(starts, a, side="right") - 1
    if len(a) and (sp.min() < 0 or (a + n > starts[sp] + lens[sp]).any()):
        return None
    if not len(a):
        return sp, a, n
    # the running max of ends is global, which is fine: spans are disjoint
    # and every range lies inside its span, so ends from earlier spans never
    # reach past the start of a later span's ranges
    end = np.maximum.accumulate(a + n)
    new = np.ones(len(a), dtype=bool)
    new[1:] = (a[1:] > end[:-1]) | (sp[1:] != sp[:-1])
    first = np.flatnonzero(new)
    gend = np.maximum.reduceat(a + n, first)
    return sp[first], a[first], gend - a[first]


def plan_rows(ranges, starts, lens, src_ptrs, dst_ptrs, src_base: int, dst_base: int,
              min_row: int = 64, max_row: int = 1 << 20):
    """Row plan for moving ``ranges`` (addresses in the KV spans) from the
    source side to the destination side, where span ``i`` sits at
    ``src_ptrs[i]`` and ``dst_ptrs[i]``.

    Every offset from a base and every length is a multiple of ``u`` (their
    gcd), so the move is "row ``sidx[k]`` of the source viewed as rows of
    ``u`` bytes goes to row ``didx[k]`` of the destination": one gather and
    one scatter whatever the number of ranges. ``e`` is the element size
    (up to 8 bytes) the rows are viewed with, so kernels move words, not
    bytes. Returns ``(u, e, idx)`` with ``idx`` an int64 array ``[2, rows]``,
    or None when the rows would be smaller than ``min_row`` (index arrays
    would rival the data) or a range is not inside one span. Rows are
    limited to the largest power of two in ``u`` up to ``max_row``: one row
    must fit the scratch buffer of a chunk."""
    import numpy as np

    merged = _merge_in_spans(ranges, starts, lens)
    if merged is None:
        return None
    sp, a, n = merged
    if not len(a):
        return 1, 1, np.zeros((2, 0), dtype=np.int64)
    rel = a - starts[sp]
    soff = src_ptrs[sp] + rel - src_base
    doff = dst_ptrs[sp] + rel - dst_base
    u = int(np.gcd.reduce(np.concatenate([soff, doff, n, [max_row]])))
    if u < min_row:
        return None
    e = 8
    while e > 1 and (u % e or src_base % e or dst_base % e):
        e //= 2
    cnt = n // u
    total = int(cnt.sum())
    # row k of range j: base row of j plus k - (first row of j in the output)
    first = np.cumsum(cnt) - cnt
    step = np.arange(total, dtype=np.int64) - np.repeat(first, cnt)
    idx = np.empty((2, total), dtype=np.int64)
    idx[0] = np.repeat(soff // u, cnt) + step
    idx[1] = np.repeat(doff // u, cnt) + step
    return u, e, idx


_DTYPE_BY_SIZE = {1: "uint8", 2: "int16", 4: "int32", 8: "int64"}


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
        # Built on the engine thread: its current stream is the engine's.
        # stage_in runs on a pull thread, where that stream is not current.
        self._engine_stream = torch.cuda.current_stream(self.device)
        self._vec = None
        if vector_enabled():
            try:
                self._vec = self._setup_vector()
            except Exception as e:  # noqa: BLE001 - per-range copies still work
                log.warning("connex PD: vectorised host staging unavailable (%s); "
                            "copying range by range", e)
        log.info("connex PD: host staging of %d KV spans (%.1f GB pinned, %s copies)",
                 len(self._spans), sum(n for _, n, _, _ in self._spans) / 1e9,
                 "vectorised" if self._vec else "per-range")

    # -- vectorised copies -------------------------------------------------------
    def _setup_vector(self):
        """Device views of the pinned mirrors (UVA), so one kernel can write
        device KV into the mirror at the right offsets (and read it back).
        A copy-engine memcpy per range is what this replaces: thousands of
        launches for a fragmented block set. None when not applicable."""
        import numpy as np

        torch = self.torch
        mptrs = []
        for _, _, _, m in self._spans:
            d = _mapped_device_ptr(m.data_ptr())
            if d is None:
                return None
            mptrs.append(d)
        starts = np.asarray(self._starts, dtype=np.int64)
        lens = np.asarray([n for _, n, _, _ in self._spans], dtype=np.int64)
        mptrs = np.asarray(mptrs, dtype=np.int64)
        vec = _Vector(starts, lens, mptrs, self._cuda_view, self._alloc, self._upload)
        # one round trip through the mapping before trusting it with KV
        start, n, dev, m = self._spans[0]
        k = min(n, 4096)
        with torch.cuda.device(self.device):
            m[:k].copy_(torch.arange(k, dtype=torch.int64).remainder_(251).to(torch.uint8))
            back = dev[:k].clone()
            dev[:k].zero_()
            vec.copy([(start, k)], to_host=False, chunk_bytes=1 << 30)
            torch.cuda.synchronize(self.device)
            ok = torch.equal(dev[:k].cpu(), m[:k])
            dev[:k].copy_(back)        # device -> host must bring these bytes back
            m[:k].zero_()
            vec.copy([(start, k)], to_host=True, chunk_bytes=1 << 30)
            torch.cuda.synchronize(self.device)
            ok = ok and torch.equal(m[:k], back.cpu())
            dev[:k].copy_(back)
            torch.cuda.synchronize(self.device)
        if not ok:
            raise RuntimeError("mapped mirror round trip mismatch")
        return vec

    def _cuda_view(self, ptr: int, nbytes: int, e: int):
        torch = self.torch
        v = torch.as_tensor(_DevView(ptr, nbytes), device=f"cuda:{self.device}")
        if v.data_ptr() != ptr:
            raise RuntimeError("device view was copied")
        return v.view(_torch_dtype(e)) if e > 1 else v

    def _alloc(self, nbytes: int):
        # caching allocator, on the current stream: freed blocks are reused
        # only by work queued after this copy on that stream
        return self.torch.empty(nbytes, dtype=self.torch.uint8, device=f"cuda:{self.device}")

    def _upload(self, idx, out) -> None:
        """Index rows to the device without blocking the calling thread: a
        pinned staging copy (the host allocator keeps it alive until the
        queued copy has read it), then an async H2D on the current stream.
        A pageable source could make the driver wait for the stream."""
        out.copy_(self.torch.from_numpy(idx).pin_memory(), non_blocking=True)

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
        if self._vec is not None:
            try:
                if self._vec.copy(ranges, to_host, _chunk_bytes()):
                    return
            except Exception as e:  # noqa: BLE001 - e.g. no device memory for the scratch
                # Copying the same bytes again range by range is harmless even
                # if part of the gather/scatter was already queued.
                if not getattr(self, "_vec_warned", False):
                    self._vec_warned = True
                    log.warning("connex PD: vectorised staging copy failed (%s); "
                                "copying range by range", e)
        self._copy_ranges(ranges, to_host)

    def _copy_ranges(self, ranges, to_host: bool) -> None:
        """One copy per merged range (fallback). A merged range that runs
        past its span (adjacent KV tensors) is split at the span end."""
        for addr, n in _merge(ranges):
            while n > 0:
                start, dev, m = self._find(addr)
                o = addr - start
                k = min(n, dev.numel() - o)
                if to_host:
                    m[o:o + k].copy_(dev[o:o + k], non_blocking=True)
                else:
                    dev[o:o + k].copy_(m[o:o + k], non_blocking=True)
                addr, n = addr + k, n - k

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
        """Copy mirror ranges to the device and wait (side stream). The copy
        is ordered after the work already queued on the engine stream: a step
        queued before the blocks were handed to this request (async
        scheduling) may still write them; it must not land after the KV."""
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            self._stream.wait_stream(self._engine_stream)
            self._copy(ranges, to_host=False)
        self._stream.synchronize()


def _chunk_bytes() -> int:
    """Bytes of device scratch one gather may use (``CONNEX_PD_STAGING_CHUNK_MB``,
    default 64): bounds the extra device memory a copy needs."""
    try:
        return max(1, int(os.getenv("CONNEX_PD_STAGING_CHUNK_MB") or 64)) << 20
    except ValueError:
        return 64 << 20


class _Vector:
    """Row-plan copies between the KV spans and their mirrors.

    Each side is addressed through one view that starts at its lowest span
    (resp. mirror) address and runs to the highest end, so ranges in any
    number of spans become one gather into a scratch buffer and one scatter
    out of it. The gaps between allocations are never indexed.

    torch rejects an indexed op whose output and an input have overlapping
    address ranges, and a device view across all KV spans can contain the
    scratch or index tensor when one of them was allocated in a gap between
    spans. The device view is therefore cut around those two tensors: at
    most three pieces, one in the common case.

    ``view(ptr, nbytes, e)`` makes a view over raw memory with elements of
    ``e`` bytes (a device tensor in production, a CPU one in tests);
    ``alloc(nbytes)`` returns a uint8 tensor on the copy's device;
    ``upload(np_array, tensor)`` fills a tensor allocated by ``alloc`` without
    blocking the caller."""

    def __init__(self, starts, lens, mirror_ptrs, view, alloc, upload):
        self.starts, self.lens, self.mirror_ptrs = starts, lens, mirror_ptrs
        self.dev_base = int(starts.min())
        self.dev_end = int((starts + lens).max())
        self.m_base = int(mirror_ptrs.min())
        self.m_end = int((mirror_ptrs + lens).max())
        self._view, self._alloc, self._upload = view, alloc, upload
        self._views: dict = {}
        self.launches = 0     # of the last copy (tests and benchmarks)
        self.pieces = 0

    def _rows(self, side: str, u: int, e: int):
        key = (side, u, e)
        v = self._views.get(key)
        if v is None:
            base, end = ((self.dev_base, self.dev_end) if side == "dev"
                         else (self.m_base, self.m_end))
            nrows = -(-(end - base) // u)
            v = self._views[key] = self._view(base, nrows * u, e).view(nrows, u // e)
        return v

    def _pieces(self, u: int, nrows: int, avoid) -> list[tuple[int, int]]:
        """Row intervals of the device view that hold none of ``avoid``
        (address ranges of tensors passed to the same kernels)."""
        cuts = []
        for a, b in avoid:
            if b > self.dev_base and a < self.dev_base + nrows * u:
                cuts.append((max(0, (a - self.dev_base) // u),
                             min(nrows, -(-(b - self.dev_base) // u))))
        out, r = [], 0
        for lo, hi in sorted(cuts):
            if lo > r:
                out.append((r, lo))
            r = max(r, hi)
        if r < nrows:
            out.append((r, nrows))
        return out

    def copy(self, ranges, to_host: bool, chunk_bytes: int) -> bool:
        """Queue the copy on the current stream; False when the plan does
        not apply (the caller copies range by range)."""
        import numpy as np

        if to_host:
            plan = plan_rows(ranges, self.starts, self.lens, self.starts, self.mirror_ptrs,
                             self.dev_base, self.m_base)
        else:
            plan = plan_rows(ranges, self.starts, self.lens, self.mirror_ptrs, self.starts,
                             self.m_base, self.dev_base)
        if plan is None:
            return False
        u, e, idx = plan
        total = idx.shape[1]
        self.launches = self.pieces = 0
        if not total:
            return True
        dev_side = 0 if to_host else 1
        chunk = max(1, min(total, chunk_bytes // u))
        scratch = self._alloc(chunk * u)
        idx_t = self._alloc(idx.nbytes)
        dev_rows = self._rows("dev", u, e)
        m_rows = self._rows("mirror", u, e)
        pieces = self._pieces(u, dev_rows.shape[0],
                              [(t.data_ptr(), t.data_ptr() + t.numel()) for t in (scratch, idx_t)])
        # device row numbers grow with the range addresses (sorted), so each
        # piece owns one contiguous slice of the plan
        rows_dev = idx[dev_side].copy()
        work = []
        for r0, r1 in pieces:
            lo, hi = np.searchsorted(rows_dev, [r0, r1])
            if hi > lo:
                idx[dev_side, lo:hi] -= r0
                work.append((r0, r1, int(lo), int(hi)))
        if sum(hi - lo for _, _, lo, hi in work) != total:
            return False   # a planned row inside a foreign tensor: cannot happen
        idx_t = idx_t.view(_torch().int64).view(2, total)
        self._upload(idx, idx_t)
        scratch = (scratch.view(_torch_dtype(e)) if e > 1 else scratch).view(chunk, u // e)
        torch = _torch()
        launches = 1
        for r0, r1, lo, hi in work:
            dv = dev_rows[r0:r1]
            src, dst = (dv, m_rows) if to_host else (m_rows, dv)
            for a in range(lo, hi, chunk):
                b = min(hi, a + chunk)
                tmp = scratch[:b - a]
                torch.index_select(src, 0, idx_t[0, a:b], out=tmp)
                dst.index_copy_(0, idx_t[1, a:b], tmp)
                launches += 2
        self.launches, self.pieces = launches, len(work)
        return True


def _torch():
    import torch
    return torch


def _torch_dtype(e: int):
    return getattr(_torch(), _DTYPE_BY_SIZE[e])


def _merge(ranges) -> list[tuple[int, int]]:
    if hasattr(ranges, "tolist"):   # block_ranges returns an array
        ranges = ranges.tolist()
    out: list[tuple[int, int]] = []
    for a, n in sorted((a, n) for a, n in ranges):
        if out and a <= out[-1][0] + out[-1][1]:
            pa, pn = out[-1]
            out[-1] = (pa, max(pa + pn, a + n) - pa)
        else:
            out.append((a, n))
    return out


def block_ranges(layouts, blocks: Sequence[int]):
    """Byte ranges of ``blocks`` in every layout, as an int64 array of
    ``(addr, n)`` rows (layout, then segment, then block order). Built with
    numpy: it runs on the engine thread for every finished prefill, and a
    Python loop over thousands of ranges costs about a millisecond."""
    import numpy as np

    b = np.asarray(blocks, dtype=np.int64).reshape(-1)
    layouts = list(layouts)
    shapes = {(lo.outer, lo.outer_stride, lo.block_stride, lo.seg_bytes) for lo in layouts}
    if len(shapes) == 1:   # the usual case: every layer laid out alike
        outer, ostride, bstride, seg = shapes.pop()
        base = np.asarray([lo.base for lo in layouts], dtype=np.int64)
        a = (base[:, None, None] + np.arange(outer, dtype=np.int64)[None, :, None] * ostride
             + b[None, None, :] * bstride).reshape(-1)
        return np.stack([a, np.full_like(a, seg)], axis=1)
    parts = []
    for lo in layouts:
        o = np.arange(lo.outer, dtype=np.int64)[:, None] * lo.outer_stride
        a = (lo.base + o + b[None, :] * lo.block_stride).reshape(-1)
        parts.append(np.stack([a, np.full_like(a, lo.seg_bytes)], axis=1))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(parts)


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
