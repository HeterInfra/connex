# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Where the bytes of one KV block live, for any paged KV cache.

An engine hands Connex its KV cache as a list of tensors (one per layer, or
one per layer and K/V). Each tensor has a ``num_blocks`` dimension; the
bytes of block ``b`` are ``outer`` equally spaced contiguous segments (e.g.
K and V of a ``[2, num_blocks, ...]`` FlashAttention cache). This module
turns such tensors into plain numbers (base address, strides, segment
length) that travel between instances, and turns a list of (local block,
remote block) pairs into the fewest one-sided reads: runs of consecutive
blocks on both sides become one read per segment.

Engine neutral and torch-free at import: tensors are only inspected through
``data_ptr()``, ``shape``, ``stride()`` and ``element_size()``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TensorLayout:
    """One KV tensor: block b, segment o starts at
    ``base + o * outer_stride + b * block_stride`` and is ``seg_bytes`` long."""

    base: int
    outer: int
    outer_stride: int
    block_stride: int
    seg_bytes: int
    num_blocks: int

    @property
    def span(self) -> tuple[int, int]:
        """(start, length) of the memory to register."""
        end = self.base + (self.outer - 1) * self.outer_stride + self.num_blocks * self.block_stride
        return self.base, end - self.base

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TensorLayout":
        return cls(**d)


def tensor_layout(t, num_blocks: int) -> TensorLayout:
    """Layout of a tensor whose dimension of size ``num_blocks`` indexes blocks
    and whose per-block slice is contiguous. The first such dimension wins
    (vLLM: ``[2, num_blocks, ...]`` or ``[num_blocks, 2, ...]``)."""
    shape, stride, es = tuple(t.shape), tuple(t.stride()), t.element_size()
    for d, n in enumerate(shape):
        if n != num_blocks:
            continue
        inner = 1
        for x in shape[d + 1:]:
            inner *= x
        # the per-block slice must be one contiguous run of `inner` elements
        if inner and _contiguous(shape[d + 1:], stride[d + 1:]) and stride[d] >= inner:
            outer = 1
            for x in shape[:d]:
                outer *= x
            if outer > 1 and d != 1:
                continue  # only a single leading dimension (K/V) is supported
            ostride = stride[0] * es if outer > 1 else 0
            return TensorLayout(base=t.data_ptr(), outer=outer, outer_stride=ostride,
                                block_stride=stride[d] * es, seg_bytes=inner * es,
                                num_blocks=num_blocks)
    raise ValueError(f"no contiguous num_blocks={num_blocks} dimension in shape {shape} "
                     f"stride {stride}")


def _contiguous(shape: Sequence[int], stride: Sequence[int]) -> bool:
    """The dims cover one dense run of memory in some order (a permuted view
    such as vLLM's HND layout is fine: bytes are copied as they lie)."""
    expect = 1
    for s, n in sorted((s, n) for n, s in zip(shape, stride) if n != 1):
        if s != expect:
            return False
        expect *= n
    return True


def block_runs(local: Sequence[int], remote: Sequence[int]) -> list[tuple[int, int, int]]:
    """Coalesce aligned block lists into (local_start, remote_start, count)."""
    if len(local) != len(remote):
        raise ValueError(f"block lists differ in length: {len(local)} vs {len(remote)}")
    runs: list[tuple[int, int, int]] = []
    for lb, rb in zip(local, remote):
        if runs:
            l0, r0, n = runs[-1]
            if lb == l0 + n and rb == r0 + n:
                runs[-1] = (l0, r0, n + 1)
                continue
        runs.append((lb, rb, 1))
    return runs


def read_ops(local_layouts: Sequence[TensorLayout], remote_layouts: Sequence[TensorLayout],
             local_blocks: Sequence[int], remote_blocks: Sequence[int]
             ) -> list[tuple[int, int, int]]:
    """(dst, src, length) reads that copy ``remote_blocks`` of the peer into
    ``local_blocks`` here, tensor by tensor. Both sides must hold the same
    model with the same block size (checked by segment size)."""
    if len(local_layouts) != len(remote_layouts):
        raise ValueError(f"peer has {len(remote_layouts)} KV tensors, we have {len(local_layouts)}")
    ops: list[tuple[int, int, int]] = []
    runs = block_runs(local_blocks, remote_blocks)
    for lo, ro in zip(local_layouts, remote_layouts):
        if (lo.seg_bytes, lo.outer) != (ro.seg_bytes, ro.outer):
            raise ValueError("KV layout mismatch between instances "
                             f"(segment {lo.seg_bytes} vs {ro.seg_bytes} bytes, "
                             f"{lo.outer} vs {ro.outer} segments per block)")
        contiguous_runs = lo.block_stride == lo.seg_bytes and ro.block_stride == ro.seg_bytes
        for o in range(lo.outer):
            lb, rb = lo.base + o * lo.outer_stride, ro.base + o * ro.outer_stride
            for l0, r0, n in runs:
                if contiguous_runs:
                    ops.append((lb + l0 * lo.block_stride, rb + r0 * ro.block_stride, n * lo.seg_bytes))
                else:
                    for i in range(n):
                        ops.append((lb + (l0 + i) * lo.block_stride,
                                    rb + (r0 + i) * ro.block_stride, lo.seg_bytes))
    return ops


def register_spans(layouts: Iterable[TensorLayout]) -> list[tuple[int, int]]:
    """Distinct (start, length) memory spans to register, merged only when
    they overlap (adjacent tensors stay separate registrations, so a failed
    or revoked registration never spans more than one tensor)."""
    spans = sorted(lo.span for lo in layouts)
    out: list[tuple[int, int]] = []
    for s, n in spans:
        if out and s < out[-1][0] + out[-1][1]:
            ps, pn = out[-1]
            out[-1] = (ps, max(ps + pn, s + n) - ps)
        else:
            out.append((s, n))
    return out
