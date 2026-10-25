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

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Optional, Sequence


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
    # Optional head geometry inside one segment (0 = unknown). Head h, token t
    # of a block segment starts at ``h * head_stride + t * token_stride`` and
    # is ``head_bytes`` long. Needed only when the two sides shard KV heads
    # differently (heterogeneous tensor parallelism).
    heads: int = 0
    head_stride: int = 0
    tokens: int = 0
    token_stride: int = 0
    head_bytes: int = 0

    @property
    def span(self) -> tuple[int, int]:
        """(start, length) of the memory to register."""
        end = self.base + (self.outer - 1) * self.outer_stride + self.num_blocks * self.block_stride
        return self.base, end - self.base

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TensorLayout":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def with_heads(self, heads: int, head_stride: int, tokens: int, token_stride: int,
                   head_bytes: int) -> "TensorLayout":
        if heads * head_bytes * tokens != self.seg_bytes:
            raise ValueError(f"{heads} heads x {tokens} tokens x {head_bytes} B != segment "
                             f"of {self.seg_bytes} B")
        return replace(self, heads=heads, head_stride=head_stride, tokens=tokens,
                       token_stride=token_stride, head_bytes=head_bytes)

    @property
    def head_major(self) -> bool:
        """All tokens of one head are contiguous (HND): a head range of a
        block is one contiguous read."""
        return self.token_stride == self.head_bytes and \
            self.head_stride == self.tokens * self.head_bytes


def tensor_layout(t, num_blocks: int, heads_dims: tuple[int, int] | None = None) -> TensorLayout:
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
            lo = TensorLayout(base=t.data_ptr(), outer=outer, outer_stride=ostride,
                              block_stride=stride[d] * es, seg_bytes=inner * es,
                              num_blocks=num_blocks)
            if heads_dims is not None:
                # (token dim, head dim) of the tensor, in either order
                td, hd = (x % len(shape) for x in heads_dims)
                hb = es   # the dims after both are the dense vector of one head and token
                for x in shape[max(td, hd) + 1:]:
                    hb *= x
                lo = lo.with_heads(shape[hd], stride[hd] * es, shape[td], stride[td] * es, hb)
            return lo
    raise ValueError(f"no contiguous num_blocks={num_blocks} dimension in shape {shape} "
                     f"stride {stride}")


def find_heads_dims(shape: tuple, num_blocks: int, kv_heads: int, block_size: int,
                layout: Optional[str] = None) -> Optional[tuple[int, int]]:
    """(token dim, head dim) of a paged KV tensor, or None if not found.

    vLLM 0.32 hands connectors per-layer tensors in the layout it resolved:
    ``LBNHC`` is ``[num_blocks, block_size, heads, C]`` and ``LBHNC``
    ``[num_blocks, heads, block_size, C]`` (C = K and V of one head and
    token); older releases use ``[2, num_blocks, block_size, heads, D]``.
    Dims after the block dim are matched by size; when the head count equals
    the block size, the layout name (order of H and N) decides."""
    try:
        b = shape.index(num_blocks)
    except ValueError:
        return None
    inner = list(range(b + 1, len(shape) - 1))
    hs = [d for d in inner if shape[d] == kv_heads]
    ts = [d for d in inner if shape[d] == block_size]
    if not kv_heads or not hs or not ts:
        return None
    if kv_heads != block_size:
        return ts[0], hs[0]
    if len(inner) < 2:
        return None
    first, second = inner[0], inner[1]
    name = (layout or "").upper()
    if "H" in name and "N" in name and name.index("H") < name.index("N"):
        return second, first     # head-major
    return first, second


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


# ---------------------------------------------------------------------------
# Heterogeneous tensor parallelism: KV heads sharded differently on each side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TPSource:
    """One remote TP rank a local rank reads from: global KV heads
    ``[head0, head0 + nheads)`` land at local head ``local_head0`` and come
    from remote head ``remote_head0`` (both relative to the rank's shard).
    ``releaser``: this local rank releases the remote rank's hold."""

    remote_rank: int
    nheads: int
    local_head0: int
    remote_head0: int
    releaser: bool
    readers: int


def tp_plan(local_rank: int, local_tp: int, remote_tp: int, total_heads: int) -> list[TPSource]:
    """Which remote ranks local rank ``local_rank`` reads its KV heads from.

    Heads are sharded contiguously (rank r holds heads ``[r*H/tp, (r+1)*H/tp)``,
    vLLM's and SGLang's convention for KV heads when ``H >= tp``). Every remote
    rank that overlaps our shard is one source; ``readers`` is how many local
    ranks read that remote rank, and the lowest of them is its ``releaser``.
    """
    for tp in (local_tp, remote_tp):
        if tp < 1 or total_heads % tp:
            raise ValueError(f"{total_heads} KV heads cannot be sharded over TP {tp} "
                             "(replicated KV heads are not supported)")
    if not 0 <= local_rank < local_tp:
        raise ValueError(f"rank {local_rank} outside TP {local_tp}")
    hl, hr = total_heads // local_tp, total_heads // remote_tp
    lo, hi = local_rank * hl, (local_rank + 1) * hl
    out = []
    for p in range(lo // hr, -(-hi // hr)):
        a, b = max(lo, p * hr), min(hi, (p + 1) * hr)
        first_reader = (p * hr) // hl
        last_reader = ((p + 1) * hr - 1) // hl
        out.append(TPSource(p, b - a, a - lo, a - p * hr, local_rank == first_reader,
                            last_reader - first_reader + 1))
    return out


def head_read_ops(local_layouts: Sequence[TensorLayout], remote_layouts: Sequence[TensorLayout],
                  local_blocks: Sequence[int], remote_blocks: Sequence[int],
                  local_head0: int, remote_head0: int, nheads: int
                  ) -> list[tuple[int, int, int]]:
    """Reads that copy heads ``[remote_head0, +nheads)`` of the peer's blocks
    into heads ``[local_head0, +nheads)`` of ours (KV head re-layout for
    heterogeneous TP). Both sides need head geometry (``with_heads``) with the
    same block size (tokens) and head size; their layouts (HND or NHD) may
    differ. Adjacent reads are merged, so a full-shard copy between equal
    layouts costs as few reads as ``read_ops``."""
    if len(local_layouts) != len(remote_layouts):
        raise ValueError(f"peer has {len(remote_layouts)} KV tensors, we have {len(local_layouts)}")
    runs = block_runs(local_blocks, remote_blocks)
    ops: list[tuple[int, int, int]] = []
    for lo, ro in zip(local_layouts, remote_layouts):
        if not (lo.heads and ro.heads):
            raise ValueError("KV head geometry unknown: cannot re-layout heads")
        if (lo.tokens, lo.head_bytes, lo.outer) != (ro.tokens, ro.head_bytes, ro.outer):
            raise ValueError("KV layout mismatch between instances "
                             f"(block of {lo.tokens} vs {ro.tokens} tokens, head of "
                             f"{lo.head_bytes} vs {ro.head_bytes} B, {lo.outer} vs {ro.outer} "
                             "segments per block)")
        if local_head0 + nheads > lo.heads or remote_head0 + nheads > ro.heads:
            raise ValueError(f"head range outside the shard ({local_head0}+{nheads} of "
                             f"{lo.heads} local, {remote_head0}+{nheads} of {ro.heads} remote)")
        # pieces of one block segment: (local offset, remote offset, length)
        if lo.head_major and ro.head_major:
            pieces = [(local_head0 * lo.head_stride, remote_head0 * ro.head_stride,
                       nheads * lo.head_stride)]
        elif lo.head_stride == lo.head_bytes and ro.head_stride == ro.head_bytes:
            pieces = [(t * lo.token_stride + local_head0 * lo.head_bytes,
                       t * ro.token_stride + remote_head0 * ro.head_bytes, nheads * lo.head_bytes)
                      for t in range(lo.tokens)]
        else:
            pieces = [((local_head0 + h) * lo.head_stride + t * lo.token_stride,
                       (remote_head0 + h) * ro.head_stride + t * ro.token_stride, lo.head_bytes)
                      for h in range(nheads) for t in range(lo.tokens)]
        for o in range(lo.outer):
            lb, rb = lo.base + o * lo.outer_stride, ro.base + o * ro.outer_stride
            for l0, r0, n in runs:
                for i in range(n):
                    ld = lb + (l0 + i) * lo.block_stride
                    rs = rb + (r0 + i) * ro.block_stride
                    for lo_off, ro_off, ln in pieces:
                        _append_merged(ops, ld + lo_off, rs + ro_off, ln)
    return ops


def _append_merged(ops: list, dst: int, src: int, n: int) -> None:
    if ops:
        d, s, m = ops[-1]
        if d + m == dst and s + m == src:
            ops[-1] = (d, s, m + n)
            return
    ops.append((dst, src, n))
