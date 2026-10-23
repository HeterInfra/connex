# SPDX-License-Identifier: Apache-2.0
"""Flat weight arena: all tensors of one model version packed into one
contiguous byte buffer, described by a manifest.

One buffer means one memory registration and large one-sided reads; the
manifest lets the receiver rebuild ``(name, tensor)`` views without copies.
"""
from dataclasses import dataclass, field

import torch

ALIGN = 256


def _dtype(name):
    return getattr(torch, name.replace("torch.", ""))


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: list
    offset: int
    nbytes: int


@dataclass
class WeightManifest:
    version: int
    entries: list = field(default_factory=list)
    total: int = 0

    def to_dict(self):
        return {"version": self.version, "total": self.total,
                "entries": [[e.name, e.dtype, list(e.shape), e.offset, e.nbytes]
                            for e in self.entries]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["version"], [TensorEntry(*e) for e in d["entries"]], d["total"])

    @classmethod
    def build(cls, version, named):
        """Layout for ``named`` (iterable of (name, tensor)); no data copied."""
        m, off = cls(version), 0
        for name, t in named:
            n = t.numel() * t.element_size()
            m.entries.append(TensorEntry(name, str(t.dtype).replace("torch.", ""),
                                         list(t.shape), off, n))
            off += (n + ALIGN - 1) // ALIGN * ALIGN
        m.total = max(off, ALIGN)
        return m

    def same_layout(self, other):
        return other is not None and self.total == other.total and [
            (e.name, e.dtype, e.shape, e.offset) for e in self.entries] == [
            (e.name, e.dtype, e.shape, e.offset) for e in other.entries]


def arena_size(named):
    return WeightManifest.build(0, named).total


def pack_into(arena, manifest, named):
    """Copy tensors into the uint8 ``arena`` at the manifest's offsets."""
    by = {e.name: e for e in manifest.entries}
    for name, t in named:
        e = by[name]
        arena[e.offset:e.offset + e.nbytes].copy_(
            t.detach().contiguous().view(-1).view(torch.uint8), non_blocking=True)


def unpack(arena, manifest):
    """Yield ``(name, tensor)`` views into ``arena`` (no copy)."""
    for e in manifest.entries:
        yield e.name, arena[e.offset:e.offset + e.nbytes].view(_dtype(e.dtype)).view(e.shape)
