# SPDX-License-Identifier: Apache-2.0
"""Expert weight moves between EP ranks with one-sided reads.

Each EP rank wraps its expert weights in an :class:`ExpertStore`: per MoE
layer, a list of parameter tensors whose first dimension is the local slot
(``w13[slot]``, ``w2[slot]``, scales...). The store registers them with a
transport (``connex.rl.transport``: connex-rs, or the in-process
``LocalTransport`` in tests) and :meth:`ExpertStore.describe` gives peers
the addresses. A move into a slot of this rank then pulls the expert's rows
from the source rank's registered memory (GPUDirect over connex-rs where
available), copies them locally when the source is on the same rank, or
calls ``loader(layer, expert)`` when no live copy exists (the checkpoint).

Rows are read straight into the destination slot. The caller makes sure the
slot is not routable while this runs (``PlacementTable.begin_moves`` then a
committed epoch), so serving can continue on the other slots.
"""
from __future__ import annotations

import base64
import os
import pickle
import socket
import time
from collections import defaultdict
from typing import Callable, Optional

import torch

from .placement import Move


def _bytes_view(t: torch.Tensor) -> torch.Tensor:
    """Flat uint8 view of a contiguous tensor (shares storage)."""
    if not t.is_contiguous():
        raise ValueError("expert parameter must be contiguous")
    return t.view(-1).view(torch.uint8) if t.dtype != torch.uint8 else t.view(-1)


def host_id():
    """Identifies this host (and container) for the same-host fast path."""
    try:
        boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        boot = ""
    return f"{socket.gethostname()}/{boot}"


class ExpertStore:
    """Expert weights of one EP rank, addressable by peers.

    ``ipc=True`` (CUDA tensors only) also exports CUDA IPC handles: a peer on
    the same host opens them once and copies rows GPU to GPU (peer-to-peer
    over PCIe/NVLink), which beats an RDMA loopback through the NIC. Peers on
    other hosts, or when opening fails, use the transport."""

    def __init__(self, rank: int, layers: list, transport=None, ipc=None):
        self.rank = rank
        self.layers = [list(ps) for ps in layers]       # [layer][param] -> tensor[slots, ...]
        self.t = transport
        self._flat = [[_bytes_view(p) for p in ps] for ps in self.layers]
        if self.t is not None:
            for ps in self.layers:
                for p in ps:
                    self.t.register(p.view(-1).view(torch.uint8) if p.dtype != torch.uint8 else p)
        self.stats = defaultdict(float)
        if ipc is None:
            ipc = os.environ.get("CONNEX_EP_IPC", "1") != "0"
        self.ipc = bool(ipc) and bool(self.layers) and self.layers[0][0].is_cuda
        self._peer_tensors = {}      # (rank, host) -> [[tensor]] opened via IPC

    @property
    def slots(self):
        return self.layers[0][0].shape[0] if self.layers else 0

    def row_bytes(self, layer, param):
        p = self.layers[layer][param]
        return p[0].numel() * p.element_size()

    def describe(self):
        """Address book for peers: transport url and, per layer and param,
        (base address, bytes per slot)."""
        d = {
            "rank": self.rank,
            "url": getattr(self.t, "url", None),
            "layers": [[[p.data_ptr(), self.row_bytes(l, i)] for i, p in enumerate(ps)]
                       for l, ps in enumerate(self.layers)],
        }
        if self.ipc:
            from torch.multiprocessing.reductions import reduce_tensor
            d["host"] = host_id()
            d["ipc"] = base64.b64encode(pickle.dumps(
                [[reduce_tensor(p) for p in ps] for ps in self.layers])).decode()
        return d

    def _open_ipc(self, peer):
        """The peer's expert tensors mapped into this process, or None."""
        if not (self.ipc and peer.get("ipc") and peer.get("host") == host_id()):
            return None
        key = (peer["rank"], peer.get("url"))
        if key not in self._peer_tensors:
            try:
                rows = pickle.loads(base64.b64decode(peer["ipc"]))
                self._peer_tensors[key] = [[fn(*args) for fn, args in ps] for ps in rows]
            except Exception:  # noqa: BLE001  (no P2P / IPC: use the transport)
                self._peer_tensors[key] = None
        return self._peer_tensors[key]

    def apply_moves(self, moves, peers: dict, slots_per_rank: int,
                    loader: Optional[Callable] = None, timeout_ms=None):
        """Execute the moves whose destination is on this rank.

        ``peers``: rank -> :meth:`describe` of that rank (this rank's own
        entry is not needed). ``loader(layer, expert)`` returns the expert's
        parameter tensors (one per param, shaped like a slot) for moves
        without a live source. Returns per-move outcomes
        ``[(move, ok, error)]``; a failed read leaves the slot marked failed
        (the caller re-plans with another source)."""
        t0 = time.perf_counter()
        mine = [m for m in moves if m.dst // slots_per_rank == self.rank]
        out = []
        # group remote reads per (source rank, layer, param) so one batch
        # carries every expert of that tensor
        remote = defaultdict(list)
        for m in mine:
            di = m.dst % slots_per_rank
            if m.src is None:
                if loader is None:
                    out.append((m, False, "no live source and no loader"))
                    continue
                for i, w in enumerate(loader(m.layer, m.expert)):
                    self.layers[m.layer][i][di].copy_(w.to(self.layers[m.layer][i].device,
                                                           self.layers[m.layer][i].dtype))
                self.stats["checkpoint_loads"] += 1
                out.append((m, True, None))
                continue
            sr, si = divmod(m.src, slots_per_rank)
            if sr == self.rank:
                for p in self.layers[m.layer]:
                    p[di].copy_(p[si])
                self.stats["local_copies"] += 1
                out.append((m, True, None))
                continue
            remote[sr].append(m)
        for sr, ms in remote.items():
            peer = peers.get(sr) if sr in peers else peers.get(str(sr))
            err = None
            if peer is None:
                err = f"no address for rank {sr}"
            else:
                try:
                    self._read_from(peer, ms, slots_per_rank, timeout_ms)
                except Exception as ex:   # TransferError or connex_rs error
                    err = repr(ex)
            out.extend((m, err is None, err) for m in ms)
        if any(m.dst // slots_per_rank == self.rank for m in moves):
            dev = self.layers[0][0].device if self.layers else None
            if dev is not None and dev.type == "cuda":
                torch.cuda.synchronize(dev)
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["moves"] += len(mine)
        return out

    def _read_from(self, peer, moves, slots_per_rank, timeout_ms):
        remote = self._open_ipc(peer)
        if remote is not None:
            devs = set()
            for m in moves:
                si, di = m.src % slots_per_rank, m.dst % slots_per_rank
                for i, p in enumerate(self.layers[m.layer]):
                    src = remote[m.layer][i][si]
                    p[di].copy_(src, non_blocking=True)
                    devs.add(src.device)
            # a peer-to-peer copy is ordered on the source device's stream:
            # wait for it there too, or "done" could precede the bytes
            for d in devs:
                torch.cuda.synchronize(d)
            self.stats["ipc_bytes"] += sum(rb for m in moves for _, rb in peer["layers"][m.layer])
            return
        url = peer["url"]
        by_tensor = defaultdict(list)            # (layer, param) -> ops
        for m in moves:
            si, di = m.src % slots_per_rank, m.dst % slots_per_rank
            for i, (base, rb) in enumerate(peer["layers"][m.layer]):
                by_tensor[(m.layer, i)].append((base + si * rb, di * rb, rb))
        read_many = getattr(self.t, "read_many", None)
        if read_many is not None:
            read_many(url, [(self._flat[l][i], ops) for (l, i), ops in by_tensor.items()],
                      timeout_ms=timeout_ms)
        else:
            for (l, i), ops in by_tensor.items():
                self.t.read_batch(url, ops, self._flat[l][i], timeout_ms=timeout_ms)
        self.stats["remote_bytes"] += sum(o[2] for ops in by_tensor.values() for o in ops)

    def close(self):
        if self.t is not None:
            for ps in self.layers:
                for p in ps:
                    try:
                        self.t.unregister(p.view(-1).view(torch.uint8) if p.dtype != torch.uint8 else p)
                    except Exception:
                        pass


def moves_by_rank(moves, slots_per_rank):
    out = defaultdict(list)
    for m in moves:
        out[m.dst // slots_per_rank].append(m)
    return dict(out)


__all__ = ["ExpertStore", "Move", "moves_by_rank"]
