# SPDX-License-Identifier: Apache-2.0
"""Test doubles for engine adapters (no GPU, no RDMA, no cluster).

* :class:`LoopbackTransferEngine` -- in-process stand-in for Mooncake's
  TransferEngine: "memory" is a dict of bytearrays, segments are looked up by
  URL in a shared registry, ``transfer_sync_read`` copies bytes. Plug it into
  :class:`connex.pp.endpoint.ConnexTEEndpoint` (``transfer_engine=``) to run
  the real signal/correlation/retry code over real ZeroMQ sockets.
* :class:`FakeLink` / :func:`fake_chain` -- an in-process
  :class:`connex.pp.hooks.StepTransport` chain with the endpoint's dedupe and
  stale rules and knobs to drop, duplicate, reorder or fail replicas.
"""

from __future__ import annotations

import itertools
import queue
import time
import threading
from collections import OrderedDict
from typing import Optional

from connex.pp.steps import CorrFilter, StepIds


class LoopbackTransferEngine:
    registry: dict = {}
    _ptr = itertools.count(1 << 20, 1 << 20)
    _lock = threading.Lock()

    def __init__(self, url: str):
        self.url = url
        self.mem: dict[int, bytearray] = {}
        LoopbackTransferEngine.registry[url] = self
        self.reads = 0

    def _alloc(self, size: int) -> int:
        with self._lock:
            p = next(self._ptr)
        self.mem[p] = bytearray(size)
        return p

    def _find(self, ptr: int):
        for base, buf in self.mem.items():
            if base <= ptr < base + max(1, len(buf)):
                return base, buf
        raise KeyError(hex(ptr))

    def allocate_managed_buffer(self, size: int) -> int:
        return self._alloc(size)

    def free_managed_buffer(self, ptr: int, size: int) -> None:
        self.mem.pop(ptr, None)

    def write_bytes_to_buffer(self, ptr: int, data: bytes, length: int) -> None:
        base, buf = self._find(ptr)
        buf[ptr - base:ptr - base + length] = data[:length]

    def read_bytes_from_buffer(self, ptr: int, length: int) -> bytes:
        base, buf = self._find(ptr)
        return bytes(buf[ptr - base:ptr - base + length])

    def transfer_sync_read(self, remote_url: str, dst_ptr: int, src_ptr: int, length: int) -> int:
        peer = LoopbackTransferEngine.registry.get(remote_url)
        if peer is None:
            return -1
        try:
            data = peer.read_bytes_from_buffer(src_ptr, length)
        except KeyError:
            return -1
        self.write_bytes_to_buffer(dst_ptr, data, length)
        self.reads += 1
        return 0

    def register_memory(self, ptr: int, length: int) -> int:
        return 0

    def unregister_memory(self, ptr: int) -> int:
        return 0


# ---- in-process step transport -------------------------------------------------

class _Wire:
    """Directed message queue with fault knobs."""

    def __init__(self):
        self.q: "queue.Queue" = queue.Queue()
        self.drop_next = 0
        self.dup_next = 0
        self.dead = False

    def put(self, item):
        if self.dead:
            return
        if self.drop_next:
            self.drop_next -= 1
            return
        self.q.put(item)
        if self.dup_next:
            self.dup_next -= 1
            self.q.put(item)


class FakeLink:
    """One stage of an in-process pipeline implementing StepTransport with
    the same matching / dedupe / stale rules as the real endpoint:

    stage 0: ids from StepIds, replies matched by id, others buffered,
    abandoned ones dropped. Stages 1..: replies of the last ``k`` steps are
    cached and re-sent for a duplicate (no recompute); ids ``k`` or more
    behind the newest are dropped as stale."""

    def __init__(self, stage_idx: int, num_stages: int, k: int = 1):
        self.stage_idx, self.num_stages = stage_idx, num_stages
        self.is_first, self.is_last = stage_idx == 0, stage_idx == num_stages - 1
        self._k = k
        self.inbox = _Wire()
        self.downstream: Optional[FakeLink] = None   # where forward steps go
        self.head: Optional[FakeLink] = None          # stage 0 (for replies)
        self.healthy = True
        self.ids = StepIds(0xC0DE << 32)
        self.outstanding: set = set()
        self.done: dict = {}
        self.filter = CorrFilter(window=k)
        self.replies: "OrderedDict[int, dict]" = OrderedDict()
        self.computed = 0
        self.sent = 0
        # fast handover: epoch of the chain each step was sent on
        self.chain_epoch = 0
        self.sent_epoch: dict = {}

    @property
    def inflight(self) -> int:
        return self._k

    def _out(self) -> "FakeLink":
        return self.head if self.is_last else self.downstream

    # stage 0
    def new_step_id(self) -> int:
        c = self.ids.next()
        self.outstanding.add(c)
        return c

    def fail_over(self, new_downstream: "FakeLink") -> None:
        """Stage 0's peer was confirmed dead: route to ``new_downstream``;
        every step sent on the old chain is now stale (lost)."""
        self.downstream = new_downstream
        self.chain_epoch += 1

    def step_stale(self, corr: int) -> bool:
        e = self.sent_epoch.get(corr)
        return e is not None and e != self.chain_epoch

    def send_step(self, td: dict, corr: int, dst=None) -> None:
        self.sent += 1
        self.sent_epoch[corr] = self.chain_epoch
        self.downstream.inbox.put((corr, dict(td)))

    def recv_step(self, corr: int, timeout_s: Optional[float] = None):
        if corr in self.done:
            self.outstanding.discard(corr)
            return self.done.pop(corr)
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else 0.2)
        while True:
            if self.step_stale(corr):
                return None              # lost to a failover: re-send now
            try:
                got, td = self.inbox.q.get(timeout=min(0.02, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    return None
                continue
            if got == corr:
                self.outstanding.discard(corr)
                return td
            if got in self.outstanding:
                self.done[got] = td

    def abandon_step(self, corr: int) -> None:
        self.outstanding.discard(corr)
        self.done.pop(corr, None)

    def downstream_healthy(self) -> bool:
        return self.downstream is not None and self.downstream.healthy and not self.downstream.inbox.dead

    # stages 1..
    def recv_forward(self, timeout_s: float = 0.2):
        while True:
            try:
                corr, td = self.inbox.q.get(timeout=timeout_s)
            except queue.Empty:
                return None, None
            if corr in self.replies:
                self._out().inbox.put((corr, dict(self.replies[corr])))
                continue
            if not self.filter.accept(corr):
                continue
            self.computed += 1
            return td, (corr, None)

    def send_reply(self, td: dict, ctx) -> None:
        corr = ctx[0]
        self.replies[corr] = dict(td)
        while len(self.replies) > self._k:
            self.replies.popitem(last=False)
        self._out().inbox.put((corr, dict(td)))


def fake_chain(num_stages: int, k: int = 1) -> list:
    """Stages 0..n-1 wired forward, last -> stage 0."""
    links = [FakeLink(i, num_stages, k) for i in range(num_stages)]
    for i, l in enumerate(links):
        l.head = links[0]
        if i + 1 < num_stages:
            l.downstream = links[i + 1]
    return links
