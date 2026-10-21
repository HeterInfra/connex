# SPDX-License-Identifier: Apache-2.0
"""Registered send-ring bookkeeping with ack-based reuse (pure Python).

The sender stages each outgoing message in a region of a registered ("send
ring") buffer; the receiver RDMA-reads it and acks the region's token
(wire.ACK_MARK). A region is reused only after its ack, or after a safety
timeout if the receiver died. Regions of replies that may be re-sent for a
duplicate request are pinned until they leave the reply cache.
"""

from __future__ import annotations

import os


class Lease:
    __slots__ = ("start", "end", "deadline", "acked", "pinned", "obj", "want_ack")

    def __init__(self, start, end, deadline, want_ack, obj=None):
        self.start, self.end, self.deadline = start, end, deadline
        self.acked = False
        self.pinned = False
        self.obj = obj  # temp buffer (None for ring regions)
        self.want_ack = want_ack


class RegionTable:
    """Bookkeeping for the registered GPU send ring plus one-off temp buffers.

    Pure Python (no torch / mooncake) so it can be unit tested. A region is
    busy while it is pinned (kept for _resend_last) or while it is neither
    acked nor past its deadline. The deadline is the ack safety timeout when
    the receiver was asked to ack, else the legacy time-based hold. alloc()
    never overwrites a busy region: it returns None and the caller falls back
    to a temporary registered buffer.
    """

    def __init__(self, size: int, token_base: int | None = None):
        self.size = size
        self.head = 0
        self.leases: dict[int, Lease] = {}
        self._seq = 0
        self.base = (int.from_bytes(os.urandom(4), "big") << 32
                     if token_base is None else token_base)
        self.expired_unacked = 0

    def _token(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFFFFFF or 1
        return self.base | self._seq

    @staticmethod
    def _live(lease: Lease, now: float) -> bool:
        return lease.pinned or (not lease.acked and lease.deadline > now)

    def alloc(self, n: int, now: float, hold_s: float, want_ack: bool):
        """Return (offset, token) or None if the ring cannot fit n bytes now."""
        if n > self.size:
            return None
        start = self.head if self.head + n <= self.size else 0
        end = start + n
        for lease in self.leases.values():
            if (lease.obj is None and self._live(lease, now)
                    and not (end <= lease.start or start >= lease.end)):
                return None
        tok = self._token()
        self.leases[tok] = Lease(start, end, now + hold_s, want_ack)
        self.head = end
        return start, tok

    def add_temp(self, obj, now: float, hold_s: float, want_ack: bool) -> int:
        tok = self._token()
        self.leases[tok] = Lease(0, 0, now + hold_s, want_ack, obj)
        return tok

    def ack(self, tok: int) -> bool:
        lease = self.leases.get(tok)
        if lease is None:
            return False
        lease.acked = True
        return True

    def pin(self, tok: int) -> None:
        lease = self.leases.get(tok)
        if lease is not None:
            lease.pinned = True

    def unpin(self, tok: int) -> None:
        lease = self.leases.get(tok)
        if lease is not None:
            lease.pinned = False

    def awaiting_ack(self) -> bool:
        return any(l.want_ack and not l.acked for l in self.leases.values())

    def collect(self, now: float) -> list:
        """Drop free leases; return the temp buffers that may be released."""
        freed = []
        for tok, lease in list(self.leases.items()):
            if self._live(lease, now):
                continue
            if lease.want_ack and not lease.acked:
                self.expired_unacked += 1
            del self.leases[tok]
            if lease.obj is not None:
                freed.append(lease.obj)
        return freed
