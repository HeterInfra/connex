# SPDX-License-Identifier: Apache-2.0
"""Step protocol state that does not touch the transport (pure Python).

Correlation ids: stage 0 stamps every pipeline step with
``corr = incarnation << 32 | seq`` and reuses it when it re-sends the step;
later stages echo it. This gives

* reply matching on stage 0 (K steps outstanding, any reply order),
* idempotent retries (a stage answers a duplicate from its reply cache
  instead of recomputing; :class:`ReplyCache`),
* stale suppression (:class:`CorrFilter`): an id K or more behind the
  newest id seen from the same stage-0 incarnation is dropped.
"""

from __future__ import annotations

import os
from collections import OrderedDict


class CorrFilter:
    """Highest correlation id seen per stage-0 incarnation (high 32 bits =
    random _corr_base of that stage 0). An id below the newest one of the same
    incarnation is a stale message (abandoned step / late retry).

    window = number of steps the sender may keep in flight (CONNEX_PP_INFLIGHT).
    With window K an id is stale only if it is K or more behind the newest id;
    ids inside the window may legitimately arrive out of order (a per-step
    retry routed around a failed replica). window=1 is the original rule."""

    def __init__(self, cap: int = 64, window: int = 1):
        self.cap = cap
        self.window = max(1, int(window))
        self.hi: "OrderedDict[int, int]" = OrderedDict()

    def accept(self, corr: int | None) -> bool:
        if not corr:
            return True  # old sender / no id: cannot judge
        h, lo = corr >> 32, corr & 0xFFFFFFFF
        m = self.hi.get(h)
        if m is not None and lo + self.window <= m:
            return False
        self.hi[h] = lo if m is None else max(m, lo)
        self.hi.move_to_end(h)
        while len(self.hi) > self.cap:
            self.hi.popitem(last=False)
        return True


class StepIds:
    """Stage-0 correlation id allocator: ``base + seq`` with a random 32-bit
    incarnation in the high bits of ``base``."""

    def __init__(self, base: int | None = None):
        self.base = (int.from_bytes(os.urandom(4), "big") << 32) if base is None else base
        self.seq = 0

    def next(self) -> int:
        self.seq += 1
        return self.base + self.seq


class ReplyCache:
    """The last ``k`` replies of a non-first stage keyed by correlation id,
    so a duplicate (sender retry) is answered from cache instead of being
    recomputed. ``on_evict(entry)`` runs for every evicted or replaced entry
    (e.g. to unpin its ring region)."""

    def __init__(self, k: int = 1, on_evict=None):
        self.k = max(1, int(k))
        self.on_evict = on_evict
        self.entries: "OrderedDict[int, object]" = OrderedDict()

    def put(self, corr: int, entry) -> None:
        old = self.entries.pop(corr, None)
        if old is not None and self.on_evict:
            self.on_evict(old)
        self.entries[corr] = entry
        while len(self.entries) > self.k:
            _, e = self.entries.popitem(last=False)
            if self.on_evict:
                self.on_evict(e)

    def get(self, corr):
        return self.entries.get(corr)

    def __contains__(self, corr) -> bool:
        return corr in self.entries

    def __len__(self) -> int:
        return len(self.entries)
