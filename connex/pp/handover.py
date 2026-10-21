# SPDX-License-Identifier: Apache-2.0
"""In-flight handover signalling (engine neutral, pure Python).

When a stage's upstream neighbour fails over to another replica, requests
that are mid-decode arrive at a replica that does not hold their KV. Connex
does not move KV; it detects the gap and has stage 0 recompute:

* stage 0 sends ``cx_ctx`` with every step: for row i, the number of tokens
  of request ``request_ids[i]`` already computed (before this step);
* every later stage compares ``cx_ctx[i]`` with what it holds
  (:meth:`EngineHooks.kv_held`); rows with ``ctx > held`` are *missing*: the
  stage releases them, still runs the batch (shape is fixed) but must not
  cache what it computes for them, and unions their ids into ``cx_missing``;
* the last stage returns ``cx_missing`` to stage 0, which discards the token
  sampled for them and re-prefills prompt + generated tokens on the current
  chain (:meth:`EngineHooks.reprefill`), aborting a request after
  ``max_rewinds`` recomputes.

Finished requests travel downstream as ``finished_req_ids`` (each id on
``repeat`` consecutive steps) so downstream stages free their KV early.

The vLLM fork's patches implement the same rules inside the runner
(integrations/vllm/patches/external_pp_reprefill.py, ConnexPPKVGuard, which
also counts the stage's own prefix-cache blocks for prefill steps). The keys
and semantics here are identical, so a vLLM stage and an engine using this
module interoperate.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Optional, Sequence

CX_CTX = "cx_ctx"
CX_MISSING = "cx_missing"
FINISHED = "finished_req_ids"
REQUEST_IDS = "request_ids"


def find_missing(request_ids: Sequence, ctx: Optional[Sequence[int]],
                 held: Callable[[object], int]) -> list:
    """Rows whose context stage 0 says exists (ctx > 0) but this stage
    does not hold: ``held(rid) < ctx[i]``. No ctx -> nothing is checked."""
    missing = []
    if not request_ids or ctx is None:
        return missing
    for i, rid in enumerate(request_ids):
        c = int(ctx[i]) if i < len(ctx) else 0
        if c > 0 and held(rid) < c:
            missing.append(rid)
    return missing


def merge_missing(upstream: Optional[Sequence], mine: Optional[Sequence]) -> Optional[list]:
    """Union of the ids reported by earlier stages and this one (order kept)."""
    if not upstream:
        return list(mine) if mine else None
    if not mine:
        return list(upstream)
    out = list(upstream)
    seen = set(out)
    out.extend(r for r in mine if r not in seen)
    return out


def blank_rows(rows: Sequence, request_ids: Sequence, missing: Iterable) -> list:
    """Copy of per-row data with the rows of ``missing`` requests emptied."""
    miss = set(missing)
    return [[] if (i < len(request_ids) and request_ids[i] in miss) else t
            for i, t in enumerate(rows)]


class KVLedger:
    """Optional bookkeeping for engines without their own per-request token
    accounting: tokens of each request this stage computed itself.

    ``kv_held(rid)`` = recorded length while ``is_live(rid)`` (default:
    always). ``commit`` after a step records ctx + scheduled tokens for every
    row that was not missing (never shrinking: a re-executed older step must
    not lower it)."""

    def __init__(self, is_live: Optional[Callable[[object], bool]] = None):
        self.kv_len: dict = {}
        self.is_live = is_live

    def kv_held(self, rid) -> int:
        if self.is_live is not None and not self.is_live(rid):
            return 0
        return self.kv_len.get(rid, 0)

    def commit(self, request_ids: Sequence, ctx: Optional[Sequence[int]],
               num_scheduled: Optional[Sequence[int]] = None, missing: Iterable = ()) -> None:
        if not request_ids or ctx is None:
            return
        miss = set(missing or ())
        for i, rid in enumerate(request_ids):
            if rid in miss:
                continue
            c = int(ctx[i]) if i < len(ctx) else 0
            n = int(num_scheduled[i]) if num_scheduled is not None and i < len(num_scheduled) else 1
            if c + n > self.kv_len.get(rid, 0) or c == 0:
                self.kv_len[rid] = c + n

    def forget(self, rids: Iterable) -> None:
        for rid in rids:
            self.kv_len.pop(rid, None)


class FinishedOutbox:
    """Stage 0: finished ids piggybacked on the next ``repeat`` sends."""

    def __init__(self, repeat: int = 2):
        self.repeat = max(1, int(repeat))
        self._q: deque = deque()  # [rid, remaining]

    def add(self, rids: Iterable) -> None:
        for r in rids:
            self._q.append([r, self.repeat])

    def take(self) -> Optional[list]:
        if not self._q:
            return None
        out = [e[0] for e in self._q]
        for e in self._q:
            e[1] -= 1
        self._q = deque(e for e in self._q if e[1] > 0)
        return out


class RewindBudget:
    """Stage 0: how often each request was re-prefilled; past ``max_rewinds``
    it must be aborted (a clean error rather than a silent wrong decode)."""

    def __init__(self, max_rewinds: int = 8):
        self.max_rewinds = max_rewinds
        self.count: dict = {}

    def split(self, missing: Iterable) -> tuple[list, list]:
        """-> (to_reprefill, to_abort)."""
        again, abort = [], []
        for rid in missing:
            n = self.count.get(rid, 0) + 1
            self.count[rid] = n
            (abort if n > self.max_rewinds else again).append(rid)
        return again, abort

    def forget(self, rids: Iterable) -> None:
        for r in rids:
            self.count.pop(r, None)
