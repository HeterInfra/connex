# SPDX-License-Identifier: Apache-2.0
"""Block allocator for the external-PP downstream stages (Stage-1+).

This file is BOTH an importable module (unit-tested in tests/test_vllm_patches.py)
and the source that patch_external_pp_allocator.py inlines into vLLM's
gpu_model_runner.py (everything between the BEGIN/END markers is copied verbatim),
so it must stay self-contained: stdlib + numpy only.

Model
-----
Block ids 1..num_blocks-1 are managed; 0 is vLLM's null block and is only ever
used as *padding* for block-table slots a request has not reached yet (the
attention kernels never read past seq_len, and slot_mapping only targets slots
the request owns, so a padding 0 is never written to by a correct step).

* Full prompt blocks are content-addressed (prefix cache): hash -> block, with a
  refcount of active requests using it. refcount 0 -> block moves to an LRU of
  evictable cached blocks (still a prefix-cache hit until evicted).
* Every other slot a request touches (partial last prompt block, decode growth)
  gets a private block, freed when the request is released.
* A block is handed out only from the free list or by evicting an unreferenced
  cached block, so a live request's KV is never overwritten (the fork's ring
  allocator wrapped around and could hand out live blocks).
* Downstream stages never see scheduler finished_req_ids, so requests are also
  released when not seen for `ttl_s` seconds, or (LRU-seen first) when the pool
  is exhausted. Requests in the current batch are never reclaimed.
"""
# BEGIN connex external-pp allocator
import hashlib as _cx_hashlib
import time as _cx_time
from array import array as _cx_array
from collections import OrderedDict as _CxOrderedDict

import numpy as _cx_np

# Toggled to True by patch_external_pp_fast_hash.py. Hashes are only compared
# within one stage process, so changing the function is safe.
_CONNEX_PP_FAST_HASH = False


def connex_pp_hash_block(prev: bytes, tokens) -> bytes:
    if _CONNEX_PP_FAST_HASH:
        return _cx_hashlib.blake2b(
            prev + _cx_array("q", tokens).tobytes(), digest_size=16
        ).digest()
    # Identical to the fork's _compute_block_hashes_for_external_pp.
    return _cx_hashlib.sha256(prev + str(tuple(tokens)).encode("utf-8")).digest()


class _CxReq:
    __slots__ = ("blocks", "hashes", "hashed_slots", "tok_hashes",
                 "next_pos", "computed_upto", "last_seen", "row")

    def __init__(self) -> None:
        self.blocks: list[int] = []          # slot -> block id
        self.hashes: list = []               # slot -> hash (None for private)
        self.hashed_slots = 0
        self.tok_hashes: list[bytes] = []    # incremental hash cache of token blocks
        self.next_pos = 0                    # position of the next token (host-tracked)
        self.computed_upto = 0
        self.last_seen = 0.0
        self.row = None                      # numpy int32 block-table row


class ConnexPPBlockAllocator:
    def __init__(self, num_blocks: int, ttl_s: float = 120.0, clock=None) -> None:
        self.num_blocks = max(int(num_blocks), 2)
        self.ttl_s = ttl_s
        self.clock = clock or _cx_time.monotonic
        self._fresh = 1                       # ids >= _fresh never handed out yet
        self._free: list[int] = []            # released private blocks
        self.hash_to_block: dict = {}
        self.block_to_hash: dict = {}
        self.ref: dict = {}                   # hashed block -> refcount
        self.evictable = _CxOrderedDict()     # hashed blocks with ref 0, LRU first
        self.computed: set = set()            # hashes whose KV is stored
        self.reqs: dict = {}
        self.stats = {"hits": 0, "misses": 0, "queries": 0, "evictions": 0,
                      "reclaimed_reqs": 0, "exhausted": 0}
        self._step = 0
        self._batch: set = set()

    # ---- raw blocks -------------------------------------------------------
    def num_free(self) -> int:
        return (self.num_blocks - self._fresh) + len(self._free) + len(self.evictable)

    def _take(self) -> int:
        if self._free:
            return self._free.pop()
        if self._fresh < self.num_blocks:
            b = self._fresh
            self._fresh += 1
            return b
        if self.evictable:
            b, _ = self.evictable.popitem(last=False)
            h = self.block_to_hash.pop(b)
            del self.hash_to_block[h]
            self.ref.pop(b, None)
            self.computed.discard(h)
            self.stats["evictions"] += 1
            return b
        return 0

    def _alloc(self) -> int:
        b = self._take()
        while b == 0:
            victim = self._stalest_idle()
            if victim is None:
                self.stats["exhausted"] += 1
                return 0  # degraded: null block, only this request's output suffers
            self.release(victim)
            self.stats["reclaimed_reqs"] += 1
            b = self._take()
        return b

    def _stalest_idle(self):
        best = None
        for rid, r in self.reqs.items():
            if rid in self._batch:
                continue
            if best is None or r.last_seen < self.reqs[best].last_seen:
                best = rid
        return best

    def _put_back(self, b: int, h) -> None:
        if b == 0:
            return
        if h is None:
            self._free.append(b)
            return
        n = self.ref.get(b, 0) - 1
        if n > 0:
            self.ref[b] = n
        else:
            self.ref[b] = 0
            self.evictable[b] = None

    # ---- requests ---------------------------------------------------------
    def release(self, req_id) -> None:
        r = self.reqs.pop(req_id, None)
        if r is None:
            return
        for b, h in zip(r.blocks, r.hashes):
            self._put_back(b, h)

    def begin_step(self, request_ids) -> None:
        now = self.clock()
        self._batch = set(request_ids)
        self._step += 1
        if self._step % 64 == 0 and self.ttl_s > 0:
            stale = [rid for rid, r in self.reqs.items()
                     if rid not in self._batch and now - r.last_seen > self.ttl_s]
            for rid in stale:
                self.release(rid)
                self.stats["reclaimed_reqs"] += 1
        for rid in request_ids:
            r = self.reqs.get(rid)
            if r is not None:
                r.last_seen = now

    def _get_req(self, rid, width: int):
        r = self.reqs.get(rid)
        if r is None:
            r = _CxReq()
            r.last_seen = self.clock()
            self.reqs[rid] = r
        if r.row is None or r.row.shape[0] != width:
            row = _cx_np.zeros(width, dtype=_cx_np.int32)
            n = min(width, len(r.blocks))
            row[:n] = r.blocks[:n]
            r.row = row
        return r

    def _hashed_block(self, h):
        self.stats["queries"] += 1
        b = self.hash_to_block.get(h)
        if b is not None:
            self.stats["hits"] += 1
            if self.ref.get(b, 0) == 0:
                self.evictable.pop(b, None)
            self.ref[b] = self.ref.get(b, 0) + 1
            return b, True
        self.stats["misses"] += 1
        b = self._alloc()
        if b == 0:
            return 0, False
        self.hash_to_block[h] = b
        self.block_to_hash[b] = h
        self.ref[b] = 1
        return b, False

    def _append_slot(self, r, b: int, h) -> None:
        idx = len(r.blocks)
        r.blocks.append(b)
        r.hashes.append(h if b != 0 else None)
        if r.row is not None and idx < r.row.shape[0]:
            r.row[idx] = b

    def _grow_private(self, r, num_slots: int) -> None:
        num_slots = min(num_slots, r.row.shape[0])
        while len(r.blocks) < num_slots:
            self._append_slot(r, self._alloc(), None)

    def token_hashes(self, r, token_ids, block_size: int) -> list:
        n_full = len(token_ids) // block_size
        th = r.tok_hashes
        if len(th) > n_full:
            del th[n_full:]
        prev = th[-1] if th else b"NONE_HASH"
        for i in range(len(th), n_full):
            prev = connex_pp_hash_block(prev, token_ids[i * block_size:(i + 1) * block_size])
            th.append(prev)
        return th

    def prefill(self, request_ids, all_token_ids, stage0_computed, num_scheduled,
                block_size: int, width: int):
        """Assign blocks for a prefill-type step. Returns
        (rows[num_reqs, width] int32, hits, queries, computed_tokens_per_req)."""
        self.begin_step(request_ids)
        hits = queries = 0
        computed_per_req = []
        rows = _cx_np.empty((len(request_ids), width), dtype=_cx_np.int32)
        for i, rid in enumerate(request_ids):
            toks = all_token_ids[i] if i < len(all_token_ids) else []
            r = self._get_req(rid, width)
            hashes = self.token_hashes(r, toks, block_size)
            n_comp = 0
            for h in hashes:
                if h in self.computed and self.hash_to_block.get(h) is not None:
                    n_comp += 1
                else:
                    break
            computed_per_req.append(min(n_comp * block_size, len(toks)))
            # Existing slots are kept (their KV is already written); only new
            # slots are content-addressed, and only while no private slot precedes them.
            for s in range(len(r.blocks), min(len(hashes), width)):
                if r.hashed_slots != s:
                    break
                b, hit = self._hashed_block(hashes[s])
                queries += 1
                hits += int(hit)
                self._append_slot(r, b, hashes[s])
                r.hashed_slots = s + 1
            s0 = stage0_computed[i] if stage0_computed is not None and i < len(stage0_computed) else 0
            ns = num_scheduled[i] if num_scheduled is not None and i < len(num_scheduled) else len(toks)
            r.computed_upto = max(r.computed_upto, min(len(toks), s0 + ns))
            r.next_pos = len(toks)
            # cover every token we know of plus the next decode position
            self._grow_private(r, r.next_pos // block_size + 1)
            rows[i] = r.row
        return rows, hits, queries, computed_per_req

    def mark_computed(self, request_ids, block_size: int) -> None:
        for rid in request_ids:
            r = self.reqs.get(rid)
            if r is None:
                continue
            n = min(r.hashed_slots, r.computed_upto // block_size)
            for s in range(n):
                h = r.hashes[s]
                if h is not None and self.hash_to_block.get(h) == r.blocks[s]:
                    self.computed.add(h)

    def decode(self, request_ids, block_size: int, width: int, counts=None):
        """Block table rows for a non-prefix-cached step (decode: 1 token per
        request, or `counts[i]` tokens). Grows private blocks lazily and advances
        the host-tracked position. Returns (rows, positions_before_step)."""
        self.begin_step(request_ids)
        rows = _cx_np.empty((len(request_ids), width), dtype=_cx_np.int32)
        pos = []
        for i, rid in enumerate(request_ids):
            r = self._get_req(rid, width)
            p = r.next_pos
            n = counts[i] if counts is not None and i < len(counts) else 1
            # cover every new position plus the next one (absorbs an off-by-one)
            self._grow_private(r, (p + n) // block_size + 1)
            r.next_pos = p + n
            rows[i] = r.row
            pos.append(p)
        return rows, pos

    def peek_positions(self, request_ids):
        """Host-tracked position of each request's next token, or None if unknown."""
        out = []
        for rid in request_ids:
            r = self.reqs.get(rid)
            if r is None or not r.blocks:
                return None
            out.append(r.next_pos)
        return out

    def live_blocks(self) -> set:
        s = set()
        for r in self.reqs.values():
            s.update(b for b in r.blocks if b)
        return s
# END connex external-pp allocator
