# SPDX-License-Identifier: Apache-2.0
"""Pure logic for re-prefill on failover (patch_external_pp_reprefill.py).

Like external_pp_alloc.py this file is BOTH an importable module (unit-tested
in tests/test_reprefill.py) and the source that the patch inlines verbatim into
vLLM's gpu_model_runner.py (everything between the BEGIN/END markers), so it
must stay self-contained: stdlib only.

ConnexPPKVGuard (stages 1+) decides, before a step runs, which of its requests
this stage cannot compute correctly because it does not hold their KV context:

  ctx[i]  = tokens of request i already computed according to stage 0
            (stage 0 sends it as "cx_ctx" on every step);
  held(i) = tokens of request i whose KV this stage wrote itself
            (kv_len, advanced by every step that ran here) while the
            request is still live in the ConnexPPBlockAllocator, or, for a
            prefill-type step, the stage's own computed prefix-cache blocks.

A request with ctx > held is "missing": typically a decode step re-routed to
a replica that never saw the request (sticky-peer failover after a downstream
replica died or left), a replica that restarted, or a request reclaimed by the
allocator's TTL/LRU. Missing requests are released from the allocator and
blanked (no token ids -> no content-addressed blocks, nothing marked
computed), so the garbage they produce in this step is written only to the
null block or a private block; their ids go back to stage 0 ("cx_missing"),
which rewinds them to num_computed_tokens = 0 and re-prefills prompt + already
generated tokens on the current chain.
"""
# BEGIN connex external-pp reprefill


class ConnexPPKVGuard:
    """Stages 1+: per-request KV-context bookkeeping (see module docstring)."""

    def __init__(self, hash_fn=None, prune_every: int = 512) -> None:
        self.hash_fn = hash_fn
        self.kv_len: dict = {}
        self.prune_every = max(1, int(prune_every))
        self.stats = {"steps": 0, "missing": 0, "missing_steps": 0}
        self._commits = 0

    def own_prefix(self, alloc, toks, block_size: int, produced=None) -> int:
        """Tokens of `toks` covered by this stage's computed prefix-cache
        blocks (consecutive from the start), as ConnexPPBlockAllocator.prefill
        would find them. `produced` (optional set of block hashes) adds blocks
        that a valid row of the SAME step writes: alloc.prefill maps equal
        hashes to the same block and the forward writes every row's KV before
        attention reads it, but alloc.computed is only filled after the
        forward, so without this a same-step shared prefix looks missing."""
        if self.hash_fn is None or not toks or block_size <= 0:
            return 0
        if alloc is None and not produced:
            return 0
        prev = b"NONE_HASH"
        n = 0
        for i in range(len(toks) // block_size):
            prev = self.hash_fn(prev, toks[i * block_size:(i + 1) * block_size])
            if (alloc is not None and prev in alloc.computed
                    and alloc.hash_to_block.get(prev) is not None) \
                    or (produced and prev in produced):
                n += 1
            else:
                break
        return n * block_size

    def _produced(self, toks, upto: int, block_size: int) -> list:
        """Hashes of the full blocks of toks[:upto]."""
        out = []
        prev = b"NONE_HASH"
        for i in range(min(upto, len(toks)) // block_size):
            prev = self.hash_fn(prev, toks[i * block_size:(i + 1) * block_size])
            out.append(prev)
        return out

    def held(self, alloc, rid) -> int:
        if alloc is None or rid not in alloc.reqs:
            return 0
        return self.kv_len.get(rid, 0)

    def check(self, alloc, request_ids, ctx, step_type, all_token_ids,
              block_size: int, num_scheduled=None) -> list:
        """Ids of this step's requests whose context this stage lacks. They
        are released from `alloc` (their next appearance must be a fresh
        prefill).

        Prefill steps: a row may also rely on prefix blocks written by another
        valid row of this same step (stage 0 caches full blocks at schedule
        time, so the second of two requests sharing a prefix arrives with
        ctx=16 and only its suffix). Earlier steps need no extra handling:
        this stage runs steps in order and marks their blocks computed before
        the next check, and a producer that was itself missing was blanked, so
        nothing it "produced" is ever trusted."""
        self.stats["steps"] += 1
        missing = []
        if not request_ids or ctx is None:
            return missing
        n = len(request_ids)
        cs = [int(ctx[i]) if i < len(ctx) else 0 for i in range(n)]
        prefill = (step_type == "prefill" and all_token_ids is not None
                   and self.hash_fn is not None)
        toks = [all_token_ids[i] if prefill and i < len(all_token_ids) else None
                for i in range(n)]
        own = [None] * n
        ok = [False] * n
        for i, rid in enumerate(request_ids):
            if cs[i] <= 0:
                ok[i] = True
                continue
            h = self.held(alloc, rid)
            if h < cs[i] and toks[i] is not None:
                own[i] = self.own_prefix(alloc, toks[i], block_size)
                h = max(h, own[i])
            ok[i] = h >= cs[i]
        if prefill and not all(ok):
            produced = set()
            done = [False] * n
            changed = True
            while changed:  # fixpoint: a row accepted via `produced` produces too
                changed = False
                for i in range(n):
                    if ok[i] and not done[i] and toks[i]:
                        done[i] = True
                        ns = (int(num_scheduled[i]) if num_scheduled is not None
                              and i < len(num_scheduled) else len(toks[i]) - cs[i])
                        new = self._produced(toks[i], cs[i] + max(ns, 0), block_size)
                        if not produced.issuperset(new):
                            produced.update(new)
                            changed = True
                if changed:
                    for i in range(n):
                        if not ok[i] and toks[i]:
                            if self.own_prefix(alloc, toks[i], block_size,
                                               produced) >= cs[i]:
                                ok[i] = True
        for i, rid in enumerate(request_ids):
            if not ok[i]:
                missing.append(rid)
        if missing:
            self.stats["missing"] += len(missing)
            self.stats["missing_steps"] += 1
            self.forget(alloc, missing)
        return missing

    def forget(self, alloc, rids) -> None:
        for rid in rids:
            self.kv_len.pop(rid, None)
            if alloc is not None:
                alloc.release(rid)

    def commit(self, alloc, request_ids, ctx, step_type, num_scheduled, missing) -> None:
        """After the step ran: requests that were computed correctly now hold
        ctx + scheduled tokens; missing ones are released again (the step
        re-created them in the allocator)."""
        if not request_ids or ctx is None:
            return
        miss = set(missing or ())
        for i, rid in enumerate(request_ids):
            if rid in miss:
                continue
            c = int(ctx[i]) if i < len(ctx) else 0
            if step_type == "decode" or num_scheduled is None or i >= len(num_scheduled):
                n = 1
            else:
                n = int(num_scheduled[i])
            # max: a re-sent older step must not shrink what we hold
            if c + n > self.kv_len.get(rid, 0) or c == 0:
                self.kv_len[rid] = c + n
        if miss:
            self.forget(alloc, miss)
        self._commits += 1
        if self._commits % self.prune_every == 0 and alloc is not None:
            for rid in [r for r in self.kv_len if r not in alloc.reqs]:
                del self.kv_len[rid]


def connex_pp_blank_rows(all_token_ids, request_ids, missing):
    """Copy of all_token_ids with the rows of `missing` requests emptied."""
    miss = set(missing)
    return [[] if (i < len(request_ids) and request_ids[i] in miss) else t
            for i, t in enumerate(all_token_ids)]


def connex_pp_merge_missing(upstream, mine):
    """Union of the missing ids reported by earlier stages and this one."""
    if not upstream:
        return list(mine) if mine else None
    if not mine:
        return list(upstream)
    out = list(upstream)
    seen = set(out)
    out.extend(r for r in mine if r not in seen)
    return out
# END connex external-pp reprefill
