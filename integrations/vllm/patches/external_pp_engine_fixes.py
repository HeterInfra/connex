# SPDX-License-Identifier: Apache-2.0
"""Pure logic for the external-PP engine fixes (patches 11-13).

Like external_pp_alloc.py this file is BOTH an importable module (unit-tested in
tests/test_engine_fixes.py) and the source that the patches inline verbatim into
vLLM's gpu_model_runner.py (everything between the BEGIN/END markers), so it
must stay self-contained: stdlib only.

* connex_pp_prefill_lens: attention lengths of a prefill-type step on stages 1+
  (patch_external_pp_chunked_seqlens.py).
* ConnexPPFinishedOutbox: stage 0 side of finished-request piggybacking
  (patch_external_pp_finished_ids.py).
* ConnexPPFinishTracker: stages 1+ side; releases finished requests in the
  ConnexPPBlockAllocator, safely under retries and reordering.
* connex_pp_host_logits_indices: stage-0 logits indices from the host copy of
  query_start_loc (patch_external_pp_host_logits_idx.py).
"""
# BEGIN connex external-pp engine fixes
from collections import OrderedDict as _CxFinOrderedDict


def connex_pp_prefill_lens(num_scheduled, num_computed=None):
    """Per-request lengths of a prefill-type external-PP step.

    num_scheduled[i]: tokens of request i in this step (the query).
    num_computed[i]:  tokens of request i already in the KV cache (stage 0's
                      num_computed_tokens; context of a later chunk or of a
                      decode request riding in a prefill batch).
    Returns (seq_lens, query_start_loc, context_lens, max_seq_len, max_query_len)
    with seq_len = context + query, as vLLM's own CommonAttentionMetadata.
    """
    n = len(num_scheduled)
    if num_computed is None or len(num_computed) != n:
        num_computed = [0] * n
    ctx = [max(0, int(c)) for c in num_computed]
    q = [int(s) for s in num_scheduled]
    qsl = [0] * (n + 1)
    for i in range(n):
        qsl[i + 1] = qsl[i] + q[i]
    seq = [c + s for c, s in zip(ctx, q)]
    return seq, qsl, ctx, (max(seq) if n else 0), (max(q) if n else 0)


def connex_pp_host_logits_indices(query_start_loc_np, num_reqs):
    """Same values as vLLM's logits_indices = query_start_loc[1:] - 1 (no spec
    decode), from the host copy, so no device sync is needed."""
    return [int(x) - 1 for x in query_start_loc_np[1:num_reqs + 1]]


class ConnexPPFinishedOutbox:
    """Stage 0: request ids finished since they were last sent downstream.

    Every id is attached to the next `repeat` sends (default 2). A retried or
    re-sent step carries the same dict, so the ids are not lost with it; the
    repeat only covers a step whose reply was answered from a reply cache or a
    replica switch. Downstream release is idempotent.
    """

    def __init__(self, repeat: int = 2, max_pending: int = 65536) -> None:
        self.repeat = max(1, int(repeat))
        self.max_pending = max_pending
        self._left = _CxFinOrderedDict()  # rid -> sends left

    def add(self, req_ids) -> None:
        for rid in req_ids:
            self._left[rid] = self.repeat
            self._left.move_to_end(rid)
        while len(self._left) > self.max_pending:
            self._left.popitem(last=False)

    def take(self) -> list:
        if not self._left:
            return []
        out = list(self._left)
        for rid in out:
            n = self._left[rid] - 1
            if n > 0:
                self._left[rid] = n
            else:
                del self._left[rid]
        return out

    def __len__(self) -> int:
        return len(self._left)


class ConnexPPFinishTracker:
    """Stages 1+: apply finished ids to the block allocator.

    on_step(alloc, finished, request_ids) runs before the step's own
    allocation. Rules:
      * a finished id that is NOT in this step is released now and
        tombstoned (bounded LRU of recently finished ids);
      * a finished id that IS in this step (cannot happen with in-order
        delivery, but a re-sent step can carry it) is deferred: released at
        the first later step that does not reference it;
      * a step that references a tombstoned id (an older step re-sent or
        reordered after the finish notice) re-creates its state in the
        allocator; it is released again at the next step that does not
        reference it, instead of leaking until the TTL.
    Steps are processed by one thread, and a step's blocks are only reused by
    later steps (stream-ordered), so release-before-allocate is safe.
    `alloc` may be None (nothing allocated yet): only bookkeeping happens.
    """

    def __init__(self, max_tombstones: int = 65536) -> None:
        self.max_tombstones = max_tombstones
        self.tombstones = _CxFinOrderedDict()
        self.deferred: set = set()
        self.revived: set = set()
        self.stats = {"released": 0, "deferred": 0, "revived": 0, "notices": 0}
        self.steps = 0

    def _tomb(self, rid) -> None:
        self.tombstones[rid] = None
        self.tombstones.move_to_end(rid)
        while len(self.tombstones) > self.max_tombstones:
            self.tombstones.popitem(last=False)

    def _release(self, alloc, rid) -> None:
        if alloc is not None and rid in alloc.reqs:
            alloc.release(rid)
            self.stats["released"] += 1
        self._tomb(rid)

    def on_step(self, alloc, finished, request_ids) -> None:
        self.steps += 1
        batch = set(request_ids or ())
        if self.deferred or self.revived:
            for rid in list(self.deferred | self.revived):
                if rid not in batch:
                    self.deferred.discard(rid)
                    self.revived.discard(rid)
                    self._release(alloc, rid)
        for rid in finished or ():
            self.stats["notices"] += 1
            if rid in batch:
                if rid not in self.deferred:
                    self.deferred.add(rid)
                    self.stats["deferred"] += 1
            else:
                self._release(alloc, rid)
        if self.tombstones:
            for rid in batch:
                if rid in self.tombstones and rid not in self.deferred \
                        and rid not in self.revived:
                    self.revived.add(rid)
                    self.stats["revived"] += 1
# END connex external-pp engine fixes


# ---- used by the patch scripts (not inlined) --------------------------------
_MOD_MARK = "# BEGIN connex external-pp engine fixes"
_MOD_ANCHOR = "# END connex external-pp allocator\n"


def inline_module(src: str):
    """Insert the block between the markers once, after the inlined
    allocator (patch_external_pp_allocator.py). Returns None if the anchor
    is missing."""
    if _MOD_MARK in src:
        return src
    if src.count(_MOD_ANCHOR) != 1:
        return None
    import os
    me = open(os.path.abspath(__file__.replace(".pyc", ".py"))).read()
    b = me.index(_MOD_MARK)
    e = me.index("# END connex external-pp engine fixes")
    block = me[b:e] + "# END connex external-pp engine fixes\n"
    return src.replace(_MOD_ANCHOR, _MOD_ANCHOR + "\n\n" + block, 1)


def run_patch(script_file: str, patch_fn, argv) -> int:
    import os
    path = argv[1]
    src = open(path).read()
    new, status = patch_fn(src)
    print(f"{os.path.basename(script_file)}: {status}")
    if new is None:
        return 1
    if new != src:
        open(path, "w").write(new)
    return 0
