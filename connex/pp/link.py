# SPDX-License-Identifier: Apache-2.0
"""Engine-facing drivers: the step loop of stage 0 and of later stages.

These encode the engine contract (docs/design/engine-contract.md) once, on
top of any :class:`connex.pp.hooks.StepTransport` (the production
:class:`connex.pp.endpoint.ConnexTEEndpoint`, or a fake):

* :class:`Stage0Link` -- submit a step, wait for its reply with per-step
  retries under the same correlation id, failover-aware give-up, K steps in
  flight, ``cx_ctx`` / ``finished_req_ids`` piggybacking, and ``cx_missing``
  handling (re-prefill or abort through :class:`EngineHooks`).
* :class:`StageWorker` -- receive a step, release finished requests, find
  rows whose KV this stage lacks, run the engine's forward, and send the
  result on with the merged ``cx_missing``.

An engine adapter may use these directly (the SGLang adapter does) or
re-implement the same rules inside the engine (the vLLM fork's patches do,
for historical reasons; same keys, same behaviour).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from connex.pp.handover import (
    CX_CTX, CX_MISSING, FINISHED, REQUEST_IDS, FinishedOutbox, RewindBudget,
    find_missing, merge_missing,
)

_log = logging.getLogger(__name__)


class StepFailed(RuntimeError):
    """A step got no reply after all retries (no healthy downstream)."""


class Stage0Link:
    """Stage-0 side of the step protocol.

    ``max_attempts`` bounds the attempts of one step while the downstream is
    not known healthy (with a healthy downstream peer a slow step keeps being
    retried, as in the vLLM fork's K>1 loop). ``recv_timeout_s`` = None uses
    the endpoint's adaptive timeout."""

    def __init__(self, transport, hooks=None, *, max_attempts: int = 5,
                 recv_timeout_s: Optional[float] = None, reprefill_max: int = 8,
                 finished_repeat: int = 2, send_ctx: bool = True,
                 max_healthy_attempts: int = 1000, log: logging.Logger = _log):
        self.t = transport
        self.hooks = hooks
        self.max_attempts = max_attempts
        self.max_healthy_attempts = max_healthy_attempts
        self.recv_timeout_s = recv_timeout_s
        self.send_ctx = send_ctx
        self.rewinds = RewindBudget(reprefill_max)
        self.outbox = FinishedOutbox(finished_repeat)
        self.log = log
        self._sent: dict[int, dict] = {}
        self.stats = {"steps": 0, "retries": 0, "reprefilled": 0, "aborted": 0}

    @property
    def inflight(self) -> int:
        return getattr(self.t, "inflight", 1)

    def finished(self, request_ids: Sequence) -> None:
        """Requests finished on stage 0: tell downstream stages (piggybacked)."""
        self.outbox.add(request_ids)
        self.rewinds.forget(request_ids)

    def submit(self, tensor_dict: dict, request_ids: Optional[Sequence] = None,
               num_computed: Optional[Sequence[int]] = None) -> int:
        """Send one step; returns its id. ``request_ids[i]`` / ``num_computed[i]``
        describe row i (tokens already computed before this step)."""
        td = dict(tensor_dict)
        if request_ids is not None:
            td.setdefault(REQUEST_IDS, list(request_ids))
            if self.send_ctx and num_computed is not None:
                td[CX_CTX] = [int(n) for n in num_computed]
        fin = self.outbox.take()
        if fin:
            td[FINISHED] = fin
        corr = self.t.new_step_id()
        self._sent[corr] = td  # a retry re-sends exactly this dict
        self.t.send_step(td, corr)
        self.stats["steps"] += 1
        return corr

    def wait(self, corr: int) -> dict:
        """Reply of step ``corr``; re-sends it on timeout. Handles
        ``cx_missing`` (hooks.reprefill / hooks.abort) and strips it."""
        attempts = 0
        while True:
            reply = self.t.recv_step(corr, self.recv_timeout_s)
            if reply is not None:
                break
            attempts += 1
            healthy = self.t.downstream_healthy()
            limit = self.max_healthy_attempts if healthy else self.max_attempts
            if attempts >= limit:
                self.t.abandon_step(corr)
                self._sent.pop(corr, None)
                raise StepFailed(f"step {corr:x}: no reply after {attempts} attempts")
            self.stats["retries"] += 1
            self.log.warning("step %x: no reply (attempt %d), re-sending", corr, attempts)
            self.t.send_step(self._sent[corr], corr)
            self._resend_stale(corr)
        self._sent.pop(corr, None)
        missing = reply.pop(CX_MISSING, None)
        if missing:
            again, abort = self.rewinds.split(missing)
            if again and self.hooks is not None:
                self.hooks.reprefill(again)
            if abort and self.hooks is not None:
                self.hooks.abort(abort)
            self.stats["reprefilled"] += len(again)
            self.stats["aborted"] += len(abort)
            reply["__cx_reprefilled__"] = again
            reply["__cx_aborted__"] = abort
        return reply

    def _resend_stale(self, current: int) -> None:
        """Fast handover: the other in-flight steps sent on a chain that has
        since failed over are lost too; re-send them all now (oldest first)
        instead of letting each wait out its own recv timeout. Transports
        without ``step_stale`` (optional) keep per-step retries only."""
        stale = getattr(self.t, "step_stale", None)
        if stale is None:
            return
        for c in sorted(c for c in self._sent if c != current):
            if stale(c):
                self.stats["retries"] += 1
                self.log.warning("step %x: in flight on a failed chain, re-sending", c)
                self.t.send_step(self._sent[c], c)

    def step(self, tensor_dict: dict, request_ids=None, num_computed=None) -> dict:
        return self.wait(self.submit(tensor_dict, request_ids, num_computed))


class StageWorker:
    """Stages 1.. : receive, check KV ownership, compute, send on.

    ``compute(tensor_dict, missing) -> dict`` is the engine's forward for this
    stage. Rows of ``missing`` requests have been released (hooks.release)
    and must not be cached by the engine; their output is garbage that stage
    0 discards. ``commit(tensor_dict, missing)`` (optional) runs after the
    forward, e.g. :meth:`connex.pp.handover.KVLedger.commit`."""

    def __init__(self, transport, hooks, compute: Callable[[dict, list], dict], *,
                 commit: Optional[Callable[[dict, list], None]] = None,
                 log: logging.Logger = _log):
        self.t = transport
        self.hooks = hooks
        self.compute = compute
        self.commit = commit
        self.log = log
        self.stats = {"steps": 0, "missing": 0, "released_finished": 0}

    def serve_one(self) -> bool:
        """Handle one step. False on receive timeout."""
        td, ctx = self.t.recv_forward()
        if td is None:
            return False
        self.stats["steps"] += 1
        fin = td.get(FINISHED)
        rids = td.get(REQUEST_IDS) or []
        if fin:
            # a request finished upstream but scheduled in this very step
            # cannot be finished yet (re-sent older step); release the rest
            live = set(rids)
            done = [r for r in fin if r not in live]
            if done:
                self.hooks.release(done)
                self.stats["released_finished"] += len(done)
        ctx_list = td.get(CX_CTX)
        mine = find_missing(rids, ctx_list, self.hooks.kv_held)
        if mine:
            self.log.warning("stage %s: %d/%d request(s) have no KV here; reporting missing",
                             getattr(self.t, "stage_idx", "?"), len(mine), len(rids))
            self.hooks.release(mine)
            self.stats["missing"] += len(mine)
        out = dict(self.compute(td, mine) or {})
        if self.commit is not None:
            self.commit(td, mine)
        merged = merge_missing(td.get(CX_MISSING), mine)
        if merged:
            out[CX_MISSING] = merged
        if not self.t.is_last:
            for k in (REQUEST_IDS, CX_CTX, FINISHED):
                if k in td and k not in out:
                    out[k] = td[k]
        self.t.send_reply(out, ctx)
        return True
