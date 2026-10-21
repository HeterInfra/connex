# SPDX-License-Identifier: Apache-2.0
"""A ``pp_group`` for SGLang backed by Connex (external, elastic PP).

SGLang's scheduler talks to its pipeline neighbours only through
``scheduler.pp_group`` (a ``GroupCoordinator``): ``send_tensor_dict`` /
``recv_tensor_dict`` with a ``"__msg_type__"`` of ``"proxy"`` (hidden
states, forward direction) or ``"output"`` (sampled tokens), plus
``is_first_rank`` / ``is_last_rank`` and a few rank properties.
:class:`ConnexPPGroup` implements that surface on a Connex step transport
(:class:`connex.pp.endpoint.ConnexTEEndpoint`, or a fake), in the *external*
topology Connex supports (docs/design/sglang-adapter.md):

* stage 0 runs the only scheduler. ``send_tensor_dict(proxy)`` starts a
  pipeline step (new correlation id, cx_ctx + finished ids attached);
  ``recv_tensor_dict()`` returns the ``"output"`` of the oldest outstanding
  step (SGLang consumes microbatch results in launch order), re-sending it on
  timeout and handing ``cx_missing`` to the engine's ``reprefill`` hook;
* stages 1.. are passive executors. ``recv_tensor_dict()`` returns the next
  step's ``"proxy"`` dict after the KV-ownership check (rows the stage lacks
  are released and listed under :data:`MISSING_HERE`, so the engine must not
  cache them); the next ``send_tensor_dict`` answers that step (forward to
  the next stage, or -- on the last stage -- the ``"output"`` back to the
  stage 0 that sent it).

What this does NOT do (engine-side, see the design doc): run SGLang with one
scheduler and stateless executors, load only this stage's layers with
pp_size=1, supply request ids / num_computed per row, and implement the
EngineHooks (KV held per request on executors, re-prefill on stage 0).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Optional, Sequence

from connex.pp.handover import (
    CX_CTX, CX_MISSING, FINISHED, REQUEST_IDS, find_missing, merge_missing,
)
from connex.pp.link import Stage0Link

MSG_TYPE = "__msg_type__"
MISSING_HERE = "__cx_missing_here__"   # stages 1..: rows this stage lacks KV for
REPREFILLED = "__cx_reprefilled__"     # stage 0: rows the engine must recompute

_log = logging.getLogger(__name__)


class _DoneWork:
    """Stands in for SGLang's P2PWork (``.work.wait()``); Connex sends are
    complete (or handed to the endpoint's sender thread) when they return."""

    class _W:
        def wait(self):
            return True

        def is_completed(self):
            return True

    def __init__(self, payload=None):
        self.work = self._W()
        self.payload = payload


class ConnexPPGroup:
    """GroupCoordinator-compatible PP group for one external stage.

    ``row_info()`` (stage 0, optional) returns ``(request_ids, num_computed)``
    of the batch being sent, used for cx_ctx; without it no handover check
    runs (pure transport). ``hooks`` implements connex.pp.hooks.EngineHooks.
    """

    def __init__(self, transport, hooks=None, *,
                 row_info: Optional[Callable[[], tuple]] = None,
                 recv_timeout_s: Optional[float] = None, max_attempts: int = 5,
                 log: logging.Logger = _log):
        self.t = transport
        self.hooks = hooks
        self.row_info = row_info
        self.log = log
        self.world_size = transport.num_stages
        self.rank_in_group = transport.stage_idx
        self.ranks = list(range(self.world_size))
        self.device_group = self.cpu_group = None
        self._pending: deque = deque()      # stage 0: corr of outstanding steps
        self._ctx: deque = deque()          # stages 1..: (ctx, td) awaiting a reply
        self._s0 = (Stage0Link(transport, hooks, recv_timeout_s=recv_timeout_s,
                               max_attempts=max_attempts, log=log)
                    if transport.is_first else None)
        self.stats = {"sent": 0, "received": 0, "missing": 0}

    # ---- GroupCoordinator properties ----------------------------------------
    @property
    def is_first_rank(self) -> bool:
        return self.t.is_first

    @property
    def is_last_rank(self) -> bool:
        return self.t.is_last

    @property
    def next_rank(self) -> int:
        return (self.rank_in_group + 1) % self.world_size

    @property
    def prev_rank(self) -> int:
        return (self.rank_in_group - 1) % self.world_size

    # ---- finished requests (stage 0) -------------------------------------------
    def finished(self, request_ids: Sequence) -> None:
        if self._s0 is not None:
            self._s0.finished(request_ids)

    # ---- data path ---------------------------------------------------------------
    def send_tensor_dict(self, tensor_dict: dict, dst: Optional[int] = None,
                         all_gather_group: Any = None, async_send: bool = False):
        """``all_gather_group`` is ignored: Connex sends the full tensors
        (one sender per stage; attention-TP slicing is not supported yet)."""
        self.stats["sent"] += 1
        if self.t.is_first:
            rids, ncomp = self.row_info() if self.row_info is not None else (None, None)
            corr = self._s0.submit(dict(tensor_dict), rids, ncomp)
            self._pending.append(corr)
            return [_DoneWork()]
        if not self._ctx:
            raise RuntimeError("ConnexPPGroup: send on a later stage without a received step "
                               "(external PP: every send answers one recv)")
        ctx, td_in = self._ctx.popleft()
        out = dict(tensor_dict)
        out.pop(MISSING_HERE, None)
        merged = merge_missing(td_in.get(CX_MISSING), td_in.get(MISSING_HERE))
        if merged:
            out[CX_MISSING] = merged
        if self.t.is_last:
            out[MSG_TYPE] = "output"
        else:
            for k in (REQUEST_IDS, CX_CTX, FINISHED):
                if k in td_in and k not in out:
                    out[k] = td_in[k]
        self.t.send_reply(out, ctx)
        return [_DoneWork()]

    def recv_tensor_dict(self, src: Optional[int] = None, all_gather_group: Any = None):
        self.stats["received"] += 1
        if self.t.is_first:
            if not self._pending:
                raise RuntimeError("ConnexPPGroup: stage 0 recv without an outstanding step")
            reply = self._s0.wait(self._pending.popleft())
            reply.setdefault(MSG_TYPE, "output")
            return reply
        while True:
            td, ctx = self.t.recv_forward()
            if td is not None:
                break
        fin = td.get(FINISHED)
        rids = td.get(REQUEST_IDS) or []
        if fin and self.hooks is not None:
            live = set(rids)
            done = [r for r in fin if r not in live]
            if done:
                self.hooks.release(done)
        mine = []
        if self.hooks is not None:
            mine = find_missing(rids, td.get(CX_CTX), self.hooks.kv_held)
            if mine:
                self.log.warning("stage %s: %d/%d request(s) have no KV here",
                                 self.t.stage_idx, len(mine), len(rids))
                self.hooks.release(mine)
                self.stats["missing"] += len(mine)
        td[MISSING_HERE] = mine
        self._ctx.append((ctx, td))
        td.setdefault(MSG_TYPE, "proxy")
        return td

    def send_recv_tensor_dict(self, send_tensor_dict, send_dst=None, recv_src=None,
                              send_all_gather_group=None, recv_all_gather_group=None):
        self.send_tensor_dict(send_tensor_dict)
        return self.recv_tensor_dict()

    # ---- object relay (not used in external mode) ---------------------------------
    def send_object(self, obj, dst: int, async_send: bool = False, tag: int = 0):
        raise NotImplementedError(
            "external PP has one scheduler (stage 0); requests are not relayed between stages")

    def recv_object(self, src: int, tag: int = 0):
        raise NotImplementedError(
            "external PP has one scheduler (stage 0); requests are not relayed between stages")

    def barrier(self):
        return None

    def destroy(self):
        close = getattr(self.t, "close", None)
        if close is not None:
            close()
