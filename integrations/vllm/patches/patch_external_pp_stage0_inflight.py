# SPDX-License-Identifier: Apache-2.0
"""Patch 7 (pipelining, CONNEX_PP_INFLIGHT=K): stage 0 keeps K steps in flight.

    python patch_external_pp_stage0_inflight.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Edits three files of the fork (all idempotent, each with its own marker):
  * v1/worker/gpu_model_runner.py      (the path given; required)
  * v1/executor/uniproc_executor.py    (../executor/ next to the runner)
  * v1/core/sched/scheduler.py         (../core/sched/ next to the runner)
The two siblings are skipped with a message when absent (runner-only copies in
unit tests). Anchors are on the runner as left by patches 0-6 and on the
executor/scheduler as shipped.

Design
------
Engine side (no engine code is edited): UniProcExecutor.max_concurrent_batches
returns K for external-PP mode when CONNEX_PP_INFLIGHT=K>1, so EngineCore uses
step_with_batch_queue (a FIFO of at most K outstanding steps; it schedules a
new step while older ones are outstanding and blocks on the oldest only when
the queue is full or nothing else is schedulable). The V1 scheduler already
skips a request whose sampled token is not back (num_new_tokens == 0), so
outstanding steps hold disjoint requests. To actually split the running set
into K microbatches the scheduler patch caps the number of RUNNING requests
scheduled per step at ceil(len(running) / K) (new requests from the waiting
queue are not capped).

Executor: runner.sample_tokens of an in-flight step returns a
_CxDeferredOutput; collective_rpc turns it into a _CxLazyFuture whose
result() runs the deferred work (recv + logits + sampling + bookkeeping) in
the calling thread, i.e. the EngineCore thread, when the engine pops that
step. Everything that touches the runner therefore stays single-threaded (no
AsyncModelRunnerOutput / async_output_thread is involved). With non_block=False
the deferred work runs immediately.

Runner (stage 0 only, i.e. pipeline_stage_mode == "external" and first rank,
and only when the communicator offers the per-step API with inflight > 1):
  * K InputBatch objects (slot 0 is the runner's own; the others are built
    with the same constructor kwargs, recorded by _cx_input_batch, and fresh
    logits processors). They share slot 0's GPU block-table and slot-mapping
    buffers: full CUDA graphs (FULL_AND_PIECEWISE, the default) bake in those
    addresses, and with private buffers slots 1..K-1 attended through slot
    0's stale tables (cross-request KV contamination seen on GPU). execute_model picks a free slot (no outstanding step)
    and swaps self.input_batch to it before _update_states.
  * Per-request state lives in self.requests (CachedRequestState); an
    InputBatch row is only trusted in the slot that last scheduled that
    request (_CxSlots.home). Before _update_states, rows of the chosen batch
    that are scheduled now but whose home is another slot are removed, so
    _update_states re-adds them from CachedRequestState (the stock path for
    un-scheduled/re-scheduled requests). Unscheduled rows are removed by
    _update_states itself. Rows of an outstanding step are never touched (a
    chunked-prefill request may be in an outstanding step and scheduled again
    for its next chunk; the outstanding row is a discarded partial sample).
  * Requests finished (aborted) while in an outstanding step: their
    CachedRequestState is kept aside (zombie) and re-inserted into
    self.requests only while that step's bookkeeping runs.
  * The external-PP branch of execute_model runs local layers as before, then
    new_step_id() + send_step() and stashes (corr, forward dict with cloned
    tensors for retries, scheduler_output, logits_indices, discard indices,
    kv/ec connector outputs) in the slot and returns None. sample_tokens
    returns the deferred object. The deferred run does recv_step(corr) with
    per-step retry (send_step under the same corr on timeout, up to
    CONNEXTE_RECV_RETRIES), compute_logits, then swaps the slot's InputBatch
    in, restores the slot's discard indices, rebuilds ExecuteModelState and
    calls the stock sample_tokens (sampling, bookkeeping, ModelRunnerOutput),
    and swaps back.

K=1 (default) or a backend without the step API: all inserted branches are
behind `_CX_PP_INFLIGHT > 1` (module constant) and the runner-side check, the
executor returns today's max_concurrent_batches and the scheduler cap is off.

Limitations
-----------
* Requires the connexte backend (new_step_id/send_step/recv_step) and
  CONNEX_PP_INFLIGHT on stage 0 AND later stages (their prefetch loop, patch 6).
* Spec decode and async scheduling: the runner falls back to K=1 behaviour
  (with a warning) but the executor still uses the batch queue. Structured
  output (guided decoding) is not supported with K>1: the engine's deferred
  grammar path would call sample_tokens with no outstanding step.
* Stage-0 TTFT/TPOT pipeline metrics (_external_pp_tpot_tracking) are not
  recorded for K>1 steps.
* Per-request latency (TPOT) does not improve; throughput does when there are
  enough concurrent requests to fill K microbatches.
"""
import os
import sys

MARK = "# connex: stage-0 pipelining (CONNEX_PP_INFLIGHT)"

# ------------------------------------------------------------------ runner
R_MODULE_ANCHOR = "class ExecuteModelState(NamedTuple):\n"
R_MODULE = MARK + '''
_CX_PP_INFLIGHT = max(1, int(os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))


def _cx_input_batch(**kw):
    """InputBatch(**kw) that remembers its kwargs so stage-0 pipelining can
    build K-1 more batches of the same shape."""
    ib = InputBatch(**kw)
    ib._cx_kwargs = kw
    return ib


class _CxDeferredOutput:
    """sample_tokens result of an outstanding stage-0 step; the patched
    UniProcExecutor runs _cx_deferred_run in the engine thread on result()."""

    __slots__ = ("_cx_deferred_run",)

    def __init__(self, fn):
        self._cx_deferred_run = fn


class _CxSlots:
    """Pure bookkeeping of K stage-0 slots (one InputBatch each)."""

    def __init__(self, k: int):
        self.k = k
        self.busy: dict = {}     # slot -> outstanding step state
        self.home: dict = {}     # req_id -> slot of the last step scheduling it
        self.zombies: dict = {}  # req_id -> (slot, state): finished while outstanding

    def acquire(self) -> int:
        for s in range(self.k):
            if s not in self.busy:
                return s
        raise RuntimeError("connex: all %d stage-0 slots are outstanding" % self.k)

    def outstanding_slot_of(self, rid):
        for s, st in self.busy.items():
            if rid in st["req_ids"]:
                return s
        return None

    def stale_rows(self, slot: int, batch_req_ids, scheduled) -> list:
        """Rows of batch `slot` scheduled now whose last scheduling was in
        another slot: their row is outdated and must be re-added."""
        return [r for r in batch_req_ids
                if r is not None and r in scheduled and self.home.get(r) != slot]

    def note_finished(self, finished, get_state) -> None:
        """Keep the state of finished (aborted) requests that are still in an
        outstanding step; forget their home slot."""
        for rid in finished:
            s = self.outstanding_slot_of(rid)
            st = get_state(rid)
            if s is not None and st is not None:
                self.zombies[rid] = (s, st)
            self.home.pop(rid, None)

    def set_home(self, slot: int, scheduled) -> None:
        for rid in scheduled:
            self.home[rid] = slot

    def zombies_of(self, slot: int) -> dict:
        return {r: st for r, (s, st) in self.zombies.items() if s == slot}

    def release(self, slot: int) -> None:
        self.busy.pop(slot, None)
        for r in [r for r, (s, _) in self.zombies.items() if s == slot]:
            del self.zombies[r]


'''

R_CTOR_OLD = "self.input_batch = InputBatch(\n"
R_CTOR_NEW = "self.input_batch = _cx_input_batch(\n"

R_EXEC_ANCHOR = '''        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("gpu_model_runner: preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)
'''
R_EXEC = '''        if _CX_PP_INFLIGHT > 1 and self._cx_k() > 1:
            self._cx_begin_step(scheduler_output)  # connex: swap in a free slot
''' + R_EXEC_ANCHOR

R_SEND_ANCHOR = '''                # ── Churn-resilient send+recv with retry ──
'''
R_SEND = '''                if _CX_PP_INFLIGHT > 1 and self._cx_k() > 1:
                    # connex: non-blocking stage-0 step (reply handled on result())
                    return self._cx_send_step(
                        forward_tensor_dict, scheduler_output, logits_indices,
                        spec_decode_metadata, spec_decode_common_attn_metadata,
                        aux_hidden_states, ec_connector_output, kv_connector_output,
                    )
''' + R_SEND_ANCHOR

R_SAMPLE_ANCHOR = '''    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
'''
R_SAMPLE = '''    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if getattr(self, "_cx_last_slot", None) is not None:
            _cx_slot, self._cx_last_slot = self._cx_last_slot, None
            return _CxDeferredOutput(
                lambda g=grammar_output, s=_cx_slot: self._cx_finish_step(s, g))
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
'''

R_METHODS_ANCHOR = "    def take_draft_token_ids(self) -> DraftTokenIds | None:\n"
R_METHODS = '''    # ---- connex: stage-0 pipelining (CONNEX_PP_INFLIGHT=K) -------------------
    def _cx_k(self) -> int:
        k = getattr(self, "_cx_k_cached", None)
        if k is not None:
            return k
        k = 1
        if (self.parallel_config.pipeline_stage_mode == "external"
                and get_pp_group().is_first_rank):
            comm = getattr(get_pp_group(), "zeromq_comm", None)
            if not (getattr(comm, "inflight", 1) > 1 and hasattr(comm, "new_step_id")
                    and hasattr(comm, "send_step") and hasattr(comm, "recv_step")):
                logger.warning("connex: CONNEX_PP_INFLIGHT=%d but the PP communicator "
                               "has no per-step API; stage 0 stays serial", _CX_PP_INFLIGHT)
            elif self.speculative_config or self.use_async_scheduling:
                logger.warning("connex: stage-0 pipelining disabled (spec decode / "
                               "async scheduling)")
            else:
                k = min(_CX_PP_INFLIGHT, int(comm.inflight))
                logger.info("connex: stage-0 pipelining with %d steps in flight", k)
        self._cx_k_cached = k
        return k

    def _cx_new_input_batch(self):
        kw = dict(self.input_batch._cx_kwargs)
        lp = self.model_config.logits_processors
        kw["logitsprocs"] = build_logitsprocs(
            self.vllm_config, self.device, self.pin_memory, self.is_pooling_model,
            tuple(lp) if lp is not None else ())
        ib = InputBatch(**kw)
        # Full CUDA graphs bake in the device addresses of slot 0's block
        # table and slot mapping: every slot must write the same GPU buffers
        # (safe, each step commits its own rows before its forward and the
        # forwards are serialised on one stream).
        for mine, base in zip(ib.block_table.block_tables,
                              self.input_batch.block_table.block_tables):
            mine.block_table.gpu = base.block_table.gpu
            mine.slot_mapping.gpu = base.slot_mapping.gpu
        return ib

    def _cx_begin_step(self, scheduler_output) -> None:
        S = getattr(self, "_cx_slots", None)
        if S is None:
            S = self._cx_slots = _CxSlots(self._cx_k())
            self._cx_batches = [self.input_batch] + [
                self._cx_new_input_batch() for _ in range(S.k - 1)]
        slot = S.acquire()
        self.input_batch = self._cx_batches[slot]
        self._cx_cur_slot = slot
        scheduled = scheduler_output.num_scheduled_tokens
        S.note_finished(scheduler_output.finished_req_ids, self.requests.get)
        for rid in S.stale_rows(slot, list(self.input_batch.req_ids), scheduled):
            self.input_batch.remove_request(rid)
        S.set_home(slot, scheduled)

    def _cx_send_step(self, forward_tensor_dict, scheduler_output, logits_indices,
                      spec_decode_metadata, spec_decode_common_attn_metadata,
                      aux_hidden_states, ec_connector_output, kv_connector_output):
        comm = get_pp_group().zeromq_comm
        slot = self._cx_cur_slot
        # Keep a private copy for retries: cudagraph output buffers are reused
        # by the next step's forward.
        td = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in forward_tensor_dict.items()}
        corr = comm.new_step_id()
        try:
            comm.send_step(td, corr)
        except Exception:
            comm.abandon_step(corr)
            raise
        n = self.num_discarded_requests
        self._cx_slots.busy[slot] = {
            "corr": corr, "td": td, "so": scheduler_output,
            "li": logits_indices, "sdm": spec_decode_metadata,
            "scm": spec_decode_common_attn_metadata, "aux": aux_hidden_states,
            "ec": ec_connector_output, "kv": kv_connector_output,
            "disc": self.discard_request_indices.np[:n].copy(),
            "req_ids": set(r for r in self.input_batch.req_ids if r is not None),
        }
        self._cx_last_slot = slot
        self.execute_model_state = None
        self.kv_connector_output = None
        return None

    def _cx_finish_step(self, slot: int, grammar_output):
        S = self._cx_slots
        st = S.busy[slot]
        comm = get_pp_group().zeromq_comm
        try:
            ret = None
            # A lost step is re-sent under the same id (downstream answers
            # duplicates from its reply cache). Past CONNEXTE_RECV_RETRIES
            # attempts keep going while the downstream chain is healthy (a
            # lost message is not a dead pipeline), up to
            # CONNEX_PP_STEP_BUDGET_S; only then give the engine up.
            t_first = time.monotonic()
            budget = float(os.environ.get("CONNEX_PP_STEP_BUDGET_S", "60") or "60")
            healthy = getattr(comm, "downstream_healthy", lambda: False)
            attempt = 0
            while True:
                if attempt:
                    logger.warning("Stage 0: step %s timed out, re-sending (%d/%d)",
                                   st["corr"], attempt, _CONNEX_PP_RECV_RETRIES)
                    comm.send_step(st["td"], st["corr"])
                    # Fast handover: every other in-flight step that was sent
                    # on the failed chain is lost too; re-send them all now
                    # (oldest first) instead of one recv timeout each.
                    stale = getattr(comm, "step_stale", None)
                    if stale is not None:
                        for o in sorted((b for i, b in S.busy.items()
                                         if i != slot and b), key=lambda b: b["corr"]):
                            if stale(o["corr"]):
                                logger.warning("Stage 0: re-sending in-flight step %s "
                                               "(chain failed over)", o["corr"])
                                comm.send_step(o["td"], o["corr"])
                ret = comm.recv_step(st["corr"])
                attempt += 1
                if ret is not None:
                    break
                if attempt >= max(1, _CONNEX_PP_RECV_RETRIES) and not (
                        healthy() and time.monotonic() - t_first < budget):
                    break
            if ret is None:
                comm.abandon_step(st["corr"])
                raise RuntimeError(
                    "Stage 0: no reply for pipelined step %s after %d attempts"
                    % (st["corr"], attempt))
            if "hidden_states" not in ret:
                raise RuntimeError("Stage 0: return tensor dict missing 'hidden_states' "
                                   f"(keys {list(ret.keys())})")
            final_hidden_states = ret["hidden_states"]
            if final_hidden_states.device != self.device:
                final_hidden_states = final_hidden_states.to(self.device)
            sample_hidden_states = final_hidden_states[st["li"]]
            logits = self.model.compute_logits(sample_hidden_states)

            cur_batch, cur_last = self.input_batch, getattr(self, "_cx_last_slot", None)
            zombies = S.zombies_of(slot)
            for rid, rs in zombies.items():
                self.requests.setdefault(rid, rs)
            self.input_batch = self._cx_batches[slot]
            self._cx_last_slot = None
            n = len(st["disc"])
            self.discard_request_indices.np[:n] = st["disc"]
            self.num_discarded_requests = n
            self.execute_model_state = ExecuteModelState(
                st["so"], logits, st["sdm"], st["scm"], final_hidden_states,
                sample_hidden_states, st["aux"], st["ec"])
            self.kv_connector_output = st["kv"]
            try:
                return self.sample_tokens(grammar_output)
            finally:
                self.input_batch = cur_batch
                self._cx_last_slot = cur_last
                self.execute_model_state = None
                for rid, rs in zombies.items():
                    if self.requests.get(rid) is rs:
                        del self.requests[rid]
        finally:
            S.release(slot)

'''

# ---------------------------------------------------------------- executor
E_MARK = "# connex: stage-0 pipelining executor (CONNEX_PP_INFLIGHT)"
E_MODULE_ANCHOR = "class UniProcExecutor(Executor):\n"
E_MODULE = E_MARK + '''
_CX_PP_INFLIGHT = max(1, int(os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))


class _CxLazyFuture(Future):
    """Future whose work runs in the thread that first calls result()."""

    def __init__(self, fn, single_value: bool = True):
        super().__init__()
        self._cx_fn = fn
        self._cx_single = single_value

    def result(self, timeout=None):
        fn, self._cx_fn = self._cx_fn, None
        if fn is not None:
            try:
                r = fn()
                self.set_result(r if self._cx_single else [r])
            except BaseException as e:  # noqa: BLE001
                self.set_exception(e)
        return super().result(timeout)


'''
E_MCB_OLD = '''    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1
'''
E_MCB_NEW = '''    def max_concurrent_batches(self) -> int:
        if (_CX_PP_INFLIGHT > 1 and getattr(
                self.vllm_config.parallel_config, "pipeline_stage_mode", None) == "external"):
            return _CX_PP_INFLIGHT
        return 2 if self.scheduler_config.async_scheduling else 1
'''
E_RPC_OLD = '''        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
'''
E_RPC_NEW = '''        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            _cx_run = getattr(result, "_cx_deferred_run", None)
            if _cx_run is not None:
                result = _cx_run()
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            _cx_run = getattr(result, "_cx_deferred_run", None)
            if _cx_run is not None:
                return _CxLazyFuture(_cx_run, single_value)
            if isinstance(result, AsyncModelRunnerOutput):
'''

# --------------------------------------------------------------- scheduler
S_MARK = "# connex: stage-0 pipelining microbatch cap (CONNEX_PP_INFLIGHT)"
S_MODULE_ANCHOR = "logger = init_logger(__name__)\n"
S_MODULE = S_MODULE_ANCHOR + S_MARK + '''
import os as _cx_os
_CX_PP_INFLIGHT = max(1, int(_cx_os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1"))
'''
S_LOOP_OLD = '''        # First, schedule the RUNNING requests.
        req_index = 0
'''
S_LOOP_NEW = S_LOOP_OLD + '''        _cx_cap = (-(-len(self.running) // _CX_PP_INFLIGHT)
                   if _CX_PP_INFLIGHT > 1 else len(self.running) + 1)
'''
S_CAP_OLD = '''            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
'''
S_CAP_NEW = '''            if len(scheduled_running_reqs) >= max(1, _cx_cap):
                break  # connex: leave the rest for the next microbatch
''' + S_CAP_OLD


def _apply(src: str, mark: str, edits) -> tuple:
    if mark in src:
        return src, "already patched"
    for old, new, count in edits:
        if src.count(old) != count:
            return None, f"anchor not found/unique ({src.count(old)}x): {old.strip()[:60]!r}"
    for old, new, _ in edits:
        src = src.replace(old, new)
    return src, "patched"


def patch_runner(src: str):
    return _apply(src, MARK, [
        (R_MODULE_ANCHOR, R_MODULE + R_MODULE_ANCHOR, 1),
        (R_CTOR_OLD, R_CTOR_NEW, 2),
        (R_EXEC_ANCHOR, R_EXEC, 1),
        (R_SEND_ANCHOR, R_SEND, 1),
        (R_SAMPLE_ANCHOR, R_SAMPLE, 1),
        (R_METHODS_ANCHOR, R_METHODS + R_METHODS_ANCHOR, 1),
    ])


def patch_executor(src: str):
    if "import os\n" not in src:
        return None, "executor: no 'import os'"
    return _apply(src, E_MARK, [
        (E_MODULE_ANCHOR, E_MODULE + E_MODULE_ANCHOR, 1),
        (E_MCB_OLD, E_MCB_NEW, 1),
        (E_RPC_OLD, E_RPC_NEW, 1),
    ])


def patch_scheduler(src: str):
    return _apply(src, S_MARK, [
        (S_MODULE_ANCHOR, S_MODULE, 1),
        (S_LOOP_OLD, S_LOOP_NEW, 1),
        (S_CAP_OLD, S_CAP_NEW, 1),
    ])


def siblings(runner_path: str) -> dict:
    v1 = os.path.dirname(os.path.dirname(os.path.abspath(runner_path)))
    return {
        "uniproc_executor.py": (os.path.join(v1, "executor", "uniproc_executor.py"), patch_executor),
        "scheduler.py": (os.path.join(v1, "core", "sched", "scheduler.py"), patch_scheduler),
    }


def _patch_file(path: str, fn) -> tuple:
    src = open(path).read()
    new, msg = fn(src)
    if new is None:
        return 1, msg
    if new is not src:
        open(path, "w").write(new)
    return 0, msg


def main(argv) -> int:
    runner = argv[1]
    rc, msg = _patch_file(runner, patch_runner)
    msgs = [f"runner {msg}"]
    for name, (path, fn) in siblings(runner).items():
        if not os.path.exists(path):
            msgs.append(f"{name} skipped (not found)")
            continue
        r, m = _patch_file(path, fn)
        rc |= r
        msgs.append(f"{name} {m}")
    status = "FAILED " if rc else ""
    print(f"{os.path.basename(__file__)}: {status}{'; '.join(msgs)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
