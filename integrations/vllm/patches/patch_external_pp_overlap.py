# SPDX-License-Identifier: Apache-2.0
"""Patch 9 (pipelining, CONNEX_PP_INFLIGHT=K>1): take the stage GPU off the
CPU critical path.

    python patch_external_pp_overlap.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Needs patches 6 and 7 (anchors are on their output). Edits the runner given
and, when present, the sibling gpu_worker.py. Idempotent (one marker per
file). Everything is behind K>1 and CONNEX_PP_OVERLAP (default 0 = patch-7
behaviour; measured slower at K=2, see docs/design/pp_inflight.md), so K=1 is untouched.

Why (profiled 2026-10-17, 2 x A40, Llama-3-8B, 32 clients, K=2): stage 0's
engine thread spent 78% of its time blocked in
`logits_indices.cpu().tolist()` right after launching the stage-0 forward,
i.e. waiting for its own GPU, and then did send, recv of the other
microbatch, logits, sampling, bookkeeping, scheduling and input prep while
the GPU sat idle (GPU0 85% busy, the pipeline bottleneck). Stage 1 likewise
blocked in send_reply (event sync) before it could launch the next step.

Changes
  * stage 0: logits_indices for the wire come from the CPU copy of
    query_start_loc (same values as query_start_loc[1:] - 1 on the GPU, the
    non-spec-decode case), so building the forward dict does not sync.
  * stage 0: _cx_send_step uses comm.send_step_async (copies enqueued now,
    a sender thread waits and signals); _cx_finish_step waits for that send
    before recv_step and re-sends synchronously on retry, as before.
  * stages 1+: the pipelined loop uses comm.send_reply_async and checks the
    previous send's future after launching the next step.
  * CONNEX_PP_PROFILE=1: stage 0 logs every 256 steps the mean ms of
    prep (begin_step -> forward enqueued), send-enqueue, the wait for the
    send/reply, recv->sampled and the microbatch size histogram.
"""
import os
import sys

MARK = "# connex: pipelining overlap (CONNEX_PP_OVERLAP)"

R_MODULE_ANCHOR = "class _CxDeferredOutput:\n"
R_MODULE = MARK + '''
_CX_PP_OVERLAP = os.environ.get("CONNEX_PP_OVERLAP", "0") == "1"
_CX_PP_PROFILE = os.environ.get("CONNEX_PP_PROFILE", "0") == "1"


class _CxStepProfile:
    """CONNEX_PP_PROFILE=1: mean per-phase ms of stage-0 pipelined steps."""

    EVERY = 256

    def __init__(self):
        import collections
        self.sums = collections.defaultdict(float)
        self.sizes = collections.Counter()
        self.n = 0

    def add(self, phase, dt):
        self.sums[phase] += dt

    def step(self, nreq):
        self.n += 1
        self.sizes[nreq] += 1
        if self.n % self.EVERY == 0:
            parts = " ".join("%s=%.2f" % (k, 1000 * v / self.EVERY)
                             for k, v in sorted(self.sums.items()))
            logger.info("connex stage-0 step profile (ms/step over %d): %s | "
                        "microbatch sizes %s", self.EVERY, parts,
                        dict(sorted(self.sizes.items())))
            self.sums.clear()
            self.sizes.clear()


'''

R_LI_OLD = '''                if isinstance(logits_indices, torch.Tensor):
                    forward_tensor_dict["logits_indices"] = logits_indices.cpu().tolist()
'''
R_LI_NEW = '''                if (_CX_PP_INFLIGHT > 1 and _CX_PP_OVERLAP and spec_decode_metadata is None
                        and isinstance(logits_indices, torch.Tensor) and self._cx_k() > 1):
                    # connex: same values as query_start_loc[1:] - 1 on the GPU,
                    # taken from the CPU copy so this does not wait for the forward
                    forward_tensor_dict["logits_indices"] = (
                        self.query_start_loc.np[1:num_reqs + 1] - 1).tolist()
                elif isinstance(logits_indices, torch.Tensor):
                    forward_tensor_dict["logits_indices"] = logits_indices.cpu().tolist()
'''

R_BEGIN_OLD = '''    def _cx_begin_step(self, scheduler_output) -> None:
        S = getattr(self, "_cx_slots", None)
'''
R_BEGIN_NEW = '''    def _cx_begin_step(self, scheduler_output) -> None:
        if _CX_PP_PROFILE:
            self._cx_t_begin = time.perf_counter()
        S = getattr(self, "_cx_slots", None)
'''

R_SEND_OLD = '''        corr = comm.new_step_id()
        try:
            comm.send_step(td, corr)
        except Exception:
            comm.abandon_step(corr)
            raise
        n = self.num_discarded_requests
'''
R_SEND_NEW = '''        corr = comm.new_step_id()
        _cx_t0 = time.perf_counter()
        sf = None
        try:
            if _CX_PP_OVERLAP and hasattr(comm, "send_step_async"):
                sf = comm.send_step_async(td, corr)
            else:
                comm.send_step(td, corr)
        except Exception:
            comm.abandon_step(corr)
            raise
        _cx_evt = None
        if _CX_PP_OVERLAP:
            _cx_evt = torch.cuda.Event()  # this step's forward + send copies
            _cx_evt.record()
        if _CX_PP_PROFILE:
            P = getattr(self, "_cx_prof", None) or _CxStepProfile()
            self._cx_prof = P
            P.add("a_prep", _cx_t0 - getattr(self, "_cx_t_begin", _cx_t0))
            P.add("b_send_enqueue", time.perf_counter() - _cx_t0)
        n = self.num_discarded_requests
'''
R_STASH_OLD = '''            "req_ids": set(r for r in self.input_batch.req_ids if r is not None),
        }
'''
R_STASH_NEW = '''            "req_ids": set(r for r in self.input_batch.req_ids if r is not None),
            "sf": sf, "evt": _cx_evt,
        }
'''
R_WRAP_OLD = '''    def _cx_finish_step(self, slot: int, grammar_output):
        S = self._cx_slots
'''
R_WRAP_NEW = '''    def _cx_finish_step(self, slot: int, grammar_output):
        """connex overlap: run recv + logits + sampling of an outstanding
        step on a side stream, so they do not queue behind the forward of
        the step just launched on the main stream (which they do not depend
        on). The side stream first waits for this step's own forward/send
        event; the main stream waits for the side stream afterwards, so the
        next step's input prep is ordered after this step's sampling."""
        st = self._cx_slots.busy.get(slot)
        if not _CX_PP_OVERLAP or st is None or st.get("evt") is None:
            return self._cx_finish_step_body(slot, grammar_output)
        side = getattr(self, "_cx_side_stream", None)
        if side is None:
            side = self._cx_side_stream = torch.cuda.Stream(priority=-1)
        main = torch.cuda.current_stream()
        side.wait_event(st["evt"])
        if isinstance(st["li"], torch.Tensor):
            st["li"].record_stream(side)  # allocated on main, read on side
        try:
            with torch.cuda.stream(side):
                return self._cx_finish_step_body(slot, grammar_output)
        finally:
            main.wait_stream(side)

    def _cx_finish_step_body(self, slot: int, grammar_output):
        S = self._cx_slots
'''

R_FINISH_OLD = '''        try:
            ret = None
'''
R_FINISH_NEW = '''        try:
            _cx_t0 = time.perf_counter()
            if st.get("sf") is not None:
                try:
                    st["sf"].result()  # async send of this step done (or failed)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Stage 0: async send of step %s failed (%s); "
                                   "re-sending", st["corr"], e)
                    comm.send_step(st["td"], st["corr"])
            _cx_t1 = time.perf_counter()
            ret = None
'''
R_RECV_OLD = '''            if "hidden_states" not in ret:
                raise RuntimeError("Stage 0: return tensor dict missing 'hidden_states' "
'''
R_RECV_NEW = '''            _cx_t2 = time.perf_counter()
''' + R_RECV_OLD
R_SAMPLE_OLD = '''            try:
                return self.sample_tokens(grammar_output)
            finally:
                self.input_batch = cur_batch
'''
R_SAMPLE_NEW = '''            try:
                _cx_out = self.sample_tokens(grammar_output)
                if _CX_PP_PROFILE and getattr(self, "_cx_prof", None) is not None:
                    _cx_t3 = time.perf_counter()
                    self._cx_prof.add("c_send_wait", _cx_t1 - _cx_t0)
                    self._cx_prof.add("d_recv", _cx_t2 - _cx_t1)
                    self._cx_prof.add("e_logits_sample", _cx_t3 - _cx_t2)
                    self._cx_prof.step(len(st["req_ids"]))
                return _cx_out
            finally:
                self.input_batch = cur_batch
'''

W_MARK = "# connex: async send_reply in the pipelined loop (CONNEX_PP_OVERLAP)"
W_OLD = '''        step_count = 0
        while True:
            try:
                tensor_dict, ctx = q.get()
                output = self.model_runner.execute_external_pipeline_step(tensor_dict)
                if output is None:
                    logger.warning("Stage %s: execute_external_pipeline_step "
                                   "returned None (step %d)", stage_idx, step_count)
                    continue
                comm.send_reply(output, ctx)
'''
W_NEW = '''        step_count = 0
        ''' + W_MARK + '''
        _async = (os.environ.get("CONNEX_PP_OVERLAP", "0") == "1"
                  and hasattr(comm, "send_reply_async"))
        prev_send = None
        while True:
            try:
                tensor_dict, ctx = q.get()
                output = self.model_runner.execute_external_pipeline_step(tensor_dict)
                if output is None:
                    logger.warning("Stage %s: execute_external_pipeline_step "
                                   "returned None (step %d)", stage_idx, step_count)
                    continue
                if _async:
                    # copies enqueued behind this step's kernels; the sender
                    # thread signals once they are done, we launch the next step
                    f = comm.send_reply_async(output, ctx)
                    if prev_send is not None:
                        try:
                            prev_send.result()
                        except Exception:
                            logger.exception("Stage %s: async send_reply failed "
                                             "(stage 0 will retry)", stage_idx)
                    prev_send = f
                else:
                    comm.send_reply(output, ctx)
'''


def _apply(src, mark, edits):
    if mark in src:
        return src, "already patched"
    for old, _new in edits:
        if src.count(old) != 1:
            return None, f"anchor not found/unique ({src.count(old)}x): {old.strip()[:60]!r}"
    for old, new in edits:
        src = src.replace(old, new)
    return src, "patched"


def patch_runner(src):
    return _apply(src, MARK, [
        (R_MODULE_ANCHOR, R_MODULE + R_MODULE_ANCHOR),
        (R_LI_OLD, R_LI_NEW),
        (R_BEGIN_OLD, R_BEGIN_NEW),
        (R_SEND_OLD, R_SEND_NEW),
        (R_STASH_OLD, R_STASH_NEW),
        (R_FINISH_OLD, R_FINISH_NEW),
        (R_WRAP_OLD, R_WRAP_NEW),
        (R_RECV_OLD, R_RECV_NEW),
        (R_SAMPLE_OLD, R_SAMPLE_NEW),
    ])


def patch_worker(src):
    return _apply(src, W_MARK, [(W_OLD, W_NEW)])


def _patch_file(path, fn):
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
    worker = os.path.join(os.path.dirname(os.path.abspath(runner)), "gpu_worker.py")
    if os.path.exists(worker):
        r, m = _patch_file(worker, patch_worker)
        rc |= r
        msgs.append(f"gpu_worker.py {m}")
    else:
        msgs.append("gpu_worker.py skipped (not found)")
    print(f"{os.path.basename(__file__)}: {'FAILED ' if rc else ''}{'; '.join(msgs)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
