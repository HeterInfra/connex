# SPDX-License-Identifier: Apache-2.0
"""Patch 6 (pipelining, CONNEX_PP_INFLIGHT=K): downstream recv/compute overlap.

    python patch_external_pp_inflight.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

apply_all.py passes the runner path; this patch edits the sibling
gpu_worker.py in the same directory (a gpu_worker.py path is also accepted).
The runner itself is not modified. Idempotent; anchors are on the fork as
shipped (the earlier patches do not touch gpu_worker.py).

What it does
------------
Stages 1..N-1 run gpu_worker._external_pipeline_loop: recv -> execute ->
send, strictly serial. With CONNEX_PP_INFLIGHT=K > 1 (and a communicator that
offers the per-step API, i.e. ConnexTECommunicator) the loop is replaced by
_connex_pipelined_loop: a daemon prefetch thread calls comm.recv_forward()
(RDMA read of step n+1 into the registered GPU recv buffer + clone) into a
bounded queue (K entries) while the loop thread computes step n and sends it
with comm.send_reply(out, ctx). ctx carries the step's correlation id and
originator endpoint, so the reply is stamped with the right id even though
the prefetch thread has already received later steps. Compute stays in one
thread, so per-request KV state (ConnexPPBlockAllocator) and the runner are
never touched concurrently; requests of different in-flight steps are
disjoint, and the allocator only reclaims idle requests by TTL/LRU, so
disjoint consecutive batches do not evict each other.

K=1 (default): the original loop runs unchanged (the gate is the first
statement after the "entering tight pipeline loop" log line).

Limitations
-----------
* Gain needs stage 0 to keep K steps in flight. This patch alone does not
  change stage 0: with an unmodified stage 0 only one step is ever in
  flight, so the prefetch thread just waits (no regression, no gain beyond
  overlapping the RDMA read with the tail of the previous send).
* Both threads use the default CUDA stream, so the clone after the RDMA read
  is ordered with compute; with CONNEXTE_GDR_RECV_SLOTS=N>1 (views, no
  clone) N must be >= K+2, otherwise keep the default 0 (clone).
* Recv-side duplicate re-sends run in the prefetch thread; they share the
  DEALER sockets with send_reply under the communicator's _send_lock.

Stage-0 design note (implemented by patch_external_pp_stage0_inflight.py, see docs/design/pp_inflight.md)
--------------------------------------------------------------------------------------------------------
The engine already has the needed scheduler behaviour: EngineCore uses
step_with_batch_queue when model_executor.max_concurrent_batches > 1, and
the V1 scheduler skips a request whose num_new_tokens is 0 (decode request
whose sampled token is not back yet), so consecutive in-flight batches are
disjoint. External PP bypasses it because stage 0 is a single-rank vLLM
(UniProcExecutor.max_concurrent_batches = 1 without async scheduling) and
its execute_model blocks on recv_tensor_dict. Required stage-0 changes:
(1) UniProcExecutor.max_concurrent_batches -> K when CONNEX_PP_INFLIGHT>1;
(2) execute_model for external-PP stage 0: run local layers, new_step_id(),
send_step(), stash (corr, forward dict, scheduler_output, logits_indices
clone, sampling metadata) in slot n % K and return without waiting;
(3) sample_tokens returns an AsyncModelRunnerOutput whose get_output() runs
recv_step(corr) with per-step retry (send_step same corr on None),
compute_logits, sampling and bookkeeping for that slot - UniProcExecutor
already runs get_output on its async_output_thread;
(4) the blocker: sampling/bookkeeping use the runner's single persistent
InputBatch, which _update_states for step n+1 rewrites (unscheduled requests
are removed) before step n's sample runs. It needs K InputBatch instances
swapped per slot (requests moving between slots are re-added from
CachedRequestState, the existing path for un/re-scheduled requests), plus
spec decode / async scheduling / structured output disabled while K>1.
Patch 7 implements this, with the deferred work run lazily in the engine
thread instead of the async output thread (see its docstring).
"""
import os
import sys

MARK = "# connex: pipelined external-PP loop (CONNEX_PP_INFLIGHT)"

GATE_ANCHOR = '''        logger.info("Stage %s: entering tight pipeline loop", stage_idx)
'''
GATE = GATE_ANCHOR + f'''        {MARK}
        _cx_comm = getattr(pp_group, "zeromq_comm", None)
        if (getattr(_cx_comm, "inflight", 1) > 1
                and hasattr(_cx_comm, "recv_forward")
                and hasattr(_cx_comm, "send_reply")):
            return self._connex_pipelined_loop(_cx_comm, stage_idx)
'''

METHOD_ANCHOR = '''    def start_external_pipeline_loop(self) -> bool:
'''
METHOD = '''    @torch.inference_mode()
    def _connex_pipelined_loop(self, comm, stage_idx: int) -> None:
        """recv of step n+1 (prefetch thread) overlaps compute+send of step n.
        Installed by connex patch_external_pp_inflight.py; K=comm.inflight."""
        import queue as _queue
        import threading as _threading
        import time as _time

        depth = max(1, int(comm.inflight))
        q: "_queue.Queue" = _queue.Queue(maxsize=depth)
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None

        def _prefetch() -> None:
            if dev is not None:
                torch.cuda.set_device(dev)
            while True:
                try:
                    td, ctx = comm.recv_forward()
                    if td is None:
                        continue
                    q.put((td, ctx))
                except Exception:
                    logger.exception("Stage %s: error in pipeline prefetch", stage_idx)
                    _time.sleep(0.1)

        _threading.Thread(target=_prefetch, name="external_pp_prefetch",
                          daemon=True).start()
        logger.info("Stage %s: pipelined loop (inflight=%d)", stage_idx, depth)
        step_count = 0
        while True:
            try:
                tensor_dict, ctx = q.get()
                output = self.model_runner.execute_external_pipeline_step(tensor_dict)
                if output is None:
                    logger.warning("Stage %s: execute_external_pipeline_step "
                                   "returned None (step %d)", stage_idx, step_count)
                    continue
                comm.send_reply(output, ctx)
                step_count += 1
                if step_count % 500 == 0:
                    logger.info("Stage %s: pipeline loop completed %d steps",
                                stage_idx, step_count)
            except Exception:
                logger.exception("Stage %s: error in pipeline loop (step %d)",
                                 stage_idx, step_count)
                _time.sleep(0.1)

'''


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    for anchor in (GATE_ANCHOR, METHOD_ANCHOR):
        if src.count(anchor) != 1:
            return None, f"anchor not found/unique: {anchor.strip()[:60]!r}"
    src = src.replace(GATE_ANCHOR, GATE)
    src = src.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR)
    return src, "patched"


def target_path(path: str) -> str:
    if os.path.basename(path) == "gpu_worker.py":
        return path
    return os.path.join(os.path.dirname(os.path.abspath(path)), "gpu_worker.py")


def main(argv) -> int:
    path = target_path(argv[1])
    if not os.path.exists(path):
        # Runner-only copies (unit tests) have no worker next to them.
        print(f"{os.path.basename(__file__)}: skipped, {path} not found")
        return 0
    src = open(path).read()
    new, msg = patch(src)
    if new is None:
        print(f"{os.path.basename(__file__)}: FAILED {msg}")
        return 1
    if new is not src:
        open(path, "w").write(new)
    print(f"{os.path.basename(__file__)}: {msg} ({os.path.basename(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
