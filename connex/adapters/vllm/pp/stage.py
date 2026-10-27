# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Stage server: one downstream pipeline stage (``--pipeline-stage-rank k``,
k >= 1) as its own process, driven by a stage-0 ConnexPPExecutor.

    python -m connex.adapters.vllm.pp.stage --connex-control-port 47001 \\
        --connex-replica 0 <the same vLLM engine arguments as stage 0> \\
        --pipeline-stage-rank 1

It builds a stock vLLM EngineCore for its stage (model load, KV cache, CUDA
graphs, all unchanged; its scheduler is never used) and then executes the
steps stage 0 sends: ``execute_model(SchedulerOutput)`` and
``sample_tokens(GrammarOutput)``, in order. The last stage sends each
step's ModelRunnerOutput back to stage 0. All stages must use the same
``--num-gpu-blocks-override`` (the block ids in the steps are stage 0's).

A replica serves one pipeline (stage-0 instance) at a time. When its owner
goes away, the first other pipeline that sends it a step takes it over
(after it dropped the old owner's requests).
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import threading

log = logging.getLogger("vllm.connex_pp")


class StageServer:
    def __init__(self, core, vllm_config, control_port: int, replica: int):
        from connex.adapters.vllm.pp.executor import CTL_SHARD, ControlLink

        self.core = core
        self.ex = core.model_executor
        self.stage = vllm_config.parallel_config.pipeline_stage_rank
        self.num_stages = vllm_config.parallel_config.pipeline_parallel_size
        self.last = self.stage == self.num_stages - 1
        self.replica = replica
        self.data_url = self.ex.local_data_url
        self.owner: str | None = None
        self.owner_peer = None
        self.live_reqs: set[str] = set()
        self.q: queue.Queue = queue.Queue()
        self.out_q: queue.Queue = queue.Queue()
        self.ctl = ControlLink(self.stage, CTL_SHARD, replica, port=control_port,
                               on_message=self._on_message, on_failed=self._on_failed)
        log.info("connex-pp: stage %d replica %d: control %s, data %s", self.stage, replica,
                 self.ctl.url, self.data_url)

    def _on_message(self, peer, msg) -> None:
        kind = msg[0]
        if kind == "HELLO":
            self.ctl.send(peer, ("HELLO_OK", self.data_url, self.owner))
        elif kind == "PING":
            self.ctl.send(peer, ("PONG",))
        elif kind in ("EXEC", "SAMPLE"):
            self.q.put((peer, msg))

    def _on_failed(self, peer, reason) -> None:
        if self.owner_peer is not None and tuple(peer) == tuple(self.owner_peer):
            log.warning("connex-pp: owner %s of stage %d replica %d failed", self.owner,
                        self.stage, self.replica)
            self.q.put((peer, ("OWNER_GONE",)))

    def _drop_requests(self) -> None:
        """Forget the requests of a previous owner (their KV belongs to it)."""
        if not self.live_reqs:
            return
        from vllm.v1.core.sched.output import SchedulerOutput

        so = SchedulerOutput.make_empty()
        so.finished_req_ids = set(self.live_reqs)
        self.out_q.put(("E", 0, 0, self.ex.execute_model(so, non_block=True), None))
        self.live_reqs.clear()

    def serve_forever(self) -> None:
        threading.Thread(target=self._reply_loop, name="cx-stage-reply", daemon=True).start()
        while True:
            peer, msg = self.q.get()
            kind = msg[0]
            if kind == "OWNER_GONE":
                self.owner, self.owner_peer = None, None
                continue
            _, seq, epoch, payload, owner = msg
            if self.owner != owner:
                if self.owner is not None:
                    self.ctl.send(peer, ("BUSY", seq))
                    continue
                self._drop_requests()
                self.owner, self.owner_peer = owner, peer
                log.info("connex-pp: stage %d replica %d now serves %s", self.stage,
                         self.replica, owner)
            if kind == "EXEC":
                so = payload
                for nr in so.scheduled_new_reqs:
                    self.live_reqs.add(nr.req_id)
                self.live_reqs.difference_update(so.finished_req_ids)
                fut = self.ex.execute_model(so, non_block=True)
                self.out_q.put(("E", seq, epoch, fut, None))
            else:
                fut = self.ex.sample_tokens(payload, non_block=True)
                self.out_q.put(("S", seq, epoch, fut, peer))

    def _reply_loop(self) -> None:
        while True:
            kind, seq, epoch, fut, peer = self.out_q.get()
            try:
                out = fut.result()
            except Exception:  # noqa: BLE001
                log.exception("connex-pp: stage %d step %d failed; exiting", self.stage, seq)
                os._exit(1)
            if kind == "S" and self.last:
                self.ctl.send(peer, ("OUT", seq, epoch, out))


def main(argv: list[str] | None = None) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor import Executor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = FlexibleArgumentParser(description="Connex pipeline stage server")
    parser.add_argument("--connex-control-port", type=int, required=True)
    parser.add_argument("--connex-replica", type=int, default=0)
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args(argv)
    os.environ.setdefault("CONNEX_PP_REPLICA", str(args.connex_replica))
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.pipeline_stage_rank in (None, 0):
        sys.exit("the stage server runs stages 1..N-1 (--pipeline-stage-rank k >= 1)")
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    core = EngineCore(vllm_config, Executor.get_class(vllm_config), log_stats=False,
                      executor_fail_callback=lambda: os._exit(1))
    server = StageServer(core, vllm_config, args.connex_control_port, args.connex_replica)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    print(f"CONNEX_PP_STAGE_READY stage={server.stage} replica={args.connex_replica} "
          f"control={server.ctl.url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
