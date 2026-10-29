# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``--worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker``

The stock GPU worker plus two things the external pipeline needs:

* before each step, hand the step's route (attached to the SchedulerOutput
  by the executor) to the PP group, so it knows which replica of the next
  stage to send to and which epoch/step it is in;
* report the data-plane URL of this worker's pipeline endpoint;
* on the last stage, compile the top-k/top-p sampling kernels at start
  (``CONNEX_PP_WARM_TOPP``, default on). vLLM compiles them on the first
  request that uses top-p; on a replica that just took over that is a
  Triton load in its first steps, on the failover's critical path.
"""

from __future__ import annotations

import logging
import os

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu_worker import Worker

log = logging.getLogger("vllm.connex_pp")

ROUTE_ATTR = "connex_pp_route"


class ConnexPPWorker(Worker):
    def execute_model(self, scheduler_output):
        group = get_pp_group()
        begin = getattr(group, "begin_step", None)
        if begin is not None:
            begin(getattr(scheduler_output, ROUTE_ATTR, None))
            group.before_step()
        return super().execute_model(scheduler_output)

    def compile_or_warm_up_model(self, *args, **kwargs):
        group = get_pp_group()
        if os.getenv("CONNEX_PP_WARM_TOPP", "1") == "1" and group.is_last_rank:
            self._register_topp_warmup()
        if hasattr(group, "blocking"):
            group.blocking = True
        try:
            return super().compile_or_warm_up_model(*args, **kwargs)
        finally:
            if hasattr(group, "blocking"):
                group.blocking = False

    def _register_topp_warmup(self) -> None:
        reg = getattr(getattr(self, "model_runner", None), "jit_warmup_registry", None)
        kc = getattr(self.vllm_config, "kernel_config", None)
        if reg is None or not getattr(kc, "enable_jit_warmup", False):
            return
        try:
            from vllm.v1.sample.ops.topk_topp_sampler import register_top_k_top_p_warmups

            with reg.activate():
                register_top_k_top_p_warmups()
        except Exception as e:  # noqa: BLE001  (an older vLLM: compiled on first use)
            log.warning("connex-pp: top-k/top-p kernel warm-up not registered: %s", e)

    def connex_pp_endpoint(self) -> str | None:
        link = getattr(get_pp_group(), "link", None)
        return None if link is None else link.url

    def connex_pp_stats(self) -> dict:
        group = get_pp_group()
        out = dict(getattr(group, "timing", {}))
        link = getattr(group, "link", None)
        if link is not None:
            out.update(link.stats)
        return out
