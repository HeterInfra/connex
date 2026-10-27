# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``--worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker``

The stock GPU worker plus two things the external pipeline needs:

* before each step, hand the step's route (attached to the SchedulerOutput
  by the executor) to the PP group, so it knows which replica of the next
  stage to send to and which epoch/step it is in;
* report the data-plane URL of this worker's pipeline endpoint.
"""

from __future__ import annotations

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu_worker import Worker

ROUTE_ATTR = "connex_pp_route"


class ConnexPPWorker(Worker):
    def execute_model(self, scheduler_output):
        group = get_pp_group()
        begin = getattr(group, "begin_step", None)
        if begin is not None:
            begin(getattr(scheduler_output, ROUTE_ATTR, None))
        return super().execute_model(scheduler_output)

    def compile_or_warm_up_model(self, *args, **kwargs):
        group = get_pp_group()
        if hasattr(group, "blocking"):
            group.blocking = True
        try:
            return super().compile_or_warm_up_model(*args, **kwargs)
        finally:
            if hasattr(group, "blocking"):
                group.blocking = False

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
