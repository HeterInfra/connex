# SPDX-License-Identifier: Apache-2.0
"""Build a ConnexPPGroup for an SGLang stage process from the environment.

    PIPELINE_STAGE_IDX, PIPELINE_NUM_STAGES, PIPELINE_REPLICA_IDX
    CONNEX_ROUTER_PORT (default 5550 + stage), FRONT_ROUTER_API (Front Router)
    CONNEX_PP_INFLIGHT, MOONCAKE_* (see connex.pp.transports)

Typical engine-side wiring (not done by this module; see the design doc)::

    from connex.adapters.sglang import make_pp_group
    scheduler.pp_group = make_pp_group(hooks=MySGLangHooks(scheduler),
                                       row_info=lambda: scheduler.cx_rows())
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from connex.adapters.sglang.pp_group import ConnexPPGroup


def make_pp_group(hooks: Any = None, row_info: Optional[Callable[[], tuple]] = None,
                  *, stage_idx: Optional[int] = None, num_stages: Optional[int] = None,
                  device: str = "cuda", **endpoint_kw) -> ConnexPPGroup:
    from connex.pp.endpoint import ConnexTEEndpoint

    stage = int(os.getenv("PIPELINE_STAGE_IDX", "0")) if stage_idx is None else stage_idx
    n = int(os.getenv("PIPELINE_NUM_STAGES", "1")) if num_stages is None else num_stages
    port = int(os.getenv("CONNEX_ROUTER_PORT", str(5550 + stage)))
    front = os.getenv("FRONT_ROUTER_API", "tcp://127.0.0.1:15555").replace("tcp://", "")
    ep = ConnexTEEndpoint(stage, n, port, front, device=device, **endpoint_kw)
    return ConnexPPGroup(ep, hooks, row_info=row_info)
