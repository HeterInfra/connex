# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Tensor-parallel pipeline stages as one mobile endpoint.

A stage that runs with tensor parallelism N is N worker processes. TP rank 0
owns the stage's pipeline endpoint (``connex.pp.endpoint``) and registers
``LID(model, stage, shard=0, replica)`` with ``group_size=N``; every other
rank registers ``LID(model, stage, shard=rank, replica)`` with the same
group size through a ``TPShardMember`` and only heartbeats. The front-router
treats the N shards as one endpoint:

* any shard missing its lease (its process died or hung), reported dead, or
  leaving fails the whole group: every shard is removed and announced as
  deregistered, and the stage's peers fail its in-flight steps over to the
  same stage of another replica chain at once (the same path as a stage that
  deregisters);
* peers only ever see shard 0 of a complete group
  (``connex.epc_client.serving_routes``), so a stage with a dead rank takes
  no new steps even while its rank 0 is still alive;
* the group rejoins under the same identity once all of its ranks have
  registered again (a restarted stage).

All-reduce inside the group stays on NCCL/NVLink; only rank 0 talks to the
other stages (see integrations/vllm/patches/patch_external_pp_tp.py).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("connex.pp")


class TPShardMember:
    """Liveness registration of one non-zero TP rank of a pipeline stage."""

    def __init__(self, stage_idx: int, rank: int, tp_size: int,
                 replica_idx: Optional[int] = None, model: Optional[str] = None,
                 front_router_api: Optional[str] = None, events: Optional[str] = None,
                 heartbeat_interval_s: Optional[float] = None):
        from connex.epc_client import EPCV2Client, EPCV2ClientConfig

        if replica_idx is None:
            replica_idx = int(os.getenv("PIPELINE_REPLICA_IDX", "0") or 0)
        model = model or os.getenv("SERVED_MODEL_NAME", "default")
        # same resolution as the stage's endpoint (connex.pp.endpoint)
        api = os.getenv("FRONT_ROUTER_API") or os.getenv("CONNEXTE_FRONT_ROUTER_API") or \
            front_router_api or "127.0.0.1:15555"
        if not api.startswith("tcp://"):
            api = "tcp://" + api
        if events is None:
            events = os.getenv("FRONT_ROUTER_EVENTS") or os.getenv(
                "CONNEXTE_FRONT_ROUTER_EVENTS") or api.rsplit(":", 1)[0] + ":15560"
        kw = {}
        if heartbeat_interval_s is not None:
            kw["heartbeat_interval_s"] = heartbeat_interval_s
        self.cfg = EPCV2ClientConfig(
            front_router_api=api, front_router_events=events, model_name=model,
            stage_idx=stage_idx, shard_idx=rank, replica_idx=replica_idx,
            group_size=tp_size, local_addresses=[f"tp-shard:{rank}/{tp_size}@{os.getpid()}"],
            ucx_enabled=False, **kw)
        self.client = EPCV2Client(self.cfg)

    def start(self) -> bool:
        ok = self.client.start()
        log.info("TP shard %d/%d of stage %d replica %d: %s", self.cfg.shard_idx,
                 self.cfg.group_size, self.cfg.stage_idx, self.cfg.replica_idx,
                 "registered" if ok else "registration failed")
        return ok

    def stop(self) -> None:
        self.client.stop()
