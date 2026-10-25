# SPDX-License-Identifier: Apache-2.0
"""Expert parallelism (MoE) with recovery and elasticity.

EP ranks are mobile endpoints; Connex owns the control plane of an EP group:

* ``PlacementTable``: versioned expert placement (expert -> physical slots on
  ranks, replicas, rank states up / draining / down, routable masks), with
  repair planning after a rank fails or drains;
* ``ExpertStore``: one rank's expert weights, registered with a one-sided
  transport (connex-rs) so a peer can pull an expert into its own slot;
* ``EPController``: applies every change through two epochs (route around,
  copy, route to the new copies), re-plans moves whose source dies.

The per-layer dispatch / combine all-to-all stays in the engine; Connex only
hands it a new placement at a step boundary. vLLM glue:
``connex.ep.vllm_ext`` (worker extension + EPLB-capable OLMoE).
"""
from .placement import (DOWN, DRAINING, UP, Move, PlacementError, PlacementTable,
                        logical_to_physical)
from .mover import ExpertStore, moves_by_rank
from .controller import EPController, LocalEngine

__all__ = ["PlacementTable", "Move", "PlacementError", "logical_to_physical",
           "UP", "DRAINING", "DOWN", "ExpertStore", "moves_by_rank",
           "EPController", "LocalEngine"]
