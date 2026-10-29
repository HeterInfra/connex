# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Port layout of the one-stage-per-instance mode, in one place.

Every downstream stage replica listens on a control port (the stage server)
and one data port per TP rank (the pipeline endpoints of its workers). On a
host network any placement must work, so no two (stage, replica, TP rank)
may share a port:

    control(stage, replica)       = BASE + 1000*stage + 16*replica
    data(stage, replica, tp_rank) = control(stage, replica) + 1 + tp_rank

with ``replica < 62`` and ``tp_rank < 15`` (``ValueError`` otherwise), so
blocks of different replicas and stages never overlap. ``BASE`` is
``CONNEX_PP_PORT_BASE`` (47000).

Used by group.py (the data port of each TP rank), docker/launch_stage.sh
(``python -m connex.adapters.vllm.pp.ports control|data STAGE REPLICA``) and
deploy/k8s/manifests.py. Importing it needs neither torch nor vLLM.
"""

from __future__ import annotations

import os
import sys

STAGE_STRIDE = 1000
REPLICA_STRIDE = 16
MAX_REPLICAS = STAGE_STRIDE // REPLICA_STRIDE          # 62 blocks of 16 ports per stage
MAX_TP = REPLICA_STRIDE - 1                           # control + 15 data ports


def base() -> int:
    return int(os.getenv("CONNEX_PP_PORT_BASE", "47000"))


def control_port(stage: int, replica: int, port_base: int | None = None) -> int:
    b = base() if port_base is None else port_base
    if stage < 0 or not 0 <= replica < MAX_REPLICAS:
        raise ValueError(f"stage {stage} / replica {replica} out of range "
                         f"(replica < {MAX_REPLICAS})")
    p = b + STAGE_STRIDE * stage + REPLICA_STRIDE * replica
    if p + REPLICA_STRIDE > 65536:
        raise ValueError(f"stage {stage} replica {replica}: port {p} out of range "
                         "(lower CONNEX_PP_PORT_BASE)")
    return p


def data_port(stage: int, replica: int, tp_rank: int = 0, port_base: int | None = None) -> int:
    if not 0 <= tp_rank < MAX_TP:
        raise ValueError(f"TP rank {tp_rank} out of range (< {MAX_TP})")
    return control_port(stage, replica, port_base) + 1 + tp_rank


def rank_port(rank0_port: int, tp_rank: int) -> int:
    """Data port of TP rank ``tp_rank`` given TP rank 0's (0: ephemeral)."""
    if not rank0_port:
        return 0
    if not 0 <= tp_rank < MAX_TP:
        raise ValueError(f"TP rank {tp_rank} out of range (< {MAX_TP})")
    return rank0_port + tp_rank


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4) or argv[0] not in ("control", "data"):
        print("usage: python -m connex.adapters.vllm.pp.ports control|data STAGE REPLICA [TP_RANK]",
              file=sys.stderr)
        return 2
    stage, replica = int(argv[1]), int(argv[2])
    if argv[0] == "control":
        print(control_port(stage, replica))
    else:
        print(data_port(stage, replica, int(argv[3]) if len(argv) > 3 else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
