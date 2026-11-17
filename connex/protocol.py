# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""Logical endpoint identity shared by the PD control plane and router."""

from dataclasses import dataclass
from typing import Tuple


class PDRole:
    """PD (Prefill-Decode) separation roles."""
    NONE = "none"           # Not in PD separation mode
    PREFILL = "prefill"     # Prefill instance (KV producer)
    DECODE = "decode"       # Decode instance (KV consumer)


@dataclass
class LogicalEndpointID:
    """Logical Endpoint Identifier (LID).
    
    A stable identity for stage replicas that survives migration.
    Physical addresses may change, but LID remains constant.
    
    Format: (model_name, stage_idx, shard_idx, replica_idx, pd_role)
    
    For PD separation:
    - pd_role="prefill": KV producer, sends KV cache after prefill
    - pd_role="decode": KV consumer, receives KV cache for decoding
    - pd_role="none": Not in PD mode (pipeline parallelism)
    """
    model_name: str
    stage_idx: int
    shard_idx: int = 0
    replica_idx: int = 0
    pd_role: str = PDRole.NONE  # NEW: "none", "prefill", or "decode"
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, LogicalEndpointID):
            return self.to_tuple() == other.to_tuple()
        if isinstance(other, tuple):
            # Support both 4-tuple and 5-tuple comparison
            my_tuple = self.to_tuple()
            if len(other) == 4:
                return my_tuple[:4] == other
            return my_tuple == other
        return False
    
    def to_tuple(self) -> Tuple[str, int, int, int, str]:
        """Convert to tuple representation (5-tuple with pd_role)."""
        return (self.model_name, self.stage_idx, self.shard_idx, self.replica_idx, self.pd_role)
    
    @property
    def is_decode(self) -> bool:
        """Check if this is a decode instance."""
        return self.pd_role == PDRole.DECODE
    
    @property
    def is_pd_mode(self) -> bool:
        """Check if this endpoint is in PD separation mode."""
        return self.pd_role != PDRole.NONE
    
    def __str__(self) -> str:
        if self.pd_role == PDRole.NONE:
            return f"LID({self.model_name}:s{self.stage_idx}:sh{self.shard_idx}:r{self.replica_idx})"
        return f"LID({self.model_name}:s{self.stage_idx}:sh{self.shard_idx}:r{self.replica_idx}:{self.pd_role})"
    
    def __repr__(self) -> str:
        return self.__str__()
