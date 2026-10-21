# SPDX-License-Identifier: Apache-2.0
"""SGLang adapter (prototype; communication side only).

:class:`ConnexPPGroup` replaces ``scheduler.pp_group`` with a Connex-backed,
GroupCoordinator-compatible object for external elastic PP. It is unit
tested with fake transports and with real Connex endpoints on localhost, not
with a running SGLang engine: the engine-side changes it relies on (one
scheduler on stage 0, passive executors on later stages, per-request KV
accounting, re-prefill) do not exist yet. See docs/design/sglang-adapter.md.

Importing this package does not import SGLang.
"""

from connex.adapters.sglang.pp_group import (  # noqa: F401
    MISSING_HERE, MSG_TYPE, REPREFILLED, ConnexPPGroup,
)
from connex.adapters.sglang.setup import make_pp_group  # noqa: F401

__all__ = ["ConnexPPGroup", "make_pp_group", "MSG_TYPE", "MISSING_HERE", "REPREFILLED"]
