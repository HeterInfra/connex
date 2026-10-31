# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Per-step timestamps for latency analysis (CONNEX_PP_TRACE=1).

Each process appends "<tag> <seq> <monotonic ns> <thread>" lines to
``$CONNEX_PP_TRACE_DIR/cxtrace-<pid>.txt`` (default /dev/shm). CLOCK_MONOTONIC
is one clock for every process on a host, so the files of the stages of one
host merge into one timeline (benchmarks/pp_upstream/trace_steps.py)."""
from __future__ import annotations

import os
import threading
import time

ON = os.getenv("CONNEX_PP_TRACE", "0") == "1"
_f = None
_lock = threading.Lock()


def tr(tag: str, seq: int) -> None:
    if not ON:
        return
    global _f
    t = time.monotonic_ns()
    with _lock:
        if _f is None:
            d = os.getenv("CONNEX_PP_TRACE_DIR", "/dev/shm")
            _f = open(os.path.join(d, f"cxtrace-{os.getpid()}.txt"), "a", buffering=1 << 16)
        _f.write(f"{tag} {seq} {t} {threading.current_thread().name}\n")
        if tag.endswith("!"):
            _f.flush()
