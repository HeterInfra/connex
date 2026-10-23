# SPDX-License-Identifier: Apache-2.0
"""Data plane selection when connex-rs is absent.

connex-rs is a separate package (pip install "connex-ai[rs]"); Connex must
import and pick a working backend without it. Runs with or without connex_rs
installed: its absence is simulated.
"""

import importlib.util
import logging
import subprocess
import sys

from connex.pp import transports


def _without_connex_rs(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "connex_rs" else real(name, *a, **k))
    monkeypatch.setattr(transports, "connexrs_module", lambda: None)


def test_auto_falls_back_without_connex_rs(monkeypatch):
    _without_connex_rs(monkeypatch)
    monkeypatch.delenv("CONNEX_PP_BACKEND", raising=False)
    monkeypatch.setattr(transports, "rdma_present", lambda: True)  # even with RDMA
    monkeypatch.setattr(transports, "MOONCAKE_AVAILABLE", True)
    assert transports.backend_from_env() == "mooncake"
    monkeypatch.setattr(transports, "MOONCAKE_AVAILABLE", False)
    assert transports.backend_from_env() == "zmq"


def test_forced_connex_rs_without_package_yields_no_engine(monkeypatch):
    _without_connex_rs(monkeypatch)
    engine, url = transports.create_connexrs_engine(
        1, 0, "m", "127.0.0.1:7000", None, logging.getLogger("t"))
    assert (engine, url) == (None, None)


def test_import_is_lazy():
    # importing connex.pp must not load the native module (it loads UCX)
    out = subprocess.run([sys.executable, "-c",
                          "import sys, connex.pp; print('connex_rs' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[-1] == "False"
