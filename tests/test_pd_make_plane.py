# SPDX-License-Identifier: Apache-2.0
"""make_plane() backend selection: an explicit "auto" auto-detects."""

import importlib.util

import pytest

from connex.pd import dataplane


@pytest.mark.parametrize("backend", [None, "", "auto", "AUTO", " auto "])
def test_auto_and_unset_select_the_same_backend(monkeypatch, backend):
    monkeypatch.delenv("CONNEX_PD_BACKEND", raising=False)
    seen = []
    monkeypatch.setattr(dataplane, "MooncakePlane", lambda ip: seen.append(ip) or "mc")
    monkeypatch.setattr(dataplane, "resolve_backend",
                        lambda name, _real=dataplane.resolve_backend:
                        "mooncake" if (name or "auto").strip().lower() == "auto" else _real(name))
    assert dataplane.make_plane("m", 0, 0, 0, "10.0.0.1", backend=backend) == "mc"
    assert seen == ["10.0.0.1"]


def test_explicit_auto_does_not_fall_back_to_loopback(monkeypatch):
    monkeypatch.delenv("CONNEX_PD_BACKEND", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name in ("connex_rs", "mooncake")
                        else real(name, *a, **k))
    # Nothing installed: auto must raise, not silently return a LoopbackPlane.
    with pytest.raises(RuntimeError, match="no KV data plane"):
        dataplane.make_plane("m", 0, 0, 0, "127.0.0.1", backend="auto")


def test_explicit_loopback_and_unknown(monkeypatch):
    assert isinstance(dataplane.make_plane("m", 0, 0, 0, "127.0.0.1", backend="loopback"),
                      dataplane.LoopbackPlane)
    with pytest.raises(ValueError):
        dataplane.make_plane("m", 0, 0, 0, "127.0.0.1", backend="bogus")
