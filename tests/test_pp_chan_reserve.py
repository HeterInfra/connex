# SPDX-License-Identifier: Apache-2.0
"""PP channel rings are held for at init, before vLLM sizes the KV cache (#39)."""

import pytest

link = pytest.importorskip("connex.adapters.vllm.pp.link")


def test_reserve_size_defaults_and_env(monkeypatch):
    monkeypatch.delenv("CONNEX_PP_CHAN_RESERVE_MB", raising=False)
    assert link.chan_reserve_bytes(4, 1 << 20) == 8 << 20          # middle: send + receive
    assert link.chan_reserve_bytes(4, 1 << 20, rings=1) == 4 << 20  # first / last stage
    monkeypatch.setenv("CONNEX_PP_CHAN_RESERVE_MB", "0")
    assert link.chan_reserve_bytes(4, 1 << 20) == 0
    monkeypatch.setenv("CONNEX_PP_CHAN_RESERVE_MB", "1.5")
    assert link.chan_reserve_bytes(4, 1 << 20) == 3 << 19


def test_first_ring_frees_the_placeholder():
    sl = link.StageLink.__new__(link.StageLink)
    sl._chan_reserve = object()
    sl._free_chan_reserve()
    assert sl._chan_reserve is None
    sl._free_chan_reserve()          # idempotent
    bare = link.StageLink.__new__(link.StageLink)
    bare._free_chan_reserve()        # built without __init__: no attribute, no error


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_placeholder_memory_is_reused_by_the_ring():
    import torch

    dev = torch.device("cuda", 0)
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved(dev)
    sl = link.StageLink.__new__(link.StageLink)
    sl._chan_reserve = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    held = torch.cuda.memory_reserved(dev)
    sl._free_chan_reserve()
    ring = torch.empty(32 << 20, dtype=torch.uint8, device=dev)  # a channel ring
    assert torch.cuda.memory_reserved(dev) == held   # no new device memory
    del ring
    torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved(dev) <= before + (2 << 20)
