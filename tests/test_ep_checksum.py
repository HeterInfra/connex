"""EP checksum (issue #13): one device-to-host copy, same values as the
per-expert loop it replaces."""
import pytest
import torch

from connex.ep.vllm_ext import _slot_sums


def _old(layers, slots, S):
    return {str(p): float(sum(w[p % S].double().sum().item() for ps in layers for w in ps))
            for p in slots}


def test_slot_sums_match_per_expert_loop():
    g = torch.Generator().manual_seed(0)
    S = 4
    layers = [[torch.randn(S, 8, 6, generator=g).half(), torch.randn(S, 6, generator=g)]
              for _ in range(5)]
    slots = [4, 5, 7]                       # rank 1 of S=4
    new, old = _slot_sums(layers, slots, S), _old(layers, slots, S)
    assert new.keys() == old.keys()
    for k in old:
        assert new[k] == pytest.approx(old[k], rel=1e-12, abs=1e-9)
    assert _slot_sums(layers, [], S) == {}
