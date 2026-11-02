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


def test_slot_sums_chunked_matches():
    """Issue #19: slots are summed in chunks (bounded float64 transient);
    a chunk of one slot gives the same values."""
    g = torch.Generator().manual_seed(1)
    S = 8
    layers = [[torch.randn(S, 16, 4, generator=g).half(), torch.randn(S, 5, generator=g)]
              for _ in range(3)]
    slots = list(range(8, 16))              # rank 1 of S=8
    old = _old(layers, slots, S)
    for budget in (1, 16 * 4 * 8 * 3, 1 << 30):
        new = _slot_sums(layers, slots, S, budget=budget)
        assert new.keys() == old.keys()
        for k in old:
            assert new[k] == pytest.approx(old[k], rel=1e-12, abs=1e-9)
