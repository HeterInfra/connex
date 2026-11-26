# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""CPU checks for the PP x EP gate tooling (benchmarks/pp_upstream/ep_gate)."""

import importlib.util
import json
from pathlib import Path

_GATE = Path(__file__).resolve().parents[1] / "benchmarks" / "pp_upstream" / "ep_gate" / "gate.py"


def _gate():
    spec = importlib.util.spec_from_file_location("ep_gate_gate", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, name, runs):
    p = tmp_path / name
    p.write_text(json.dumps([{"prompt": f"p{i}", "text": "", "ids": ids} for i, ids in enumerate(runs)]))
    return str(p)


def test_cmp_exact_and_first_diff(tmp_path, capsys):
    g = _gate()
    a = _write(tmp_path, "a.json", [[1, 2, 3], [4, 5, 6], [7, 8]])
    assert g.cmp(a, _write(tmp_path, "b.json", [[1, 2, 3], [4, 5, 6], [7, 8]]))
    assert "3/3 exact" in capsys.readouterr().out
    assert not g.cmp(a, _write(tmp_path, "c.json", [[1, 2, 3], [4, 9, 6], [7]]))
    out = capsys.readouterr().out
    assert "1/3 exact" in out and "[1, 1]" in out


def test_prompts_fixed():
    g = _gate()
    assert len(g.PROMPTS) == 20 and len(set(g.PROMPTS)) == 20
