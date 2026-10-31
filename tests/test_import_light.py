# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``import connex`` (and the router/proxy entry points) must not pull in
torch or vLLM: the package resolves its public names lazily."""

import subprocess
import sys

import pytest

CHECK = ("import sys, {mod}; heavy = [m for m in ('torch', 'vllm') if m in sys.modules]; "
         "print(','.join(heavy))")


@pytest.mark.parametrize("mod", ["connex", "connex.pp", "connex.front_router",
                                 "connex.directory_server", "connex.serving.proxy"])
def test_import_does_not_load_torch(mod):
    out = subprocess.run([sys.executable, "-c", CHECK.format(mod=mod)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"{mod} imported {out.stdout.strip()}"


def test_lazy_names_resolve():
    import connex

    assert set(connex.__all__) <= set(dir(connex))
    assert connex.RouteTable.__name__ == "RouteTable"
    with pytest.raises(AttributeError):
        connex.no_such_name
