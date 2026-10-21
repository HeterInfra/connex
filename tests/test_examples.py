# SPDX-License-Identifier: Apache-2.0
"""The README's generic example runs end to end on CPU."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mobile_endpoints_example():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "mobile_endpoints.py")],
                         capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    for line in ("handover:   [('demo', 1, 0, 0)] b'step-42'", "migrate:    b'after-migrate'",
                 "scale-down: [('demo', 1, 0, 0)]"):
        assert line in out.stdout, out.stdout
