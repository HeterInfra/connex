# SPDX-License-Identifier: Apache-2.0
"""scripts/check_release.py detects what it should and passes clean files."""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _scan(tmp_path, content, extra=None):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for f in ("check_release.py", "release_patterns.txt"):
        shutil.copy(os.path.join(ROOT, "scripts", f), root / "scripts" / f)
    (root / "x.py").write_text(content)
    args = [sys.executable, str(root / "scripts" / "check_release.py"), "--all-files"]
    if extra:
        (tmp_path / "p.local").write_text(extra)
        args += ["--extra", str(tmp_path / "p.local")]
    return subprocess.run(args, capture_output=True, text=True)


def test_clean(tmp_path):
    r = _scan(tmp_path, 'URL = "http://127.0.0.1:8000"\nDOC = "192.0.2.10"\n')
    assert r.returncode == 0, r.stdout


def test_findings(tmp_path):
    key = "-----BEGIN RSA " + "PRIVATE KEY-----"
    home = "/" + "home/alice/x"
    ip = "172.31." + "4.5"
    r = _scan(tmp_path, f'a = "{key}"\nb = "{home}"\nc = "{ip}"\npassword = "hunter2hunter2"\n')
    assert r.returncode == 1
    for needle in ("private key", "home directory", "ipv4", "password"):
        assert needle in r.stdout, r.stdout


def test_extra_patterns_and_allow_marker(tmp_path):
    r = _scan(tmp_path, 'n = "secretnode7"\nm = "secretnode8"  # release-scan: allow\n',
              extra="node\tsecretnode\\d\n")
    assert r.returncode == 1 and r.stdout.count("[node]") == 1


def test_allow_block(tmp_path):
    r = _scan(tmp_path, '# release-scan: allow-begin\nn = "secretnode7"\n# release-scan: allow-end\nm = "secretnode8"\n',
              extra="node\tsecretnode\\d\n")
    assert r.returncode == 1
    assert "secretnode8" in r.stdout and "secretnode7" not in r.stdout, r.stdout
