#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release scanner: fail if the tree contains environment-specific or secret data.

    python scripts/check_release.py                 # scan git-tracked files (or the whole tree)
    python scripts/check_release.py --all-files     # scan every file under the root
    python scripts/check_release.py --list          # print the active patterns

Checks
  * generic patterns in scripts/release_patterns.txt (IPv4 literals outside an
    allowlist, home/data directories, internal registries, e-mail addresses,
    proxies, ...)
  * secret patterns (private keys, API tokens, password assignments)
  * extra deployment-specific patterns from a private file that is NOT
    committed: --extra FILE, or $CONNEX_RELEASE_PATTERNS, or
    ./.release-patterns.local if present (one regex per line, '#' comments)

A line can be exempted with a trailing marker `release-scan: allow` (use for
documentation examples only), and a block with `release-scan: allow-begin` ...
`release-scan: allow-end` (e.g. HTML comments around the citation). Exit status 1 if anything is found.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS_FILE = os.path.join(ROOT, "scripts", "release_patterns.txt")
LOCAL_FILE = os.path.join(ROOT, ".release-patterns.local")
ALLOW_MARK = "release-scan: allow"
# Lines between these markers are exempt too (e.g. the BibTeX citation block,
# whose author list is public by design).
ALLOW_BEGIN, ALLOW_END = "release-scan: allow-begin", "release-scan: allow-end"
SKIP_FILES = {"scripts/release_patterns.txt", "LICENSE"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".so", ".tgz", ".gz", ".whl", ".ico"}

SECRET_PATTERNS = [
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("GitHub token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    ("Slack token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("sk- style API key", r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    ("HF token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("Notion token", r"\b(?:secret_|ntn_)[A-Za-z0-9]{30,}\b"),
    ("password/token assignment",
     r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*['\"][^'\"\s$<{]{8,}['\"]"),
    ("URL with credentials", r"[a-z]+://[^/\s:@]+:[^/\s:@]+@"),
]

IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?:/\d{1,2})?(?![\w.])")
# loopback / unspecified / documentation ranges / link-local metadata are fine
ALLOWED_NETS = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "0.0.0.0/32", "255.255.255.255/32",
    "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
    # synthetic addresses used throughout the unit tests
    "10.0.0.0/24", "10.9.9.9/32", "1.2.3.4/32", "9.9.9.9/32")]


def load_patterns(path):
    out = []
    if not path or not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, _, rx = line.partition("\t") if "\t" in line else (line, "", line)
        out.append((name.strip(), (rx or name).strip()))
    return out


def tracked_files(all_files):
    if not all_files:
        try:
            r = subprocess.run(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"],
                               capture_output=True, text=True, check=True)
            return [f for f in r.stdout.splitlines() if os.path.isfile(os.path.join(ROOT, f))]
        except Exception:
            pass
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", "build") and not x.endswith(".egg-info")]
        out += [os.path.relpath(os.path.join(d, f), ROOT) for f in files]
    return out


def bad_ip(s):
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    return not any(ip in n for n in ALLOWED_NETS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-files", action="store_true")
    ap.add_argument("--extra", default=os.environ.get("CONNEX_RELEASE_PATTERNS", LOCAL_FILE))
    ap.add_argument("--no-ip", action="store_true", help="skip the IPv4 literal check")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    pats = [(n, re.compile(r)) for n, r in
            load_patterns(PATTERNS_FILE) + load_patterns(a.extra) + SECRET_PATTERNS]
    if a.list:
        for n, r in pats:
            print(f"{n}\t{r.pattern}")
        return 0
    me = os.path.relpath(os.path.abspath(__file__), ROOT)
    hits = 0
    for rel in sorted(tracked_files(a.all_files)):
        if rel in SKIP_FILES or rel == me or rel.startswith(".release-patterns") \
                or os.path.splitext(rel)[1].lower() in SKIP_EXT:
            continue
        # a file *name* can leak as well
        for n, r in pats:
            if r.search(rel):
                print(f"{rel}: [{n}] in path")
                hits += 1
        try:
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        in_allow = False
        for i, line in enumerate(text.splitlines(), 1):
            if ALLOW_BEGIN in line:
                in_allow = True
            if ALLOW_END in line:
                in_allow = False
                continue
            if in_allow or ALLOW_MARK in line:
                continue
            found = [n for n, r in pats if r.search(line)]
            if not a.no_ip:
                found += [f"ipv4 {m.group(1)}" for m in IPV4.finditer(line) if bad_ip(m.group(1))]
            for n in dict.fromkeys(found):
                print(f"{rel}:{i}: [{n}] {line.strip()[:160]}")
                hits += 1
    if hits:
        print(f"\nrelease scan FAILED: {hits} finding(s)")
        return 1
    print(f"release scan passed ({len(pats)} patterns + IPv4 check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
