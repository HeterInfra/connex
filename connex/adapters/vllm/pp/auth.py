# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Authenticated control frames of the external pipeline.

The control messages between stage 0 and the stage servers carry vLLM
objects (SchedulerOutput, GrammarOutput, ModelRunnerOutput), so they are
pickled, and unpickling runs code. Every control frame therefore carries an
HMAC-SHA256 tag over its payload, keyed with a secret all stages of a
deployment share; a frame without a valid tag is dropped before anything is
unpickled.

The secret comes from ``CONNEX_PP_SECRET`` (the value) or
``CONNEX_PP_SECRET_FILE`` (a file holding it; surrounding whitespace is
ignored), at least 16 bytes. ``docker/launch_stage.sh`` and
``deploy/k8s/manifests.py`` set it up. ``CONNEX_PP_AUTH=off`` turns the check
off (plain pickle, for a trusted test network only); without a secret and
without that, a control endpoint refuses to start.

Frame: ``b"CXA1" + tag[:16] + payload``. The tag binds only the payload: a
frame captured on the network can be replayed (the pipeline tolerates
duplicate control messages: a step carries its epoch and sequence number).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

log = logging.getLogger("vllm.connex_pp")

MAGIC = b"CXA1"
TAG_LEN = 16
MIN_SECRET = 16


class AuthError(RuntimeError):
    pass


def load_secret(env=None) -> bytes | None:
    """The shared secret, or None when authentication is off. Raises
    AuthError when it is neither configured nor explicitly turned off."""
    env = os.environ if env is None else env
    if env.get("CONNEX_PP_AUTH", "on").lower() in ("off", "0", "no", "false"):
        return None
    val = env.get("CONNEX_PP_SECRET")
    path = env.get("CONNEX_PP_SECRET_FILE")
    if not val and path:
        try:
            with open(path, "rb") as f:
                val = f.read().strip()
        except OSError as e:
            raise AuthError(f"CONNEX_PP_SECRET_FILE={path}: {e}") from e
    if not val:
        raise AuthError(
            "connex-pp control frames are authenticated: set CONNEX_PP_SECRET or "
            "CONNEX_PP_SECRET_FILE to the same secret on every stage (docker/launch_stage.sh "
            "and deploy/k8s/manifests.py do), or CONNEX_PP_AUTH=off on a trusted network")
    key = val.encode() if isinstance(val, str) else bytes(val)
    if len(key) < MIN_SECRET:
        raise AuthError(f"CONNEX_PP_SECRET must have at least {MIN_SECRET} bytes")
    return key


class FrameAuth:
    """seal()/open() control frames with ``key`` (None: no authentication)."""

    def __init__(self, key: bytes | None):
        self.key = key
        self.rejected = 0
        if key is None:
            log.warning("connex-pp: CONNEX_PP_AUTH=off: control frames are not authenticated; "
                        "anyone who reaches a control port can run code in this process")

    def seal(self, payload: bytes) -> bytes:
        if self.key is None:
            return payload
        tag = hmac.new(self.key, payload, hashlib.sha256).digest()[:TAG_LEN]
        return MAGIC + tag + payload

    def open(self, frame):
        """The payload (a memoryview) of a frame with a valid tag, else None."""
        view = memoryview(frame)
        if self.key is None:
            return view
        n = len(MAGIC) + TAG_LEN
        if len(view) < n or view[:len(MAGIC)] != MAGIC:
            self.rejected += 1
            return None
        tag = view[len(MAGIC):n]
        payload = view[n:]
        want = hmac.new(self.key, payload, hashlib.sha256).digest()[:TAG_LEN]
        if not hmac.compare_digest(tag, want):
            self.rejected += 1
            return None
        return payload
