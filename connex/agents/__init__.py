# SPDX-License-Identifier: Apache-2.0
"""Agent sessions with stable logical ids that can move between serving
instances (load balancing, scale-down, failure) without the client noticing.

``SessionProxy`` extends the resume proxy (``connex.serving.proxy``) with a
session table: each session (``session_id`` in the body, or the
``X-Connex-Session`` header) has a logical id, a placement (the instance
that holds its prefix state) and an epoch that increases on every move.
Requests of a session are routed to its placement (sticky). ``migrate``
moves a session: it first makes the target hold the session's prefix state
(through a ``KVMover`` that transfers the KV, or by re-prefilling the
conversation on the target so its prefix cache holds it), then bumps the
epoch, repoints the session, and cuts in-flight generations over with the
proxy's resume (prompt + tokens emitted so far). If the placement dies the
next attempt goes elsewhere and re-prefills there.
"""
from .session import KVMover, ReprefillMover, Session, SessionProxy

__all__ = ["KVMover", "ReprefillMover", "Session", "SessionProxy"]
