# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Stage 0's step control messages, sent off the engine thread.

Stage 0 sends each step's EXEC (the SchedulerOutput) and SAMPLE (the
grammar output) to one replica of every downstream stage. A connex-rs
``send_msg`` over TCP takes tens of microseconds for a small message and
about 0.3 ms (1 ms at p99) for 20 KB, and a prefill SchedulerOutput is tens
of KB: sent inline, PP=4 puts three such sends in a row on the engine
thread, twice per step. Here the engine thread only queues the bytes and
one thread per peer sends them.

What holds:

* Per peer, messages leave in the order they were queued (one FIFO, one
  thread). A stage executes EXEC and SAMPLE in arrival order, so it sees
  the same sequence as with inline sends. Different peers are not ordered
  against each other; a stage only reads its own control stream (hidden
  states that arrive before their EXEC wait in the data plane's inbox).
* Bounded: at most ``depth`` messages queued per peer; ``put`` waits for
  room (backpressure), and gives up when the peer is dropped or the sender
  closed meanwhile.
* A failed send is logged and reported to ``on_error`` (once per run of
  consecutive failures to that peer), so the owner can fail the peer over
  as it does for a connex-rs failure event. A failure of the sender itself
  (not of one send) is kept and raised by every later ``put``.
* ``drop(peer)`` discards what is queued for a peer that left the route:
  queued steps of a replica that failed must not reach a restarted
  incarnation under the same address. The send in progress completes.
* ``close(drain_s)`` lets the queues drain for up to ``drain_s``, then
  discards the rest; later ``put`` calls return False.

No import of vLLM here (CPU tests run without it).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from connex.adapters.vllm.pp.trace import tr

log = logging.getLogger("vllm.connex_pp")


class _Peer:
    __slots__ = ("url", "q", "cv", "gen", "busy", "fails", "thread")

    def __init__(self, url, lock: threading.Lock):
        self.url = url
        self.q: deque = deque()
        self.cv = threading.Condition(lock)
        self.gen = 0          # bumped by drop(): a put waiting for room gives up
        self.busy = False     # a send in progress
        self.fails = 0        # consecutive failed sends
        self.thread: threading.Thread | None = None


class CtlSender:
    def __init__(self, send, depth: int = 64, on_error=None, name: str = "cx-ctl-tx"):
        """``send(peer, data)`` does one send (connex-rs ``send_msg``; it is
        thread-safe, so one thread per peer is fine). ``on_error(peer,
        exc)`` runs on the peer's sender thread and must not block long."""
        self._send = send
        self.depth = max(1, int(depth))
        self.on_error = on_error
        self.name = name
        self.lock = threading.Lock()
        self.peers: dict = {}
        self.closed = False
        self.error: BaseException | None = None
        self.sent = 0
        self.failed = 0
        self.dropped = 0

    # ------------------------------------------------------------ engine side
    def put(self, peer, data: bytes, tag: tuple | None = None) -> bool:
        """Queue ``data`` for ``peer``. True when queued; False when the
        sender is closed or the peer was dropped while waiting for room.
        Raises when the sender has failed."""
        with self.lock:
            self._check()
            if self.closed:
                return False
            p = self.peers.get(peer)
            if p is None:
                p = self.peers[peer] = _Peer(peer, self.lock)
                p.thread = threading.Thread(target=self._run, args=(p,),
                                            name=f"{self.name}", daemon=True)
                p.thread.start()
            gen = p.gen
            while len(p.q) >= self.depth:
                p.cv.wait(1.0)
                self._check()
                if self.closed or p.gen != gen:
                    self.dropped += 1
                    return False
            p.q.append((data, tag))
            p.cv.notify_all()
        return True

    def _check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"connex-pp: control sender failed: {self.error!r}") from self.error

    def drop(self, peer) -> int:
        """Discard what is queued for ``peer``; returns how many."""
        with self.lock:
            p = self.peers.get(peer)
            if p is None:
                return 0
            n = len(p.q)
            p.q.clear()
            p.gen += 1
            p.fails = 0     # back on a route later, its first failure counts
            self.dropped += n
            p.cv.notify_all()
        if n:
            log.info("connex-pp: dropped %d queued control message(s) to %s", n, peer)
        return n

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every queue is empty and no send is in progress."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.lock:
            while True:
                if self.error is not None or all(not p.q and not p.busy
                                                 for p in self.peers.values()):
                    return self.error is None
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                # Any peer's condition shares this lock; senders notify all
                # of theirs when they finish a send.
                cv = next(iter(self.peers.values())).cv
                cv.wait(0.05 if left is None else min(left, 0.05))

    def close(self, drain_s: float = 1.0) -> None:
        if drain_s > 0:
            self.flush(drain_s)
        with self.lock:
            self.closed = True
            for p in self.peers.values():
                self.dropped += len(p.q)
                p.q.clear()
                p.gen += 1
                p.cv.notify_all()

    # ------------------------------------------------------------ sender side
    def _run(self, p: _Peer) -> None:
        try:
            while True:
                with self.lock:
                    while not p.q and not self.closed:
                        p.cv.wait()
                    if not p.q:
                        return
                    data, tag = p.q.popleft()
                    p.busy = True
                    p.cv.notify_all()     # room for a waiting put
                err = None
                try:
                    self._send(p.url, data)
                except Exception as e:  # noqa: BLE001
                    err = e
                if tag is not None:
                    tr(tag[0], tag[1])
                with self.lock:
                    p.busy = False
                    if err is None:
                        self.sent += 1
                        recovered, p.fails = p.fails, 0
                    else:
                        self.failed += 1
                        p.fails += 1
                    first = err is not None and p.fails == 1
                    p.cv.notify_all()
                if err is None:
                    if recovered:
                        log.info("connex-pp: control sends to %s work again after %d "
                                 "failure(s)", p.url, recovered)
                    continue
                if first:
                    log.warning("connex-pp: control send to %s failed: %s", p.url, err)
                    if self.on_error is not None:
                        try:
                            self.on_error(p.url, err)
                        except Exception:  # noqa: BLE001
                            log.exception("connex-pp: control send error handler failed")
        except BaseException as e:  # noqa: BLE001
            # Not a failed send: the sender itself is broken. Every later
            # put raises it, so the engine thread does not keep queueing
            # steps nobody sends.
            log.exception("connex-pp: control sender for %s died", p.url)
            with self.lock:
                self.error = e
                p.busy = False
                for x in self.peers.values():
                    x.cv.notify_all()
