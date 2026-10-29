# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""GPU-driven cross-host channel: the sender's NIC writes the receiver's ring
and flag, no Python thread on either side is on the data path.

The receiver owns a ring of N fixed-size slots (GPU memory for hidden
states, pinned host memory for the small sampled-token payloads) and a pair
of flag arrays in pinned host memory, all registered with connex-rs, and
tells the sender where they are (``info``). Per message ``v`` in slot ``s``:

    sender stream:    copy -> local staging     write doorbell = v+1
    sender connex-rs: wait doorbell >= v+1  [read free[s] until >= gate]
                      put staging -> ring slot s   (remotely complete)
                      put ready[s] = v+1
    receiver stream:  wait ready[s] >= v+1   use slot s   write free[s] = v+1

The doorbell is a word of the sender's pinned memory that its own stream
writes (``cuStreamWriteValue32``) and its connex-rs progress thread polls
(``Engine.put_chain``); the flag write follows the data write only after the
data put is remotely complete. Like the same-host CUDA IPC channel
(ipcchan.py) and like NCCL, nothing on the path needs the GIL, so a CUDA call
that holds the GIL and waits for the device (a first Triton kernel load)
cannot deadlock against it.

Values are counts modulo 2^32 compared cyclically on both ends (ipcchan's
``reached``; connex-rs>=0.1.5 compares the same way). A chain is cancelled
through a separate word of the sender's pinned memory that only the CPU
writes, one per epoch (``cancel_below``), never through the doorbell: a
stream write still queued could overwrite a cancel value there.

Every RMA from or into GPU memory is at least ``ZCOPY_MIN`` bytes: below
UCX's zero-copy threshold UCX copies with the CPU, which on a GPU pointer
is a segfault (connex-rs>=0.1.5 rejects such a transfer instead). Slots are
at least that large and the senders' buffers are allocated in such units.

Python only touches the flags on failure (``release_all``) and on an
aborted epoch (``release_below``), to let streams go that wait for data that
will not come.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time

import numpy as np
import torch

from connex.adapters.vllm.pp.ipcchan import (AHEAD, MASK, _ALIGN, _check, _cu, _mark, _wait,
                                             _write, reached)

log = logging.getLogger("vllm.connex_pp")

# Smallest RMA on GPU memory (UCX_ZCOPY_THRESH set by link._setup_ucx_env).
ZCOPY_MIN = 4096
_CANCEL_WORDS = 64
_qids = itertools.count(1)
_cids = itertools.count(1)


def _dev_ptr(host_ptr: int) -> int:
    cu = _cu()
    return int(_check(cu.cuMemHostGetDevicePointer(host_ptr, 0), "cuMemHostGetDevicePointer"))


def _register(engine, t: torch.Tensor) -> None:
    rc = engine.register_memory(t.data_ptr(), t.numel() * t.element_size())
    if rc < 0:
        raise RuntimeError(f"connex-rs: registering {t.numel() * t.element_size()} bytes failed ({rc})")


def slot_size(nbytes: int) -> int:
    """Slot size for messages of up to ``nbytes`` (aligned, >= ZCOPY_MIN)."""
    return max((nbytes + _ALIGN - 1) // _ALIGN * _ALIGN, ZCOPY_MIN)


class PushRx:
    """Receiver end. ``host``: ring in pinned host memory (small payloads
    copied to the GPU by the caller) instead of GPU memory."""

    def __init__(self, engine, url: str, device: torch.device, num_slots: int,
                 slot_bytes: int, host: bool = False):
        self.engine = engine
        self.device = device
        self.n = num_slots
        self.slot_bytes = slot_size(slot_bytes)
        size = self.n * self.slot_bytes
        if host:
            self.ring = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        else:
            self.ring = torch.empty(size, dtype=torch.uint8, device=device)
        self.flags = torch.zeros(2 * self.n, dtype=torch.int32, pin_memory=True)
        self.arr = self.flags.numpy().view(np.uint32)
        self.dptr = _dev_ptr(self.flags.data_ptr())
        _register(engine, self.ring)
        _register(engine, self.flags)
        self.registered = True
        self.cid = next(_cids)
        self.info = (self.cid, url, self.ring.data_ptr(), self.slot_bytes, self.n,
                     self.flags.data_ptr())
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.retired = False
        self.last = -1        # last message index waited for (counted / keyed)
        self.hi = 0           # highest count waited for
        self.pending: dict[int, int] = {}   # message -> epoch, waited for, not done
        # Sampled-token channels: the (epoch, first seq) the sender pushes
        # from (None: nothing yet); set by the receiver when it tells the
        # sender (CHAN / CHANX).
        self.from_key: tuple | None = None
        self.created = time.monotonic()
        self.lock = threading.Lock()

    def wait(self, v: int, epoch: int = 0) -> torch.Tensor:
        """Queue a wait for message ``v`` on the current stream; returns the
        slot's bytes (valid until ``done(v)``)."""
        s = v % self.n
        stream = torch.cuda.current_stream(self.device)
        self.streams[stream.cuda_stream] = stream
        _wait(stream, self.dptr + 4 * s, v + 1)
        with self.lock:
            self.pending[v] = epoch
            self.hi = max(self.hi, v + 1)
        return self.ring[s * self.slot_bytes:(s + 1) * self.slot_bytes]

    def done(self, v: int) -> None:
        """Queue on the current stream: the slot of message ``v`` may be reused."""
        if self.retired:
            return
        _write(torch.cuda.current_stream(self.device), self.dptr + 4 * (self.n + v % self.n),
               v + 1)
        with self.lock:
            for j in [j for j in self.pending if j <= v]:
                del self.pending[j]

    def skip_to(self, v: int) -> None:
        """Before waiting for ``v``: mark the slots of the messages since the
        last one waited for as free (messages of aborted steps, or ones that
        came another way), so the sender's gates for later messages open."""
        for j in range(max(self.last + 1, v - self.n), v):
            self.done(j)
        self.last = max(self.last, v)

    def fill(self, v: int, payload: bytes | None) -> None:
        """Deliver message ``v`` from the CPU: a payload that came the slow
        way (host rings only), or None to let the wait go without data."""
        s = v % self.n
        if payload is not None and self.ring.device.type == "cpu":
            n = min(len(payload), self.slot_bytes)
            self.ring.numpy()[s * self.slot_bytes:s * self.slot_bytes + n] = \
                np.frombuffer(memoryview(payload)[:n], dtype=np.uint8)
        if not reached(self.arr[s], v + 1):
            self.arr[s] = (v + 1) & MASK

    def release_below(self, epoch: int) -> int:
        """Let the waits for messages of epochs before ``epoch`` go: their
        sender may never push them (replaced, or hung). A push that still
        comes lands before the slot's next message (the sender's gate and
        chain order), so nothing later is overwritten."""
        with self.lock:
            old = [v for v, ep in self.pending.items() if ep < epoch]
        for v in old:
            self.fill(v, None)
        return len(old)

    def release_all(self) -> None:
        self.arr[:self.n] = (self.hi + AHEAD) & MASK

    reassert = release_all

    def retire(self) -> list:
        """The sender is gone or replaced: let every wait go. A put the
        sender's NIC still had in flight could land after the release and
        lower a flag; the fence (unregistering, which completes once the
        sender has dropped our keys after its in-flight operations) runs on
        a thread and re-asserts the release after it, and the graveyard
        re-asserts it until the streams are past."""
        self.retired = True
        self.release_all()
        threading.Thread(target=self._fence, name="cx-pp-fence", daemon=True).start()
        return _mark(self.streams)

    def _fence(self) -> None:
        self._unregister()
        self.release_all()

    def _unregister(self) -> None:
        with self.lock:
            if not self.registered:
                return
            self.registered = False
        for t in (self.ring, self.flags):
            try:
                self.engine.unregister_memory(t.data_ptr())
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._unregister()
        self.ring = None


class PushTx:
    """Sender end, opened from a PushRx's ``info``. The data comes from
    registered local memory the caller manages (``src``), or, with
    ``staging`` slots, from a registered staging ring of this channel."""

    def __init__(self, engine, device: torch.device, info, staging: bool = False,
                 host_staging: bool = False, timeout_ms: int = 10000,
                 arm_timeout_ms: int | None = 60000):
        self.cid, self.url, self.ring_ptr, self.slot_bytes, self.n, self.flags_ptr = info
        if self.slot_bytes < ZCOPY_MIN:
            raise ValueError(f"push channel slots of {self.slot_bytes} bytes (< {ZCOPY_MIN})")
        self.engine = engine
        self.device = device
        self.timeout_ms = timeout_ms
        self.arm_timeout_ms = arm_timeout_ms
        self.bell = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.bell_dptr = _dev_ptr(self.bell.data_ptr())
        # Cancel words, one per epoch (ring of _CANCEL_WORDS); only the CPU
        # writes them.
        self.cancel = torch.zeros(_CANCEL_WORDS, dtype=torch.int32, pin_memory=True)
        self.cancel_np = self.cancel.numpy()
        self.word_epoch = [-1] * _CANCEL_WORDS
        self.queue = next(_qids)
        self.i = 0              # counted channels: next index
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.staging = None
        self.host_staging = host_staging
        if staging:
            size = self.n * self.slot_bytes
            self.staging = (torch.empty(size, dtype=torch.uint8, pin_memory=True) if host_staging
                            else torch.empty(size, dtype=torch.uint8, device=device))
            _register(engine, self.staging)
        self.slot_t: list = [None] * self.n   # staging slot -> chain reading it
        self.lock = threading.Lock()
        self.dead = False
        self.extra = None
        self.created = time.monotonic()
        self.first_done: float | None = None
        # Fetch the peer's keys for the ring and the flags now, not on the
        # first push (the first access to a region costs a round trip).
        self.scratch = torch.zeros(4, dtype=torch.int32)
        try:
            for i, addr in enumerate((self.ring_ptr, self.flags_ptr)):
                engine.get_async(self.url, self.scratch.data_ptr() + 4 * i, addr, 4, timeout_ms)
        except Exception:  # noqa: BLE001  (the first push fetches them instead)
            pass

    def _word(self, epoch: int) -> int:
        k = epoch % _CANCEL_WORDS
        if self.word_epoch[k] != epoch:
            self.word_epoch[k] = epoch
            # A chain of an epoch already aborted (a stale step still being
            # executed) starts cancelled, even if no word existed for it yet.
            self.cancel_np[k] = 1 if (self.dead or epoch < getattr(self, "cancel_floor", 0)) \
                else 0
        return self.cancel.data_ptr() + 4 * k

    def cancel_below(self, epoch: int) -> None:
        """Drop the chains of epochs before ``epoch`` that still wait for
        their doorbell or gate (aborted steps)."""
        self.cancel_floor = max(getattr(self, "cancel_floor", 0), epoch)
        for k, ep in enumerate(self.word_epoch):
            if 0 <= ep < epoch:
                self.cancel_np[k] = 1

    def push(self, v: int, src_ptr: int, nbytes: int, gate: int | None = None,
             epoch: int = 0, doorbell: bool = True):
        """Queue on the current stream the doorbell for message ``v`` (the
        data at ``src_ptr`` must be written by work queued before this call)
        and arm the chain that moves it. ``gate``: wait until the receiver
        freed the slot's previous message (free[s] >= gate). ``nbytes`` is
        rounded up to ZCOPY_MIN: ``src_ptr`` must have that many registered
        bytes. Returns the connex-rs transfer (done once the flag is
        written, or failed)."""
        nbytes = max(nbytes, ZCOPY_MIN)
        if nbytes > self.slot_bytes:
            raise ValueError("message larger than the channel slot")
        s = v % self.n
        if doorbell:
            stream = torch.cuda.current_stream(self.device)
            self.streams[stream.cuda_stream] = stream
            _write(stream, self.bell_dptr, v + 1)
        gate_ptr = self.flags_ptr + 4 * (self.n + s) if gate is not None else 0
        return self.engine.put_chain(
            self.url, self.bell.data_ptr() if doorbell else 0, (v + 1) & MASK, gate_ptr,
            (gate or 0) & MASK, src_ptr, self.ring_ptr + s * self.slot_bytes, nbytes,
            self.flags_ptr + 4 * s, (v + 1) & MASK, self.queue, self.timeout_ms,
            cancel=self._word(epoch), arm_timeout_ms=self.arm_timeout_ms)

    def push_staged(self, v: int, chunks: list[torch.Tensor], gate: int | None = None,
                    epoch: int = 0):
        """Copy ``chunks`` (byte views) into staging slot ``v % n`` on the
        current stream, then push it. The previous chain that read the slot
        must be done first (waited for on the CPU; with n slots that is n
        messages back, normally long done)."""
        s = v % self.n
        prev = self.slot_t[s]
        if prev is not None and not prev.done():
            try:
                prev.wait(self.timeout_ms)
            except Exception:  # noqa: BLE001  (failed: the slot is free anyway)
                pass
        off = s * self.slot_bytes
        start = off
        for c in chunks:
            n = c.numel()
            self.staging[off:off + n].copy_(c, non_blocking=True)
            off += n
        n = off - start
        if not self.host_staging:
            n = max(n, ZCOPY_MIN)
        t = self.push(v, self.staging.data_ptr() + start, max(n, 4), gate, epoch)
        self.slot_t[s] = t
        return t

    def staged_bytes(self, v: int, n: int) -> bytes:
        """The staged payload of message ``v`` (host staging), for resending
        it the slow way after a failed chain."""
        s = v % self.n
        return self.staging[s * self.slot_bytes:s * self.slot_bytes + n].numpy().tobytes()

    def warm(self, src_ptr: int | None = None):
        """A first chain right away (no doorbell): connects and fetches the
        keys before a step needs the channel. Uses index ``self.i`` (the
        receiver skips it like an unconsumed message)."""
        v = self.i
        self.i += 1
        if src_ptr is None:
            src_ptr = self.staging.data_ptr() + (v % self.n) * self.slot_bytes
        return self.push(v, src_ptr, ZCOPY_MIN, None, 0, doorbell=False)

    def retire(self) -> list:
        """The receiver is gone or left the route: cancel the chains that
        wait (connex-rs drops them once it reads a cancel word)."""
        self.dead = True
        self.cancel_np[:] = 1
        return _mark(self.streams)

    def close(self) -> None:
        if self.staging is not None:
            try:
                self.engine.unregister_memory(self.staging.data_ptr())
            except Exception:  # noqa: BLE001
                pass
            self.staging = None
