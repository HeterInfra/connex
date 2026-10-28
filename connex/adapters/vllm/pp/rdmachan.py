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

Python only touches the flags on failure (``release_all``) and on an
aborted epoch (``release``), to let streams go that wait for data that will
not come.
"""

from __future__ import annotations

import itertools
import logging
import threading

import torch

from connex.adapters.vllm.pp.ipcchan import _ALIGN, _check, _cu, _mark, _wait, _write

log = logging.getLogger("vllm.connex_pp")

CANCEL = 0x7FFFFFFF
_qids = itertools.count(1)
_cids = itertools.count(1)


def _dev_ptr(host_ptr: int) -> int:
    cu = _cu()
    return int(_check(cu.cuMemHostGetDevicePointer(host_ptr, 0), "cuMemHostGetDevicePointer"))


def _register(engine, t: torch.Tensor) -> None:
    rc = engine.register_memory(t.data_ptr(), t.numel() * t.element_size())
    if rc < 0:
        raise RuntimeError(f"connex-rs: registering {t.numel() * t.element_size()} bytes failed ({rc})")


class PushRx:
    """Receiver end. ``host``: ring in pinned host memory (small payloads
    copied to the GPU by the caller) instead of GPU memory."""

    def __init__(self, engine, url: str, device: torch.device, num_slots: int,
                 slot_bytes: int, host: bool = False):
        self.engine = engine
        self.device = device
        self.n = num_slots
        self.slot_bytes = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        size = self.n * self.slot_bytes
        if host:
            self.ring = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        else:
            self.ring = torch.empty(size, dtype=torch.uint8, device=device)
        self.flags = torch.zeros(2 * self.n, dtype=torch.int32, pin_memory=True)
        self.arr = self.flags.numpy()
        self.dptr = _dev_ptr(self.flags.data_ptr())
        _register(engine, self.ring)
        _register(engine, self.flags)
        self.cid = next(_cids)
        self.info = (self.cid, url, self.ring.data_ptr(), self.slot_bytes, self.n,
                     self.flags.data_ptr())
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.retired = False
        self.last = -1        # counted channels: last index waited for

    def wait(self, v: int) -> torch.Tensor:
        """Queue a wait for message ``v`` on the current stream; returns the
        slot's bytes (valid until ``done(v)``)."""
        s = v % self.n
        stream = torch.cuda.current_stream(self.device)
        self.streams[stream.cuda_stream] = stream
        _wait(stream, self.dptr + 4 * s, v + 1)
        return self.ring[s * self.slot_bytes:(s + 1) * self.slot_bytes]

    def done(self, v: int) -> None:
        """Queue on the current stream: the slot of message ``v`` may be reused."""
        if self.retired:
            return
        _write(torch.cuda.current_stream(self.device), self.dptr + 4 * (self.n + v % self.n),
               v + 1)

    def fill(self, v: int, payload: bytes | None) -> None:
        """Host rings only: deliver message ``v`` from the CPU (a payload
        that came the slow way, or None to let the wait go without data)."""
        s = v % self.n
        if payload is not None and self.ring.device.type == "cpu":
            n = min(len(payload), self.slot_bytes)
            self.ring.numpy()[s * self.slot_bytes:s * self.slot_bytes + n] = \
                memoryview(payload)[:n]
        if self.arr[s] < v + 1:
            self.arr[s] = v + 1

    def release_all(self) -> None:
        self.arr[:self.n] = CANCEL

    def retire(self) -> list:
        self.retired = True
        self.release_all()
        self.arr[self.n:] = CANCEL   # a sender polling a free flag gives up
        return _mark(self.streams)

    def close(self) -> None:
        for t in (self.ring, self.flags):
            try:
                self.engine.unregister_memory(t.data_ptr())
            except Exception:  # noqa: BLE001
                pass
        self.ring = None


class PushTx:
    """Sender end, opened from a PushRx's ``info``. The data comes from
    registered local memory the caller manages (``src``), or, with
    ``staging`` slots, from a registered staging ring of this channel."""

    def __init__(self, engine, device: torch.device, info, staging: bool = False,
                 host_staging: bool = False, timeout_ms: int = 10000):
        self.cid, self.url, self.ring_ptr, self.slot_bytes, self.n, self.flags_ptr = info
        self.engine = engine
        self.device = device
        self.timeout_ms = timeout_ms
        self.bell = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.bell_np = self.bell.numpy()
        self.bell_dptr = _dev_ptr(self.bell.data_ptr())
        self.queue = next(_qids)
        self.i = 0              # counted channels: next index
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.staging = None
        if staging:
            size = self.n * self.slot_bytes
            self.staging = (torch.empty(size, dtype=torch.uint8, pin_memory=True) if host_staging
                            else torch.empty(size, dtype=torch.uint8, device=device))
            _register(engine, self.staging)
        self.lock = threading.Lock()
        self.dead = False
        # Fetch the peer's keys for the ring and the flags now, not on the
        # first push (the first access to a region costs a round trip).
        self.scratch = torch.zeros(4, dtype=torch.int32)
        try:
            for i, addr in enumerate((self.ring_ptr, self.flags_ptr)):
                engine.get_async(self.url, self.scratch.data_ptr() + 4 * i, addr, 4, timeout_ms)
        except Exception:  # noqa: BLE001  (the first push fetches them instead)
            pass

    def push(self, v: int, src_ptr: int, nbytes: int, gate: int | None = None):
        """Queue on the current stream the doorbell for message ``v`` (the
        data at ``src_ptr`` must be written by work queued before this call)
        and arm the chain that moves it. ``gate``: wait until the receiver
        freed the slot's previous message (free[s] >= gate). Returns the
        connex-rs transfer (done once the flag is written, or failed)."""
        if nbytes > self.slot_bytes:
            raise ValueError("message larger than the channel slot")
        s = v % self.n
        stream = torch.cuda.current_stream(self.device)
        self.streams[stream.cuda_stream] = stream
        _write(stream, self.bell_dptr, v + 1)
        gate_ptr = self.flags_ptr + 4 * (self.n + s) if gate else 0
        t = self.engine.put_chain(self.url, self.bell.data_ptr(), v + 1, gate_ptr,
                                  gate or 0, src_ptr, self.ring_ptr + s * self.slot_bytes,
                                  nbytes, self.flags_ptr + 4 * s, v + 1, self.queue,
                                  self.timeout_ms)
        return t

    def push_staged(self, v: int, chunks: list[torch.Tensor]):
        """Copy ``chunks`` (byte views) into staging slot ``v % n`` on the
        current stream, then push it."""
        s = v % self.n
        off = s * self.slot_bytes
        start = off
        for c in chunks:
            n = c.numel()
            self.staging[off:off + n].copy_(c, non_blocking=True)
            off += n
        return self.push(v, self.staging.data_ptr() + start, max(off - start, 4))

    def retire(self) -> list:
        """The receiver is gone: cancel the chains that wait (connex-rs
        drops them once it sees the doorbell's cancel value)."""
        self.dead = True
        self.bell_np[0] = CANCEL
        return _mark(self.streams)

    def close(self) -> None:
        if self.staging is not None:
            try:
                self.engine.unregister_memory(self.staging.data_ptr())
            except Exception:  # noqa: BLE001
                pass
            self.staging = None
