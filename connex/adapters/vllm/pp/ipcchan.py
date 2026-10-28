# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""GPU-driven same-host channel between two pipeline-stage processes.

The sender owns a ring of N fixed-size slots in its GPU memory, exported to
the receiver over CUDA IPC, and a pair of flag arrays in shared host memory
(``/dev/shm``, registered as mapped pinned memory in both processes):

    sender stream:   wait free[s] >= i-N+1   copy -> slot s   write ready[s] = i+1
    receiver stream: wait ready[s] >= i+1    use slot s       write free[s]  = i+1

Every step is a CUDA stream operation (``cuStreamWaitValue32`` /
``cuStreamWriteValue32``), so like a NCCL send/recv nothing on the data path
depends on a Python thread. That matters: a stream wait that only a Python
thread can release deadlocks against CUDA calls that hold the GIL and wait
for the device (a first Triton kernel load does), see hostwait.py.

Python only touches the flags on failure (``release_all``), to let streams
go that wait for a peer that died.
"""

from __future__ import annotations

import mmap
import os
import threading

import numpy as np
import torch

_ALIGN = 256


def _cu():
    from cuda.bindings import driver as cu  # cuda-python

    return cu


def _check(res, what):
    cu = _cu()
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")
    return res[1] if isinstance(res, tuple) and len(res) > 1 else None


class _Flags:
    """2*n int32 in /dev/shm, mapped into the GPU's address space."""

    def __init__(self, name: str, n: int, create: bool):
        self.name, self.n = name, n
        path = "/dev/shm/" + name
        size = 2 * n * 4
        if create:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
            os.ftruncate(fd, 4096)
        else:
            fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(fd, 4096)
        os.close(fd)
        self.arr = np.frombuffer(self.mm, dtype=np.int32, count=2 * n)
        cu = _cu()
        self.host_ptr = np.frombuffer(self.mm, dtype=np.uint8).ctypes.data
        flags = (cu.CU_MEMHOSTREGISTER_DEVICEMAP | cu.CU_MEMHOSTREGISTER_PORTABLE)
        _check(cu.cuMemHostRegister(self.host_ptr, 4096, flags), "cuMemHostRegister")
        self.dptr = int(_check(cu.cuMemHostGetDevicePointer(self.host_ptr, 0),
                               "cuMemHostGetDevicePointer"))
        self.size = size

    def ready_addr(self, s: int) -> int:
        return self.dptr + 4 * s

    def free_addr(self, s: int) -> int:
        return self.dptr + 4 * (self.n + s)

    def close(self, unlink: bool) -> None:
        try:
            _cu().cuMemHostUnregister(self.host_ptr)
        except Exception:  # noqa: BLE001
            pass
        if unlink:
            try:
                os.unlink("/dev/shm/" + self.name)
            except OSError:
                pass


def _wait(stream: torch.cuda.Stream, addr: int, value: int) -> None:
    cu = _cu()
    _check(cu.cuStreamWaitValue32(cu.CUstream(stream.cuda_stream), cu.CUdeviceptr(addr),
                                  value & 0x7FFFFFFF,
                                  cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
           "cuStreamWaitValue32")


def _write(stream: torch.cuda.Stream, addr: int, value: int) -> None:
    cu = _cu()
    _check(cu.cuStreamWriteValue32(cu.CUstream(stream.cuda_stream), cu.CUdeviceptr(addr),
                                   value & 0x7FFFFFFF,
                                   cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT),
           "cuStreamWriteValue32")


_counter = 0
_counter_lock = threading.Lock()


def _remove_stale_flags() -> None:
    """Flag files of processes that died without cleaning up (SIGKILL)."""
    try:
        names = [n for n in os.listdir("/dev/shm") if n.startswith("cxpp-")]
    except OSError:
        return
    for n in names:
        try:
            pid = int(n.split("-")[1])
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                os.unlink("/dev/shm/" + n)
            except OSError:
                pass
        except (ValueError, PermissionError, IndexError):
            pass


class TxChannel:
    """Sender end; ``info`` is what the receiver needs to open it."""

    def __init__(self, device: torch.device, num_slots: int, slot_bytes: int, tag: str):
        global _counter
        with _counter_lock:
            _counter += 1
            cid = _counter
        if cid == 1:
            _remove_stale_flags()
        self.n = num_slots
        self.slot_bytes = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        self.ring = torch.empty(self.n * self.slot_bytes, dtype=torch.uint8, device=device)
        self.flags = _Flags(f"cxpp-{os.getpid()}-{tag}-{cid}", self.n, create=True)
        self.flags.arr[:] = 0
        self.i = 0  # next message index
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.info = (self.flags.name, self.n, self.slot_bytes,
                     self.ring.untyped_storage()._share_cuda_())

    def push(self, chunks: list[torch.Tensor]) -> int:
        """Queue on the current stream: wait for the slot, copy ``chunks``
        (byte views, packed in order) into it, mark it ready. Returns the
        message index."""
        i = self.i
        self.i += 1
        s = i % self.n
        stream = torch.cuda.current_stream(self.ring.device)
        self.streams[stream.cuda_stream] = stream
        if i >= self.n:
            _wait(stream, self.flags.free_addr(s), i - self.n + 1)
        off = s * self.slot_bytes
        for c in chunks:
            n = c.numel()
            if off + n > (s + 1) * self.slot_bytes:
                raise ValueError("message larger than the channel slot")
            self.ring[off:off + n].copy_(c, non_blocking=True)
            off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        _write(stream, self.flags.ready_addr(s), i + 1)
        return i

    def release_all(self) -> None:
        """The receiver is gone: let pending pushes go."""
        self.flags.arr[self.n:] = 0x7FFFFFFF

    def retire(self) -> list:
        """Let pending pushes go; returns events after which the channel's
        memory may be released (streams may still reference the flags)."""
        self.release_all()
        return _mark(self.streams)

    def close(self) -> None:
        self.flags.close(unlink=True)
        self.ring = None


class RxChannel:
    """Receiver end, opened from a TxChannel's ``info``."""

    def __init__(self, device: torch.device, info):
        name, self.n, self.slot_bytes, handle = info
        self.flags = _Flags(name, self.n, create=False)
        args = list(handle)
        args[0] = device.index if device.index is not None else 0
        storage = torch.UntypedStorage._new_shared_cuda(*args)
        self.ring = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (storage.nbytes(),))
        self.device = device
        self.i = 0  # next index for counted receives
        self.pending: set[int] = set()
        self.streams: dict[int, torch.cuda.Stream] = {}
        self.retired = False

    def wait(self, i: int) -> torch.Tensor:
        """Queue a wait for message ``i`` on the current stream; returns the
        slot's bytes (valid until ``done(i)``)."""
        s = i % self.n
        stream = torch.cuda.current_stream(self.device)
        self.streams[stream.cuda_stream] = stream
        _wait(stream, self.flags.ready_addr(s), i + 1)
        self.pending.add(i)
        return self.ring[s * self.slot_bytes:(s + 1) * self.slot_bytes]

    def done(self, i: int) -> None:
        """Queue on the current stream: slot of message ``i`` may be reused."""
        if self.retired:
            return
        _write(torch.cuda.current_stream(self.device), self.flags.free_addr(i % self.n), i + 1)
        self.pending.discard(i)

    def release_all(self) -> None:
        """The sender is gone: let pending waits go (the data is garbage)."""
        self.flags.arr[:self.n] = 0x7FFFFFFF

    def retire(self) -> list:
        """Let pending waits go; returns events after which the mapping of
        the sender's ring and the flags may be released."""
        self.retired = True
        self.release_all()
        return _mark(self.streams)

    def close(self) -> None:
        self.flags.close(unlink=False)
        self.ring = None  # drops the IPC mapping (a dead sender's memory is freed)


def _mark(streams: dict) -> list:
    evs = []
    for st in streams.values():
        with torch.cuda.device(st.device):
            ev = torch.cuda.Event()
            ev.record(st)
        evs.append(ev)
    return evs


class Graveyard:
    """Channels whose peer is gone. Their flags were released at once (so
    no stream waits forever); the memory is unmapped only once every stream
    that used the channel has passed the release, otherwise a queued wait
    or copy would touch unmapped memory."""

    def __init__(self):
        self.items: list = []
        self.lock = threading.Lock()

    def add(self, ch) -> None:
        evs = ch.retire()
        with self.lock:
            self.items.append((ch, evs))

    def drain(self) -> None:
        with self.lock:
            keep, done = [], []
            for ch, evs in self.items:
                (done if all(e.query() for e in evs) else keep).append((ch, evs))
            self.items = keep
        for ch, _ in done:
            ch.close()
