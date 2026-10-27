# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Let a CUDA stream wait for data that a host thread delivers later.

The V2 model runner receives the sampled tokens of step T asynchronously:
it issues the receive on a side stream, records an event, and only waits for
that event pp_size steps later. NCCL gives this for free. A host-side
transport needs the side stream to block until the bytes have arrived:
``cuStreamWaitValue32`` on a flag in pinned host memory does that without
occupying an SM, and a host-to-device copy queued behind it moves the bytes.

    slot = waiter.post(tensor)        # on the current stream: wait + H2D copy
    ...                               # later, from any thread:
    waiter.fill(slot, payload_bytes)  # writes the staging buffer, sets the flag
    waiter.release(slot)              # or: let the stream go without data
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch



class HostFlagWaiter:
    def __init__(self, device: torch.device, num_slots: int = 256, slot_bytes: int = 1 << 16):
        from cuda.bindings import driver as cu  # cuda-python

        self.cu = cu
        self.device = device
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        self.flags = torch.zeros(num_slots, dtype=torch.int32, pin_memory=True)
        self.flags_np = self.flags.numpy()
        self.staging = torch.empty(num_slots * slot_bytes, dtype=torch.uint8, pin_memory=True)
        self.staging_np = self.staging.numpy()
        err, self.flags_dptr = cu.cuMemHostGetDevicePointer(self.flags.data_ptr(), 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemHostGetDevicePointer failed: {err}")
        self.gen = np.zeros(num_slots, dtype=np.int64)
        self.busy: dict[int, torch.cuda.Event | None] = {}
        self.next = 0
        self.lock = threading.Lock()

    def _free_slot(self) -> int:
        while True:
            with self.lock:
                for _ in range(self.num_slots):
                    i = self.next
                    self.next = (self.next + 1) % self.num_slots
                    ev = self.busy.get(i, "free")
                    if isinstance(ev, str) or (ev is not None and ev.query()):
                        self.busy[i] = None
                        return i
            torch.cuda.current_stream(self.device).synchronize()

    def post(self, tensor: torch.Tensor) -> int:
        """Queue on the current stream: wait for the slot's flag, then copy
        the slot's staging bytes into ``tensor``. Returns the slot."""
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.slot_bytes:
            raise ValueError(f"{nbytes} bytes > slot size {self.slot_bytes}")
        slot = self._free_slot()
        stream = torch.cuda.current_stream(self.device)
        with self.lock:
            self.gen[slot] += 1
            target = int(self.gen[slot]) & 0x7FFFFFFF
        cu = self.cu
        (err,) = cu.cuStreamWaitValue32(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(int(self.flags_dptr) + 4 * slot),
            target,
            cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {err}")
        src = self.staging[slot * self.slot_bytes: slot * self.slot_bytes + nbytes]
        tensor.view(-1).view(torch.uint8).copy_(src, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)
        with self.lock:
            self.busy[slot] = ev
        return slot

    def post_wait(self) -> int:
        """Queue on the current stream only the wait for the slot's flag."""
        slot = self._free_slot()
        stream = torch.cuda.current_stream(self.device)
        with self.lock:
            self.gen[slot] += 1
            target = int(self.gen[slot]) & 0x7FFFFFFF
        cu = self.cu
        (err,) = cu.cuStreamWaitValue32(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(int(self.flags_dptr) + 4 * slot),
            target,
            cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {err}")
        ev = torch.cuda.Event()
        ev.record(stream)
        with self.lock:
            self.busy[slot] = ev
        return slot

    def no_wait(self) -> int:
        """A slot for a receive that the caller waits for on the CPU."""
        slot = self._free_slot()
        with self.lock:
            self.gen[slot] += 1
            self.busy[slot] = None
        return slot

    def wait_filled(self, slot: int, timeout: float | None = None) -> bool:
        target = int(self.gen[slot]) & 0x7FFFFFFF
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.flags_np[slot] != target:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.0001)
        with self.lock:
            self.busy[slot] = "free"
        return True

    def fill(self, slot: int, payload: bytes | memoryview | None) -> None:
        """Write the payload (None: leave the bytes, just release) and set
        the flag."""
        if payload is not None:
            n = len(payload)
            base = slot * self.slot_bytes
            self.staging_np[base:base + n] = np.frombuffer(payload, dtype=np.uint8)
        with self.lock:
            target = int(self.gen[slot]) & 0x7FFFFFFF
        self.flags_np[slot] = target

    def release(self, slot: int) -> None:
        self.fill(slot, None)
