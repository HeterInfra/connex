# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``--pp-group-cls connex.adapters.vllm.pp.group.ConnexPPGroup``

A vLLM ``GroupCoordinator`` for the pipeline-parallel group whose
inter-stage traffic goes over connex-rs (UCX; GPUDirect RDMA where the NIC
allows) instead of NCCL:

* ``isend_tensor_dict`` / ``irecv_tensor_dict``: the hidden states and
  residual passed from stage to stage;
* ``make_sibling_device_group`` / ``sibling_broadcast``: the V2 model
  runner's broadcast of the sampled tokens from the last stage to the other
  stages (received asynchronously, see :mod:`hostwait`).

Two modes:

* **in-process** (stock vLLM PP: all stages are ranks of one
  torch.distributed world, ``--pipeline-parallel-size N``): every rank opens
  a link and the links exchange URLs over the group's CPU (gloo) group.
  Nothing else changes; this is the drop-in "PP without NCCL" mode.
* **external** (``--pipeline-stage-rank k``: each stage is its own vLLM
  instance, driven by :mod:`connex.adapters.vllm.pp.executor`): the group
  reports stage k of N; the peers of each step come from the step's route,
  which the executor attaches to the SchedulerOutput and the worker
  (:mod:`connex.adapters.vllm.pp.worker`) hands to :meth:`begin_step`.
  With TP > 1, TP rank 0 is the stage's endpoint and shares what it
  receives with the other TP ranks over the TP group.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group

from connex.adapters.vllm.pp.link import StageLink

log = logging.getLogger("vllm.connex_pp")

MODEL_ID = os.getenv("CONNEX_PP_MODEL_ID", "cxpp")


class ConnexPPGroup(GroupCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        # for placeholder tensors of steps without upstream data
        self._max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        model_config = vllm_config.model_config
        self._hidden = model_config.get_hidden_size() if model_config else 0
        self._dtype = model_config.dtype if model_config else torch.bfloat16
        # Ring for the steps in flight: (pp + 1) steps of hidden states +
        # residual at the largest batch.
        elem = torch.empty(0, dtype=self._dtype).element_size()
        step_bytes = self._max_batched * max(self._hidden, 1) * 2 * elem
        ring_mb = int(os.getenv("CONNEX_PP_RING_MB", "0")) or max(
            64, -(-(parallel_config.pipeline_parallel_size + 1) * step_bytes // (1 << 20)))
        self.stage_rank = getattr(parallel_config, "pipeline_stage_rank", None)
        self.external = self.stage_rank is not None
        self.tp = get_tp_group()
        self.tp_rank = self.tp.rank_in_group
        self.tp_size = self.tp.world_size
        self._step_key: tuple = (0, 0)
        self._send_count = 0
        self._recv_count = 0
        self._bcast_idx = 0
        self._route: list[str] = []
        self._last_shapes: dict[str, tuple] | None = None
        self._tp_sibling = None
        self._waiter = None
        self._pending_slots: dict[tuple, int] = {}
        self._local_step = self.stage_rank is not None
        # While set (warmup), receives wait on the CPU instead of queueing a
        # stream wait: a stream wait that only a Python thread of this
        # process can release deadlocks against CUDA calls that hold the GIL
        # and synchronize the device, such as a first kernel (module) load,
        # and warmup is where the kernels load.
        self.blocking = False
        self.link: StageLink | None = None
        self.timing = {"send_s": 0.0, "recv_wait_s": 0.0, "steps": 0}

        if self.external:
            num_stages = parallel_config.pipeline_parallel_size
            # Stage k of N; the other stages are not ranks of this world.
            self.rank_in_group = self.stage_rank
            self.world_size = num_stages
            self.ranks = [self.rank if i == self.stage_rank else -1 - i
                          for i in range(num_stages)]
            self.is_endpoint = self.tp_rank == 0
            replica = int(os.getenv("CONNEX_PP_REPLICA", "0"))
            port = int(os.getenv("CONNEX_PP_DATA_PORT", "0"))
            if self.is_endpoint:
                self.link = StageLink(MODEL_ID, self.stage_rank, 0, replica, self.device,
                                      port=port, ring_mb=ring_mb)
                self.link.abort_callbacks.append(self._on_abort)
            log.info("Connex PP group: stage %d of %d, TP rank %d%s", self.stage_rank,
                     num_stages, self.tp_rank,
                     f", endpoint {self.link.url}" if self.link else "")
        else:
            # In-process PP: one link per rank, URLs exchanged over gloo.
            self.is_endpoint = True
            if self.world_size > 1:
                self.link = StageLink(MODEL_ID, self.rank_in_group, self.tp_rank, 0,
                                      self.device, ring_mb=ring_mb)
                urls: list[Any] = [None] * self.world_size
                torch.distributed.all_gather_object(urls, self.link.url, group=self.cpu_group)
                self._route = list(urls)
                # One connection per pair, opened by the lower rank (two
                # simultaneous connects would replace each other).
                for i, url in enumerate(urls):
                    if i > self.rank_in_group:
                        self.link.connect(url)
                for i in range(self.world_size):
                    if i != self.rank_in_group:
                        peer = (MODEL_ID, i, self.tp_rank, 0)
                        if not self.link.wait_up(peer):
                            raise RuntimeError(f"connex-pp: no link to PP rank {i}")
                torch.distributed.barrier(group=self.cpu_group)
                log.info("Connex PP group (in-process): rank %d of %d at %s",
                         self.rank_in_group, self.world_size, self.link.url)

    # ------------------------------------------------------------- per step
    def begin_step(self, meta: dict | None) -> None:
        """External mode: the step's (epoch, seq) and the data URLs of the
        stages on its route (set by the worker before execute_model)."""
        if meta is None:
            # A step that did not come from the pipeline (warmup / dummy run
            # of this stage alone): no inter-stage traffic.
            self._local_step = self.external
            return
        self._local_step = False
        epoch, seq, route = meta["epoch"], meta["seq"], meta["route"]
        if self.link is not None and epoch > self.link.min_epoch:
            self.link.abort(epoch)
        self._step_key = (epoch, seq)
        self._route = route
        self._bcast_idx = 0
        self._max_tokens = meta.get("max_tokens", 1)

    # Keys are (epoch, seq, idx): idx -1 for the hidden states of a step,
    # 0.. for the sampled-token broadcasts within it.
    def _key(self, sending: bool) -> tuple:
        if self.external:
            return (*self._step_key, -1)
        # In-process: the k-th send of a rank pairs with the k-th receive of
        # its successor.
        if sending:
            self._send_count += 1
            return (0, self._send_count, -1)
        self._recv_count += 1
        return (0, self._recv_count, -1)

    def _on_abort(self, epoch: int) -> None:
        # Let every stream that still waits for sampled tokens of an older
        # epoch go (the requests are recomputed, the bytes are not used).
        if self._waiter is None:
            return
        for key in [k for k in self._pending_slots if k[0] < epoch]:
            self._waiter.release(self._pending_slots.pop(key))

    # --------------------------------------------------- inter-stage tensors
    def isend_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None,
                          all_gather_tensors=None):
        if self.world_size <= 1:
            return []
        if self._local_step:
            return []
        key = self._key(sending=True)
        if not self.is_endpoint:
            return []
        rank = (self.rank_in_group + 1) % self.world_size if dst is None else dst
        tensors = {k: v for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}
        meta = {k: v for k, v in tensor_dict.items() if not isinstance(v, torch.Tensor)}
        t0 = time.perf_counter()
        self.link.send_tensors(self._route[rank], key, tensors, meta or None)
        self.timing["send_s"] += time.perf_counter() - t0
        return []

    def send_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None,
                         all_gather_tensors=None):
        self.isend_tensor_dict(tensor_dict, dst, all_gather_group, all_gather_tensors)
        return None

    def irecv_tensor_dict(self, src=None, all_gather_group=None, all_gather_tensors=None):
        if self.world_size <= 1:
            return None, [], []
        if self._local_step:
            return self._aborted_dict(), [], []
        key = self._key(sending=False)
        td = None
        if self.is_endpoint:
            t0 = time.perf_counter()
            got = self.link.recv_tensors(key, blocking=self.blocking)
            self.timing["recv_wait_s"] += time.perf_counter() - t0
            self.timing["steps"] += 1
            if got is None:
                td = self._aborted_dict()
            else:
                tensors, meta = got
                td = dict(tensors)
                if meta:
                    td.update(meta)
                self._last_shapes = {k: (v.shape[1:], v.dtype) for k, v in tensors.items()}
        if self.external and self.tp_size > 1:
            td = self.tp.broadcast_tensor_dict(td, src=0)
        return td, [], []

    def recv_tensor_dict(self, src=None, all_gather_group=None, all_gather_tensors=None):
        return self.irecv_tensor_dict(src, all_gather_group, all_gather_tensors)[0]

    def _aborted_dict(self) -> dict:
        """Zeros of the expected shapes for a step whose upstream is gone
        or that has none (warmup): its values are never used."""
        n = getattr(self, "_max_tokens", None) or self._max_batched
        shapes = self._last_shapes or {"hidden_states": ((self._hidden,), self._dtype),
                                       "residual": ((self._hidden,), self._dtype)}
        return {k: torch.zeros((n, *s), dtype=dt, device=self.device)
                for k, (s, dt) in shapes.items()}

    # ------------------------------------------- sampled-token broadcast (V2)
    def make_sibling_device_group(self, group_desc: str | None = None):
        if not self.external and os.getenv("CONNEX_PP_SAMPLED_NCCL") == "1":
            self._sampled_nccl = True
            return super().make_sibling_device_group(group_desc)
        if self.external and self.tp_size > 1:
            # Rank 0 receives from the last stage; the other TP ranks get the
            # tokens from it over a side communicator of the TP group.
            self._tp_sibling = self.tp.make_sibling_device_group(group_desc)
        return None

    def sibling_broadcast(self, tensor: torch.Tensor, src: int, group) -> None:
        if getattr(self, "_sampled_nccl", False):
            return super().sibling_broadcast(tensor, src, group)
        if self._local_step:
            return  # the receiver keeps whatever the tensor holds
        if self.is_last_rank:
            self._send_sampled(tensor)
        else:
            self._recv_sampled(tensor)

    def _bkey(self) -> tuple:
        if self.external:
            k = (*self._step_key, self._bcast_idx)
            self._bcast_idx += 1
            return k
        self._bcast_idx += 1
        return (0, self._bcast_idx, 0)

    def _send_sampled(self, tensor: torch.Tensor) -> None:
        key = self._bkey()
        if not self.is_endpoint:
            return
        stream = torch.cuda.current_stream(self.device)
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)
        num = self.world_size - 1
        peers = self._route[:num]

        def payload(host=host):
            return host.numpy().tobytes()

        for url in peers:
            self.link.send_bytes(url, key, payload, event=ev)

    def _recv_sampled(self, tensor: torch.Tensor) -> None:
        key = self._bkey()
        if self.is_endpoint and self.blocking:
            data = self.link.take(key)
            if data is not None:
                tensor.view(-1).view(torch.uint8).copy_(
                    torch.frombuffer(bytearray(data), dtype=torch.uint8))
        elif self.is_endpoint:
            if self._waiter is None:
                from connex.adapters.vllm.pp.hostwait import HostFlagWaiter

                self._waiter = HostFlagWaiter(self.device)
            slot = self._waiter.post(tensor)
            self._pending_slots[key] = slot
            # The filler releases the slot at once if the key's epoch was
            # already aborted.
            self._watch(key, slot)
        if self.external and self.tp_size > 1:
            torch.distributed.broadcast(tensor, src=self.tp.ranks[0], group=self._tp_sibling)

    def _watch(self, key: tuple, slot: int) -> None:
        """Fill ``slot`` when the bytes for ``key`` arrive (a waiter thread)."""
        _SampledFiller.get(self).add(key, slot)

    def destroy(self):
        if self.link is not None:
            self.link.close()
        super().destroy()


class _SampledFiller:
    """One thread per group that moves arriving sampled-token payloads into
    the waiter's staging slots, in the order the receives were posted."""

    _inst: dict[int, "_SampledFiller"] = {}

    @classmethod
    def get(cls, group: ConnexPPGroup) -> "_SampledFiller":
        inst = cls._inst.get(id(group))
        if inst is None:
            inst = cls._inst[id(group)] = cls(group)
        return inst

    def __init__(self, group: ConnexPPGroup):
        import queue
        import threading

        self.group = group
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, name="cx-pp-sampled", daemon=True).start()

    def add(self, key: tuple, slot: int) -> None:
        self.q.put((key, slot))

    def _loop(self) -> None:
        g = self.group
        while True:
            key, slot = self.q.get()
            data = g.link.take(key)   # None once the key's epoch is aborted
            if g._pending_slots.get(key) != slot:
                continue  # already released by an abort
            g._pending_slots.pop(key, None)
            g._waiter.fill(slot, data)
