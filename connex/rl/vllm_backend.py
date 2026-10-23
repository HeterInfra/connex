# SPDX-License-Identifier: Apache-2.0
"""Connex weight-transfer backend for vLLM's pluggable weight-transfer API
(``vllm.distributed.weight_transfer``).

Worker side (``ConnexWeightTransferEngine``): at ``init_transfer_engine``
each worker joins the Connex weight registry under a logical id and
registers a receive arena with connex-rs. ``update_weights({"version": v})``
pulls ``v`` with one-sided reads from the trainer or from any engine that
already holds it (tree fan-out; a source that dies mid-transfer is replaced
and the pull continues), loads it through ``model.load_weights`` inside
vLLM's layerwise-reload lifecycle, then offers its copy to later pullers.

Trainer side (``ConnexTrainerWeightTransferEngine``): ``send_weights``
packs the ``WeightSource`` into a registered, double-buffered arena,
publishes it as the next version, and drives start / update / finish on the
client (one engine, or ``PoolClient`` for many, where a failed engine is
dropped instead of failing the round).

Selected with ``--weight-transfer-config '{"backend": "connex"}'`` once
``register()`` has run in the engine process (``connex.rl.vllm_plugin`` does
it through vLLM's general-plugin entry point).
"""
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import torch

from vllm.distributed.weight_transfer.base import (TrainerInitInfo,
                                                   TrainerWeightTransferEngine,
                                                   WeightTransferEngine,
                                                   WeightTransferInitInfo,
                                                   WeightTransferUpdateInfo)

from .registry import RegistryClient
from .sync import DEFAULT_CHUNK, WeightPublisher, WeightSubscriber
from .transport import advertise_ip as _advertise_ip, make_transport

BACKEND = "connex"


@dataclass
class ConnexWeightTransferInitInfo(WeightTransferInitInfo):
    registry_url: str
    member: str | None = None          # logical id prefix of this engine
    transport: str = "connex-rs"       # "connex-rs" | "local" (tests)
    chunk_bytes: int = DEFAULT_CHUNK


@dataclass
class ConnexWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    version: int | None = None         # None: latest


class ConnexWeightTransferEngine(
        WeightTransferEngine[ConnexWeightTransferInitInfo, ConnexWeightTransferUpdateInfo]):
    init_info_cls = ConnexWeightTransferInitInfo
    update_info_cls = ConnexWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(self, config, vllm_config, device, model):
        super().__init__(config, vllm_config, device, model)
        self.sub = None
        self.transport = None
        self.last_stats = {}

    def init_transfer_engine(self, init_info):
        pc = self.parallel_config
        rank = getattr(pc, "rank", 0)
        base = init_info.member or f"{socket.gethostname()}-{os.getpid()}"
        member = f"{base}/r{rank}"
        if self.transport is None:
            if init_info.transport == "local":
                self.transport = make_transport("local", name=member)
            else:
                self.transport = make_transport(
                    "connex-rs", lid=("connex-rl", 1, rank, abs(hash(base)) % (1 << 31)),
                    advertise_ip=_advertise_ip())
        self.sub = WeightSubscriber(RegistryClient(init_info.registry_url), self.transport,
                                    member, device=self.device,
                                    chunk_bytes=init_info.chunk_bytes)

    def start_weight_update(self):
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
        initialize_layerwise_reload(self.model)

    def finish_weight_update(self):
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload
        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info):
        if self.sub is None:
            raise RuntimeError("connex weight transfer not initialized")
        m = self.sub.pull(update_info.version)
        self.model.load_weights(self.sub.tensors())
        torch.accelerator.synchronize()
        self.sub.commit(m.version)
        self.last_stats = dict(self.sub.stats)

    def shutdown(self):
        if self.sub is not None:
            self.sub.close()
            self.sub = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None


@dataclass
class ConnexTrainerInitInfo(TrainerInitInfo):
    backend: ClassVar[str] = BACKEND
    registry_url: str = ""
    transport: str = "connex-rs"
    device: str = "cuda"
    chunk_bytes: int = DEFAULT_CHUNK
    member: str | None = None          # worker-side member prefix (single client)
    buffers: int = 2                   # publisher arenas (1 halves trainer memory)


class ConnexTrainerWeightTransferEngine(TrainerWeightTransferEngine[ConnexTrainerInitInfo]):
    init_info_cls = ConnexTrainerInitInfo

    def __init__(self, *, client, source=None, is_sender=True, publisher=None):
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.pub = publisher
        self.version = 0

    @classmethod
    def trainer_init(cls, init_info, *, client, source=None):
        if source is None:
            raise ValueError("connex trainer weight transfer requires a WeightSource")
        pub = None
        if init_info.is_sender:
            t = make_transport(init_info.transport, name="trainer",
                               lid=("connex-rl", 0, 0, 0), advertise_ip=_advertise_ip())
            pub = WeightPublisher(RegistryClient(init_info.registry_url), t,
                                  device=init_info.device, buffers=init_info.buffers)
            client.init_weight_transfer_engine(asdict(ConnexWeightTransferInitInfo(
                registry_url=init_info.registry_url, member=init_info.member,
                transport=init_info.transport, chunk_bytes=init_info.chunk_bytes)))
        return cls(client=client, source=source, is_sender=init_info.is_sender, publisher=pub)

    def publish(self):
        """Pack and publish the next version without driving any engine."""
        named = list(self.source)        # every rank joins the gather
        if not self.is_sender:
            return None
        self.version = self.pub.publish(named)
        return self.version

    def send_weights(self):
        v = self.publish()
        if not self.is_sender:
            return
        self.client.start_weight_update()
        self.client.update_weights({"version": v})
        self.client.finish_weight_update(weight_version=str(v))

    def shutdown(self):
        if self.pub is not None:
            self.pub.t.close()


class PoolClient:
    """``VLLMWeightSyncClient`` over many engines: calls them in parallel and
    drops the ones that fail (dead or gone) instead of failing the round."""

    def __init__(self, clients: dict[str, Any], on_failure=None):
        self.clients = dict(clients)
        self.on_failure = on_failure
        self.failed = {}

    def _all(self, fn):
        with ThreadPoolExecutor(max_workers=max(1, len(self.clients))) as ex:
            futs = {n: ex.submit(fn, c) for n, c in self.clients.items()}
        for n, f in futs.items():
            try:
                f.result()
            except Exception as e:
                self.failed[n] = repr(e)
                self.clients.pop(n, None)
                if self.on_failure:
                    self.on_failure(n, e)
        if not self.clients:
            raise RuntimeError(f"every engine failed: {self.failed}")

    def add(self, name, client):
        self.clients[name] = client

    def init_weight_transfer_engine(self, init_info):
        self._all(lambda c: c.init_weight_transfer_engine(
            dict(init_info, member=init_info.get("member") or None)))

    def start_weight_update(self):
        self._all(lambda c: c.start_weight_update())

    def update_weights(self, update_info):
        self._all(lambda c: c.update_weights(update_info))

    def finish_weight_update(self, weight_version=None):
        self._all(lambda c: c.finish_weight_update(weight_version))


_registered = False


def register():
    """Register the backend with vLLM's factories (idempotent)."""
    global _registered
    if _registered:
        return
    from vllm.distributed.weight_transfer.factory import (WeightTransferEngineFactory,
                                                          WeightTransferTrainerFactory)
    for fac, cls in ((WeightTransferEngineFactory, ConnexWeightTransferEngine),
                     (WeightTransferTrainerFactory, ConnexTrainerWeightTransferEngine)):
        if BACKEND not in fac._registry:
            fac.register_engine(BACKEND, cls)
    _registered = True
