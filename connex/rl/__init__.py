# SPDX-License-Identifier: Apache-2.0
"""Elastic, versioned weight synchronisation for RL post-training.

A trainer publishes weight version ``v`` (``WeightPublisher``); rollout
engines pull it with one-sided reads of registered buffers
(``WeightSubscriber``), from the trainer or from engines that already hold
``v`` (tree fan-out), and apply it in one step. The ``WeightRegistry`` tracks
members (rollout endpoints by logical id), the latest version and who holds
it. ``RolloutPool`` routes rollouts so that one request never sees two weight
versions, while instances join, leave, fail and update.

Engine glue: ``connex.rl.vllm_ext`` (a vLLM worker extension) and
``connex.rl.sglang_ext``.
"""
from .weights import WeightManifest, pack_into, unpack, arena_size
from .transport import LocalTransport, ConnexRsTransport, make_transport, TransferError
from .registry import WeightRegistry, RegistryServer, RegistryClient
from .sync import WeightPublisher, WeightSubscriber
from .rollout import RolloutPool, RolloutInstance, InstanceState

__all__ = [
    "WeightManifest", "pack_into", "unpack", "arena_size",
    "LocalTransport", "ConnexRsTransport", "make_transport", "TransferError",
    "WeightRegistry", "RegistryServer", "RegistryClient",
    "WeightPublisher", "WeightSubscriber",
    "RolloutPool", "RolloutInstance", "InstanceState",
]
