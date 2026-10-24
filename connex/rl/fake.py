# SPDX-License-Identifier: Apache-2.0
"""Fake rollout engine for CPU tests and the churn simulation.

Its "model" is a weight tensor pulled through a real ``WeightSubscriber``;
the token sequence of a request is a pure function of (prompt, weights), so
a test can check that every token of a response came from one version.
"""
import asyncio

import torch

from .rollout import Aborted, InstanceDead, RolloutInstance


def token(seed, i, w):
    return int((seed * 1000003 + i * 7919 + int(w)) % 50000)


class FakeInstance(RolloutInstance):
    supports_resume = True

    def __init__(self, name, subscriber=None, tpot_s=0.001, update_s=0.0):
        self.name = name
        self.sub = subscriber
        self.version = None
        self.w = None
        self.tpot_s = tpot_s
        self.update_s = update_s
        self.dead = False
        self._abort = asyncio.Event()
        self.served = 0

    def kill(self):
        self.dead = True

    async def generate(self, req, partial=None):
        if self.dead:
            raise InstanceDead("down")
        start = len(partial or [])
        w, out = self.w, []
        for i in range(start, req["max_tokens"]):
            await asyncio.sleep(self.tpot_s)
            if self.dead:
                raise InstanceDead("died", partial=out)
            if self._abort.is_set():
                raise Aborted()
            out.append(token(req["seed"], i, w))
        self.served += 1
        return {"tokens": out}

    async def abort_all(self):
        self._abort.set()
        await asyncio.sleep(0)
        await asyncio.sleep(self.tpot_s * 2)
        self._abort.clear()

    async def update(self, version):
        if self.dead:
            raise InstanceDead("down")
        await asyncio.sleep(self.update_s)
        if self.sub is not None:
            def apply(named):
                for _, t in named:
                    self.w = float(t.view(-1)[0])
                    break
            st = await asyncio.to_thread(self.sub.sync, apply, version)
            self.version = st["version"]
        else:
            self.w, self.version = float(version), version


def expected(req, w):
    return [token(req["seed"], i, w) for i in range(req["max_tokens"])]


def weights_for(version, n=4096):
    return [("w", torch.full((n,), float(version))), ("b", torch.arange(16.0) + version)]
