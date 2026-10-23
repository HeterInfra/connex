# SPDX-License-Identifier: Apache-2.0
"""Rollout pool: route rollouts over elastic engines without mixing weight
versions inside a request.

Invariant: a request is stamped with the pool's weight version when it is
admitted and is only ever executed by an instance at exactly that version.
An instance changes version only when it has no request in flight (it is
drained, or its requests were aborted and re-queued first), so no request
ever spans two versions.

* ``add`` an instance: it catches up to the pool's version before it takes
  traffic (the subscriber pulls the latest version from the tree).
* ``update(v)``: the pool version becomes ``v``; new requests wait for an
  instance at ``v``. Each instance stops admitting, drains its in-flight
  requests (they finish at the old version, which is allowed: they started
  there) or, past ``drain_timeout``, aborts them; aborted requests are
  re-queued and restart from scratch at ``v``. ``rolling=k`` updates ``k``
  instances at a time so the rest keep serving.
* An instance that dies mid-request: the request is re-executed on another
  instance at the same version if there is one (with the tokens generated so
  far when the instance supports ``resume``), otherwise it restarts at the
  pool version. An instance that fails its update is removed.
"""
import asyncio
import enum
import logging
import time

log = logging.getLogger("connex.rl.rollout")


class InstanceState(enum.Enum):
    JOINING = "joining"
    SERVING = "serving"
    UPDATING = "updating"
    DEAD = "dead"


class InstanceDead(Exception):
    """Raised by an instance whose engine died; ``partial`` = tokens so far."""

    def __init__(self, msg="", partial=None):
        super().__init__(msg)
        self.partial = partial or []


class Aborted(Exception):
    """The pool aborted the request on this instance (for an update)."""


class RolloutInstance:
    """Interface the pool drives. Implementations: ``FakeInstance`` (tests),
    ``connex.rl.engines.HTTPInstance`` (vLLM / SGLang servers)."""

    name = "instance"
    version = None
    supports_resume = False

    async def generate(self, req, partial=None):   # -> {"tokens": [...], ...}
        raise NotImplementedError

    async def abort_all(self):
        pass

    async def update(self, version):
        raise NotImplementedError

    async def close(self):
        pass


class _Slot:
    def __init__(self, inst):
        self.inst = inst
        self.state = InstanceState.JOINING
        self.inflight = {}          # task -> request record
        self.idle = asyncio.Event()
        self.idle.set()


class RolloutPool:
    def __init__(self, version=0, drain_timeout=None, max_attempts=6):
        self.version = version
        self.drain_timeout = drain_timeout
        self.max_attempts = max_attempts
        self.slots = {}
        self.changed = asyncio.Condition()
        self.stats = {"requests": 0, "completed": 0, "restarted": 0, "resumed": 0,
                      "aborted_for_update": 0, "instance_failures": 0, "failed": 0,
                      "tokens": 0, "updates": []}

    # -- membership -------------------------------------------------------------
    async def add(self, inst):
        """Catch ``inst`` up to the latest weights, then serve. If it came up
        at a newer version than the pool's (the trainer already published the
        next one), it is parked until ``update`` moves the pool there."""
        s = _Slot(inst)
        self.slots[inst.name] = s
        if inst.version != self.version:
            try:
                await inst.update(self.version)
            except Exception as e:
                log.warning("%s failed to catch up to v%s: %s", inst.name, self.version, e)
                self.slots.pop(inst.name, None)
                raise
        await self._set(s, InstanceState.SERVING if inst.version == self.version
                        else InstanceState.JOINING)

    async def remove(self, name, drain=True):
        s = self.slots.get(name)
        if s is None:
            return
        await self._set(s, InstanceState.UPDATING)   # stop admitting
        if drain:
            await s.idle.wait()
        else:
            await self._abort(s)
        self.slots.pop(name, None)
        await s.inst.close()
        async with self.changed:
            self.changed.notify_all()

    def mark_dead(self, name):
        s = self.slots.pop(name, None)
        if s is not None:
            s.state = InstanceState.DEAD
            self.stats["instance_failures"] += 1

    async def _set(self, s, state):
        s.state = state
        async with self.changed:
            self.changed.notify_all()

    def serving(self, version=None):
        v = self.version if version is None else version
        return [s for s in self.slots.values()
                if s.state is InstanceState.SERVING and s.inst.version == v]

    # -- requests ---------------------------------------------------------------
    async def _pick(self, version, exclude=()):
        async with self.changed:
            while True:
                c = [s for s in self.serving(version) if s.inst.name not in exclude] \
                    or self.serving(version)
                if c:
                    return min(c, key=lambda s: len(s.inflight))
                if version != self.version:
                    return None    # nobody left at the old version
                await self.changed.wait()

    async def generate(self, req):
        """Run one rollout; returns {"tokens", "version", "instance", "attempts"}."""
        self.stats["requests"] += 1
        version = self.version
        partial, tried, attempts = [], [], 0
        while attempts < self.max_attempts:
            s = await self._pick(version, exclude=tried)
            if s is None:            # the old version is gone: restart at the new one
                version, partial, tried = self.version, [], []
                self.stats["restarted"] += 1
                continue
            attempts += 1
            task = asyncio.current_task()
            s.inflight[task] = req
            s.idle.clear()
            try:
                use = partial if (partial and s.inst.supports_resume) else None
                if partial and use is None:
                    partial = []
                out = await s.inst.generate(req, partial=use)
                toks = (use or []) + list(out["tokens"])
                self.stats["completed"] += 1
                self.stats["tokens"] += len(toks)
                return {"tokens": toks, "version": version, "instance": s.inst.name,
                        "attempts": attempts}
            except Aborted:
                self.stats["aborted_for_update"] += 1
                version, partial, tried = self.version, [], []
            except InstanceDead as e:
                log.warning("instance %s died during a request", s.inst.name)
                self.mark_dead(s.inst.name)
                tried.append(s.inst.name)
                if s.inst.supports_resume and e.partial:
                    partial = (partial or []) + list(e.partial)
                    self.stats["resumed"] += 1
            finally:
                s.inflight.pop(task, None)
                if not s.inflight:
                    s.idle.set()
        self.stats["failed"] += 1
        raise RuntimeError(f"request failed after {attempts} attempts")

    # -- weight updates ---------------------------------------------------------
    async def _abort(self, s):
        if s.inflight:
            await s.inst.abort_all()
            await s.idle.wait()

    async def _update_one(self, s, version):
        await self._set(s, InstanceState.UPDATING)
        if self.drain_timeout is None:
            await s.idle.wait()
        else:
            try:
                await asyncio.wait_for(s.idle.wait(), self.drain_timeout)
            except asyncio.TimeoutError:
                await self._abort(s)
        try:
            if s.inst.version != version:
                await s.inst.update(version)
            if s.inst.version != version:
                raise RuntimeError(f"came up at v{s.inst.version}, wanted v{version}")
        except Exception as e:
            log.warning("%s failed to update to v%s: %s; removing it", s.inst.name, version, e)
            self.mark_dead(s.inst.name)
            async with self.changed:
                self.changed.notify_all()
            return False
        await self._set(s, InstanceState.SERVING)
        return True

    async def update(self, version, rolling=None):
        """Move the pool to weight ``version``; returns the number of instances
        that updated. ``rolling=k``: at most k instances out of service at a time."""
        t0 = time.perf_counter()
        self.version = version
        async with self.changed:
            self.changed.notify_all()
        slots = list(self.slots.values())
        k = rolling or len(slots) or 1
        ok = 0
        for i in range(0, len(slots), k):
            res = await asyncio.gather(*(self._update_one(s, version) for s in slots[i:i + k]))
            ok += sum(res)
        self.stats["updates"].append({"version": version, "s": time.perf_counter() - t0,
                                      "instances": ok, "of": len(slots)})
        return ok
