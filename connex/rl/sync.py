# SPDX-License-Identifier: Apache-2.0
"""Trainer-side publisher and engine-side subscriber.

Publish (trainer): pack the named tensors of version ``v`` into a registered
arena (double-buffered, so readers of ``v-1`` are never overwritten by
``v+1``), then announce ``v`` with its manifest in the registry.

Pull (engine): ask the registry for a source of ``v``, read the arena in
chunks with one-sided reads, and on a read error (source died or left)
report it, get another source and continue with the remaining chunks. The
trainer always holds ``v``, so a pull completes as long as the trainer is
alive. After the last chunk the subscriber hands the arena to ``apply``
(the engine's ``load_weights``) in one call, then registers as a holder of
``v`` so later pullers can read from it (tree fan-out).
"""
import logging
import threading
import time

import torch

from .registry import PUBLISHER
from .transport import TransferError
from .weights import WeightManifest, pack_into, unpack

log = logging.getLogger("connex.rl")

DEFAULT_CHUNK = 64 << 20


class WeightPublisher:
    def __init__(self, registry, transport, device="cpu", buffers=2, idle_timeout=30.0):
        self.reg = registry
        self.t = transport
        self.device = torch.device(device)
        self.nbuf = buffers
        self.arenas = []        # [(tensor, version)]
        self.manifest = None
        self.version = 0
        self.idle_timeout = idle_timeout

    def publish(self, named, version=None):
        """``named``: list of (name, tensor). Returns the new version."""
        named = list(named)
        v = version if version is not None else self.version + 1
        m = WeightManifest.build(v, named)
        if len(self.arenas) < self.nbuf:
            slot = len(self.arenas)
        else:   # reuse the slot of the oldest version
            slot = min(range(self.nbuf), key=lambda i: self.arenas[i][1])
        if slot < len(self.arenas):
            a, old = self.arenas[slot]
            if old is not None:
                ok = self.reg.wait_idle(version=old, member=PUBLISHER,
                                        timeout=self.idle_timeout)
                if not (ok or {}).get("ok", True):
                    log.warning("publisher: readers of v%s still active after %.0fs; "
                                "overwriting its arena (they may read torn weights)",
                                old, self.idle_timeout)
            if a.numel() < m.total:
                self.t.unregister(a)
                a = torch.empty(m.total, dtype=torch.uint8, device=self.device)
                self.t.register(a)
                self.arenas[slot][0] = a
        else:
            a = torch.empty(m.total, dtype=torch.uint8, device=self.device)
            self.t.register(a)
            self.arenas.append([a, None])
        pack_into(a, m, named)
        if a.is_cuda:
            torch.cuda.synchronize(a.device)
        self.arenas[slot][1] = v
        self.reg.publish(version=v, manifest=m.to_dict(), url=self.t.url, addr=a.data_ptr())
        self.version, self.manifest = v, m
        return v


class WeightSubscriber:
    """One per engine worker. ``member`` is its logical id in the registry."""

    def __init__(self, registry, transport, member, device="cpu", chunk_bytes=DEFAULT_CHUNK,
                 max_source_failures=16, heartbeat_s=2.0, window=4):
        self.reg = registry
        self.t = transport
        self.member = member
        self.device = torch.device(device)
        self.chunk = chunk_bytes
        self.window = window       # chunks in flight per source
        self.max_fail = max_source_failures
        self.arena = None
        self.manifest = None
        self.version = None
        self.stats = {}
        self.reg.join(member=member, url=transport.url)
        self._stop = threading.Event()
        if heartbeat_s:
            threading.Thread(target=self._hb, args=(heartbeat_s,), daemon=True).start()

    def _hb(self, every):
        while not self._stop.wait(every):
            try:
                if not self.reg.heartbeat(member=self.member)["ok"]:
                    self.reg.join(member=self.member, url=self.t.url)
            except Exception:  # registry restarting: keep trying
                pass

    def latest(self):
        lt = self.reg.get_latest()
        return None if lt is None else lt["version"]

    def pull(self, version=None):
        """Fetch the latest version (at least ``version``) into the local arena; returns
        the manifest. Does not touch the model."""
        lt = self.reg.get_latest()
        if lt is None:
            raise RuntimeError("no weight version published yet")
        if version is not None and version > lt["version"]:
            lt = self.reg.get_latest(wait_newer_than=version - 1, timeout=60.0)
            if lt["version"] < version:
                raise RuntimeError(f"version {version} not published (latest {lt['version']})")
        # holders keep only the last two versions: an older request is served
        # with the latest one, which the caller sees in the returned manifest
        version, md = lt["version"], lt["manifest"]
        m = WeightManifest.from_dict(md)
        t0 = time.perf_counter()
        # stop serving the old copy before overwriting it
        if self.arena is not None:
            self.reg.retract(member=self.member)
        self.manifest = None      # the arena is about to be overwritten
        for attempt in range(3):
            stale, failures, sources = self._read(version, m)
            if not stale:
                break
            log.warning("%s: a source of v%d overwrote its copy during the read; "
                        "reading again", self.member, version)
        else:
            raise RuntimeError(f"v{version}: sources kept changing during the read")
        if self.arena.is_cuda:
            torch.cuda.synchronize(self.arena.device)
        self.manifest = m
        self.stats = {"version": version, "bytes": m.total, "pull_s": time.perf_counter() - t0,
                      "sources": sources, "source_failures": failures}
        return m

    def _release(self, version, source):
        r = self.reg.release(version=version, source=source, member=self.member)
        return bool((r or {}).get("stale"))

    def _read(self, version, m):
        """Read ``version`` into the arena; returns (stale, failures, sources).
        ``stale``: a source overwrote its copy while assigned to us."""
        if self.arena is None or self.arena.numel() < m.total:
            if self.arena is not None:
                self.t.unregister(self.arena)
            self.arena = torch.empty(m.total, dtype=torch.uint8, device=self.device)
            self.t.register(self.arena)
        offs = list(range(0, m.total, self.chunk))
        stale = False
        failures, bad, sources = 0, set(), []
        src = None
        i = 0
        try:
            while i < len(offs):
                if src is None:
                    src = self.reg.assign(version=version, member=self.member, exclude=sorted(bad))
                    if src is None:
                        raise RuntimeError(f"no source holds version {version}")
                    sources.append(src["member"])
                ops = [(src["addr"] + off, off, min(self.chunk, m.total - off))
                       for off in offs[i:i + self.window]]
                try:
                    self.t.read_batch(src["url"], ops, self.arena)
                    i += len(ops)
                except TransferError as e:
                    failures += 1
                    log.warning("%s: read of v%d from %s failed (%s); switching source",
                                self.member, version, src["member"], e)
                    self.reg.report_failure(version=version, source=src["member"],
                                            reporter=self.member)
                    stale |= self._release(version, src["member"])
                    if src["member"] != PUBLISHER:
                        bad.add(src["member"])
                    src = None
                    if failures > self.max_fail:
                        raise
        except BaseException:
            # give back the assignment we still hold (e.g. too many failures)
            if src is not None:
                try:
                    self._release(version, src["member"])
                except Exception:  # noqa: BLE001
                    pass
            raise
        if src is not None:
            stale |= self._release(version, src["member"])
        return stale, failures, sources

    def tensors(self):
        return unpack(self.arena, self.manifest)

    def commit(self, version):
        """Mark ``version`` applied and offer it to other pullers."""
        self.version = version
        self.reg.done(version=version, member=self.member, url=self.t.url,
                      addr=self.arena.data_ptr())

    def sync(self, apply, version=None):
        """pull + apply(iterable of (name, tensor)) + commit. Returns stats."""
        m = self.pull(version)
        t0 = time.perf_counter()
        apply(self.tensors())
        if self.arena.is_cuda:
            torch.cuda.synchronize(self.arena.device)
        self.stats["apply_s"] = time.perf_counter() - t0
        self.commit(m.version)
        return self.stats

    def close(self, leave=True):
        self._stop.set()
        if leave:
            try:
                self.reg.leave(member=self.member)
            except Exception:
                pass
