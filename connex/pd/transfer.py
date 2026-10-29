# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Pull-based KV block transfer between a prefill and a decode instance.

Engine neutral. A prefill worker runs a ``KVProducer``: after a request's
prefill it *holds* the request's KV blocks under a transfer id and serves a
small control channel (ZeroMQ REP). A decode worker runs a ``KVConsumer``:
it *acquires* the transfer (learning the producer's block ids and, once per
peer, its KV layout and data-plane URL), reads the blocks with one-sided
reads into its own blocks, and later *releases* them.

Why hold until release instead of freeing after the first read: endpoint
mobility. If the decode instance dies mid-generation, another decode
instance can acquire the same transfer and pull the prompt KV again instead
of re-running prefill. A hold is a lease: every acquire renews it for
``retain_s`` seconds and an expired hold is freed, so a dead decode never
strands prefill memory. ``release_after_pull`` frees right after the first
successful read (the conventional behaviour, least prefill memory).

Failure semantics. Every failure is reported per request, never raised into
the engine: control-channel timeout (prefill dead or unreachable), unknown
transfer (hold expired or prefill restarted), layout mismatch, or a failed
read (with connex-rs a dead peer fails the read in milliseconds). The engine
adapter turns a failed request into load errors for its local blocks, and the
engine recomputes or fails the request; the PD router then retries.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import msgpack
import zmq

from connex.pd.dataplane import ZCOPY_THRESH
from connex.pd.host_staging import block_ranges, register_with_staging
from connex.pd.kv_layout import TPSource, TensorLayout, head_read_ops, read_ops, register_spans

log = logging.getLogger("connex.pd")

PROTO_VERSION = 1


def _pack(o) -> bytes:
    return msgpack.packb(o, use_bin_type=True)


def _unpack(b: bytes):
    return msgpack.unpackb(b, raw=False)


@dataclass
class _Hold:
    req_id: str
    blocks: list[int]
    deadline: float
    acquired: int = 0
    released_by: set = field(default_factory=set)
    ready: object = None    # host staging: event of the device -> mirror copy
    owner: object = None    # ``who`` of the latest acquire (single-reader release check)


class KVProducer:
    """Prefill side: holds KV blocks of finished prefills and serves them."""

    def __init__(self, plane, layouts: Sequence[TensorLayout], bind_ip: str = "0.0.0.0",
                 port: int = 0, retain_s: float = 60.0, engine_id: str = "",
                 block_size: int = 0, register: bool = True, device: str = ""):
        self.plane = plane
        self.layouts = list(layouts)
        self.retain_s = retain_s
        self.engine_id = engine_id
        self.block_size = block_size
        self.device = device
        self._holds: dict[str, _Hold] = {}
        self._released: set[str] = set()   # transfer ids released before their hold arrived
        self._finished: set[str] = set()   # req ids whose blocks may be freed
        self._lock = threading.Lock()
        self.stats = {"holds": 0, "acquires": 0, "releases": 0, "expired": 0, "unknown": 0}
        # GPUDirect when the plane can register the KV cache, else (or with
        # CONNEX_PD_HOST_STAGING=1) a pinned host mirror (host_staging.py)
        if register:
            self._staging = register_with_staging(plane, register_spans(self.layouts),
                                                  "KV producer")
        else:
            self._staging = getattr(plane, "_connex_staging", None)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        if port:
            self._sock.bind(f"tcp://{bind_ip}:{port}")
            self.port = port
        else:
            self.port = self._sock.bind_to_random_port(f"tcp://{bind_ip}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="connex-pd-producer", daemon=True)
        self._thread.start()

    # -- engine side ------------------------------------------------------------
    def hold(self, req_id: str, transfer_id: str, blocks: Sequence[int]) -> None:
        with self._lock:
            if transfer_id in self._released:
                self._released.discard(transfer_id)
                self._finished.add(req_id)
                return
        ready = None
        if self._staging is not None:   # queued after the prefill on the engine stream
            ready = self._staging.stage_out(block_ranges(self.layouts, blocks))
        with self._lock:
            self._holds[transfer_id] = _Hold(req_id, list(blocks), time.monotonic() + self.retain_s,
                                             ready=ready)
            self.stats["holds"] += 1

    def drop(self, transfer_id: str) -> None:
        """The engine freed the request itself (abort): forget the hold."""
        with self._lock:
            self._holds.pop(transfer_id, None)

    def take_finished(self) -> set[str]:
        """Request ids whose blocks can be freed now (released or expired)."""
        now = time.monotonic()
        with self._lock:
            for tid, h in list(self._holds.items()):
                if h.deadline < now:
                    del self._holds[tid]
                    self._finished.add(h.req_id)
                    self.stats["expired"] += 1
                    log.info("KV hold %s (req %s) expired after %d acquires", tid, h.req_id, h.acquired)
            out, self._finished = self._finished, set()
        return out

    def held(self) -> int:
        with self._lock:
            return len(self._holds)

    # -- control channel -------------------------------------------------------
    def _handle(self, m: dict) -> dict:
        op = m.get("op")
        if op == "meta":
            return {"ok": True, "v": PROTO_VERSION, "url": self.plane.local_url,
                    "backend": self.plane.name, "engine_id": self.engine_id,
                    "block_size": self.block_size, "device": self.device,
                    "layouts": [(self._staging.translate(lo) if self._staging else lo).to_dict()
                                for lo in self.layouts]}
        tid = m.get("tid")
        with self._lock:
            h = self._holds.get(tid)
            if op == "acquire":
                if h is None:
                    self.stats["unknown"] += 1
                    return {"ok": False, "why": "unknown"}
                h.acquired += 1
                if m.get("who"):
                    h.owner = m.get("who")
                h.deadline = time.monotonic() + self.retain_s
                self.stats["acquires"] += 1
                return {"ok": True, "blocks": h.blocks, "_wait": h.ready}
            if op == "release":
                self.stats["releases"] += 1
                # several decode TP ranks may read this rank (decode TP larger
                # than prefill TP): free once all `of` of them released
                of = int(m.get("of") or 1)
                who = m.get("who")
                if h is not None and of <= 1 and who and h.owner and who != h.owner:
                    # stale release from a decode instance the request already
                    # moved away from: the hold now belongs to the newer reader,
                    # freeing it would let the engine reuse blocks mid-read
                    self.stats["stale_releases"] = self.stats.get("stale_releases", 0) + 1
                    return {"ok": True, "stale": True}
                if h is not None and of > 1:
                    h.released_by.add(m.get("who"))
                    if len(h.released_by) < of:
                        return {"ok": True, "pending": of - len(h.released_by)}
                if h is None:
                    self._released.add(tid)   # hold not seen yet (or already gone)
                    if len(self._released) > 100000:
                        self._released.clear()
                else:
                    del self._holds[tid]
                    self._finished.add(h.req_id)
                return {"ok": True}
            if op == "ping":
                return {"ok": True, "held": len(self._holds)}
        return {"ok": False, "why": f"bad op {op!r}"}

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(100):
                continue
            try:
                msg = _unpack(self._sock.recv())
                rep = self._handle(msg)
                ev = rep.pop("_wait", None)
                if ev is not None:
                    ev.synchronize()   # staged copy done before the reader reads
            except Exception as e:  # noqa: BLE001
                rep = {"ok": False, "why": f"error: {e}"}
            self._sock.send(_pack(rep))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close(0)


@dataclass
class PullResult:
    req_id: str
    ok: bool
    why: str = ""
    nbytes: int = 0
    seconds: float = 0.0
    failed_blocks: list[int] = field(default_factory=list)


class _PeerMeta:
    def __init__(self, d: dict):
        self.url = d["url"]
        self.backend = d.get("backend")
        self.engine_id = d.get("engine_id")
        self.block_size = d.get("block_size", 0)
        self.device = d.get("device", "")
        self.layouts = [TensorLayout.from_dict(x) for x in d["layouts"]]


class KVConsumer:
    """Decode side: pulls held KV blocks from producers, asynchronously."""

    def __init__(self, plane, layouts: Sequence[TensorLayout], block_size: int = 0,
                 workers: int = 4, ctrl_timeout_ms: int = 1000, register: bool = True,
                 release_after_pull: bool = False, acquire_wait_s: float = 2.0,
                 device: str = "", who: str = ""):
        self.plane = plane
        self.device = device
        self.who = who or f"{os.getpid()}@{id(self):x}"   # identifies this rank in releases
        # CONNEX_PD_INJECT_PULL_DELAY_S: test-only delay before each pull, to
        # kill a prefill instance while a transfer is pending
        self.inject_delay_s = float(os.getenv("CONNEX_PD_INJECT_PULL_DELAY_S", "0") or 0)
        self.acquire_wait_s = acquire_wait_s
        self.layouts = list(layouts)
        self.block_size = block_size
        self.ctrl_timeout_ms = ctrl_timeout_ms
        self.release_after_pull = release_after_pull
        self._spans = register_spans(self.layouts)
        if register:
            self._staging = register_with_staging(plane, self._spans, "KV consumer")
        else:
            self._staging = getattr(plane, "_connex_staging", None)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="connex-pd-pull")
        self._meta: dict[str, _PeerMeta] = {}
        self._lock = threading.Lock()
        self._done: list[PullResult] = []
        self._ctx = zmq.Context.instance()
        self.stats = {"pulls": 0, "failed": 0, "bytes": 0, "seconds": 0.0}

    def _call(self, addr: str, msg: dict) -> Optional[dict]:
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, self.ctrl_timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self.ctrl_timeout_ms)
        try:
            s.connect(f"tcp://{addr}")
            s.send(_pack(msg))
            return _unpack(s.recv())
        except zmq.ZMQError:
            return None
        finally:
            s.close(0)

    def peer_meta(self, addr: str, refresh: bool = False) -> Optional[_PeerMeta]:
        with self._lock:
            m = self._meta.get(addr)
        if m is not None and not refresh:
            return m
        r = self._call(addr, {"op": "meta"})
        if not r or not r.get("ok"):
            return None
        m = _PeerMeta(r)
        with self._lock:
            self._meta[addr] = m
        return m

    def forget_peer(self, addr: str) -> None:
        with self._lock:
            self._meta.pop(addr, None)

    def pull(self, req_id: str, addr: str, transfer_id: str, local_blocks: Sequence[int],
             first_block: int = 0,
             sources: Optional[Sequence[tuple[str, TPSource]]] = None) -> None:
        """Start pulling remote blocks [first_block, first_block + len(local))
        of ``transfer_id`` from the producer at ``addr`` (host:port) into
        ``local_blocks``. The result shows up in ``take_done()``.

        ``sources`` (heterogeneous TP): (producer address, ``TPSource``) per
        remote TP rank this rank reads its KV heads from; ``addr`` is then
        ignored. The request succeeds only if every source was read."""
        self._pool.submit(self._pull, req_id, addr, transfer_id, list(local_blocks), first_block,
                          list(sources) if sources else None)

    def _pull(self, req_id, addr, tid, local, first, sources=None) -> None:
        t0 = time.monotonic()
        res = PullResult(req_id, False, failed_blocks=local)
        try:
            if sources:
                res = self._pull_sources(req_id, sources, tid, local, first, t0)
            else:
                res = self._pull_inner(req_id, addr, tid, local, first, t0)
        except Exception as e:  # noqa: BLE001
            res.why = f"error: {e}"
            log.exception("KV pull of %s failed", req_id)
        if not res.ok:
            res.failed_blocks = list(local)
            log.warning("KV pull req=%s tid=%s from %s failed: %s", req_id, tid, addr, res.why)
        with self._lock:
            self.stats["pulls"] += 1
            self.stats["failed"] += 0 if res.ok else 1
            self.stats["bytes"] += res.nbytes
            self.stats["seconds"] += res.seconds
            self._done.append(res)

    def _pull_sources(self, req_id, sources, tid, local, first, t0) -> PullResult:
        if self.inject_delay_s:
            time.sleep(self.inject_delay_s)
        nbytes = 0
        for addr, src in sources:
            r = self._pull_inner(req_id, addr, tid, local, first, t0, src=src, delay=False)
            if not r.ok:
                r.why = f"TP rank {src.remote_rank} ({addr}): {r.why}"
                return r
            nbytes += r.nbytes
        return PullResult(req_id, True, nbytes=nbytes, seconds=time.monotonic() - t0)

    def _pull_inner(self, req_id, addr, tid, local, first, t0, src: Optional[TPSource] = None,
                    delay: bool = True) -> PullResult:
        if self.inject_delay_s and delay:   # fault injection: widen the transfer window
            time.sleep(self.inject_delay_s)
        # The decode request can overtake the prefill engine handing the hold
        # to its worker (that happens on its next scheduler step): "unknown"
        # is retried for a short while before it counts as gone.
        delay = 0.002
        while True:
            r = self._call(addr, {"op": "acquire", "tid": tid, "who": self.who})
            if (r is None or r.get("ok") or r.get("why") != "unknown"
                    or time.monotonic() - t0 > self.acquire_wait_s):
                break
            time.sleep(delay)
            delay = min(delay * 2, 0.05)
        if r is None:
            self.forget_peer(addr)
            return PullResult(req_id, False, "producer unreachable")
        if not r.get("ok"):
            return PullResult(req_id, False, f"producer: {r.get('why')}")
        remote_all = r["blocks"]
        if not local:
            return PullResult(req_id, True)
        remote = remote_all[first:first + len(local)]
        if len(remote) < len(local):
            return PullResult(req_id, False, f"producer holds {len(remote_all)} blocks, "
                                             f"need {first + len(local)}")
        meta = self.peer_meta(addr)
        if meta is None:
            return PullResult(req_id, False, "producer meta unavailable")
        if self.block_size and meta.block_size and meta.block_size != self.block_size:
            return PullResult(req_id, False, f"block size {meta.block_size} != {self.block_size}")
        if self.device and self.device == meta.device and self.plane.name == "connex-rs":
            # UCX 1.16 (HPC-X) serves a get between two processes on the same
            # GPU with bcopy over rc_mlx5, i.e. a CPU memcpy into device
            # memory: the decode process segfaults. Refuse instead.
            return PullResult(req_id, False, "prefill and decode share a GPU; connex-rs "
                                             "needs them on different GPUs")
        ops = self._ops(meta, local, remote, src)
        if self.plane.name == "connex-rs" and ops and min(n for _, _, n in ops) < ZCOPY_THRESH:
            return PullResult(req_id, False, f"KV segment of {min(n for _, _, n in ops)} B is below "
                                             f"the {ZCOPY_THRESH} B zero-copy threshold of connex-rs")
        bad = self._outside(ops)
        if bad:
            return PullResult(req_id, False, f"read outside registered KV memory: {bad}")
        if self._staging is not None:
            rc = self.plane.read(meta.url, self._staging.mirror_ops(ops))
            if rc == 0:
                self._staging.stage_in([(d, n) for d, _, n in ops])
        else:
            rc = self.plane.read(meta.url, ops)
        if rc != 0:
            self.forget_peer(addr)
            return PullResult(req_id, False, f"{self.plane.name} read returned {rc}")
        if self.release_after_pull:
            self._call(addr, self._release_msg(tid, src))
        n = sum(x[2] for x in ops)
        return PullResult(req_id, True, nbytes=n, seconds=time.monotonic() - t0)

    def _ops(self, meta: _PeerMeta, local, remote, src: Optional[TPSource]):
        if src is not None:
            lo, ro = self.layouts[0], meta.layouts[0]
            whole = (src.nheads == lo.heads == ro.heads and lo.seg_bytes == ro.seg_bytes
                     and lo.head_stride == ro.head_stride and lo.token_stride == ro.token_stride)
            if not whole:
                return head_read_ops(self.layouts, meta.layouts, local, remote,
                                     src.local_head0, src.remote_head0, src.nheads)
        return read_ops(self.layouts, meta.layouts, local, remote)

    def _release_msg(self, tid: str, src: Optional[TPSource] = None) -> dict:
        m = {"op": "release", "tid": tid, "who": self.who}
        if src is not None and src.readers > 1:
            m.update(of=src.readers)
        return m

    def _outside(self, ops) -> Optional[tuple]:
        """First op whose destination is not inside registered local memory (a
        DMA there would corrupt or crash the process), else None."""
        spans = self._spans
        for dst, src, n in ops:
            if not any(s <= dst and dst + n <= s + ln for s, ln in spans):
                return (hex(dst), n)
        return None

    def release(self, addr: str, transfer_id: str, src: Optional[TPSource] = None) -> None:
        """Tell the producer it may free the blocks (fire and forget). With a
        ``TPSource`` read by several local ranks, the producer frees once all
        of them released."""
        self._pool.submit(self._call, addr, self._release_msg(transfer_id, src))

    def take_done(self) -> list[PullResult]:
        with self._lock:
            out, self._done = self._done, []
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)
