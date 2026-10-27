# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Point-to-point link between pipeline stages over connex-rs (UCX).

One :class:`StageLink` per process (per TP rank 0 of a stage, or per rank in
the in-process mode). It moves two kinds of traffic:

* tensor dicts (hidden states + residual between stages): the sender copies
  the tensors into a registered GPU ring and sends a small header; the
  receiver pulls the bytes with one one-sided read (GPUDirect RDMA when the
  NIC supports it) into its own registered ring and acks, which frees the
  sender's slot;
* small byte messages (the sampled-token broadcast of the V2 model runner,
  control): inline in the message.

Messages are keyed by ``(epoch, seq, idx)``. ``epoch`` changes when the
pipeline fails over to another replica; :meth:`abort` releases everything a
caller still waits for in an older epoch, so a stage whose upstream died
never blocks forever.

Peer-failure events from connex-rs are passed to ``on_peer_failed``.
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

log = logging.getLogger("vllm.connex_pp")

ZCOPY_THRESH = 4096
_ALIGN = 256

# message kinds
TENSORS = 0
ACK = 1
BYTES = 2
ABORT = 3
CONTROL = 4
READY = 5     # the data of an announced tensor dict is in place

READ_TIMEOUT_MS = int(os.getenv("CONNEX_PP_READ_TIMEOUT_MS", "10000"))
_DTYPES = {str(d): d for d in (torch.float16, torch.bfloat16, torch.float32, torch.int64,
                               torch.int32, torch.uint8, torch.float8_e4m3fn)}


def _setup_ucx_env(ip: str) -> None:
    os.environ.setdefault("UCX_TCP_CM_REUSEADDR", "y")
    # Reads into GPU memory must stay zero-copy (see connex.pd.dataplane).
    os.environ.setdefault("UCX_ZCOPY_THRESH", str(ZCOPY_THRESH))
    # No CUDA IPC between processes: a mapping of a dead peer's memory keeps
    # its GPU allocation alive (see insight on connex-rs + cuda_ipc).
    if os.getenv("CONNEX_RS_CUDA_IPC", "0") != "1" and not os.getenv("UCX_TLS"):
        os.environ["UCX_TLS"] = "^cuda_ipc"
    if not os.getenv("UCX_NET_DEVICES"):
        from connex.pp.transports import ucx_net_devices_for_ip

        nd = os.getenv("CONNEX_RS_NET_DEVICES", "auto")
        nd = ucx_net_devices_for_ip(ip) if nd == "auto" else nd
        if nd:
            os.environ["UCX_NET_DEVICES"] = nd


def local_ip() -> str:
    for name in ("CONNEX_RS_LOCAL_IP", "CONNEX_HOST_IP", "VLLM_HOST_IP"):
        if os.getenv(name):
            return os.environ[name]
    try:
        from vllm.utils.network_utils import get_ip

        return get_ip()
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


def make_engine(model: str, stage: int, shard: int, replica: int, port: int = 0,
                ip: str | None = None):
    """A connex_rs.Engine (raises ImportError when connex-rs is missing)."""
    import connex_rs  # type: ignore

    ip = ip or local_ip()
    _setup_ucx_env(ip)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # connex-rs classifies GPU memory with the registering thread's
        # current CUDA context.
        torch.cuda.set_device(torch.cuda.current_device())
    return connex_rs.Engine(
        model, stage, shard, replica, listen_ip=ip, port=port,
        heartbeat_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_MS", "1000")),
        heartbeat_timeout_ms=int(os.getenv("CONNEX_RS_HEARTBEAT_TIMEOUT_MS", "3000")),
        spin_us=int(os.getenv("CONNEX_RS_SPIN_US", "200")),
    )


class _Ring:
    """First-fit ring allocator over one registered buffer; frees may come
    in any order."""

    def __init__(self, size: int):
        self.size = size
        self.head = 0          # next allocation offset
        self.live: dict[int, int] = {}   # offset -> length
        self.cv = threading.Condition()

    def _fits(self, n: int) -> int | None:
        if n > self.size:
            raise ValueError(f"message of {n} bytes exceeds ring of {self.size} bytes "
                             "(raise CONNEX_PP_RING_MB)")
        if not self.live:
            self.head = 0
            return 0
        start = self.head
        if start + n > self.size:
            start = 0
        end = start + n
        for off, ln in self.live.items():
            if off < end and start < off + ln:
                return None
        return start

    def alloc(self, n: int, timeout: float | None = None) -> int:
        n = (n + _ALIGN - 1) // _ALIGN * _ALIGN
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cv:
            while True:
                off = self._fits(n)
                if off is not None:
                    self.live[off] = n
                    self.head = off + n
                    return off
                rem = None if deadline is None else deadline - time.monotonic()
                if rem is not None and rem <= 0:
                    raise TimeoutError("ring full")
                self.cv.wait(0.002)
                self.cv.release()
                try:
                    drain_frees()
                finally:
                    self.cv.acquire()

    def free(self, off: int) -> None:
        with self.cv:
            self.live.pop(off, None)
            self.cv.notify_all()


@dataclass
class _Layout:
    name: str
    dtype: torch.dtype
    shape: tuple
    offset: int
    nbytes: int


def _layout(tensors: dict[str, torch.Tensor]) -> tuple[list[_Layout], int]:
    out, off = [], 0
    for k, t in tensors.items():
        n = t.numel() * t.element_size()
        out.append(_Layout(k, t.dtype, tuple(t.shape), off, n))
        off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
    return out, off


@dataclass
class _Inbox:
    """Delivered items by key, with waiters."""
    items: dict = field(default_factory=dict)
    cv: threading.Condition = field(default_factory=threading.Condition)


class StageLink:
    """See the module docstring."""

    def __init__(self, model: str, stage: int, shard: int, replica: int,
                 device: torch.device, port: int = 0, ring_mb: int | None = None,
                 on_peer_failed: Callable[[tuple], None] | None = None,
                 on_control: Callable[[tuple, Any], None] | None = None):
        self.device = device
        si = float(os.getenv("CONNEX_PP_SWITCH_INTERVAL_US", "0"))
        if si > 0:
            # The receive/send threads wait for the GIL behind the engine's
            # main thread; Python hands it over only every switch interval
            # (5 ms by default), which then shows up in every hop.
            import sys

            sys.setswitchinterval(si / 1e6)
        self.engine = make_engine(model, stage, shard, replica, port)
        self.url = self.engine.local_url
        ring_mb = ring_mb or int(os.getenv("CONNEX_PP_RING_MB", "256"))
        nbytes = ring_mb << 20
        self.send_buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        for buf in (self.send_buf, self.recv_buf):
            rc = self.engine.register_memory(buf.data_ptr(), buf.numel())
            if rc < 0:
                raise RuntimeError(f"connex-rs: registering {buf.numel()} bytes failed ({rc})")
        self.send_ring = _Ring(nbytes)
        self.recv_ring = _Ring(nbytes)
        self.copy_stream = torch.cuda.Stream(device)
        self.on_peer_failed = on_peer_failed
        self.on_control = on_control
        self.min_epoch = 0
        self.abort_callbacks: list[Callable[[int], None]] = []
        self.inbox = _Inbox()
        self.up_peers: set[tuple] = set()
        self._out_lock = threading.Lock()
        self._post_lock = threading.Lock()
        self._posted: dict[tuple, tuple] = {}   # key -> (recv ring offset, wait slot)
        self._ready: dict[tuple, tuple] = {}    # key -> (peer, READY) before the post
        self._prev_off: int | None = None
        self._waiter = None
        self._outstanding: dict[int, tuple] = {}
        self._msg_id = 0
        self._tx: queue.Queue = queue.Queue()
        self._closed = False
        self.stats = {"sent": 0, "recv": 0, "bytes_sent": 0, "bytes_recv": 0,
                      "read_s": 0.0}
        # connex-rs releases the GIL while waiting; these threads only parse
        # headers and post reads.
        for target, name in ((self._rx_loop, "cx-pp-rx"), (self._tx_loop, "cx-pp-tx"),
                             (self._event_loop, "cx-pp-ev"), (_orphan_watch, "cx-pp-orphan")):
            threading.Thread(target=target, name=name, daemon=True).start()

    # ------------------------------------------------------------------ send
    def send_tensors(self, peer_url: str, key: tuple, tensors: dict[str, torch.Tensor],
                     meta: Any = None) -> None:
        """Queue ``tensors`` for ``peer_url``. Stream-ordered: the copy into
        the ring is issued on the current stream, so the caller may reuse
        the tensors after the call."""
        layout, total = _layout(tensors)
        off = self.send_ring.alloc(max(total, _ALIGN))
        base = self.send_buf[off:off + total]
        for lay in layout:
            t = tensors[lay.name]
            dst = base[lay.offset:lay.offset + lay.nbytes]
            dst.copy_(t.contiguous().view(-1).view(torch.uint8), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        with self._out_lock:
            self._msg_id += 1
            mid = self._msg_id
            # message id -> (ring offset, epoch, peer id); freed by the ack,
            # by the peer's failure, or by an abort of the message's epoch
            self._outstanding[mid] = (off, key[0], _peer_id(peer_url))
        # The header goes now, while the GPU still computes the tensors (like
        # vLLM's metadata send), so the receiver can set up its step; the
        # data is announced once the copy is done.
        lay = [(x.name, str(x.dtype), x.shape, x.offset, x.nbytes) for x in layout]
        try:
            self.engine.send_msg(peer_url, pickle.dumps((TENSORS, key, lay, total, meta),
                                                        protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:  # noqa: BLE001  (peer failed / unknown)
            log.warning("connex-pp: send to %s failed: %s", peer_url, e)
            self._release_sent(mid)
            return
        self._tx.put((ev, peer_url, (READY, key, self.url, self.send_buf.data_ptr() + off,
                                     mid, total)))

    def _release_sent(self, mid: int) -> None:
        with self._out_lock:
            ent = self._outstanding.pop(mid, None)
        if ent is not None:
            self.send_ring.free(ent[0])

    def _release_sent_where(self, pred) -> int:
        with self._out_lock:
            mids = [m for m, ent in self._outstanding.items() if pred(ent)]
            ents = [self._outstanding.pop(m) for m in mids]
        for off, _, _ in ents:
            self.send_ring.free(off)
        return len(ents)

    def send_bytes(self, peer_url: str, key: tuple, data: Any, event=None) -> None:
        """Send a small picklable payload (after ``event`` completes)."""
        self._tx.put((event, peer_url, (BYTES, key, self.url, data)))

    def send_abort(self, peer_url: str, epoch: int) -> None:
        self._tx.put((None, peer_url, (ABORT, epoch)))

    def send_control(self, peer: Any, msg: Any) -> None:
        self._tx.put((None, peer, (CONTROL, msg)))

    def _tx_loop(self) -> None:
        while not self._closed:
            item = self._tx.get()
            if item is None:
                return
            ev, peer, hdr = item
            if ev is not None:
                ev.synchronize()
            if hdr[0] == BYTES and callable(hdr[3]):
                hdr = hdr[:3] + (hdr[3](),)
            try:
                self.engine.send_msg(peer, pickle.dumps(hdr, protocol=pickle.HIGHEST_PROTOCOL))
                if hdr[0] == READY:
                    self.stats["sent"] += 1
                    self.stats["bytes_sent"] += hdr[5]
            except Exception as e:  # noqa: BLE001  (peer failed / unknown)
                log.warning("connex-pp: send to %s failed: %s", peer, e)
                if hdr[0] == READY:
                    self._release_sent(hdr[4])

    # --------------------------------------------------------------- receive
    def _rx_loop(self) -> None:
        while not self._closed:
            try:
                m = self.engine.recv_msg(timeout_ms=200)
            except Exception:  # noqa: BLE001  (engine closed)
                if self._closed:
                    return
                time.sleep(0.05)
                continue
            if m is not None:
                self._handle(*m)

    def _handle(self, peer, data) -> None:
        try:
            hdr = pickle.loads(data)
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: bad message from %s: %s", peer, e)
            return
        kind = hdr[0]
        if kind == ACK:
            self._release_sent(hdr[1])
        elif kind == TENSORS:
            _, key, layout, total, meta = hdr
            self._deliver(key, _Header(layout, total, meta))
        elif kind == READY:
            self._on_ready(peer, hdr)
        elif kind == BYTES:
            _, key, _src, payload = hdr
            self._deliver(key, payload)
        elif kind == ABORT:
            self.abort(hdr[1])
        elif kind == CONTROL:
            if self.on_control is not None:
                try:
                    self.on_control(peer, hdr[1])
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: control handler failed")

    def _on_ready(self, peer, hdr) -> None:
        _, key, src_url, src_ptr, mid, total = hdr
        with self._post_lock:
            posted = self._posted.pop(key, None)
            if posted is None:
                if key[0] >= self.min_epoch:
                    self._ready[key] = (peer, hdr)
                    return
        if posted is None:   # aborted epoch
            self._ack(peer, mid)
            return
        self._read_into(peer, hdr, posted)

    def _ack(self, peer, mid: int) -> None:
        # Directly, not through the send queue: an ack must not wait behind
        # sends that wait for the GPU (the sender's ring could fill up).
        try:
            self.engine.send_msg(peer, pickle.dumps((ACK, mid)))
        except Exception:  # noqa: BLE001  (peer gone: its slots are released anyway)
            pass

    def _read_into(self, peer, hdr, posted) -> None:
        _, key, src_url, src_ptr, mid, total = hdr
        off, slot = posted
        rc = self.engine.transfer_sync_read(src_url, self.recv_buf.data_ptr() + off, src_ptr,
                                            total, READ_TIMEOUT_MS)
        # The sender's slot is free once the read is done; ack off the
        # critical path.
        self._ack(peer, mid)
        if rc != 0:
            log.warning("connex-pp: read of %d bytes from %s failed (%d)", total, src_url, rc)
        self.stats["recv"] += 1
        self.stats["bytes_recv"] += total
        self._waiter.fill(slot, None)   # the stream may use the data now

    def _release(self, item) -> None:
        pass  # headers and byte payloads hold no ring space

    def _deliver(self, key, item) -> None:
        with self.inbox.cv:
            if key[0] < self.min_epoch:
                self._release(item)
                return
            self.inbox.items[key] = item
            self.inbox.cv.notify_all()

    def take(self, key: tuple, timeout: float | None = None):
        """Block until the item for ``key`` arrives. Returns None when the
        key's epoch was aborted (or on timeout)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.inbox.cv:
            while True:
                if key in self.inbox.items:
                    return self.inbox.items.pop(key)
                if key[0] < self.min_epoch or self._closed:
                    return None
                rem = None if deadline is None else deadline - time.monotonic()
                if rem is not None and rem <= 0:
                    return None
                self.inbox.cv.wait(rem if rem is not None else 0.5)

    def poll(self, key: tuple):
        with self.inbox.cv:
            return self.inbox.items.pop(key, None)

    def recv_tensors(self, key: tuple, timeout: float | None = None,
                     blocking: bool = False):
        """Tensors for ``key``, or None when its epoch was aborted.

        Returns as soon as the sender's header is here: the tensors are views
        of the receive ring, and the current stream waits until their bytes
        have landed (like a NCCL receive), so the caller can queue its work
        while the sender's GPU is still busy. They stay valid until the next
        call."""
        item = self.take(key, timeout)
        stream = torch.cuda.current_stream(self.device)
        # The previous step's tensors were consumed by work queued before now.
        if self._prev_off is not None:
            ev = torch.cuda.Event()
            ev.record(stream)
            _free_after(ev, self.recv_ring, self._prev_off)
            self._prev_off = None
        if item is None:
            return None
        if self._waiter is None:
            from connex.adapters.vllm.pp.hostwait import HostFlagWaiter

            self._waiter = HostFlagWaiter(self.device)
        off = self.recv_ring.alloc(max(item.total, _ALIGN))
        slot = self._waiter.post_wait() if not blocking else self._waiter.no_wait()
        self._prev_off = off
        with self._post_lock:
            stashed = self._ready.pop(key, None)
            if stashed is None:
                self._posted[key] = (off, slot)
        if stashed is not None:
            self._read_into(stashed[0], stashed[1], (off, slot))
        if blocking:
            # Wait on the CPU until the bytes are here (no stream wait).
            self._waiter.wait_filled(slot)
        base = self.recv_buf[off:off + item.total]
        out = {}
        for name, dtype, shape, loff, n in item.layout:
            dt = _DTYPES.get(dtype) or getattr(torch, dtype.replace("torch.", ""))
            out[name] = base[loff:loff + n].view(dt).view(shape)
        return out, item.meta

    def abort(self, epoch: int) -> None:
        """Release waiters for epochs older than ``epoch``."""
        with self.inbox.cv:
            if epoch <= self.min_epoch:
                return
            self.min_epoch = epoch
            for key in [k for k in self.inbox.items if k[0] < epoch]:
                self._release(self.inbox.items.pop(key))
            self.inbox.cv.notify_all()
        with self._post_lock:
            gone = [k for k in self._posted if k[0] < epoch]
            posted = [self._posted.pop(k) for k in gone]
            stale = [self._ready.pop(k) for k in [k for k in self._ready if k[0] < epoch]]
        for _off, slot in posted:
            self._waiter.fill(slot, None)
        for peer, hdr in stale:
            self._ack(peer, hdr[4])
        # Sends of older epochs will not be needed (a live receiver drops
        # them); a dead one never acks.
        self._release_sent_where(lambda ent: ent[1] < epoch)
        for cb in list(self.abort_callbacks):
            try:
                cb(epoch)
            except Exception:  # noqa: BLE001
                log.exception("connex-pp: abort callback failed")

    # ---------------------------------------------------------------- events
    def _event_loop(self) -> None:
        while not self._closed:
            try:
                ev = self.engine.wait_event(timeout_ms=500)
            except Exception:  # noqa: BLE001
                return
            if ev is None:
                continue
            if ev.get("type") in ("up", "joined"):
                with self.inbox.cv:
                    self.up_peers.add(tuple(ev.get("peer") or ()))
                    self.inbox.cv.notify_all()
                continue
            if ev.get("type") != "failed":
                continue
            peer = tuple(ev.get("peer") or ())
            with self.inbox.cv:
                self.up_peers.discard(peer)
            n = self._release_sent_where(lambda ent: ent[2] == peer)
            if n:
                log.warning("connex-pp: released %d send slot(s) of failed peer %s", n, peer)
            log.warning("connex-pp: peer %s failed (%s)", ev.get("peer"), ev.get("reason"))
            if self.on_peer_failed is not None:
                try:
                    self.on_peer_failed(tuple(ev.get("peer") or ()))
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: failure handler failed")

    def connect(self, url: str) -> None:
        self.engine.connect(url)

    def wait_up(self, peer: tuple, timeout: float = 60.0) -> bool:
        """Wait until the link to ``peer`` (model, stage, shard, replica) is
        up, whichever side connected."""
        deadline = time.monotonic() + timeout
        with self.inbox.cv:
            while peer not in self.up_peers:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self.inbox.cv.wait(min(rem, 0.5))
        return True

    def close(self) -> None:
        self._closed = True
        self._tx.put(None)
        try:
            self.engine.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _Header:
    layout: list
    total: int
    meta: Any


_pending_frees: deque = deque()
_pending_lock = threading.Lock()


def _peer_id(url: str) -> tuple:
    """(model, stage, shard, replica) of "ip:port#model|stage|shard|replica"."""
    try:
        model, stage, shard, replica = url.split("#", 1)[1].split("|")
        return (model, int(stage), int(shard), int(replica))
    except (IndexError, ValueError):
        return ()


def _orphan_watch() -> None:
    """Exit when the parent (the executor / stage server) is gone: a worker
    blocked on a step that will never arrive must not outlive it."""
    parent = os.getppid()
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            log.warning("connex-pp: parent process %d exited; exiting", parent)
            os._exit(1)


def drain_frees() -> None:
    done = []
    with _pending_lock:
        while _pending_frees and _pending_frees[0][0].query():
            done.append(_pending_frees.popleft())
    for _, r, o in done:
        r.free(o)


def _free_after(ev: torch.cuda.Event, ring: _Ring, off: int) -> None:
    """Free ``off`` once ``ev`` has completed (checked lazily)."""
    with _pending_lock:
        _pending_frees.append((ev, ring, off))
    drain_frees()
