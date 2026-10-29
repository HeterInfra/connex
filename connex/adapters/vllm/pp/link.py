# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Point-to-point link between pipeline stages over connex-rs (UCX).

One :class:`StageLink` per process (per TP rank 0 of a stage, or per rank in
the in-process mode). It moves two kinds of traffic:

* tensor dicts (hidden states + residual between stages): the sender copies
  the tensors into a registered GPU ring and sends a small header (at launch
  time, before the GPU has computed them). Then, by peer:

  - same host: CUDA IPC channel (ipcchan), the GPUs synchronise on flags;
  - another host, once the receiver has announced a push channel (rdmachan):
    the sender's connex-rs writes the bytes and a flag into the receiver's
    ring when the sender's stream rings a doorbell; the receiving stream
    waits on the flag;
  - otherwise (the first message from a new sender): the receiver pulls the
    bytes with one one-sided read (GPUDirect RDMA when the NIC supports it)
    into its own registered ring and acks, which frees the sender's slot;
    the receiving side waits on the CPU;
* small byte messages (the sampled-token broadcast of the V2 model runner,
  control): inline in the message.

Messages are keyed by ``(epoch, seq, idx)``. ``epoch`` changes when the
pipeline fails over to another replica; :meth:`abort` releases everything a
caller still waits for in an older epoch, so a stage whose upstream died
never blocks forever.

Peer-failure events from connex-rs are passed to ``on_peer_failed``.

Failures never turn into silently wrong data: a hop whose bytes cannot be
delivered for a live epoch (a failed read, a failed push chain, a push on a
channel we do not know) stops this process (``fatal``), so the pipeline
fails over and the step is recomputed; for an aborted epoch they are
dropped.
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
CHAN = 6      # a receiver's push channel (rdmachan) for the sender to use
WANT = 7      # a sender asks for a push channel (warming a spare's hop)
CHANX = 8     # a receiver updates what a push channel carries (sampled tokens)
CHANFAIL = 9  # a sender's chain into one of our channels failed

# Same-host stage pairs copy over CUDA IPC (CONNEX_PP_CUDA_IPC=0: RDMA loopback).
IPC_ENABLED = os.getenv("CONNEX_PP_CUDA_IPC", "1") == "1"
# Cross-host pairs: the sender's NIC writes our ring and flag (rdmachan), so
# the receiving stream can wait without a Python thread on the path.
# CONNEX_PP_RDMA_PUSH=0: receive with one-sided reads, waiting on the CPU.
PUSH_ENABLED = os.getenv("CONNEX_PP_RDMA_PUSH", "1") == "1"
_push_ok: bool | None = None


def push_supported() -> bool:
    """PUSH_ENABLED and a connex-rs whose ``Engine.put_chain`` takes a cancel
    word and compares cyclically (connex-rs>=0.1.5); without it cross-host
    receives wait on the CPU. Both ends of a hop are assumed to run the same
    install."""
    global _push_ok
    if _push_ok is None:
        try:
            import connex_rs  # type: ignore

            sig = getattr(getattr(connex_rs.Engine, "put_chain", None), "__text_signature__", "")
            _push_ok = PUSH_ENABLED and "cancel" in (sig or "")
        except ImportError:
            _push_ok = False
    return _push_ok


READ_TIMEOUT_MS = int(os.getenv("CONNEX_PP_READ_TIMEOUT_MS", "10000"))
# A push chain whose doorbell (the sender's own stream) does not ring in
# this long fails (a hung GPU), instead of holding its queue forever.
CHAIN_ARM_TIMEOUT_MS = int(os.getenv("CONNEX_PP_CHAIN_ARM_TIMEOUT_MS", "60000"))
# Waiting for space in the send ring longer than this raises.
RING_TIMEOUT_S = float(os.getenv("CONNEX_PP_RING_TIMEOUT_S", "60"))
# Re-announce a push channel to a sender that still sends without it.
REANNOUNCE_S = 1.0
# RMA on GPU memory is at least this long (see rdmachan.ZCOPY_MIN).
ZCOPY_MIN = ZCOPY_THRESH
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
                    raise TimeoutError(f"ring full for {timeout:.1f} s ({len(self.live)} "
                                       "messages held)")
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
                 on_control: Callable[[tuple, Any], None] | None = None,
                 ipc_slot_bytes: int = 0, ipc_slots: int = 3,
                 sampled_slots: int = 64, sampled_slot_bytes: int = 1 << 16):
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
        # Called with a reason when this process must stop (see fatal()).
        self.on_fatal: Callable[[str], None] | None = None
        self.min_epoch = 0
        self.abort_callbacks: list[Callable[[int], None]] = []
        self.inbox = _Inbox()
        self.up_peers: set[tuple] = set()
        self._out_lock = threading.Lock()
        self._post_lock = threading.Lock()
        self._posted: dict[tuple, tuple] = {}   # key -> (recv ring offset, wait slot)
        self._ready: dict[tuple, tuple] = {}    # key -> (peer, READY) before the post
        self._prev_off: int | None = None
        # GPU-driven same-host channels (ipcchan): ours to each peer, and
        # the peers' to us; plus the receive whose slot is released next.
        self.ipc_slot_bytes, self.ipc_slots = ipc_slot_bytes, ipc_slots
        # What a push channel of each tag looks like: (slots, slot bytes, host ring)
        self.chan_specs = {"h": (ipc_slots, ipc_slot_bytes, False),
                           "s": (sampled_slots, sampled_slot_bytes, True)}
        self._chan_lock = threading.RLock()
        self._tx_chans: dict[tuple, Any] = {}
        self._rx_chans: dict[tuple, Any] = {}
        self._rx_done: tuple | None = None
        self._rx_last: dict[str, int] = {}
        self._graveyard = None
        self._ip = self.url.split(":", 1)[0]
        self._waiter = None
        self._outstanding: dict[int, tuple] = {}
        self._msg_id = 0
        # push channels (rdmachan) we receive on, by id (the current one of
        # a (peer, tag) and up to one older one whose pushes may still be
        # consumed), and when we last announced one to a sender incarnation
        self._push_rx: dict[int, Any] = {}
        self._retired_cids: deque = deque(maxlen=256)
        self._announced: dict[tuple, float] = {}
        self.nonce = os.urandom(8).hex()
        self._reap: queue.Queue = queue.Queue()
        self.push_fallback: dict[tuple, tuple] = {}
        self._tx: queue.Queue = queue.Queue()
        self._closed = False
        self._fatal = False
        self.stats = {"sent": 0, "recv": 0, "bytes_sent": 0, "bytes_recv": 0,
                      "read_s": 0.0, "chain_err": 0, "fallback_fill": 0}
        # connex-rs releases the GIL while waiting; these threads only parse
        # headers and post reads.
        for target, name in ((self._rx_loop, "cx-pp-rx"), (self._tx_loop, "cx-pp-tx"),
                             (self._event_loop, "cx-pp-ev"), (_orphan_watch, "cx-pp-orphan"),
                             (self._reap_loop, "cx-pp-reap")):
            threading.Thread(target=target, name=name, daemon=True).start()

    # ---------------------------------------------------------------- failure
    def fatal(self, reason: str) -> None:
        """This hop cannot deliver the data of a live step: stop the process
        (its stage fails over, the step is recomputed) rather than letting a
        stream go on with bytes that never arrived. ``on_fatal`` may
        override (tests)."""
        log.critical("connex-pp: %s; stopping this process so the pipeline fails over", reason)
        self._fatal = True
        if self.on_fatal is not None:
            self.on_fatal(reason)
            return
        logging.shutdown()
        os._exit(70)

    # ------------------------------------------------------------------ send
    def send_tensors(self, peer_url: str, key: tuple, tensors: dict[str, torch.Tensor],
                     meta: Any = None) -> None:
        """Queue ``tensors`` for ``peer_url``. Stream-ordered: the copy into
        the ring is issued on the current stream, so the caller may reuse
        the tensors after the call."""
        if not peer_url:
            return  # no live replica on the route (failover in progress)
        layout, total = _layout(tensors)
        if self.same_host(peer_url) and total <= self.ipc_slot_bytes:
            self._send_ipc(peer_url, key, tensors, layout, total, meta)
            return
        tx = self._tx_chans.get((_peer_id(peer_url), ("p", "h")))
        if tx is not None and not tx.dead and total <= tx.slot_bytes:
            if self._send_push(tx, peer_url, key, tensors, layout, total, meta):
                return
        # At least ZCOPY_MIN bytes: the receiver reads that much (GPU RMA).
        off = self.send_ring.alloc(max(total, ZCOPY_MIN), timeout=RING_TIMEOUT_S)
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
        path = ("rdma", self.nonce) if push_supported() and self.ipc_slot_bytes > 0 else None
        try:
            self.engine.send_msg(peer_url, pickle.dumps((TENSORS, key, lay, total, meta, path),
                                                        protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:  # noqa: BLE001  (peer failed / unknown)
            log.warning("connex-pp: send to %s failed: %s", peer_url, e)
            self._release_sent(mid)
            return
        self._tx.put((ev, peer_url, (READY, key, self.url, self.send_buf.data_ptr() + off,
                                     mid, total)))

    def _send_push(self, tx, peer_url, key, tensors, layout, total, meta) -> bool:
        """Cross host, receiver-driven: copy into our registered ring, ring
        the doorbell on the stream; connex-rs puts the bytes into the
        receiver's slot and sets its flag once the copy is done."""
        # The chain puts at least ZCOPY_MIN bytes (GPU RMA below UCX's
        # zero-copy threshold crashes in UCX): allocate that much.
        off = self.send_ring.alloc(max(total, ZCOPY_MIN), timeout=RING_TIMEOUT_S)
        base = self.send_buf[off:off + total]
        for lay in layout:
            t = tensors[lay.name]
            base[lay.offset:lay.offset + lay.nbytes].copy_(
                t.contiguous().view(-1).view(torch.uint8), non_blocking=True)
        i = tx.i
        gate = i - tx.n + 1 if i >= tx.n else None
        try:
            t = tx.push(i, self.send_buf.data_ptr() + off, total, gate, epoch=key[0])
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: push to %s failed: %s", peer_url, e)
            self.send_ring.free(off)   # nothing was armed; the copy is stream-ordered
            tx.dead = True
            return False
        tx.i += 1
        self._reap.put(_Reap(t, off, tx, peer_url, key, "h", i))
        lay = [(x.name, str(x.dtype), x.shape, x.offset, x.nbytes) for x in layout]
        try:
            self.engine.send_msg(peer_url, pickle.dumps(
                (TENSORS, key, lay, total, meta, ("push", tx.cid, i)),
                protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: send to %s failed: %s", peer_url, e)
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += total
        self.stats["push"] = self.stats.get("push", 0) + 1
        return True

    def push_sampled(self, tx, peer_url: str, key: tuple, chunks: list) -> bool:
        """Sampled tokens over a receiver's host-ring push channel (see
        group._send_sampled); False when the channel cannot take it."""
        v = key[1] * 4 + key[2]
        gate = v - tx.n + 1 if v >= tx.n else None
        try:
            t = tx.push_staged(v, chunks, gate=gate, epoch=key[0])
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: sampled-token push to %s failed: %s", peer_url, e)
            tx.dead = True
            return False
        nbytes = sum(c.numel() for c in chunks)
        self._reap.put(_Reap(t, None, tx, peer_url, key, "s", v, nbytes))
        return True

    def _reap_loop(self) -> None:
        """Free the ring space of pushed messages once connex-rs is done
        with them, in completion order (one stuck chain must not hold the
        others' space), and handle failed chains (see _chain_failed)."""
        pending: list = []
        while not self._closed:
            try:
                item = self._reap.get(timeout=None if not pending else 0.001)
                if item is None:
                    return
                pending.append(item)
                while True:   # take whatever else is queued
                    item = self._reap.get_nowait()
                    if item is None:
                        return
                    pending.append(item)
            except queue.Empty:
                pass
            keep = []
            for r in pending:
                if not r.t.done():
                    keep.append(r)
                    continue
                err = None
                try:
                    r.t.wait(0)
                except Exception as e:  # noqa: BLE001
                    err = e
                if r.off is not None:
                    self.send_ring.free(r.off)
                if err is None:
                    if r.tx.first_done is None:
                        r.tx.first_done = time.monotonic()
                        log.info("connex-pp: first chain on push channel %s to %s done "
                                 "%.1f ms after it opened", r.tx.cid, r.peer_url,
                                 (r.tx.first_done - r.tx.created) * 1e3)
                elif not (isinstance(err, RuntimeError) and "canceled" in str(err)):
                    self._chain_failed(r, err)
            pending = keep

    def _chain_failed(self, r: "_Reap", err: Exception) -> None:
        """A push chain failed (no connection, a failed put/flush/read, a
        lost key, a doorbell or gate that timed out). The channel is not used
        again (later messages go the read path, and the receiver announces a
        new channel). The message itself: dropped when its epoch is aborted;
        sampled tokens are sent again the slow way (the receiver fills its
        slot from the CPU); hidden states tell the receiver, which cannot use
        the step (CHANFAIL)."""
        r.tx.dead = True
        self.stats["chain_err"] += 1
        if r.kind == "w":
            log.warning("connex-pp: warm-up chain on push channel %s to %s failed: %r",
                        r.tx.cid, r.peer_url, err)
            return
        if r.key[0] < self.min_epoch:
            return
        log.warning("connex-pp: push of %s to %s failed: %r", r.key, r.peer_url, err)
        if r.kind == "s":
            try:
                payload = r.tx.staged_bytes(r.v, r.nbytes)
            except Exception:  # noqa: BLE001
                payload = None
            if payload is not None:
                self._tx.put((None, r.peer_url, (BYTES, r.key, self.url, payload)))
            return
        try:
            self.engine.send_msg(r.peer_url, pickle.dumps((CHANFAIL, r.tx.cid, r.v, r.key,
                                                           repr(err)[:200])))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: telling %s about the failed push failed: %s", r.peer_url, e)

    def push_rx(self, peer: tuple, tag: str, slots: int | None = None,
                slot_bytes: int | None = None, host: bool | None = None,
                announce: Any = None):
        """A new receive channel for pushes from ``peer`` (replacing the
        previous one for ``tag``), announced to the peer."""
        from connex.adapters.vllm.pp.rdmachan import PushRx

        d_slots, d_bytes, d_host = self.chan_specs[tag if isinstance(tag, str) else tag[0]]
        slots = slots or d_slots
        slot_bytes = slot_bytes or d_bytes
        host = d_host if host is None else host
        torch.cuda.set_device(self.device)
        k = (tuple(peer), ("p", tag))
        ch = PushRx(self.engine, self.url, self.device, slots, slot_bytes, host=host)
        ch.from_key = announce
        with self._chan_lock:
            old = self._rx_chans.get(k)
            self._rx_chans[k] = ch
            self._push_rx[ch.cid] = ch
            # Pushes into the previous channel may still be consumed (the
            # sender switches when it gets ours): keep it; retire the one
            # before it.
            older = [c for c in self._push_rx.values()
                     if c is not ch and c is not old and getattr(c, "key", None) == k]
            ch.key = k
        for c in older:
            self._retire_rx(c)
        try:
            self.engine.send_msg(tuple(peer), pickle.dumps((CHAN, tag, ch.info, announce)))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: announcing a channel to %s failed: %s", peer, e)
        return ch

    def _retire_rx(self, ch) -> None:
        with self._chan_lock:
            self._push_rx.pop(ch.cid, None)
            for k in [k for k, c in self._rx_chans.items() if c is ch]:
                del self._rx_chans[k]
        self._bury(ch)

    def _bury(self, ch) -> None:
        if hasattr(ch, "from_key"):   # a PushRx of ours
            self._retired_cids.append(ch.cid)
        if self._graveyard is None:
            from connex.adapters.vllm.pp.ipcchan import Graveyard

            self._graveyard = Graveyard()
        self._graveyard.add(ch)

    def announce(self, peer: tuple, tag: str, announce: Any = None):
        """``push_rx`` unless a live channel for (peer, tag) exists."""
        ch = self._rx_chans.get((tuple(peer), ("p", tag)))
        if ch is not None and not ch.retired:
            return ch
        return self.push_rx(peer, tag, announce=announce)

    def update_chan(self, ch, extra: Any) -> None:
        """Tell the sender of push channel ``ch`` what it now carries."""
        ch.from_key = extra
        try:
            self.engine.send_msg(ch.key[0], pickle.dumps((CHANX, ch.cid, extra)))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: updating a channel of %s failed: %s", ch.key[0], e)

    def want_chan(self, url: str, tag: str) -> None:
        """Ask ``url`` (a receiver of ours) for a push channel of ``tag`` now,
        so that the first message to it does not go the slow way."""
        if (_peer_id(url), ("p", tag)) in self._tx_chans:
            return
        self._tx.put((None, url, (WANT, tag)))

    def _on_chan(self, peer, tag, info, extra) -> None:
        from connex.adapters.vllm.pp.rdmachan import PushTx

        torch.cuda.set_device(self.device)
        k = (tuple(peer), ("p", tag))
        host = tag != "h"
        tx = PushTx(self.engine, self.device, info, staging=host, host_staging=host,
                    timeout_ms=READ_TIMEOUT_MS, arm_timeout_ms=CHAIN_ARM_TIMEOUT_MS)
        tx.extra = extra
        log.info("connex-pp: push channel %s (%s) to %s opened", tx.cid, tag, peer)
        with self._chan_lock:
            old = self._tx_chans.get(k)
            self._tx_chans[k] = tx
        if old is not None:
            self._bury(old)
        # A first chain now: connection, keys and the path are warm before a
        # step depends on them.
        try:
            if tx.staging is not None:
                self._reap.put(_Reap(tx.warm(), None, tx, peer, (self.min_epoch, -1, -1),
                                     "w", 0))
            else:
                off = self.send_ring.alloc(ZCOPY_MIN, timeout=1.0)
                self._reap.put(_Reap(tx.warm(self.send_buf.data_ptr() + off), off, tx, peer,
                                     (self.min_epoch, -1, -1), "w", 0))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: warming push channel %s failed: %s", tx.cid, e)

    def _on_chanx(self, peer, cid, extra) -> None:
        with self._chan_lock:
            for (p, _tag), tx in self._tx_chans.items():
                if p == tuple(peer) and getattr(tx, "cid", None) == cid:
                    tx.extra = extra
                    return

    def _on_chanfail(self, peer, cid, v, key, err) -> None:
        if key[0] < self.min_epoch:
            return   # an aborted step: nobody uses it
        ch = self._push_rx.get(cid)
        if (ch is not None and ch.retired) or cid in self._retired_cids:
            return   # we gave that channel up (its waits were released)
        # The step's stream waits for bytes that will not come; letting it go
        # would compute on garbage. Stop: the pipeline fails over.
        self.fatal(f"push of {key} from {peer} into channel {cid} failed ({err})")

    def same_host(self, url: str) -> bool:
        return (bool(url) and IPC_ENABLED and self.ipc_slot_bytes > 0
                and url.split(":", 1)[0] == self._ip)

    def tx_chan(self, url: str, tag: tuple, slot_bytes: int, slots: int):
        """Our channel to ``url`` for ``tag``; (channel, created)."""
        from connex.adapters.vllm.pp.ipcchan import TxChannel

        k = (_peer_id(url), tag)
        with self._chan_lock:   # abort() and the event thread iterate the dict
            ch = self._tx_chans.get(k)
            if ch is not None:
                return ch, False
            ch = self._tx_chans[k] = TxChannel(self.device, slots, slot_bytes, f"{tag[0]}")
        return ch, True

    def rx_chan(self, peer: tuple, tag: tuple, info):
        from connex.adapters.vllm.pp.ipcchan import RxChannel

        k = (tuple(peer), tag)
        with self._chan_lock:   # abort() and the event thread iterate the dict
            ch = self._rx_chans.get(k)
            if ch is None or ch.flags.name != info[0]:
                if ch is not None:
                    self.close_chans(lambda kk, k=k: kk == k)
                ch = self._rx_chans[k] = RxChannel(self.device, info)
        return ch

    def close_chans(self, pred) -> None:
        with self._chan_lock:
            gone = []
            for d in (self._tx_chans, self._rx_chans):
                for k in [k for k in d if pred(k)]:
                    gone.append(d.pop(k))
            for ch in gone:
                if getattr(ch, "cid", None) in self._push_rx and \
                        self._push_rx[ch.cid] is ch:
                    del self._push_rx[ch.cid]
            # older push channels of the same keys
            for cid, ch in list(self._push_rx.items()):
                if getattr(ch, "key", None) is not None and pred(ch.key):
                    gone.append(self._push_rx.pop(cid))
        for ch in gone:
            self._bury(ch)
        self.drain_chans()

    def retain_tx(self, keep: set) -> None:
        """Retire our channels to peers outside ``keep`` (peers that left the
        route and are no spares): their chains are cancelled and the memory
        released, instead of holding ring space and polls."""
        self.close_chans(lambda k: k in self._tx_chans and k[0] and k[0] not in keep)

    def drain_chans(self) -> None:
        if self._graveyard is not None:
            self._graveyard.drain()

    def _send_ipc(self, peer_url, key, tensors, layout, total, meta) -> None:
        # Stream-ordered: the copy into our ring and the "ready" flag are
        # queued behind the work that computed the tensors; the header goes
        # now (at launch time), so the receiver can queue its step.
        ch, _ = self.tx_chan(peer_url, ("h",), self.ipc_slot_bytes, self.ipc_slots)
        chunks = [tensors[x.name].contiguous().view(-1).view(torch.uint8) for x in layout]
        i = ch.push(chunks)
        lay = [(x.name, str(x.dtype), x.shape, x.offset, x.nbytes) for x in layout]
        try:
            self.engine.send_msg(peer_url, pickle.dumps(
                (TENSORS, key, lay, total, meta, ("ipc", ch.info, i)),
                protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: send to %s failed: %s", peer_url, e)
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += total
        self.stats["ipc"] = self.stats.get("ipc", 0) + 1

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
                try:
                    self._handle(*m)
                except Exception:  # noqa: BLE001  (a malformed message must not stop receiving)
                    log.exception("connex-pp: handling a message from %s failed", m[0])

    def _handle(self, peer, data) -> None:
        try:
            hdr = loads_plain(data)
            kind = hdr[0]
        except Exception as e:  # noqa: BLE001
            log.warning("connex-pp: bad message from %s: %s", peer, e)
            return
        if kind == ACK:
            self._release_sent(hdr[1])
        elif kind == TENSORS:
            _, key, layout, total, meta = hdr[:5]
            ipc = hdr[5] if len(hdr) > 5 else None
            self._deliver(key, _Header(layout, total, meta, ipc, tuple(peer)))
        elif kind == CHAN:
            try:
                self._on_chan(peer, hdr[1], hdr[2], hdr[3])
            except Exception:  # noqa: BLE001
                log.exception("connex-pp: opening a push channel to %s failed", peer)
        elif kind == CHANX:
            self._on_chanx(peer, hdr[1], hdr[2])
        elif kind == WANT:
            if push_supported() and not self._closed:
                try:
                    self.announce(tuple(peer), hdr[1])
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: announcing a channel to %s failed", peer)
        elif kind == CHANFAIL:
            self._on_chanfail(tuple(peer), *hdr[1:5])
        elif kind == READY:
            self._on_ready(peer, hdr)
        elif kind == BYTES:
            _, key, _src, payload = hdr
            with self.inbox.cv:
                fb = self.push_fallback.pop(key, None)
                if fb is None:
                    if key[0] >= self.min_epoch:
                        self.inbox.items[key] = payload
                        self.inbox.cv.notify_all()
                    return
                # The sender had no channel yet (or its push failed) for a
                # message we wait for on one: deliver it into the slot from
                # here. Under the inbox lock, so that a receive registering
                # its fallback (expect_push) sees either this or the item.
                self._fill_fallback(key, fb, payload)
        elif kind == ABORT:
            self.abort(hdr[1])
        elif kind == CONTROL:
            if self.on_control is not None:
                try:
                    self.on_control(peer, hdr[1])
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: control handler failed")

    def _fill_fallback(self, key, fb, payload) -> None:
        log.warning("connex-pp: %s came without push; filling from the CPU", key)
        self.stats["fallback_fill"] += 1
        fb[0].fill(fb[1], payload)

    def expect_push(self, key: tuple, ch, v: int) -> None:
        """A receive of ``key`` waits on push channel ``ch`` for message
        ``v``. If the sender sends it the slow way instead (no channel yet,
        or a failed chain), fill the slot from the CPU: now when it is
        already here, else when it arrives. One lock with _handle(BYTES)."""
        with self.inbox.cv:
            payload = self.inbox.items.pop(key, None)
            if payload is None:
                self.push_fallback[key] = (ch, v)
                return
        self._fill_fallback(key, (ch, v), payload)

    def forget_push(self, key: tuple) -> None:
        with self.inbox.cv:
            self.push_fallback.pop(key, None)

    def _on_ready(self, peer, hdr) -> None:
        _, key, src_url, src_ptr, mid, total = hdr[:6]
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
        _, key, src_url, src_ptr, mid, total = hdr[:6]
        off, slot = posted
        # Both rings hold at least ZCOPY_MIN bytes for the message (GPU RMA
        # below UCX's zero-copy threshold crashes in UCX).
        rc = self.engine.transfer_sync_read(src_url, self.recv_buf.data_ptr() + off, src_ptr,
                                            max(total, ZCOPY_MIN), READ_TIMEOUT_MS)
        # The sender's slot is free once the read is done; ack off the
        # critical path.
        self._ack(peer, mid)
        if rc != 0:
            if key[0] < self.min_epoch:
                self._waiter.fill(slot, None)   # aborted: nobody uses the data
                return
            # Do not let the stream (or the blocking receive) go on with
            # bytes that never arrived.
            self.fatal(f"read of {total} bytes for {key} from {src_url} failed ({rc})")
            return
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

    def prune_inbox(self, epoch: int, seq_floor: int) -> int:
        """Drop items of ``epoch`` older than ``seq_floor`` nobody took (a
        payload that also came over a push channel, a channel message of a
        step that ran locally): they would stay forever otherwise."""
        with self.inbox.cv:
            old = [k for k in self.inbox.items if k[0] == epoch and 0 <= k[1] < seq_floor]
            for k in old:
                del self.inbox.items[k]
            for k in [k for k in self.push_fallback if k[0] == epoch and k[1] < seq_floor]:
                del self.push_fallback[k]
        return len(old)

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
        if self._rx_done is not None:
            ch, i = self._rx_done
            ch.done(i)   # the previous step's consumers were queued before now
            self._rx_done = None
        if item is None:
            return None
        path = item.ipc[0] if item.ipc is not None else None
        if path == "push":
            _, cid, i = item.ipc
            ch = self._push_rx.get(cid)
            if ch is None or ch.retired:
                if key[0] < self.min_epoch:
                    return None
                # Its bytes went into a channel we no longer have: they
                # cannot be waited for, and the stream must not use garbage.
                raise RuntimeError(f"connex-pp: push for {key} on unknown channel {cid} "
                                   f"from {item.peer}")
            ch.skip_to(i)   # messages of aborted steps we never consumed
            buf = ch.wait(i, key[0])   # released by the sender's NIC: safe even in warmup
            self._rx_done = (ch, i)
            self.stats["recv"] += 1
            self.stats["bytes_recv"] += item.total
            self.stats["push"] = self.stats.get("push", 0) + 1
            out = {}
            for name, dtype, shape, loff, n in item.layout:
                dt = _DTYPES.get(dtype) or getattr(torch, dtype.replace("torch.", ""))
                out[name] = buf[loff:loff + n].view(dt).view(shape)
            return out, item.meta
        if path == "rdma":
            # A sender that can push but has no channel of ours (yet, or any
            # more: it gave a failed one up). Announce one; again after a
            # while if it keeps sending without.
            ak = (item.peer, item.ipc[1])
            now = time.monotonic()
            if self.ipc_slot_bytes > 0 and now - self._announced.get(ak, -1e9) > REANNOUNCE_S:
                self._announced[ak] = now
                self.push_rx(item.peer, "h")
        elif item.ipc is not None:
            _, info, i = item.ipc
            ch = self.rx_chan(item.peer, ("h",), info)
            last = self._rx_last.get(info[0], -1)
            for j in range(max(last + 1, i - ch.n), i):
                ch.done(j)   # messages of aborted steps we never consumed
            self._rx_last[info[0]] = i
            buf = ch.wait(i, key[0])
            if blocking:
                stream.synchronize()  # released by the sender's GPU, not by us
            self._rx_done = (ch, i)
            self.stats["recv"] += 1
            self.stats["bytes_recv"] += item.total
            out = {}
            for name, dtype, shape, loff, n in item.layout:
                dt = _DTYPES.get(dtype) or getattr(torch, dtype.replace("torch.", ""))
                out[name] = buf[loff:loff + n].view(dt).view(shape)
            return out, item.meta
        if self._waiter is None:
            from connex.adapters.vllm.pp.hostwait import HostFlagWaiter

            self._waiter = HostFlagWaiter(self.device)
        off = self.recv_ring.alloc(max(item.total, ZCOPY_MIN), timeout=RING_TIMEOUT_S)
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
            for key in [k for k in self.push_fallback if k[0] < epoch]:
                del self.push_fallback[key]
            self.inbox.cv.notify_all()
        with self._post_lock:
            gone = [k for k in self._posted if k[0] < epoch]
            posted = [self._posted.pop(k) for k in gone]
            stale = [self._ready.pop(k) for k in [k for k in self._ready if k[0] < epoch]]
        for _off, slot in posted:
            self._waiter.fill(slot, None)
        for peer, hdr in stale:
            self._ack(peer, hdr[4])
        # Streams waiting on channels for messages of older epochs: their
        # sender may never send them (replaced, or hung while alive).
        with self._chan_lock:
            rx = list(self._push_rx.values()) + [c for c in self._rx_chans.values()
                                                 if hasattr(c, "release_below")]
            tx = [c for c in self._tx_chans.values() if hasattr(c, "cancel_below")]
        n = 0
        for ch in {id(c): c for c in rx}.values():
            n += ch.release_below(epoch)
        if n:
            log.info("connex-pp: released %d channel wait(s) of epochs < %d", n, epoch)
        # Pushes of older epochs that still wait for their doorbell or gate
        # are dropped (they would hold their queue).
        for ch in tx:
            ch.cancel_below(epoch)
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
                self.drain_chans()
            except Exception:  # noqa: BLE001
                log.exception("connex-pp: releasing channels failed")
            try:
                ev = self.engine.wait_event(timeout_ms=200)
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
            self.close_chans(lambda k, peer=peer: k[0] == peer)
            for ak in [ak for ak in list(self._announced) if ak[0] == peer]:
                self._announced.pop(ak, None)
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
        self._reap.put(None)
        try:
            self.engine.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _Reap:
    """A push chain whose completion frees ``off`` of the send ring."""
    t: Any
    off: int | None
    tx: Any
    peer_url: Any
    key: tuple
    kind: str        # "h" hidden states, "s" sampled tokens, "w" warm-up
    v: int
    nbytes: int = 0


@dataclass
class _Header:
    layout: list
    total: int
    meta: Any
    ipc: Any = None
    peer: tuple = ()


_pending_frees: deque = deque()
_pending_lock = threading.Lock()


class _PlainUnpickler(pickle.Unpickler):
    """Unpickles only plain data (tuples, lists, dicts, str, bytes, numbers,
    None, bool): every message of this link is built from those. Any global
    (a class or a function) is refused, so a peer that can reach the data
    port cannot make this process run code by sending a crafted pickle."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} not allowed in a pp message")


def loads_plain(data) -> Any:
    import io

    return _PlainUnpickler(io.BytesIO(bytes(data))).load()


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
