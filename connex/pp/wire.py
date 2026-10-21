# SPDX-License-Identifier: Apache-2.0
"""Connex PP wire protocol (pure stdlib; see docs/design/engine-contract.md).

Every data-plane message is ONE ZeroMQ frame sent from a DEALER to the
receiving stage's ROUTER (the ROUTER prepends the sender identity frame).
Two kinds exist:

* data signal::

      b"CONNEXTE" | !Q src_ptr | !Q total | !I url_len | url | !Q corr | tail

  ``src_ptr``/``total`` name ``total`` bytes in the sender's registered
  memory at transfer-engine segment ``url``; the receiver pulls them with
  one RDMA READ. ``corr`` is the step correlation id (0 = none). ``tail`` is

  - empty: host-buffer payload (``!Qd`` tensor_len, send_ts, safetensors
    bytes, pickled non-tensor dict), see :func:`encode_host_payload`;
  - ``b"G1" + pickle({"layout", "nt"})``: legacy GPUDirect header;
  - ``b"G2" + !QIIHH(token, layout_len, nt_len, ack_ep_len, origin_len)
    + layout + ack_ep + origin + nt``: GPUDirect header v2, where the
    payload is the raw tensor bytes laid out per ``layout``.

* ack: ``b"CXACK" + !Q token`` (receiver -> sender's ROUTER, G2 only):
  the RDMA read of the region leased under ``token`` completed.

A ZeroMQ-only fallback (no transfer engine) uses a different, multi-frame
format, see :mod:`connex.pp.zmq_base`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

WIRE_MAGIC = b"CONNEXTE"
ACK_MARK = b"CXACK"
ACK = struct.Struct("!Q")
SIG_FIXED = struct.Struct("!QQI")
CORR = struct.Struct("!Q")
HOST_HDR = struct.Struct("!Qd")
G2 = struct.Struct("!QIIHH")
TAG_G1 = b"G1"
TAG_G2 = b"G2"

# Highest header revision this implementation sends by default / accepts.
WIRE_VERSION = 2

# Reserved non-tensor keys (the engine must not use them for its own data).
KEY_ORIGIN = "__stage0_endpoint__"   # stage-0 ROUTER endpoint for the reply
KEY_SEND_TS = "__send_timestamp__"   # dropped by receivers

# dtype codes of the G2 layout. Append only: codes are on the wire.
DT_NAMES = ("float32", "float16", "bfloat16", "float64", "int64", "int32",
            "int16", "int8", "uint8", "bool", "float8_e4m3fn", "float8_e5m2")
DT_CODE = {n: i for i, n in enumerate(DT_NAMES)}


def align(n: int, a: int = 256) -> int:
    return (n + a - 1) // a * a


# ---- G2 layout ---------------------------------------------------------------
# layout: !H n, then per tensor !HBB(keylen, dtype code, ndim) key
# !{ndim}q shape !QQ(offset, nbytes).

def encode_layout(layout) -> bytes | None:
    """layout = [(key, dtype_name, shape, offset, nbytes)]. None if a dtype
    has no code (caller falls back to the G1 header)."""
    parts = [struct.pack("!H", len(layout))]
    for k, dt, shape, off, nb in layout:
        code = DT_CODE.get(dt)
        if code is None:
            return None
        kb = k.encode("utf-8")
        parts.append(struct.pack("!HBB", len(kb), code, len(shape)))
        parts.append(kb)
        parts.append(struct.pack(f"!{len(shape)}q", *shape))
        parts.append(struct.pack("!QQ", off, nb))
    return b"".join(parts)


def decode_layout(b: bytes) -> list:
    (n,), pos, out = struct.unpack_from("!H", b, 0), 2, []
    for _ in range(n):
        klen, code, ndim = struct.unpack_from("!HBB", b, pos)
        pos += 4
        k = b[pos:pos + klen].decode("utf-8")
        pos += klen
        shape = struct.unpack_from(f"!{ndim}q", b, pos)
        pos += 8 * ndim
        off, nb = struct.unpack_from("!QQ", b, pos)
        pos += 16
        out.append((k, DT_NAMES[code], tuple(shape), off, nb))
    return out


def encode_g2(token: int, layout_b: bytes, ack_ep: bytes, origin: bytes, nt_b: bytes) -> bytes:
    return (TAG_G2 + G2.pack(token, len(layout_b), len(nt_b), len(ack_ep), len(origin))
            + layout_b + ack_ep + origin + nt_b)


def decode_g2(body: bytes):
    """body = tail after b"G2". Returns (token, layout_b, ack_ep, origin, nt_b)."""
    token, ll, nl, al, ol = G2.unpack_from(body, 0)
    p = G2.size
    layout_b = body[p:p + ll]; p += ll
    ack_ep = body[p:p + al].decode("utf-8"); p += al
    origin = body[p:p + ol].decode("utf-8"); p += ol
    nt_b = body[p:p + nl]
    return token, layout_b, ack_ep, origin, nt_b


class LayoutCodec:
    """Sender: (keys, dtypes, shapes) -> (layout, total, layout bytes).
    Receiver: layout bytes -> parsed layout. Both bounded caches, so the
    steady state does no pickling and no layout parsing."""

    def __init__(self, cap: int = 512):
        self.cap = cap
        self.enc: dict = {}
        self.dec: dict[bytes, list] = {}

    def layout_for(self, key, sizes):
        """key = tuple((name, dtype_str, shape)); sizes = per-tensor nbytes."""
        hit = self.enc.get(key)
        if hit is None:
            layout, total = [], 0
            for (k, dt, shape), nb in zip(key, sizes):
                layout.append((k, dt, shape, total, nb))
                total = align(total + nb)
            hit = (layout, total, encode_layout(layout))
            if len(self.enc) >= self.cap:
                self.enc.clear()
            self.enc[key] = hit
        return hit

    def parse(self, b: bytes) -> list:
        hit = self.dec.get(b)
        if hit is None:
            hit = decode_layout(b)
            if len(self.dec) >= self.cap:
                self.dec.clear()
            self.dec[b] = hit
        return hit


# ---- data signal ---------------------------------------------------------------

def encode_signal(src_ptr: int, total: int, url: bytes, corr: int, tail: bytes = b"") -> bytes:
    return WIRE_MAGIC + SIG_FIXED.pack(src_ptr, total, len(url)) + url + CORR.pack(corr) + tail


@dataclass
class Signal:
    src_ptr: int
    total: int
    url: str
    corr: int | None     # None: sender predates correlation ids
    tail: bytes          # bytes after corr (b"", b"G1..." or b"G2...")

    @property
    def tag(self) -> bytes:
        return self.tail[:2]


def decode_signal(data: bytes) -> Signal:
    if not data.startswith(WIRE_MAGIC):
        raise ValueError("not a CONNEXTE signal")
    meta = data[len(WIRE_MAGIC):]
    if len(meta) < SIG_FIXED.size:
        raise ValueError("Invalid metadata length")
    src_ptr, total, ul = SIG_FIXED.unpack_from(meta, 0)
    if len(meta) < SIG_FIXED.size + ul:
        raise ValueError("Incomplete sender URL")
    url = meta[SIG_FIXED.size:SIG_FIXED.size + ul].decode("utf-8")
    rest = meta[SIG_FIXED.size + ul:]
    corr = CORR.unpack_from(rest, 0)[0] if len(rest) >= 8 else None
    return Signal(src_ptr, total, url, corr, rest[8:] if len(rest) >= 8 else b"")


def signal_corr(sig: bytes) -> int:
    """Correlation id of an encoded signal (for tracing)."""
    ul = struct.unpack("!I", sig[24:28])[0]
    return CORR.unpack(sig[28 + ul:][:8])[0]


def encode_ack(token: int) -> bytes:
    return ACK_MARK + ACK.pack(token)


def decode_ack(data: bytes) -> int | None:
    if data.startswith(ACK_MARK) and len(data) >= len(ACK_MARK) + ACK.size:
        return ACK.unpack_from(data, len(ACK_MARK))[0]
    return None


# ---- correlation ids ---------------------------------------------------------------
# corr = (incarnation << 32) | seq. incarnation = 32 random bits chosen by each
# stage-0 process at start; seq counts its steps (1, 2, ...). Later stages echo
# the id of the step they answer. 0 = no id.

def corr_incarnation(corr: int) -> int:
    return corr >> 32


def corr_seq(corr: int) -> int:
    return corr & 0xFFFFFFFF
