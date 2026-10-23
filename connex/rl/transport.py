# SPDX-License-Identifier: Apache-2.0
"""One-sided read transports for weight arenas.

``ConnexRsTransport`` wraps ``connex_rs.Engine`` (UCX; RDMA / GPUDirect when
available, TCP otherwise): the owner registers its arena, readers pull byte
ranges with ``transfer_sync_read``. A dead owner surfaces as an error within
the per-op timeout instead of a hang, which is what lets a broadcast finish
for the survivors.

``LocalTransport`` is an in-process stand-in with the same surface (for CPU
tests and single-process benchmarks), with failure injection.
"""
import os
import socket
import subprocess
import threading
import uuid


def advertise_ip():
    """Address peers should connect to: $CONNEX_ADVERTISE_IP, else the first
    InfiniBand (ib*) IPv4 address, else the address of the host name."""
    ip = os.environ.get("CONNEX_ADVERTISE_IP")
    if ip:
        return ip
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True,
                             timeout=2).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) > 3 and f[1].startswith("ib"):
                return f[3].split("/")[0]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


class TransferError(RuntimeError):
    pass


class LocalTransport:
    """Shared in-process registry: url -> {base_ptr: tensor}."""
    _owners = {}
    _dead = set()
    _lock = threading.Lock()

    def __init__(self, name=None):
        self.url = f"local://{name or uuid.uuid4().hex[:8]}"
        with self._lock:
            self._owners[self.url] = {}
            self._dead.discard(self.url)
        self.reads = 0
        self.bytes = 0

    def register(self, tensor):
        with self._lock:
            self._owners[self.url][tensor.data_ptr()] = tensor
        return tensor.data_ptr()

    def unregister(self, tensor):
        with self._lock:
            self._owners[self.url].pop(tensor.data_ptr(), None)

    def read(self, remote_url, remote_addr, dst, dst_off, length, timeout_ms=None):
        with self._lock:
            if remote_url in self._dead or remote_url not in self._owners:
                raise TransferError(f"peer {remote_url} unreachable")
            for base, t in self._owners[remote_url].items():
                if base <= remote_addr and remote_addr + length <= base + t.numel():
                    src = t[remote_addr - base:remote_addr - base + length]
                    break
            else:
                raise TransferError(f"{remote_url}: address range not registered")
        dst[dst_off:dst_off + length].copy_(src)
        self.reads += 1
        self.bytes += length

    def read_batch(self, remote_url, ops, dst, timeout_ms=None):
        for remote_addr, dst_off, length in ops:
            self.read(remote_url, remote_addr, dst, dst_off, length, timeout_ms)

    def kill(self):
        """Simulate process death: further reads from this url fail."""
        with self._lock:
            self._dead.add(self.url)

    def close(self):
        with self._lock:
            self._owners.pop(self.url, None)


advertise_ip_ = advertise_ip


class ConnexRsTransport:
    def __init__(self, lid=("connex-rl", 0, 0, 0), listen_ip="0.0.0.0", port=0,
                 advertise_ip=None, timeout_ms=10000):
        import connex_rs  # optional dependency
        model, stage, shard, replica = lid
        self.e = connex_rs.Engine(model, stage, shard, replica, listen_ip=listen_ip,
                                  port=port, advertise_ip=advertise_ip or advertise_ip_(),
                                  heartbeat_ms=1000)
        self.url = self.e.local_url
        self.timeout_ms = timeout_ms
        self.reads = 0
        self.bytes = 0

    def register(self, tensor):
        n = tensor.numel() * tensor.element_size()
        rc = self.e.register_memory(tensor.data_ptr(), n)
        if rc != 0:
            raise TransferError(f"register_memory failed: {rc}")
        return tensor.data_ptr()

    def unregister(self, tensor):
        self.e.unregister_memory(tensor.data_ptr())

    def read(self, remote_url, remote_addr, dst, dst_off, length, timeout_ms=None):
        try:
            rc = self.e.transfer_sync_read(remote_url, dst.data_ptr() + dst_off, remote_addr,
                                           length, timeout_ms or self.timeout_ms)
        except Exception as ex:  # connex_rs raises on peer failure / timeout
            raise TransferError(f"{remote_url}: {ex}") from ex
        if rc != 0:
            raise TransferError(f"{remote_url}: read failed rc={rc} "
                                "(-2 peer failed, -3 timeout, -4 invalid)")
        self.reads += 1
        self.bytes += length

    def read_batch(self, remote_url, ops, dst, timeout_ms=None):
        """Issue all reads of ``ops`` [(remote_addr, dst_off, len)] at once
        and wait for all; any failure raises TransferError (the caller
        re-reads the whole batch elsewhere: reads are idempotent)."""
        base = dst.data_ptr()
        to = timeout_ms or self.timeout_ms
        hs, err = [], None
        try:
            for remote_addr, dst_off, length in ops:
                hs.append(self.e.get_async(remote_url, base + dst_off, remote_addr, length, to))
        except Exception as ex:
            err = ex
        for h in hs:
            try:
                if not h.wait(to + 1000):
                    err = err or TimeoutError("read timed out")
            except Exception as ex:
                err = err or ex
        if err is not None:
            raise TransferError(f"{remote_url}: {err}") from err
        self.reads += len(ops)
        self.bytes += sum(o[2] for o in ops)

    def close(self):
        try:
            self.e.close()
        except Exception:
            pass


def make_transport(kind=None, **kw):
    kind = kind or os.environ.get("CONNEX_RL_TRANSPORT", "connex-rs")
    if kind == "local":
        return LocalTransport(kw.get("name"))
    return ConnexRsTransport(**{k: v for k, v in kw.items() if k != "name"})
