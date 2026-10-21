# SPDX-License-Identifier: Apache-2.0
"""Data-plane backends of a pipeline endpoint.

The endpoint moves a message as: sender stages bytes in registered memory,
sends a small ZeroMQ signal (connex.pp.wire), receiver pulls the bytes with
one one-sided read. Any backend with the :class:`TransferEngine` surface can
carry it:

* connex-rs (``connex_rs.Engine``, Rust over UCX; RDMA / TCP, GPU memory
  with GPUDirect) -- :func:`create_connexrs_engine`. Besides the
  TransferEngine surface it reports peer failures (UCX endpoint errors,
  heartbeat) as events, which the endpoint uses as its primary failure
  signal, and it lets a restarted stage rejoin under the same identity;
* Mooncake Transfer Engine (RDMA / TCP; GPU memory with GPUDirect) --
  :func:`create_mooncake_engine`;
* in-process loopback (:class:`connex.pp.testing.LoopbackTransferEngine`) for
  tests;
* none: the endpoint falls back to the ZeroMQ-only path
  (:mod:`connex.pp.zmq_base`).

UCX: the UCX-based communicator (integrations/vllm/connex_communicator.py,
built on :mod:`connex.ucx_transport`) is a separate, older data plane with
its own wire format; it is not routed through this interface yet (see
docs/design/engine-contract.md, "Status").
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Protocol

try:
    from mooncake.engine import TransferEngine as _MooncakeTE  # type: ignore
    MOONCAKE_AVAILABLE = True
except ImportError:  # optional dependency
    _MooncakeTE = None
    MOONCAKE_AVAILABLE = False



def connexrs_module():
    """connex_rs (optional, pip install connex[rs]) or None. Imported only
    when the backend is used: it loads the system UCX into the process."""
    try:
        import connex_rs  # type: ignore
    except ImportError:
        return None
    return connex_rs if hasattr(connex_rs, "Engine") else None


BACKENDS = ("connex-rs", "mooncake", "zmq")
DEFAULT_BACKEND = "auto"


def rdma_present(sysfs: str = "/sys/class/infiniband", dev: str = "/dev/infiniband") -> bool:
    """An RDMA device is usable: listed in sysfs (visible in containers
    too) and its uverbs node is present and accessible (needs the device
    passed into the container)."""
    try:
        if not any(not d.startswith(".") for d in os.listdir(sysfs)):
            return False
        return any(d.startswith("uverbs") and os.access(os.path.join(dev, d), os.R_OK | os.W_OK)
                   for d in os.listdir(dev))
    except OSError:
        return False


def backend_from_env() -> str:
    """CONNEX_PP_BACKEND: connex-rs | mooncake | zmq | auto (default).

    auto = connex-rs when it is installed and an RDMA device is present (the
    setting it was validated in: GPUDirect RDMA, ms failure events,
    same-identity rejoin); otherwise Mooncake when installed (without RDMA,
    UCX's TCP path for one-sided reads is several times slower than
    Mooncake's); otherwise ZeroMQ only. "zmq" = no transfer engine."""
    b = (os.getenv("CONNEX_PP_BACKEND") or DEFAULT_BACKEND).strip().lower()
    b = {"connexrs": "connex-rs", "connex_rs": "connex-rs", "rs": "connex-rs",
         "none": "zmq", "zeromq": "zmq"}.get(b, b)
    if b == "auto":
        import importlib.util

        if importlib.util.find_spec("connex_rs") is not None and rdma_present():
            return "connex-rs"
        return "mooncake" if MOONCAKE_AVAILABLE else "zmq"
    if b not in BACKENDS:
        raise ValueError(f"CONNEX_PP_BACKEND must be one of {BACKENDS} or auto, got {b!r}")
    return b


class TransferEngine(Protocol):
    """The subset of Mooncake's TransferEngine the endpoint uses."""

    def allocate_managed_buffer(self, size: int) -> int: ...
    def free_managed_buffer(self, ptr: int, size: int) -> None: ...
    def write_bytes_to_buffer(self, ptr: int, data: bytes, length: int) -> None: ...
    def read_bytes_from_buffer(self, ptr: int, length: int) -> bytes: ...
    def transfer_sync_read(self, remote_url: str, dst_ptr: int, src_ptr: int, length: int) -> int: ...
    def register_memory(self, ptr: int, length: int) -> int: ...
    def unregister_memory(self, ptr: int) -> int: ...


def _local_ip(stage_idx: int, front_router_addr: str, net, log: logging.Logger,
              *env: str) -> str:
    """First of the given env vars, then CONNEX_HOST_IP / VLLM_HOST_IP, the
    IPv4 of interface net1, else the front router's host."""
    for name in (*env, "CONNEX_HOST_IP", "VLLM_HOST_IP"):
        if os.getenv(name):
            return os.environ[name]
    try:
        import netifaces

        if "net1" in netifaces.interfaces():
            addrs = netifaces.ifaddresses("net1")
            if netifaces.AF_INET in addrs:
                return addrs[netifaces.AF_INET][0]["addr"]
    except Exception:  # noqa: BLE001
        pass
    ip, _ = net.split_host_port(front_router_addr)
    log.warning("Stage %s: Could not get local IP, using front_router host: %s", stage_idx, ip)
    return ip


def _if_ipv4(ifname: str) -> str | None:
    import fcntl
    import socket
    import struct as _st

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            r = fcntl.ioctl(s.fileno(), 0x8915,  # SIOCGIFADDR
                            _st.pack("256s", ifname[:15].encode()))
        return socket.inet_ntoa(r[20:24])
    except OSError:
        return None


def ucx_net_devices_for_ip(ip: str, sysfs: str = "/sys/class/infiniband") -> str | None:
    """"<rdma dev>:1,<netdev>" for the RDMA device whose network interface
    has address ``ip`` (or the only RDMA device); None without RDMA."""
    try:
        devs = sorted(os.listdir(sysfs))
    except OSError:
        return None
    found = []
    for d in devs:
        try:
            nets = sorted(os.listdir(os.path.join(sysfs, d, "device", "net")))
        except OSError:
            nets = []
        for n in nets:
            found.append((d, n))
            if _if_ipv4(n) == ip:
                return f"{d}:1,{n}"
    if len(devs) == 1:
        return f"{devs[0]}:1" + (f",{found[0][1]}" if found else "")
    return None


def create_connexrs_engine(stage_idx: int, replica_idx: int, model: str,
                           front_router_addr: str, net, log: logging.Logger,
                           engine_cls=None):
    """Start a connex-rs engine from the environment.

    CONNEX_RS_LOCAL_IP (else MOONCAKE_LOCAL_IP / CONNEX_HOST_IP /
    VLLM_HOST_IP / net1 / front router host); CONNEX_RS_PORT, else
    CONNEX_RS_BASE_PORT (default MOONCAKE_BASE_PORT + 1000 when that is set,
    else 51000 + 100 * replica) + stage_idx -- a fixed port, so a restarted
    stage comes back at the same URL; CONNEX_RS_HEARTBEAT_MS (default 1000, 0 = off) and
    CONNEX_RS_HEARTBEAT_TIMEOUT_MS (default 3000) for hung/unreachable
    peers; CONNEX_RS_SPIN_US (progress-thread busy poll, default 500);
    CONNEX_RS_NET_DEVICES -> UCX_NET_DEVICES when that is unset: "auto"
    (default) = the RDMA device whose IPoIB interface carries the local IP,
    plus that interface for TCP (e.g. "mlx5_0:1,ibs102"; device names differ
    per host). Pinning matters: UCX otherwise also tries calico / tailscale
    interfaces.

    Returns ``(engine, local_url)``; ``(None, None)`` when connex_rs is not
    installed. Raises on initialisation errors.
    """
    if engine_cls is None:
        mod = connexrs_module()
        if mod is None:
            log.warning("Stage %s: connex_rs not installed (pip install connex[rs])", stage_idx)
            return None, None
        engine_cls = mod.Engine
    # Restarted stages listen on the same port at once.
    os.environ.setdefault("UCX_TCP_CM_REUSEADDR", "y")
    ip = _local_ip(stage_idx, front_router_addr, net, log, "CONNEX_RS_LOCAL_IP", "MOONCAKE_LOCAL_IP")
    nd = os.getenv("CONNEX_RS_NET_DEVICES", "auto")
    if nd == "auto":
        nd = ucx_net_devices_for_ip(ip)
    if nd and not os.getenv("UCX_NET_DEVICES"):
        os.environ["UCX_NET_DEVICES"] = nd
        log.info("Stage %s: UCX_NET_DEVICES=%s", stage_idx, nd)
    if os.getenv("CONNEX_RS_PORT"):
        port = int(os.environ["CONNEX_RS_PORT"])
    else:
        if os.getenv("CONNEX_RS_BASE_PORT"):
            base = int(os.environ["CONNEX_RS_BASE_PORT"])
        elif os.getenv("MOONCAKE_BASE_PORT"):  # already per replica in our manifests
            base = int(os.environ["MOONCAKE_BASE_PORT"]) + 1000
        else:
            base = 51000 + 100 * replica_idx
        port = base + stage_idx
    hb = int(os.getenv("CONNEX_RS_HEARTBEAT_MS", "1000"))
    hb_to = int(os.getenv("CONNEX_RS_HEARTBEAT_TIMEOUT_MS", "3000"))
    if hb and hb_to < 1000:
        log.warning("Stage %s: CONNEX_RS_HEARTBEAT_TIMEOUT_MS=%s < 1000 gives false failures "
                    "under load; using 1000", stage_idx, hb_to)
        hb_to = 1000
    engine = engine_cls(model, stage_idx, 0, replica_idx, listen_ip=ip, port=port,
                        heartbeat_ms=hb, heartbeat_timeout_ms=hb_to,
                        spin_us=int(os.getenv("CONNEX_RS_SPIN_US", "500")))
    url = engine.local_url
    log.info("Stage %s: connex-rs engine at %s (heartbeat %s/%s ms)", stage_idx, url, hb, hb_to)
    return engine, url


def create_mooncake_engine(stage_idx: int, front_router_addr: str, net,
                           log: logging.Logger, engine_cls=None):
    """Initialise a Mooncake TransferEngine from the environment.

    MOONCAKE_METADATA_SERVER (required; "P2PHANDSHAKE" for no metadata
    service), MOONCAKE_BASE_PORT (+stage_idx), MOONCAKE_PROTOCOL (tcp|rdma),
    MOONCAKE_DEVICE_NAME, MOONCAKE_METADATA_BACKEND (etcd|redis),
    MOONCAKE_LOCAL_IP / CONNEX_HOST_IP / VLLM_HOST_IP, else the IPv4 of
    interface net1, else the front router's host.

    Returns ``(engine, local_url)`` or ``(None, None)`` when unavailable
    (the caller falls back to ZeroMQ). Raises on initialisation errors.
    """
    engine_cls = engine_cls or _MooncakeTE
    if engine_cls is None:
        log.info("Stage %s: Mooncake not available, using ZeroMQ fallback", stage_idx)
        return None, None
    metadata_server = os.getenv("MOONCAKE_METADATA_SERVER")
    if metadata_server is None:
        log.warning("Stage %s: MOONCAKE_METADATA_SERVER not set, using ZeroMQ fallback", stage_idx)
        return None, None

    base_port = int(os.getenv("MOONCAKE_BASE_PORT", "50000"))
    protocol = os.getenv("MOONCAKE_PROTOCOL", "tcp")
    device_name = os.getenv("MOONCAKE_DEVICE_NAME", "")
    metadata_backend = os.getenv("MOONCAKE_METADATA_BACKEND")

    local_ip = (os.getenv("MOONCAKE_LOCAL_IP") or os.getenv("CONNEX_HOST_IP")
                or os.getenv("VLLM_HOST_IP"))
    if not local_ip:
        try:
            import netifaces

            if "net1" in netifaces.interfaces():
                addrs = netifaces.ifaddresses("net1")
                if netifaces.AF_INET in addrs:
                    local_ip = addrs[netifaces.AF_INET][0]["addr"]
                    log.info("Stage %s: Using net1 IP for Mooncake: %s", stage_idx, local_ip)
        except Exception:  # noqa: BLE001
            pass
    if not local_ip:
        local_ip, _ = net.split_host_port(front_router_addr)
        log.warning("Stage %s: Could not get local IP, using front_router host: %s",
                    stage_idx, local_ip)

    local_url = net.join_host_port(local_ip, base_port + stage_idx)
    log.info("Stage %s: ConnexTE Mooncake local URL: %s", stage_idx, local_url)

    engine = engine_cls()
    if metadata_backend is None:
        engine.initialize(local_url, metadata_server, protocol, device_name)
    else:
        supported = ["etcd", "redis"]
        backend = metadata_backend.lower()
        if backend not in supported:
            raise ValueError(f"Mooncake metadata backend must be one of {supported}, got: {backend}")
        engine.initialize_ext(local_url, metadata_server, protocol, device_name, backend)
    # P2P handshake: peers open our segment by ip:rpc_port (Mooncake picks the
    # RPC port itself), not by the local server name.
    if str(metadata_server).upper() == "P2PHANDSHAKE" and hasattr(engine, "get_rpc_port"):
        local_url = net.join_host_port(local_ip, int(engine.get_rpc_port()))
        log.info("Stage %s: P2P handshake target %s", stage_idx, local_url)
    return engine, local_url


def verify_engine(engine) -> Optional[str]:
    """Allocate/free a small managed buffer. Returns an error string or None."""
    try:
        buf = engine.allocate_managed_buffer(1024)
        if buf <= 0:
            raise RuntimeError("Test buffer allocation returned invalid pointer")
        engine.free_managed_buffer(buf, 1024)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)
