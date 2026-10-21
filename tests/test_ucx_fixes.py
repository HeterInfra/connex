# SPDX-License-Identifier: Apache-2.0
"""Regression tests for UCX backend fixes (see docs/design/ucx-transport-testing.md).

B  UCX_SOCKADDR_TLS_PRIORITY is passed as an init option, and RDMA mode never
   silently downgrades to TCP.
A  RDMA mode requires the UCP registration cache for rendezvous, else eager.
C  A peer without a UCX address is reached over ZeroMQ, never a guessed port.
D  GPU-direct send/recv between two processes on two GPUs.
E  stop_sync shuts the UCX loop down without "Task exception was never retrieved".
"""

import ast
import logging
import os
import socket
import subprocess
import sys
import textwrap

import pytest

import connex.ucx_transport as ut
from connex.ucx_transport import UCX_AVAILABLE, TransportMode, UCXTransport

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeUcp:
    """Stands in for the ucp module; ``fail(options)`` decides if init raises."""

    def __init__(self, fail):
        self.fail = fail
        self.calls = []

    def init(self, options):
        self.calls.append((dict(options), os.environ.get("UCX_SOCKADDR_TLS_PRIORITY")))
        err = self.fail(options)
        if err:
            raise err


def _transport(mode):
    return UCXTransport("127.0.0.1", _free_port(), mode=mode, device="cpu",
                        buffer_size=1024, num_buffers=1)


# ------------------------------------------------------------------- bug B

def test_sockaddr_env_passed_as_option_not_env(monkeypatch):
    # Emulates ucx-py's ucx_config_to_dict choking on the env var.
    def fail(options):
        if "UCX_SOCKADDR_TLS_PRIORITY" in os.environ:
            return ValueError("too many values to unpack (expected 2)")
        return None

    fake = _FakeUcp(fail)
    monkeypatch.setattr(ut, "ucp", fake, raising=False)
    monkeypatch.setattr(ut, "UCX_AVAILABLE", True)  # fake ucp works without ucx-py
    monkeypatch.setenv("UCX_SOCKADDR_TLS_PRIORITY", "rdmacm")
    monkeypatch.delenv("UCX_TLS", raising=False)
    t = _transport(TransportMode.UCX_RDMA)
    opts = t._build_ucp_options()
    assert opts["SOCKADDR_TLS_PRIORITY"] == "rdmacm"
    t._init_ucp(opts)
    assert t.mode == TransportMode.UCX_RDMA
    assert len(fake.calls) == 1
    assert fake.calls[0][1] is None  # env hidden during ucp.init
    assert fake.calls[0][0]["SOCKADDR_TLS_PRIORITY"] == "rdmacm"
    assert os.environ["UCX_SOCKADDR_TLS_PRIORITY"] == "rdmacm"  # restored


def test_rdma_init_failure_raises_instead_of_tcp_downgrade(monkeypatch):
    fake = _FakeUcp(lambda o: RuntimeError("no RDMA device"))
    monkeypatch.setattr(ut, "ucp", fake, raising=False)
    monkeypatch.setattr(ut, "UCX_AVAILABLE", True)  # fake ucp works without ucx-py
    monkeypatch.delenv("CONNEX_UCX_ALLOW_TCP_FALLBACK", raising=False)
    t = _transport(TransportMode.UCX_RDMA)
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        t._init_ucp(t._build_ucp_options())
    assert t.mode == TransportMode.UCX_RDMA


def test_rdma_tcp_downgrade_only_when_allowed_and_loud(monkeypatch, caplog):
    fake = _FakeUcp(lambda o: None if o.get("TLS") == "tcp,cuda_copy" else RuntimeError("no RDMA"))
    monkeypatch.setattr(ut, "ucp", fake, raising=False)
    monkeypatch.setattr(ut, "UCX_AVAILABLE", True)  # fake ucp works without ucx-py
    monkeypatch.setenv("CONNEX_UCX_ALLOW_TCP_FALLBACK", "1")
    t = _transport(TransportMode.UCX_RDMA)
    opts = t._build_ucp_options()
    with caplog.at_level(logging.ERROR, logger="connex.ucx_transport"):
        t._init_ucp(opts)
    assert t.mode == TransportMode.UCX_TCP
    assert opts["TLS"] == "tcp,cuda_copy"
    assert any("DOWNGRADING" in r.getMessage() for r in caplog.records)


def _start_with_env_child(out):
    t = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.UCX_TCP,
                     device="cpu", buffer_size=1024, num_buffers=1)
    t.start_sync()
    out.put((t.mode.value, t.get_ucx_config()["sockaddr_tls_priority"]))
    t.stop_sync()


@pytest.mark.skipif(not UCX_AVAILABLE, reason="ucx-py not installed")
def test_real_ucp_init_with_sockaddr_env(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setenv("UCX_SOCKADDR_TLS_PRIORITY", "tcp")
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    p = ctx.Process(target=_start_with_env_child, args=(out,))
    p.start()
    try:
        assert out.get(timeout=60) == ("ucx_tcp", "tcp")
    finally:
        p.join(10)
        if p.is_alive():
            p.kill()


# ------------------------------------------------------------------- bug A

def test_rdma_requires_rcache_for_rendezvous(monkeypatch):
    monkeypatch.setattr(ut, "UCX_AVAILABLE", True)  # option building needs no ucx-py
    monkeypatch.setattr(ut, "_USER_RNDV_THRESH", None)
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.delenv("UCX_RCACHE_ENABLE", raising=False)
    opts = _transport(TransportMode.UCX_RDMA)._build_ucp_options()
    assert opts.get("RCACHE_ENABLE") == "y"
    assert "RNDV_THRESH" not in opts
    tcp = _transport(TransportMode.UCX_TCP)._build_ucp_options()
    assert tcp["RNDV_THRESH"] == "inf" and "RCACHE_ENABLE" not in tcp


def test_rdma_without_rcache_falls_back_to_eager(monkeypatch):
    monkeypatch.setattr(ut, "_USER_RNDV_THRESH", None)
    monkeypatch.setenv("CONNEX_UCX_RCACHE_RETRIES", "3")
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.delenv("UCX_RCACHE_ENABLE", raising=False)
    fake = _FakeUcp(lambda o: RuntimeError("Unsupported operation")
                    if o.get("RCACHE_ENABLE") == "y" else None)
    monkeypatch.setattr(ut, "ucp", fake, raising=False)
    monkeypatch.setattr(ut, "UCX_AVAILABLE", True)  # fake ucp works without ucx-py
    t = _transport(TransportMode.UCX_RDMA)
    opts = t._build_ucp_options()
    t._init_ucp(opts)
    assert len(fake.calls) == 4  # 3 rcache attempts + eager
    assert opts["RNDV_THRESH"] == "inf"
    assert "RCACHE_ENABLE" not in opts
    assert t.mode == TransportMode.UCX_RDMA


# ------------------------------------------------------------------- bug C

def _load_resolver():
    # connex_communicator imports vLLM; load just the pure helper.
    path = os.path.join(ROOT, "integrations", "vllm", "connex_communicator.py")
    tree = ast.parse(open(path).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_peer_ucx_address")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns["resolve_peer_ucx_address"]


def test_peer_ucx_address_resolution():
    resolve = _load_resolver()
    assert resolve("10.0.0.5:51234") == ("10.0.0.5", 51234)
    assert resolve(" 10.0.0.5:51234 ") == ("10.0.0.5", 51234)
    assert resolve("[fe80::1]:7000") == ("fe80::1", 7000)
    # No UCX address advertised -> ZeroMQ path (None), never router_port + 1000.
    for bad in (None, "", "10.0.0.5", "10.0.0.5:", ":5000", "10.0.0.5:abc", "10.0.0.5:0",
                "10.0.0.5:70000"):
        assert resolve(bad) is None, bad


def test_communicator_has_no_guessed_ucx_port():
    src = open(os.path.join(ROOT, "integrations", "vllm", "connex_communicator.py")).read()
    assert "peer_ucx_port = self.router_port + 1000" not in src


# ------------------------------------------------------------------- bug D

def _gpu_rx(port, ready, out):
    import torch
    torch.cuda.set_device(1)
    t = UCXTransport("127.0.0.1", port, mode=TransportMode.UCX_TCP, device="cuda",
                     port_fallback=16)
    t.start_sync()
    ready.put(t.local_port)
    got = []
    for _ in range(3):
        r = t.recv_from_queue(timeout=20)
        if r is None:
            got.append(None)
            break
        d = r[1]
        if isinstance(d, torch.Tensor):
            got.append(("T", d.device.index, tuple(d.shape), str(d.dtype), d.float().sum().item()))
        else:
            got.append(("B", bytes(d)))
    out.put(got)
    t.stop_sync()


def _cuda_pair_available():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        return False


@pytest.mark.skipif(not UCX_AVAILABLE or not _cuda_pair_available(),
                    reason="needs ucx-py and 2 CUDA devices")
def test_gpu_direct_two_process():
    import multiprocessing as mp
    import torch

    ctx = mp.get_context("spawn")
    ready, out = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_gpu_rx, args=(_free_port(), ready, out))
    p.start()
    try:
        port = ready.get(timeout=60)
        torch.cuda.set_device(0)
        tx = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.UCX_TCP, device="cuda",
                          port_fallback=16)
        tx.start_sync()
        ep = tx.connect_sync("127.0.0.1", port)
        a = torch.arange(1024, device="cuda:0", dtype=torch.float16).reshape(4, 256)
        b = torch.ones(3, 5, device="cuda:0")
        tx.send_batch_gpu_direct(ep, b"HEADER01", [a, b])
        got = out.get(timeout=60)
        tx.stop_sync()
        assert got == [
            ("B", b"HEADER01"),
            ("T", 1, (4, 256), "torch.float16", 523776.0),
            ("T", 1, (3, 5), "torch.float32", 15.0),
        ]
    finally:
        p.join(10)
        if p.is_alive():
            p.kill()


# ------------------------------------------------------------------- bug E

@pytest.mark.skipif(not UCX_AVAILABLE, reason="ucx-py not installed")
def test_stop_sync_no_task_exception_at_exit(tmp_path):
    script = textwrap.dedent(f"""
        import multiprocessing as mp, socket, sys
        sys.path.insert(0, {ROOT!r})
        from connex.ucx_transport import UCXTransport, TransportMode

        def fp():
            s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

        def rx(port, ready, out):
            t = UCXTransport("127.0.0.1", port, mode=TransportMode.UCX_TCP, device="cpu",
                             buffer_size=1024, num_buffers=1, port_fallback=16)
            t.start_sync(); ready.put(t.local_port)
            r = t.recv_from_queue(timeout=20)
            out.put(bytes(r[1]) if r else None)
            t.stop_sync()

        if __name__ == "__main__":
            ctx = mp.get_context("spawn"); ready = ctx.Queue(); out = ctx.Queue()
            p = ctx.Process(target=rx, args=(fp(), ready, out)); p.start(); port = ready.get(timeout=60)
            tx = UCXTransport("127.0.0.1", fp(), mode=TransportMode.UCX_TCP, device="cpu",
                              buffer_size=1024, num_buffers=1, port_fallback=16)
            tx.start_sync()
            ep = tx.connect_sync("127.0.0.1", port)
            for _ in range(3):
                tx.send_sync(ep, b"x" * 4096)
            assert out.get(timeout=30) == b"x" * 4096
            tx.stop_sync()
            p.join(20)
            print("DONE")
    """)
    path = tmp_path / "stop_clean.py"
    path.write_text(script)
    for _ in range(2):
        r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                           timeout=180, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "DONE" in r.stdout
        for bad in ("never retrieved", "Bad file descriptor", "Task was destroyed"):
            assert bad not in r.stderr, r.stderr[-2000:]
