# SPDX-License-Identifier: Apache-2.0
"""Loopback send/recv tests for UCXTransport (ZeroMQ fallback and UCX)."""

import socket

import pytest

from connex.ucx_transport import UCX_AVAILABLE, TransportMode, UCXTransport


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_zmq_loopback_send_recv():
    port = _free_port()
    rx = UCXTransport("127.0.0.1", port, mode=TransportMode.ZEROMQ, device="cpu",
                      buffer_size=1024, num_buffers=1)
    tx = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.ZEROMQ,
                      device="cpu", buffer_size=1024, num_buffers=1)
    rx.start_sync()
    tx.start_sync()
    try:
        ep = tx.connect_sync("127.0.0.1", port)
        tx.send_sync(ep, b"hello connex")
        assert rx.recv_sync(timeout_ms=2000) == b"hello connex"
        assert tx.get_stats()["messages_sent"] == 1
        assert rx.get_stats()["messages_received"] == 1
    finally:
        tx.stop_sync()
        rx.stop_sync()


def test_recv_timeout_returns_none():
    rx = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.ZEROMQ,
                      device="cpu", buffer_size=1024, num_buffers=1)
    rx.start_sync()
    try:
        assert rx.recv_sync(timeout_ms=50) is None
    finally:
        rx.stop_sync()


def _ucx_receiver(port, out):
    rx = UCXTransport("127.0.0.1", port, mode=TransportMode.UCX_TCP,
                      device="cpu", buffer_size=1024, num_buffers=1)
    rx.start_sync()
    out.put("ready")
    got = rx.recv_from_queue(timeout=10)
    out.put(bytes(got[1]) if got else None)
    rx.stop_sync()


@pytest.mark.skipif(not UCX_AVAILABLE, reason="ucx-py not installed")
def test_ucx_tcp_loopback_send_recv():
    # UCX is initialised once per process, so the receiver runs in a child
    # process, matching the one-transport-per-stage deployment model.
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    port = _free_port()
    out = ctx.Queue()
    proc = ctx.Process(target=_ucx_receiver, args=(port, out))
    proc.start()
    try:
        assert out.get(timeout=30) == "ready"
        tx = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.UCX_TCP,
                          device="cpu", buffer_size=1024, num_buffers=1)
        tx.start_sync()
        ep = tx.connect_sync("127.0.0.1", port)
        tx.send_sync(ep, b"hello ucx")
        assert out.get(timeout=15) == b"hello ucx"
        assert tx.close_endpoint(ep)
        assert not tx.close_endpoint(ep)
        tx.stop_sync()
    finally:
        proc.join(10)
        if proc.is_alive():
            proc.kill()


def test_close_endpoint_zmq():
    port = _free_port()
    rx = UCXTransport("127.0.0.1", port, mode=TransportMode.ZEROMQ, device="cpu",
                      buffer_size=1024, num_buffers=1)
    tx = UCXTransport("127.0.0.1", _free_port(), mode=TransportMode.ZEROMQ,
                      device="cpu", buffer_size=1024, num_buffers=1)
    rx.start_sync()
    tx.start_sync()
    try:
        ep = tx.connect_sync("127.0.0.1", port)
        assert tx.close_endpoint(ep)
        assert not tx.close_endpoint(ep)  # already gone
        ep2 = tx.connect_sync("127.0.0.1", port)  # reconnect creates a new socket
        assert ep2 is not ep
        tx.send_sync(ep2, b"again")
        assert rx.recv_sync(timeout_ms=2000) == b"again"
    finally:
        tx.stop_sync()
        rx.stop_sync()


def _start_on_busy_port(port, out, fallback=0):
    from connex.ucx_transport import UCXListenerError
    t = UCXTransport("127.0.0.1", port, mode=TransportMode.UCX_TCP, device="cpu",
                     buffer_size=1024, num_buffers=1, port_fallback=fallback)
    try:
        t.start_sync()
        out.put(f"started:{t.local_port}")
    except UCXListenerError as e:
        out.put("UCXListenerError" if str(port) in str(e) else f"bad message: {e}")
    except Exception as e:  # wrong exception type
        out.put(f"{type(e).__name__}: {e}")


@pytest.mark.skipif(not UCX_AVAILABLE, reason="ucx-py not installed")
def test_ucx_listener_on_busy_port_raises():
    import multiprocessing as mp

    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        p = ctx.Process(target=_start_on_busy_port, args=(port, out))
        p.start()
        assert out.get(timeout=60) == "UCXListenerError"
        p.join(10)
        if p.is_alive():
            p.kill()

        # With fallback the transport moves to a later port and reports it.
        p = ctx.Process(target=_start_on_busy_port, args=(port, out, 4))
        p.start()
        got = out.get(timeout=60)
        assert got.startswith("started:") and int(got.split(":")[1]) != port
        p.join(10)
        if p.is_alive():
            p.kill()
    finally:
        blocker.close()
