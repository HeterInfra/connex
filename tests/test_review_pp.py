# SPDX-License-Identifier: Apache-2.0
"""Regression tests for pipeline-parallel review fixes."""
import socket
import threading

import pytest

from connex.pp.routing import StaticMembership
from connex.pp.testing import LoopbackTransferEngine


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mk(monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    made = []

    def make(stage, n, k=1):
        port = _port()
        url = f"loop-{stage}-{port}"
        ep = core.ConnexTEEndpoint(stage, n, port, "127.0.0.1:1", device="cpu",
                                   membership=StaticMembership(stage, 0),
                                   transfer_engine=LoopbackTransferEngine(url),
                                   local_url=url, inflight=k)
        made.append(ep)
        return ep

    yield make
    for ep in made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass


def _close_within(ep, timeout_s=5.0) -> bool:
    t = threading.Thread(target=ep.close, daemon=True)
    t.start()
    t.join(timeout_s)
    return not t.is_alive()


def test_close_does_not_hang_on_ack_sockets(mk):
    """GDR acks open DEALER sockets; close() must close them, else
    zmq Context.term() blocks forever."""
    ep = mk(1, 2)
    ep._send_ack(f"tcp://127.0.0.1:{_port()}", 7)
    assert ep._ack_socks
    assert _close_within(ep)


def test_close_stops_signal_thread(mk):
    from concurrent.futures import ThreadPoolExecutor
    ep = mk(0, 2, k=2)
    ex = ep._signal_ex = ThreadPoolExecutor(1, thread_name_prefix="connexte_signal")
    ex.submit(lambda: None).result(5)
    assert _close_within(ep)
    assert ex._shutdown and ep._signal_ex is None
