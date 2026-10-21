# SPDX-License-Identifier: Apache-2.0
"""Two pipeline stages that find each other through the directory and exchange
a tensor. Runs on one machine with no GPU; uses UCX if ucx-py is installed,
otherwise ZeroMQ.

    python examples/two_stage_pipeline.py
"""

import multiprocessing as mp
import socket
import time

import torch

from connex.directory import DirectoryClient
from connex.directory_server import DirectoryServer
from connex.ucx_transport import UCX_AVAILABLE, TransportMode, UCXTransport

MODE = TransportMode.UCX_TCP if UCX_AVAILABLE else TransportMode.ZEROMQ


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stage1(directory: str, port: int, ready, result) -> None:
    """Downstream stage: JOIN the directory, receive one activation, LEAVE."""
    transport = UCXTransport("127.0.0.1", port, mode=MODE, device="cpu",
                             buffer_size=1 << 20, num_buffers=1)
    transport.start_sync()
    client = DirectoryClient(directory, "demo", stage_idx=1,
                             local_addresses=[f"127.0.0.1:{port}"])
    client.start()
    client.register()
    ready.set()

    if MODE == TransportMode.ZEROMQ:
        data = transport.recv_sync(timeout_ms=10_000)
    else:
        _peer, data = transport.recv_from_queue(timeout=10)
    result.put(torch.frombuffer(bytearray(data), dtype=torch.float32).tolist())

    client.deregister()
    client.stop()
    transport.stop_sync()


def main() -> None:
    server = DirectoryServer(f"127.0.0.1:{free_port()}")
    server.start()

    ctx = mp.get_context("spawn")
    ready, result = ctx.Event(), ctx.Queue()
    proc = ctx.Process(target=stage1, args=(server.bind_addr, free_port(), ready, result))
    proc.start()
    ready.wait(30)

    # Upstream stage: discover stage 1 by its logical ID, then send.
    client = DirectoryClient(server.bind_addr, "demo", stage_idx=0)
    client.start()
    peer = None
    for _ in range(50):  # the registration may not be visible yet
        peer = client.lookup_stage(1)
        if peer is not None and peer.primary_address:
            break
        time.sleep(0.1)
    else:
        raise SystemExit("stage 1 did not appear in the directory")
    host, port = peer.primary_address.rsplit(":", 1)

    transport = UCXTransport("127.0.0.1", free_port(), mode=MODE, device="cpu",
                             buffer_size=1 << 20, num_buffers=1)
    transport.start_sync()
    ep = transport.connect_sync(host, int(port))
    transport.send_sync(ep, torch.arange(4, dtype=torch.float32))
    print(f"[{MODE.value}] stage 1 received:", result.get(timeout=15))

    proc.join(10)
    print("members after stage 1 left:", list(server.members()))
    transport.close_endpoint(ep)
    transport.stop_sync()
    client.stop()
    server.stop()


if __name__ == "__main__":
    main()
