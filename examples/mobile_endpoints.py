# SPDX-License-Identifier: Apache-2.0
"""Mobile endpoints on one machine, no GPU: a sender keeps talking to the
replicas of one logical endpoint group while they join, fail, rejoin, migrate
to a new address and leave. Nothing here is pipeline-specific: the group could
be the decode pool of a PD deployment, the FFN workers of an attention-FFN
split, or one stage of several pipeline replicas.

    python examples/mobile_endpoints.py
"""

import socket
import time

from connex.directory import DirectoryClient
from connex.directory_server import DirectoryServer
from connex.elastic import ElasticPeerSet
from connex.ucx_transport import TransportMode, UCXTransport

GROUP = 1  # the logical group the sender talks to (the "stage" field of the LID)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def transport() -> UCXTransport:
    t = UCXTransport("127.0.0.1", free_port(), mode=TransportMode.ZEROMQ,
                     device="cpu", buffer_size=1 << 16, num_buffers=1)
    t.start_sync()
    return t


def until(cond, timeout=5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise SystemExit("timed out")
        time.sleep(0.01)


def main() -> None:
    # Control plane: the directory holds routes, leases and membership events.
    directory = DirectoryServer(f"127.0.0.1:{free_port()}", f"127.0.0.1:{free_port()}")
    directory.start()

    def member(replica: int, rx: UCXTransport) -> DirectoryClient:
        """A worker JOINs under the logical id (model, group, shard, replica)."""
        c = DirectoryClient(directory.bind_addr, "demo", stage_idx=GROUP, replica_idx=replica,
                            local_addresses=[f"127.0.0.1:{rx.local_port}"])
        c.start()
        c.register()
        return c

    # Sender: resolves the group through the directory, never through fixed ranks.
    owner = DirectoryClient(directory.bind_addr, "demo", stage_idx=0)
    owner.start()
    tx = transport()
    peers = ElasticPeerSet(owner, tx, stage_idx=GROUP, drain_grace_s=0.1)
    peers.start()

    rx_a, rx_b = transport(), transport()
    a = member(0, rx_a)                         # replica 0 joins
    b = member(1, rx_b)                         # replica 1 joins (scale up)
    peers.wait_ready(2)
    print("replicas:  ", peers.peers())
    for _ in range(2):
        peers.send(peers.pick(), b"step")
    print("received:  ", rx_a.recv_sync(timeout_ms=2000), rx_b.recv_sync(timeout_ms=2000))

    # Replica 1 dies with a step in flight. The sender detects it (here: as
    # if a reply timed out), reports it, and re-sends the step to a survivor.
    in_flight = b"step-42"
    peers.mark_failed(("demo", GROUP, 0, 1))   # evicted for every sender
    peers.send(peers.pick(), in_flight)
    print("handover:  ", peers.peers(), rx_a.recv_sync(timeout_ms=2000))

    # Replica 1 comes back under the same logical id (scale up again).
    b.register()
    peers.wait_ready(2)
    print("rejoin:    ", peers.peers())

    # Migrate: replica 0 moves to a new address under the same logical id.
    rx_a2 = transport()
    a.register(addresses=[f"127.0.0.1:{rx_a2.local_port}"])
    until(lambda: peers._ready.get(("demo", GROUP, 0, 0), (None, ""))[1]
          .endswith(f":{rx_a2.local_port}"))
    peers.send(("demo", GROUP, 0, 0), b"after-migrate")
    print("migrate:   ", rx_a2.recv_sync(timeout_ms=2000))

    b.deregister()                              # scale down
    until(lambda: peers.peers() == [("demo", GROUP, 0, 0)])
    print("scale-down:", peers.peers())

    peers.stop()
    for c in (a, b, owner):
        c.stop()
    for t in (tx, rx_a, rx_b, rx_a2):
        t.stop_sync()
    directory.stop()


if __name__ == "__main__":
    main()
