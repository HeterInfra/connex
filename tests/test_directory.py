# SPDX-License-Identifier: Apache-2.0
"""JOIN / lookup / LEAVE tests for DirectoryClient against DirectoryServer."""

import socket

import pytest

from connex.directory import DirectoryClient
from connex.directory_server import DirectoryServer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    srv = DirectoryServer(f"127.0.0.1:{_free_port()}")
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def clients(server):
    made = []

    def make(stage, addrs=None):
        c = DirectoryClient(server.bind_addr, "m", stage_idx=stage, local_addresses=addrs)
        c.start()
        made.append(c)
        return c

    yield make
    for c in made:
        c.stop()


def test_join_then_lookup_from_peer(server, clients):
    s0 = clients(0, ["127.0.0.1:6000"])
    s1 = clients(1)
    assert s0.register()
    assert ("m", 0, 0, 0) in server.members()
    info = s1.lookup_stage(0)
    assert info is not None
    assert info.primary_address == "127.0.0.1:6000"
    assert s1.lookup(("m", 5, 0, 0)) is None


def test_leave_removes_member(server, clients):
    s0 = clients(0, ["127.0.0.1:6000"])
    s1 = clients(1)
    assert s0.register()
    assert s1.lookup_stage(0) is not None
    epoch_before = server.epoch
    assert s0.deregister()
    assert server.epoch == epoch_before + 1
    assert ("m", 0, 0, 0) not in server.members()
    s1.invalidate_cache()
    assert s1.lookup_stage(0) is None
    assert not s0.deregister()  # second leave: not a member any more


def test_rejoin_after_leave_and_assignment(server, clients):
    s0 = clients(0, ["127.0.0.1:6000"])
    s1 = clients(1)
    assert s0.register() and s0.deregister()
    assert s0.register(["127.0.0.1:6001"])
    assert s1.request_assignment(0, "forward") == ("127.0.0.1:6001", "m-0-0-0")


def test_migration_updates_address(server, clients):
    s0 = clients(0, ["127.0.0.1:6000"])
    s1 = clients(1)
    assert s0.register()
    assert s0.notify_migration(["127.0.0.1:7000"], target_epoch=5)
    s1.invalidate_cache()
    assert s1.lookup_stage(0).primary_address == "127.0.0.1:7000"


def test_register_without_addresses_fails():
    c = DirectoryClient("127.0.0.1:1", "m", stage_idx=0)
    assert not c.register()
