# SPDX-License-Identifier: Apache-2.0
"""connex.rl: arena packing, registry tree fan-out, pull with source failure,
join catch-up, and the rollout pool's no-mixed-versions invariant (CPU)."""
import asyncio
import threading

import pytest
import torch

from connex.rl import (LocalTransport, RegistryClient, RegistryServer, RolloutPool,
                       WeightManifest, WeightPublisher, WeightRegistry, WeightSubscriber,
                       pack_into, unpack)
from connex.rl.fake import FakeInstance, expected, weights_for


def named(v):
    g = torch.Generator().manual_seed(v)
    return [("a", torch.randn(1000, generator=g)), ("b", torch.randn(7, 3, generator=g).half()),
            ("c", torch.randint(0, 9, (5,), generator=g))]


def test_pack_unpack_roundtrip():
    nt = named(1)
    m = WeightManifest.build(1, nt)
    a = torch.empty(m.total, dtype=torch.uint8)
    pack_into(a, m, nt)
    m2 = WeightManifest.from_dict(m.to_dict())
    for (n1, t1), (n2, t2) in zip(nt, unpack(a, m2)):
        assert n1 == n2 and t1.dtype == t2.dtype and torch.equal(t1, t2)
    assert all(e.offset % 256 == 0 for e in m.entries)


def setup(n, fanout=2, chunk=512):
    reg = WeightRegistry(fanout=fanout)
    pub = WeightPublisher(reg, LocalTransport("trainer"))
    subs = [WeightSubscriber(reg, LocalTransport(f"e{i}"), f"e{i}", chunk_bytes=chunk,
                             heartbeat_s=0) for i in range(n)]
    return reg, pub, subs


def check(sub, v):
    for (n1, t1), (n2, t2) in zip(named(v), sub.tensors()):
        assert n1 == n2 and torch.equal(t1, t2)


def test_publish_pull_tree_fanout():
    reg, pub, subs = setup(6, fanout=2)
    v = pub.publish(named(1))
    for s in subs:          # sequential: later pullers read from earlier ones
        s.sync(lambda it: None)
        check(s, v)
    srcs = [s.stats["sources"][0] for s in subs]
    assert srcs[:2] == ["trainer", "trainer"] or srcs[0] == "trainer"
    assert any(x != "trainer" for x in srcs), srcs           # tree, not star
    assert reg.status()["members"] == {f"e{i}": 1 for i in range(6)}


def test_source_dies_mid_pull_survivor_completes():
    reg, pub, subs = setup(3, fanout=1, chunk=256)
    pub.publish(named(1))
    subs[0].sync(lambda it: None)
    v2 = pub.publish(named(2))
    subs[0].sync(lambda it: None)          # e0 holds v2 and is the preferred peer
    # e0 dies while e1 reads from it: kill after the first chunk
    orig = subs[1].t.read
    calls = [0]

    def read(url, *a, **k):
        calls[0] += 1
        if calls[0] == 2:
            subs[0].t.kill()
        return orig(url, *a, **k)
    subs[1].t.read = read
    st = subs[1].sync(lambda it: None)
    check(subs[1], v2)
    assert st["source_failures"] >= 1 and st["sources"][0] == "e0"
    assert "e0" not in reg.status()["holders"][str(v2)]


def test_join_catches_up_to_latest():
    reg, pub, subs = setup(1)
    for v in range(1, 4):
        pub.publish(named(v))
    late = WeightSubscriber(reg, LocalTransport("late"), "late", heartbeat_s=0)
    st = late.sync(lambda it: None)
    assert st["version"] == 3
    check(late, 3)


def test_publisher_double_buffer_waits_for_readers():
    reg, pub, subs = setup(1)
    pub.publish(named(1))
    src = reg.assign(version=1, member="x")     # a reader holds v1's arena
    pub.publish(named(2))                        # other slot: no wait
    done = threading.Event()
    threading.Thread(target=lambda: (pub.publish(named(3)), done.set()), daemon=True).start()
    assert not done.wait(0.3)                    # slot of v1 is busy
    reg.release(version=1, source=src["member"])
    assert done.wait(5)


def test_registry_over_http():
    srv = RegistryServer(WeightRegistry(fanout=2))
    try:
        c = RegistryClient(f"http://127.0.0.1:{srv.port}")
        pub = WeightPublisher(c, LocalTransport("t-http"))
        s = WeightSubscriber(c, LocalTransport("s-http"), "s", heartbeat_s=0)
        pub.publish(named(5))
        s.sync(lambda it: None)
        check(s, 5)
        assert c.status()["members"]["s"] == 1
        with pytest.raises(RuntimeError):
            c.publish(version=1, manifest={}, url="", addr=0)
    finally:
        srv.close()


def test_member_expiry():
    reg = WeightRegistry(member_ttl_s=0.0)
    reg.join(member="m")
    assert reg.expire() == ["m"]


# -- rollout pool -------------------------------------------------------------

def pool_rig(n, tpot=0.002):
    reg = WeightRegistry(fanout=2)
    pub = WeightPublisher(reg, LocalTransport("trainer-p"))
    pub.publish(weights_for(1))
    insts = [FakeInstance(f"r{i}", WeightSubscriber(reg, LocalTransport(f"r{i}"), f"r{i}",
                                                    heartbeat_s=0), tpot_s=tpot)
             for i in range(n)]
    return reg, pub, insts


def assert_consistent(req, res):
    assert res["tokens"] == expected(req, float(res["version"])), "mixed weight versions"


def test_pool_update_never_mixes_versions():
    async def main():
        reg, pub, insts = pool_rig(3)
        pool = RolloutPool(version=1)
        for i in insts:
            await pool.add(i)
        reqs = [{"seed": k, "max_tokens": 30} for k in range(24)]
        tasks = [asyncio.create_task(pool.generate(r)) for r in reqs]
        await asyncio.sleep(0.02)
        pub.publish(weights_for(2))
        await pool.update(2, rolling=1)
        late = [{"seed": 100 + k, "max_tokens": 10} for k in range(6)]
        res = await asyncio.gather(*tasks, *(pool.generate(r) for r in late))
        for r, o in zip(reqs + late, res):
            assert_consistent(r, o)
        assert {o["version"] for o in res[-6:]} == {2}
        assert all(i.version == 2 for i in insts)
    asyncio.run(main())


def test_pool_abort_on_update_restarts_at_new_version():
    async def main():
        reg, pub, insts = pool_rig(2, tpot=0.01)
        pool = RolloutPool(version=1, drain_timeout=0.02)
        for i in insts:
            await pool.add(i)
        reqs = [{"seed": k, "max_tokens": 40} for k in range(4)]
        tasks = [asyncio.create_task(pool.generate(r)) for r in reqs]
        await asyncio.sleep(0.03)
        pub.publish(weights_for(2))
        await pool.update(2)
        res = await asyncio.gather(*tasks)
        for r, o in zip(reqs, res):
            assert_consistent(r, o)
            assert o["version"] == 2
        assert pool.stats["aborted_for_update"] == 4
    asyncio.run(main())


def test_pool_churn_join_and_failure():
    async def main():
        reg, pub, insts = pool_rig(3, tpot=0.003)
        pool = RolloutPool(version=1)
        for i in insts[:2]:
            await pool.add(i)
        reqs = [{"seed": k, "max_tokens": 25} for k in range(30)]
        tasks = [asyncio.create_task(pool.generate(r)) for r in reqs]
        await asyncio.sleep(0.02)
        insts[0].kill()                       # failure mid-rollout
        pub.publish(weights_for(2))
        await pool.add(insts[2])              # joins mid-training: pulls the latest (v2)
        assert insts[2].version == 2 and "r2" not in [s.inst.name for s in pool.serving()]
        await pool.update(2)                  # dead instance r0 is dropped
        res = await asyncio.gather(*tasks)
        for r, o in zip(reqs, res):
            assert_consistent(r, o)
        assert pool.stats["instance_failures"] == 1
        assert pool.stats["completed"] == 30 and pool.stats["failed"] == 0
        assert sorted(pool.slots) == ["r1", "r2"]
    asyncio.run(main())
