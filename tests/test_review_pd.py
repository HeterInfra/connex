# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Regression tests for PD / resume-proxy review fixes."""
import asyncio

import pytest

from connex.serving.proxy import Pool, Proxy, StreamBroken


class _BreakAfter(Proxy):
    """Every attempt emits ``n`` tokens (optionally with a finish_reason on
    the last) and then breaks before [DONE]."""

    def __init__(self, n, finish=None, **kw):
        super().__init__(Pool(["http://a", "http://b"]), probe_interval=0, **kw)
        self.n, self.fin, self.calls = n, finish, 0

    async def _attempt(self, st, ep, path):
        self.calls += 1
        base = st.k
        for i in range(self.n):
            last = i == self.n - 1
            c = {"id": "x", "created": 0, "model": "m",
                 "choices": [{"index": 0, "text": f" w{i}",
                              "finish_reason": self.fin if last else None}],
                 "usage": {"completion_tokens": i + 1}}
            out = st.on_chunk(c, base)
            if out is not None:
                yield out
        raise StreamBroken("EOF before [DONE]")


async def _collect(proxy, body):
    out = []
    async for c in proxy.generate("completions", body):
        out.append(c)
    return out


def test_break_after_max_tokens_is_success_not_failure():
    p = _BreakAfter(4)
    out = asyncio.run(_collect(p, {"prompt": "a", "max_tokens": 4}))
    st = out[-1]
    assert st.k == 4 and st.finish == "length"
    assert p.calls == 1 and p.stats["failed"] == 0


def test_break_after_finish_reason_does_not_resume():
    p = _BreakAfter(3, finish="stop")
    out = asyncio.run(_collect(p, {"prompt": "a", "max_tokens": 10}))
    assert p.calls == 1, "resumed past a finished request: duplicate generation"
    assert out[-1].finish == "stop" and out[-1].k == 3


def test_max_tokens_none_falls_back_to_max_completion_tokens():
    p = _BreakAfter(2)
    out = asyncio.run(_collect(p, {"prompt": "a", "max_tokens": None,
                                   "max_completion_tokens": 2}))
    assert p.calls == 1 and out[-1].k == 2


def test_expired_blacklist_rearms_dead_event_without_prober():
    async def go():
        pool = Pool(["http://a"], blacklist_s=0.01)
        ep = pool.eps["http://a"]
        pool.mark_bad(ep, "x")
        assert ep.dead.is_set()
        await asyncio.sleep(0.02)
        assert pool.pick() is ep
        assert not ep.dead.is_set(), "stale dead event would abort every first-token wait"
    asyncio.run(go())


def test_stale_release_does_not_free_newer_readers_hold():
    from connex.pd.transfer import KVProducer

    class _Plane:
        name, local_url = "loopback", "x"

    prod = KVProducer(_Plane(), [], bind_ip="127.0.0.1", register=False)
    try:
        prod.hold("r1", "T", [1, 2])
        assert prod._handle({"op": "acquire", "tid": "T", "who": "decodeA"})["ok"]
        # decode A dies/aborts; decode B re-acquires the same transfer
        assert prod._handle({"op": "acquire", "tid": "T", "who": "decodeB"})["ok"]
        # A's late release must not free B's blocks
        prod._handle({"op": "release", "tid": "T", "who": "decodeA"})
        assert prod.held() == 1 and prod.take_finished() == set()
        prod._handle({"op": "release", "tid": "T", "who": "decodeB"})
        assert prod.held() == 0 and prod.take_finished() == {"r1"}
        # releases without ``who`` (older consumers) keep working
        prod.hold("r2", "U", [3])
        prod._handle({"op": "acquire", "tid": "U", "who": "decodeA"})
        prod._handle({"op": "release", "tid": "U"})
        assert prod.held() == 0
    finally:
        prod.close()


def test_pd_admin_requires_loopback_or_token():
    from types import SimpleNamespace

    from connex.serving.pd_router import ElasticPool, PDRouter

    r = PDRouter(ElasticPool([]), ElasticPool([]), probe_interval=0)
    req = lambda remote, auth="": SimpleNamespace(remote=remote, headers={"Authorization": auth})
    assert r._admin_allowed(req("127.0.0.1"))
    assert not r._admin_allowed(req("10.0.0.5"))
    r.admin_token = "s3cret"
    assert r._admin_allowed(req("10.0.0.5", "Bearer s3cret"))
    assert not r._admin_allowed(req("10.0.0.5", "Bearer nope"))


def test_bind_ip_defaults_to_advertised_ipv4():
    try:
        from connex.adapters.vllm.pd_connector import _bind_ip
    except Exception as e:  # noqa: BLE001  (vLLM missing or built for another torch)
        pytest.skip(f"vLLM connector not importable: {e}")

    assert _bind_ip({}, "192.0.2.3") == "192.0.2.3"
    assert _bind_ip({}, "node-a") == "0.0.0.0"
    assert _bind_ip({"bind_ip": "0.0.0.0"}, "192.0.2.3") == "0.0.0.0"
