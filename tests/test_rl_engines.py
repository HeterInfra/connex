# SPDX-License-Identifier: Apache-2.0
"""connex.rl engine glue on CPU: the SGLang custom loader, the vLLM backend
(skipped without a vLLM that has the weight-transfer API) and the HTTP
rollout instances against fake servers."""
import asyncio
import base64
import pickle
from types import SimpleNamespace

import pytest
import torch
from aiohttp import web

from connex.rl import (LocalTransport, RegistryServer, WeightPublisher, WeightRegistry)
from connex.rl import sglang_ext
from connex.rl.engines import SGLangInstance, VLLMInstance
from connex.rl.rollout import Aborted, InstanceDead


def named(v):
    g = torch.Generator().manual_seed(v)
    return [("a.weight", torch.randn(64, 8, generator=g)), ("b.bias", torch.randn(8, generator=g))]


class FakeModel:
    def __init__(self):
        self.loaded = {}

    def load_weights(self, weights):
        for n, t in weights:
            self.loaded[n] = t.clone()


@pytest.fixture
def registry():
    srv = RegistryServer(WeightRegistry(fanout=2))
    yield srv, f"http://127.0.0.1:{srv.port}"
    srv.close()


def test_sglang_control_payload_roundtrip_and_pickle():
    p = sglang_ext.control_payload("http://r:1", 7, "m")
    again = pickle.loads(pickle.dumps(p))     # what SGLangInstance sends over HTTP
    assert sglang_ext.parse_control(again) == {"registry_url": "http://r:1", "version": 7,
                                               "member": "m", "transport": "connex-rs",
                                               "chunk_bytes": sglang_ext.DEFAULT_CHUNK}
    assert sglang_ext.parse_control([("x", torch.zeros(1))]) is None


def test_sglang_loader_pulls_and_loads(registry, monkeypatch):
    srv, url = registry
    pub = WeightPublisher(srv.reg, LocalTransport("sgl-trainer"))
    monkeypatch.setattr(sglang_ext, "_sub", None)
    monkeypatch.setenv("CONNEX_RL_DEVICE", "cpu")
    model = FakeModel()
    pub.publish(named(1))
    sglang_ext.connex_load(model, sglang_ext.control_payload(url, 1, "s0", transport="local"))
    assert torch.equal(model.loaded["a.weight"].cpu(), named(1)[0][1])
    pub.publish(named(2))
    sglang_ext.connex_load(model, sglang_ext.control_payload(url, 2, "s0", transport="local"))
    assert torch.equal(model.loaded["b.bias"].cpu(), named(2)[1][1])
    assert srv.reg.status()["members"] == {"s0/r0": 2}
    # plain tensors still load like SGLang's default
    sglang_ext.connex_load(model, [("c", torch.ones(2))])
    assert "c" in model.loaded
    sglang_ext._sub.close()


def test_vllm_backend_receive_weights(registry):
    pytest.importorskip("vllm.distributed.weight_transfer.base", exc_type=ImportError)
    from connex.rl.vllm_backend import (ConnexWeightTransferEngine, PoolClient, register)
    srv, url = registry
    pub = WeightPublisher(srv.reg, LocalTransport("vllm-trainer"))
    vc = SimpleNamespace(parallel_config=SimpleNamespace(rank=0), model_config=None)
    model = FakeModel()
    eng = ConnexWeightTransferEngine(SimpleNamespace(backend="connex"), vc,
                                     torch.device("cpu"), model)
    eng.init_transfer_engine(eng.parse_init_info({"registry_url": url, "member": "e",
                                                  "transport": "local"}))
    pub.publish(named(3))
    eng.receive_weights(eng.parse_update_info({"version": 1}))
    assert torch.equal(model.loaded["a.weight"], named(3)[0][1])
    assert srv.reg.status()["members"] == {"e/r0": 1}
    eng.shutdown()
    register()
    register()                                    # idempotent

    class Bad:
        def start_weight_update(self):
            raise RuntimeError("down")

    class Good:
        calls = 0

        def start_weight_update(self):
            Good.calls += 1
    pc = PoolClient({"bad": Bad(), "good": Good()})
    pc.start_weight_update()
    assert list(pc.clients) == ["good"] and "bad" in pc.failed and Good.calls == 1
    with pytest.raises(RuntimeError):
        PoolClient({"bad": Bad()}).start_weight_update()


# -- HTTP instances against fake servers ----------------------------------------

async def serve(routes):
    app = web.Application()
    for method, path, h in routes:
        app.router.add_route(method, path, h)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def test_vllm_instance_protocol():
    seen = []

    async def rec(request):
        body = await request.read()
        seen.append((request.path, dict(request.query), body and await request.json()))
        if request.path == "/v1/completions":
            j = await request.json()
            fin = "abort" if j["prompt"] == [9] else "length"
            return web.json_response({"choices": [{"finish_reason": fin, "logprobs": {
                "tokens": ["token_id:5", "token_id:6"]}}]})
        return web.json_response({})

    async def go():
        runner, url = await serve([("POST", "/{p:.*}", rec)])
        inst = VLLMInstance("v", url, "m", "http://reg", member="v")
        out = await inst.generate({"prompt_ids": [1, 2], "max_tokens": 2})
        assert out["tokens"] == [5, 6]
        with pytest.raises(Aborted):
            await inst.generate({"prompt_ids": [9], "max_tokens": 2})
        await inst.update(4)
        assert inst.version == 4
        paths = [p for p, _, _ in seen]
        assert paths[-4:] == ["/init_weight_transfer_engine", "/start_weight_update",
                              "/update_weights", "/finish_weight_update"]
        assert seen[-2][2] == {"update_info": {"version": 4}}
        await inst.update(5)                      # init only once
        assert [p for p, _, _ in seen].count("/init_weight_transfer_engine") == 1
        await inst.abort_all()
        assert seen[-2][0] == "/pause" and seen[-2][1]["mode"] == "abort"
        await inst.close()
        await runner.cleanup()
        with pytest.raises(InstanceDead):
            await inst.generate({"prompt_ids": [1], "max_tokens": 1})
        await inst.close()
    asyncio.run(go())


def test_sglang_instance_protocol():
    seen = []

    async def gen(request):
        j = await request.json()
        fr = {"type": "abort"} if j["input_ids"] == [9] else {"type": "length"}
        return web.json_response({"output_ids": [3, 4], "meta_info": {
            "finish_reason": fr, "weight_version": "2"}})

    async def upd(request):
        seen.append(await request.json())
        return web.json_response({"success": True})

    async def go():
        runner, url = await serve([("POST", "/generate", gen),
                                   ("POST", "/update_weights_from_tensor", upd)])
        inst = SGLangInstance("s", url, "m", "http://reg:1", member="s", tp_size=2)
        out = await inst.generate({"prompt_ids": [1], "max_tokens": 2, "seed": 3})
        assert out["tokens"] == [3, 4] and out["weight_version"] == "2"
        with pytest.raises(Aborted):
            await inst.generate({"prompt_ids": [9], "max_tokens": 2})
        await inst.update(2)
        b = seen[-1]
        assert b["load_format"] == sglang_ext.LOADER and b["weight_version"] == "2"
        assert len(b["serialized_named_tensors"]) == 2
        ctl = sglang_ext.parse_control(pickle.loads(base64.b64decode(
            b["serialized_named_tensors"][0])))
        assert ctl["version"] == 2 and ctl["registry_url"] == "http://reg:1"
        await inst.close()
        await runner.cleanup()
    asyncio.run(go())
