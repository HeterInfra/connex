# SPDX-License-Identifier: Apache-2.0
"""Rollout instances backed by real engine servers (HTTP).

* ``VLLMInstance``: a ``vllm serve`` started with
  ``--weight-transfer-config '{"backend": "connex"}'`` and
  ``VLLM_SERVER_DEV_MODE=1`` (for the RL routes). Updates go through vLLM's
  weight-transfer routes; the bytes come over connex-rs.
* ``SGLangInstance``: an SGLang server started with
  ``--custom-weight-loader connex.rl.sglang_ext.connex_load``; updates go
  through ``/update_weights_from_tensor`` with a control payload.

Both generate from token ids with greedy or seeded sampling and report the
tokens; an HTTP/connection failure is ``InstanceDead``; a request the engine
aborted (``finish_reason == "abort"``) is ``Aborted``.
"""
import asyncio
import base64
import io
import json
import pickle

import aiohttp

from .rollout import Aborted, InstanceDead, RolloutInstance
from .sglang_ext import LOADER, control_payload

RETRY = (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError)


class _HTTP(RolloutInstance):
    def __init__(self, name, url, model, registry_url, member=None, timeout=600):
        self.name = name
        self.url = url.rstrip("/")
        self.model = model
        self.registry_url = registry_url
        self.member = member or name
        self.version = None
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._s = None
        self.last_update = {}

    async def _session(self):
        if self._s is None or self._s.closed:
            # keep idle connections shorter than the servers' keep-alive (uvicorn:
            # 5 s), so a pooled connection is never reused after the server closed it
            self._s = aiohttp.ClientSession(
                timeout=self.timeout, trust_env=False,
                connector=aiohttp.TCPConnector(keepalive_timeout=2.0, limit=0))
        return self._s

    async def _post(self, path, json=None, params=None):
        s = await self._session()
        for attempt in (0, 1):
            try:
                async with s.post(self.url + path, json=json, params=params) as r:
                    body = await r.read()
                    if r.status != 200:
                        raise InstanceDead(f"{path}: HTTP {r.status}: {body[:300]!r}")
                    return body
            except aiohttp.ServerDisconnectedError as e:
                # a stale pooled connection looks the same as a dead server;
                # a dead server also refuses the retry's new connection
                if attempt:
                    raise InstanceDead(f"{path}: {e!r}") from e
            except RETRY as e:
                raise InstanceDead(f"{path}: {e!r}") from e

    async def close(self):
        if self._s is not None:
            await self._s.close()


class VLLMInstance(_HTTP):
    """``req``: {"prompt_ids", "max_tokens", "temperature", "seed"}."""

    _inited = False

    async def init(self):
        await self._post("/init_weight_transfer_engine", {"init_info": {
            "registry_url": self.registry_url, "member": self.member}})

    async def generate(self, req, partial=None):
        body = {"model": self.model, "prompt": list(req["prompt_ids"]),
                "max_tokens": req["max_tokens"], "temperature": req.get("temperature", 0.0),
                "seed": req.get("seed"), "ignore_eos": req.get("ignore_eos", False),
                "logprobs": 0, "return_tokens_as_token_ids": True}
        out = json.loads(await self._post("/v1/completions", body))
        ch = out["choices"][0]
        if ch.get("finish_reason") == "abort":
            raise Aborted()
        toks = ch.get("token_ids")
        if toks is None:
            toks = [int(t.split(":", 1)[1]) for t in ch["logprobs"]["tokens"]]
        return {"tokens": toks, "finish_reason": ch.get("finish_reason")}

    async def abort_all(self):
        await self._post("/pause", params={"mode": "abort", "clear_cache": "false"})
        await self._post("/resume")

    async def update(self, version):
        if not self._inited:
            await self.init()
            self._inited = True
        await self._post("/start_weight_update")
        await self._post("/update_weights", {"update_info": {"version": version}})
        await self._post("/finish_weight_update", {"weight_version": str(version)})
        self.version = version


class SGLangInstance(_HTTP):
    def __init__(self, *a, tp_size=1, **kw):
        super().__init__(*a, **kw)
        self.tp_size = tp_size
        self._has_sessions = None

    async def generate(self, req, partial=None):
        sp = {"max_new_tokens": req["max_tokens"], "temperature": req.get("temperature", 0.0),
              "ignore_eos": req.get("ignore_eos", False)}
        if req.get("seed") is not None:
            sp["sampling_seed"] = req["seed"]
        out = json.loads(await self._post("/generate", {"input_ids": list(req["prompt_ids"]),
                                                      "sampling_params": sp}))
        fr = (out.get("meta_info") or {}).get("finish_reason") or {}
        if isinstance(fr, dict) and fr.get("type") == "abort":
            raise Aborted()
        return {"tokens": out["output_ids"], "finish_reason": fr,
                "weight_version": (out.get("meta_info") or {}).get("weight_version")}

    async def abort_all(self):
        await self._post("/abort_request", {"abort_all": True})

    def payload(self, version):
        buf = io.BytesIO()
        pickle.dump(control_payload(self.registry_url, version, self.member), buf)
        s = base64.b64encode(buf.getvalue()).decode()
        return [s] * self.tp_size

    async def _session_call(self, path):
        """begin/end_weight_update (SGLang >= 0.5.2x); absent on older servers."""
        if self._has_sessions is False:
            return
        s = await self._session()
        try:
            async with s.post(self.url + path, json={}) as r:
                body = await r.read()
                if r.status == 404:
                    self._has_sessions = False
                    return
                if r.status != 200:
                    raise InstanceDead(f"{path}: HTTP {r.status}: {body[:300]!r}")
                self._has_sessions = True
        except RETRY as e:
            raise InstanceDead(f"{path}: {e!r}") from e

    async def update(self, version):
        await self._session_call("/begin_weight_update")
        await self._post("/update_weights_from_tensor", {
            "serialized_named_tensors": self.payload(version), "load_format": LOADER,
            "flush_cache": True, "weight_version": str(version)})
        await self._session_call("/end_weight_update")
        self.version = version
