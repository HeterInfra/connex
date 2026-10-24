# SPDX-License-Identifier: Apache-2.0
"""Deterministic fake vLLM OpenAI server for testing the resume proxy on CPU.

The "model" is greedy and stateless: the next token id is a pure function of
the last two token ids, so generating from prompt+partial output continues
exactly where an interrupted stream stopped. Vocabulary: token i <-> word
``w{i}``; each token's text is " w{i}".

It implements the subset of vLLM's API the proxy relies on:
``/v1/completions`` and ``/v1/chat/completions`` (stream and non-stream,
``stream_options.include_usage`` / ``continuous_usage_stats``, ``logprobs`` +
``return_tokens_as_token_ids``, ``return_token_ids``, prompt as token ids,
chat ``continue_final_message``), ``/tokenize`` and ``/v1/models``.

Failure injection (``FakeBackend.fail``): ``("reset", k)`` aborts the TCP
connection after k tokens, ``("stall", k)`` stops sending after k tokens but
keeps the socket open (what a SIGTERMed uvicorn draining a dead engine looks
like), ``("http500", 0)`` answers 500. ``fail_count`` limits how many requests
are affected (default: all).

    python -m connex.serving.fake_backend --port 5100 --tpot-ms 0
"""
import argparse
import asyncio
import json
import time

from aiohttp import web

VOCAB = 1000
BOS = 1
CHAT_PREFIX = [2, 3]  # stands in for the chat template's header tokens


def tokenize(text, add_bos=True):
    ids = [BOS] if add_bos else []
    for w in text.split():
        ids.append(int(w[1:]) % VOCAB if w.startswith("w") and w[1:].isdigit()
                   else sum(map(ord, w)) % VOCAB)
    return ids


def next_token(ids):
    a = ids[-1] if ids else 0
    b = ids[-2] if len(ids) > 1 else 0
    return (a * 31 + b * 17 + 7) % (VOCAB - 10) + 10


def detok(tid):
    return f" w{tid}"


def chat_ids(messages, add_generation_prompt=True, continue_final=False):
    ids = [BOS]
    for i, m in enumerate(messages):
        last = i == len(messages) - 1
        ids += CHAT_PREFIX + tokenize(m.get("content") or "", add_bos=False)
        if not (last and continue_final):
            ids.append(4)  # end-of-turn
    if add_generation_prompt and not continue_final:
        ids += CHAT_PREFIX
    return ids


class FakeBackend:
    def __init__(self, tpot_ms=0.0, ttft_ms=0.0, model="fake", stamp=False):
        self.stamp = stamp          # add the send time ("_t") to every chunk
        self.tpot = tpot_ms / 1000
        self.ttft = ttft_ms / 1000
        self.model = model
        self.fail = None
        self.fail_count = None
        self.requests = []          # request bodies, for assertions
        self.runner = None
        self.port = None

    def _take_fail(self):
        if self.fail is None:
            return None
        if self.fail_count is not None:
            if self.fail_count <= 0:
                return None
            self.fail_count -= 1
        return self.fail

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/tokenize", self.tokenize)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/health", self.models)
        return app

    async def start(self, port=0, host="127.0.0.1"):
        self.runner = web.AppRunner(self.app(), shutdown_timeout=0.1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host, port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return f"http://{host}:{self.port}"

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    async def models(self, req):
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model"}]})

    async def tokenize(self, req):
        b = await req.json()
        if "messages" in b:
            ids = chat_ids(b["messages"], b.get("add_generation_prompt", True),
                           b.get("continue_final_message", False))
        else:
            ids = tokenize(b["prompt"], b.get("add_special_tokens", True))
        return web.json_response({"tokens": ids, "count": len(ids), "max_model_len": 4096})

    async def completions(self, req):
        b = await req.json()
        self.requests.append(b)
        p = b["prompt"]
        ids = list(p) if isinstance(p, list) else tokenize(p)
        return await self._generate(req, b, ids, chat=False)

    async def chat(self, req):
        b = await req.json()
        self.requests.append(b)
        ids = chat_ids(b["messages"], b.get("add_generation_prompt", True),
                       b.get("continue_final_message", False))
        return await self._generate(req, b, ids, chat=True)

    async def _generate(self, req, b, ids, chat):
        fail = self._take_fail()
        if fail and fail[0] == "http500":
            return web.json_response({"error": "boom"}, status=500)
        if (b.get("max_tokens") or 1) <= 0:
            return web.json_response({"error": "max_tokens must be positive"}, status=400)
        n_prompt = len(ids)
        max_tok = b.get("max_tokens") or b.get("max_completion_tokens") or 16
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{len(self.requests)}"
        created = int(time.time())
        want_lp = b.get("logprobs") is not None and b.get("logprobs") is not False
        as_ids = b.get("return_tokens_as_token_ids", False)
        want_tids = b.get("return_token_ids", False)
        so = b.get("stream_options") or {}
        gen = []
        seq = list(ids)
        for _ in range(max_tok):
            t = next_token(seq)
            seq.append(t)
            gen.append(t)
        obj_type = "chat.completion.chunk" if chat else "text_completion"

        def usage(k):
            return {"prompt_tokens": n_prompt, "completion_tokens": k, "total_tokens": n_prompt + k}

        def lp_for(t):
            tok = f"token_id:{t}" if as_ids else detok(t)
            if chat:
                return {"content": [{"token": tok, "logprob": -0.1, "top_logprobs": []}]}
            return {"tokens": [tok], "token_logprobs": [-0.1], "top_logprobs": None, "text_offset": [0]}

        if not b.get("stream"):
            text = "".join(detok(t) for t in gen)
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text},
                          "finish_reason": "length"}
            else:
                choice = {"index": 0, "text": text, "finish_reason": "length", "logprobs": None}
            return web.json_response({"id": rid, "object": "chat.completion" if chat else "text_completion",
                                      "created": created, "model": self.model, "choices": [choice],
                                      "usage": usage(len(gen))})

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)

        async def send(obj):
            await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if self.ttft:
            await asyncio.sleep(self.ttft)
        if fail and fail[1] == 0 and fail[0] == "stall":
            await asyncio.sleep(3600)
        if chat:
            await send({"id": rid, "object": obj_type, "created": created, "model": self.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                     "logprobs": None, "finish_reason": None}],
                        **({"usage": usage(0)} if so.get("continuous_usage_stats") else {})})
        for k, t in enumerate(gen):
            if fail and fail[1] == k:
                if fail[0] == "reset":
                    await asyncio.sleep(0.02)   # let already-written chunks leave
                    req.transport.abort()
                    return resp
                if fail[0] == "stall":
                    await asyncio.sleep(3600)
            last = k == len(gen) - 1
            if chat:
                ch = {"index": 0, "delta": {"content": detok(t)},
                      "logprobs": lp_for(t) if want_lp else None,
                      "finish_reason": "length" if last else None}
            else:
                ch = {"index": 0, "text": detok(t), "logprobs": lp_for(t) if want_lp else None,
                      "finish_reason": "length" if last else None, "stop_reason": None}
            if want_tids:
                ch["token_ids"] = [t]
            obj = {"id": rid, "object": obj_type, "created": created, "model": self.model, "choices": [ch]}
            if so.get("continuous_usage_stats"):
                obj["usage"] = usage(k + 1)
            if self.stamp:
                obj["_t"] = time.time()
            await send(obj)
            if self.tpot:
                await asyncio.sleep(self.tpot)
        if so.get("include_usage"):
            await send({"id": rid, "object": obj_type, "created": created, "model": self.model,
                        "choices": [], "usage": usage(len(gen))})
        await resp.write(b"data: [DONE]\n\n")
        return resp


class FakePDBackend(FakeBackend):
    """Fake prefill or decode instance for the PD router (``role`` = "prefill"
    | "decode"). Instances of one deployment share ``kv`` (transfer id ->
    held prompt ids and owner), standing in for the Connex KV connector:

    * prefill: a request with ``kv_transfer_params.do_remote_decode`` stores
      its prompt under the transfer id and returns ``kv_transfer_params``.
    * decode: a request with ``do_remote_prefill`` "pulls" the held prompt;
      it fails with HTTP 500 (vLLM's ``kv_load_failure_policy="fail"``) if the
      hold is gone, the owner is down (``alive`` False) or the held ids are
      not a prefix of the request's prompt (that would be wrong KV).
    """

    def __init__(self, role, kv, **kw):
        super().__init__(**kw)
        self.role = role
        self.kv = kv
        self.alive = True
        self.prefills = 0
        self.pulls = 0
        self.url = None

    async def start(self, port=0, host="127.0.0.1"):
        self.url = await super().start(port, host)
        self.alive = True
        return self.url

    async def stop(self):
        self.alive = False
        await super().stop()

    async def _generate(self, req, b, ids, chat):
        p = b.get("kv_transfer_params") or {}
        if self.role == "prefill" and p.get("do_remote_decode"):
            fail = self._take_fail()
            if fail and fail[0] == "http500":
                return web.json_response({"error": "boom"}, status=500)
            if fail and fail[0] in ("reset", "stall"):
                if fail[0] == "stall":
                    await asyncio.sleep(3600)
                req.transport.abort()
                return web.Response()
            self.prefills += 1
            tid = p["transfer_id"]
            self.kv[tid] = {"ids": list(ids), "owner": self}
            return web.json_response({
                "id": "cmpl-p", "object": "text_completion", "created": int(time.time()),
                "model": self.model,
                "choices": [{"index": 0, "text": detok(next_token(ids)), "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": 1,
                          "total_tokens": len(ids) + 1},
                "kv_transfer_params": {"do_remote_prefill": True, "do_remote_decode": False,
                                       "transfer_id": tid, "remote_ctrl": self.url,
                                       "remote_num_tokens": len(ids)}})
        if self.role == "decode":
            if not p.get("do_remote_prefill"):
                return web.json_response({"error": "decode instance needs kv_transfer_params"},
                                         status=400)
            held = self.kv.get(p.get("transfer_id"))
            n = p.get("remote_num_tokens", 0)
            if (held is None or not held["owner"].alive or held["ids"] != ids[:n]
                    or len(held["ids"]) != n):
                return web.json_response({"error": "KV load failed"}, status=500)
            self.pulls += 1
        return await super()._generate(req, b, ids, chat)


class FakeSGLangPDBackend(FakeBackend):
    """Fake SGLang PD instance: the router sends the request to a prefill and
    a decode instance concurrently with ``bootstrap_room``; the decode waits
    (up to ``wait_s``) for the prefill to publish the room in the shared
    ``rooms`` dict, and fails if it does not or if the prefill is down."""

    def __init__(self, role, rooms, wait_s=2.0, **kw):
        super().__init__(**kw)
        self.role, self.rooms, self.wait_s = role, rooms, wait_s
        self.alive = True
        self.prefills = 0

    async def start(self, port=0, host="127.0.0.1"):
        self.url = await super().start(port, host)
        self.alive = True
        return self.url

    async def stop(self):
        self.alive = False
        await super().stop()

    async def _generate(self, req, b, ids, chat):
        room = b.get("bootstrap_room")
        if room is None:
            return web.json_response({"error": "bootstrap_room required"}, status=400)
        if self.role == "prefill":
            fail = self._take_fail()
            if fail and fail[0] == "http500":
                return web.json_response({"error": "boom"}, status=500)
            self.prefills += 1
            self.rooms[room] = (list(ids), self)
            b = {**b, "stream": False, "max_tokens": 1}
            return await super()._generate(req, b, ids, chat)
        t0 = time.monotonic()
        while room not in self.rooms:
            if time.monotonic() - t0 > self.wait_s:
                return web.json_response({"error": "KV transfer timed out"}, status=500)
            await asyncio.sleep(0.005)
        held, owner = self.rooms[room]
        if held != ids or not owner.alive:
            return web.json_response({"error": "KV transfer failed"}, status=500)
        return await super()._generate(req, b, ids, chat)


def reference(prompt, max_tokens):
    """Uninterrupted greedy output of the fake model for a completion: (ids, text)."""
    seq = tokenize(prompt) if isinstance(prompt, str) else list(prompt)
    out = []
    for _ in range(max_tokens):
        t = next_token(seq)
        seq.append(t)
        out.append(t)
    return out, "".join(detok(t) for t in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5100)
    ap.add_argument("--tpot-ms", type=float, default=0)
    ap.add_argument("--ttft-ms", type=float, default=0)
    ap.add_argument("--stamp", action="store_true", help="add send timestamps to chunks")
    a = ap.parse_args()
    web.run_app(FakeBackend(a.tpot_ms, a.ttft_ms, stamp=a.stamp).app(), host=a.host, port=a.port,
                access_log=None)


if __name__ == "__main__":
    main()
