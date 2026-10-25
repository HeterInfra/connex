# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel (MoE) experiments for ``connex.ep`` on vLLM.

Load generator plus control-plane events, one JSON per run::

    # steady state (no event); any vLLM EP server
    ep_bench.py run --url http://127.0.0.1:8301 --out steady.json

    # drain EP rank 1 at t=5 s under load (server started with
    # --enable-eplb, redundant experts and the Connex worker extension)
    ep_bench.py run --url ... --event drain --event-at 5 --out drain.json

    # migrate K experts (all layers) to rank 1 under load
    ep_bench.py run --url ... --event migrate --experts 8 --out mig.json

    # teacher-forced check of a run: every generated token must be the
    # argmax (within --tol nats) given its own prefix
    ep_bench.py score --url http://127.0.0.1:8301 --file drain.json

Events (``--event``):

* ``drain``: ``EPController.drain_rank(1)``; then the drained rank's slots are
  overwritten with NaN (``connex_ep_poison``): if any token still reached
  them, outputs would turn to garbage and fail the score; then the rank
  rejoins (``join_rank``) and is routed to again.
* ``migrate``: ``--experts`` experts are copied from rank 0 into rank-1
  slots (one-sided reads over connex-rs) and made their only replica
  (the original slot is staged out), so every token for them now runs on
  moved weights.

Request load, prompts and the score check reuse ``benchmarks/pd/pd_bench.py``.

SGLang (``--engine sglang``): the server runs with ``CONNEX_EP=1`` and the
controller talks to the control endpoint of TP rank 0 (``--ctrl``, default
server port + 1000). Requests and the score use SGLang's native
``/generate`` with token ids (prompts tokenized locally with ``--tokenizer``)::

    ep_bench.py run --engine sglang --url http://127.0.0.1:8301 \
        --tokenizer <model dir> --event drain --out drain.json
    ep_bench.py score --engine sglang --url http://127.0.0.1:8302 \
        --tokenizer <model dir> --file drain.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "pd"))
import pd_bench  # noqa: E402

from connex.ep import EPController  # noqa: E402
from connex.ep.vllm_ext import VLLMEPEngine  # noqa: E402


def make_engine(a):
    if getattr(a, "engine", "vllm") == "sglang":
        from connex.ep.sglang_ext import SGLangEPEngine
        ctrl = a.ctrl or "http://127.0.0.1:%d" % (int(a.url.rsplit(":", 1)[1]) + 1000)
        return SGLangEPEngine(ctrl, async_moves=not a.sync_moves)
    return VLLMEPEngine(a.url, async_moves=not a.sync_moves)


# -- SGLang native API ----------------------------------------------------------
_tok = {}


def sglang_prompt_ids(a):
    """Prompts as token ids (tokenized here, so load and score agree)."""
    if "ids" not in _tok:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.tokenizer)
        ps = pd_bench.prompts(a.n, a.prompt_len, a.seed, text=a.text)
        _tok["ids"] = [tk(p)["input_ids"] if isinstance(p, str) else p for p in ps]
    return _tok["ids"]


async def one_sglang(session, url, model, prompt, max_tokens, timeout):
    """``pd_bench.one`` for SGLang's ``/generate`` (streaming, greedy)."""
    body = {"input_ids": prompt, "stream": True,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0,
                                "ignore_eos": True}}
    t0 = time.perf_counter()
    first, ids, times, err = None, [], [], None
    try:
        async with session.post(url + "/generate", json=body,
                                timeout=pd_bench.aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:200]}"
            else:
                async for line in r.content:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    c = json.loads(line[6:])
                    if "error" in c:
                        err = str(c["error"])[:200]
                        break
                    out = c.get("output_ids") or []
                    n = (c.get("meta_info") or {}).get("completion_tokens", len(ids) + len(out))
                    new = out[len(ids):] if len(out) >= n else out    # cumulative or delta
                    now = time.perf_counter()
                    for t in new:
                        ids.append(t)
                        times.append(now)
                    if new and first is None:
                        first = now
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()
    return {"ok": err is None and len(ids) == max_tokens, "err": err, "n": len(ids), "ids": ids,
            "ttft": (first - t0) if first else None, "e2e": t1 - t0,
            "tpot": ((times[-1] - first) / (len(ids) - 1)) if first and len(ids) > 1 else None,
            "max_gap": max((b - a for a, b in zip(times, times[1:])), default=None),
            "start": t0}


async def score_sglang(a):
    """Teacher-forced check on SGLang: every generated token must be the
    argmax (within ``tol`` nats) given its own prefix."""
    d = json.load(open(a.file))
    ra = argparse.Namespace(**d["args"])
    ra.tokenizer = a.tokenizer
    ps = sglang_prompt_ids(ra)
    sem = asyncio.Semaphore(8)
    async with pd_bench.aiohttp.ClientSession(trust_env=False) as s:
        async def go(r):
            p, gen = ps[r["i"]], r.get("ids") or []
            body = {"input_ids": p + gen, "return_logprob": True, "top_logprobs_num": 1,
                    "logprob_start_len": len(p) - 1,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
            async with sem:
                async with s.post(a.url + "/generate", json=body) as resp:
                    return r, gen, (await resp.json())["meta_info"]
        out = await asyncio.gather(*(go(r) for r in d["requests"] if r.get("ok")))
    tot = bad = bad_reqs = 0
    worst, per = 0.0, []
    for r, gen, mi in out:
        lps, tops = mi["input_token_logprobs"], mi["input_top_logprobs"]
        # entries start at token len(p) - 1; the first generated token is next
        off = len(lps) - len(gen)
        nb, first = 0, None
        for k, t in enumerate(gen):
            lp, tid = lps[off + k][0], lps[off + k][1]
            assert tid == t, (tid, t)
            gap = max(x[0] for x in tops[off + k]) - lp
            worst = max(worst, gap)
            if gap > a.tol:
                nb += 1
                first = k if first is None else first
        tot += len(gen)
        bad += nb
        bad_reqs += nb > 0
        if nb:
            per.append((r["i"], nb, first))
    return {"file": a.file, "requests": len(out), "tokens": tot, "non_argmax_tokens": bad,
            "requests_with_non_argmax": bad_reqs, "worst_gap": round(worst, 3), "tol": a.tol,
            "detail": per[:20]}


def controller(a):
    active_replicas = a.active_replicas
    eng = make_engine(a)
    ctl = EPController(eng.table(), eng)
    ctl.table.active_replicas = active_replicas
    t0 = time.perf_counter()
    ctl.refresh_peers()
    init_s = time.perf_counter() - t0
    ctl.push()
    return eng, ctl, init_s


def ev_drain(eng, ctl, a, log):
    S = ctl.table.slots_per_rank
    rep = ctl.drain_rank(1)
    log("drain", rep, apply=eng.last.get("apply"))
    poisoned = eng.rpc("connex_ep_poison", json.dumps(list(range(S, 2 * S))))
    log("poison", {"slots": S, "per_rank": poisoned})
    time.sleep(a.hold)
    # the poisoned rank must reload before it can serve: copy every slot back
    # from rank 0 (same experts, linear layout) and rejoin
    ctl.table.mark_up(1)
    for l in range(ctl.table.num_layers):
        ctl.table.staging.update((l, p) for p in range(S, 2 * S))
    ctl.table._dirty = True
    rep = ctl._run_moves("rejoin rank 1", [
        m for l in range(ctl.table.num_layers) for m in _copy_back(ctl, l, S)])
    log("rejoin", rep, move=eng.last.get("move"))


def _copy_back(ctl, layer, S):
    from connex.ep import Move
    row = ctl.table.slot_map[layer]
    for p in range(S, 2 * S):
        e = row[p]
        src = next(q for q in range(S) if row[q] == e)
        yield Move(layer, e, p, src, e)


def ev_migrate(eng, ctl, a, log):
    S = ctl.table.slots_per_rank
    L = ctl.table.num_layers
    experts = list(range(a.experts))
    for e in experts:
        # destination: the rank-1 slot that holds another expert's spare copy
        dst = S + ((e + S // 2) % S)
        rep = ctl.migrate(e, dst)
        log("migrate", rep, expert=e, dst=dst, move=eng.last.get("move"))
    # stage out every other replica of the moved experts: only moved weights serve
    for l in range(L):
        for e in experts:
            for p in ctl.table.replicas(l, e):
                if p != S + ((e + S // 2) % S):
                    ctl.table.staging.add((l, p))
    ctl.table._dirty = True
    t = ctl.push()
    log("isolate", {"apply_s": t, "replicas_e0": ctl.table.routable_replicas(0, 0)},
        apply=eng.last.get("apply"))


EVENTS = {"drain": ev_drain, "migrate": ev_migrate}


def run(a):
    events = []

    def log(kind, rep, **kw):
        events.append({"t": time.perf_counter() - t_start, "kind": kind, "report": rep, **kw})
        print(f"[{events[-1]['t']:.2f}s] {kind}: {json.dumps(rep)[:400]}", flush=True)

    eng = ctl = None
    if a.event or a.active_replicas:
        eng, ctl, init_s = controller(a)
        events.append({"t": 0, "kind": "init", "report": {"init_s": init_s}})
    t_start = time.perf_counter()
    th = None
    if a.event:
        def fire():
            time.sleep(a.event_at)
            try:
                EVENTS[a.event](eng, ctl, a, log)
            except Exception as ex:  # noqa: BLE001
                log("event-error", {"error": repr(ex)})
        th = threading.Thread(target=fire, daemon=True)
        th.start()
    if a.engine == "sglang":
        ids = sglang_prompt_ids(a)
        pd_bench.prompts = lambda n, *_, **__: ids[:n]
        pd_bench.one = one_sglang
    res, wall = asyncio.run(pd_bench.run_load(a))
    if th:
        th.join(timeout=600)
    s = pd_bench.summary(res, wall, a.max_tokens)
    out = {"args": vars(a), "summary": s, "events": events, "requests": res}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k: s[k] for k in ("requests", "ok", "failed", "throughput_tok_s")}))
    print("tpot p50", s["tpot"]["p50"], "max_gap max", s["max_gap"]["max"])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8301")
    r.add_argument("--model", default="olmoe")
    r.add_argument("--n", type=int, default=128)
    r.add_argument("--prompt-len", type=int, default=256)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--qps", type=float, default=0)
    r.add_argument("--duration", type=float, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--text", action="store_true", default=True)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--ref", default=None)
    r.add_argument("--event", choices=sorted(EVENTS), default=None)
    r.add_argument("--event-at", type=float, default=5.0)
    r.add_argument("--hold", type=float, default=5.0)
    r.add_argument("--experts", type=int, default=8)
    r.add_argument("--active-replicas", type=int, default=None,
                   help="route at most this many replicas per expert (others warm standby)")
    r.add_argument("--sync-moves", action="store_true",
                   help="copy weights inside the RPC (blocks the engine) instead of in "
                        "a background thread")
    r.add_argument("--out", required=True)
    for x in (r, (s := sub.add_parser("score"))):
        x.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
        x.add_argument("--tokenizer", default=None, help="sglang: model dir for prompt ids")
    r.add_argument("--ctrl", default=None, help="sglang: control endpoint of TP rank 0")
    s.add_argument("--url", default="http://127.0.0.1:8301")
    s.add_argument("--model", default="olmoe")
    s.add_argument("--file", required=True)
    s.add_argument("--tol", type=float, default=0.5)
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        fn = score_sglang if a.engine == "sglang" else pd_bench.run_score
        print(json.dumps(asyncio.run(fn(a))))


if __name__ == "__main__":
    main()
