# SPDX-License-Identifier: Apache-2.0
"""SGLang glue for ``connex.ep`` (upstream SGLang, no fork; tested with 0.5.22).

SGLang keeps the expert placement of an EP deployment in one
``ExpertLocationMetadata`` per process: ``physical_to_logical_map``
``[layers, slots]``, ``logical_to_all_physical_map`` ``[layers, experts, X]``
(-1 padded) and ``logical_to_all_physical_map_num_valid``. Each MoE layer's
top-k routing reads views of these tensors (``ExpertLocationDispatchInfo``;
the ``dynamic`` dispatch picks replica ``row % num_valid``), so writing them
in place reroutes tokens, also inside captured CUDA graphs.

Control path. The ``connex_ep`` ``sglang.srt.plugins`` entry point (opt-in
with ``CONNEX_EP=1``) adds a small control endpoint to the scheduler of TP
rank 0 (HTTP, ``CONNEX_EP_PORT``; default: a free port, advertised in the
server log and in ``ctl_file(server_port)``). A command
posted there is put into the list of requests that rank 0 broadcasts to the
other TP ranks at the top of every scheduler iteration (the same broadcast
that carries the timeout aborts), so every rank runs it between the same two
forward steps. Each rank runs it, the per-rank results are gathered over the
TP CPU group, and rank 0 answers the HTTP call. The commands are those of the
vLLM worker extension (:class:`connex.ep.vllm_ext.ConnexEPWorker`):

* ``connex_ep_info``      geometry and the physical->logical map loaded;
* ``connex_ep_init``      register the expert weights with connex-rs;
* ``connex_ep_apply``     install a ``PlacementTable.view()`` into the
                          expert location metadata, in place;
* ``connex_ep_move_start`` / ``connex_ep_move_poll`` / ``connex_ep_move``
                          pull expert weights into this rank's slots;
* ``connex_ep_load``      tokens routed per physical slot (needs
                          ``--expert-distribution-recorder-mode stat``).

Serving needs ``--ep-num-redundant-experts`` (room for replicas) and
``--ep-dispatch-algorithm dynamic`` (the replica pick goes through the
metadata) and a model with SGLang EPLB support (Qwen2-MoE, Qwen3-MoE,
DeepSeek-V2/V3, ...). Do not pass ``--enable-eplb``: Connex owns the map.
No Mooncake / elastic-EP kernels are used. The plugin also makes ``TopK``
apply the metadata on MoE paths that do not pass it (see
:func:`_patch_topk`); without that, SGLang's Qwen2-MoE ignores replicas
when there is no all-to-all backend.

With the overlap scheduler the previous batch can still run on the GPU while
the scheduler handles input, so ``apply`` waits for the device before it
writes the maps and again after: the next batch sees the new epoch entirely.

Controller side: :class:`SGLangEPEngine` is the ``EPController`` engine
handle for one SGLang server (all its EP ranks).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time

from .vllm_ext import ConnexEPWorker, VLLMEPEngine

log = logging.getLogger("connex.ep")

_installed = False


class ConnexEPCmd:
    """One control command; rides SGLang's per-iteration request broadcast."""

    def __init__(self, seq, method, args):
        self.seq = seq
        self.method = method
        self.args = list(args)
        self.t_enq = time.time()


def register():
    """``sglang.srt.plugins`` entry point: opt-in with ``CONNEX_EP=1``."""
    if os.environ.get("CONNEX_EP", "") not in ("1", "true", "yes"):
        return
    install()


def install():
    """Patch SGLang's scheduler with the control path (idempotent)."""
    global _installed
    if _installed:
        return
    from sglang.srt.managers.scheduler import Scheduler
    _installed = True

    orig_init = Scheduler.__init__
    orig_poll = Scheduler._poll_timeout_aborts
    orig_process = Scheduler.process_input_requests

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            _setup(self)
        except Exception:  # noqa: BLE001  (never take the server down)
            log.exception("connex.ep: control endpoint not started")

    def _poll_timeout_aborts(self):
        out = orig_poll(self)
        q = getattr(self, "_connex_ep_q", None)
        if q is not None and not q.empty():
            out = list(out)
            while True:
                try:
                    out.append(q.get_nowait())
                except queue.Empty:
                    break
        return out

    def process_input_requests(self, recv_reqs):
        if recv_reqs and any(isinstance(r, ConnexEPCmd) for r in recv_reqs):
            for r in recv_reqs:
                if isinstance(r, ConnexEPCmd):
                    _run_cmd(self, r)
            recv_reqs = [r for r in recv_reqs if not isinstance(r, ConnexEPCmd)]
        return orig_process(self, recv_reqs)

    Scheduler.__init__ = __init__
    Scheduler._poll_timeout_aborts = _poll_timeout_aborts
    Scheduler.process_input_requests = process_input_requests
    _patch_topk()


def _patch_topk():
    """Route through the expert location metadata on every MoE path.

    Some models call ``TopK`` without ``expert_location_dispatch_info`` on
    their non-all-to-all path (SGLang 0.5.22's Qwen2-MoE: only the DeepEP
    path passes it). Top-k ids then stay logical and are used as physical
    slot ids, so with redundant experts every token goes to the first
    ``num_experts`` slots (EP rank 0) and replicas are never used. The
    ``TopK`` module knows its layer id: build the dispatch info from it when
    the caller did not, whenever a dispatch algorithm is configured."""
    from sglang.srt.layers.moe.topk import TopK
    orig = TopK.forward_cuda
    if getattr(orig, "_connex_ep", False):
        return

    def forward_cuda(self, hidden_states, router_logits, *a, **kw):
        if kw.get("expert_location_dispatch_info") is None and \
                getattr(self, "layer_id", None) is not None:
            from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
            from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
            md = get_global_expert_location_metadata()
            if md is not None and self.layer_id < md.num_layers:
                kw["expert_location_dispatch_info"] = \
                    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
        return orig(self, hidden_states, router_logits, *a, **kw)

    forward_cuda._connex_ep = True
    TopK.forward_cuda = forward_cuda


def _tp_rank():
    from sglang.srt.runtime_context import get_parallel
    return get_parallel().tp_rank


def _setup(sched):
    sched._connex_ep = SGLangEPWorker(sched)
    kind = os.environ.get("CONNEX_EP_TRANSPORT", "connex-rs")
    if kind != "none":
        # create the endpoint now, not at attach time under load
        try:
            sched._connex_ep._transport({"transport": kind})
        except Exception:  # noqa: BLE001
            log.exception("connex.ep: transport %s not created at start", kind)
    if _tp_rank() != 0:
        return
    sched._connex_ep_q = queue.Queue()
    sched._connex_ep_wait = {}
    port = int(os.environ.get("CONNEX_EP_PORT") or 0)    # 0: any free port
    _serve(sched, port)
    _advertise(sched)


def _server_port(sched):
    sa = getattr(sched, "server_args", None)
    base = getattr(sa, "port", None)
    if base is None:
        try:
            from sglang.srt.runtime_context import get_serving
            base = getattr(get_serving(), "port", None)
        except Exception:  # noqa: BLE001
            base = None
    return base


def ctl_file(server_port):
    """Where TP rank 0 advertises its control endpoint for ``server_port``
    (``CONNEX_EP_CTL_FILE`` overrides)."""
    import tempfile
    return os.environ.get("CONNEX_EP_CTL_FILE") or os.path.join(
        tempfile.gettempdir(), f"connex-ep-ctl-{server_port}.json")


def _advertise(sched):
    host, port = sched._connex_ep_srv.server_address[:2]
    url = f"http://{host}:{port}"
    info = {"url": url, "port": port, "server_port": _server_port(sched), "pid": os.getpid()}
    path = ctl_file(info["server_port"])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, path)
    sched._connex_ep_ctl = info
    log.info("connex.ep control endpoint %s (advertised in %s)", url, path)


def _run_cmd(sched, cmd):
    """Runs on every TP rank at the same point of the scheduler loop."""
    import torch.distributed as dist
    w = sched._connex_ep
    queue_s = time.time() - getattr(cmd, "t_enq", time.time())
    t0 = time.perf_counter()
    try:
        fn = getattr(w, cmd.method)
        if not cmd.method.startswith("connex_ep_"):
            raise AttributeError(cmd.method)
        res = {"ok": True, "result": fn(*cmd.args)}
    except Exception as ex:  # noqa: BLE001
        log.exception("connex.ep %s failed", cmd.method)
        res = {"ok": False, "error": repr(ex)}
    res["rank_s"] = time.perf_counter() - t0
    res["queue_s"] = queue_s
    group = getattr(sched, "tp_cpu_group", None)
    n = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
    if n > 1:
        out = [None] * n
        dist.all_gather_object(out, res, group=group)
    else:
        out = [res]
    waiting = getattr(sched, "_connex_ep_wait", None)     # TP rank 0 only
    if waiting is not None:
        slot = waiting.pop(cmd.seq, None)
        if slot is not None:
            slot["res"] = out
            slot["ev"].set()


def _serve(sched, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seq = [0]
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200, {"ok": True})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError as ex:
                return self._reply(400, {"error": repr(ex)})
            with lock:
                seq[0] += 1
                s = seq[0]
            slot = {"ev": threading.Event(), "res": None}
            sched._connex_ep_wait[s] = slot
            sched._connex_ep_q.put(ConnexEPCmd(s, req["method"], req.get("args") or []))
            if not slot["ev"].wait(float(req.get("timeout", 120))):
                sched._connex_ep_wait.pop(s, None)
                return self._reply(504, {"error": "timeout"})
            self._reply(200, {"results": slot["res"]})

    srv = ThreadingHTTPServer((os.environ.get("CONNEX_EP_HOST", "127.0.0.1"), port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="connex-ep-ctl").start()
    sched._connex_ep_srv = srv


class SGLangEPWorker(ConnexEPWorker):
    """Per-scheduler state; the RPC methods are those of the vLLM extension."""

    def __init__(self, sched):
        self.sched = sched

    def _transport(self, cfg=None):
        """The rank's transport for ``cfg`` (kind, group), created once.
        Creating a connex-rs endpoint takes 0.2-1.1 s and holds the
        interpreter, which stalls the scheduler loop; the plugin therefore
        creates the default one when the scheduler starts (before serving),
        see :func:`_setup`."""
        from connex.rl.transport import make_transport
        cfg = cfg or {}
        key = (cfg.get("transport", "connex-rs"), int(cfg.get("group", 0)))
        cache = self.__dict__.setdefault("_connex_ep_transports", {})
        if key not in cache:
            rank = self._ep_rank()
            cache[key] = make_transport(key[0], name=f"ep-r{rank}",
                                        lid=("connex-ep", 0, rank, key[1]))
        return cache[key]

    # -- engine access -------------------------------------------------------
    def _md(self):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        md = get_global_expert_location_metadata()
        if md is None:
            raise RuntimeError("no expert location metadata: the model has no SGLang EPLB "
                               "support or EP is off")
        return md

    def _model(self):
        return self.sched.tp_worker.model_runner.model

    def _ep_rank(self):
        from sglang.srt.runtime_context import get_parallel
        return get_parallel().moe_ep_rank

    def _dispatch(self):
        from sglang.srt.runtime_context import get_exec
        return get_exec().moe.ep_dispatch_algorithm

    def _ep_geometry(self):
        md = self._md()
        return md, md.ep_size, md.num_local_physical_experts

    def _expert_layers(self):
        """Per MoE layer (by layer id): the expert parameter tensors, each
        viewed as ``[local slots, ...]`` (fused shared-expert rows excluded)."""
        if getattr(self, "_connex_ep_layers", None) is not None:
            return self._connex_ep_layers
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        md, R, S = self._ep_geometry()
        by_layer = {}
        for m in self._model().modules():
            if isinstance(m, FusedMoE) and getattr(m, "layer_id", None) is not None:
                n = m.num_local_experts
                ps = [p for _, p in m.named_parameters(recurse=False)
                      if p.dim() >= 1 and p.shape[0] == n]
                by_layer[m.layer_id] = [p.data[:S] for p in ps]
        missing = [l for l in range(md.num_layers) if l not in by_layer]
        if missing:
            raise RuntimeError(f"layers {missing[:4]}... have no FusedMoE: dense layers are not "
                               "supported yet")
        self._connex_ep_layers = [by_layer[l] for l in range(md.num_layers)]
        return self._connex_ep_layers

    # -- RPCs ----------------------------------------------------------------
    def connex_ep_info(self):
        md, R, S = self._ep_geometry()
        return {
            "ep_rank": self._ep_rank(), "num_ranks": R, "slots_per_rank": S,
            "num_layers": md.num_layers, "num_experts": md.num_logical_experts,
            "max_replicas": md.logical_to_all_physical_map.shape[-1],
            "slot_map": md.physical_to_logical_map_cpu.tolist(),
            "dispatch": self._dispatch(),
            "epoch": getattr(self, "_connex_ep_epoch", 0),
        }

    def connex_ep_init(self, cfg="{}"):
        """Register the expert weights with connex-rs (and export CUDA IPC
        handles) in a background thread: registering tens of GB takes
        0.3-2 s and would stall the scheduler loop. Returns
        ``{"pending": True}`` until done, then ``ExpertStore.describe()``;
        call again to poll."""
        cfg = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
        store = getattr(self, "_connex_ep_store", None)
        if store is not None:
            return store.describe()
        job = getattr(self, "_connex_ep_init_job", None)
        if job is None:
            import threading
            import torch
            from .mover import ExpertStore
            layers, rank = self._expert_layers(), self._ep_rank()
            dev = layers[0][0].device if layers else None
            job = self._connex_ep_init_job = {"done": False, "error": None}

            def run():
                try:
                    if dev is not None and dev.type == "cuda":
                        torch.cuda.set_device(dev)
                    t0 = time.perf_counter()
                    t = self._transport(cfg)
                    t1 = time.perf_counter()
                    st = ExpertStore(rank, layers, t)
                    job["describe"] = st.describe()
                    log.info("connex.ep attach, rank %d: transport %.0f ms, registration "
                             "%.0f ms", rank, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3)
                    self._connex_ep_store = st
                except Exception as ex:  # noqa: BLE001
                    job["error"] = repr(ex)
                job["done"] = True

            threading.Thread(target=run, daemon=True, name="connex-ep-init").start()
            self._connex_ep_epoch = getattr(self, "_connex_ep_epoch", 0)
        if not job["done"]:
            return {"pending": True, "rank": self._ep_rank()}
        if job["error"]:
            self._connex_ep_init_job = None          # let the caller retry
            raise RuntimeError(job["error"])
        return job["describe"]

    def connex_ep_apply(self, view):
        import torch
        from .placement import logical_to_physical
        t0 = time.perf_counter()
        view = json.loads(view) if isinstance(view, str) else view
        cur = getattr(self, "_connex_ep_epoch", 0)
        if view["epoch"] <= cur and cur > 0:
            return {"applied": False, "epoch": cur}
        md, R, S = self._ep_geometry()
        if md.logical_to_all_physical_map.shape[0] == 0:
            raise RuntimeError("empty expert location metadata")
        if view["num_ranks"] != R or view["slots_per_rank"] != S:
            raise ValueError(f"view geometry {view['num_ranks']}x{view['slots_per_rank']} "
                             f"!= engine {R}x{S}")
        L, width = md.num_layers, md.logical_to_all_physical_map.shape[-1]
        nl = len(view["slot_map"])
        l2p, cnt = [], []
        for layer in range(nl):
            a, c = logical_to_physical(view, layer)
            if min(c) == 0:
                raise ValueError(f"layer {layer}: an expert has no routable replica")
            l2p.append(a)
            cnt.append(c)
        w = max(len(a[0]) for a in l2p)
        if w > width:
            raise ValueError(f"{w} replicas per expert, engine allows {width}")
        if nl != L:
            l2p, cnt = l2p * L, cnt * L
        rows = view["slot_map"] if nl == L else view["slot_map"] * L
        dev = md.logical_to_all_physical_map.device
        dt = md.logical_to_all_physical_map.dtype
        l2p_t = torch.tensor(l2p, dtype=dt)
        cnt_t = torch.tensor(cnt, dtype=md.logical_to_all_physical_map_num_valid.dtype)
        rows_t = None
        if rows != getattr(self, "_connex_ep_rows", None):
            rows_t = torch.tensor(rows, dtype=md.physical_to_logical_map.dtype)
        rank_map = None
        if md.logical_to_rank_dispatch_physical_map is not None:
            # static dispatch: this rank's pick, a local replica when it has one
            me = self._ep_rank()
            rank_map = torch.tensor(
                [[next((p for p in ps if p // S == me), ps[0]) for ps in
                  [[p for p in a[e] if p >= 0] for e in range(len(a))]] for a in l2p],
                dtype=md.logical_to_rank_dispatch_physical_map.dtype)
        if dev.type == "cuda" and os.environ.get("CONNEX_EP_APPLY_SYNC", "0") == "1":
            # blocking variant: wait out the overlapped batch on the CPU
            torch.cuda.synchronize(dev)
            self._write_maps(md, dev, rows, rows_t, l2p_t, w, width, cnt_t, rank_map)
            torch.cuda.synchronize(dev)
        elif dev.type == "cuda":
            # The scheduler thread runs on its schedule stream; the previous
            # batch may still run on the forward stream (overlap scheduler)
            # and read the maps. Order the writes after it on the GPU, not by
            # blocking the CPU: the schedule stream waits for the forward
            # stream, writes, and the next forward waits for the schedule
            # stream (SGLang's run_batch does forward_stream.wait_stream).
            st = torch.cuda.current_stream(dev)
            fwd = getattr(self.sched, "forward_stream", None)
            if fwd is not None and fwd != st:
                st.wait_stream(fwd)
            src = [x.pin_memory() for x in (l2p_t, cnt_t) + tuple(
                x for x in (rows_t, rank_map) if x is not None)]
            self._write_maps(md, dev, rows, rows_t, l2p_t, w, width, cnt_t, rank_map)
            ev = torch.cuda.Event()
            ev.record(st)
            # moves into slots this epoch staged out must not start before
            # the batches that could still read them are done
            self._connex_ep_applied = (ev, src)
        else:
            self._write_maps(md, dev, rows, rows_t, l2p_t, w, width, cnt_t, rank_map)
        self._connex_ep_epoch = view["epoch"]
        return {"applied": True, "epoch": view["epoch"], "rank": self._ep_rank(),
                "apply_ms": (time.perf_counter() - t0) * 1e3}

    def _write_maps(self, md, dev, rows, rows_t, l2p_t, w, width, cnt_t, rank_map):
        nb = dev.type == "cuda"
        if rows_t is not None:
            md.physical_to_logical_map.copy_(rows_t.to(dev, non_blocking=nb))
            md.physical_to_logical_map_cpu.copy_(rows_t)
            self._connex_ep_rows = rows
        prev_w = getattr(self, "_connex_ep_width", width)
        md.logical_to_all_physical_map[..., :w].copy_(l2p_t.to(dev, non_blocking=nb))
        md.logical_to_all_physical_map_cpu[..., :w].copy_(l2p_t)
        if prev_w > w:
            md.logical_to_all_physical_map[..., w:prev_w].fill_(-1)
            md.logical_to_all_physical_map_cpu[..., w:prev_w].fill_(-1)
        self._connex_ep_width = w
        md.logical_to_all_physical_map_num_valid.copy_(cnt_t.to(dev, non_blocking=nb))
        if rank_map is not None:
            md.logical_to_rank_dispatch_physical_map.copy_(rank_map.to(dev, non_blocking=nb))

    def _connex_ep_before_move(self):
        """Background move thread: wait until the epoch that staged the
        destination slots is in effect and earlier batches are done."""
        applied = getattr(self, "_connex_ep_applied", None)
        if applied is not None:
            applied[0].synchronize()

    def connex_ep_checksum(self, slots="[]"):
        import torch
        slots = json.loads(slots) if isinstance(slots, str) else slots
        md, R, S = self._ep_geometry()
        rank = self._ep_rank()
        mine = [p for p in (slots or range(rank * S, (rank + 1) * S)) if p // S == rank]
        layers = self._expert_layers()
        out = {str(p): float(sum(w[p % S].double().sum().item() for ps in layers for w in ps))
               for p in mine}
        torch.cuda.synchronize()
        return out

    def connex_ep_poison(self, slots="[]", value="nan"):
        import torch
        slots = json.loads(slots) if isinstance(slots, str) else slots
        md, R, S = self._ep_geometry()
        rank = self._ep_rank()
        n = 0
        for p in slots:
            if p // S == rank:
                for ps in self._expert_layers():
                    for w in ps:
                        w[p % S].fill_(float(value))
                n += 1
        torch.cuda.synchronize()
        return n

    def connex_ep_load(self):
        """Tokens per physical slot over the steps in SGLang's expert
        distribution recorder buffer (``--expert-distribution-recorder-mode
        stat``, recording started with ``/start_expert_distribution_record``;
        ``None`` otherwise)."""
        try:
            from sglang.srt.eplb.expert_distribution import \
                get_global_expert_distribution_recorder
            rec = get_global_expert_distribution_recorder()
            acc = getattr(rec, "_accumulator", None)
            data = getattr(acc, "_global_physical_count_of_buffered_step", None)
            if data is None:
                return None
            return data.get_all().sum(dim=0).sum(dim=0).cpu().tolist()
        except Exception:  # noqa: BLE001
            return None


class SGLangEPEngine(VLLMEPEngine):
    """``EPController`` engine handle for one SGLang server (all EP ranks).

    ``ctrl_url`` is the control endpoint of TP rank 0: ``CONNEX_EP_PORT`` when
    set, else a free port advertised in :func:`ctl_file` (see
    :meth:`from_server`)."""

    def __init__(self, ctrl_url, **kw):
        super().__init__(ctrl_url, **kw)

    @classmethod
    def from_server(cls, server_url, **kw):
        """Handle for the SGLang server at ``server_url`` (same host): reads
        the control endpoint its TP rank 0 advertised."""
        port = int(server_url.rstrip("/").rsplit(":", 1)[1])
        with open(ctl_file(port)) as f:
            return cls(json.load(f)["url"], **kw)

    def describe(self, poll_s=0.02):
        """Every rank's ``ExpertStore.describe()``; registration runs in the
        background on the ranks, so poll until all are ready."""
        t_end = time.time() + self.timeout
        while True:
            res = self.rpc("connex_ep_init", self.cfg)
            if not any(isinstance(d, dict) and d.get("pending") for d in res):
                return {d["rank"]: d for d in res}
            if time.time() > t_end:
                raise TimeoutError("connex_ep_init still pending")
            time.sleep(poll_s)

    def rpc(self, method, *args):
        r = self.s.post(self.base + "/rpc",
                        json={"method": method,
                              "args": [a if isinstance(a, str) else json.dumps(a) for a in args],
                              "timeout": self.timeout},
                        timeout=self.timeout + 5)
        if r.status_code != 200:
            raise RuntimeError(f"{method}: HTTP {r.status_code} {r.text[:300]}")
        out = r.json()["results"]
        errs = [x["error"] for x in out if not x["ok"]]
        if errs:
            raise RuntimeError(f"{method} failed: {errs}")
        self.last.setdefault("rank_s", {})[method] = [x["rank_s"] for x in out]
        self.last.setdefault("queue_s", {})[method] = out[0].get("queue_s")
        return [x["result"] for x in out]


__all__ = ["register", "install", "SGLangEPWorker", "SGLangEPEngine", "ConnexEPCmd"]

