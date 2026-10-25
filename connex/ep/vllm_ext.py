# SPDX-License-Identifier: Apache-2.0
"""vLLM glue for ``connex.ep`` (upstream vLLM, no fork; tested with 0.32.0).

Worker side: ``--worker-extension-cls connex.ep.vllm_ext.ConnexEPWorker``
adds methods that the controller calls through ``collective_rpc`` (HTTP
``/collective_rpc`` with ``VLLM_SERVER_DEV_MODE=1``). vLLM runs utility
calls between two engine steps on every worker, which is exactly the safe
cutover point a placement epoch needs:

* ``connex_ep_info``      geometry and the physical->logical map vLLM loaded;
* ``connex_ep_init``      register the expert weights with connex-rs;
                          returns this rank's ``ExpertStore.describe()``;
* ``connex_ep_apply``     install a ``PlacementTable.view()`` into vLLM's
                          EPLB maps (``physical_to_logical``,
                          ``logical_to_physical``, replica counts), in place,
                          so the CUDA graphs keep their pointers;
* ``connex_ep_move``      pull expert weights into this rank's slots;
* ``connex_ep_load``      tokens routed per physical slot since start.

vLLM's own EPLB rebalancing must stay off (``step_interval`` larger than the
run): Connex owns the map. EPLB needs a model that implements vLLM's
``MixtureOfExperts`` interface; :func:`register` adds it to OLMoE (vLLM's
``OlmoeForCausalLM`` builds its MoE layer without EPLB), the models that
already have it (Qwen3-MoE, DeepSeek-V2/V3, Mixtral, ...) need nothing.

Controller side: :class:`VLLMEPEngine` is the ``EPController`` engine handle
for one vLLM server (all its EP ranks).
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("connex.ep")

_registered = False


def register():
    """Make vLLM's OLMoE EPLB-capable (idempotent; a no-op without vLLM)."""
    global _registered
    if _registered:
        return
    try:
        from vllm.model_executor.models import olmoe
        from vllm.model_executor.models.interfaces import MixtureOfExperts
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError:
        return
    _registered = True
    if hasattr(olmoe.OlmoeForCausalLM, "set_eplb_state"):
        return                                    # upstream has it already
    orig_factory = olmoe.FusedMoEFactory

    def factory(*args, **kw):
        from vllm.config import get_current_vllm_config
        pc = get_current_vllm_config().parallel_config
        if pc.enable_eplb:
            kw.setdefault("enable_eplb", True)
            kw.setdefault("num_redundant_experts", pc.eplb_config.num_redundant_experts)
        return orig_factory(*args, **kw)

    olmoe.FusedMoEFactory = factory

    class OlmoeForCausalLMEP(olmoe.OlmoeForCausalLM, MixtureOfExperts):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            cfg = vllm_config.model_config.hf_config
            red = vllm_config.parallel_config.eplb_config.num_redundant_experts \
                if vllm_config.parallel_config.enable_eplb else 0
            self.moe_layers = [l.mlp.experts for l in self.model.layers
                               if isinstance(getattr(l, "mlp", None), olmoe.OlmoeMoE)]
            self.expert_weights = []
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = cfg.num_experts
            self.num_routed_experts = cfg.num_experts
            self.num_redundant_experts = red
            self.num_physical_experts = cfg.num_experts + red
            ep = self.moe_layers[0]
            self.num_local_physical_experts = getattr(
                ep, "local_num_experts", self.num_physical_experts)

        def update_physical_experts_metadata(self, num_physical_experts,
                                             num_local_physical_experts):
            self.num_physical_experts = num_physical_experts
            self.num_local_physical_experts = num_local_physical_experts
            self.num_redundant_experts = num_physical_experts - self.num_logical_experts
            for m in self.moe_layers:
                m.update_expert_map()

    OlmoeForCausalLMEP.__name__ = "OlmoeForCausalLM"
    olmoe.OlmoeForCausalLMEP = OlmoeForCausalLMEP
    ModelRegistry.register_model("OlmoeForCausalLM", OlmoeForCausalLMEP)


# register at import: the worker imports this module (worker extension)
# before it builds the model
register()


class ConnexEPWorker:
    """Mixed into vLLM's worker (``--worker-extension-cls``)."""

    # -- helpers -------------------------------------------------------------
    def _ep_model_state(self):
        st = self.model_runner.eplb_state
        if st is None:
            raise RuntimeError("EPLB is off: start vLLM with --enable-expert-parallel "
                               "--enable-eplb")
        return next(iter(st.model_states.values()))

    def _ep_rank(self):
        from vllm.distributed import get_ep_group
        return get_ep_group().rank_in_group

    def _ep_geometry(self):
        from vllm.distributed import get_ep_group
        ms = self._ep_model_state()
        P = ms.physical_to_logical_map.shape[1]
        R = get_ep_group().world_size
        return ms, R, P // R

    # -- RPCs (string args: vLLM's HTTP collective_rpc passes strings) --------
    def connex_ep_info(self):
        ms, R, S = self._ep_geometry()
        return {
            "ep_rank": self._ep_rank(), "num_ranks": R, "slots_per_rank": S,
            "num_layers": ms.physical_to_logical_map.shape[0],
            "num_experts": ms.model.num_logical_experts,
            "max_replicas": ms.logical_to_physical_map.shape[-1],
            "slot_map": ms.physical_to_logical_map.cpu().tolist(),
            "epoch": getattr(self, "_connex_ep_epoch", 0),
        }

    def connex_ep_init(self, cfg="{}"):
        cfg = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
        from connex.rl.transport import make_transport
        from .mover import ExpertStore
        if getattr(self, "_connex_ep_store", None) is not None:
            return self._connex_ep_store.describe()
        ms = self._ep_model_state()
        model = ms.model
        layers = [list(m.get_expert_weights()) for m in model.moe_layers]
        rank = self._ep_rank()
        kind = cfg.get("transport", "connex-rs")
        t = make_transport(kind, name=f"ep-r{rank}",
                           lid=("connex-ep", 0, rank, int(cfg.get("group", 0))))
        self._connex_ep_store = ExpertStore(rank, layers, t)
        self._connex_ep_epoch = getattr(self, "_connex_ep_epoch", 0)
        return self._connex_ep_store.describe()

    def connex_ep_apply(self, view):
        import torch
        from .placement import logical_to_physical
        t0 = time.perf_counter()
        view = json.loads(view) if isinstance(view, str) else view
        cur = getattr(self, "_connex_ep_epoch", 0)
        if view["epoch"] <= cur and cur > 0:
            return {"applied": False, "epoch": cur}
        ms, R, S = self._ep_geometry()
        if view["num_ranks"] != R or view["slots_per_rank"] != S:
            raise ValueError(f"view geometry {view['num_ranks']}x{view['slots_per_rank']} "
                             f"!= engine {R}x{S}")
        L = ms.physical_to_logical_map.shape[0]
        width = ms.logical_to_physical_map.shape[-1]
        nl = len(view["slot_map"])
        # routing lists at their real width (replicas per expert), not
        # vLLM's padded width (MAX_EXPERT_REDUNDANCY + 1 = 1024): building
        # and copying the padded tensor took ~90 ms per epoch
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
        dev = ms.logical_to_physical_map.device
        rows = view["slot_map"] if nl == L else view["slot_map"] * L
        if rows != getattr(self, "_connex_ep_rows", None):
            ms.physical_to_logical_map.copy_(torch.tensor(
                rows, dtype=ms.physical_to_logical_map.dtype).to(dev, non_blocking=True))
            self._connex_ep_rows = rows
        prev_w = getattr(self, "_connex_ep_width", width)
        ms.logical_to_physical_map[..., :w].copy_(torch.tensor(
            l2p, dtype=ms.logical_to_physical_map.dtype).to(dev, non_blocking=True))
        if prev_w > w:
            ms.logical_to_physical_map[..., w:prev_w].fill_(-1)
        self._connex_ep_width = w
        ms.logical_replica_count.copy_(torch.tensor(
            cnt, dtype=ms.logical_replica_count.dtype).to(dev, non_blocking=True))
        torch.cuda.current_stream(dev).synchronize()
        self._connex_ep_epoch = view["epoch"]
        return {"applied": True, "epoch": view["epoch"], "rank": self._ep_rank(),
                "apply_ms": (time.perf_counter() - t0) * 1e3}

    def connex_ep_move(self, req):
        from .placement import Move
        req = json.loads(req) if isinstance(req, str) else req
        store = getattr(self, "_connex_ep_store", None)
        if store is None:
            raise RuntimeError("call connex_ep_init first")
        moves = [Move.from_dict(m) for m in req["moves"]]
        peers = {int(k): v for k, v in req["peers"].items()}
        t0 = time.perf_counter()
        res = store.apply_moves(moves, peers, req["slots_per_rank"],
                                timeout_ms=req.get("timeout_ms"))
        return {"rank": self._ep_rank(), "seconds": time.perf_counter() - t0,
                "results": [[m.to_dict(), ok, err] for m, ok, err in res]}

    def connex_ep_move_start(self, req):
        """Start the moves of ``req`` in a background thread (one-sided reads
        land directly in the slots; local copies run on a side stream) and
        return at once: the engine keeps stepping while weights move. Poll
        with ``connex_ep_move_poll``."""
        import threading
        import torch
        from .placement import Move
        req = json.loads(req) if isinstance(req, str) else req
        store = getattr(self, "_connex_ep_store", None)
        if store is None:
            raise RuntimeError("call connex_ep_init first")
        prev = getattr(self, "_connex_ep_job", None)
        if prev is not None and not prev["done"]:
            raise RuntimeError("a move is already running")
        moves = [Move.from_dict(m) for m in req["moves"]]
        peers = {int(k): v for k, v in req["peers"].items()}
        dev = store.layers[0][0].device
        job = {"done": False, "results": [], "seconds": None, "error": None}

        def run():
            t0 = time.perf_counter()
            try:
                torch.cuda.set_device(dev)
                side = torch.cuda.Stream(dev)
                with torch.cuda.stream(side):
                    res = store.apply_moves(moves, peers, req["slots_per_rank"],
                                            timeout_ms=req.get("timeout_ms"))
                side.synchronize()
                job["results"] = [[m.to_dict(), ok, err] for m, ok, err in res]
            except Exception as ex:  # noqa: BLE001
                job["error"] = repr(ex)
            job["seconds"] = time.perf_counter() - t0
            job["done"] = True

        self._connex_ep_job = job
        threading.Thread(target=run, daemon=True, name="connex-ep-move").start()
        return {"rank": self._ep_rank(), "started": True}

    def connex_ep_move_poll(self):
        job = getattr(self, "_connex_ep_job", None)
        if job is None:
            return {"rank": self._ep_rank(), "done": True, "results": [], "seconds": 0}
        return {"rank": self._ep_rank(), **job}

    def connex_ep_checksum(self, slots="[]"):
        """Per local slot of this rank (global ids in ``slots``; all when
        empty): a float64 sum over every MoE layer's weights. For tests."""
        import torch
        slots = json.loads(slots) if isinstance(slots, str) else slots
        ms, R, S = self._ep_geometry()
        rank = self._ep_rank()
        mine = [p for p in (slots or range(rank * S, (rank + 1) * S)) if p // S == rank]
        layers = [list(m.get_expert_weights()) for m in ms.model.moe_layers]
        out = {}
        for p in mine:
            i = p % S
            out[str(p)] = float(sum(w[i].double().sum().item() for ps in layers for w in ps))
        torch.cuda.synchronize()
        return out

    def connex_ep_poison(self, slots="[]", value="nan"):
        """Overwrite the given global slots of this rank with ``value``
        (default NaN). For tests only: proves that no token reaches them."""
        import torch
        slots = json.loads(slots) if isinstance(slots, str) else slots
        ms, R, S = self._ep_geometry()
        rank = self._ep_rank()
        layers = [list(m.get_expert_weights()) for m in ms.model.moe_layers]
        n = 0
        for p in slots:
            if p // S == rank:
                for ps in layers:
                    for w in ps:
                        w[p % S].fill_(float(value))
                n += 1
        torch.cuda.synchronize()
        return n

    def connex_ep_load(self):
        ms = self._ep_model_state()
        w = getattr(ms, "expert_load_window", None)
        if w is None:
            return None
        return w.sum(dim=0).sum(dim=0).cpu().tolist()


class VLLMEPEngine:
    """``EPController`` engine handle for one vLLM server (all EP ranks).

    Talks to ``/collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``)."""

    def __init__(self, base_url, transport="connex-rs", group=0, timeout=120.0,
                 async_moves=True, poll_s=0.005):
        import requests
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.trust_env = False
        self.cfg = {"transport": transport, "group": group}
        self.timeout = timeout
        self.async_moves = async_moves
        self.poll_s = poll_s
        self.last = {}

    def rpc(self, method, *args):
        r = self.s.post(f"{self.base}/collective_rpc",
                        json={"method": method,
                              "args": [a if isinstance(a, str) else json.dumps(a) for a in args],
                              "timeout": self.timeout},
                        timeout=self.timeout + 5)
        r.raise_for_status()
        out = r.json()
        return out.get("results", out) if isinstance(out, dict) else out

    def info(self):
        return self.rpc("connex_ep_info")

    def table(self):
        """The placement the engine runs now, at the engine's epoch: a new
        controller continues the epoch sequence (an engine ignores views
        whose epoch is not newer than the one it runs)."""
        from .placement import PlacementTable
        infos = self.info()
        i = infos[0]
        return PlacementTable(i["num_experts"], i["num_ranks"], i["slots_per_rank"],
                              i["slot_map"], epoch=max(x["epoch"] for x in infos))

    def describe(self):
        return {d["rank"]: d for d in self.rpc("connex_ep_init", self.cfg)}

    def apply(self, view):
        res = self.rpc("connex_ep_apply", view)
        self.last["apply"] = res
        newer = [r["epoch"] for r in res if not r.get("applied") and r["epoch"] > view["epoch"]]
        if newer:
            from .placement import PlacementError
            raise PlacementError(f"engine runs epoch {max(newer)} > {view['epoch']}: another "
                                 "controller owns it, or adopt its state with table()")
        return res

    def move(self, moves, peers, slots_per_rank):
        from .placement import Move
        out = []
        req = {"moves": [m.to_dict() for m in moves],
               "peers": {str(k): v for k, v in peers.items()},
               "slots_per_rank": slots_per_rank}
        if self.async_moves:
            self.rpc("connex_ep_move_start", req)
            t_end = time.time() + self.timeout
            while True:
                res = self.rpc("connex_ep_move_poll")
                if all(r["done"] for r in res) or time.time() > t_end:
                    break
                time.sleep(self.poll_s)
            self.last["move"] = [{"rank": r["rank"], "seconds": r["seconds"]} for r in res]
            errs = [r["error"] for r in res if r.get("error")]
            if errs:
                raise RuntimeError(f"expert move failed: {errs}")
        else:
            res = self.rpc("connex_ep_move", req)
        if not self.async_moves:
            self.last["move"] = [{"rank": r["rank"], "seconds": r["seconds"]} for r in res]
        for r in res:
            out += [(Move.from_dict(m), ok, err) for m, ok, err in r["results"]]
        return out

    def load(self):
        return self.rpc("connex_ep_load")


__all__ = ["register", "ConnexEPWorker", "VLLMEPEngine"]
