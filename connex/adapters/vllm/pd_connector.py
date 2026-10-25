# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Connex KV connector for prefill-decode disaggregation in upstream vLLM.

Loaded out of tree, no vLLM changes::

    vllm serve <model> --kv-transfer-config '{"kv_connector": "ConnexConnector",
        "kv_connector_module_path": "connex.adapters.vllm.pd_connector",
        "kv_role": "kv_producer",            # prefill; "kv_consumer" on decode
        "kv_connector_extra_config": {"ctrl_port": 17100}}'

Protocol (pull based, see ``connex.pd.transfer``):

1. The PD router sends the prompt to a prefill instance with
   ``kv_transfer_params={"do_remote_decode": true, "transfer_id": T}`` and
   ``max_tokens=1``. When the request finishes the scheduler keeps its blocks
   (delayed free) and returns ``kv_transfer_params`` naming the transfer and
   the prefill's KV control address; the prefill worker *holds* the blocks
   under ``T`` as a renewable lease.
2. The router sends the request to a decode instance with those params. The
   decode scheduler reports the prompt tokens as externally computed (async
   load); its worker acquires ``T``, reads the blocks with one-sided reads
   over connex-rs (GPUDirect RDMA when available) or Mooncake, and reports
   the request as received. A failed acquire or read is reported as load
   errors for the request's blocks: vLLM then fails the request
   (``kv_load_failure_policy="fail"``, the default; the PD router retries on
   another prefill) or recomputes it locally (``"recompute"``).
3. When the request finishes on the decode instance, its worker releases
   ``T`` and the prefill frees the blocks. If the decode instance dies first,
   the router resumes the request on another decode instance with the same
   params plus the tokens generated so far: the new instance acquires ``T``
   again and pulls the prompt KV, and computes only the generated tokens.
   The hold expires after ``retain_s`` without acquires, so a lost decode
   never strands prefill memory. ``release`` = ``"after_pull"`` frees right
   after the first read instead.

``kv_connector_extra_config`` keys (all optional):

=================  =============================================================
``backend``        ``auto`` (default) | ``connex-rs`` | ``mooncake`` | ``loopback``
``host``           address peers use to reach this instance (default:
                   ``CONNEX_HOST_IP``, ``VLLM_HOST_IP``, else vLLM's ``get_ip``)
``ctrl_port``      base port of the KV control channel; TP rank r uses
                   ``ctrl_port + r`` (default ``kv_port``)
``rs_port``        base port of the connex-rs listener (default 0 = any)
``retain_s``       prefill hold lease, renewed by each acquire (default 60)
``release``        ``"on_finish"`` (default) | ``"after_pull"``
``front_router``   ``tcp://host:port`` of a Connex front-router: register this
                   instance in the PD pools (with ``http_url``, ``replica``)
``http_url``       this instance's OpenAI API base URL, advertised to the router
``replica``        replica index of this instance in its pool
=================  =============================================================

Tensor parallelism: decode TP rank r reads the KV heads of its shard from
every prefill TP rank that holds some of them (``connex.pd.kv_layout.tp_plan``),
so prefill and decode may use different TP sizes (KV head re-layout; the
connector asks vLLM for head-major ``LBHNC`` blocks, where one head of one
block is a single contiguous read). With equal TP sizes rank r reads whole
blocks from rank r.

Scope: attention KV caches whose per-block slice is contiguous (vLLM's
FlashAttention/FlashInfer layouts); equal block size on prefill and decode;
KV heads divisible by both TP sizes (no replicated KV heads); no hybrid
(Mamba/sliding window) groups, no pipeline parallelism inside an instance.
"""
from __future__ import annotations

import logging
import os
import uuid
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from connex.pd.dataplane import make_plane
from connex.pd.kv_layout import find_heads_dims, tensor_layout, tp_plan
from connex.pd.membership import PD_STAGE, PDMember
from connex.pd.transfer import KVConsumer, KVProducer
from connex.protocol import PDRole

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

log = logging.getLogger("vllm.connex_pd")  # under vLLM's handlers


@dataclass
class _Pull:
    req_id: str
    ctrl: str            # host:base_port of the producer
    transfer_id: str
    local_blocks: list[int]
    first_block: int
    remote_tp: int = 0   # 0: unknown (older producer), read rank r from rank r


@dataclass
class ConnexConnectorMetadata(KVConnectorMetadata):
    holds: list[tuple[str, str, list[int]]] = field(default_factory=list)   # (req, tid, blocks)
    drops: list[str] = field(default_factory=list)                          # tids
    pulls: list[_Pull] = field(default_factory=list)
    releases: list[tuple[str, str, int]] = field(default_factory=list)      # (ctrl, tid, remote tp)


def _host(extra: dict) -> str:
    h = extra.get("host") or os.getenv("CONNEX_HOST_IP") or os.getenv("VLLM_HOST_IP")
    if h:
        return h
    from vllm.utils.network_utils import get_ip

    return get_ip()


def _device_uuid() -> str:
    try:
        import torch

        return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    except Exception:  # noqa: BLE001
        return ""


def _split(addr: str) -> tuple[str, int]:
    h, p = addr.rsplit(":", 1)
    return h, int(p)


class ConnexConnector(KVConnectorBase_V1):
    """KVConnectorBase_V1 over Connex: pull-based KV transfer with renewable
    holds, so a request can be re-pulled by another decode instance."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig" = None):
        super().__init__(vllm_config, role, kv_cache_config)
        ktc = vllm_config.kv_transfer_config
        self.extra: dict[str, Any] = dict(ktc.kv_connector_extra_config or {})
        self.is_producer = ktc.kv_role in ("kv_producer", "kv_both")
        self.is_consumer = ktc.kv_role in ("kv_consumer", "kv_both")
        self.block_size = vllm_config.cache_config.block_size
        self.host = _host(self.extra)
        self.ctrl_base = int(self.extra.get("ctrl_port") or ktc.kv_port)
        self.engine_id = ktc.engine_id or str(uuid.uuid4())
        self.replica = int(self.extra.get("replica",
                                          zlib.crc32(self.engine_id.encode()) & 0x7FFFFFF))
        self.model = vllm_config.model_config.served_model_name or vllm_config.model_config.model
        if isinstance(self.model, (list, tuple)):
            self.model = self.model[0]
        self.release_after_pull = self.extra.get("release", "on_finish") == "after_pull"
        if role == KVConnectorRole.SCHEDULER:
            self._init_scheduler()
        else:
            self._init_worker(vllm_config)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> Optional[str]:
        """Head-major blocks (HND): one KV head of a block is one contiguous
        run, so a decode rank can read just its heads from a prefill rank of
        a different tensor-parallel size with reads of block_size x head_dim
        elements (4 KB for 16-token blocks of 128-dim bf16 heads) instead of
        one small read per token. ``kv_layout`` in the extra config overrides
        it (``"none"`` keeps vLLM's default)."""
        mc = getattr(vllm_config, "model_config", None)
        if mc is None or getattr(mc, "use_mla", False):
            return None
        ktc = vllm_config.kv_transfer_config
        extra = (ktc.kv_connector_extra_config or {}) if ktc is not None else {}
        layout = str(extra.get("kv_layout", "LBHNC"))
        return None if layout.lower() == "none" else layout

    # ======================================================================
    # Scheduler side
    # ======================================================================
    def _init_scheduler(self) -> None:
        self._meta = ConnexConnectorMetadata()
        self._local_hit: dict[str, int] = {}          # req -> locally computed tokens
        self._remote: dict[str, tuple[str, str, int]] = {}  # decode req -> (ctrl, tid, remote tp)
        # membership is registered by the workers, one shard per TP rank

    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> tuple[int, bool]:
        p = request.kv_transfer_params
        if not p or not p.get("do_remote_prefill"):
            return 0, False
        n_prompt = len(request.prompt_token_ids or ())
        remote = min(int(p.get("remote_num_tokens", n_prompt)), n_prompt)
        # Load whole blocks only (plus the partial tail block when the remote
        # prefix is the full prompt): the local prefix hit is block aligned.
        count = remote - num_computed_tokens
        if count > 0:
            self._local_hit[request.request_id] = num_computed_tokens
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        p = request.kv_transfer_params
        if not p or not p.get("do_remote_prefill"):
            return
        p["do_remote_prefill"] = False   # one transfer per scheduling of the request
        ctrl, tid = p.get("remote_ctrl"), p.get("transfer_id")
        if not ctrl or not tid:
            log.warning("request %s: kv_transfer_params without remote_ctrl/transfer_id: %s",
                        request.request_id, p)
            return
        rtp = int(p.get("remote_tp_size") or 0)
        self._remote[request.request_id] = (ctrl, tid, rtp)
        local: list[int] = []
        first = 0
        if num_external_tokens > 0:
            hit = self._local_hit.pop(request.request_id, 0)
            first = hit // self.block_size
            last = -(-(hit + num_external_tokens) // self.block_size)
            local = list(blocks.get_block_ids()[0][first:last])
        # an empty pull still acquires: the router learns nothing is lost, and
        # the hold is renewed until this request releases it
        self._meta.pulls.append(_Pull(request.request_id, ctrl, tid, local, first, rtp))

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta, self._meta = self._meta, ConnexConnectorMetadata()
        return meta

    def request_finished(self, request: "Request",
                         block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        from vllm.v1.request import RequestStatus

        p = request.kv_transfer_params
        rid = request.request_id
        self._local_hit.pop(rid, None)
        if not p:
            return False, None
        if rid in self._remote or p.get("do_remote_prefill"):
            ctrl, tid, rtp = self._remote.pop(rid, (p.get("remote_ctrl"), p.get("transfer_id"),
                                                    int(p.get("remote_tp_size") or 0)))
            if ctrl and tid and not self.release_after_pull:
                self._meta.releases.append((ctrl, tid, rtp))
            return False, None
        if not p.get("do_remote_decode") or not self.is_producer:
            return False, None
        tid = p.get("transfer_id") or rid
        ids = block_ids[0] if block_ids and isinstance(block_ids[0], (list, tuple)) else block_ids
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED or not ids:
            return False, None   # aborted or nothing to send: free now
        self._meta.holds.append((rid, tid, list(ids)))
        return True, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "transfer_id": tid,
            "remote_ctrl": f"{self.host}:{self.ctrl_base}",
            "remote_engine_id": self.engine_id,
            "remote_num_tokens": len(request.prompt_token_ids or ()),
            "remote_tp_size": self._vllm_config.parallel_config.tensor_parallel_size,
        }

    # ======================================================================
    # Worker side
    # ======================================================================
    def _init_worker(self, vllm_config: "VllmConfig") -> None:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        try:
            self.rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001  (not initialised in some unit setups)
            self.rank = 0
        pc = vllm_config.parallel_config
        self.tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        try:
            self.kv_heads = int(vllm_config.model_config.get_num_kv_heads(pc))   # per rank
        except Exception:  # noqa: BLE001
            self.kv_heads = 0
        self._plans: dict[int, list] = {}
        self.producer: Optional[KVProducer] = None
        self.consumer: Optional[KVConsumer] = None
        self._errors: set[int] = set()
        self._recving: set[str] = set()

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        num_blocks = self._kv_cache_config.num_blocks
        t0 = next(iter(kv_caches.values()))
        heads_dims = find_heads_dims(tuple(t0.shape), num_blocks, self.kv_heads, self.block_size,
                                 getattr(self._vllm_config.cache_config, "kv_cache_layout", None))
        layouts = [tensor_layout(t, num_blocks, heads_dims) for t in kv_caches.values()]
        log.info("ConnexConnector: %d KV tensors, first %s %s stride %s, %s",
                 len(layouts), t0.dtype, tuple(t0.shape), t0.stride(), layouts[0])
        role = PDRole.PREFILL if self.is_producer and not self.is_consumer else PDRole.DECODE
        plane = make_plane(self.model, PD_STAGE[role], self.replica, self.rank, self.host,
                           backend=self.extra.get("backend"),
                           port=int(self.extra.get("rs_port", 0)) and
                           int(self.extra["rs_port"]) + self.rank)
        log.info("ConnexConnector rank %d: %s data plane at %s, %d KV tensors",
                 self.rank, plane.name, plane.local_url, len(layouts))
        dev = _device_uuid()
        # one registration per process: producer and consumer share the plane
        if self.is_producer:
            self.producer = KVProducer(plane, layouts, port=self.ctrl_base + self.rank,
                                       retain_s=float(self.extra.get("retain_s", 60.0)),
                                       engine_id=self.engine_id, block_size=self.block_size,
                                       device=dev)
        if self.is_consumer:
            self.consumer = KVConsumer(plane, layouts, block_size=self.block_size,
                                       register=self.producer is None,
                                       release_after_pull=self.release_after_pull,
                                       device=dev, who=f"{self.engine_id}/{self.rank}")
        self.has_heads = heads_dims is not None
        self._start_membership(role, plane)

    def _start_membership(self, role: str, plane) -> None:
        """Register this TP rank as shard ``rank`` of the instance's group and
        report peers that the data plane sees die. A group (one instance) is
        one endpoint: when any rank dies the front-router fails all of it,
        and the PD router stops using the instance."""
        self._member = None
        fr = self.extra.get("front_router")
        if not fr:
            return
        url = self.extra.get("http_url") or f"http://{self.host}:{os.getenv('VLLM_PORT', '8000')}"
        self._member = PDMember(fr, self.model, role, self.replica, url,
                                kv_addr=f"{self.host}:{self.ctrl_base}" if self.is_producer else None,
                                shard=self.rank, group_size=self.tp_size)
        self._member.start()
        if hasattr(getattr(plane, "e", None), "wait_event"):
            from connex.pd.group import PeerDeathReporter

            self._reporter = PeerDeathReporter(plane.e, fr, self.model)
            self._reporter.start()

    def _rank_addr(self, ctrl: str, rank: Optional[int] = None) -> str:
        h, p = _split(ctrl)
        return f"{h}:{p + (self.rank if rank is None else rank)}"

    def _sources(self, ctrl: str, remote_tp: int):
        """(producer address, TPSource) per prefill rank this rank reads, or
        None for the plain rank-r-from-rank-r pull (equal TP, or unknown)."""
        if remote_tp <= 0 or not self.kv_heads or not getattr(self, "has_heads", False):
            if remote_tp > 0 and remote_tp != self.tp_size:
                raise ValueError(f"prefill TP {remote_tp} != decode TP {self.tp_size} and the KV "
                                 "head geometry is unknown")
            return None
        plan = self._plans.get(remote_tp)
        if plan is None:
            plan = tp_plan(self.rank, self.tp_size, remote_tp, self.kv_heads * self.tp_size)
            self._plans[remote_tp] = plan
        return [(self._rank_addr(ctrl, s.remote_rank), s) for s in plan]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, ConnexConnectorMetadata)
        if self.producer is not None:
            for rid, tid, blocks in meta.holds:
                self.producer.hold(rid, tid, blocks)
        if self.consumer is not None:
            for pl in meta.pulls:
                self._recving.add(pl.req_id)
                try:
                    srcs = self._sources(pl.ctrl, pl.remote_tp)
                except ValueError as e:
                    log.warning("request %s: %s", pl.req_id, e)
                    self._fail_now(pl)
                    continue
                self.consumer.pull(pl.req_id, self._rank_addr(pl.ctrl), pl.transfer_id,
                                   pl.local_blocks, pl.first_block, sources=srcs)
            for ctrl, tid, rtp in meta.releases:
                try:
                    srcs = self._sources(ctrl, rtp)
                except ValueError:
                    srcs = None
                if srcs is None:
                    self.consumer.release(self._rank_addr(ctrl), tid)
                else:
                    for addr, src in srcs:
                        self.consumer.release(addr, tid, src)

    def _fail_now(self, pl: _Pull) -> None:
        from connex.pd.transfer import PullResult

        with self.consumer._lock:
            self.consumer._done.append(PullResult(pl.req_id, False, "TP mismatch",
                                                  failed_blocks=list(pl.local_blocks)))

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: "torch.Tensor",
                      attn_metadata: Any, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(self, finished_req_ids: set[str]
                     ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        sending = self.producer.take_finished() if self.producer is not None else set()
        recving: set[str] = set()
        if self.consumer is not None:
            for r in self.consumer.take_done():
                recving.add(r.req_id)
                self._recving.discard(r.req_id)
                if not r.ok:
                    self._errors.update(r.failed_blocks)
        return (sending or None), (recving or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        out, self._errors = self._errors, set()
        return out

    def shutdown(self) -> None:
        member = getattr(self, "_member", None)
        if member is not None:
            member.stop()
        rep = getattr(self, "_reporter", None)
        if rep is not None:
            rep.stop()
        for x in (getattr(self, "consumer", None), getattr(self, "producer", None)):
            if x is not None:
                x.close()
