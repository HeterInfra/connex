# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""PD pool membership through the Connex front-router.

Every prefill and decode instance registers one logical endpoint
``LID(model, stage=PD_STAGE[role], shard=0, replica)`` whose addresses carry
its serving URL (``http://...``) and, for prefill instances, the KV control
address (``kv=host:port``). Registration, heartbeats (leases) and the
deregistration on a graceful stop come from ``connex.epc_client``; an
instance that dies stops heartbeating and is evicted by the router after its
lease. The PD router reads the two pools with ``fetch_pools``.

The front-router keys LIDs by (model, stage, shard, replica), so the PD role
is mapped to a reserved stage index (``PD_STAGE``); ``connex.protocol``
carries the same role as ``LogicalEndpointID.pd_role``.

Tensor parallelism: an instance with TP size n is one endpoint made of n
shards. Each TP rank registers ``LID(model, stage, shard=rank, replica)``
with ``group_size=n`` from its own worker process, so the death of any one
rank stops that rank's heartbeats, and a peer that sees it die (connex-rs
peer-death event) reports it with ``report_failure``. Either way the
front-router fails the whole group, and ``fetch_pools`` lists an instance
only while all n shards are registered: a group with a dead rank takes no
new requests, and it rejoins under the same identity once every rank is
back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import zmq

from connex.protocol import LogicalEndpointID, PDRole

log = logging.getLogger("connex.pd")

PD_STAGE = {PDRole.PREFILL: 1000, PDRole.DECODE: 1001}
STAGE_ROLE = {v: k for k, v in PD_STAGE.items()}
ROUTE_ACTIVE = 1


def pd_lid(model: str, role: str, replica: int, shard: int = 0) -> LogicalEndpointID:
    return LogicalEndpointID(model, PD_STAGE[role], shard, replica, pd_role=role)


@dataclass
class PDMemberInfo:
    role: str
    replica: int
    url: str
    kv_addr: Optional[str] = None
    extra: list[str] = field(default_factory=list)
    group_size: int = 1


def _gsize(route: dict) -> int:
    return int(route.get("group_size") or (route.get("metadata") or {}).get("group_size") or 1)


def complete_groups(routes: list[dict]) -> list[dict]:
    """Routes (compact dicts) of ungrouped endpoints, plus shard 0 of every
    group whose shards are all active; incomplete groups are left out."""
    out, groups = [], {}
    for r in routes:
        lid = r.get("lid") or {}
        n = _gsize(r)
        if n <= 1:
            out.append(r)
            continue
        if r.get("state", ROUTE_ACTIVE) != ROUTE_ACTIVE:
            continue
        key = (lid.get("model_name"), lid.get("stage_idx"), lid.get("replica_idx"))
        groups.setdefault(key, {})[int(lid.get("shard_idx", 0))] = r
    for shards in groups.values():
        lead = shards.get(0)
        if lead is not None and set(shards) >= set(range(_gsize(lead))):
            out.append(lead)
    return out


def _parse(route: dict) -> Optional[PDMemberInfo]:
    lid = route.get("lid") or {}
    role = STAGE_ROLE.get(lid.get("stage_idx"))
    if role is None or route.get("state", ROUTE_ACTIVE) != ROUTE_ACTIVE:
        return None
    url, kv, extra = None, None, []
    for a in route.get("addresses") or []:
        if a.startswith("http://") or a.startswith("https://"):
            url = url or a.rstrip("/")
        elif a.startswith("kv="):
            kv = a[3:]
        else:
            extra.append(a)
    if url is None:
        return None
    return PDMemberInfo(role, int(lid.get("replica_idx", 0)), url, kv, extra,
                        _gsize(route))


def fetch_pools(front_router_api: str, model: str, timeout_ms: int = 2000
                ) -> Optional[dict[str, list[PDMemberInfo]]]:
    """Active prefill and decode members of ``model``, or None if the
    front-router did not answer."""
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        s.connect(front_router_api)
        s.send_json({"method": "get_routes",
                     "params": {"model_name": model, "stage_idx": list(PD_STAGE.values()),
                                "compact": True}})
        rep = s.recv_json()
    except zmq.ZMQError:
        return None
    finally:
        s.close(0)
    if "error" in rep:
        log.warning("front-router: %s", rep["error"])
        return None
    pools: dict[str, list[PDMemberInfo]] = {PDRole.PREFILL: [], PDRole.DECODE: []}
    for r in complete_groups(rep.get("routes", [])):
        m = _parse(r)
        if m is not None:
            pools[m.role].append(m)
    for v in pools.values():
        v.sort(key=lambda m: m.replica)
    return pools


def report_failure(front_router_api: str, lid: LogicalEndpointID, reason: str,
                   timeout_ms: int = 1000) -> Optional[list[str]]:
    """Tell the front-router that ``lid`` died; it fails the whole group the
    LID belongs to. Returns the failed route keys, or None without an answer."""
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        s.connect(front_router_api)
        s.send_json({"method": "report_failure",
                     "params": {"lid": {"model_name": lid.model_name, "stage_idx": lid.stage_idx,
                                        "shard_idx": lid.shard_idx,
                                        "replica_idx": lid.replica_idx},
                                "reason": reason}})
        rep = s.recv_json()
    except zmq.ZMQError:
        return None
    finally:
        s.close(0)
    return list(rep.get("failed") or [])


class PDMember:
    """Registers one PD instance (or one TP rank of it: ``shard``,
    ``group_size``) with the front-router and keeps its lease.

    ``stop()`` deregisters: a graceful leave (scale-down) that the PD router
    sees before the instance goes away."""

    def __init__(self, front_router_api: str, model: str, role: str, replica: int,
                 http_url: str, kv_addr: Optional[str] = None,
                 heartbeat_interval_s: float = 1.0, events: Optional[str] = None,
                 shard: int = 0, group_size: int = 1, extra_addresses=()):
        from connex.epc_client import EPCV2Client, EPCV2ClientConfig

        addrs = [http_url.rstrip("/")] + ([f"kv={kv_addr}"] if kv_addr else []) + \
            list(extra_addresses)
        cfg = EPCV2ClientConfig(
            front_router_api=front_router_api,
            front_router_events=events or front_router_api.rsplit(":", 1)[0] + ":15560",
            model_name=model, stage_idx=PD_STAGE[role], replica_idx=replica,
            shard_idx=shard, group_size=group_size,
            local_addresses=addrs, heartbeat_interval_s=heartbeat_interval_s,
            ucx_enabled=False)
        self.lid = pd_lid(model, role, replica, shard)
        self.client = EPCV2Client(cfg)

    def start(self) -> bool:
        ok = self.client.start()
        if ok:
            log.info("PD member %s registered", self.lid)
        else:
            log.warning("PD member %s: front-router registration failed", self.lid)
        return ok

    def stop(self) -> None:
        self.client.stop()
