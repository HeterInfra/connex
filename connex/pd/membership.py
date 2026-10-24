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


def pd_lid(model: str, role: str, replica: int) -> LogicalEndpointID:
    return LogicalEndpointID(model, PD_STAGE[role], 0, replica, pd_role=role)


@dataclass
class PDMemberInfo:
    role: str
    replica: int
    url: str
    kv_addr: Optional[str] = None
    extra: list[str] = field(default_factory=list)


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
    return PDMemberInfo(role, int(lid.get("replica_idx", 0)), url, kv, extra)


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
    for r in rep.get("routes", []):
        m = _parse(r)
        if m is not None:
            pools[m.role].append(m)
    for v in pools.values():
        v.sort(key=lambda m: m.replica)
    return pools


class PDMember:
    """Registers one PD instance with the front-router and keeps its lease.

    ``stop()`` deregisters: a graceful leave (scale-down) that the PD router
    sees before the instance goes away."""

    def __init__(self, front_router_api: str, model: str, role: str, replica: int,
                 http_url: str, kv_addr: Optional[str] = None,
                 heartbeat_interval_s: float = 1.0, events: Optional[str] = None):
        from connex.epc_client import EPCV2Client, EPCV2ClientConfig

        addrs = [http_url.rstrip("/")] + ([f"kv={kv_addr}"] if kv_addr else [])
        cfg = EPCV2ClientConfig(
            front_router_api=front_router_api,
            front_router_events=events or front_router_api.rsplit(":", 1)[0] + ":15560",
            model_name=model, stage_idx=PD_STAGE[role], replica_idx=replica,
            local_addresses=addrs, heartbeat_interval_s=heartbeat_interval_s,
            ucx_enabled=False)
        self.lid = pd_lid(model, role, replica)
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
