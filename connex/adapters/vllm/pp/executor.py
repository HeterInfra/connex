# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``--distributed-executor-backend connex.adapters.vllm.pp.executor.ConnexPPExecutor``

Drives a pipeline whose stages are separate vLLM instances
(``--pipeline-stage-rank k``), with replicas per stage and in-flight
failover. It is the stock multiprocessing executor for this instance's
own stage (its TP workers), plus:

* on stage 0 (the instance that serves the API and schedules): each step's
  SchedulerOutput and grammar output are also sent to one replica of every
  downstream stage (the route), and the step's ModelRunnerOutput comes back
  from the last stage. The hidden states flow stage to stage over the PP
  group (:mod:`group`), not through this executor.
* failover: when a replica on the route dies (connex-rs peer-failure event,
  or a step timeout confirmed by a ping), the route moves to another
  replica of that stage. Steps in flight are completed as lost, and every
  request that was running reports its KV as lost
  (``KVConnectorOutput.lost_kv_req_ids``), so the scheduler drops their
  in-flight outputs and recomputes them; clients see a delay, not an error.

On stages 1.. the executor is only the local part; the stage server
(:mod:`stage`) feeds it the steps it receives.

Configuration (stage 0):
  CONNEX_PP_STAGES       JSON list, one entry per downstream stage, of the
                         stage servers' control addresses "ip:port" (list
                         position = replica index), e.g.
                         '[["10.0.0.2:47001","10.0.0.3:47001"]]'
  CONNEX_PP_REPLICA      this pipeline's replica index (preferred route)
  CONNEX_PP_STEP_TIMEOUT_S   backstop for a step without reply (60): the
                         stage that does not answer a ping, else the first
                         one behind on the step, is failed over
  CONNEX_PP_REPLACE_TIMEOUT_S how long to wait for a live replica (120)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from connex.adapters.vllm.pp.link import CONTROL, make_engine
from connex.adapters.vllm.pp.worker import ROUTE_ATTR

log = logging.getLogger("vllm.connex_pp")

MODEL_ID = os.getenv("CONNEX_PP_MODEL_ID", "cxpp")
CTL_SHARD = 1000          # LID shard of a stage server's control endpoint
HEAD_SHARD = 2000         # LID shard of a stage-0 executor


def control_url(hostport: str, stage: int, replica: int) -> str:
    return f"{hostport}#{MODEL_ID}|{stage}|{CTL_SHARD}|{replica}"


class ControlLink:
    """connex-rs messaging without GPU memory: pickled control messages and
    peer-failure events."""

    def __init__(self, stage: int, shard: int, replica: int, port: int = 0,
                 on_message=None, on_failed=None):
        self.engine = make_engine(MODEL_ID, stage, shard, replica, port)
        self.url = self.engine.local_url
        self.on_message = on_message
        self.on_failed = on_failed
        self.closed = False
        self.send_lock = threading.Lock()
        threading.Thread(target=self._rx, name="cx-ctl-rx", daemon=True).start()
        threading.Thread(target=self._ev, name="cx-ctl-ev", daemon=True).start()

    def send(self, peer, msg, raw: bytes | None = None, quiet: bool = False) -> bool:
        data = raw if raw is not None else pickle.dumps((CONTROL, msg),
                                                        protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with self.send_lock:
                self.engine.send_msg(peer, data)
            return True
        except Exception as e:  # noqa: BLE001
            if not quiet:
                log.warning("connex-pp: control send to %s failed: %s", peer, e)
            return False

    @staticmethod
    def encode(msg) -> bytes:
        return pickle.dumps((CONTROL, msg), protocol=pickle.HIGHEST_PROTOCOL)

    def _rx(self) -> None:
        while not self.closed:
            try:
                m = self.engine.recv_msg(timeout_ms=200)
            except Exception:  # noqa: BLE001
                if self.closed:
                    return
                time.sleep(0.05)
                continue
            if m is None:
                continue
            peer, data = m
            try:
                hdr = pickle.loads(data)
            except Exception as e:  # noqa: BLE001
                log.warning("connex-pp: bad control message: %s", e)
                continue
            if not (isinstance(hdr, tuple) and len(hdr) == 2):
                log.warning("connex-pp: malformed control message from %s", peer)
                continue
            if hdr[0] == CONTROL and self.on_message is not None:
                try:
                    self.on_message(tuple(peer), hdr[1])
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: control handler failed")

    def _ev(self) -> None:
        while not self.closed:
            try:
                ev = self.engine.wait_event(timeout_ms=500)
            except Exception:  # noqa: BLE001
                return
            if ev and ev.get("type") == "failed" and self.on_failed is not None:
                try:
                    self.on_failed(tuple(ev.get("peer") or ()), ev.get("reason"))
                except Exception:  # noqa: BLE001
                    log.exception("connex-pp: failure handler failed")

    def close(self) -> None:
        self.closed = True
        try:
            self.engine.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class Replica:
    stage: int
    idx: int
    ctl_url: str
    data_url: str | None = None
    alive: bool = False
    busy: bool = False          # owned by another pipeline
    pong: float = 0.0
    alive_since: float = 0.0
    done_seq: int = -1          # last step it reported done (with the pong)
    quarantine_until: float = 0.0   # failed over for a hung step: not chosen before


@dataclass
class Pending:
    seq: int
    epoch: int
    req_ids: list[str]
    event: threading.Event = field(default_factory=threading.Event)
    output: ModelRunnerOutput | None = None
    t_send: float = 0.0
    lost: bool = False


def pick_stuck(route: list, late: list, t_ping: float) -> tuple:
    """The replica to fail over for steps ``late`` (not completed within the
    step timeout), after a ping sent at ``t_ping``: one that did not answer;
    else, since a hung GPU (or a stream waiting for data that will not come)
    still answers, the first stage behind on the oldest late step, else the
    last one. A wrong guess costs a recompute, and the next timeout tries
    again. (None, "") when the steps completed meanwhile."""
    if not late:
        return None, ""
    silent = next((r for r in route if r is not None and r.pong < t_ping), None)
    if silent is not None:
        return silent, "step timeout, no pong"
    seq = min(p.seq for p in late)
    victim = next((r for r in route if r is not None and r.done_seq < seq),
                  next((r for r in reversed(route) if r is not None), None))
    return victim, f"step {seq} timeout, stage answers but is stuck"


class ConnexPPExecutor(MultiprocExecutor):
    supports_pp = True

    # ---------------------------------------------------------------- setup
    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        tp, pp, pcp = super()._get_parallel_sizes()
        if self.parallel_config.pipeline_stage_rank is None:
            raise ValueError("ConnexPPExecutor needs --pipeline-stage-rank (one stage per "
                             "instance) and --pp-group-cls connex.adapters.vllm.pp.group."
                             "ConnexPPGroup")
        return tp, 1, pcp

    def _post_init_executor(self) -> None:
        pc = self.parallel_config
        self.stage = pc.pipeline_stage_rank
        self.num_stages = pc.pipeline_parallel_size
        self.head = self.stage == 0
        endpoints = self.collective_rpc("connex_pp_endpoint")
        # One endpoint per TP rank when every rank has one (group.py
        # TP_ENDPOINTS), ";"-separated; else TP rank 0's.
        self.local_data_url = (";".join(endpoints) if len(endpoints) > 1 and all(endpoints)
                               else endpoints[0])
        if self.head:
            self._init_head()

    def _init_head(self) -> None:
        if "lost_all_kv" not in getattr(KVConnectorOutput, "__dataclass_fields__", {}):
            # Without it a failover would decode on a replica that lacks the
            # requests' KV: fail closed.
            raise RuntimeError("ConnexPPExecutor needs a vLLM with "
                               "KVConnectorOutput.lost_all_kv (see "
                               "integrations/vllm/README.md)")
        stages = json.loads(os.environ.get("CONNEX_PP_STAGES", "[]"))
        if len(stages) != self.num_stages - 1:
            raise ValueError(f"CONNEX_PP_STAGES must list {self.num_stages - 1} downstream "
                             f"stage(s), got {stages!r}")
        self.replica_id = int(os.getenv("CONNEX_PP_REPLICA", "0"))
        self.owner = f"{self.replica_id}:{os.getpid()}"
        self.step_timeout = float(os.getenv("CONNEX_PP_STEP_TIMEOUT_S", "60"))
        self.replace_timeout = float(os.getenv("CONNEX_PP_REPLACE_TIMEOUT_S", "120"))
        self.max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.replicas: list[list[Replica]] = [
            [Replica(k + 1, i, control_url(hp, k + 1, i)) for i, hp in enumerate(lst)]
            for k, lst in enumerate(stages)
        ]
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.epoch = 1
        self.seq = 0
        self.pending: dict[int, Pending] = {}
        self.await_sample: deque[int] = deque()
        self.live_reqs: set[str] = set()
        self.route: list[Replica | None] = [None] * (self.num_stages - 1)
        # After a failover, until the engine has received an output that
        # reports the lost KV: steps are completed as lost, not sent.
        self.lost_pending = False
        self.failovers: list[dict] = []
        self.ctl = ControlLink(0, HEAD_SHARD, self.replica_id,
                               on_message=self._on_message, on_failed=self._on_failed)
        for lst in self.replicas:
            for r in lst:
                self._hello(r)
        deadline = time.monotonic() + float(os.getenv("CONNEX_PP_CONNECT_TIMEOUT_S", "600"))
        # Give every replica a moment to answer, so the preferred one is used.
        settle = time.monotonic() + float(os.getenv("CONNEX_PP_SETTLE_S", "10"))
        with self.cv:
            while (time.monotonic() < settle
                   and not all(r.alive for lst in self.replicas for r in lst)):
                self.cv.wait(0.2)
            while not self._choose_route():
                if time.monotonic() > deadline:
                    raise RuntimeError("connex-pp: no live replica for some stage")
                self.cv.wait(1.0)
                for lst in self.replicas:
                    for r in lst:
                        if not r.alive:
                            self._hello(r)
        threading.Thread(target=self._watchdog, name="cx-pp-watchdog", daemon=True).start()
        log.info("connex-pp: stage 0 of %d, route %s", self.num_stages,
                 [f"s{r.stage}r{r.idx}" for r in self.route])

    # -------------------------------------------------------------- routing
    def _choose_route(self) -> bool:
        """Keep live route members; pick a live, free replica for the rest
        (prefer the one with our replica index). Caller holds the lock."""
        ok = True
        for k, lst in enumerate(self.replicas):
            cur = self.route[k]
            if cur is not None and cur.alive and not cur.busy:
                continue
            cands = [r for r in lst if r.alive and not r.busy and r.data_url]
            # A replica failed over for a hung step answers pings; prefer
            # any other for a while.
            now = time.monotonic()
            cands.sort(key=lambda r: (r.quarantine_until > now,
                                      r.idx != self.replica_id % max(1, len(lst)), r.idx))
            self.route[k] = cands[0] if cands else None
            ok = ok and bool(cands)
        return ok

    def _route_urls(self) -> list[str]:
        return [self.local_data_url] + [r.data_url if r else None for r in self.route]

    def _spare_urls(self) -> dict:
        """Data URLs of the live replicas off the route, by stage: the
        endpoints warm their hops to them (group._route_changed)."""
        now = time.monotonic()
        out = {}
        for k, lst in enumerate(self.replicas):
            cur = self.route[k]
            urls = [r.data_url for r in lst if r is not cur and r.alive and not r.busy
                    and r.data_url and r.quarantine_until <= now]
            if urls:
                out[k + 1] = urls
        return out

    def _hello(self, r: Replica) -> None:
        # Our control URL goes along: a replica that restarts announces
        # itself to it once ready, instead of waiting for our next HELLO.
        self.ctl.send(r.ctl_url, ("HELLO", self.owner, self.ctl.url), quiet=True)

    # ------------------------------------------------------------- messages
    def _find(self, peer: tuple) -> Replica | None:
        # peer = (model, stage, shard, replica)
        if len(peer) < 4 or peer[2] != CTL_SHARD:
            return None
        stage, idx = int(peer[1]), int(peer[3])
        if 1 <= stage < self.num_stages and idx < len(self.replicas[stage - 1]):
            return self.replicas[stage - 1][idx]
        return None

    def _on_message(self, peer: tuple, msg) -> None:
        kind = msg[0]
        if kind == "OUT":
            _, seq, epoch, out = msg
            with self.lock:
                p = self.pending.get(seq)
                if p is None or p.epoch != epoch or p.event.is_set():
                    return
                p.output = out
                p.event.set()
            return
        r = self._find(peer)
        if r is None:
            return
        if kind == "HELLO_OK":
            _, data_url, owner = msg[:3]
            with self.cv:
                r.data_url = data_url
                if not r.alive:
                    r.alive_since = time.monotonic()
                    log.info("connex-pp: replica s%dr%d is up (%s)", r.stage, r.idx,
                             "announced itself" if len(msg) > 3 and msg[3] else "hello")
                r.alive = True
                r.busy = owner not in (None, self.owner)
                self.cv.notify_all()
        elif kind == "READY":
            # A (re)started replica announcing itself: confirm with a HELLO
            # (it reports its data URL and owner in the answer).
            self._hello(r)
        elif kind == "PONG":
            if len(msg) > 1:
                r.done_seq = msg[1]
            r.pong = time.monotonic()
        elif kind == "BUSY":
            log.warning("connex-pp: s%dr%d is owned by another pipeline", r.stage, r.idx)
            with self.lock:
                r.busy = True
            # Not on this (the control receive) thread: the failover waits
            # for HELLO_OK of a replacement, which this thread delivers, and
            # OUT replies of other steps must not stall behind it.
            threading.Thread(target=self._failover, args=(r, "busy"), name="cx-pp-failover",
                             daemon=True).start()

    def _on_failed(self, peer: tuple, reason) -> None:
        r = self._find(peer)
        if r is None:
            return
        if (r.alive and time.monotonic() - r.alive_since < 30.0
                and self._answers_ping(r, timeout=0.5)):
            # A late event about an earlier incarnation (a replica restarted
            # under the same identity) or a failed connect attempt: the live
            # one answers. Failing it over would recompute every request
            # again and again.
            log.info("connex-pp: ignoring failure event for s%dr%d (%s): it answers",
                     r.stage, r.idx, reason)
            return
        log.warning("connex-pp: replica s%dr%d failed (%s)", r.stage, r.idx, reason)
        with self.lock:
            r.alive = False
        self._failover(r, f"peer failed: {reason}")

    def _answers_ping(self, r: Replica, timeout: float = 1.5) -> bool:
        t = time.monotonic()
        if not self.ctl.send(r.ctl_url, ("PING",), quiet=True):
            return False
        while time.monotonic() - t < timeout:
            if r.pong >= t:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------- failover
    def _failover(self, r: Replica, why: str) -> None:
        t0 = time.monotonic()
        with self.cv:
            if self.route[r.stage - 1] is not r:
                return  # a spare
            self.epoch += 1
            epoch = self.epoch
            lost = [p for p in self.pending.values() if not p.event.is_set()]
            self.lost_pending = True
            self.route[r.stage - 1] = None
            # Complete the lost steps now: the engine must not wait for them.
            for p in lost:
                p.lost = True
                p.output = None
                p.event.set()
            log.warning("connex-pp: failover of stage %d (%s): epoch %d, %d step(s) in flight "
                        "lost, %d request(s) to recompute", r.stage, why, epoch, len(lost),
                        len(self.live_reqs))
            deadline = time.monotonic() + self.replace_timeout
            while not self._choose_route():
                if time.monotonic() > deadline:
                    log.error("connex-pp: no replacement for stage %d", r.stage)
                    break
                for x in self.replicas[r.stage - 1]:
                    if not x.alive:
                        self.ctl.send(x.ctl_url, ("HELLO", self.owner), quiet=True)
                self.cv.wait(0.5)
            urls = [u for u in self._route_urls() if u]
            self.failovers.append({"stage": r.stage, "from": r.idx,
                                   "to": getattr(self.route[r.stage - 1], "idx", None),
                                   "t": time.time(), "lost_steps": len(lost),
                                   "choose_s": time.monotonic() - t0})
        # Release every endpoint still waiting for data of older epochs.
        self._send_abort(urls, epoch)
        log.warning("connex-pp: route now %s (%.1f ms after the failure was seen)",
                    [f"s{x.stage}r{x.idx}" if x else None for x in self.route],
                    (time.monotonic() - t0) * 1e3)

    def _send_abort(self, urls: list[str], epoch: int) -> None:
        """ABORT of the epochs before ``epoch`` to every endpoint, off this
        thread and not under the control send lock: a send to an endpoint
        that needs a connection (or is gone) must not hold the failover or
        the next step's EXEC."""
        raw = pickle.dumps((3, epoch))  # link.ABORT

        def one(u):
            try:
                self.ctl.engine.send_msg(u, raw)
            except Exception as e:  # noqa: BLE001
                log.warning("connex-pp: abort to %s failed: %s", u, e)

        for u in [e for u in urls for e in u.split(";")]:
            threading.Thread(target=one, args=(u,), name="cx-pp-abort", daemon=True).start()

    def _watchdog(self) -> None:
        tick = 0
        while not self.ctl.closed:
            time.sleep(1.0)
            tick += 1
            if tick % 2 == 0:
                # (Re)discover replicas that are down: a restarted one becomes
                # a spare (it also announces itself, see stage.py).
                for lst in self.replicas:
                    for r in lst:
                        if not r.alive:
                            self._hello(r)
            now = time.monotonic()
            with self.lock:
                late = [p for p in self.pending.values()
                        if not p.event.is_set() and now - p.t_send > self.step_timeout]
                route = list(self.route)
            if not late:
                continue
            # Confirm before failing over: a slow step is not a dead peer.
            t_ping = time.monotonic()
            for r in route:
                if r is not None:
                    self.ctl.send(r.ctl_url, ("PING",))
            time.sleep(2.0)
            with self.lock:
                still = [p for p in late if not p.event.is_set()]
            victim, why = pick_stuck(route, still, t_ping)
            if victim is None:
                continue
            if "answers" in why:
                victim.quarantine_until = time.monotonic() + 4 * self.step_timeout
            with self.lock:
                victim.alive = False
            self._failover(victim, why)

    # --------------------------------------------------------------- steps
    def execute_model(self, scheduler_output, non_block: bool = False):
        if not getattr(self, "head", False):
            return super().execute_model(scheduler_output, non_block)
        so = scheduler_output
        forward = so.total_num_scheduled_tokens > 0
        with self.cv:
            # A stage without a live replica (e.g. restarting in place): wait
            # for one instead of running a step nobody downstream will see.
            # Such a step would never complete, and stage 0's receive for it
            # would hold its streams (and, with TP > 1, its in-group
            # broadcast) forever.
            deadline = time.monotonic() + self.replace_timeout
            while any(r is None for r in self.route) and time.monotonic() < deadline:
                self.cv.wait(0.5)
            self.seq += 1
            seq, epoch = self.seq, self.epoch
            doomed = self.lost_pending
            route_urls = self._route_urls()
            for nr in so.scheduled_new_reqs:
                self.live_reqs.add(nr.req_id)
            self.live_reqs.difference_update(so.finished_req_ids)
            if forward:
                self.await_sample.append(seq)
                p = Pending(seq, epoch, list(so.num_scheduled_tokens), t_send=time.monotonic())
                self.pending[seq] = p
                if doomed:
                    p.lost = True
                    p.event.set()
            route = list(self.route)
            spares = self._spare_urls()
        setattr(so, ROUTE_ATTR, {"epoch": epoch, "seq": seq, "route": route_urls,
                                 "max_tokens": self.max_tokens, "doomed": doomed,
                                 "spares": spares})
        if not doomed:
            raw = ControlLink.encode(("EXEC", seq, epoch, so, self.owner))
            for r in route:
                if r is not None:
                    self.ctl.send(r.ctl_url, None, raw=raw)
        return super().execute_model(so, non_block)

    def sample_tokens(self, grammar_output, non_block: bool = False):
        if not getattr(self, "head", False):
            return super().sample_tokens(grammar_output, non_block)
        with self.lock:
            seq = self.await_sample.popleft()
            p = self.pending[seq]
            route = list(self.route)
        if not p.lost:
            # The step can complete only from here on: the watchdog's step
            # timeout runs from now, not from execute_model.
            p.t_send = time.monotonic()
            raw = ControlLink.encode(("SAMPLE", seq, p.epoch, grammar_output, self.owner))
            for r in route:
                if r is not None:
                    self.ctl.send(r.ctl_url, None, raw=raw)
        local = super().sample_tokens(grammar_output, non_block=True)
        fut = _StepFuture(self, local, p)
        return fut if non_block else fut.result()

    def _lost_output(self, p: Pending) -> ModelRunnerOutput:
        """Output of a step that was lost in a failover: no tokens. The first
        one reports that the replaced stage's KV cache is gone, so the
        scheduler recomputes every running request (their in-flight outputs,
        this one included, are dropped) and drops its prefix cache."""
        out = ModelRunnerOutput(
            req_ids=list(p.req_ids),
            req_id_to_index={r: i for i, r in enumerate(p.req_ids)},
            sampled_token_ids=[[] for _ in p.req_ids],
        )
        self._attach_lost(out)
        return out

    def _attach_lost(self, out: ModelRunnerOutput) -> None:
        with self.lock:
            if not self.lost_pending:
                return
            self.lost_pending = False
            # Steps scheduled from now on run on the new route. Bump the
            # epoch once more so every endpoint lets go of the steps that
            # were completed as lost.
            self.epoch += 1
            epoch = self.epoch
            urls = [u for u in self._route_urls() if u]
        kvo = out.kv_connector_output or KVConnectorOutput()
        kvo.lost_all_kv = True
        out.kv_connector_output = kvo
        self._send_abort(urls, epoch)

    def connex_pp_failovers(self) -> list[dict]:
        return list(getattr(self, "failovers", []))

    def shutdown(self):
        ctl = getattr(self, "ctl", None)
        if ctl is not None:
            ctl.close()
        super().shutdown()


class _StepFuture:
    """sample_tokens() future of stage 0: the local (non-last stage) result
    is drained, the ModelRunnerOutput comes from the last stage."""

    def __init__(self, ex: ConnexPPExecutor, local, p: Pending):
        self.ex = ex
        self.local = local
        self.p = p
        self._out = None

    def done(self) -> bool:
        return self.p.event.is_set()

    def result(self, timeout=None):
        if self._out is not None:
            return self._out
        local_out = self.local.result()
        self.p.event.wait()
        with self.ex.lock:
            self.ex.pending.pop(self.p.seq, None)
        if self.p.lost or self.p.output is None:
            self._out = self.ex._lost_output(self.p)
        else:
            out = self.p.output
            if out.kv_connector_output is None and local_out is not None:
                out.kv_connector_output = getattr(local_out, "kv_connector_output", None)
            self.ex._attach_lost(out)
            self._out = out
        return self._out
