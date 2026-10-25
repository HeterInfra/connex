# SPDX-License-Identifier: Apache-2.0
"""Replica-level recovery for EP groups whose engine cannot survive the loss
of one rank.

In vLLM (0.32) and SGLang (0.5.22, without its Mooncake elastic-EP backend)
the ranks of one EP group share collectives every step (the MoE all-to-all /
all-gather, the tensor-parallel all-reduce): when one rank dies the whole
group stops. Recovery is then a level up: each EP group is one mobile
endpoint (logical id ``ep-group/<k>``, address = its API URL). The
supervisor

1. probes every group (``/health``; process exit when it owns the process),
2. on failure removes the group from the routing set (the endpoints file the
   resume proxy re-reads every second; the proxy itself already resumes the
   group's in-flight requests on a healthy group from the tokens they
   produced, see ``connex.serving.proxy``),
3. rebuilds the group (``restart`` callback: start the engine again), waits
   until it serves, and puts it back in the routing set (rejoin).

Timings of every step are recorded in ``events``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request

log = logging.getLogger("connex.ep")


DEFAULT_PROBE_TIMEOUT = 3.0     # SGLang's /health answers after >= 1 s


def _healthy(url, timeout=None, path=None):
    """GET ``url + path`` answers 200 within ``timeout`` seconds. Defaults:
    ``CONNEX_EP_PROBE_TIMEOUT`` (3 s) and ``CONNEX_EP_PROBE_PATH``
    (``/health``)."""
    if timeout is None:
        timeout = float(os.environ.get("CONNEX_EP_PROBE_TIMEOUT", DEFAULT_PROBE_TIMEOUT))
    if path is None:
        path = os.environ.get("CONNEX_EP_PROBE_PATH", "/health")
    try:
        req = urllib.request.Request(url.rstrip("/") + "/" + path.lstrip("/"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def pid_alive(pid):
    """True if ``pid`` runs and is not a zombie (a killed rank whose parent has
    not reaped it yet still answers kill(pid, 0))."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class EPGroup:
    """``rank_pids``: optional callable returning the pids of the group's rank
    processes; any of them gone marks the group failed at the next check,
    without waiting for the API server to notice (vLLM keeps answering
    /health for several seconds after a worker died)."""

    def __init__(self, name, url, restart=None, proc=None, rank_pids=None, on_rejoin=None):
        self.name = name
        self.on_rejoin = on_rejoin    # callable(group) after a rebuilt group serves again
        self.rank_pids = rank_pids
        self._pids = None
        self.url = url.rstrip("/")
        self.restart = restart        # callable() -> new proc (or None)
        self.proc = proc              # optional subprocess.Popen
        self.state = "up"
        self.fails = 0


class EPGroupSupervisor:
    def __init__(self, groups, endpoints_file, probe_s=0.5, fail_after=2,
                 start_timeout=600.0, probe=None, probe_timeout=None, probe_path=None):
        """``probe(url) -> bool`` checks one group; by default an HTTP GET of
        ``probe_path`` with ``probe_timeout`` (see :func:`_healthy`)."""
        if probe is None:
            def probe(url, _t=probe_timeout, _p=probe_path):
                return _healthy(url, _t, _p)
        self.groups = {g.name: g for g in groups}
        self.file = endpoints_file
        self.probe_s = probe_s
        self.fail_after = fail_after
        self.start_timeout = start_timeout
        self.probe = probe
        self.events = []
        self._stop = threading.Event()
        self._th = None
        self._lock = threading.Lock()
        self.write_endpoints()

    def _event(self, kind, group, **kw):
        e = {"t": time.time(), "kind": kind, "group": group, **kw}
        self.events.append(e)
        log.info("ep-group %s %s %s", group, kind, kw)

    def write_endpoints(self):
        urls = [g.url for g in self.groups.values() if g.state == "up"]
        tmp = self.file + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(urls) + "\n")
        os.replace(tmp, self.file)

    def check_once(self):
        for g in list(self.groups.values()):
            if g.state != "up":
                continue
            dead = g.proc is not None and g.proc.poll() is not None
            if not dead and g.rank_pids is not None:
                if not g._pids:
                    try:
                        g._pids = list(g.rank_pids() or [])
                    except Exception:  # noqa: BLE001
                        g._pids = None
                dead = bool(g._pids) and not all(pid_alive(p) for p in g._pids)
                if dead:
                    self.on_failure(g, "rank process exited")
                    continue
            ok = (not dead) and self.probe(g.url)
            g.fails = 0 if ok else g.fails + 1
            if dead or g.fails >= self.fail_after:
                self.on_failure(g, "process exited" if dead else "health probe")

    def on_failure(self, g, why):
        with self._lock:
            if g.state != "up":
                return
            g.state = "down"
            self.write_endpoints()
            self._event("down", g.name, why=why)
        if g.restart is not None:
            threading.Thread(target=self._rebuild, args=(g,), daemon=True).start()

    def _rebuild(self, g):
        t0 = time.time()
        g.state = "rebuilding"
        self._event("rebuild-start", g.name)
        try:
            if g.proc is not None and g.proc.poll() is None:
                g.proc.kill()
            g.proc = g.restart()
        except Exception as ex:  # noqa: BLE001
            self._event("rebuild-failed", g.name, error=repr(ex))
            g.state = "down"
            return
        while time.time() - t0 < self.start_timeout and not self._stop.is_set():
            if g.proc is not None and g.proc.poll() is not None:
                self._event("rebuild-failed", g.name, error="exited during start")
                g.state = "down"
                return
            if self.probe(g.url):
                with self._lock:
                    g.state = "up"
                    g.fails = 0
                    g._pids = None          # re-read the new rank processes
                    self.write_endpoints()
                self._event("rejoin", g.name, rebuild_s=time.time() - t0)
                if g.on_rejoin is not None:
                    try:
                        g.on_rejoin(g)
                    except Exception as ex:  # noqa: BLE001
                        self._event("rejoin-hook-failed", g.name, error=repr(ex))
                return
            time.sleep(self.probe_s)
        self._event("rebuild-failed", g.name, error="timeout")
        g.state = "down"

    def start(self):
        def loop():
            while not self._stop.is_set():
                self.check_once()
                self._stop.wait(self.probe_s)
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)


class RankDeathWatcher:
    """Group failure across hosts, from the transport.

    The process watch of ``EPGroup.rank_pids`` only sees ranks on the
    supervisor's own host. ``RankDeathWatcher`` links a connex-rs engine to
    the EP transport of every rank (``ExpertStore.describe()["url"]``, one
    identity ``("connex-ep", 0, rank, group)`` per rank) and fails the
    rank's group in the supervisor as soon as connex-rs reports that peer
    dead: the dead process's kernel resets its connections, wherever it ran.
    Timeouts (hung or partitioned ranks) stay with the health probe.

    ``engine``: a connex-rs ``Engine`` (duck-typed: ``connect``,
    ``wait_event``); by default a new one pinned to this host's RDMA
    device."""

    def __init__(self, supervisor, engine=None):
        from connex.pd.group import GroupFailureTracker

        if engine is None:
            from connex.rl.transport import ConnexRsTransport

            self._transport = ConnexRsTransport(lid=("connex-ep-watch", 0, 0, 0))
            engine = self._transport.e
        self.sup = supervisor
        self.engine = engine
        self.tracker = GroupFailureTracker("connex-ep", [0])
        self.groups: dict[tuple, str] = {}     # rank identity -> group name
        self.reports: list[tuple[float, str, tuple]] = []
        self._stop = threading.Event()
        self._th = None

    @staticmethod
    def identity(url):
        """``ip:port#model|stage|shard|replica`` -> (model, stage, shard, replica)."""
        m, st, sh, r = url.split("#", 1)[1].split("|")
        return (m, int(st), int(sh), int(r))

    def watch(self, group_name, urls):
        """Link to the ranks of ``group_name`` (call again after a rebuild:
        the new incarnations have new addresses, the same identities)."""
        for u in urls:
            self.groups[self.identity(u)] = group_name
            self.engine.connect(u)

    def handle(self, ev):
        lid = self.tracker.on_event(ev)
        if lid is None:
            return None
        key = (lid.model_name, lid.stage_idx, lid.shard_idx, lid.replica_idx)
        name = self.groups.get(key)
        g = self.sup.groups.get(name) if name else None
        if g is None or g.state != "up":
            return None
        self.reports.append((time.time(), name, key))
        self.sup.on_failure(g, f"rank {lid.shard_idx} died (transport)")
        return name

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    ev = self.engine.wait_event(100)
                except Exception:  # noqa: BLE001  (engine closed)
                    return
                if ev is not None:
                    self.handle(ev)
        self._th = threading.Thread(target=loop, daemon=True, name="connex-ep-rank-watch")
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)


__all__ = ["EPGroup", "EPGroupSupervisor", "RankDeathWatcher", "pid_alive"]
