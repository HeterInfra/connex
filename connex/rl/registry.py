# SPDX-License-Identifier: Apache-2.0
"""Weight registry: the control plane of elastic weight sync.

It knows the latest published version and its manifest, which members
(rollout engines, by logical id) are alive and at which version, and who
holds a full copy of each version and can serve it. ``assign`` hands a
subscriber a source with spare fan-out, which turns the broadcast into a
tree that grows as members finish: the trainer serves the first ``fanout``
readers, each finished reader serves ``fanout`` more, and so on.

``WeightRegistry`` is the in-process object; ``RegistryServer`` serves it
over HTTP/JSON (stdlib only, so it also works inside engine worker
processes) and ``RegistryClient`` has the same methods.
"""
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUBLISHER = "trainer"


class WeightRegistry:
    def __init__(self, fanout=2, member_ttl_s=10.0):
        self.fanout = fanout
        self.ttl = member_ttl_s
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.latest = None          # {"version", "manifest"}
        self.holders = {}           # version -> {member: {"url","addr","load","publisher"}}
        self.members = {}           # member -> {"url","version","last_seen"}
        self.events = []            # (t, kind, member, detail) for observability

    def _ev(self, kind, member, detail=None):
        self.events.append((time.time(), kind, member, detail))
        del self.events[:-1000]

    # -- membership -------------------------------------------------------------
    def join(self, member, url=""):
        with self.lock:
            self.members[member] = {"url": url, "version": None, "last_seen": time.time()}
            self._ev("join", member)
            return {"latest": self.latest["version"] if self.latest else None}

    def heartbeat(self, member):
        with self.lock:
            if member in self.members:
                self.members[member]["last_seen"] = time.time()
                return {"ok": True}
            return {"ok": False}

    def leave(self, member, reason="leave"):
        with self.cv:
            self.members.pop(member, None)
            for hs in self.holders.values():
                hs.pop(member, None)
            self._ev(reason, member)
            self.cv.notify_all()
            return {"ok": True}

    def expire(self):
        now = time.time()
        with self.lock:
            dead = [m for m, i in self.members.items() if now - i["last_seen"] > self.ttl]
        for m in dead:
            self.leave(m, "expired")
        return dead

    # -- versions ---------------------------------------------------------------
    def publish(self, version, manifest, url, addr):
        with self.cv:
            if self.latest and version <= self.latest["version"]:
                raise ValueError(f"version {version} <= latest {self.latest['version']}")
            self.latest = {"version": version, "manifest": manifest}
            self.holders[version] = {PUBLISHER: {"url": url, "addr": addr, "load": 0,
                                                 "publisher": True}}
            for v in [v for v in self.holders if v < version - 1]:
                del self.holders[v]
            self._ev("publish", PUBLISHER, version)
            self.cv.notify_all()
            return {"ok": True}

    def get_latest(self, wait_newer_than=None, timeout=0.0):
        with self.cv:
            if wait_newer_than is not None:
                self.cv.wait_for(lambda: self.latest and self.latest["version"] > wait_newer_than,
                                 timeout=timeout)
            return self.latest

    def assign(self, version, member, exclude=()):
        """Pick a source for ``member`` to read ``version`` from."""
        with self.lock:
            hs = {m: h for m, h in self.holders.get(version, {}).items()
                  if m != member and m not in exclude}
            if not hs:
                return None
            # prefer peers with spare fan-out (keeps load off the trainer), then
            # the least loaded; the publisher only when it has spare fan-out too
            def key(item):
                m, h = item
                full = h["load"] >= self.fanout
                return (full, h["load"] if full else 0, h["publisher"], h["load"])
            m, h = min(hs.items(), key=key)
            h["load"] += 1
            return {"member": m, "url": h["url"], "addr": h["addr"]}

    def release(self, version, source):
        with self.cv:
            h = self.holders.get(version, {}).get(source)
            if h and h["load"] > 0:
                h["load"] -= 1
            self.cv.notify_all()
            return {"ok": True}

    def report_failure(self, version, source, reporter):
        """``reporter`` could not read from ``source``: stop handing it out."""
        with self.cv:
            h = self.holders.get(version, {})
            if source in h and not h[source]["publisher"]:
                del h[source]
            self._ev("source_failed", source, {"version": version, "by": reporter})
            self.cv.notify_all()
            return {"ok": True}

    def retract(self, member, timeout=5.0):
        """``member`` is about to overwrite its arena: stop serving and wait
        (bounded) until nobody is reading from it."""
        with self.cv:
            def busy():
                return sum(h[member]["load"] for h in self.holders.values() if member in h)
            self.cv.wait_for(lambda: busy() == 0, timeout=timeout)
            for hs in self.holders.values():
                hs.pop(member, None)
            return {"ok": True}

    def wait_idle(self, version, member=PUBLISHER, timeout=5.0):
        with self.cv:
            ok = self.cv.wait_for(
                lambda: self.holders.get(version, {}).get(member, {"load": 0})["load"] == 0,
                timeout=timeout)
            return {"ok": bool(ok)}

    def done(self, version, member, url, addr):
        """``member`` holds a full copy of ``version`` and has applied it."""
        with self.cv:
            if member not in self.members:
                self.members[member] = {"url": url, "version": None, "last_seen": time.time()}
            self.members[member].update(version=version, last_seen=time.time())
            if version in self.holders:
                self.holders[version][member] = {"url": url, "addr": addr, "load": 0,
                                                 "publisher": False}
            self._ev("applied", member, version)
            self.cv.notify_all()
            return {"ok": True}

    def status(self):
        with self.lock:
            return {"latest": self.latest["version"] if self.latest else None,
                    "members": {m: i["version"] for m, i in self.members.items()},
                    "holders": {str(v): {m: h["load"] for m, h in hs.items()}
                                for v, hs in self.holders.items()}}

    def wait_all(self, version, timeout=60.0):
        with self.cv:
            return self.cv.wait_for(
                lambda: all(i["version"] == version for i in self.members.values()),
                timeout=timeout)


_METHODS = {"join", "heartbeat", "leave", "expire", "publish", "get_latest", "assign",
            "release", "report_failure", "retract", "wait_idle", "done", "status", "wait_all"}


class RegistryServer:
    def __init__(self, registry=None, host="0.0.0.0", port=0, expire_interval=1.0):
        self.reg = registry or WeightRegistry()
        reg = self.reg

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                name = self.path.strip("/")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                try:
                    if name not in _METHODS:
                        raise KeyError(name)
                    out, code = {"result": getattr(reg, name)(**body)}, 200
                except Exception as e:  # report to the client
                    out, code = {"error": f"{type(e).__name__}: {e}"}, 400
                data = json.dumps(out).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self.httpd.serve_forever, daemon=True),
                         threading.Thread(target=self._expire_loop, args=(expire_interval,),
                                          daemon=True)]
        for t in self._threads:
            t.start()

    def _expire_loop(self, every):
        while not self._stop.wait(every):
            self.reg.expire()

    def close(self):
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class RegistryClient:
    """Same methods as WeightRegistry, over HTTP."""

    def __init__(self, url, timeout=120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        # cluster-internal: never go through an HTTP(S)_PROXY from the env
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def __getattr__(self, name):
        if name not in _METHODS:
            raise AttributeError(name)

        def call(**kw):
            req = urllib.request.Request(f"{self.url}/{name}", data=json.dumps(kw).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with self._open(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["result"]
            except urllib.error.HTTPError as e:
                body = e.read()
                try:
                    msg = json.loads(body).get("error")
                except ValueError:
                    msg = f"HTTP {e.code}: {body[:200]!r}"
                raise RuntimeError(msg) from None
        return call
