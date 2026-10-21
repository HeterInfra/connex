# SPDX-License-Identifier: Apache-2.0
"""connex.pp with fake engines (no vLLM / SGLang import, no GPU, no cluster).

Two layers:
* in-process fake transports (connex.pp.testing.fake_chain) for the step
  drivers and the handover contract;
* the real ConnexTEEndpoint over real ZeroMQ sockets on localhost with an
  in-process loopback transfer engine and static membership, for the full
  data path including failover to another replica.
"""

import socket
import sys
import threading
import time

import pytest
import torch
import zmq

from connex.pp import (
    CX_CTX, CX_MISSING, KVLedger, Stage0Link, StageWorker, StaticMembership, StepFailed,
    find_missing, merge_missing, wire,
)
from connex.pp.routing import Lid, Route
from connex.pp.steps import CorrFilter, ReplyCache
from connex.pp.testing import LoopbackTransferEngine, fake_chain


def test_no_engine_imported():
    import connex.pp  # noqa: F401
    import connex.adapters.vllm  # noqa: F401
    import connex.adapters.sglang  # noqa: F401
    assert not any(m == "sglang" or m.startswith("sglang.") for m in sys.modules)


# ---- fake engines -----------------------------------------------------------------

class Stage0Engine:
    def __init__(self):
        self.reprefilled, self.aborted = [], []

    def kv_held(self, rid):
        return 0

    def release(self, rids):
        pass

    def reprefill(self, rids):
        self.reprefilled.extend(rids)

    def abort(self, rids):
        self.aborted.extend(rids)


class DownstreamEngine:
    """Holds per-request KV lengths (KVLedger); forward adds 1 to 'h'."""

    def __init__(self, name="s"):
        self.name = name
        self.ledger = KVLedger()
        self.released = []
        self.forwards = 0

    def kv_held(self, rid):
        return self.ledger.kv_held(rid)

    def release(self, rids):
        rids = list(rids)
        self.released.extend(rids)
        self.ledger.forget(rids)

    def reprefill(self, rids):
        raise AssertionError("only stage 0 re-prefills")

    def abort(self, rids):
        raise AssertionError("only stage 0 aborts")

    def compute(self, td, missing):
        self.forwards += 1
        out = {"h": td["h"] + 1}
        if "nsched" in td:  # engine metadata travels with the engine's own dict
            out["nsched"] = td["nsched"]
        return out

    def commit(self, td, missing):
        self.ledger.commit(td.get("request_ids"), td.get(CX_CTX), td.get("nsched"), missing)


def _worker(link, eng):
    return StageWorker(link, eng, eng.compute, commit=eng.commit)


class Pump:
    """Run StageWorkers in background threads."""

    def __init__(self, workers):
        self.workers, self.stop = workers, threading.Event()
        self.threads = [threading.Thread(target=self._run, args=(w,), daemon=True) for w in workers]

    def _run(self, w):
        while not self.stop.is_set():
            try:
                w.serve_one()
            except Exception:  # noqa: BLE001  (endpoint closed under us)
                if self.stop.is_set():
                    return
                raise

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        for t in self.threads:
            t.join(5)


def _prefill_then_decode(s0, rids, prompt=4, decode=3):
    """Drive requests: one prefill step then `decode` decode steps."""
    n = {r: 0 for r in rids}
    out = s0.step({"h": torch.zeros(len(rids)), "nsched": [prompt] * len(rids)},
                  rids, [n[r] for r in rids])
    for r in rids:
        n[r] = prompt
    replies = [out]
    for _ in range(decode):
        out = s0.step({"h": torch.zeros(len(rids))}, rids, [n[r] for r in rids])
        replies.append(out)
        for r in rids:
            n[r] += 1
    return replies, n


# ---- steady state -----------------------------------------------------------------

def test_steady_state_three_stages():
    l0, l1, l2 = fake_chain(3)
    e0, e1, e2 = Stage0Engine(), DownstreamEngine(), DownstreamEngine()
    s0 = Stage0Link(l0, e0, recv_timeout_s=0.5)
    with Pump([_worker(l1, e1), _worker(l2, e2)]):
        replies, n = _prefill_then_decode(s0, ["a", "b"])
    assert all(torch.equal(r["h"], torch.full((2,), 2.0)) for r in replies)
    assert all(CX_MISSING not in r for r in replies)
    assert e1.ledger.kv_len == e2.ledger.kv_len == {"a": 7, "b": 7}
    assert e0.reprefilled == [] and s0.stats["retries"] == 0


def test_finished_ids_release_downstream():
    l0, l1 = fake_chain(2)
    e1 = DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.5, finished_repeat=2)
    with Pump([_worker(l1, e1)]):
        _prefill_then_decode(s0, ["a", "b"], decode=1)
        s0.finished(["a"])
        s0.step({"h": torch.zeros(1)}, ["b"], [5])
        s0.step({"h": torch.zeros(1)}, ["b"], [6])
        s0.step({"h": torch.zeros(1)}, ["b"], [7])
    assert e1.released == ["a", "a"] and "a" not in e1.ledger.kv_len  # sent on 2 steps


# ---- duplicate / retry ---------------------------------------------------------------

def test_lost_reply_is_retried_and_answered_from_cache():
    l0, l1, l2 = fake_chain(3)
    e1, e2 = DownstreamEngine(), DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.2)
    with Pump([_worker(l1, e1), _worker(l2, e2)]):
        _prefill_then_decode(s0, ["a"], decode=1)
        l0.inbox.drop_next = 1          # the next reply to stage 0 is lost
        out = s0.step({"h": torch.zeros(1)}, ["a"], [5])
    assert torch.equal(out["h"], torch.full((1,), 2.0))
    assert s0.stats["retries"] == 1
    # the re-sent step was answered from the stages' reply caches, not recomputed
    assert e1.forwards == e2.forwards == 3
    assert e1.ledger.kv_len == {"a": 6}


def test_duplicated_message_does_not_recompute():
    l0, l1 = fake_chain(2)
    e1 = DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.5)
    with Pump([_worker(l1, e1)]):
        l1.inbox.dup_next = 1           # network duplicates the forward step
        s0.step({"h": torch.zeros(1), "nsched": [4]}, ["a"], [0])
        s0.step({"h": torch.zeros(1)}, ["a"], [4])
    assert e1.forwards == 2 and l1.computed == 2


def test_no_reply_and_unhealthy_downstream_fails_step():
    l0, l1 = fake_chain(2)
    l1.inbox.dead = True
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.02, max_attempts=3)
    with pytest.raises(StepFailed):
        s0.step({"h": torch.zeros(1)}, ["a"], [0])
    assert not l0.outstanding and l0.sent == 3


# ---- stale filter ---------------------------------------------------------------------

def test_stale_step_dropped_inside_window_accepted():
    f = CorrFilter(window=2)
    b = 0x77 << 32
    assert f.accept(b + 5) and f.accept(b + 4)       # 1 behind: inside K=2
    assert not f.accept(b + 3)                       # 2 behind: stale
    assert f.accept((0x78 << 32) + 1)                # other stage-0 incarnation
    assert f.accept(0)                               # no id: cannot judge


def test_fake_chain_drops_stale_resend():
    l0, l1 = fake_chain(2)
    e1 = DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.5)
    with Pump([_worker(l1, e1)]):
        s0.step({"h": torch.zeros(1), "nsched": [4]}, ["a"], [0])
        s0.step({"h": torch.zeros(1)}, ["a"], [4])
        s0.step({"h": torch.zeros(1)}, ["a"], [5])
        old = l0.ids.base + 1                         # a late copy of step 1
        l0.downstream.replies.clear()                 # not in the reply cache any more
        l1.inbox.put((old, {"h": torch.zeros(1)}))
        time.sleep(0.3)
    assert e1.forwards == 3


def test_reply_cache_evicts_oldest():
    ev = []
    c = ReplyCache(2, on_evict=ev.append)
    for i in range(4):
        c.put(i, f"r{i}")
    assert list(c.entries) == [2, 3] and ev == ["r0", "r1"]


# ---- K > 1, out of order ----------------------------------------------------------------

def test_k_in_flight_out_of_order_waits():
    l0, l1, l2 = fake_chain(3, k=3)
    e1, e2 = DownstreamEngine(), DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=0.5)
    with Pump([_worker(l1, e1), _worker(l2, e2)]):
        ids = [s0.submit({"h": torch.full((1,), float(i)), "nsched": [4]}, [f"r{i}"], [0])
               for i in range(3)]
        outs = {c: s0.wait(c) for c in reversed(ids)}  # wait in reverse order
    assert [outs[c]["h"].item() for c in ids] == [2.0, 3.0, 4.0]
    assert not l0.outstanding and not l0.done


# ---- failover + missing-KV report ------------------------------------------------------------

def test_failover_to_fresh_replica_reports_missing_and_reprefills():
    l0, l1a, l2 = fake_chain(3)
    l1b = fake_chain(3)[1]
    l1b.head, l1b.downstream = l0, l2
    e0, e1a, e1b, e2 = Stage0Engine(), DownstreamEngine("1a"), DownstreamEngine("1b"), DownstreamEngine()
    s0 = Stage0Link(l0, e0, recv_timeout_s=0.5)
    with Pump([_worker(l1a, e1a), _worker(l1b, e1b), _worker(l2, e2)]):
        _, n = _prefill_then_decode(s0, ["a", "b"], decode=2)
        l0.downstream = l1b                  # stage 1 replica a died; failover to b
        out = s0.step({"h": torch.zeros(3)}, ["a", "b", "c"], [n["a"], n["b"], 0])
        assert out["__cx_reprefilled__"] == ["a", "b"]
        assert e0.reprefilled == ["a", "b"]
        assert CX_MISSING not in out
        # stage 0 rewinds a and b to 0 and re-prefills prompt + generated tokens
        out = s0.step({"h": torch.zeros(3), "nsched": [6, 6, 1]}, ["a", "b", "c"], [0, 0, 1])
        assert "__cx_reprefilled__" not in out
        out = s0.step({"h": torch.zeros(2)}, ["a", "b"], [6, 6])
        assert "__cx_reprefilled__" not in out
    assert sorted(e1b.released) == ["a", "b"]
    assert e1b.ledger.kv_len["a"] == 7
    # stage 2 held a and b all along: it was not asked to drop them
    assert e2.released == []


def test_rewind_budget_aborts():
    l0, l1 = fake_chain(2)
    e0, e1 = Stage0Engine(), DownstreamEngine()
    s0 = Stage0Link(l0, e0, recv_timeout_s=0.5, reprefill_max=2)
    with Pump([_worker(l1, e1)]):
        for _ in range(3):
            s0.step({"h": torch.zeros(1)}, ["a"], [9])   # stage 1 never holds 'a'
    assert e0.reprefilled == ["a", "a"] and e0.aborted == ["a"]


def test_find_missing_and_merge_rules():
    held = {"a": 5, "b": 3}.get
    assert find_missing(["a", "b", "c"], [5, 4, 0], lambda r: held(r, 0)) == ["b"]
    assert find_missing(["a"], None, lambda r: 0) == []
    assert merge_missing(["x"], ["y", "x"]) == ["x", "y"]
    assert merge_missing(None, []) is None


def test_ledger_matches_vllm_guard_rules():
    """KVLedger + find_missing reproduce ConnexPPKVGuard's decode rules."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "integrations/vllm/patches/external_pp_reprefill.py"
    spec = importlib.util.spec_from_file_location("rp", p)
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    class Alloc:
        def __init__(self):
            self.reqs = set()

        def release(self, r):
            self.reqs.discard(r)
    al, g, led = Alloc(), rp.ConnexPPKVGuard(), KVLedger(is_live=lambda r: r in al.reqs)
    script = [(["a"], [0], [8]), (["a"], [8], None), (["a", "n"], [9, 0], None),
              (["a"], [15], None), (["a"], [0], [16]), (["a"], [16], None)]
    for rids, ctx, ns in script:
        st = "prefill" if ns else "decode"
        m1 = g.check(al, rids, ctx, st, None, 16)
        m2 = find_missing(rids, ctx, led.kv_held)
        assert m1 == m2
        led.forget(m2)
        al.reqs.update(r for r in rids if r not in m1)
        g.commit(al, rids, ctx, st, ns, m1)
        led.commit(rids, ctx, ns, m2)
        assert {r: v for r, v in g.kv_len.items()} == led.kv_len


# ---- real endpoint over ZeroMQ + loopback transfer engine ------------------------------------

def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def real(monkeypatch):
    import connex.pp.endpoint as core
    monkeypatch.setenv("CONNEX_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(core, "_CONNEXTE_RECV_TIMEOUT_MS", 400)
    made = []

    def mk(stage, n, replica=0, k=1, routes=None):
        port = _port()
        mem = StaticMembership(stage, replica)
        if routes is not None:
            mem.routes = routes
        url = f"loop-{stage}-{replica}-{port}"
        ep = core.ConnexTEEndpoint(stage, n, port, "127.0.0.1:1", device="cpu",
                                   membership=mem, transfer_engine=LoopbackTransferEngine(url),
                                   local_url=url, inflight=k)
        made.append(ep)
        return ep, mem

    yield mk
    for ep in made:
        try:
            ep.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.parametrize("k", [1, 2])
def test_real_endpoints_steady_state(real, k):
    routes = []
    eps = []
    for s in range(3):
        ep, mem = real(s, 3, k=k, routes=routes)
        mem.add(s, 0, ep.router_endpoint)   # mem.routes is the shared list
        eps.append(ep)
    assert eps[0]._mooncake_available and eps[0]._gdr is None  # host path
    e1, e2 = DownstreamEngine(), DownstreamEngine()
    s0 = Stage0Link(eps[0], Stage0Engine(), recv_timeout_s=2.0)
    with Pump([_worker(eps[1], e1), _worker(eps[2], e2)]):
        replies, _ = _prefill_then_decode(s0, ["a", "b"], decode=4)
    assert all(torch.equal(r["h"], torch.full((2,), 2.0)) for r in replies)
    assert e2.ledger.kv_len == {"a": 8, "b": 8}


def test_real_endpoints_failover_reprefills(real):
    from connex.epc_client import EventType
    routes = []
    s0ep, _ = real(0, 3, routes=routes)
    s1a, _ = real(1, 3, replica=0, routes=routes)
    s1b, _ = real(1, 3, replica=1, routes=routes)
    s2, _ = real(2, 3, routes=routes)
    for ep, rep in ((s0ep, 0), (s1a, 0), (s1b, 1), (s2, 0)):
        routes.append(Route(Lid(ep.stage_idx, rep), [ep.router_endpoint]))
    e0, e1a, e1b, e2 = Stage0Engine(), DownstreamEngine("1a"), DownstreamEngine("1b"), DownstreamEngine()
    s0 = Stage0Link(s0ep, e0, recv_timeout_s=0.5, max_attempts=10)
    pump_a = Pump([_worker(s1a, e1a)])
    with Pump([_worker(s1b, e1b), _worker(s2, e2)]), pump_a:
        _, n = _prefill_then_decode(s0, ["a", "b"], decode=2)
        assert s0ep._sticky_peer[1] == 0 and e1b.forwards == 0
        # replica 1a leaves: stop serving, router drops it, event reaches stage 0
        pump_a.stop.set()
        pump_a.threads[0].join(5)
        routes[:] = [r for r in routes if (r.lid.stage_idx, r.lid.replica_idx) != (1, 0)]
        s0ep._epc_client.emit(EventType.ENDPOINT_DEREGISTERED, 1, 0)
        out = s0.step({"h": torch.zeros(2)}, ["a", "b"], [n["a"], n["b"]])
        assert e0.reprefilled == ["a", "b"] and out["__cx_reprefilled__"] == ["a", "b"]
        assert s0ep._sticky_peer[1] == 1
        out = s0.step({"h": torch.zeros(2), "nsched": [7, 7]}, ["a", "b"], [0, 0])
        assert "__cx_reprefilled__" not in out
        out = s0.step({"h": torch.zeros(2)}, ["a", "b"], [7, 7])
        assert "__cx_reprefilled__" not in out and torch.equal(out["h"], torch.full((2,), 2.0))
    assert sorted(e1b.released) == ["a", "b"] and e1b.ledger.kv_len == {"a": 8, "b": 8}
    assert e2.released == []


def test_wire_signal_roundtrip():
    tail = wire.encode_g2(7, wire.encode_layout([("h", "bfloat16", (2, 3), 0, 12)]),
                          b"tcp://a:1", b"tcp://s0:1", b"")
    sig = wire.encode_signal(0x1000, 256, b"seg:1", (5 << 32) | 9, tail)
    d = wire.decode_signal(sig)
    assert (d.src_ptr, d.total, d.url, d.corr, d.tag) == (0x1000, 256, "seg:1", (5 << 32) | 9, b"G2")
    tok, lb, ack, origin, nt = wire.decode_g2(d.tail[2:])
    assert (tok, ack, origin, nt) == (7, "tcp://a:1", "tcp://s0:1", b"")
    assert wire.decode_layout(lb) == [("h", "bfloat16", (2, 3), 0, 12)]
    assert wire.signal_corr(sig) == (5 << 32) | 9
    assert wire.decode_ack(wire.encode_ack(42)) == 42


# ---- fast handover: all in-flight steps fail over at once -------------------------------

def test_multi_slot_failover_resends_all_inflight_at_once():
    """K=4: stage 1 replica a dies with 4 steps in flight. One recv notices
    the failover and Stage0Link re-sends every stale step together, so the
    chain recovers in ~one step, not one recv timeout per slot."""
    l0, l1a, l2 = fake_chain(3, k=4)
    l1b = fake_chain(3, k=4)[1]
    l1b.head, l1b.downstream = l0, l2
    e1b, e2 = DownstreamEngine("1b"), DownstreamEngine()
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=5.0)
    l1a.inbox.dead = True                        # a: SIGKILLed, nothing gets through
    with Pump([_worker(l1b, e1b), _worker(l2, e2)]):
        ids = [s0.submit({"h": torch.full((1,), float(i)), "nsched": [4]}, [f"r{i}"], [0])
               for i in range(4)]
        threading.Timer(0.1, lambda: l0.fail_over(l1b)).start()
        t = time.monotonic()
        outs = [s0.wait(ids[0])]
        assert l0.sent == 8                      # all 4 re-sent together
        outs += [s0.wait(c) for c in ids[1:]]
        took = time.monotonic() - t
    assert [o["h"].item() for o in outs] == [2.0, 3.0, 4.0, 5.0]
    assert took < 1.5, took                      # not 4 x 5 s
    assert l0.sent == 8 and s0.stats["retries"] == 4
    assert not l0.outstanding and not l0.done


def test_fresh_steps_after_failover_are_not_resent():
    l0, l1a, l2 = fake_chain(3, k=2)
    l1b = fake_chain(3, k=2)[1]
    l1b.head, l1b.downstream = l0, l2
    l0.fail_over(l1b)                            # before anything was sent
    s0 = Stage0Link(l0, Stage0Engine(), recv_timeout_s=2.0)
    with Pump([_worker(l1b, DownstreamEngine()), _worker(l2, DownstreamEngine())]):
        ids = [s0.submit({"h": torch.zeros(1), "nsched": [4]}, [f"r{i}"], [0]) for i in range(2)]
        [s0.wait(c) for c in ids]
    assert l0.sent == 2 and s0.stats["retries"] == 0


def test_real_endpoints_multi_slot_failover_by_probe(real):
    """Real endpoints, K=4: stage-1 replica a is killed (port closed) with 4
    steps in flight. UNHEALTHY -> TCP probe refused -> failover at once; the
    stale steps are re-sent together and complete on replica b."""
    from connex.epc_client import EventType
    routes = []
    s0ep, _ = real(0, 3, k=4, routes=routes)
    s1a, _ = real(1, 3, replica=0, k=4, routes=routes)
    s1b, _ = real(1, 3, replica=1, k=4, routes=routes)
    s2, _ = real(2, 3, k=4, routes=routes)
    for ep, rep in ((s0ep, 0), (s1a, 0), (s1b, 1), (s2, 0)):
        routes.append(Route(Lid(ep.stage_idx, rep), [ep.router_endpoint]))
    s0 = Stage0Link(s0ep, Stage0Engine(), recv_timeout_s=10.0)
    pump_a = Pump([_worker(s1a, DownstreamEngine("1a"))])
    with Pump([_worker(s1b, DownstreamEngine("1b")), _worker(s2, DownstreamEngine())]), pump_a:
        _prefill_then_decode(s0, ["w"], decode=1)  # warm: sticky peer = a
        assert s0ep._sticky_peer[1] == 0
        pump_a.stop.set()
        pump_a.threads[0].join(5)
        s1a.close()                              # dead: connects are refused
        ids = [s0.submit({"h": torch.full((1,), float(i)), "nsched": [4]}, [f"r{i}"], [0])
               for i in range(4)]
        t = time.monotonic()
        s0ep._epc_client.emit(EventType.ENDPOINT_UNHEALTHY, 1, 0)   # missed lease
        outs = [s0.wait(c) for c in ids]
        took = time.monotonic() - t
    assert [o["h"].item() for o in outs] == [2.0, 3.0, 4.0, 5.0]
    assert s0ep._sticky_peer[1] == 1 and s0ep._chain_epoch == 1
    assert took < 3.0, took                      # not 4 x 10 s
    assert s0.stats["retries"] >= 4


# ---- fast handover: K=1 races (send and recv on different threads) -----------------------

def _k1_pair(real):
    routes = []
    s0ep, _ = real(0, 2, routes=routes)
    s1, _ = real(1, 2, routes=routes)
    for ep in (s0ep, s1):
        routes.append(Route(Lid(ep.stage_idx, 0), [ep.router_endpoint]))
    return s0ep, s1


def _serve_k1(s1, times=1):
    td = s1.recv_tensor_dict()
    for _ in range(times):
        s1.send_tensor_dict({"h": td["h"] + 1})


def test_k1_late_duplicate_reply_not_taken_for_next_step(real):
    """A re-sent step's late duplicate reply reaches stage 0 before the
    send thread has stamped the next step: it must be dropped, never taken
    for the next step (that gave wrong tokens and an engine crash)."""
    s0ep, s1 = _k1_pair(real)
    s0ep.send_tensor_dict({"h": torch.zeros(1)})
    _serve_k1(s1, times=2)                       # reply + late duplicate
    assert s0ep.recv_tensor_dict()["h"].item() == 1.0
    time.sleep(0.2)                              # the duplicate is waiting
    got = {}
    th = threading.Thread(target=lambda: got.setdefault("r", s0ep.recv_tensor_dict()))
    th.start()                                   # recv first, as in the fork
    time.sleep(0.05)
    s0ep.send_tensor_dict({"h": torch.full((1,), 10.0)})
    _serve_k1(s1)
    th.join(5)
    assert got["r"]["h"].item() == 11.0


class _StealOnce:
    """Wraps the real ROUTER socket: the first message polled is taken by
    'another thread' (what _drain_acks on the send path does)."""

    def __init__(self, ep):
        self.ep, self.sock, self.stolen = ep, ep.router_socket, False

    def poll(self, *a, **k):
        return self.sock.poll(*a, **k)

    def recv_multipart(self, flags=0):
        if not self.stolen:
            self.stolen = True
            self.ep._stash.append(self.sock.recv_multipart())
            raise zmq.Again()
        return self.sock.recv_multipart(flags)

    def __getattr__(self, name):
        return getattr(self.sock, name)


def test_k1_message_taken_by_drain_acks_is_not_a_timeout(real):
    s0ep, s1 = _k1_pair(real)
    s0ep.send_tensor_dict({"h": torch.zeros(1)})
    _serve_k1(s1)
    real_sock = s0ep.router_socket
    s0ep.router_socket = _StealOnce(s0ep)
    try:
        t = time.monotonic()
        out = s0ep.recv_tensor_dict()
    finally:
        s0ep.router_socket = real_sock
    assert out is not None and out["h"].item() == 1.0
    assert time.monotonic() - t < 0.3
