# SPDX-License-Identifier: Apache-2.0
"""Stage 0's EXEC/SAMPLE sends leave the engine thread (issue #39): per-peer
order, errors, drop on failover, shutdown, backpressure, and the inline
mode (CONNEX_PP_ASYNC_CTRL=0). CPU only, no vLLM needed."""
import logging
import pickle
import threading
import time
from types import SimpleNamespace

import pytest

from connex.adapters.vllm.pp.ctltx import CtlSender

from _fake_vllm import load_executor


class Wire:
    """send_msg stand-in: records (peer, message) in send order, optionally
    slow, failing for some peers, or blocking until released."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.log = []
        self.lock = threading.Lock()
        self.fail = set()
        self.gate = None     # threading.Event: sends wait for it

    def send(self, peer, data):
        if self.gate is not None:
            self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)
        if peer in self.fail:
            raise OSError(f"no route to {peer}")
        with self.lock:
            self.log.append((peer, pickle.loads(data)))

    def to(self, peer):
        with self.lock:
            return [m for p, m in self.log if p == peer]


def _wait(cond, t=5.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ----------------------------------------------------------- CtlSender alone
def test_per_peer_fifo_across_kinds_and_peers():
    w = Wire()
    tx = CtlSender(w.send)
    for seq in range(50):
        for kind in ("EXEC", "SAMPLE"):
            for peer in ("a", "b", "c"):
                tx.put(peer, pickle.dumps((kind, seq)))
    assert tx.flush(5)
    want = [(k, s) for s in range(50) for k in ("EXEC", "SAMPLE")]
    for peer in ("a", "b", "c"):
        assert w.to(peer) == want
    assert tx.sent == 300 and tx.failed == 0


def test_failed_send_is_reported_once_per_run_and_others_continue(caplog):
    w = Wire()
    w.fail.add("b")
    errs = []
    tx = CtlSender(w.send, on_error=lambda p, e: errs.append((p, type(e))))
    with caplog.at_level(logging.WARNING, logger="vllm.connex_pp"):
        for seq in range(5):
            tx.put("a", pickle.dumps(("EXEC", seq)))
            tx.put("b", pickle.dumps(("EXEC", seq)))
        tx.flush(5)
    assert errs == [("b", OSError)]          # once, not per message
    assert tx.failed == 5 and len(w.to("a")) == 5
    assert sum("control send to b failed" in r.message for r in caplog.records) == 1
    # it works again, then fails again: reported again
    w.fail.clear()
    tx.put("b", pickle.dumps(("EXEC", 5)))
    tx.flush(5)
    w.fail.add("b")
    tx.put("b", pickle.dumps(("EXEC", 6)))
    tx.flush(5)
    assert errs == [("b", OSError)] * 2


def test_sender_failure_is_raised_on_the_next_put():
    def broken(peer, data):
        raise SystemExit("boom")      # not an ordinary send error
    tx = CtlSender(broken)
    tx.put("a", b"x")
    assert _wait(lambda: tx.error is not None)
    with pytest.raises(RuntimeError, match="control sender failed"):
        tx.put("a", b"y")
    with pytest.raises(RuntimeError):
        tx.put("other", b"y")


def test_queue_is_bounded_and_put_waits_for_room():
    w = Wire()
    w.gate = threading.Event()
    tx = CtlSender(w.send, depth=4)
    for i in range(5):          # one in flight (held by the gate), four queued
        assert tx.put("a", pickle.dumps(i))
    done = threading.Event()
    threading.Thread(target=lambda: (tx.put("a", pickle.dumps(5)), done.set()),
                     daemon=True).start()
    assert not done.wait(0.2)                # backpressure: the engine waits
    assert len(tx.peers["a"].q) == 4
    w.gate.set()
    assert done.wait(5)
    tx.flush(5)
    assert w.to("a") == list(range(6))


def test_drop_discards_queue_and_releases_a_waiting_put():
    w = Wire()
    w.gate = threading.Event()
    tx = CtlSender(w.send, depth=2)
    for i in range(3):
        tx.put("a", pickle.dumps(i))
    res = []
    t = threading.Thread(target=lambda: res.append(tx.put("a", pickle.dumps(9))), daemon=True)
    t.start()
    time.sleep(0.1)
    assert tx.drop("a") == 2
    t.join(5)
    assert res == [False]                    # the waiting put gave up
    tx.put("b", pickle.dumps(7))             # other peers unaffected
    w.gate.set()
    tx.flush(5)
    assert w.to("a") == [0]                  # only the send already in progress
    assert w.to("b") == [7]
    tx.put("a", pickle.dumps(10))            # usable again (back on the route)
    tx.flush(5)
    assert w.to("a") == [0, 10]


def test_close_drains_then_discards():
    w = Wire(delay=0.01)
    tx = CtlSender(w.send)
    for i in range(5):
        tx.put("a", pickle.dumps(i))
    tx.close(drain_s=2.0)
    assert w.to("a") == list(range(5))
    assert tx.put("a", b"x") is False        # closed
    w2 = Wire()
    w2.gate = threading.Event()
    tx2 = CtlSender(w2.send)
    for i in range(5):
        tx2.put("a", pickle.dumps(i))
    t0 = time.monotonic()
    tx2.close(drain_s=0.1)                   # a hung peer bounds shutdown
    assert time.monotonic() - t0 < 1.0
    w2.gate.set()
    time.sleep(0.1)
    assert w2.to("a") == [0] and tx2.dropped == 4


# ------------------------------------------------- through the PP executor
@pytest.fixture
def exmod():
    mod, restore = load_executor()
    yield mod
    restore()


def _head(ex, wire, async_ctrl=True):
    h = ex.ConnexPPExecutor.__new__(ex.ConnexPPExecutor)
    h.head = True
    h.lock = threading.RLock()
    h.cv = threading.Condition(h.lock)
    h.num_stages = 4
    h.replicas = [[ex.Replica(k, 0, f"ctl{k}", f"d{k}", alive=True,
                              alive_since=time.monotonic() - 100)] for k in (1, 2, 3)]
    h.route = [lst[0] for lst in h.replicas]
    h.epoch, h.seq, h.replica_id, h.owner = 1, 0, 0, "o"
    h.pending, h.live_reqs, h.failovers = {}, set(), []
    h.await_sample = __import__("collections").deque()
    h.lost_pending = False
    h.replace_timeout = 0.5
    h.max_tokens = 8192
    h.local_data_url = "d0"
    sent_inline = []

    def inline(peer, msg, raw=None, quiet=False):
        wire.send(peer, raw)
        sent_inline.append(peer)
        return True
    h.ctl = SimpleNamespace(encode=lambda m: pickle.dumps(m[:3]), send=inline,
                            engine=SimpleNamespace(send_msg=wire.send), close=lambda: None)
    h.sent_inline = sent_inline
    h.ctl_tx = CtlSender(wire.send, on_error=h._on_send_failed) if async_ctrl else None
    return h


def _so(n=16):
    return SimpleNamespace(total_num_scheduled_tokens=n, num_scheduled_tokens={"r": n},
                           scheduled_new_reqs=[], finished_req_ids=set())


def _steps(h, n):
    for _ in range(n):
        h.execute_model(_so())
        h.sample_tokens(None, non_block=True)


def test_executor_order_per_stage(exmod):
    w = Wire(delay=0.0005)
    h = _head(exmod, w)
    _steps(h, 20)
    assert h.ctl_tx.flush(10)
    want = [(k, s, 1) for s in range(1, 21) for k in ("EXEC", "SAMPLE")]
    for k in (1, 2, 3):
        assert w.to(f"ctl{k}") == want
    assert not h.sent_inline


def test_executor_does_not_wait_for_slow_sends(exmod):
    w = Wire(delay=0.02)                    # 20 ms per send, 3 peers
    h = _head(exmod, w)
    t0 = time.monotonic()
    _steps(h, 3)                            # inline: 3 x 2 x 3 x 20 ms = 360 ms
    assert time.monotonic() - t0 < 0.1
    assert h.ctl_tx.flush(5)
    assert len(w.log) == 18


def test_inline_mode(exmod):
    w = Wire()
    h = _head(exmod, w, async_ctrl=False)
    _steps(h, 3)
    assert h.sent_inline == ["ctl1", "ctl2", "ctl3"] * 6
    assert w.to("ctl2") == [(k, s, 1) for s in (1, 2, 3) for k in ("EXEC", "SAMPLE")]


@pytest.mark.parametrize("env,want", [(None, True), ("1", True), ("0", False)])
def test_init_head_reads_the_env(exmod, monkeypatch, env, want):
    """CONNEX_PP_ASYNC_CTRL=0 sends inline; anything else queues."""
    if env is None:
        monkeypatch.delenv("CONNEX_PP_ASYNC_CTRL", raising=False)
    else:
        monkeypatch.setenv("CONNEX_PP_ASYNC_CTRL", env)
    monkeypatch.setenv("CONNEX_PP_STAGES", '[["10.0.0.2:1"]]')
    monkeypatch.setenv("CONNEX_PP_SETTLE_S", "0")
    monkeypatch.setenv("CONNEX_PP_CTL_QUEUE", "7")
    holder = {}

    class Link:
        def __init__(self, *a, on_message=None, on_failed=None, **k):
            self.url, self.closed, self.on_message = "me", True, on_message   # no watchdog loop
            self.engine = SimpleNamespace(send_msg=lambda p, d: None)
            holder["link"] = self

        def send(self, peer, msg, raw=None, quiet=False):
            if msg and msg[0] == "HELLO":   # the replica answers at once
                self.on_message((exmod.MODEL_ID, 1, exmod.CTL_SHARD, 0),
                                ("HELLO_OK", "d1", None))
            return True
    monkeypatch.setattr(exmod, "ControlLink", Link)
    h = exmod.ConnexPPExecutor.__new__(exmod.ConnexPPExecutor)
    h.num_stages = 2
    h.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(
        max_num_batched_tokens=8192))
    h._init_head()
    assert (h.ctl_tx is not None) == want
    if want:
        assert h.ctl_tx.depth == 7 and h.ctl_tx.on_error == h._on_send_failed


def test_failed_send_fails_the_replica_over(exmod):
    w = Wire()
    w.fail.add("ctl2")
    h = _head(exmod, w)
    calls = []
    h._failover = lambda r, why: calls.append((r.stage, why))
    _steps(h, 2)
    assert _wait(lambda: calls)
    assert calls[0][0] == 2 and "control send failed" in calls[0][1]
    assert not h.replicas[1][0].alive
    assert len(calls) == 1                  # once, not per message


def test_failover_drops_what_is_queued_for_the_replaced_replica(exmod):
    w = Wire()
    w.gate = threading.Event()              # nothing leaves yet
    h = _head(exmod, w)
    h._send_abort = lambda urls, epoch: None
    h.replicas[1].append(exmod.Replica(2, 1, "ctl2b", "d2b", alive=True))
    _steps(h, 3)
    r = h.replicas[1][0]
    r.alive = False
    h._failover(r, "test")
    assert h.route[1].ctl_url == "ctl2b" and h.epoch == 2
    assert not h.ctl_tx.peers["ctl2"].q     # queued steps of the dead replica: gone
    w.gate.set()
    assert h.ctl_tx.flush(5)
    assert len(w.to("ctl2")) <= 1           # at most the send already in progress
    assert len(w.to("ctl1")) == 6 and len(w.to("ctl3")) == 6


def test_shutdown_drains_the_queues(exmod):
    w = Wire(delay=0.005)
    h = _head(exmod, w)
    _steps(h, 4)
    h.shutdown()
    assert len(w.log) == 24
    assert h.ctl_tx.closed
