"""PP head: failure events are handled off the event thread, and events that
predate the current incarnation are dropped without a PING (issue #1)."""
import threading
import time

import pytest

pytest.importorskip("vllm")

from connex.adapters.vllm.pp import executor as ex  # noqa: E402


def _head(replace_timeout=5.0):
    h = ex.ConnexPPExecutor.__new__(ex.ConnexPPExecutor)
    h.lock = threading.Lock()
    h.cv = threading.Condition(h.lock)
    h.replicas = [[ex.Replica(stage=1, idx=0, ctl_url="a")],
                  [ex.Replica(stage=2, idx=0, ctl_url="b")]]
    h.num_stages = 3
    h.route = [h.replicas[0][0], h.replicas[1][0]]
    for lst in h.replicas:
        for r in lst:
            r.alive, r.alive_since = True, time.monotonic()
    h.epoch, h.seq, h.replica_id = 0, 0, 0
    h.pending, h.live_reqs, h.failovers = {}, set(), []
    h.lost_pending = False
    h.replace_timeout = replace_timeout
    h.pings = 0

    def ping(r, timeout=1.5):
        h.pings += 1
        return False   # the old incarnation is gone
    h._answers_ping = ping
    h._hello_all = lambda reps: None
    h._send_abort = lambda urls, epoch: None
    h._find = lambda peer: h.replicas[peer[0] - 1][0]
    return h


def test_stale_event_dropped_without_ping():
    h = _head()
    r = h.replicas[0][0]
    h._on_failed((1,), "old", ts=r.alive_since - 1.0)
    assert h.pings == 0 and r.alive


def test_failover_without_spare_does_not_block_event_thread():
    h = _head(replace_timeout=3.0)
    t0 = time.monotonic()
    h._on_failed((1,), "dead", ts=time.monotonic() + 1)     # no spare: waits 3 s
    assert time.monotonic() - t0 < 0.5
    # a second stage's failure is seen at once, not after the first wait
    h.replicas[1][0].alive_since = 0.0
    h._on_failed((2,), "dead", ts=time.monotonic() + 1)
    assert time.monotonic() - t0 < 0.5
    assert not h.replicas[1][0].alive
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and h.route[1] is not None:
        time.sleep(0.05)
    assert h.route == [None, None]
