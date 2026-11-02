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


def test_failed_connect_to_down_replica_is_quiet(caplog):
    """Issue #14: a failed re-HELLO to a replica that is already down is no
    state change: no warning, no failover."""
    h = _head()
    r = h.replicas[0][0]
    r.alive = False
    h.route[0] = None
    called = []
    h._failover = lambda *a: called.append(a)
    with caplog.at_level("INFO", logger=ex.log.name):
        h._on_failed((1,), "connect refused", ts=time.monotonic())
    time.sleep(0.05)
    assert not called and not [x for x in caplog.records if x.levelname == "WARNING"]


def test_rehello_backs_off_and_resets():
    h = _head()
    r = h.replicas[0][0]
    r.alive = False
    sent = []
    h._hello = lambda x, resolve=False: sent.append(x.idx)
    t = 1000.0
    times = []
    for k in range(120):        # two minutes, one tick per second
        n = len(sent)
        h._rehello_down(t + k)
        if len(sent) > n:
            times.append(k)
    assert times[:5] == [0, 2, 6, 14, 30]
    assert len(times) <= 7      # was 60 (every 2 s)
    # up again: the next outage starts at 2 s again
    h._on_message = ex.ConnexPPExecutor._on_message.__get__(h)
    h.owner = "o"
    h._on_message((1, 1, ex.CTL_SHARD, 0), ("HELLO_OK", "d", None))
    assert r.alive and r.hello_every == 2.0


def test_step_timeout_follows_measured_steps():
    """Issue #15: a hung stage is found in seconds, not after 60 s."""
    h = _head()
    h.step_timeout, h.step_timeout_min = 60.0, 10.0
    h.step_times = __import__("collections").deque(maxlen=512)
    assert h.current_step_timeout() == 60.0
    h.step_times.extend([(0.03, 64)] * 40)
    assert h.current_step_timeout(64) == 10.0
    h.step_times.append((1.0, 64))
    assert h.current_step_timeout(64) == 20.0
    h.step_times.append((5.0, 64))
    assert h.current_step_timeout(64) == 60.0


def test_step_timeout_compares_steps_of_the_same_size():
    """Issue #18: after many small decode steps, a step larger than every
    measured one (a long prefill) keeps the backstop instead of the 10 s
    floor, so a healthy stage is not failed over."""
    h = _head()
    h.step_timeout, h.step_timeout_min = 60.0, 10.0
    h.step_times = __import__("collections").deque(maxlen=512)
    h.step_times.extend([(0.03, 32)] * 40)
    assert h.current_step_timeout(32) == 10.0
    assert h.current_step_timeout(8192) == 60.0       # never seen a step this large
    h.step_times.append((1.5, 8192))
    assert h.current_step_timeout(8192) == 30.0       # 20 x the slowest large step
    assert h.current_step_timeout(32) == 30.0         # small steps: bounded by every step
