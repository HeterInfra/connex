# SPDX-License-Identifier: Apache-2.0
"""CONNEX_PP_INFLIGHT=K: per-step correlation, matching, retry/dedupe and the
gpu_worker pipelined-loop patch. Mock only (no GPU, Mooncake or cluster)."""

import ast
import os
import pickle
import shutil
import subprocess
import sys
import threading

import pytest

from test_connexte_protocol import FakeRouter, _comm, _sig, cx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "integrations", "vllm", "patches")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fork import FORK_V1  # noqa: E402  (set VLLM_FORK_DIR)
WORKER_DIR = os.path.join(FORK_V1, "worker")
BASE = 0xABCD << 32


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    monkeypatch.setattr(cx, "METRICS_ENABLED", False)


def _reply(corr, mb, origin=b"tcp://s0:1"):
    nt = pickle.dumps({"mb": mb})
    return _sig(corr, cx._encode_g2(0, cx._encode_layout([]), b"", origin, nt))


def _k(stage, k, msgs=()):
    c, sent = _comm(stage, msgs=msgs)
    c._corr_base, c._corr_seq = BASE, 0
    c._init_inflight(k)
    return c, sent


def _capture_sends(c):
    """Replace the data path: record (corr, origin) the send path would stamp."""
    out = []

    def send(td, dst=None):
        out.append((c._corr_for_send(), c._origin_ep(), td))
    c.send_tensor_dict = send
    return out


# ---- stale filter with a window ----------------------------------------------

def test_corr_filter_window_accepts_out_of_order_inside_k():
    f = cx._CorrFilter(window=4)
    assert f.accept(BASE + 10)
    assert f.accept(BASE + 8) and f.accept(BASE + 7)   # 7 = 10 - 3: inside
    assert not f.accept(BASE + 6)                       # 4 behind: stale
    assert f.accept(BASE + 11)


def test_corr_filter_window_one_is_original_rule():
    f = cx._CorrFilter()
    assert f.accept(BASE + 3) and f.accept(BASE + 3)
    assert not f.accept(BASE + 2)


# ---- stage 0: several outstanding ids ------------------------------------------

def test_stage0_ids_unique_and_outstanding():
    c, _ = _k(0, 4)
    ids = [c.new_step_id() for _ in range(4)]
    assert len(set(ids)) == 4 and c._mb_outstanding == set(ids)
    assert all(i >> 32 == 0xABCD for i in ids)


def test_stage0_out_of_order_replies_matched_by_id():
    c, _ = _k(0, 3)
    a, b, d = (c.new_step_id() for _ in range(3))
    c.router_socket = FakeRouter([_reply(d, 3), _reply(a, 1), _reply(b, 2)])
    assert c.recv_step(a, timeout_s=1)["mb"] == 1      # d buffered on the way
    assert set(c._mb_done) == {d}
    assert c.recv_step(b, timeout_s=1)["mb"] == 2
    assert c.recv_step(d, timeout_s=1)["mb"] == 3       # from the buffer
    assert not c._mb_outstanding and not c._mb_done


def test_stage0_in_order_replies():
    c, _ = _k(0, 4)
    ids = [c.new_step_id() for _ in range(4)]
    c.router_socket = FakeRouter([_reply(i, n) for n, i in enumerate(ids)])
    assert [c.recv_step(i, timeout_s=1)["mb"] for i in ids] == [0, 1, 2, 3]


def test_stage0_drops_unknown_and_abandoned_replies():
    c, _ = _k(0, 2)
    a, b = c.new_step_id(), c.new_step_id()
    c.abandon_step(a)
    c.router_socket = FakeRouter([_reply(a, 1), _reply(BASE + 999, 9), _reply(b, 2)])
    assert c.recv_step(b, timeout_s=1)["mb"] == 2
    assert not c._mb_done


def test_stage0_timeout_then_per_step_retry_keeps_id():
    c, _ = _k(0, 2)
    sent = _capture_sends(c)
    a, b = c.new_step_id(), c.new_step_id()
    c.send_step({"x": 1}, a)
    c.send_step({"x": 2}, b)
    c.router_socket = FakeRouter([_reply(b, 2)])
    assert c.recv_step(a, timeout_s=0.05) is None       # a lost, b buffered
    c.send_step({"x": 1}, a)                            # retry only a
    c.router_socket.msgs.append(_reply(a, 1))
    assert c.recv_step(a, timeout_s=1)["mb"] == 1
    assert c.recv_step(b, timeout_s=1)["mb"] == 2
    assert [s[0] for s in sent] == [a, b, a]
    assert getattr(c._tls, "corr", None) is None        # override cleared


def test_stage0_concurrent_waiters_get_their_own_reply():
    c, _ = _k(0, 4)
    ids = [c.new_step_id() for _ in range(4)]
    c.router_socket = FakeRouter([_reply(i, n) for n, i in reversed(list(enumerate(ids)))])
    got = {}

    def wait(i):
        got[i] = c.recv_step(i, timeout_s=2)["mb"]
    ts = [threading.Thread(target=wait, args=(i,)) for i in ids]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert got == {i: n for n, i in enumerate(ids)}


def test_k1_step_api_is_the_legacy_path():
    c, _ = _comm(0, msgs=[_reply(BASE + 1, 7)])
    c._corr_base, c._corr_seq = BASE, 0
    assert c.inflight == 1
    i = c.new_step_id()
    assert c._pending_corr == i and c._corr_for_send() == i
    assert c.recv_step(i)["mb"] == 7 and c._pending_corr is None


def test_k1_legacy_stage0_still_drops_other_ids():
    c, _ = _comm(0, msgs=[_reply(BASE + 5, 5), _reply(BASE + 6, 6)])
    c._pending_corr = BASE + 6
    assert c.recv_tensor_dict()["mb"] == 6


def test_k_gt1_stage0_without_step_api_behaves_like_k1():
    # Stage 0 not using the per-step API (unpatched runner): legacy matching.
    c, _ = _k(0, 4, msgs=[_reply(BASE + 5, 5), _reply(BASE + 6, 6)])
    c._pending_corr = BASE + 6
    assert c.recv_tensor_dict()["mb"] == 6


# ---- later stages: per-step context, dedupe ring ---------------------------------

def test_prefetch_does_not_clobber_reply_context():
    c, _ = _k(3, 4, msgs=[_reply(BASE + 1, 1, b"tcp://s0a:1"),
                          _reply(BASE + 2, 2, b"tcp://s0b:1")])
    sent = _capture_sends(c)
    td1, ctx1 = c.recv_forward()
    td2, ctx2 = c.recv_forward()          # prefetched before step 1 is answered
    c.send_reply({"o": 1}, ctx1)
    c.send_reply({"o": 2}, ctx2)
    assert [(s[0], s[1]) for s in sent] == [(BASE + 1, "tcp://s0a:1"),
                                             (BASE + 2, "tcp://s0b:1")]


def test_done_ring_resends_any_of_last_k_and_pins_regions():
    c, sent = _k(1, 2)
    toks = [c._gdr.alloc(64, 0.0, 10.0, True)[1] for _ in range(3)]
    for n, t in enumerate(toks):
        c._remember_done(BASE + n, ("sig", b"S%d" % n), 2, t)
    assert list(c._done_ring) == [BASE + 1, BASE + 2]
    assert not c._gdr.leases[toks[0]].pinned
    assert c._gdr.leases[toks[1]].pinned and c._gdr.leases[toks[2]].pinned
    c.router_socket = FakeRouter([_reply(BASE + 1, 0), _reply(BASE + 3, 3)])
    td, ctx = c.recv_forward()
    assert sent == [b"S1"] and ctx[0] == BASE + 3 and td["mb"] == 3


def test_retry_of_unanswered_step_inside_window_is_processed():
    c, _ = _k(1, 4, msgs=[_reply(BASE + 4, 4), _reply(BASE + 2, 2), _reply(BASE + 0, 0),
                          _reply(BASE + 5, 5)])
    assert c.recv_forward()[1][0] == BASE + 4
    assert c.recv_forward()[1][0] == BASE + 2      # 2 behind: retry, accepted
    assert c.recv_forward()[1][0] == BASE + 5      # 0 is 4 behind: stale


def test_k1_downstream_keeps_single_last_done():
    c, sent = _comm(1, msgs=[_sig(9), _sig(10)])
    c._last_done = (9, ("sig", b"ORIG"), 2)
    td, ctx = c.recv_forward()
    assert sent == [b"ORIG"] and ctx[0] == 10 and c._corr_in == 10


# ---- gpu_worker patch ---------------------------------------------------------

@pytest.fixture
def fork_copy(tmp_path):
    if not os.path.exists(os.path.join(WORKER_DIR, "gpu_worker.py")):
        pytest.skip("fork source not available")
    for f in ("gpu_model_runner.py", "gpu_worker.py"):
        shutil.copy(os.path.join(WORKER_DIR, f), tmp_path / f)
    return tmp_path


def test_apply_all_patches_worker_idempotently(fork_copy):
    runner = str(fork_copy / "gpu_model_runner.py")
    snaps = []
    for _ in range(2):
        p = subprocess.run([sys.executable, os.path.join(PATCHES, "apply_all.py"), runner],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        snaps.append(((fork_copy / "gpu_model_runner.py").read_text(),
                      (fork_copy / "gpu_worker.py").read_text()))
    assert snaps[0] == snaps[1]
    assert "inflight" in p.stdout and "already patched" in p.stdout.splitlines()[-1]
    worker = snaps[1][1]
    ast.parse(worker)
    ast.parse(snaps[1][0])
    assert "def _connex_pipelined_loop(self, comm, stage_idx: int)" in worker
    # K=1: the original serial loop is still there right after the gate
    gate = worker.index("return self._connex_pipelined_loop(_cx_comm, stage_idx)")
    assert worker.index("tensor_dict = pp_group.recv_tensor_dict()", gate) > gate


def test_worker_patch_skips_runner_only_copy(tmp_path):
    if not os.path.exists(os.path.join(WORKER_DIR, "gpu_model_runner.py")):
        pytest.skip("fork source not available")
    shutil.copy(os.path.join(WORKER_DIR, "gpu_model_runner.py"), tmp_path / "gpu_model_runner.py")
    p = subprocess.run([sys.executable, os.path.join(PATCHES, "patch_external_pp_inflight.py"),
                        str(tmp_path / "gpu_model_runner.py")], capture_output=True, text=True)
    assert p.returncode == 0 and "skipped" in p.stdout


def test_pipelined_loop_overlaps_and_preserves_ctx(fork_copy):
    """Run the inserted method against a fake comm/runner: replies carry the
    ctx of their own step and order is preserved."""
    subprocess.run([sys.executable, os.path.join(PATCHES, "patch_external_pp_inflight.py"),
                    str(fork_copy / "gpu_worker.py")], check=True, capture_output=True)
    tree = ast.parse((fork_copy / "gpu_worker.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_connex_pipelined_loop")
    fn.decorator_list = []
    import logging
    import types as _t

    class Stop(BaseException):
        pass

    fake_torch = _t.SimpleNamespace(cuda=_t.SimpleNamespace(is_available=lambda: False))
    ns = {"torch": fake_torch, "logger": logging.getLogger("t")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "w", "exec"), ns)

    msgs = [({"i": i}, (BASE + i, f"o{i}")) for i in range(6)]
    replies, done = [], threading.Event()

    class Comm:
        inflight = 3

        def recv_forward(self):
            if msgs:
                return msgs.pop(0)
            threading.Event().wait()  # park the daemon prefetch thread

        def send_reply(self, out, ctx):
            replies.append((out["i"], ctx))
            if len(replies) == 6:
                done.set()
                raise Stop

    runner = _t.SimpleNamespace(execute_external_pipeline_step=lambda td: td)
    w = _t.SimpleNamespace(model_runner=runner)
    t = threading.Thread(target=lambda: _swallow(Stop, ns["_connex_pipelined_loop"], w, Comm(), 1),
                         daemon=True)
    t.start()
    assert done.wait(5)
    assert replies == [(i, (BASE + i, f"o{i}")) for i in range(6)]


def _swallow(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        pass
