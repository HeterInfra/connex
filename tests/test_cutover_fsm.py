# SPDX-License-Identifier: Apache-2.0
"""State machine tests for CutoverFSM."""

import pytest

from connex.cutover_fsm import CutoverFSM, CutoverState


def _full_cutover(fsm, sid="s0"):
    fsm.start_cutover(sid, "10.0.0.1:5000", "10.0.0.2:5000", old_epoch=1, new_epoch=2)
    assert fsm.prepare_complete(sid, new_connection=object())
    assert fsm.start_traffic_switch(sid, last_old_seqno=10)
    assert fsm.cutover_complete(sid)
    assert not fsm.update_drain_progress(sid, acked_seqno=9)
    assert fsm.update_drain_progress(sid, acked_seqno=10)
    return fsm.finalize(sid)


def test_happy_path_visits_all_states():
    seen = []
    fsm = CutoverFSM(stage_idx=0, on_state_change=lambda st, ctx: seen.append(ctx.state))
    ctx = _full_cutover(fsm)
    assert ctx.state == CutoverState.IDLE
    assert CutoverState.DRAINING in seen and CutoverState.FINALIZING in seen
    assert not fsm.is_cutting_over("s0")
    assert fsm.get_stats()["successful_cutovers"] == 1


def test_duplicate_start_rejected():
    fsm = CutoverFSM(stage_idx=0)
    fsm.start_cutover("s0", "a", "b", 1, 2)
    with pytest.raises(RuntimeError):
        fsm.start_cutover("s0", "a", "c", 1, 3)


def test_out_of_order_transition_rejected():
    fsm = CutoverFSM(stage_idx=0)
    fsm.start_cutover("s0", "a", "b", 1, 2)
    assert not fsm.start_traffic_switch("s0", 5)  # still PREPARING
    assert not fsm.cutover_complete("s0")


def test_abort_clears_cutover():
    fsm = CutoverFSM(stage_idx=0)
    fsm.start_cutover("s0", "a", "b", 1, 2)
    assert fsm.abort("s0", "peer died")
    assert not fsm.is_cutting_over("s0")
    _full_cutover(fsm)  # can restart after abort
