"""Stage startup timeline (issue #2): every phase is logged with its time."""
import time

from connex.adapters.vllm.pp import stage


def test_proc_age_is_positive():
    a = stage._proc_age()
    assert a is None or a >= 0.0


def test_phases_line_sums_marks():
    ph = stage._Phases()
    time.sleep(0.05)
    ph.mark("a")
    ph.mark("b")
    line = ph.line()
    assert "process start -> main" in line and " a 0." in line and "total" in line
    assert ph.total() >= 0.05
