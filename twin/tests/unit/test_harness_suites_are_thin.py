"""harness.suites.{s1,s2,s3,s4} are deliberately thin. Pins the deferral
itself and its citation, not any real behavior — each needs real
corpora/runtime that don't exist until their own PLAN.md phase."""

from __future__ import annotations

import pytest

from twin.harness.suites import s2, s3, s4

# s1 is no longer pinned here — twin/PLAN.md Phase 2 landed real behavior;
# see tests/unit/test_harness_suites_s1.py.


def test_s2_run_held_out_tasks_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 11"):
        s2.run_held_out_tasks(task_ids=[], decider=object())


def test_s3_build_time_series_feed_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 11"):
        s3.build_time_series_feed(principal_id="p1", window_start="2026-01-01", window_end="2026-02-01")


def test_s4_build_blind_test_set_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 12"):
        s4.build_blind_test_set(principal_samples=[], twin_samples=[])
