"""`editing.transitions`: the reason -> kind table, the evidence downgrade
and the frequency caps. Pure; no ffmpeg."""

from __future__ import annotations

import pytest

from ai_studio.core.enums import Severity, TransitionKind, TransitionReason
from ai_studio.core.errors import UnknownKeyError
from ai_studio.editing import transitions as tr


def test_the_table_maps_meaning_to_effect_and_default_is_a_hard_cut() -> None:
    assert tr.choose(TransitionReason.DEFAULT).kind is TransitionKind.HARD_CUT
    t = tr.choose(TransitionReason.TIME_PASSING)
    assert t.kind is TransitionKind.DISSOLVE and t.overlap_s == tr.DISSOLVE_S == t.audio_fade_s
    assert t.downgraded_from is None


def test_a_hard_cut_dips_the_audio_but_overlaps_nothing() -> None:
    t = tr.choose(TransitionReason.DEFAULT)
    assert t.overlap_s == 0.0 and t.audio_fade_s == tr.AUDIO_DIP_S


def test_without_evidence_wipe_whip_and_zoom_downgrade_to_a_hard_cut() -> None:
    for reason, kind in ((TransitionReason.TOPIC_CHANGE, TransitionKind.WIPE),
                         (TransitionReason.DRILL_DOWN, TransitionKind.ZOOM_PUNCH)):
        t = tr.choose(reason)
        assert t.kind is TransitionKind.HARD_CUT and t.downgraded_from is kind and t.reason is reason


def test_with_evidence_the_effect_survives() -> None:
    t = tr.choose(TransitionReason.TOPIC_CHANGE, tr.Evidence(occlusion=True))
    assert t.kind is TransitionKind.WIPE and t.overlap_s == tr.WIPE_S


def test_an_unknown_reason_raises() -> None:
    with pytest.raises(UnknownKeyError, match="transition reason"):
        tr.choose("crossfade")  # type: ignore[arg-type]


def test_the_wipe_cap_turns_the_fourth_wipe_into_a_hard_cut() -> None:
    reasons = [TransitionReason.TOPIC_CHANGE] * 5
    evidence = [tr.Evidence(occlusion=True)] * 5
    planned = tr.plan(reasons, evidence)
    assert [t.kind for t in planned] == [TransitionKind.WIPE] * 3 + [TransitionKind.HARD_CUT] * 2
    assert planned[3].downgraded_from is TransitionKind.WIPE


def test_check_flags_the_ratio_and_the_caps() -> None:
    many = [tr.choose(TransitionReason.TIME_PASSING)] * 3 + [tr.choose(TransitionReason.DEFAULT)]
    findings = tr.check(many)
    assert [f.rule_id for f in findings] == ["T-RATIO"] and findings[0].severity is Severity.WARN
    one = [tr.choose(TransitionReason.TIME_PASSING)] + [tr.choose(TransitionReason.DEFAULT)] * 8
    assert tr.check(one) == []  # one motivated transition is always allowed
    wipe = tr.choose(TransitionReason.TOPIC_CHANGE, tr.Evidence(occlusion=True))
    over = tr.check([wipe] * 4 + [tr.choose(TransitionReason.DEFAULT)] * 40)
    assert any(f.rule_id == "T-CAP-WIPE" and f.severity is Severity.FAIL for f in over)


def test_plan_with_mismatched_evidence_raises() -> None:
    with pytest.raises(ValueError, match="evidence"):
        tr.plan([TransitionReason.DEFAULT], [])
