"""`editing.captions`: line breaking, read speed, emoji, colour keys."""

from __future__ import annotations

import pytest

from ai_studio.core.enums import Severity
from ai_studio.core.errors import UnknownKeyError
from ai_studio.editing import captions as cap


def test_short_text_is_one_line_and_long_text_breaks_after_punctuation() -> None:
    assert cap.break_lines("沒事,明天照常開。") == ["沒事,明天照常開。"]
    assert cap.break_lines("我不走,這攤是我爸留下的,明天照常開。") == ["我不走,這攤是我爸留下的,", "明天照常開。"]


def test_text_with_no_punctuation_breaks_hard_at_the_limit() -> None:
    text = "一" * 20
    assert cap.break_lines(text) == ["一" * 15, "一" * 5]


def test_three_lines_worth_raises() -> None:
    with pytest.raises(ValueError, match="more than 2 lines"):
        cap.break_lines("一" * 31)
    with pytest.raises(ValueError, match="empty"):
        cap.break_lines("   ")


def test_emoji_are_stripped_and_cjk_kept() -> None:
    assert cap.strip_emoji("明天見 👋🏻✨ 好嗎") == "明天見  好嗎"
    assert cap.strip_emoji("沒事。") == "沒事。"


def test_read_speed_thresholds() -> None:
    fast = cap.check([("一二三四五六七八九十一二三四五", 2.0, "w")])  # 7.5 chars/s
    assert [f.rule_id for f in fast] == ["C-READ-FAIL"]
    warn = cap.check([("一二三四五六七八九十一二", 2.0, "w")])  # 6 chars/s
    assert [(f.rule_id, f.severity) for f in warn] == [("C-READ-WARN", Severity.WARN)]
    assert cap.check([("沒事。", 3.0, "w")]) == []


def test_emoji_colour_and_dwell_findings() -> None:
    findings = cap.check([("好 🎉", 0.3, "x")])
    ids = {f.rule_id for f in findings}
    assert {"C-EMOJI", "C-COLOR", "C-DWELL"} <= ids
    none = cap.check([("好", 0.0, "w")])
    assert [f.rule_id for f in none] == ["C-DWELL"] and none[0].severity is Severity.FAIL


def test_unknown_colour_key_raises_rather_than_rendering_white() -> None:
    assert cap.resolve_color("w") == "&H00FFFFFF"
    with pytest.raises(UnknownKeyError, match="caption colour key"):
        cap.resolve_color("gold")
