"""`editing.rhythm`: the pacing band, the metronome floor, the slow-run rule."""

from __future__ import annotations

import pytest

from ai_studio.core.enums import Severity
from ai_studio.editing import rhythm

POLICY = rhythm.PacingPolicy(min_s=2.0, warn_s=8.0, fail_s=12.5, total_band_s=(55.0, 65.0))
TEMPLATE = [2.5, 4.79, 5.5, 4.625, 8.71, 6.0, 5.54, 4.5, 5.625, 8.0]  # 55.8 s, ten segments


def _ids(findings: list) -> list[str]:
    return [f.rule_id for f in findings]


def test_six_equal_ten_second_shots_are_a_metronome() -> None:
    findings = rhythm.check([10.125] * 6, POLICY)
    assert "R-CV" in _ids(findings)
    assert all(f.severity is Severity.FAIL for f in findings if f.rule_id == "R-CV")


def test_the_drama_template_passes_clean() -> None:
    findings = rhythm.check(TEMPLATE, POLICY)
    assert [f for f in findings if f.severity is Severity.FAIL] == []
    assert _ids(findings) == ["R-BAND-WARN"]  # the 8.71 s conflict beat is slow, by design


def test_two_consecutive_slow_segments_fail() -> None:
    findings = rhythm.check([3.0, 9.0, 9.5, 3.0, 4.0], rhythm.PacingPolicy(min_s=2.0, warn_s=8.0, fail_s=12.5))
    assert "R-CONSEC-SLOW" in _ids(findings)


def test_band_and_total_are_enforced() -> None:
    findings = rhythm.check([1.0, 13.0, 5.0, 6.0], POLICY)
    ids = _ids(findings)
    assert "R-BAND-MIN" in ids and "R-BAND-FAIL" in ids and "R-TOTAL" in ids


def test_cv_is_population_over_mean_and_zero_for_one_value() -> None:
    assert rhythm.coefficient_of_variation([10.0]) == 0.0
    assert rhythm.coefficient_of_variation([1.0, 3.0]) == pytest.approx(0.5)


def test_an_inverted_band_raises() -> None:
    with pytest.raises(ValueError, match="min <= warn <= fail"):
        rhythm.PacingPolicy(min_s=5.0, warn_s=4.0, fail_s=6.0)
    with pytest.raises(ValueError, match="positive"):
        rhythm.check([1.0, 0.0], POLICY)
