"""The Teacher interface itself — provider-agnostic. SPEC.md §5.2, D9.
Gemini-specific tests live in test_teacher_gemini.py; nothing here touches
any vendor SDK."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin.teacher import Teacher, TeacherCallLedger, TeacherError, TeacherRateExhausted


def test_gemini_teacher_is_not_re_exported() -> None:
    """The isolation is a property of the import graph, not a comment: using
    GeminiTeacher MUST require spelling out `twin.teacher.gemini` explicitly,
    so SPEC.md §5.2's decided v1 binding never looks like the only option a
    caller can reach for `twin.teacher` itself."""
    import twin.teacher

    assert "GeminiTeacher" not in dir(twin.teacher)
    assert set(twin.teacher.__all__) == {"Teacher", "TeacherCallLedger", "TeacherError", "TeacherRateExhausted"}


def test_teacher_and_errors_are_importable() -> None:
    # Protocol/exception classes exist and are importable from the package root.
    assert Teacher is not None
    assert issubclass(TeacherRateExhausted, TeacherError)


class TestTeacherCallLedger:
    def test_starts_at_zero(self, tmp_path: Path) -> None:
        assert TeacherCallLedger(tmp_path / "ledger.json").calls_today() == 0

    def test_records_and_counts(self, tmp_path: Path) -> None:
        ledger = TeacherCallLedger(tmp_path / "ledger.json")
        ledger.record_call()
        ledger.record_call()
        assert ledger.calls_today() == 2

    def test_resets_on_a_new_day(self, tmp_path: Path) -> None:
        ledger = TeacherCallLedger(tmp_path / "ledger.json")
        yesterday = datetime(2026, 8, 26, tzinfo=UTC)
        today = datetime(2026, 8, 27, tzinfo=UTC)
        ledger.record_call(yesterday)
        ledger.record_call(yesterday)
        assert ledger.calls_today(today) == 0

    def test_rejects_a_corrupted_ledger_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(TeacherError, match="corrupted"):
            TeacherCallLedger(path).calls_today()
