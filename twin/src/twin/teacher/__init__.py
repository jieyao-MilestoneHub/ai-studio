"""Teacher provider interface. SPEC.md §5.2, D9, §7.1/D12.

Deliberately re-exports only the provider-agnostic contract — `Teacher`,
`TeacherError`, `TeacherRateExhausted`, `TeacherCallLedger`. `GeminiTeacher`
(SPEC.md §5.2's v1 binding) is NOT re-exported here: using it requires
`from twin.teacher.gemini import GeminiTeacher` explicitly, so that binding
never looks like the only option a caller can reach for.
"""

from __future__ import annotations

from twin.teacher.base import Teacher, TeacherCallLedger, TeacherError, TeacherRateExhausted

__all__ = ["Teacher", "TeacherCallLedger", "TeacherError", "TeacherRateExhausted"]
