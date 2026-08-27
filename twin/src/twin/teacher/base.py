"""The Teacher interface. SPEC.md §5.2, D9.

`Teacher` is the swappable contract every concrete binding (e.g.
`twin.teacher.gemini.GeminiTeacher`) implements. This module has no vendor
SDK dependency at all — the "Vendor SDK coupling confined to teacher.py"
import-linter contract exists so that whichever concrete provider is chosen
lives in its own submodule of `twin.teacher`, never here and never anywhere
above `twin.teacher`.

D9: call strategy MUST be "少次、大批" — a metered/free tier's bottleneck is
typically RPD, not TPM, so callers should inject as much context as fits and
ask for many structured items back in one call, not one call per item.
`TeacherCallLedger` below exists to make that discipline's failure symptom
("配額耗盡，資料工廠停擺") loud and early rather than a mysterious 429 mid-run —
and it is provider-agnostic: nothing about it is Gemini-specific, so whichever
`Teacher` implementation is actually run against real data can reuse it under
its own quota.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class TeacherError(Exception):
    """The Teacher refused, failed, or returned something unusable."""


class TeacherRateExhausted(TeacherError):
    """D9's named failure symptom: a metered/free tier's request quota is used up."""


class Teacher(Protocol):
    """SPEC.md §5.2: "Teacher MUST 透過 teacher.py 介面存取，實作可替換." One
    method, deliberately generic — Phase 2's S1 question generation, Phase 3's
    interview post-processing, and Phase 9's trajectory synthesis all differ
    only in `prompt` and `response_schema`, not in how they call the Teacher.
    """

    def generate(self, prompt: str, *, response_schema: type[T]) -> T: ...


class TeacherCallLedger:
    """Durable, day-scoped count of Teacher calls — mirrors the shape of
    `ai_studio.runtime.budget.SpendLedger` (PLAN.md §3.3), sized to a request
    count instead of a dollar amount."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _day_key(self, when: datetime | None = None) -> str:
        return (when or datetime.now(UTC)).date().isoformat()

    def _read(self) -> dict[str, Any]:
        fresh: dict[str, Any] = {"day": self._day_key(), "calls": 0}
        if not self.path.is_file():
            return fresh
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TeacherError(
                f"{self.path} is corrupted (invalid JSON) — refusing to silently "
                f"reset the daily Teacher call ledger, which could re-grant an "
                f"already-exhausted quota: {exc}"
            ) from exc
        if not isinstance(data, dict) or "day" not in data:
            raise TeacherError(
                f"{self.path} is not a recognisable ledger (missing 'day' key) — "
                f"refusing to silently reset the daily Teacher call count."
            )
        if data["day"] != fresh["day"]:
            return fresh
        data.setdefault("calls", 0)
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def record_call(self, when: datetime | None = None) -> None:
        data = self._read()
        data["day"] = self._day_key(when)
        data["calls"] = int(data["calls"]) + 1
        self._write(data)

    def calls_today(self, when: datetime | None = None) -> int:
        data = self._read()
        if data["day"] != self._day_key(when):
            return 0
        return int(data["calls"])
