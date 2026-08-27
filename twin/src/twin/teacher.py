"""Teacher provider interface. SPEC.md §5.2, D9, §7.1/D12.

One of exactly two files allowed to couple to a vendor SDK (the other is
`launch/*.sh`, outside import-linter's reach) — enforced by the "Vendor SDK
coupling confined to teacher.py" contract in pyproject.toml. `train.py` and
everything above it MUST NOT know this binds to Gemini; they only see `Teacher`.

D9: call strategy MUST be "少次、大批" — the free tier's bottleneck is RPD, not
TPM, so callers should inject as much context as fits (up to 1M tokens) and ask
for many structured items back in one call, not one call per item. `TeacherCallLedger`
below exists to make the D9 failure symptom ("RPD 耗盡，資料工廠停擺") loud and
early rather than a mysterious 429 mid-run.

Unverified against a live account: SPEC.md/D8 requires the Teacher's GCP project
to never have billing enabled, and that project does not exist yet (twin/PLAN.md
Phase 0's remaining manual step) — so `GeminiTeacher` below is correct against
the installed `google-genai==2.20.0` SDK's actual signatures (checked by
introspection, not guessed), but has never made a real call.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from twin.config.settings import Settings

T = TypeVar("T", bound=BaseModel)


class TeacherError(Exception):
    """The Teacher refused, failed, or returned something unusable."""


class TeacherRateExhausted(TeacherError):
    """D9's named failure symptom: the free tier's daily request quota is used up."""


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
                f"already-exhausted RPD quota: {exc}"
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


class GeminiTeacher:
    """SPEC.md §5.2 v1 binding: Gemini Flash free tier (1,500 RPD / 15 RPM / 1M
    TPM). `client` is injected so tests never need real credentials or network."""

    def __init__(
        self,
        *,
        model: str,
        client: genai.Client,
        ledger: TeacherCallLedger,
        rpd: int = 1500,
    ) -> None:
        self.model = model
        self._client = client
        self.ledger = ledger
        self.rpd = rpd

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiTeacher:
        if settings.gemini_api_key is None:
            raise TeacherError(
                "TWIN_GEMINI_API_KEY is not set — the Phase 0 GCP project must "
                "exist and its key configured before GeminiTeacher can be used "
                "(SPEC.md §5.2, D8: that project MUST never have billing enabled)."
            )
        if settings.gemini_model is None:
            raise TeacherError(
                "TWIN_GEMINI_MODEL is not set. SPEC.md §5.2 only commits to the "
                "'Gemini Flash 系列' family, not one pinned model ID — pick a "
                "current free-tier-eligible Flash model explicitly rather than "
                "relying on a guessed default (see config.settings.Settings.gemini_model)."
            )
        client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        ledger = TeacherCallLedger(settings.teacher_ledger_path)
        return cls(model=settings.gemini_model, client=client, ledger=ledger, rpd=settings.gemini_rpd)

    def generate(self, prompt: str, *, response_schema: type[T]) -> T:
        self._refuse_if_exhausted()
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        self.ledger.record_call()
        parsed = response.parsed
        if not isinstance(parsed, response_schema):
            raise TeacherError(
                f"Gemini did not return a parsed {response_schema.__name__}: "
                f"got {type(parsed).__name__}. Raw text: {str(response.text)[:500]}"
            )
        return parsed

    def _refuse_if_exhausted(self) -> None:
        used = self.ledger.calls_today()
        if used >= self.rpd:
            raise TeacherRateExhausted(
                f"{used}/{self.rpd} Gemini Flash free-tier calls already used today "
                f"(D9) — refusing to call again rather than risk falling through to "
                f"billed usage (SPEC.md §5.2, D8: the Teacher project MUST never "
                f"have billing enabled). Wait for the daily quota to reset, or batch "
                f"more per call (D9: 少次、大批)."
            )
