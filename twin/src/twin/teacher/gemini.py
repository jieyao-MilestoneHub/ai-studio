"""SPEC.md §5.2/D9's v1 Teacher binding: Gemini Flash's free tier.

This is *one example implementation* of the `Teacher` interface (`twin.teacher.
base.Teacher`), decided by SPEC.md for v1 — not an architectural commitment
that nothing above `twin.teacher` should assume otherwise. Nothing outside
`twin.teacher` needs to know this module exists: `twin/teacher/__init__.py`
deliberately does not re-export `GeminiTeacher`, so using it requires spelling
out `from twin.teacher.gemini import GeminiTeacher` explicitly. If Gemini
turns out not to be the Teacher actually run against real data, write and
inject a different `Teacher` implementation — nothing above this module
changes.

One of exactly two files allowed to couple to a vendor SDK (the other is
`launch/*.sh`, outside import-linter's reach) — enforced by the "Vendor SDK
coupling confined to teacher.py" contract in pyproject.toml.

Unverified against a live account: SPEC.md/D8 requires the Teacher's GCP
project to never have billing enabled, and that project does not exist yet
(twin/PLAN.md Phase 0's remaining manual step) — so the calls below are
correct against the installed `google-genai==2.20.0` SDK's actual signatures
(checked by introspection, not guessed), but have never made a real call.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from twin.config.settings import Settings
from twin.teacher.base import T, TeacherCallLedger, TeacherError, TeacherRateExhausted


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
        sleep: Callable[[float], None] = time.sleep,
        max_rate_limit_retries: int = 3,
    ) -> None:
        self.model = model
        self._client = client
        self.ledger = ledger
        self.rpd = rpd
        self._sleep = sleep
        self._max_rate_limit_retries = max_rate_limit_retries

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
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                break
            except genai_errors.ClientError as exc:
                # The free tier is 15 RPM as well as 1,500 RPD (SPEC.md §5.2). A
                # burst of per-item calls (2026-08-30: 30 interview paraphrase
                # calls) trips the per-minute limit first; that is a pause, not
                # D9's "quota exhausted" — the RPD ledger is what guards that.
                if exc.code != 429 or attempt >= self._max_rate_limit_retries:
                    raise
                attempt += 1
                self._sleep(_retry_delay_seconds(str(exc)))
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


_RETRY_IN = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


def _retry_delay_seconds(message: str, *, default: float = 20.0, cap: float = 75.0) -> float:
    match = _RETRY_IN.search(message)
    delay = float(match.group(1)) + 1.0 if match else default
    return min(delay, cap)
