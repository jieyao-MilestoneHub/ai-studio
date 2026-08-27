"""Runtime settings. Mirrors ai_studio.config.settings's shape (PLAN.md §3.3):
`SecretStr` for anything sensitive, environment/`.env` only, a process-wide
singleton with `get_settings(refresh=True)` for tests.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from the environment, and from `.env` in development.

    `.env` is gitignored (twin/.gitignore's python section covers dotfiles via
    the repo-root pattern already in place; no twin-specific secret has landed
    here yet, so there is no `.env.example` to keep in sync until one does).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ---------------------------------------------------------- principal
    principal_id: str = Field(
        default="default",
        alias="TWIN_PRINCIPAL_ID",
        description="SPEC.md §2.1: all data is isolated by principal_id. A "
        "single-operator v1 tool defaults to one principal rather than making "
        "every ingest call site pass it explicitly.",
    )

    # ---------------------------------------------------------- Teacher (SPEC.md §5.2, D8/D9, D12/§7.1)
    gemini_api_key: SecretStr | None = Field(default=None, alias="TWIN_GEMINI_API_KEY")
    gemini_model: str | None = Field(
        default=None,
        alias="TWIN_GEMINI_MODEL",
        description="No default on purpose. SPEC.md §5.2 commits only to the "
        "'Gemini Flash 系列' family, not one pinned model ID, and free-tier "
        "eligibility is model-specific — guessing a string here risks the exact "
        "D8 failure symptom (an unexpected bill) if it silently resolves to a "
        "non-free-tier model. MUST be set once the Phase 0 GCP project exists.",
    )
    gemini_rpd: int = Field(
        default=1500,
        ge=0,
        alias="TWIN_GEMINI_RPD",
        description="SPEC.md §5.2's documented free-tier ceiling (requests/day). "
        "D9's failure symptom is this being exhausted mid-ingest, not an overspend.",
    )
    teacher_ledger_path: Path = Field(
        default=Path("data/.teacher_call_ledger.json"),
        alias="TWIN_TEACHER_LEDGER_PATH",
        description="Local bookkeeping only (mirrors ai_studio.runtime.budget's "
        "own local Path ledger) — not a data artifact, so SPEC.md §7.2's URI "
        "requirement doesn't apply to it.",
    )

    # ---------------------------------------------------------- ingest / memory storage (SPEC.md §7.2)
    fragment_store_uri: str = Field(
        default="file://./data/fragments.jsonl",
        alias="TWIN_FRAGMENT_STORE_URI",
        description="fsspec URI. SPEC.md §7.2: all data-artifact paths MUST be "
        "URIs; MUST NOT be bare local paths.",
    )


_settings: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Process-wide settings singleton. `refresh=True` re-reads for tests."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
