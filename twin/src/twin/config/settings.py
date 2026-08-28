"""Runtime settings. Mirrors ai_studio.config.settings's shape (PLAN.md §3.3):
`SecretStr` for anything sensitive, environment/`.env` only, a process-wide
singleton with `get_settings(refresh=True)` for tests.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
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
    trajectory_store_uri: str = Field(
        default="file://./data/trajectories.jsonl",
        alias="TWIN_TRAJECTORY_STORE_URI",
        description="fsspec URI read by twin.train.data.load_training_examples "
        "(via twin.ingest.store). Same reasoning as fragment_store_uri.",
    )
    checkpoint_store_uri: str | None = Field(
        default=None,
        alias="TWIN_CHECKPOINT_STORE_URI",
        description="fsspec URI, the root twin.train.checkpoint syncs training "
        "checkpoints and the final AdapterManifest under (SPEC.md §7.2/§7.4). No "
        "default on purpose, same reasoning as gemini_model: guessing an R2 "
        "bucket path risks silently writing checkpoints somewhere wrong once the "
        "real bucket exists. MUST be set before running train.py for real.",
    )
    adapter_encryption_key: SecretStr | None = Field(
        default=None,
        alias="TWIN_ADAPTER_ENCRYPTION_KEY",
        description="fernet.Fernet key (see core.encryption.generate_key / "
        "examples/generate_adapter_encryption_key.py). No default — SPEC.md §8: "
        "'Adapter 為個資...MUST 加密儲存'. A generated-and-shipped default key "
        "would defeat the point entirely, so this MUST be set before "
        "train.py writes or reads any adapter weights for real.",
    )
    transcript_store_uri: str = Field(
        default="file://./transcripts/",
        alias="TWIN_TRANSCRIPT_STORE_URI",
        description="INTERVIEW.md §6.3/§8 I-D, SPEC.md D32: where raw "
        "session recordings and post-processed transcript *files* live "
        "(retained until §7's quality check passes) — MUST NOT enter "
        "cross-cloud storage, since no consent/anonymization flow is "
        "implemented. This field alone does NOT enforce that for `Fragment` "
        "records: interview-transcript blocks become `SourceClass.SELF_REPORT` "
        "`Fragment.content` verbatim (SPEC.md D26), and those flow through "
        "the general `fragment_store_uri`/`write_fragments_jsonl` path — the "
        "actual Fragment-level guardrail is `ingest.store.write_fragments_"
        "jsonl` refusing any SELF_REPORT fragment on a non-file:// URI, not "
        "this setting. Unlike fragment_store_uri/checkpoint_store_uri (any "
        "fsspec URI), this field's *scheme* itself is constrained, not just "
        "its default value — see the field_validator below.",
    )

    @field_validator("transcript_store_uri")
    @classmethod
    def _transcript_store_must_stay_local(cls, value: str) -> str:
        if not value.startswith("file://"):
            raise ValueError(
                f"TWIN_TRANSCRIPT_STORE_URI MUST be a file:// URI (INTERVIEW.md "
                f"§6.3/§8 I-D: transcripts and audio MUST NOT enter cross-cloud "
                f"storage) — got {value!r}"
            )
        return value


_settings: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Process-wide settings singleton. `refresh=True` re-reads for tests."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
