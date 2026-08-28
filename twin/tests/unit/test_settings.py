"""Runtime settings guardrails. INTERVIEW.md §6.3/§8 I-D, SPEC.md D32:
transcripts and audio MUST NOT enter cross-cloud storage — this is the one
setting whose *scheme* is constrained, not just its default value.

Constructs via env vars + `get_settings(refresh=True)`, not `Settings(field=
...)` kwargs — every field is `alias=`-only, so the alias (the env var name)
is what the constructor actually accepts (see test_teacher_gemini.py's own
note on this).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.config.settings import Settings, get_settings


def test_transcript_store_uri_defaults_to_a_file_scheme(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from a real local twin/.env
    monkeypatch.delenv("TWIN_TRANSCRIPT_STORE_URI", raising=False)
    settings = get_settings(refresh=True)
    assert settings.transcript_store_uri.startswith("file://")


def test_transcript_store_uri_rejects_a_non_file_scheme(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TWIN_TRANSCRIPT_STORE_URI", "r2://twin-checkpoints/transcripts/")
    with pytest.raises(ValueError, match="MUST be a file"):
        Settings()
