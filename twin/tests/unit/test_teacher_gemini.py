"""GeminiTeacher — SPEC.md §5.2/D9's v1 Teacher binding. Never touches the
network: the SDK client is injected (a fake stands in), so nothing here needs
the Phase 0 GCP project to exist. Importing this module is the explicit,
friction-ful opt-in the twin.teacher package's __init__.py is designed to
require (see test_teacher.py::test_gemini_teacher_is_not_re_exported)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from twin.config.settings import get_settings
from twin.teacher.base import TeacherCallLedger, TeacherError, TeacherRateExhausted
from twin.teacher.gemini import GeminiTeacher


class _Recipe(BaseModel):
    name: str


class _OtherModel(BaseModel):
    value: int


@dataclass
class _FakeResponse:
    parsed: object
    text: str = ""


@dataclass
class _FakeModels:
    response: _FakeResponse
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


@dataclass
class _FakeClient:
    models: _FakeModels


def _teacher(tmp_path: Path, *, parsed: object, rpd: int = 10) -> tuple[GeminiTeacher, _FakeClient]:
    client = _FakeClient(models=_FakeModels(response=_FakeResponse(parsed=parsed)))
    ledger = TeacherCallLedger(tmp_path / "ledger.json")
    teacher = GeminiTeacher(model="gemini-flash-test", client=client, ledger=ledger, rpd=rpd)  # type: ignore[arg-type]
    return teacher, client


class TestGeminiTeacherGenerate:
    def test_returns_the_parsed_object_and_records_a_call(self, tmp_path: Path) -> None:
        expected = _Recipe(name="x")
        teacher, client = _teacher(tmp_path, parsed=expected)

        result = teacher.generate("prompt", response_schema=_Recipe)

        assert result == expected
        assert client.models.calls[0]["model"] == "gemini-flash-test"
        assert client.models.calls[0]["contents"] == "prompt"
        assert teacher.ledger.calls_today() == 1

    def test_sends_json_mime_type_and_the_schema(self, tmp_path: Path) -> None:
        teacher, client = _teacher(tmp_path, parsed=_Recipe(name="x"))
        teacher.generate("prompt", response_schema=_Recipe)
        config = client.models.calls[0]["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is _Recipe

    def test_refuses_when_rpd_is_exhausted(self, tmp_path: Path) -> None:
        teacher, client = _teacher(tmp_path, parsed=_Recipe(name="x"), rpd=2)
        teacher.ledger.record_call()
        teacher.ledger.record_call()

        with pytest.raises(TeacherRateExhausted, match="2/2"):
            teacher.generate("prompt", response_schema=_Recipe)
        assert client.models.calls == []  # never actually called the client

    def test_raises_if_the_returned_type_does_not_match_the_schema(self, tmp_path: Path) -> None:
        teacher, _ = _teacher(tmp_path, parsed=_OtherModel(value=1))
        with pytest.raises(TeacherError, match="did not return"):
            teacher.generate("prompt", response_schema=_Recipe)


class TestGeminiTeacherFromSettings:
    """Constructs via env vars + get_settings(refresh=True), not Settings(field=...)
    kwargs — every field here is `alias=`-only (mirrors ai_studio.config.settings),
    so the alias (the env var name) is what the constructor actually accepts."""

    def test_requires_an_api_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # A real twin/.env now exists for local development (both fields set) —
        # pydantic-settings reads env_file as a source independent of
        # os.environ, so delenv alone doesn't simulate "unset" once that file
        # is present. chdir to an empty dir so env_file=".env" resolves to
        # nothing, isolating this test from local developer/CI environment.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TWIN_GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("TWIN_GEMINI_MODEL", "gemini-flash-test")
        settings = get_settings(refresh=True)
        with pytest.raises(TeacherError, match="TWIN_GEMINI_API_KEY"):
            GeminiTeacher.from_settings(settings)

    def test_requires_a_model(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TWIN_GEMINI_API_KEY", "k")
        monkeypatch.delenv("TWIN_GEMINI_MODEL", raising=False)
        settings = get_settings(refresh=True)
        with pytest.raises(TeacherError, match="TWIN_GEMINI_MODEL"):
            GeminiTeacher.from_settings(settings)
