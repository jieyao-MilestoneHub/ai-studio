"""`ai-studio understand`: the offline smoke test for the understanding path,
mirroring `ai-studio generate --provider stub` for the generation path -- no
GPU, no RunPod account, no money."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ai_studio.cli.main import app

runner = CliRunner()


def test_understand_stub_roundtrip_prints_a_description(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 42)

    result = runner.invoke(app, ["understand", str(photo), "--kind", "image"])

    assert result.exit_code == 0, result.output
    assert "photo.jpg" in result.output
    assert "42 bytes" in result.output


def test_understand_rejects_an_unknown_kind(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x")

    result = runner.invoke(app, ["understand", str(photo), "--kind", "smell"])

    assert result.exit_code != 0


def test_understand_rejects_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["understand", str(tmp_path / "nope.jpg"), "--kind", "image"])
    assert result.exit_code != 0
