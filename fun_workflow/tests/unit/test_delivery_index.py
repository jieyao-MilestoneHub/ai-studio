"""`storage.index`: the map from a delivered file's random name back to its request."""

from __future__ import annotations

import json
from pathlib import Path

from fun_workflow.storage.index import append_delivery, index_path


def test_append_delivery_writes_one_json_line_with_hash_and_size(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir()
    clip = files / "OD5_XWwNXUU.png"
    clip.write_bytes(b"png-bytes")

    rec = append_delivery(files, token="OD5_XWwNXUU", job_id=85, kind="image", path=clip, cost_usd=0.013)
    rec2 = append_delivery(files, token="abc", job_id=86, kind="video", path=files / "missing.mp4")

    lines = [json.loads(line) for line in index_path(files).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and rec is not None and rec2 is not None
    assert lines[0]["token"] == "OD5_XWwNXUU" and lines[0]["job_id"] == 85 and lines[0]["kind"] == "image"
    assert lines[0]["bytes"] == 9 and len(lines[0]["sha256"]) == 64 and lines[0]["cost_usd"] == 0.013
    assert lines[0]["ts"].endswith("+00:00")
    assert lines[1]["bytes"] is None and lines[1]["sha256"] is None  # a missing file is recorded, not fatal


def test_an_unwritable_index_is_a_warning_not_an_error(tmp_path: Path, caplog) -> None:
    import logging

    files = tmp_path / "not-a-dir"
    files.write_text("I am a file, not a directory", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="fun_workflow.index"):
        rec = append_delivery(files, token="t", job_id=1, kind="image", path=tmp_path / "x.png")
    assert rec is None
    assert any("delivery index not written" in r.getMessage() for r in caplog.records)
