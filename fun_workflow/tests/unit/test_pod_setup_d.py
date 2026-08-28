"""The pod-setup extension this package ships: FaceDetailer for /短劇."""

from __future__ import annotations

from fun_workflow import paths


def test_the_extension_is_found_and_can_never_kill_the_setup() -> None:
    extras = paths.pod_setup_extras()
    assert [p.name for p in extras] == ["face_repair.sh"]
    body = extras[0].read_text(encoding="utf-8")
    assert "die " not in body and "die\n" not in body
    assert "set -e" not in body, "best effort: no step may abort the script"
    assert body.rstrip().endswith("exit 0")
    assert "ComfyUI-Impact-Pack" in body and "ComfyUI-Impact-Subpack" in body
    assert "face_yolov8m.pt" in body
    assert ".venv-cu128/bin/pip" in body, "node-pack requirements go into ComfyUI's own venv"
