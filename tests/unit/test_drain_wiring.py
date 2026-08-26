"""The queue must actually get drained, and the result must be reachable.

`pipeline/drain.py` was fully written and fully tested and had no caller: no CLI
command, and the VPS timers only ran `session open`, `reap` and `close`. The
window would open on a paid GPU, jobs would sit at `parsed`, and the pod would
terminate two hours later having rendered nothing. That failure is silent, it
recurs daily, and it costs $45-60/month for no videos -- so the wiring is worth
holding in place with tests rather than trusting to memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_studio.cli.main import app
from ai_studio.config.settings import get_settings

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_session_drain_is_a_real_command() -> None:
    """Without this the window is a bill with no product."""
    result = runner.invoke(app, ["session", "drain", "--help"])
    assert result.exit_code == 0, result.output


def test_drain_exits_cleanly_when_no_window_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs on a five-minute timer all day, so 'no window' is the normal
    case and must be success, not failure -- a unit that fails 280 times a day
    trains everyone to ignore it."""
    from ai_studio.runtime import session as sess

    monkeypatch.setattr(sess, "load_state", lambda: None)
    result = runner.invoke(app, ["session", "drain"])
    assert result.exit_code == 0, result.output
    assert "no window" in result.output


def test_the_api_serves_the_directory_drain_writes_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AI_STUDIO_FILES_DIR has to move both ends together.

    drain writes `<files_dir>/<token>.mp4` and the status page links
    `/files/<token>.mp4`. When the API ignored the setting, the clip rendered
    perfectly and then 404'd at delivery -- the most expensive place to fail.
    """
    from ai_studio.api.main import create_app

    target = tmp_path / "elsewhere"
    monkeypatch.setenv("AI_STUDIO_FILES_DIR", str(target))
    get_settings(refresh=True)
    try:
        app_ = create_app(queue=None, handler=None)
        assert Path(app_.state.files_dir) == target
    finally:
        monkeypatch.delenv("AI_STUDIO_FILES_DIR", raising=False)
        get_settings(refresh=True)


def test_session_drain_constructs_both_the_video_and_image_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`drain.py` was rewritten to dispatch by media_kind, but a queue that only
    ever gets a `comfyui` provider silently drops every image job forever — the
    same class of "wired but not actually wired" bug this file exists to catch."""
    import shutil
    from datetime import datetime, timedelta, timezone

    from ai_studio.runtime import session as sess

    # Isolated cwd: session_drain's default JobQueue() and its relative
    # workflows/ lookup both resolve off the cwd, and this must not touch the
    # real repo's runs/queue.sqlite3.
    (tmp_path / "workflows").mkdir()
    for name in ("h3_fl2va_turbo.json", "h3_fl2va_turbo_fp8.json", "flux_dev.json"):
        shutil.copy(REPO / "workflows" / name, tmp_path / "workflows" / name)
    monkeypatch.chdir(tmp_path)

    requested: list[str] = []

    class _Stub:
        def capabilities(self):
            class Caps:
                native_width = 8
                native_height = 8
                native_fps = 1

            return Caps()

    def _fake_get_provider(name: str, **kwargs):
        requested.append(name)
        return _Stub()

    now = datetime.now(timezone.utc)
    fake_session = sess.Session(
        pod_id="pod-1",
        gpu="NVIDIA GeForce RTX 4090",
        datacenter="EUR-IS-2",
        cloud="COMMUNITY",
        cost_per_hr=0.354,
        opened_at=(now - timedelta(minutes=5)).isoformat(),
        window_end=(now + timedelta(hours=2)).isoformat(),
        tier_label="4090/COMMUNITY",
        vram_gb=24,
        low_vram=True,
    )
    monkeypatch.setattr(sess, "load_state", lambda: fake_session)
    monkeypatch.setattr("ai_studio.cli.main.get_provider", _fake_get_provider)

    result = runner.invoke(app, ["session", "drain"])
    assert result.exit_code == 0, result.output
    assert set(requested) == {"comfyui", "flux"}, (
        "session drain must build a provider for every media_kind the queue can hold"
    )


def test_the_vps_runbook_documents_manual_drain_recovery() -> None:
    """`session drain` is off every timer now -- the worker renders each parsed
    job itself, on demand, inside business hours. `test_deploy_scripts.py`
    already holds the worker service's own installation and enablement in
    place; what is left unique to this command is the manual recovery path
    printed for an operator when the worker wedges holding an open pod.

    Checked as text because the alternative is a runbook that quietly points
    at a command that stopped existing.
    """
    script = (REPO / "deploy" / "vps_setup.sh").read_text(encoding="utf-8")
    assert "ai-studio session drain" in script, (
        "the runbook no longer tells an operator how to recover a wedged worker"
    )
