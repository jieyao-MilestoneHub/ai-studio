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

from videogen.cli.main import app
from videogen.config.settings import get_settings

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
    from videogen.runtime import session as sess

    monkeypatch.setattr(sess, "load_state", lambda: None)
    result = runner.invoke(app, ["session", "drain"])
    assert result.exit_code == 0, result.output
    assert "no window" in result.output


def test_the_api_serves_the_directory_drain_writes_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """VIDEOGEN_FILES_DIR has to move both ends together.

    drain writes `<files_dir>/<token>.mp4` and the status page links
    `/files/<token>.mp4`. When the API ignored the setting, the clip rendered
    perfectly and then 404'd at delivery -- the most expensive place to fail.
    """
    from videogen.api.main import create_app

    target = tmp_path / "elsewhere"
    monkeypatch.setenv("VIDEOGEN_FILES_DIR", str(target))
    get_settings(refresh=True)
    try:
        app_ = create_app(queue=None, handler=None)
        assert Path(app_.state.files_dir) == target
    finally:
        monkeypatch.delenv("VIDEOGEN_FILES_DIR", raising=False)
        get_settings(refresh=True)


def test_the_vps_installs_a_drain_timer() -> None:
    """Provisioning must schedule the step that makes videos.

    Checked as text because the alternative is discovering it on a live pod at
    11:00, having paid for the GPU.
    """
    script = (REPO / "deploy" / "vps_setup.sh").read_text(encoding="utf-8")
    assert "session drain" in script, "the VPS never renders anything"
    assert "for phase in open drain reap close" in script, (
        "the drain unit is defined but never enabled"
    )
