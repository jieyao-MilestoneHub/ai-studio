"""The GPU side's pre-launch checklist, and the checklist machinery itself.

There is no stub in this project, so this checklist is what stands between the
implementation and the one affordable live run. That gives it an unusual
property: **a bug here is invisible in exactly the way it costs the most.** A
check that returns PASS when it could not actually run reads as "verified",
and the forty minutes of GPU time get spent discovering what it did not check.

So most of this file is about the checklist's own honesty rather than about
the things it checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_studio.checks import CheckResult, Status, stamp, summarise
from ai_studio.cli import preflight
from ai_studio.cli.main import app
from ai_studio.cli.preflight import run_all
from ai_studio.config import settings as settings_mod
from ai_studio.config.settings import get_settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run every check as if on a bare machine.

    Without this the results depend on whoever's `.env` is lying around, which
    would make the whole file report differently on the VM than in CI.
    """
    for name in (
        "LINE_CHANNEL_SECRET",
        "LINE_CHANNEL_ACCESS_TOKEN",
        "LINE_ALLOWED_GROUP_ID",
        "LINE_ALLOWED_USER_IDS",
        "RUNPOD_API_KEY",
        "AI_STUDIO_LLM_ENDPOINT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    # `.env` is read by pydantic-settings on construction, so point it at a
    # file that does not exist rather than trusting the repo not to have one.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_mod, "ENV_FILE", tmp_path / ".env")
    get_settings(refresh=True)
    yield
    monkeypatch.undo()
    get_settings(refresh=True)


# ------------------------------------------------------- the checklist's honesty


def test_a_check_that_cannot_run_skips_rather_than_passing() -> None:
    """The failure mode that would make this whole module worse than nothing:
    "we could not verify this" rendering as "this is verified"."""
    assert preflight.check_placement().status is Status.SKIP


def test_every_skip_says_why() -> None:
    """A skip with no reason is indistinguishable from a check nobody wrote."""
    results = run_all(run_suite=False)
    for result in results:
        if result.status is Status.SKIP:
            assert result.detail.strip(), f"check {result.number} skipped silently"


def test_green_means_all_pass_not_merely_no_failures() -> None:
    """A checklist that reports "fine" with three unknowns on it is worse
    than no checklist, because it gets believed."""
    nine_pass = [CheckResult(n, f"c{n}", Status.PASS, "") for n in range(1, 10)]
    with_a_skip = [*nine_pass[:-1], CheckResult(9, "c9", Status.SKIP, "no key")]
    with_a_fail = [*nine_pass[:-1], CheckResult(9, "c9", Status.FAIL, "broken")]

    assert summarise(nine_pass)[0] is True
    assert summarise(with_a_skip)[0] is False
    assert summarise(with_a_fail)[0] is False


def test_an_empty_result_list_is_never_green() -> None:
    """Guard against the whole runner silently doing nothing."""
    assert summarise([])[0] is False


def test_a_check_that_raises_becomes_a_failure_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broken check must not cost you the other results."""

    def boom() -> CheckResult:
        raise RuntimeError("the disk fell off")

    monkeypatch.setattr(preflight, "check_graphs", boom)
    results = run_all(run_suite=False)

    assert len(results) == 4
    broken = [r for r in results if r.status is Status.FAIL]
    assert any("the disk fell off" in r.detail for r in broken)


def test_all_checks_run_in_order() -> None:
    results = run_all(run_suite=False)

    assert [r.number for r in results] == list(range(1, 5))
    assert len({r.name for r in results}) == 4


# --------------------------------------------------------- the offline checks


def test_the_graphs_check_loads_every_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline and free, and it catches the graph error that would otherwise
    surface at submit time on a billing pod."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    result = preflight.check_graphs()

    assert result.status is Status.PASS
    assert "flux_dev.json" in result.detail


# ------------------------------------------------------------------- the CLI


def test_preflight_is_a_real_command() -> None:
    result = runner.invoke(app, ["preflight", "--help"])

    assert result.exit_code == 0, result.output
    assert "--skip-suite" in result.output


def test_the_command_exits_nonzero_when_it_is_not_green() -> None:
    """A zero here on a bare machine would say "everything is proved" when
    the placement check was skipped."""
    result = runner.invoke(app, ["preflight", "--skip-suite"])

    assert result.exit_code == 1
    assert "not green" in result.output


def test_the_command_exits_zero_when_every_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_all",
        lambda **_: [CheckResult(n, f"check {n}", Status.PASS, "ok") for n in range(1, 5)],
    )

    result = runner.invoke(app, ["preflight", "--skip-suite"])

    assert result.exit_code == 0, result.output
    assert "all green" in result.output


def test_the_command_never_opens_a_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole premise is that it costs nothing. `runpodctl` is the only
    way a pod gets created, so nothing may shell out to it."""
    from ai_studio.runtime import session as sess

    def forbidden(*args: str, **kwargs: object) -> dict[str, object]:
        raise AssertionError(f"preflight shelled out to runpodctl: {args}")

    monkeypatch.setattr(sess, "_runpodctl", forbidden)
    result = runner.invoke(app, ["preflight", "--skip-suite"])

    assert result.exit_code == 1  # not green, but nothing was created


# -------------------------------------------------------------------- stamp


def test_the_stamp_is_pasteable_ascii() -> None:
    """It goes into a run log next to figures graded by the number-honesty
    rule, and through a cp950 console on the way."""
    from datetime import datetime, timezone

    results = run_all(run_suite=False)
    text = stamp(results, when=datetime(2026, 8, 25, 3, 0, tzinfo=timezone.utc))

    assert text.isascii(), text
    assert "2026-08-25T03:00:00+00:00" in text
    assert "NOT green" in text
    for result in results:
        assert f"{result.number}. [{result.status.value}]" in text
