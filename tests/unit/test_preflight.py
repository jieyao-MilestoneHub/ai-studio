"""The pre-launch checklist.

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

from ai_studio.cli import preflight
from ai_studio.cli.main import app
from ai_studio.cli.preflight import CheckResult, Status, run_all, stamp, summarise
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
    get_settings(refresh=True)
    yield
    get_settings(refresh=True)


# ------------------------------------------------------- the checklist's honesty


def test_a_check_that_cannot_run_skips_rather_than_passing() -> None:
    """The failure mode that would make this whole module worse than nothing:
    "we could not verify this" rendering as "this is verified"."""
    assert preflight.check_signature_and_dedupe().status is Status.SKIP
    assert preflight.check_placement().status is Status.SKIP


def test_the_conversion_check_runs_offline_and_passes() -> None:
    """The rewriter is gpt-oss on the pod, which does not exist before a
    window; the check proves the queue -> prompt -> `_rendered` path with a
    scripted reply instead of skipping, so it is a real assertion."""
    result = preflight.check_queue_and_conversion()
    assert result.status is Status.PASS, result.detail
    assert "built_by=llm" in result.detail


def test_every_skip_says_why() -> None:
    """A skip with no reason is indistinguishable from a check nobody wrote."""
    results = run_all(run_suite=False)
    for result in results:
        if result.status is Status.SKIP:
            assert result.detail.strip(), f"check {result.number} skipped silently"


def test_green_means_all_nine_not_merely_no_failures() -> None:
    """Phase 4's own definition of done is nine passes. A checklist that
    reports "fine" with three unknowns on it is worse than no checklist,
    because it gets believed."""
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
    """One broken check must not cost you the other eight results."""

    def boom() -> CheckResult:
        raise RuntimeError("the disk fell off")

    monkeypatch.setattr(preflight, "check_graphs", boom)
    results = run_all(run_suite=False)

    assert len(results) == 9
    broken = [r for r in results if r.status is Status.FAIL]
    assert any("the disk fell off" in r.detail for r in broken)


def test_all_nine_checks_run_in_plan_order() -> None:
    results = run_all(run_suite=False)

    assert [r.number for r in results] == list(range(1, 10))
    assert len({r.name for r in results}) == 9


# --------------------------------------------------------- the offline checks


def test_the_graphs_check_loads_every_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline and free, and it catches the graph error that would otherwise
    surface at submit time on a billing pod."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    result = preflight.check_graphs()

    assert result.status is Status.PASS
    assert "flux_dev.json" in result.detail


def test_the_graphs_check_skips_rather_than_failing_outside_the_repo() -> None:
    """`workflows/` is resolved relative to the cwd. Not being in the repo is
    an operator mistake, not a broken graph, and reporting it as a broken graph
    would send someone looking in the wrong place."""
    result = preflight.check_graphs()

    assert result.status is Status.SKIP
    assert "repo root" in result.detail


def test_the_range_check_proves_206_end_to_end() -> None:
    """LINE's video message needs it and nothing about the failure says so."""
    result = preflight.check_files_range()

    assert result.status is Status.PASS
    assert "bytes 0-99/1024" in result.detail


def test_the_out_of_hours_check_is_the_one_that_guards_real_money() -> None:
    """It runs at any hour because the clock is injected — otherwise it could
    only be verified for two hours a day, which is when nobody is looking."""
    result = preflight.check_out_of_hours()

    assert result.status is Status.PASS, result.detail
    assert "no pod created" in result.detail


def test_the_out_of_hours_check_output_is_ascii() -> None:
    """The reply it inspects is Chinese and the Windows console is cp950. A
    detail string that quoted it would render as mojibake in the one place an
    operator reads it."""
    result = preflight.check_out_of_hours()

    assert result.detail.isascii(), result.detail


# ----------------------------------------------------------- the push guard


def test_the_push_check_never_sends_unless_asked() -> None:
    """The only check that messages real people and spends real quota. Opt-in
    behind a flag, not merely gated on credentials being present."""
    assert preflight.check_push(send=False).status is Status.SKIP
    assert "pass --push" in preflight.check_push(send=False).detail


def test_the_push_check_still_needs_credentials_when_asked() -> None:
    result = preflight.check_push(send=True)

    assert result.status is Status.SKIP
    assert "LINE_CHANNEL_ACCESS_TOKEN" in result.detail


def test_run_all_does_not_push_by_default() -> None:
    """The default path must be unable to send anything, whatever is in .env."""
    results = run_all(run_suite=False)
    push = next(r for r in results if r.number == 5)

    assert push.status is Status.SKIP


# ------------------------------------------------------------------- the CLI


def test_preflight_is_a_real_command() -> None:
    result = runner.invoke(app, ["preflight", "--help"])

    assert result.exit_code == 0, result.output
    assert "--push" in result.output


def test_the_command_exits_nonzero_when_it_is_not_green() -> None:
    """This exit code is the Phase 7 gate. A zero here on a bare machine would
    say "everything is proved" when six things were skipped."""
    result = runner.invoke(app, ["preflight", "--skip-suite"])

    assert result.exit_code == 1
    assert "not green" in result.output


def test_the_command_exits_zero_when_every_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_all",
        lambda **_: [CheckResult(n, f"check {n}", Status.PASS, "ok") for n in range(1, 10)],
    )

    result = runner.invoke(app, ["preflight", "--skip-suite"])

    assert result.exit_code == 0, result.output
    assert "all nine green" in result.output


def test_the_command_never_opens_a_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole premise of Phase 4 is that it costs nothing. `runpodctl` is
    the only way a pod gets created, so nothing may shell out to it."""
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
