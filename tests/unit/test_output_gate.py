"""The output gate, and the fact that it is actually connected.

The failure this whole file exists for is not loud: **a broken generation is not
an error anywhere in the stack.** A NaN latent decodes to a flat black or flat
grey image, ComfyUI saves it, `/history` reports success, `probe()` returns
sensible dimensions, and the bot pushes it to the group. So does a render whose
LoRA silently failed to load.

Two halves are tested separately, because they fail separately: the rules
themselves against fixtures, and the *wiring* — a gate nobody calls is exactly
as useful as no gate, which is the shape `test_drain_wiring.py` exists to
remember.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ai_studio.core.errors import GateFailure
from ai_studio.gates.core import GateContext, selftest
from ai_studio.gates.output_gate import output_gate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "output_gate"


def _report(name: str):
    return output_gate(GateContext(FIXTURES / name))


def _failed(name: str) -> set[str]:
    return {f.rule_id for f in _report(name).failures}


# --------------------------------------------------------------- the rules


def test_a_good_render_passes_cleanly() -> None:
    """Guard against a gate that fails everything and so proves nothing."""
    assert _failed("good") == set()


def test_a_flat_render_is_caught() -> None:
    """The one that matters. Every pixel the same brightness is what a NaN
    latent decodes to, and nothing upstream calls it an error."""
    assert "OUT-FLAT" in _failed("black")


def test_a_silent_clip_is_caught() -> None:
    """H3 generates picture and audio in one pass. A silent file does not mean
    a quiet scene, it means the audio VAE never ran."""
    assert "OUT-AUDIO" in _failed("silent")


def test_the_gate_catches_its_own_rules_on_the_bad_fixtures() -> None:
    """The shell's own contract: every gate ships a deliberately-bad fixture,
    so a rule that silently stops working is distinguishable from a rule that
    is being satisfied."""
    selftest(output_gate, FIXTURES / "black", expect_fail="OUT-FLAT")
    selftest(output_gate, FIXTURES / "silent", expect_fail="OUT-AUDIO")


def test_wrong_dimensions_are_caught(tmp_path: Path) -> None:
    """A render at a size the provider was not configured for means the graph
    and the capabilities disagree — and every downstream size calculation, the
    poster's aspect ratio included, is then wrong."""
    _copy(tmp_path, "good", output={"width": 1024, "height": 576})

    assert "OUT-DIMS" in {f.rule_id for f in output_gate(GateContext(tmp_path)).failures}


def test_a_truncated_file_is_caught(tmp_path: Path) -> None:
    _copy(tmp_path, "good", output={"size_bytes": 0})

    assert "OUT-SIZE" in {f.rule_id for f in output_gate(GateContext(tmp_path)).failures}


def test_a_duration_that_does_not_match_the_frame_count_is_caught(
    tmp_path: Path,
) -> None:
    """124 frames at 24fps is 5.17s. A file claiming 3s lost frames somewhere."""
    _copy(tmp_path, "good", output={"duration_s": 3.0})

    assert "OUT-DURATION" in {
        f.rule_id for f in output_gate(GateContext(tmp_path)).failures
    }


def test_a_missing_luma_measurement_warns_rather_than_fails(tmp_path: Path) -> None:
    """Not measuring is not the same as measuring a failure. ffmpeg missing on
    the host must not throw away a clip that was paid for."""
    _copy(tmp_path, "good", drop=("luma",))
    report = output_gate(GateContext(tmp_path))

    assert report.failures == ()
    assert "OUT-FLAT-UNMEASURED" in {f.rule_id for f in report.findings}


def test_a_missing_artifact_says_which_stage_did_not_run(tmp_path: Path) -> None:
    (tmp_path / "provider_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(GateFailure, match=r"output\.json"):
        output_gate(GateContext(tmp_path))


def _copy(dest: Path, fixture: str, *, output: dict | None = None, drop=()) -> None:
    src = FIXTURES / fixture
    (dest / "provider_manifest.json").write_text(
        (src / "provider_manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = json.loads((src / "output.json").read_text(encoding="utf-8"))
    payload.update(output or {})
    for key in drop:
        payload.pop(key, None)
    (dest / "output.json").write_text(json.dumps(payload), encoding="utf-8")


# ------------------------------------------------------------- the wiring


class _StubWorkflow:
    source = "workflows/h3_fl2va_turbo.json"


class _StubProvider:
    workflow = _StubWorkflow()

    def capabilities(self) -> Any:
        from ai_studio.providers.comfyui import h3_capabilities

        return h3_capabilities()


class _Job:
    id = 7
    token = "tok-verify"

    def __init__(self, kind: Any) -> None:
        self.media_kind = kind


@pytest.mark.ffmpeg
def test_verify_output_rejects_a_flat_clip_end_to_end(tmp_path: Path) -> None:
    """The whole path for real: ffmpeg measures it, the artifacts get written,
    the gate reads them back, and a black clip comes out rejected."""
    from ai_studio import media
    from ai_studio.core.enums import MediaKind
    from ai_studio.pipeline.worker import verify_output

    if media.which(media.get_settings().ffmpeg_bin) is None:
        pytest.skip("ffmpeg is not on PATH")

    black = tmp_path / "black.mp4"
    subprocess.run(
        [media.get_settings().ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:size=864x480:rate=24:duration=5.167",
         "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
         "-shortest", "-pix_fmt", "yuv420p", str(black)],
        check=True,
    )

    verdict = verify_output(
        _Job(MediaKind.VIDEO), black, _StubProvider(), runs_dir=tmp_path / "runs"
    )

    assert verdict is not None and "OUT-FLAT" in verdict
    run_dir = tmp_path / "runs" / "tok-verify"
    assert (run_dir / "provider_manifest.json").is_file(), "the manifest was not written"
    assert (run_dir / "output.json").is_file()
    assert (run_dir / "gates" / "output.json").is_file(), "no gate report was kept"


@pytest.mark.asyncio
async def test_a_rejected_render_is_not_delivered_as_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring, without ffmpeg. A flat clip must reach the user as a failure
    they are told about, not as a video they wait for and never get."""
    from ai_studio.pipeline import worker
    from ai_studio.pipeline.queue import JobQueue, JobState

    monkeypatch.setattr(worker, "verify_output", lambda *a, **kw: "[OUT-FLAT] flat")

    from test_worker import FakeHost, _files, _parsed  # type: ignore[import-not-found]

    with JobQueue(tmp_path / "q.sqlite3") as queue:
        job = _parsed(queue)
        host = FakeHost()
        report = worker.WorkerReport()

        action = await worker.tick(
            queue, host, files_dir=_files(tmp_path), report=report
        )

        assert action == "rejected"
        assert report.rejected == 1
        assert queue.by_id(job.id).state is JobState.FAILED
        assert host.delivered == [(job.id, None)], "the user was not told"
        assert "REJECTED=1" in report.summary()
