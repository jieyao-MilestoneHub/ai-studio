"""PRE gate over `plan.json`: pacing, transitions, captions -- before any
GPU-second is spent.

A pure function of one on-disk artifact, like every gate. `plan.json` is
generic: segments with intended durations and the clip each renders in,
the cut reason at each clip boundary, caption cues bound to segments, and
the pacing band the caller wants applied. It knows nothing about beats,
shots or LINE; the drama writes it and reads the report back.

```json
{"fps": 24,
 "pacing": {"min_s": 2.0, "warn_s": 8.0, "fail_s": 12.5, "cv_floor": 0.11,
            "max_consecutive_slow": 1, "total_band_s": [55, 65]},
 "segments": [{"segment_id": "sg_..", "shot_id": "sh_..", "scene_id": "sc_..",
               "subcut_index": 0, "intended_duration_s": 2.5, "clip": "1"}],
 "transitions": [{"after_clip": "1", "reason": "default"}],
 "cues": [{"cue_id": "cue_..", "segment_id": "sg_..", "text": "...", "color_key": "w"}]}
```

Rule ids: `P-*` are this gate's own structural checks; `R-*`, `T-*` and
`C-*` come from `editing.rhythm`, `editing.transitions` and
`editing.captions` respectively.
"""

from __future__ import annotations

from typing import Any

from ai_studio.core.enums import TransitionReason
from ai_studio.core.models import GateReport
from ai_studio.editing import captions, rhythm, transitions
from ai_studio.gates.core import GateContext, GateRun

GATE = "plan_gate"
PLAN_FILE = "plan.json"


def run(ctx: GateContext) -> GateReport:
    plan = ctx.artifact(PLAN_FILE)
    gate = GateRun(GATE)

    fps = plan.get("fps")
    gate.assert_that(isinstance(fps, int) and fps > 0, "P-FPS", f"plan fps must be a positive integer, got {fps!r}")
    segments: list[dict[str, Any]] = list(plan.get("segments") or [])
    gate.assert_that(bool(segments), "P-SEGMENTS", "plan has no segments")

    durations: list[float] = []
    by_id: dict[str, float] = {}
    for seg in segments:
        seg_id = str(seg.get("segment_id", ""))
        d = float(seg.get("intended_duration_s", 0.0))
        durations.append(d)
        by_id[seg_id] = d
        if isinstance(fps, int) and fps > 0:
            frames = d * fps
            gate.assert_that(
                abs(frames - round(frames)) < 1e-6, "P-QUANTUM",
                f"segment {seg_id} is {d}s, not a whole number of frames at {fps} fps",
                where=seg_id, observed=f"{frames:.4f}", expected="integer",
            )
    gate.count("segments", len(segments))

    pacing: dict[str, Any] = plan.get("pacing") or {}
    if gate.assert_that(bool(pacing), "P-PACING", "plan carries no pacing policy"):
        band = pacing.get("total_band_s")
        policy = rhythm.PacingPolicy(
            min_s=float(pacing["min_s"]), warn_s=float(pacing["warn_s"]), fail_s=float(pacing["fail_s"]),
            cv_floor=float(pacing.get("cv_floor", rhythm.CV_FLOOR)),
            max_consecutive_slow=int(pacing.get("max_consecutive_slow", 1)),
            total_band_s=(float(band[0]), float(band[1])) if band else None,
        )
        if durations and all(d > 0 for d in durations):
            for f in rhythm.check(durations, policy):
                gate.assert_that(False, f.rule_id, f.message, severity=f.severity, where=f.where,
                                 observed=f.observed, expected=f.expected, source_url=f.source_url)
            gate.count("duration_cv", round(rhythm.coefficient_of_variation(durations), 4))
            gate.count("total_s", round(sum(durations), 3))

    reasons: list[TransitionReason] = []
    for t in plan.get("transitions") or []:
        raw = str(t.get("reason", ""))
        try:
            reasons.append(TransitionReason(raw))
        except ValueError:
            gate.assert_that(False, "P-REASON", f"unknown transition reason {raw!r}", observed=raw,
                             expected=[r.value for r in TransitionReason])
    planned = transitions.plan(reasons)
    # Every segment boundary is a splice; the ones inside a clip are hard
    # cuts by construction and count toward the >= 90% rule like any other.
    internal = max(len(segments) - 1 - len(planned), 0)
    for f in transitions.check([*planned, *([transitions.hard_cut()] * internal)]):
        gate.assert_that(False, f.rule_id, f.message, severity=f.severity, observed=f.observed,
                         expected=f.expected, source_url=f.source_url)
    gate.count("hard_cuts", sum(t.is_hard_cut for t in planned))
    gate.count("dissolves", sum(t.kind.value == "dissolve" for t in planned))

    cue_rows: list[tuple[str, float, str]] = []
    for cue in plan.get("cues") or []:
        seg_id = str(cue.get("segment_id", ""))
        if not gate.assert_that(seg_id in by_id, "P-SEGMENT-REF",
                                f"cue {cue.get('cue_id')} points at unknown segment {seg_id!r}", where=seg_id):
            continue
        cue_rows.append((str(cue.get("text", "")), by_id[seg_id], str(cue.get("color_key", "w"))))
    for f in captions.check(cue_rows):
        gate.assert_that(False, f.rule_id, f.message, severity=f.severity, where=f.where,
                         observed=f.observed, expected=f.expected, source_url=f.source_url)
    gate.count("cues", len(cue_rows))
    return gate.report()


def describe(report: GateReport) -> str:
    """One ASCII line for a state file or a log: passed / passed with N warning(s) / failed: ..."""
    if report.failures:
        return "failed: " + "; ".join(f"[{f.rule_id}] {f.message}" for f in report.failures)[:300]
    if report.warnings:
        return f"passed with {len(report.warnings)} warning(s): " + ", ".join(f.rule_id for f in report.warnings)
    return "passed"
