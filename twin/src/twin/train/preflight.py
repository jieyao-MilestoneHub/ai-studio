"""Pre-training dataset gate. Run on the exact `datasets.Dataset` that is
about to be trained on, before a single GPU-second (the same reasoning as
ai-studio's PRE gates: a post-hoc check on a 5-hour run is a receipt, not a
check). Every rule here is a defect that an actual run surfaced first:

- `\\uXXXX` escapes in a tool-call target (T v1, 2026-08-30: `json.dumps`
  default ensure_ascii — the adapter learned to emit escapes);
- empty reply targets (would teach "call reply with nothing");
- self-report share too small to matter (D19: 30/15,673 = 0.2% is noise) —
  or, if a future set ever inverts this, too large to still be a LINE-behaviour
  model.

Fail loudly (twin/CLAUDE.md): raise, never warn-and-continue.
"""

from __future__ import annotations

import json
from typing import Any

import datasets
from pydantic import BaseModel, ConfigDict

from twin.ingest.interview_trajectories import INTERVIEW_SURFACE

MIN_SELF_REPORT_SHARE = 0.01  # below this, D19's "dominant source" is not in the training signal at all
MAX_SELF_REPORT_SHARE = 0.25


class PreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: int
    reply_targets: int
    no_action_targets: int
    self_report_targets: int
    self_report_share: float
    escaped_targets: int
    empty_targets: int

    def render(self) -> str:
        return (
            f"preflight: {self.samples} samples — {self.reply_targets} reply targets, {self.no_action_targets} no_action, "
            f"{self.self_report_targets} self-report ({self.self_report_share:.1%}); "
            f"{self.escaped_targets} \\u-escaped, {self.empty_targets} empty"
        )


def inspect_dataset(dataset: datasets.Dataset) -> PreflightReport:
    reply = no_action = self_report = escaped = empty = 0
    for example in dataset:
        for message in example["messages"]:
            if message["role"] != "assistant":
                continue
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                no_action += 1
                continue
            reply += 1
            arguments: str = tool_calls[0]["function"]["arguments"]
            if "\\u" in arguments:
                escaped += 1
            parsed: dict[str, Any] = json.loads(arguments)
            content = str(parsed.get("content", ""))
            if not content.strip():
                empty += 1
            if parsed.get("surface") == INTERVIEW_SURFACE:
                self_report += 1
    share = self_report / len(dataset) if len(dataset) else 0.0
    return PreflightReport(
        samples=len(dataset),
        reply_targets=reply,
        no_action_targets=no_action,
        self_report_targets=self_report,
        self_report_share=share,
        escaped_targets=escaped,
        empty_targets=empty,
    )


def assert_dataset_trainable(dataset: datasets.Dataset, *, require_self_report: bool) -> PreflightReport:
    """`require_self_report=False` is for the kill/resume toy run and any
    deliberately LINE-only experiment; production runs after Phase 3 pass
    True."""
    report = inspect_dataset(dataset)
    problems: list[str] = []
    if report.samples == 0:
        problems.append("no training samples")
    if report.escaped_targets:
        problems.append(f"{report.escaped_targets} tool-call targets contain \\u escapes (train.formatting ensure_ascii regression)")
    if report.empty_targets:
        problems.append(f"{report.empty_targets} reply targets are empty")
    if require_self_report and report.self_report_share < MIN_SELF_REPORT_SHARE:
        problems.append(
            f"self-report is {report.self_report_share:.2%} of samples (< {MIN_SELF_REPORT_SHARE:.0%}) — D19 says it is the "
            f"dominant persona source; add/augment interview trajectories before spending GPU time"
        )
    if report.self_report_share > MAX_SELF_REPORT_SHARE:
        problems.append(f"self-report is {report.self_report_share:.0%} of samples (> {MAX_SELF_REPORT_SHARE:.0%}) — behaviour data is being drowned")
    if problems:
        raise ValueError("dataset preflight failed:\n  - " + "\n  - ".join(problems) + f"\n  ({report.render()})")
    return report
