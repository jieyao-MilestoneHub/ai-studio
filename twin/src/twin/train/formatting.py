"""Trajectory -> SFTTrainer-ready chat examples. SPEC.md §4.10 (Step union),
§5.3/D16 (tool-name masking happens here, at data-build time — not in
model.py or run.py), §6.2/§6.3/§11-A/D28/D29 (no_action and reply are
ordinary steps, no special branch — "回覆為 tool call": replying IS the twin
choosing to call `reply`, at the same level as every other tool), C1/C2/C3
(only what the trajectory already carries goes into the formatted text — no
memory-store calls, no tool JSON schema).

Deliberately a separate module from `train.data`, not an extension of it:
`train.data.load_training_examples` is the one function PLAN.md §3.5 names as
the target of an un-skippable CI test guarding SPEC.md §4.8's split filter.
Mixing unrelated formatting logic into that file would widen the blast
radius of the one module that must never grow a way to be bypassed.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import datasets

from twin.core.trajectory import (
    ActionStep,
    NoActionStep,
    ReflectionStep,
    Step,
    ToolCallStep,
    Trajectory,
)
from twin.ingest.interview_trajectories import INTERVIEW_SURFACE
from twin.train.data import load_training_examples
from twin.train.masking import mask_tool_names


def _rng_for(seed: int, trajectory_id: str) -> random.Random:
    """Per-trajectory RNG derived from (seed, trajectory_id) rather than one
    Random advanced across the dataset iterator — masking results MUST NOT
    depend on iteration order (data-contract skill rule 8's "order MUST NOT
    affect the result" discipline, same reasoning as core.hashing)."""
    return random.Random(f"{seed}:{trajectory_id}")


def _reply_as_tool_call(step: ActionStep) -> ToolCallStep:
    """SPEC.md §11-A/D28/D29: "回覆為 tool call" — reply MUST sit at the same
    level as every other tool, no special calling path, precisely so the
    send-gate level can change without a retrain. Converting `ActionStep` into
    a synthetic `ToolCallStep(tool="reply", ...)` *before* masking is what
    makes that real: it lets `reply` enter the exact same §5.3/D16 name-
    permutation pool as `recall`/`web_search`, instead of the model learning
    "reply" as a hardcoded, never-masked literal.

    `result_digest` here is a local, training-time-only hash of
    (surface, content) — it is NOT required to bit-match
    `agent.tools.reply.ReplyTool`'s own digest computation. `twin.train` MUST
    NOT import `twin.agent` (the Layered spine import-linter contract: `agent`
    sits above `train`), and the model only needs to see a structurally
    consistent tool_call -> tool-response shape, not a byte-identical hash.
    """
    digest = hashlib.sha256(f"{step.surface}\x1f{step.content}".encode()).hexdigest()[:16]
    return ToolCallStep(tool="reply", args={"surface": step.surface, "content": step.content}, result_digest=digest)


def trajectory_to_messages(trajectory: Trajectory, *, seed: int) -> list[dict[str, Any]]:
    """One Trajectory -> one OpenAI-style message list, tool names already
    masked (SPEC.md §5.3/D16), with `reply` folded into that same masked
    tool-call vocabulary (see `_reply_as_tool_call`).

    Raises on a `ReflectionStep`: Phase 4's slim trajectory set MUST NOT
    contain any (reflection-step generation and its serialization format are
    Phase 10 decisions, SPEC.md §5.4, not yet made). A trajectory that somehow
    carries one reaching this function is a data-hygiene bug, not something to
    silently reformat by guessing.

    `NoActionStep.reason` is deliberately NOT written into the returned
    messages — it stays corpus metadata (available for SPEC.md §4.11 negative-
    class auditing) rather than trained-on text. SPEC.md doesn't decide this
    serialization choice explicitly; this is this module's own interpretation
    of "呼叫零個工具即為 no_action，無特殊分支" applied at the data layer.
    """
    rng = _rng_for(seed, trajectory.trajectory_id)
    steps_for_masking: list[Step] = [
        _reply_as_tool_call(step) if isinstance(step, ActionStep) else step for step in trajectory.steps
    ]
    masked_steps, masked_tools = mask_tool_names(steps_for_masking, trajectory.available_tools, rng=rng)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"available tools: {', '.join(masked_tools)}"},
        {"role": "user", "content": trajectory.observation},
    ]
    for step in masked_steps:
        if isinstance(step, ToolCallStep):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            # ensure_ascii=False: with the default, every CJK character in the
                            # principal's replies became a 6-char `\uXXXX` escape in the training
                            # target, and the first real adapter (run_e6a366ee73958e69) learned to
                            # emit exactly that — found in the 2026-08-30 S1 smoke test.
                            "function": {"name": step.tool, "arguments": json.dumps(step.args, ensure_ascii=False)},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "name": step.tool, "content": step.result_digest})
        elif isinstance(step, NoActionStep):
            messages.append({"role": "assistant", "content": "", "tool_calls": []})
        elif isinstance(step, ReflectionStep):
            raise NotImplementedError(
                "trajectory_to_messages: encountered a ReflectionStep. Phase 4's "
                "slim trajectory set MUST NOT contain these — reflection "
                "serialization format is a Phase 10 decision (SPEC.md §5.4), not "
                "yet made. This trajectory should not have reached training; "
                "treat this as a data-hygiene bug, not a format to guess at."
            )
        else:  # pragma: no cover - ActionStep is converted away above; anything else is a closed union
            raise AssertionError(f"unreachable: unknown step type {step!r}")
    return messages


def _is_self_report(trajectory: Trajectory) -> bool:
    return any(isinstance(step, ActionStep) and step.surface == INTERVIEW_SURFACE for step in trajectory.steps)


def build_sft_dataset(
    trajectories_uri: str, *, seed: int, self_report_upsample: int = 1
) -> tuple[datasets.Dataset, list[str]]:
    """Reads only `split == train` (via `train.data.load_training_examples`,
    the un-skippable §4.8 filter), formats each trajectory, and materializes
    a map-style `datasets.Dataset` via `Dataset.from_list` — never
    `from_generator(streaming=True)` and never `.to_iterable_dataset()`.

    Map-style is a hard requirement, not a style preference: TRL/Accelerate's
    `resume_from_checkpoint` seeks to a dataloader sample cursor rather than
    re-iterating already-seen data only for a map-style dataset (SPEC.md
    §7.4 items 5-6). A streaming `IterableDataset` degrades resume into slow
    re-iteration and, in the worst case, the exact "fake convergence" symptom
    the checkpoint contract exists to prevent.

    Returns `(dataset, trajectory_ids_in_order)` so the caller can feed that
    id list straight into `core.hashing.dataset_hash()` without a second read
    of `trajectories_uri`. Self-report trajectories (an `ActionStep` on the
    `interview` surface, `ingest.interview_trajectories`) are repeated
    `self_report_upsample` times — and their ids repeated with them, so the
    dataset hash reflects the upsampling (see `TrainingConfig.self_report_
    upsample`). Repeats are contiguous here; the Trainer's random sampler
    spreads them across the epoch.
    """
    if self_report_upsample < 1:
        raise ValueError(f"self_report_upsample must be >= 1, got {self_report_upsample}")
    examples: list[dict[str, Any]] = []
    ids: list[str] = []
    for trajectory in load_training_examples(trajectories_uri):
        repeats = self_report_upsample if _is_self_report(trajectory) else 1
        example = {
            "trajectory_id": trajectory.trajectory_id,
            "messages": trajectory_to_messages(trajectory, seed=seed),
        }
        for _ in range(repeats):
            examples.append(example)
            ids.append(trajectory.trajectory_id)
    return datasets.Dataset.from_list(examples), ids
