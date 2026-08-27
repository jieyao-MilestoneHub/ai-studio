"""Tool-name masking. SPEC.md §5.3/D16: "工具名稱 MUST 於訓練時做隨機置換或遮蔽，
強迫模型學習選擇準則而非特定工具名" — this is the actual mechanism C2/C3 rely on
(the model learns *when* to reach for a tool, not a memorized name for it).
Pure and deterministic given a seed (SPEC.md §7.5's reproducibility binding),
operates only on already-typed `Trajectory` data — no model dependency.
"""

from __future__ import annotations

import random

from twin.core.trajectory import Step, ToolCallStep


def mask_tool_names(
    steps: list[Step], available_tools: list[str], *, rng: random.Random
) -> tuple[list[Step], list[str]]:
    """Randomly permutes tool names: every real name maps to another real name
    from the same trajectory's tool vocabulary, consistently within this call
    (the same name always maps to the same replacement), so the model still
    sees a coherent — just renamed — set of tools rather than a nonsense one."""
    tool_call_names = {step.tool for step in steps if isinstance(step, ToolCallStep)}
    unique_tools = sorted(tool_call_names | set(available_tools))
    shuffled = unique_tools[:]
    rng.shuffle(shuffled)
    rename = dict(zip(unique_tools, shuffled, strict=True))

    masked_steps: list[Step] = [
        step.model_copy(update={"tool": rename[step.tool]}) if isinstance(step, ToolCallStep) else step
        for step in steps
    ]
    masked_available_tools = [rename[tool] for tool in available_tools]
    return masked_steps, masked_available_tools
