"""Workflow graph validation, run before anything is submitted.

The load-bearing check here is the MiniMax H3 turbo trap, which is worth
stating in full because it is the most expensive mistake available in this
project and it disguises itself as a win.

The H3 turbo LoRA cannot be driven by ComfyUI's stock LoRA loader. The pruned
model replaces its AdaLN branch with a lookup table, so the LoRA has to go
through the dedicated ``MiniMaxH3TurboLoRA`` node, and sampling has to go
through ``MiniMaxH3TurboSampler`` rather than the generic ``KSamplerSelect``.
Wire it the stock way and you get vertical comb artifacts and banded gradients.

The trap is that **the broken path is faster**: 2.53 s/iteration against 9.8
s/iteration for the correct one. Benchmark it and you will conclude you found a
6.3x free speedup. You did not — the model is skipping work and emitting
garbage. Turbo's real speedup over 20-step base is about 1.7x. [reported]

A wrong-but-fast path that produces plausible-looking output is exactly the
kind of failure a human review misses and a mechanical check catches, so this
runs on every submission rather than being left to discipline.
"""

from __future__ import annotations

from typing import Any

from ai_studio.core.errors import GraphValidationError

TURBO_LORA_NODE = "MiniMaxH3TurboLoRA"
TURBO_SAMPLER_NODE = "MiniMaxH3TurboSampler"

STOCK_LORA_NODES = frozenset({"LoraLoader", "LoraLoaderModelOnly"})
"""Fine for ordinary LoRAs. Silently destructive for the H3 turbo LoRA."""

STOCK_SAMPLER_NODES = frozenset({"KSamplerSelect", "KSampler", "KSamplerAdvanced"})

_TURBO_HINTS = ("turbo", "lightning", "4step", "4-step")


def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Node map, ignoring our own `_ai_studio` metadata block."""
    return {
        node_id: node
        for node_id, node in graph.items()
        if not node_id.startswith("_") and isinstance(node, dict)
    }


def _class_types(graph: dict[str, Any]) -> dict[str, str]:
    return {nid: str(n.get("class_type", "")) for nid, n in _nodes(graph).items()}


def _mentions_turbo(node: dict[str, Any]) -> bool:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return False
    for value in inputs.values():
        if isinstance(value, str) and any(h in value.lower() for h in _TURBO_HINTS):
            return True
    return False


def uses_turbo_lora(graph: dict[str, Any]) -> bool:
    """True if any node loads something that looks like a turbo/step-distilled LoRA."""
    for node in _nodes(graph).values():
        if str(node.get("class_type", "")) == TURBO_LORA_NODE:
            return True
        if str(node.get("class_type", "")) in STOCK_LORA_NODES and _mentions_turbo(node):
            return True
    return False


def validate_graph(graph: dict[str, Any], *, expect_turbo: bool | None = None) -> None:
    """Raise `GraphValidationError` if the graph is malformed or mis-wired.

    `expect_turbo=True` additionally asserts the turbo path is actually present,
    so a workflow that silently lost its LoRA does not quietly render at base
    speed and bill for it.
    """
    problems: list[str] = []

    nodes = _nodes(graph)
    if not nodes:
        raise GraphValidationError("workflow contains no nodes")

    classes = _class_types(graph)

    # --- structural: every link must point at a node that exists
    for node_id, node in nodes.items():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            problems.append(f"node {node_id} ({classes[node_id]}) has no inputs mapping")
            continue
        for key, value in inputs.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                target = value[0]
                if target not in nodes:
                    problems.append(
                        f"node {node_id}.{key} links to node {target!r}, which does not exist"
                    )
        if not classes[node_id]:
            problems.append(f"node {node_id} has no class_type")

    # --- the turbo trap
    turbo = uses_turbo_lora(graph)

    if expect_turbo and not turbo:
        problems.append(
            f"expected the turbo path but found no {TURBO_LORA_NODE}. "
            "Rendering would fall back to full base steps at roughly 1.7x the cost."
        )

    if turbo:
        stock_lora = [nid for nid, cls in classes.items() if cls in STOCK_LORA_NODES]
        if stock_lora:
            problems.append(
                f"turbo LoRA driven through stock loader node(s) {stock_lora} "
                f"({', '.join(sorted({classes[n] for n in stock_lora}))}). "
                f"Use {TURBO_LORA_NODE} instead. The stock path runs ~4x faster and "
                "produces vertical comb artifacts and banding — it looks like a "
                "speedup and is actually broken output."
            )
        if TURBO_LORA_NODE not in classes.values():
            problems.append(f"turbo workflow is missing a {TURBO_LORA_NODE} node")

        stock_sampler = [nid for nid, cls in classes.items() if cls in STOCK_SAMPLER_NODES]
        if stock_sampler:
            problems.append(
                f"turbo workflow samples through {stock_sampler} "
                f"({', '.join(sorted({classes[n] for n in stock_sampler}))}). "
                f"The turbo path requires {TURBO_SAMPLER_NODE}."
            )
        if TURBO_SAMPLER_NODE not in classes.values():
            problems.append(f"turbo workflow is missing a {TURBO_SAMPLER_NODE} node")

    if problems:
        raise GraphValidationError(
            "workflow graph failed validation:\n  - " + "\n  - ".join(problems)
        )
