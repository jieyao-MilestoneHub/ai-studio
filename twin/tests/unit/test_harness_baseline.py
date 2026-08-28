"""B0/B1/B2 baseline prompt construction and sample generation. EVAL.md §3.4."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from twin.harness.baseline import (
    generate_baseline_samples,
    render_b0_prompt,
    render_b1_prompt,
    render_b2_prompt,
)
from twin.harness.schema import HarnessError, S1Item
from twin.harness.shard import sample_id


def _item(item_id: str = "i1") -> S1Item:
    return S1Item(
        item_id=item_id,
        item_type="preference",
        prompt=f"prompt-{item_id}",
        options=["a", "b"],
        source_fragment_ids=["frag-1"],
    )


@dataclass
class _FakeBackend:
    reply: str = "candidate answer"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_render_b0_prompt_contains_no_persona_or_transcript_text() -> None:
    prompt = render_b0_prompt(_item())
    assert "prompt-i1" in prompt
    assert "persona" not in prompt.lower()
    assert "transcript" not in prompt.lower()


def test_render_b1_prompt_contains_persona_text() -> None:
    prompt = render_b1_prompt(_item(), persona_text="I am a careful planner.")
    assert "I am a careful planner." in prompt
    assert "prompt-i1" in prompt


def test_render_b2_prompt_contains_transcript_text() -> None:
    prompt = render_b2_prompt(_item(), transcript_text="A: tell me about yourself. B: ...")
    assert "tell me about yourself" in prompt
    assert "prompt-i1" in prompt


def test_generate_baseline_samples_rejects_b1_without_persona_text() -> None:
    backend = _FakeBackend()
    with pytest.raises(HarnessError, match="persona_text"):
        generate_baseline_samples(items=[_item()], baseline="B1", backend=backend)
    assert backend.prompts == []


def test_generate_baseline_samples_rejects_b2_without_transcript_text() -> None:
    backend = _FakeBackend()
    with pytest.raises(HarnessError, match="transcript_text"):
        generate_baseline_samples(items=[_item()], baseline="B2", backend=backend)
    assert backend.prompts == []


def test_generate_baseline_samples_all_use_twin_source_label() -> None:
    backend = _FakeBackend()
    samples = generate_baseline_samples(items=[_item("a"), _item("b")], baseline="B0", backend=backend)
    assert all(sample.source_label == "twin" for sample in samples)
    assert len(samples) == 2


def test_generate_baseline_samples_sample_id_is_content_derived() -> None:
    backend = _FakeBackend(reply="a specific answer")
    samples = generate_baseline_samples(items=[_item()], baseline="B0", backend=backend)
    assert samples[0].sample_id == sample_id("a specific answer")


def test_generate_baseline_samples_b1_calls_backend_with_persona_prompt() -> None:
    backend = _FakeBackend()
    generate_baseline_samples(items=[_item()], baseline="B1", backend=backend, persona_text="persona-marker")
    assert "persona-marker" in backend.prompts[0]
