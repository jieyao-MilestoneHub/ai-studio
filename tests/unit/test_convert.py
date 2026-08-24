"""Colloquial text -> validated H3 prompt.

The property that matters: `prompts/h3.py`'s validation stands between a model
hallucination and a submitted GPU job. A reply that would produce a prompt the
model cannot follow must be rejected, not rendered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videogen.llm.endpoint import ScriptedLlmClient
from videogen.pipeline.convert_worker import DEFAULT_DURATION_S, convert_job, convert_pending
from videogen.pipeline.queue import JobQueue, JobState
from videogen.prompts.convert import (
    ConversionError,
    build_prompt,
    convert,
    template_prompt,
)

DURATION = DEFAULT_DURATION_S  # 124 frames at 24fps


def _good(**over) -> str:
    payload = {
        "shots": [
            {
                "style": "Live-action, cinematic",
                "description": "an orange tabby cat walks along a wet asphalt road in rain",
                "camera": {
                    "motion": "tracking_shot",
                    "amplitude": "small",
                    "speed": "slow",
                    "toward": "the cat",
                },
            },
            {
                "cut_at_s": 2.5,
                "description": "the same cat rendered as thick impasto pixel art",
                "camera": {"motion": "push_in", "amplitude": "small", "speed": "slow"},
            },
        ],
        "overall_soundscape": "Steady rain hisses on asphalt with small wet paw steps.",
        "non_diegetic_music": "Sparse piano at a slow tempo over a low sustained string.",
    }
    payload.update(over)
    return json.dumps(payload)


# --------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_a_good_reply_becomes_a_schema_conformant_prompt() -> None:
    prompt, how = await convert("一隻橘貓走在雨中", ScriptedLlmClient(_good()), duration_s=DURATION)

    assert how == "llm"
    assert len(prompt.shots) == 2
    rendered = prompt.render()
    assert rendered.startswith("integrated_multimodal_description: [Shot 1]")
    assert "At 00:02.500, the camera cuts to" in rendered
    assert "overall_soundscape:" in rendered
    assert "non_diegetic_music:" in rendered


@pytest.mark.asyncio
async def test_camera_words_are_rendered_from_the_closed_vocabulary() -> None:
    prompt, _ = await convert("貓", ScriptedLlmClient(_good()), duration_s=DURATION)
    assert "The camera follows the subject with small amplitude at slow speed" in prompt.render()


@pytest.mark.asyncio
async def test_code_fences_and_chatter_around_the_json_are_tolerated() -> None:
    noisy = f"Sure! Here is the plan:\n```json\n{_good()}\n```\nHope that helps."
    prompt, how = await convert("貓", ScriptedLlmClient(noisy), duration_s=DURATION)
    assert how == "llm"
    assert len(prompt.shots) == 2


# ------------------------------------------------- the h3 schema as a guard


def test_a_cut_outside_the_clip_is_rejected() -> None:
    """h3.py's own rule, doing the work."""
    payload = json.loads(_good())
    payload["shots"][1]["cut_at_s"] = 99.0
    with pytest.raises(ConversionError, match="H3 schema validation"):
        build_prompt(payload, DURATION)


def test_non_increasing_cuts_are_rejected() -> None:
    payload = json.loads(_good())
    payload["shots"].append({"cut_at_s": 1.0, "description": "goes backwards"})
    with pytest.raises(ConversionError, match="H3 schema validation"):
        build_prompt(payload, DURATION)


def test_a_shot_with_no_description_is_rejected() -> None:
    payload = json.loads(_good())
    payload["shots"][0]["description"] = ""
    with pytest.raises(ConversionError, match="no description"):
        build_prompt(payload, DURATION)


def test_an_invented_camera_word_loses_the_move_not_the_clip() -> None:
    """A closed vocabulary is exactly what a model gets almost right."""
    payload = json.loads(_good())
    payload["shots"][0]["camera"] = {"motion": "swooping dramatically"}
    prompt = build_prompt(payload, DURATION)
    assert prompt.shots[0].camera is None


def test_a_malformed_speaker_id_loses_the_line_not_the_clip() -> None:
    payload = json.loads(_good())
    payload["shots"][0]["dialogue"] = [
        {"speaker_id": "narrator", "identity": "x", "text": "hello"},
        {"speaker_id": "S1", "identity": "The baker", "text": "First batch."},
    ]
    prompt = build_prompt(payload, DURATION)
    assert len(prompt.shots[0].dialogue) == 1
    assert prompt.shots[0].dialogue[0].speaker_id == "S1"


def test_too_many_shots_are_truncated() -> None:
    payload = {
        "shots": [{"style": "Live-action", "description": f"shot {i}", "cut_at_s": i * 0.5}
                  for i in range(10)],
        "overall_soundscape": "Ambience.",
    }
    payload["shots"][0].pop("cut_at_s")
    prompt = build_prompt(payload, DURATION)
    assert len(prompt.shots) == 4


# ----------------------------------------------------------------- fallback


@pytest.mark.asyncio
async def test_invalid_json_retries_then_falls_back_to_a_template() -> None:
    """A dropped request is worse than a mediocre one."""
    client = ScriptedLlmClient("not json at all", "still not json")
    prompt, how = await convert("一隻橘貓走在雨中", client, duration_s=DURATION)

    assert how.startswith("template (llm failed")
    assert len(client.calls) == 2, "it should have retried exactly once"
    assert len(prompt.shots) == 1
    assert "一隻橘貓走在雨中" in prompt.render()


@pytest.mark.asyncio
async def test_a_second_attempt_that_succeeds_is_recorded_as_a_retry() -> None:
    client = ScriptedLlmClient("garbage", _good())
    _, how = await convert("貓", client, duration_s=DURATION)
    assert how == "llm-retry"


@pytest.mark.asyncio
async def test_a_network_failure_falls_back_rather_than_raising() -> None:
    class Broken:
        async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
            raise TimeoutError("endpoint cold and slow")

    prompt, how = await convert("貓", Broken(), duration_s=DURATION)
    assert "template" in how
    assert prompt.render()


@pytest.mark.asyncio
async def test_no_client_at_all_still_produces_a_valid_prompt() -> None:
    """The whole pipeline must be runnable with no LLM deployed."""
    prompt, how = await convert("一隻貓", None, duration_s=DURATION)
    assert how == "template"
    assert prompt.render()


def test_the_template_prompt_is_schema_valid() -> None:
    assert template_prompt("一隻貓在雨中走", DURATION).render().startswith(
        "integrated_multimodal_description:"
    )


# ------------------------------------------------------------- queue wiring


@pytest.mark.asyncio
async def test_convert_job_makes_a_queued_request_claimable(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-1", "Cgroup", "一隻橘貓走在雨中")
        assert q.claim_next() is None, "unparsed work must not reach a GPU"

        how = await convert_job(q, job.id, ScriptedLlmClient(_good()))
        assert how == "llm"

        claimed = q.claim_next()
        assert claimed is not None
        assert claimed.state is JobState.RUNNING
        assert claimed.prompt is not None
        assert claimed.prompt["_built_by"] == "llm"
        assert claimed.prompt["_rendered"].startswith("integrated_multimodal_description:")


@pytest.mark.asyncio
async def test_convert_pending_catches_up_a_backlog(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        for i in range(3):
            q.enqueue(f"evt-{i}", "Cgroup", f"貓 {i}")

        tally = await convert_pending(q, ScriptedLlmClient(_good(), _good(), _good()))
        assert tally == {"llm": 3}
        assert q.counts().get("parsed") == 3


@pytest.mark.asyncio
async def test_convert_job_on_an_unknown_id_is_a_no_op(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        assert "skipped" in await convert_job(q, 999, ScriptedLlmClient(_good()))
