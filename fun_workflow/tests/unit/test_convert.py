"""Colloquial text -> validated H3 prompt.

The property that matters: `prompts/h3.py`'s validation stands between a model
hallucination and a submitted GPU job. A reply that would produce a prompt the
model cannot follow must be rejected, not rendered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_studio.llm.scripted import ScriptedLlmClient
from ai_studio.prompts.convert import (
    ConversionError,
    build_prompt,
    convert,
    template_prompt,
)

from fun_workflow.pipeline.convert_worker import DEFAULT_DURATION_S, convert_job, convert_pending
from fun_workflow.pipeline.queue import JobQueue, JobState

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

        how = await convert_job(q, job.id, ScriptedLlmClient(_good()), prompt_mode="structured")
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

        tally = await convert_pending(q, ScriptedLlmClient(_good(), _good(), _good()), prompt_mode="structured")
        assert tally == {"llm": 3}
        assert q.counts().get("parsed") == 3


@pytest.mark.asyncio
async def test_convert_job_on_an_unknown_id_is_a_no_op(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        assert "skipped" in await convert_job(q, 999, ScriptedLlmClient(_good()))


# ------------------------------------------------------------ image-to-video


@pytest.mark.asyncio
async def test_i2v_briefs_the_model_about_the_photo_and_sets_the_mode() -> None:
    """Without the brief, "變油畫風格" against a portrait came back as a market
    scene in traditional-painting style -- a plan for a picture H3 was never
    shown. The photo is the first frame; the prompt must be about it."""
    from ai_studio.prompts.convert import I2V_BRIEF
    from ai_studio.prompts.h3 import H3Mode

    client = ScriptedLlmClient(_good())
    prompt, how = await convert(
        "變油畫風格", client, duration_s=DURATION, mode=H3Mode.I2VA
    )

    assert how == "llm"
    assert prompt.mode is H3Mode.I2VA
    assert "<Picture 1>" in prompt.render()
    user_message = client.calls[0][1]
    assert user_message.startswith(I2V_BRIEF)
    assert "Do not invent a different scene" in user_message


@pytest.mark.asyncio
async def test_t2v_does_not_mention_a_photo() -> None:
    from ai_studio.prompts.h3 import H3Mode

    client = ScriptedLlmClient(_good())
    prompt, _ = await convert("一隻橘貓走在雨中", client, duration_s=DURATION)
    assert prompt.mode is H3Mode.T2VA
    assert "Picture 1" not in client.calls[0][1]
    assert "Picture 1" not in prompt.render()


@pytest.mark.asyncio
async def test_the_template_fallback_keeps_the_i2v_mode() -> None:
    from ai_studio.prompts.h3 import H3Mode

    prompt, how = await convert("變油畫風格", None, duration_s=DURATION, mode=H3Mode.I2VA)
    assert how == "template" and prompt.mode is H3Mode.I2VA


@pytest.mark.asyncio
async def test_convert_job_uses_i2v_when_the_request_carries_a_photo(tmp_path: Path) -> None:
    from ai_studio.prompts.h3 import H3Mode

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-p", "Cgroup", "變油畫風格", first_frame_path="/incoming/x.jpg")
        client = ScriptedLlmClient(_good())
        await convert_job(q, job.id, client, prompt_mode="structured")
        assert q.by_id(job.id).prompt["mode"] == H3Mode.I2VA.value
        assert "Picture 1" in client.calls[0][1]


# ------------------------------------------------------ prompt-writing rules


def test_the_system_prompt_forbids_embellishment_and_keeps_the_users_words() -> None:
    """The brief that the LLM works from. Pinned because a prompt rewrite that
    quietly drops "do not embellish" is exactly how a user's 「橘貓」 becomes
    "a majestic feline" -- the model would still validate, the clip would
    still render, and nobody would know why it was the wrong cat."""
    from ai_studio.prompts.convert import I2V_BRIEF, SYSTEM_PROMPT

    for rule in (
        "do not embellish",
        "Every concrete word the user used",
        "Proper nouns and on-screen text stay verbatim",
        "not speaking, lips closed",
        "at most 2",
        "cuts. One shot is fine",
        # The community H3 guidance (2026-08-27): action first, one camera
        # move, audio directed, negatives inside the description, one job.
        "Lead with the ACTION",
        "Exactly ONE camera move per shot",
        "No dialogue.",
        "No text, no subtitles, no logos, no watermark, no extra people.",
        "Give the clip ONE job",
        "Never write bare praise",
    ):
        assert rule in SYSTEM_PROMPT, rule
    assert "describe only what\nhappens next" in I2V_BRIEF
    assert "Picture 1 is the opening frame" in I2V_BRIEF


def test_the_flux_system_prompt_keeps_the_users_words_too() -> None:
    from ai_studio.prompts.flux import SYSTEM_PROMPT

    assert "translate faithfully, do not embellish" in SYSTEM_PROMPT
    assert "Proper nouns stay\n  verbatim" in SYSTEM_PROMPT
    # Flux community guidance: sentences in a fixed order, 40-60 words, no
    # weights, no negatives, never "white background".
    for rule in ("40-60 words", "No negatives", "white background", "(word:1.2)"):
        assert rule in SYSTEM_PROMPT, rule


def test_the_h3_few_shot_example_parses_through_the_real_builder() -> None:
    """The example in the system prompt is the model's strongest instruction.
    If it ever stops validating, the model is being shown a shape the code
    would reject -- so it is built here, with the real parser."""
    from ai_studio.prompts.convert import SYSTEM_PROMPT, build_prompt, extract_json

    example = SYSTEM_PROMPT.split("total duration 10.12 seconds:", 1)[1]
    prompt = build_prompt(extract_json(example), duration_s=10.12)
    assert len(prompt.shots) == 2
    assert prompt.shots[1].cut_at_s == 6.0
    assert prompt.non_diegetic_music == "N/A"
    assert "No dialogue." in prompt.overall_soundscape


# ------------------------------------------------------------------ raw mode


@pytest.mark.asyncio
async def test_raw_mode_sends_the_users_words_verbatim_and_calls_no_llm(tmp_path: Path) -> None:
    """The default. The person typing knows what they want; every rewrite is
    a place their words can be lost, and every LLM call is $0.02 of a warm
    4090 worker."""
    class _Explodes:
        async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
            raise AssertionError("raw mode must not call the LLM")

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-raw", "Cgroup", "  一隻橘貓走在雨中  ")
        how = await convert_job(q, job.id, _Explodes(), prompt_mode="raw")
        plan = q.by_id(job.id).prompt

    assert how == "raw"
    assert plan["_rendered"] == "一隻橘貓走在雨中"
    assert plan["_built_by"] == "raw"
    assert plan["mode"] == "t2va" and plan["duration_s"] == DEFAULT_DURATION_S


@pytest.mark.asyncio
async def test_raw_mode_keeps_the_picture_binding_line_for_image_to_video(tmp_path: Path) -> None:
    from ai_studio.prompts.h3 import I2VA_INSTRUCTION

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-raw-i2v", "Cgroup", "變油畫風格", first_frame_path="/in/x.jpg")
        await convert_job(q, job.id, None, prompt_mode="raw")
        plan = q.by_id(job.id).prompt

    assert plan["mode"] == "i2va"
    assert plan["_rendered"].startswith(I2VA_INSTRUCTION)
    assert plan["_rendered"].endswith("變油畫風格")


@pytest.mark.asyncio
async def test_raw_mode_for_an_image_is_just_the_text(tmp_path: Path) -> None:
    from fun_workflow.core.kinds import JobKind

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-raw-img", "Cgroup", "亞洲辣妹", media_kind=JobKind.IMAGE)
        await convert_job(q, job.id, None, prompt_mode="raw")
        plan = q.by_id(job.id).prompt

    assert plan["_rendered"] == "亞洲辣妹" and "mode" not in plan


# --------------------------------------------------------- requested length


def test_clamp_duration_snaps_and_bounds() -> None:
    from fun_workflow.pipeline.convert_worker import (
        DEFAULT_DURATION_S,
        MAX_DURATION_S,
        clamp_duration,
    )

    assert clamp_duration(None) == DEFAULT_DURATION_S
    assert clamp_duration(15) == MAX_DURATION_S            # the offered ceiling
    assert clamp_duration(999) == MAX_DURATION_S           # above -> clamped
    assert clamp_duration(1) == 124 / 24                   # below floor -> floor
    assert abs(clamp_duration(8) - 192 / 24) < 1e-6        # snapped to the grid


@pytest.mark.asyncio
async def test_raw_mode_honours_a_requested_length(tmp_path: Path) -> None:
    from fun_workflow.pipeline.convert_worker import MAX_DURATION_S

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-15s", "Cgroup", "一隻貓", requested_seconds=15.0)
        await convert_job(q, job.id, None, prompt_mode="raw")
        plan = q.by_id(job.id).prompt

    assert plan["duration_s"] == MAX_DURATION_S
