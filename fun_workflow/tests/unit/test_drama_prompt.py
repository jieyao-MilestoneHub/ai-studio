"""The screenwriter: three model calls in, one validated `Screenplay` out.

What is asserted is what protects the money: the anchor and the world bible
cannot be dropped, the shot count and the beat template cannot drift, the
framings alternate, and there is no silent fallback that would send six
clips of nothing to the GPU.
"""

from __future__ import annotations

import json

import pytest
from ai_studio.llm.scripted import ScriptedLlmClient
from ai_studio.prompts.h3 import I2VA_INSTRUCTION
from pydantic import ValidationError

from fun_workflow.core.drama_spec import (
    BEAT_TEMPLATE,
    SHOT_COUNT,
    Beat,
    CharacterAnchor,
    DramaShot,
    Framing,
    Screenplay,
    SubShot,
    WorldBible,
)
from fun_workflow.prompts import drama

APPEARANCE = "25-year-old Asian woman, oval face, small mole under right eye, dark chin-length straight hair"

OUTLINE = {
    "title": "夜市的信",
    "logline": "A night-market stall owner finds a letter that says the market closes tomorrow.",
    "style": "Live-action, cinematic",
    "anchor": {
        "name": "阿玲",
        "appearance": APPEARANCE,
        "wardrobe": "a faded red apron over a white t-shirt",
        "voice": "soft, low, slightly hoarse",
    },
    "world": {
        "location": "a narrow night-market food stall facing one row of stalls",
        "light": "warm tungsten string lights from above left",
        "signature_prop": "a folded paper letter",
    },
    "beats": {b.value: f"{b.value} beat" for b in Beat},
    "overall_soundscape": "Sizzling oil, a crowd murmuring, scooters passing.",
    "non_diegetic_music": "N/A",
}

# Ten sub-shots whose framings alternate, one push-in on the turn (shot 4).
FRAMINGS = {
    1: ["close-up", "medium"], 2: ["wide", "close-up"], 3: ["over-the-shoulder"],
    4: ["wide", "medium close-up"], 5: ["close-up", "medium"], 6: ["wide"],
}


def _shots(indices: list[int]) -> dict:
    shots = []
    for i in indices:
        subs = []
        for k, framing in enumerate(FRAMINGS[i], start=1):
            sub = {"framing": framing, "action": f"the lead does thing {i}.{k}",
                   "camera": {"motion": "static_shot"}}
            if i == 4 and k == 2:
                sub["camera"] = {"motion": "push_in", "amplitude": "small", "speed": "slow"}
            if i == 3:
                sub["line"] = "明天就要收了嗎?"
            subs.append(sub)
        shots.append({"index": i, "scene": f"the stall, beat {i}", "sub_shots": subs,
                      "cut_reason": "time_passing" if i == 4 else "default"})
    return {"shots": shots}


def _client(*replies: dict) -> ScriptedLlmClient:
    return ScriptedLlmClient(*(json.dumps(r, ensure_ascii=False) for r in replies))


def _good_client() -> ScriptedLlmClient:
    return _client(OUTLINE, _shots([1, 2, 3]), _shots([4, 5, 6]))


def _screenplay(shots_1: dict | None = None, shots_2: dict | None = None) -> Screenplay:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    world = WorldBible.model_validate(OUTLINE["world"])
    kw = {"anchor": anchor, "world": world}
    shots = drama.build_shots(shots_1 or _shots([1, 2, 3]), expected=[1, 2, 3], **kw) + drama.build_shots(
        shots_2 or _shots([4, 5, 6]), expected=[4, 5, 6], **kw
    )
    return Screenplay(title="t", logline="l", anchor=anchor, world=world, shots=tuple(shots), overall_soundscape="quiet")


# ------------------------------------------------------------------ happy path


async def test_three_calls_make_a_six_shot_screenplay() -> None:
    client = _good_client()
    screenplay, how = await drama.write_screenplay("夜市老闆娘發現一封信", client)

    assert how == "llm"
    assert len(client.calls) == 3
    assert [s.index for s in screenplay.shots] == list(range(1, SHOT_COUNT + 1))
    assert [s.beat for s in screenplay.shots] == [slot.beat for slot in BEAT_TEMPLATE]
    assert [s.frames for s in screenplay.shots] == [158, 243, 192, 243, 243, 209]
    assert screenplay.anchor.appearance == APPEARANCE
    assert screenplay.shots[2].dialogue[0].text == "明天就要收了嗎?"
    assert screenplay.shots[2].dialogue[0].identity == "阿玲, soft, low, slightly hoarse"
    assert screenplay.shots[3].cut_reason.value == "time_passing"
    assert len(screenplay.sub_shots()) == 10
    assert set(screenplay.beats) == set(Beat)
    assert screenplay.beats[Beat.TURN] == "turn beat"


async def test_the_anchor_and_the_world_are_in_every_keyframe_prompt_verbatim() -> None:
    screenplay, _ = await drama.write_screenplay("x", _good_client())
    for shot in screenplay.shots:
        assert shot.keyframe_prompt.startswith(APPEARANCE)
        assert "faded red apron" in shot.keyframe_prompt
        assert screenplay.world.prefix() in shot.keyframe_prompt
        assert f"{shot.sub_shots[0].framing.value} shot" in shot.keyframe_prompt


async def test_the_model_never_gets_to_write_the_face_and_hears_the_seam() -> None:
    """The shots prompt hands the model the name and the slot sizes, not the
    appearance; the second call is told how shot 3 ended."""
    client = _good_client()
    await drama.write_screenplay("x", client)
    _, shots_user_1 = client.calls[1]
    _, shots_user_2 = client.calls[2]
    assert APPEARANCE not in shots_user_1
    assert "阿玲" in shots_user_1
    assert "Shot 1 (HOOK, 6.6 s, 2 sub-shots: 2.5 s then 4.1 s)" in shots_user_1
    assert "Shot 3 (CONFLICT, 8.0 s, 1 sub-shot)" in shots_user_1
    assert "ended on a over-the-shoulder framing" in shots_user_2


async def test_a_retry_is_recorded_as_llm_retry() -> None:
    client = _client({"garbage": True}, OUTLINE, _shots([1, 2, 3]), _shots([4, 5, 6]))
    _, how = await drama.write_screenplay("x", client)
    assert how == "llm-retry"


async def test_a_retry_is_told_the_exact_violation_it_made() -> None:
    """📏 2026-08-29 (job 107): the model repeated a validation failure
    verbatim on attempt 2 because the retry sent the same prompt. Now the
    retry states the specific error."""
    bad_shots = _shots([4, 5, 6])
    bad_shots["shots"][2]["sub_shots"][0]["action"] = "the phone buzzes with a new message, screen off"
    client = _client(OUTLINE, _shots([1, 2, 3]), bad_shots, _shots([4, 5, 6]))
    _, how = await drama.write_screenplay("x", client)
    assert how == "llm-retry"
    _, retried_user = client.calls[-1]
    assert "Your previous reply was invalid: shot 6 sub-shot 1" in retried_user
    assert "the lead must be in the action" in retried_user


def test_h3_prompt_cuts_inside_the_clip_at_the_template_time() -> None:
    screenplay = _screenplay()
    rendered = drama.h3_prompt(screenplay.shots[0], screenplay).render()
    assert rendered.startswith(I2VA_INSTRUCTION)
    assert APPEARANCE in rendered and screenplay.world.prefix() in rendered
    assert "[Shot 2] At 00:02.500, the camera cuts to" in rendered
    assert rendered.count("lips remain closed") == 2  # both sub-shots silent
    assert drama.h3_prompt(screenplay.shots[0], screenplay).duration_s == pytest.approx(158 / 24)

    spoken = drama.h3_prompt(screenplay.shots[2], screenplay).render()
    assert "<d>[Mandarin Chinese] 明天就要收了嗎?</d>" in spoken
    assert "[Shot 2]" not in spoken  # the conflict beat is one held shot


def test_subshots_off_folds_the_second_action_into_one_held_shot() -> None:
    screenplay = _screenplay()
    prompt = drama.h3_prompt(screenplay.shots[0], screenplay, subshots=False)
    assert len(prompt.shots) == 1
    assert "then the lead does thing 1.2" in prompt.shots[0].description


def test_status_payload_carries_shots_the_page_already_knows_how_to_render() -> None:
    screenplay = _screenplay()
    payload = drama.screenplay_payload(screenplay, "llm")
    assert payload["_built_by"] == "llm"
    assert [s["index"] for s in payload["shots"]] == [1, 2, 3, 4, 5, 6]
    assert payload["shots"][0]["description"].startswith("[hook 6.6s] [close-up]")
    assert Screenplay.model_validate(payload["screenplay"]) == screenplay


def test_the_older_one_framing_reply_shape_still_parses_for_one_sub_shot_beats() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    world = WorldBible.model_validate(OUTLINE["world"])
    old = {"shots": [{"index": 3, "scene": "s", "framing": "medium", "action": "the lead nods",
                      "dialogue": [{"speaker_id": "S1", "text": "嗯。"}]}]}
    [shot] = drama.build_shots(old, expected=[3], anchor=anchor, world=world)
    assert shot.dialogue[0].text == "嗯。" and shot.framing is Framing.MEDIUM


# --------------------------------------------------------------------- refusals


async def test_no_client_means_no_drama_not_a_template() -> None:
    with pytest.raises(drama.ScreenplayError, match="gpt-oss"):
        await drama.write_screenplay("x", None)


async def test_a_model_that_keeps_returning_garbage_fails_loudly() -> None:
    client = _client({"nope": 1}, {"nope": 2})
    with pytest.raises(drama.ScreenplayError, match="after 2 attempts"):
        await drama.write_screenplay("x", client)


async def test_a_missing_beat_is_rejected() -> None:
    beats = dict(OUTLINE["beats"])
    del beats["turn"]
    bad = dict(OUTLINE, beats=beats)
    client = _client(bad, bad)
    with pytest.raises(drama.ScreenplayError, match="'turn' is missing"):
        await drama.write_screenplay("x", client)


def test_an_unknown_framing_or_wrong_sub_shot_count_is_a_screenplay_error() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    world = WorldBible.model_validate(OUTLINE["world"])
    bad = _shots([1, 2, 3])
    bad["shots"][0]["sub_shots"][0]["framing"] = "extreme close-up"
    with pytest.raises(drama.ScreenplayError, match="unknown framing"):
        drama.build_shots(bad, expected=[1, 2, 3], anchor=anchor, world=world)
    short = _shots([1, 2, 3])
    short["shots"][0]["sub_shots"].pop()
    with pytest.raises(drama.ScreenplayError, match="needs 2 sub-shot"):
        drama.build_shots(short, expected=[1, 2, 3], anchor=anchor, world=world)


def test_a_repeated_framing_across_the_seam_does_not_validate() -> None:
    second = _shots([4, 5, 6])
    second["shots"][0]["sub_shots"][0]["framing"] = "over-the-shoulder"  # shot 3 ended on it
    with pytest.raises(ValidationError, match="consecutive framings must differ"):
        _screenplay(shots_2=second)


def test_a_second_push_in_or_one_off_the_turn_does_not_validate() -> None:
    first = _shots([1, 2, 3])
    first["shots"][0]["sub_shots"][1]["camera"] = {"motion": "push_in"}
    with pytest.raises(ValidationError, match="one push-in per drama"):
        _screenplay(shots_1=first)
    second = _shots([4, 5, 6])
    second["shots"][0]["sub_shots"][1]["camera"] = {"motion": "static_shot"}
    second["shots"][1]["sub_shots"][0]["camera"] = {"motion": "push_in"}
    with pytest.raises(ValidationError, match="belongs to the turn"):
        _screenplay(shots_2=second)


def test_a_screenplay_whose_keyframe_lost_the_anchor_does_not_validate() -> None:
    good = _screenplay()
    drifted = good.shots[3].model_copy(update={"keyframe_prompt": "a woman with short hair, wide shot"})
    shots = (*good.shots[:3], drifted, *good.shots[4:])
    with pytest.raises(ValidationError, match="does not contain the anchor"):
        Screenplay(title="t", logline="l", anchor=good.anchor, world=good.world, shots=shots, overall_soundscape="q")


def test_a_shot_off_its_template_slot_does_not_validate() -> None:
    sub = SubShot(index=1, framing=Framing.MEDIUM, action="the lead waits")
    sub2 = SubShot(index=2, framing=Framing.WIDE, action="the lead leaves")
    with pytest.raises(ValidationError, match="must be 158 frames"):
        DramaShot(index=1, beat=Beat.HOOK, frames=243, scene="s", sub_shots=(sub, sub2), keyframe_prompt="k")
    with pytest.raises(ValidationError, match="has 2 sub-shot"):
        DramaShot(index=1, beat=Beat.HOOK, frames=158, scene="s", sub_shots=(sub,), keyframe_prompt="k")
    with pytest.raises(ValidationError, match="is the hook"):
        DramaShot(index=1, beat=Beat.TURN, frames=158, scene="s", sub_shots=(sub, sub2), keyframe_prompt="k")


def test_shot_indices_must_be_one_to_six() -> None:
    good = _screenplay()
    with pytest.raises(ValidationError, match="exactly 6 shots"):
        Screenplay(title="t", logline="l", anchor=good.anchor, world=good.world, shots=good.shots[:1], overall_soundscape="q")


def test_the_character_sheet_stands_in_the_world_not_a_studio() -> None:
    screenplay = _screenplay()
    sheets = drama.character_sheet_prompts(screenplay.anchor, screenplay.world)
    assert set(sheets) == {"front", "three_quarter"}
    for prompt in sheets.values():
        assert prompt.startswith(APPEARANCE) and screenplay.world.prefix() in prompt
        assert "studio" not in prompt


def test_no_slot_is_above_the_measured_frame_ceiling() -> None:
    from fun_workflow.core.drama_spec import MAX_SHOT_FRAMES

    assert MAX_SHOT_FRAMES == 243
    assert all(slot.frames <= MAX_SHOT_FRAMES for slot in BEAT_TEMPLATE)


def test_a_sub_shot_without_the_lead_is_a_screenplay_error() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    world = WorldBible.model_validate(OUTLINE["world"])
    bad = _shots([4, 5, 6])
    bad["shots"][2]["sub_shots"][0]["action"] = "the phone buzzes with a new message, screen off"
    with pytest.raises(drama.ScreenplayError, match="the lead must be in the action"):
        drama.build_shots(bad, expected=[4, 5, 6], anchor=anchor, world=world)


def test_beats_must_cover_all_six_or_be_left_empty() -> None:
    good = _screenplay()
    with pytest.raises(ValidationError, match="missing"):
        Screenplay(
            title="t", logline="l", anchor=good.anchor, world=good.world, shots=good.shots,
            overall_soundscape="q", beats={Beat.HOOK: "only one"},
        )
    with pytest.raises(ValidationError, match="non-empty"):
        Screenplay(
            title="t", logline="l", anchor=good.anchor, world=good.world, shots=good.shots,
            overall_soundscape="q", beats={b: "" if b is Beat.HOOK else "x" for b in Beat},
        )
    # An empty dict (the pre-this-change shape) still constructs -- backward compatible.
    ok = Screenplay(
        title="t", logline="l", anchor=good.anchor, world=good.world, shots=good.shots, overall_soundscape="q",
    )
    assert ok.beats == {}


async def test_screenplay_payload_carries_the_beats_forward() -> None:
    screenplay, how = await drama.write_screenplay("x", _good_client())
    payload = drama.screenplay_payload(screenplay, how)
    assert payload["screenplay"]["beats"]["turn"] == "turn beat"
    assert Screenplay.model_validate(payload["screenplay"]).beats == screenplay.beats
