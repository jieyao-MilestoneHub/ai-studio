"""The screenwriter: three model calls in, one validated `Screenplay` out.

What is asserted is what protects the money: the anchor cannot be dropped,
the shot count cannot drift, and there is no silent fallback that would send
six clips of nothing to the GPU.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_studio.core.drama_spec import SHOT_COUNT, CharacterAnchor, DramaShot, Screenplay
from ai_studio.llm.scripted import ScriptedLlmClient
from ai_studio.prompts import drama
from ai_studio.prompts.h3 import I2VA_INSTRUCTION

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
    "beats": [f"beat {i}" for i in range(1, SHOT_COUNT + 1)],
    "overall_soundscape": "Sizzling oil, a crowd murmuring, scooters passing.",
    "non_diegetic_music": "N/A",
}


def _shots(indices: list[int]) -> dict:
    framings = ["medium", "close-up", "over-the-shoulder", "wide", "medium close-up", "close-up"]
    return {
        "shots": [
            {
                "index": i,
                "scene": f"the night market stall, string lights, steam, beat {i}",
                "framing": framings[i - 1],
                "action": f"the lead does thing {i}",
                "camera": {"motion": "push_in", "amplitude": "small", "speed": "slow"},
                **({"dialogue": [{"speaker_id": "S1", "identity": "the lead", "language": "Mandarin Chinese", "text": "明天就要收了嗎?"}]} if i == 3 else {}),
                "cut_reason": "time_passing" if i == 4 else "default",
            }
            for i in indices
        ]
    }


def _client(*replies: dict) -> ScriptedLlmClient:
    return ScriptedLlmClient(*(json.dumps(r, ensure_ascii=False) for r in replies))


def _good_client() -> ScriptedLlmClient:
    return _client(OUTLINE, _shots([1, 2, 3]), _shots([4, 5, 6]))


# ------------------------------------------------------------------ happy path


async def test_three_calls_make_a_six_shot_screenplay() -> None:
    client = _good_client()
    screenplay, how = await drama.write_screenplay("夜市老闆娘發現一封信", client)

    assert how == "llm"
    assert len(client.calls) == 3
    assert [s.index for s in screenplay.shots] == list(range(1, SHOT_COUNT + 1))
    assert screenplay.anchor.appearance == APPEARANCE
    assert screenplay.shots[2].dialogue[0].text == "明天就要收了嗎?"
    assert screenplay.shots[3].cut_reason.value == "time_passing"


async def test_the_anchor_is_in_every_keyframe_prompt_verbatim() -> None:
    screenplay, _ = await drama.write_screenplay("x", _good_client())
    for shot in screenplay.shots:
        assert shot.keyframe_prompt.startswith(APPEARANCE)
        assert "faded red apron" in shot.keyframe_prompt


async def test_the_model_never_gets_to_write_the_face() -> None:
    """The shots prompt hands the model the name, not the appearance, and the
    keyframe prompt is composed here -- so a model that ignores the rule and
    describes a different face in `scene` cannot replace the anchor."""
    client = _good_client()
    await drama.write_screenplay("x", client)
    _, shots_user_1 = client.calls[1]
    assert APPEARANCE not in shots_user_1
    assert "阿玲" in shots_user_1


async def test_a_retry_is_recorded_as_llm_retry() -> None:
    client = _client({"garbage": True}, OUTLINE, _shots([1, 2, 3]), _shots([4, 5, 6]))
    _, how = await drama.write_screenplay("x", client)
    assert how == "llm-retry"


def test_h3_prompt_is_image_to_video_at_the_default_length() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    shots = drama.build_shots(_shots([1, 2, 3]), expected=[1, 2, 3], anchor=anchor) + drama.build_shots(
        _shots([4, 5, 6]), expected=[4, 5, 6], anchor=anchor
    )
    screenplay = Screenplay(
        title="t", logline="l", anchor=anchor, shots=tuple(shots),
        overall_soundscape="quiet", non_diegetic_music="N/A",
    )
    rendered = drama.h3_prompt(screenplay.shots[0], screenplay, duration_s=10.125).render()
    assert rendered.startswith(I2VA_INSTRUCTION)
    assert APPEARANCE in rendered
    assert "lips remain closed" in rendered  # silent shot
    spoken = drama.h3_prompt(screenplay.shots[2], screenplay, duration_s=10.125).render()
    assert "<d>[Mandarin Chinese] 明天就要收了嗎?</d>" in spoken


def test_status_payload_carries_shots_the_page_already_knows_how_to_render() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    shots = drama.build_shots(_shots([1, 2, 3]), expected=[1, 2, 3], anchor=anchor) + drama.build_shots(
        _shots([4, 5, 6]), expected=[4, 5, 6], anchor=anchor
    )
    screenplay = Screenplay(
        title="t", logline="l", anchor=anchor, shots=tuple(shots),
        overall_soundscape="quiet",
    )
    payload = drama.screenplay_payload(screenplay, "llm")
    assert payload["_built_by"] == "llm"
    assert [s["index"] for s in payload["shots"]] == [1, 2, 3, 4, 5, 6]
    assert Screenplay.model_validate(payload["screenplay"]) == screenplay


# --------------------------------------------------------------------- refusals


async def test_no_client_means_no_drama_not_a_template() -> None:
    with pytest.raises(drama.ScreenplayError, match="gpt-oss"):
        await drama.write_screenplay("x", None)


async def test_a_model_that_keeps_returning_garbage_fails_loudly() -> None:
    client = _client({"nope": 1}, {"nope": 2})
    with pytest.raises(drama.ScreenplayError, match="after 2 attempts"):
        await drama.write_screenplay("x", client)


async def test_five_beats_are_rejected() -> None:
    bad = dict(OUTLINE, beats=OUTLINE["beats"][:5])
    client = _client(bad, bad)
    with pytest.raises(drama.ScreenplayError, match="exactly 6 beats"):
        await drama.write_screenplay("x", client)


def test_a_screenplay_whose_keyframe_lost_the_anchor_does_not_validate() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    good = drama.build_shots(_shots([1, 2, 3]), expected=[1, 2, 3], anchor=anchor) + drama.build_shots(
        _shots([4, 5, 6]), expected=[4, 5, 6], anchor=anchor
    )
    drifted = good[3].model_copy(update={"keyframe_prompt": "a woman with short hair, wide shot"})
    shots = (*good[:3], drifted, *good[4:])
    with pytest.raises(ValidationError, match="does not contain the anchor"):
        Screenplay(title="t", logline="l", anchor=anchor, shots=shots, overall_soundscape="q")


def test_shot_indices_must_be_one_to_six() -> None:
    anchor = CharacterAnchor.model_validate(OUTLINE["anchor"])
    shot = DramaShot(
        index=1, scene="s", framing="medium", action="a",
        keyframe_prompt=drama.keyframe_prompt(anchor, framing="medium", scene="s"),
    )
    with pytest.raises(ValidationError, match="exactly 6 shots"):
        Screenplay(title="t", logline="l", anchor=anchor, shots=(shot,), overall_soundscape="q")
