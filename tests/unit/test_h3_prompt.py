"""The prompt builder must reproduce the official schema exactly.

Assertions here are quoted from
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_studio.prompts.h3 import (
    Amplitude,
    CameraMotion,
    Dialogue,
    H3Mode,
    H3Prompt,
    PromptShot,
    Speed,
    camera_phrase,
    format_cut_time,
)


def _shot1(**kw: object) -> PromptShot:
    base: dict[str, object] = {
        "index": 1,
        "style": "Live-action, cinematic",
        "description": "a medium-wide shot frames a baker opening the shutters",
    }
    base.update(kw)
    return PromptShot(**base)  # type: ignore[arg-type]


def _prompt(**kw: object) -> H3Prompt:
    base: dict[str, object] = {
        "duration_s": 5.0,
        "shots": (_shot1(),),
        "overall_soundscape": "Wooden shutters scrape open over a quiet street.",
    }
    base.update(kw)
    return H3Prompt(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ structure


def test_t2va_has_no_instruction_line() -> None:
    text = _prompt().render()
    assert text.startswith("integrated_multimodal_description:")


def test_the_three_core_fields_are_present_and_blank_line_separated() -> None:
    text = _prompt().render()
    assert "\n\nintegrated_multimodal_description" not in text  # it is first
    assert "\n\noverall_soundscape: " in text
    assert "\n\nnon_diegetic_music: " in text


def test_keyframe_modes_put_the_instruction_first_then_one_blank_line() -> None:
    for mode in (H3Mode.I2VA, H3Mode.FL2VA, H3Mode.L2VA):
        text = _prompt(mode=mode).render()
        head, blank, rest = text.split("\n", 2)
        assert head and blank == ""
        assert rest.startswith("integrated_multimodal_description:")


def test_fl2va_instruction_states_duration_to_two_decimals() -> None:
    text = _prompt(mode=H3Mode.FL2VA, duration_s=8.0).instruction()
    assert "8.00-second mark" in text
    assert "Picture 1 (from Shot 1) aligns with the 0.00-second mark" in text


def test_i2va_instruction_is_the_exact_official_sentence() -> None:
    assert _prompt(mode=H3Mode.I2VA).instruction() == (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )


# ---------------------------------------------------------------- shot rules


def test_shot_one_must_not_carry_a_timestamp() -> None:
    with pytest.raises(ValidationError, match="must not carry a timestamp"):
        PromptShot(index=1, cut_at_s=0.0, style="Cinematic", description="x")


def test_shot_one_must_declare_a_style() -> None:
    with pytest.raises(ValidationError, match="overall style"):
        PromptShot(index=1, description="x")


def test_later_shots_require_a_cut_time() -> None:
    with pytest.raises(ValidationError, match="needs a cut time"):
        PromptShot(index=2, description="x")


def test_cut_times_must_strictly_increase() -> None:
    shots = (
        _shot1(),
        PromptShot(index=2, cut_at_s=3.0, description="a close-up"),
        PromptShot(index=3, cut_at_s=3.0, description="another"),
    )
    with pytest.raises(ValidationError, match="not strictly after"):
        _prompt(shots=shots)


def test_cut_times_must_fall_inside_the_video() -> None:
    shots = (_shot1(), PromptShot(index=2, cut_at_s=9.0, description="late"))
    with pytest.raises(ValidationError, match=r"outside the 5\.0s video"):
        _prompt(shots=shots)


def test_cut_timestamp_format_matches_the_guide() -> None:
    assert format_cut_time(3.5) == "00:03.500"
    assert format_cut_time(65.25) == "01:05.250"


# ------------------------------------------------------------------- camera


def test_camera_motion_reads_as_prose_not_stacked_labels() -> None:
    assert camera_phrase(
        CameraMotion.PUSH_IN, Amplitude.SMALL, Speed.SLOW, toward="the folded letter"
    ) == ("The camera pushes in with small amplitude at slow speed toward the folded letter.")


def test_medium_amplitude_and_normal_speed_are_omitted() -> None:
    phrase = camera_phrase(CameraMotion.STATIC_SHOT)
    assert phrase == "The camera holds a static shot."
    assert "amplitude" not in phrase and "speed" not in phrase


# ----------------------------------------------------------------- dialogue


def test_dialogue_keeps_identity_outside_the_tag_and_words_inside() -> None:
    line = Dialogue(
        speaker_id="S1",
        identity="The young woman with a quiet, breathy voice",
        text="I get off at the next station.",
    ).render()
    assert line.endswith("<d>[English] I get off at the next station.</d>")
    assert line.startswith("The young woman with a quiet, breathy voice (S1) says:")


def test_voiceover_uses_the_exact_phrase_and_notes_closed_lips() -> None:
    line = Dialogue(
        speaker_id="S1", identity="The man", text="I still remember that road.", voiceover=True
    ).render()
    assert "says in an off-screen voiceover" in line
    assert "lips remain completely closed" in line


def test_compound_speaker_ids_are_accepted_and_junk_is_not() -> None:
    Dialogue(speaker_id="S1,S2", identity="The two children", text="Wait for us!")
    with pytest.raises(ValidationError):
        Dialogue(speaker_id="narrator", identity="x", text="y")
