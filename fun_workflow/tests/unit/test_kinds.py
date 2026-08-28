"""`core.kinds.JobKind`: the queue's discriminator, mapped onto ai-studio's
model kinds."""

from __future__ import annotations

from ai_studio.core.enums import MediaKind

from fun_workflow.core.kinds import JobKind


def test_values_are_the_strings_the_queue_stores() -> None:
    assert {k.value for k in JobKind} == {
        "video", "image", "image_understand", "audio_understand", "video_understand", "chat", "drama",
    }


def test_every_kind_but_drama_maps_onto_a_model_kind() -> None:
    assert JobKind.DRAMA.model_kind is None
    for kind in JobKind:
        if kind is not JobKind.DRAMA:
            assert kind.model_kind is MediaKind(kind.value)
    assert not hasattr(MediaKind, "DRAMA"), "a drama is a pipeline, not a model ai-studio serves"


def test_the_two_sides_of_the_card() -> None:
    assert {k for k in JobKind if k.is_generation} == {JobKind.VIDEO, JobKind.IMAGE, JobKind.DRAMA}
    assert {k for k in JobKind if k.is_understanding} == {
        JobKind.IMAGE_UNDERSTAND, JobKind.AUDIO_UNDERSTAND, JobKind.VIDEO_UNDERSTAND,
    }
