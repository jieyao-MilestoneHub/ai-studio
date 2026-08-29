"""`render.timeline.resolve_timeline`: absolute time computed once, on frames,
with dissolves subtracted exactly once and sub-cuts dipping no audio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_studio.core.enums import TransitionKind, TransitionReason
from ai_studio.core.models import Segment
from ai_studio.editing import transitions as tr
from ai_studio.render import timeline as tl


def _seg(i: int, seconds: float) -> Segment:
    return Segment(segment_id=f"sg{i}", shot_id=f"sh{i}", scene_id="sc", subcut_index=0, intended_duration_s=seconds)


def test_segments_lay_end_to_end_on_whole_frames() -> None:
    segs = [_seg(0, 2.5), _seg(1, 4.79), _seg(2, 8.7)]
    clip_of = {"sg0": "1", "sg1": "1", "sg2": "2"}
    t = tl.resolve_timeline(segs, clip_of, [tr.choose(TransitionReason.DEFAULT)])
    assert [s.start_frame for s in t.segments] == [0, 60, 175]
    assert t.segments[1].end_s == pytest.approx(175 / 24)
    assert t.clip_offsets == (0.0, pytest.approx(175 / 24))
    assert t.total_s == pytest.approx((175 + 209) / 24)


def test_a_dissolve_is_subtracted_once_and_only_at_the_clip_boundary() -> None:
    segs = [_seg(0, 5.0), _seg(1, 5.0), _seg(2, 5.0)]
    clip_of = {"sg0": "a", "sg1": "a", "sg2": "b"}
    t = tl.resolve_timeline(segs, clip_of, [tr.choose(TransitionReason.TIME_PASSING)])
    inner, outer = t.boundaries
    assert inner.kind is TransitionKind.HARD_CUT and inner.audio_fade_s == 0.0 and not inner.clip_boundary
    assert outer.kind is TransitionKind.DISSOLVE and outer.overlap_s == 0.5 and outer.clip_boundary
    assert t.total_s == pytest.approx(14.5)
    assert t.segments[2].start_s == pytest.approx(9.5)
    assert t.clip_offsets == (0.0, pytest.approx(9.5))


def test_a_clip_boundary_hard_cut_carries_the_audio_dip() -> None:
    t = tl.resolve_timeline([_seg(0, 3.0), _seg(1, 3.0)], {"sg0": "a", "sg1": "b"}, [tr.choose(TransitionReason.DEFAULT)])
    assert t.boundaries[0].audio_fade_s == tr.AUDIO_DIP_S and t.total_s == 6.0


def test_transition_count_must_match_clip_boundaries() -> None:
    with pytest.raises(ValueError, match="1 clip boundaries but 0"):
        tl.resolve_timeline([_seg(0, 3.0), _seg(1, 3.0)], {"sg0": "a", "sg1": "b"}, [])


def test_an_unrenderable_kind_raises_rather_than_cutting_silently() -> None:
    wipe = tr.choose(TransitionReason.TOPIC_CHANGE, tr.Evidence(occlusion=True))
    with pytest.raises(ValueError, match="cannot render a wipe"):
        tl.resolve_timeline([_seg(0, 3.0), _seg(1, 3.0)], {"sg0": "a", "sg1": "b"}, [wipe])


def test_offsets_json_round_trips(tmp_path: Path) -> None:
    t = tl.resolve_timeline([_seg(0, 2.0), _seg(1, 2.0)], {"sg0": "a", "sg1": "b"}, [tr.choose(TransitionReason.DEFAULT)])
    path = tl.write_offsets(tmp_path, t)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "offsets.json"
    assert data["total_s"] == 4.0 and data["clips"] == ["a", "b"]
    assert data["boundaries"][0]["kind"] == "hard_cut"
    assert t.segment("sg1").start_frame == 48
    with pytest.raises(KeyError):
        t.segment("nope")
