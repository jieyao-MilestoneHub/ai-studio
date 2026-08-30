"""`render.captions_ass`: cue windows come from the timeline, the title card
is the only typed time, unknown colours raise, and the text is valid ASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_studio.core.enums import CaptionKind, TransitionReason
from ai_studio.core.errors import UnknownKeyError
from ai_studio.core.models import CaptionCue, Segment
from ai_studio.editing import transitions as tr
from ai_studio.render import captions_ass as ass
from ai_studio.render.timeline import resolve_timeline


def _timeline():
    segs = [Segment(segment_id=f"sg{i}", shot_id=f"sh{i}", scene_id="sc", subcut_index=0, intended_duration_s=d)
            for i, d in enumerate((2.5, 4.0, 3.0))]
    return resolve_timeline(segs, {"sg0": "1", "sg1": "1", "sg2": "2"}, [tr.choose(TransitionReason.DEFAULT)])


def test_cues_are_windowed_to_their_segment_and_never_straddle_a_cut() -> None:
    t = _timeline()
    cues = [CaptionCue(cue_id="c1", segment_id="sg1", text="沒事,明天照常開。"),
            CaptionCue(cue_id="c2", segment_id="sg2", text="我不走。")]
    events = ass.cue_events(cues, t)
    assert events[0].start_s == pytest.approx(2.6) and events[0].end_s == pytest.approx(6.4)
    assert events[1].start_s == pytest.approx(6.6) and events[1].end_s == pytest.approx(9.4)
    assert events[0].text == "沒事,明天照常開。" and events[0].style == "Main"


def test_the_title_card_owns_the_first_1_5_seconds_with_a_fade() -> None:
    event = ass.title_card("夜市的信 🎬")
    assert (event.start_s, event.end_s, event.style) == (0.0, 1.5, "Title")
    assert event.text == "夜市的信" and event.tags == "\\fad(0,300)"
    assert event.line().startswith("Dialogue: 0,0:00:00.00,0:00:01.50,Title,,0,0,0,,{\\fad(0,300)}夜市的信")
    with pytest.raises(ValueError, match="needs a title"):
        ass.title_card("🎬")


def test_an_unknown_colour_key_or_a_missing_segment_raises() -> None:
    t = _timeline()
    with pytest.raises(UnknownKeyError):
        ass.cue_events([CaptionCue(cue_id="c", segment_id="sg0", text="x", color_key="gold")], t)
    with pytest.raises(KeyError):
        ass.cue_events([CaptionCue(cue_id="c", segment_id="nope", text="x")], t)
    with pytest.raises(UnknownKeyError, match="caption kind for ASS style"):
        ass.cue_events([CaptionCue(cue_id="c", segment_id="sg1", text="x", kind=CaptionKind.HOOK)], t)


def test_narration_cues_use_the_sub_style_dialogue_uses_main() -> None:
    t = _timeline()
    cues = [
        CaptionCue(cue_id="c1", segment_id="sg1", text="好。", kind=CaptionKind.MAIN),
        CaptionCue(cue_id="c2", segment_id="sg2", text="稍後。", kind=CaptionKind.SUB),
    ]
    main, sub = ass.cue_events(cues, t)
    assert main.style == "Main" and sub.style == "Sub"


def test_long_lines_break_and_braces_are_escaped() -> None:
    t = _timeline()
    [event] = ass.cue_events([CaptionCue(cue_id="c", segment_id="sg1", text="我不走,這攤是我爸留下的,明天{照常}開。")], t)
    assert "\\N" in event.text and "{" not in event.text


def test_render_is_a_valid_ass_document(tmp_path: Path) -> None:
    t = _timeline()
    styles = ass.default_styles("Noto Sans CJK TC", (864, 480))
    events = [ass.title_card("t"), *ass.cue_events([CaptionCue(cue_id="c", segment_id="sg1", text="好。")], t)]
    path = ass.write(tmp_path / "captions.ass", ass.AssDocument(play_res=(864, 480), styles=styles, events=tuple(events)))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("[Script Info]\nScriptType: v4.00+\nPlayResX: 864\nPlayResY: 480")
    assert "Style: Main,Noto Sans CJK TC,31,&H00FFFFFF" in text
    assert "Style: Sub,Noto Sans CJK TC,26,&H00FFFFFF" in text and ",8,20,20," in text  # top-aligned
    assert "Style: Title,Noto Sans CJK TC,48,&H00FFFFFF" in text and ",3,6,0,5," in text  # boxed, centred
    assert text.count("Dialogue:") == 2
    assert ass._time(3661.239) == "1:01:01.24"


def test_marker_event_places_a_short_caption_anywhere_on_the_timeline() -> None:
    event = ass.marker_event("稍後", start_s=12.5, duration_s=1.0)
    assert (event.start_s, event.end_s, event.style) == (12.5, 13.5, "Sub")
    assert event.text == "稍後" and event.tags == "\\fad(0,200)"
    with pytest.raises(ValueError, match="needs text"):
        ass.marker_event("🎬", start_s=0.0, duration_s=1.0)
    with pytest.raises(ValueError, match="positive duration"):
        ass.marker_event("x", start_s=0.0, duration_s=0.0)


def test_start_offsets_reserve_time_for_a_marker_before_the_cue() -> None:
    t = _timeline()
    cues = [CaptionCue(cue_id="c", segment_id="sg1", text="好。")]
    plain = ass.cue_events(cues, t)[0]
    reserved = ass.cue_events(cues, t, start_offsets={"sg1": 1.0})[0]
    assert reserved.start_s == pytest.approx(plain.start_s + 1.0)
    assert reserved.end_s == plain.end_s

    # A reservation that swallows the whole segment still raises loudly.
    with pytest.raises(ValueError, match="cannot hold caption"):
        ass.cue_events(cues, t, start_offsets={"sg1": 10.0})
