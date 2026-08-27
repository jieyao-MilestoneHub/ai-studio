"""Colloquial Chinese → a validated MiniMax H3 prompt.

The measured reason this module exists: holding seed, resolution and scene
constant and changing only the prompt moved the quality score from 26.0 (free
prose) to 367.6 (the official structured schema). Holding the prose constant and
rendering at five times the pixels changed nothing. [reported]

So a LINE user's "一隻橘毛走在下雨天的路上" has to become a schema-conformant
prompt before it is worth spending a GPU-minute on.

**The LLM never produces the final string.** It produces JSON, which is validated
into `H3Prompt` — so every rule in `prompts/h3.py` (Shot 1 must declare a style,
cut times must strictly increase and fall inside the clip, camera motion comes
from a closed vocabulary, speaker ids must match a pattern) stands between a
hallucination and a submitted job. A model that invents `"camera": "swooping
dramatically"` fails validation rather than producing a prompt the model cannot
follow.

This module is pure: it takes a client conforming to `LlmClient` and does no I/O
of its own, so it stays in L1 and is testable with no network.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

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
)

MAX_SHOTS = 4
"""More than this in a five-second clip is unwatchable, whatever the model says."""


class LlmClient(Protocol):
    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str: ...


SYSTEM_PROMPT = """\
You convert a casual description of a video (often Traditional Chinese) into a
structured shot plan for the MiniMax H3 video model. Reply with JSON only, no
prose and no code fence.

Schema:
{
  "shots": [
    {
      "style": "<REQUIRED on the first shot only: overall visual style, e.g. \
'Live-action, cinematic' or '2D-animated'>",
      "cut_at_s": <omit for the first shot; for later shots the second at which \
the cut happens, strictly increasing, strictly less than the total duration>,
      "description": "<English. What is visible: subject, appearance, action, \
setting, light, notable objects. Concrete and observable, never abstract mood.>",
      "camera": {
        "motion": "<one of: zoom_in zoom_out push_in pull_out pan_left pan_right \
truck_left truck_right tilt_up tilt_down pedestal_up pedestal_down arc_shot \
tracking_shot static_shot shake_slightly shake_strongly pov roll_cw roll_ccw>",
        "amplitude": "<small|medium|large>",
        "speed": "<slow|normal|fast>",
        "toward": "<optional: what the camera moves toward>"
      },
      "dialogue": [
        {"speaker_id": "S1",
         "identity": "<who they are and how they sound>",
         "language": "English",
         "text": "<the exact spoken words>"}
      ]
    }
  ],
  "overall_soundscape": "<1-4 English sentences: ambience, physical action \
sounds, non-verbal human sounds. No dialogue, no non-diegetic music.>",
  "non_diegetic_music": "<1-3 English sentences on instrumentation, tempo, \
rhythm and dynamics, or exactly N/A. Never abstract mood words.>"
}

Rules:
- The request is the spec. Translate it faithfully; do not embellish, do not
  "improve", do not add subjects, props, settings or moods it did not name.
  Every concrete word the user used (objects, colours, clothing, styles,
  places, names, numbers) must survive into the description as its plain
  English equivalent. Proper nouns and on-screen text stay verbatim, in
  double quotes. If the request is thin, the description is short -- never
  padded with invented detail.
- Give the clip ONE job: one subject doing one clearly visible thing. If the
  request lists several ideas, keep the first as the action and fold the rest
  in as setting or a second shot -- never a montage.
- Lead with the ACTION. The first sentence of every description names the
  subject and its physical action with a motion quality ("walks slowly, each
  paw placed deliberately", "turns her head in one quick snap"). Static
  adjectives come after the verb, never before it.
- Name the shot type in words (wide shot / medium shot / close-up / low angle
  / over-the-shoulder). Never write bare praise: no "cinematic", "stunning",
  "high quality", "masterpiece", "8k".
- Exactly ONE camera move per shot, from the closed vocabulary. Never combine
  moves ("orbit while pushing in"). static_shot is a good default.
- Restate the same subject in every shot (same person, same clothes, same
  colours) so the face and outfit do not change across a cut.
- At most __MAX_SHOTS__ shots; for clips of 10 seconds or less at most 2
  cuts. One shot is fine and often better. The first shot has no cut_at_s;
  later ones strictly increase and stay strictly below the total duration.
  A style change part-way is a second shot at a stated time.
- Audio is rendered in the same pass, so direct it explicitly. Name the
  concrete sounds (footsteps on gravel, rain on a tin roof, a kettle
  whistling). Omit dialogue unless the request implies speech; when a person
  is on screen and silent, say "not speaking, lips closed" and give them an
  action. If nobody speaks, end overall_soundscape with "No dialogue."
- non_diegetic_music: instrumentation, tempo and dynamics tied to events
  ("solo piano, slow, enters when the door opens") -- or exactly N/A. Never
  a mood word (no "melancholic", "epic", "emotional").
- Carry the user's 「不要…」 list, and always finish the LAST shot's
  description with the negatives as one sentence: "No text, no subtitles, no
  logos, no watermark, no extra people." There is no separate negative
  prompt; the description is the only place it can go.
- description and every audio field are English, even when the request is
  Chinese.

- Output ONE line of minified JSON: no line breaks, no indentation, no
  comments, nothing before the opening brace or after the closing one.

Example -- request: 「一隻橘貓走在下雨的路上,然後變成梵谷風格的像素貓」,
total duration 10.12 seconds:
{"shots":[{"style":"Live-action, cinematic","description":"Medium tracking shot: an orange tabby cat walks slowly along a wet asphalt road in steady rain, each paw placed deliberately, fur damp and flattened, streetlights reflecting in puddles. Not speaking. No people.","camera":{"motion":"tracking_shot","amplitude":"small","speed":"slow","toward":"the cat"}},{"cut_at_s":6.0,"description":"Close-up: the same orange tabby cat, now rendered as pixel art in the style of a Van Gogh painting with thick swirling brushstroke pixels, keeps walking in the same rain. No text, no subtitles, no logos, no watermark, no extra people.","camera":{"motion":"static_shot","amplitude":"small","speed":"normal"}}],"overall_soundscape":"Steady rain hisses on asphalt and taps on distant rooftops; soft wet paw steps; a car passes far away. No dialogue.","non_diegetic_music":"N/A"}
""".replace("__MAX_SHOTS__", str(MAX_SHOTS))

I2V_BRIEF = """\
A reference photo is supplied as Picture 1 and it IS the first frame of the
video: it already fixes the subject, setting, framing and style. Open the
first shot by stating that role -- "Picture 1 is the opening frame; keep the
same person, clothing, setting and framing" -- and then describe only what
happens next: the action and camera, not a re-description of the picture.
Do not invent a different scene, subject or setting: the request is about
this photo. If the request names a style change (e.g. "become an oil
painting"), describe the picture's own content transforming into that style
over the clip, starting from the photo exactly as it is. Nobody speaks unless
the request says so. The picture already answers "what is in frame"; spend
the words on the action and its motion quality."""
"""What the model is told on top of SYSTEM_PROMPT for image-to-video. Without
it, "變油畫風格" against a portrait came back as "a traditional Chinese painting
of a bustling market" -- a perfectly good shot plan for a scene that was not
in the picture H3 was then told to animate. Observed live, 2026-08-26."""

_MOTIONS = {m.name.lower(): m for m in CameraMotion}
_AMPLITUDES = {"small": Amplitude.SMALL, "medium": Amplitude.MEDIUM, "large": Amplitude.LARGE}
_SPEEDS = {"slow": Speed.SLOW, "normal": Speed.NORMAL, "fast": Speed.FAST}


class ConversionError(Exception):
    """The model's JSON could not be turned into a valid prompt."""


# --------------------------------------------------------------------- parsing


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply.

    Models add code fences and commentary however firmly you ask them not to,
    so this takes the outermost braces rather than trusting the whole string.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ConversionError(f"no JSON object in the reply: {raw[:200]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ConversionError(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConversionError("top-level JSON was not an object")
    return payload


def _camera(spec: Any) -> str | None:
    """Render a camera block into prose, or None if it is unusable.

    An unknown motion word is dropped rather than raising: losing the camera
    move still leaves a usable prompt, and the closed vocabulary is exactly the
    kind of thing a model gets almost right.
    """
    if not isinstance(spec, dict):
        return None
    motion = _MOTIONS.get(str(spec.get("motion", "")).strip().lower())
    if motion is None:
        return None
    return camera_phrase(
        motion,
        _AMPLITUDES.get(str(spec.get("amplitude", "")).lower(), Amplitude.MEDIUM),
        _SPEEDS.get(str(spec.get("speed", "")).lower(), Speed.NORMAL),
        toward=str(spec["toward"]).strip() if spec.get("toward") else None,
    )


def _dialogue(items: Any) -> tuple[Dialogue, ...]:
    lines: list[Dialogue] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        try:
            lines.append(
                Dialogue(
                    speaker_id=str(item.get("speaker_id") or "S1"),
                    identity=str(item.get("identity") or "A speaker"),
                    language=str(item.get("language") or "English"),
                    text=str(item["text"]),
                )
            )
        except ValidationError:
            continue  # a malformed speaker id loses the line, not the clip
    return tuple(lines)


def build_prompt(
    payload: dict[str, Any], duration_s: float, *, mode: H3Mode = H3Mode.T2VA
) -> H3Prompt:
    """Turn model JSON into an `H3Prompt`. Raises on anything unusable."""
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ConversionError("no shots in the reply")

    shots: list[PromptShot] = []
    for index, spec in enumerate(raw_shots[:MAX_SHOTS], start=1):
        if not isinstance(spec, dict):
            raise ConversionError(f"shot {index} was not an object")
        description = str(spec.get("description") or "").strip()
        if not description:
            raise ConversionError(f"shot {index} has no description")

        cut = None if index == 1 else spec.get("cut_at_s")
        shots.append(
            PromptShot(
                index=index,
                cut_at_s=None if index == 1 else float(cut) if cut is not None else None,
                style=str(spec.get("style") or "Live-action, cinematic") if index == 1 else None,
                description=description,
                camera=_camera(spec.get("camera")),
                dialogue=_dialogue(spec.get("dialogue")),
            )
        )

    try:
        return H3Prompt(
            mode=mode,
            duration_s=duration_s,
            shots=tuple(shots),
            overall_soundscape=str(payload.get("overall_soundscape") or "").strip()
            or "Quiet ambient room tone with faint distant movement.",
            non_diegetic_music=str(payload.get("non_diegetic_music") or "N/A").strip(),
        )
    except ValidationError as exc:
        # The h3 module's own rules rejected it — a cut outside the clip, a
        # non-increasing cut, a first shot without a style. This is the guard
        # doing its job.
        raise ConversionError(f"failed H3 schema validation: {exc}") from exc


# -------------------------------------------------------------------- fallback


def template_prompt(text: str, duration_s: float, *, mode: H3Mode = H3Mode.T2VA) -> H3Prompt:
    """A single-shot prompt built without a model.

    Used when the LLM is unavailable or keeps producing invalid JSON. The result
    is worse than a real conversion — closer to the 26.0 free-prose score than
    the 367.6 structured one — but a mediocre clip beats a dropped request, and
    it keeps the whole pipeline runnable with no LLM at all.
    """
    return H3Prompt(
        mode=mode,
        duration_s=duration_s,
        shots=(
            PromptShot(
                index=1,
                style="Live-action, cinematic",
                description=f"a medium shot showing {text.strip().rstrip('.')}",
                camera=camera_phrase(CameraMotion.STATIC_SHOT),
            ),
        ),
        overall_soundscape=(
            "Ambient sound appropriate to the scene continues throughout, with "
            "the subject's own movement audible over it."
        ),
        non_diegetic_music="N/A",
    )


# --------------------------------------------------------------------- convert


async def convert(
    text: str,
    client: LlmClient | None,
    *,
    duration_s: float,
    mode: H3Mode = H3Mode.T2VA,
    attempts: int = 2,
) -> tuple[H3Prompt, str]:
    """Convert `text` into an `H3Prompt`. Returns `(prompt, how)`.

    `mode=H3Mode.I2VA` means a photo will be the first frame: the model is
    briefed to describe *that* picture moving, not to invent a scene.

    `how` is "llm", "llm-retry" or "template", and is recorded so a run's quality
    can be traced back to how its prompt was built. Never raises: a request that
    cannot be converted still gets a template prompt rather than being dropped.
    """
    if client is None:
        return template_prompt(text, duration_s, mode=mode), "template"

    user = (
        (I2V_BRIEF + "\n\n" if mode is H3Mode.I2VA else "")
        + f"Total duration: {duration_s:.2f} seconds.\n"
        f"Request: {text.strip()}\n"
        "Reply with the JSON object only."
    )

    last_error = ""
    for attempt in range(attempts):
        try:
            reply = await client.complete(SYSTEM_PROMPT, user)
            prompt = build_prompt(_extract_json(reply), duration_s, mode=mode)
            return prompt, "llm" if attempt == 0 else "llm-retry"
        except (ConversionError, ValidationError) as exc:
            last_error = str(exc)
        except Exception as exc:  # network, timeout, anything at all
            last_error = f"{type(exc).__name__}: {exc}"

    return template_prompt(text, duration_s, mode=mode), f"template (llm failed: {last_error[:200]})"
