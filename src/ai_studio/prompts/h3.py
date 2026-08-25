"""MiniMax H3 structured prompt builder.

This is the highest-leverage module in the project, and the reason is measured
rather than asserted. Holding seed, resolution, and scene constant and changing
only the prompt moved the quality score from 26.0 (free prose) to 205.9 (more
specific prose) to 367.6 (the official structured schema). Holding the prose
constant and rendering at five times the pixels changed it by nothing:
608x352 scored 29.2, 864x480 scored 30.3, 1344x768 scored 26.0. [reported]

The operational conclusion is blunt: **a blurry result is a prompt problem, not
a resolution problem.** So this module makes it hard to submit unstructured
prose in the first place. You assemble typed fields; the schema is the output,
not something you are trusted to remember.

Schema source, quoted and followed exactly:
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_studio.core.errors import UnknownKeyError

# --------------------------------------------------------------------- modes


class H3Mode(str, Enum):
    """The four task shapes. Each has its own mandatory first line."""

    T2VA = "t2va"
    """Text only. No alignment instruction; starts straight at the core fields."""

    I2VA = "i2va"
    """First frame given; develop forward from it."""

    FL2VA = "fl2va"
    """First and last frame given; describe the path between them."""

    L2VA = "l2va"
    """Last frame given; infer a plausible earlier state that lands on it."""


# ------------------------------------------------------------------- camera


class CameraMotion(str, Enum):
    """The closed camera vocabulary from section 4.3.

    A closed set on purpose. "The camera swoops dramatically" is not in the
    model's training vocabulary; `ARC_SHOT` is. Inventing motion words is the
    quiet way a structured prompt decays back into prose.
    """

    ZOOM_IN = "zooms in"
    ZOOM_OUT = "zooms out"
    PUSH_IN = "pushes in"
    PULL_OUT = "pulls out"
    PAN_LEFT = "pans left"
    PAN_RIGHT = "pans right"
    TRUCK_LEFT = "trucks left"
    TRUCK_RIGHT = "trucks right"
    TILT_UP = "tilts up"
    TILT_DOWN = "tilts down"
    PEDESTAL_UP = "pedestals up"
    PEDESTAL_DOWN = "pedestals down"
    ARC_SHOT = "moves in an arc around the subject"
    TRACKING_SHOT = "follows the subject"
    STATIC_SHOT = "holds a static shot"
    SHAKE_SLIGHTLY = "shakes slightly"
    SHAKE_STRONGLY = "shakes strongly"
    POV = "takes the subject's point of view"
    ROLL_CW = "rolls clockwise"
    ROLL_CCW = "rolls counterclockwise"


class Amplitude(str, Enum):
    SMALL = "with small amplitude"
    MEDIUM = ""
    """Medium is the default and the guide says to omit it."""
    LARGE = "with large amplitude"


class Speed(str, Enum):
    SLOW = "at slow speed"
    NORMAL = ""
    """Normal is the default and the guide says to omit it."""
    FAST = "at fast speed"


STYLES: frozenset[str] = frozenset(
    {
        "Cinematic",
        "live-action",
        "2D-animated",
        "3D CG",
        "claymation",
        "watercolor",
        "vintage film",
    }
)
"""The styles named in section 4.1. Other values are allowed but warned about."""


def camera_phrase(
    motion: CameraMotion,
    amplitude: Amplitude = Amplitude.MEDIUM,
    speed: Speed = Speed.NORMAL,
    *,
    toward: str | None = None,
) -> str:
    """Render camera motion as a natural sentence, not stacked labels.

    The guide is explicit that motion belongs in the prose as an action, not
    appended as tags:

    >>> camera_phrase(CameraMotion.PUSH_IN, Amplitude.SMALL, Speed.SLOW,
    ...               toward="the folded letter in her hands")
    'The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.'
    """
    parts = ["The camera", motion.value]
    if amplitude.value:
        parts.append(amplitude.value)
    if speed.value:
        parts.append(speed.value)
    if toward:
        parts.append(f"toward {toward}" if motion is not CameraMotion.PAN_RIGHT else toward)
    return " ".join(parts).rstrip() + "."


# ----------------------------------------------------------------- dialogue


class Dialogue(BaseModel):
    """One spoken or sung line.

    The guide's split is strict and easy to get wrong: identity, action, and
    delivery go *outside* `<d>`; only the language tag and the verbatim words go
    inside. Original wording and punctuation are preserved — never translated,
    never tidied.
    """

    model_config = ConfigDict(frozen=True)

    speaker_id: str = Field(pattern=r"^S\d+(,S\d+)*$", description="e.g. 'S1' or 'S1,S2'")
    identity: str = Field(description="Who they are and how they sound, on first appearance.")
    language: str = "English"
    text: str
    voiceover: bool = False

    def render(self) -> str:
        verb = "says in an off-screen voiceover" if self.voiceover else "says"
        line = f"{self.identity} ({self.speaker_id}) {verb}: <d>[{self.language}] {self.text}</d>"
        if self.voiceover:
            # The guide requires this immediately after every voiceover block.
            line += " while their lips remain completely closed."
        return line


# --------------------------------------------------------------------- shots


class PromptShot(BaseModel):
    """One shot inside `integrated_multimodal_description`.

    Note this is the *prompt's* notion of a shot, distinct from
    `ai_studio.core.models.Shot`, which is a pipeline planning unit. One planned
    Shot maps to one generation call, which may contain several PromptShots.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    cut_at_s: float | None = Field(
        default=None,
        description="None for Shot 1. Required and strictly increasing thereafter.",
    )

    style: str | None = Field(
        default=None,
        description="Required on Shot 1: overall style plus initial composition.",
    )
    description: str
    camera: str | None = None
    dialogue: tuple[Dialogue, ...] = ()

    @model_validator(mode="after")
    def _check_shot(self) -> PromptShot:
        if self.index == 1:
            if self.cut_at_s is not None:
                raise ValueError("Shot 1 must not carry a timestamp (guide 4.2)")
            if not self.style:
                raise ValueError("Shot 1 must state the overall style and composition (guide 4.1)")
        elif self.cut_at_s is None:
            raise ValueError(f"Shot {self.index} needs a cut time (guide 4.2)")
        return self

    def render(self) -> str:
        parts: list[str] = [f"[Shot {self.index}]"]
        if self.cut_at_s is not None:
            parts.append(f"At {format_cut_time(self.cut_at_s)}, the camera cuts to")
        if self.style:
            parts.append(f"{self.style},")
        parts.append(self.description.rstrip("."))
        body = " ".join(parts).rstrip() + "."
        if self.camera:
            body += " " + self.camera
        for line in self.dialogue:
            body += " " + line.render()
        return body


def format_cut_time(seconds: float) -> str:
    """``MM:SS.mmm`` as the guide's examples use (``00:03.500``)."""
    if seconds < 0:
        raise ValueError(f"negative cut time: {seconds}")
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{rem:06.3f}"


# -------------------------------------------------------------------- prompt


class H3Prompt(BaseModel):
    """A complete, schema-conformant MiniMax H3 prompt."""

    model_config = ConfigDict(frozen=True)

    mode: H3Mode = H3Mode.T2VA
    duration_s: float = Field(gt=0)

    shots: tuple[PromptShot, ...]
    overall_soundscape: str = Field(
        description=(
            "1-4 sentences: ambience, physical action sounds, non-verbal human "
            "sounds. Dialogue and diegetic music belong in the shots, not here. "
            "'N/A' only when the user explicitly asked for total silence."
        )
    )
    non_diegetic_music: str = Field(
        default="N/A",
        description=(
            "1-3 sentences on instrumentation, tempo, rhythm, dynamics — never "
            "abstract mood words. Music the characters can hear is diegetic and "
            "belongs in the shots. 'N/A' when there is none."
        ),
    )

    @model_validator(mode="after")
    def _check_prompt(self) -> H3Prompt:
        if not self.shots:
            raise ValueError("a prompt needs at least one shot")
        if self.shots[0].index != 1:
            raise ValueError("shots must start at index 1")

        previous = 0.0
        for shot in self.shots[1:]:
            cut = shot.cut_at_s
            assert cut is not None  # guaranteed by PromptShot validation
            if cut <= previous:
                raise ValueError(
                    f"Shot {shot.index} cuts at {cut}s, not strictly after {previous}s"
                )
            if cut >= self.duration_s:
                raise ValueError(
                    f"Shot {shot.index} cuts at {cut}s, outside the {self.duration_s}s video"
                )
            previous = cut

        indices = [s.index for s in self.shots]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"shot indices must be sequential from 1: {indices}")
        return self

    # -------------------------------------------------------------- rendering

    def instruction(self) -> str:
        """The mandatory first line for keyframe modes. Empty for T2VA."""
        last = self.shots[-1].index
        stamp = f"{self.duration_s:.2f}"
        if self.mode is H3Mode.T2VA:
            return ""
        if self.mode is H3Mode.I2VA:
            return (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        if self.mode is H3Mode.FL2VA:
            return (
                "How the reference pictures align with the target video — "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
                f"target video; Picture 2 (from Shot {last}) aligns with the "
                f"{stamp}-second mark of the target video."
            )
        if self.mode is H3Mode.L2VA:
            return (
                "How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot {last}]) aligns with the {stamp}-second "
                "mark of the target video."
            )
        raise UnknownKeyError("H3 mode", self.mode, list(H3Mode))

    def render(self) -> str:
        """The final prompt string, ready to inject into a ComfyUI graph.

        Layout is fixed by the guide: instruction line, one blank line, then the
        three core fields separated by blank lines.
        """
        body = " ".join(shot.render() for shot in self.shots)
        fields = "\n\n".join(
            (
                f"integrated_multimodal_description: {body}",
                f"overall_soundscape: {self.overall_soundscape}",
                f"non_diegetic_music: {self.non_diegetic_music}",
            )
        )
        head = self.instruction()
        return f"{head}\n\n{fields}" if head else fields
