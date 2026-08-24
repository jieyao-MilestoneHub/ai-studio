"""Build a MiniMax H3 prompt from typed fields.

Run: `uv run python examples/build_prompt.py`

The point of the builder is that the official schema is the *output* rather than
something you have to remember. Prompt structure is the single biggest quality
lever on this model — free prose scored 26.0 against 367.6 for the structured
schema, while a five-fold increase in resolution changed nothing. [reported]
"""

from __future__ import annotations

from videogen.prompts.h3 import (
    Amplitude,
    CameraMotion,
    Dialogue,
    H3Mode,
    H3Prompt,
    PromptShot,
    Speed,
    camera_phrase,
)

prompt = H3Prompt(
    mode=H3Mode.T2VA,
    duration_s=10.0,
    shots=(
        PromptShot(
            index=1,
            # Shot 1 must open with style and initial composition.
            style="Live-action, cinematic",
            description=(
                "a medium-wide shot frames a baker opening the shutters of a small "
                "street bakery before sunrise, warm interior light spilling onto wet "
                'pavement, a hand-painted sign reading "Morning Loaf" above the door'
            ),
            camera=camera_phrase(
                CameraMotion.PUSH_IN,
                Amplitude.SMALL,
                Speed.SLOW,
                toward="the fresh loaf on the wooden counter",
            ),
            dialogue=(
                Dialogue(
                    speaker_id="S1",
                    identity="The middle-aged baker with a calm, slightly raspy voice",
                    text="First batch of the morning.",
                ),
            ),
        ),
        PromptShot(
            index=2,
            cut_at_s=5.0,
            description=(
                "a close-up of steam rising from sliced bread while the baker's final "
                "words carry over from the previous shot"
            ),
            camera=camera_phrase(CameraMotion.STATIC_SHOT),
        ),
    ),
    overall_soundscape=(
        "Wooden shutters scrape open over a quiet street as trays clink softly "
        "inside the bakery. A doorbell rings once, followed by light footsteps "
        "and the crisp sound of bread being sliced."
    ),
    non_diegetic_music=(
        "A soft acoustic-guitar pattern at a moderate tempo, joined by sparse "
        "upright-bass notes and a gentle fade at the end."
    ),
)

if __name__ == "__main__":
    print(prompt.render())
