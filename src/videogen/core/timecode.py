"""Frame/second conversion. Pure arithmetic, no I/O.

Everything downstream of `resolve_timeline` works in frames, because cut points
and transition durations are only meaningful on frame boundaries. The editing
grammar's transition durations (whip 0.30s, wipe 0.50s, zoom 0.33s) are quoted
in seconds but land on exact frames at 24 and 30 fps, which is not an accident.
"""

from __future__ import annotations

from fractions import Fraction

DEFAULT_FPS = 24
"""MiniMax H3 renders at 24fps. Delivering at the model's native rate avoids
frame interpolation, which is blacklisted: `minterpolate` smears generative
artifacts into something worse than judder."""


def seconds_to_frames(seconds: float, fps: int = DEFAULT_FPS) -> int:
    """Round to the nearest frame. Half-frames round up, deterministically."""
    if seconds < 0:
        raise ValueError(f"negative duration: {seconds}")
    return int(Fraction(seconds).limit_denominator(1000) * fps + Fraction(1, 2))


def frames_to_seconds(frames: int, fps: int = DEFAULT_FPS) -> float:
    if frames < 0:
        raise ValueError(f"negative frame count: {frames}")
    return frames / fps


def quantize_seconds(seconds: float, fps: int = DEFAULT_FPS) -> float:
    """Snap a duration to the nearest whole frame.

    Use this on anything an author typed. A 2.5s hold at 24fps is 60 frames; a
    2.51s hold is also 60 frames, and pretending otherwise puts sub-frame drift
    into the timeline that accumulates across a hundred cuts.
    """
    return frames_to_seconds(seconds_to_frames(seconds, fps), fps)


def snap_to_quantum(seconds: float, quantum: float | None) -> float:
    """Round a requested clip length up to what the model can actually emit.

    Many video models only produce fixed lengths (H3: 5s or 10s). Asking for 7s
    silently gets you something else, so the planner snaps up and the plan gate
    asserts the result is representable.
    """
    if quantum is None:
        return seconds
    if quantum <= 0:
        raise ValueError(f"quantum must be positive: {quantum}")
    steps = int(Fraction(seconds) / Fraction(quantum).limit_denominator(1000))
    if steps * quantum < seconds:
        steps += 1
    return max(1, steps) * quantum


def format_timecode(seconds: float, fps: int = DEFAULT_FPS) -> str:
    """``HH:MM:SS:FF`` — for gate findings and log lines a human has to read."""
    total = seconds_to_frames(seconds, fps)
    frames = total % fps
    secs = total // fps
    return f"{secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}:{frames:02d}"
