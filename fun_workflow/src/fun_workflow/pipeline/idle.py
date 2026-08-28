"""How long a quiet pod is worth keeping after each kind of job.

Numbers differ because the reloads cost differently: Flux comes back into
VRAM in 📏 ~15 s, H3's 32B text encoder in 📏 60-90 s. The understanding
grace is `[speculative]` -- nothing has measured a lazy-load cost for
moondream3, Qwen2-Audio or Qwen2.5-VL on this hardware yet, and the three
are not even the same size, so one shared number is a starting point, not a
considered answer. A pod is worth keeping only while the chance of the next
request within the grace, times the reload it would save, beats the idle
minutes -- and in a group chat the next message usually comes within five
minutes or not for hours. The reaper log says how often a pod was closed and
reopened within a few minutes, which is the number that tunes these.

CHAT is deliberately the longest grace, and for a different reason than the
others: a `/himonkey` conversation's cadence is many short exchanges with
ordinary pauses in between, and every reopen has a real fixed floor
(`ai_studio.runtime.budget.MIN_SESSION_MINUTES` worth of billing) *and*
consumes one of the day's pod opens -- a ceiling that, once hit, blocks new
video and image sessions too. A grace long enough to survive a
conversation's ordinary pauses is both cheaper and safer for the rest of the
service than one short enough to reopen repeatedly. `[speculative]` -- retune
from the reaper log once real chat traffic exists, not from this guess.

DRAMA equals the video grace because a drama's last GPU job is an H3 clip.
What makes it safe for a 15-30 minute render is not the number:
`pipeline.drama` touches activity after *every* fetched still or clip, so
the grace only ever measures a real gap, never a long render.

The pod runtime keeps the clock and reads back whatever grace the last
touch recorded (`ai_studio.runtime.session.touch_activity`); it holds no
table of kinds.
"""

from __future__ import annotations

from fun_workflow.core.kinds import JobKind

GRACE_MINUTES_BY_KIND: dict[JobKind, float] = {
    JobKind.IMAGE: 5.0,
    JobKind.VIDEO: 10.0,
    JobKind.IMAGE_UNDERSTAND: 5.0,
    JobKind.AUDIO_UNDERSTAND: 5.0,
    JobKind.VIDEO_UNDERSTAND: 5.0,
    JobKind.CHAT: 15.0,
    JobKind.DRAMA: 10.0,
}


def grace_for(kind: JobKind) -> float:
    """Raises on a kind with no entry -- fail loudly, never a silent default."""
    return GRACE_MINUTES_BY_KIND[kind]
