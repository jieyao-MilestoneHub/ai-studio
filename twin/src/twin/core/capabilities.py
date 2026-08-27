"""Per-surface capability declaration. SPEC.md §4.3/§6.4 — whether a surface's
exposure signal (read receipts, chat-open events) is actually obtainable MUST
be a code-produced declaration, not a remembered comment, so harness.suites.s3
can refuse a surface automatically rather than relying on someone to notice.

Mirrors ai_studio.providers' Protocol-plus-capabilities-one-layer-below shape
(PLAN.md §3.3 row 1). No concrete instance ships from this module: SPEC.md
§11 item H marks LINE's exposure-signal availability as *unvalidated*, not
decided false, and hardcoding a guess here would quietly prejudge a question
that's explicitly still open. The real instance ships once Phase 8/11
actually runs that validation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SurfaceCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    surface: str
    exposure_signal_available: bool
    exposure_signal_note: str = ""
