"""The `extra=` keys this package's records carry that ai-studio's
allow-list does not know: the request's delivery token, who asked, what the
push did. Passed to `ai_studio.core.observability.configure_logging` by the
composition root."""

from __future__ import annotations

EXTRA_FIELDS: tuple[str, ...] = (
    "token", "built_by", "user", "message_id", "outcome", "deferred", "resident",
    "attempts", "quota_exhausted", "to",
)
