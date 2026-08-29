"""Settings for the request-taking side: LINE credentials, per-user caps,
delivery directories, the public origin, `/短劇` knobs.

Separate from `Settings` (GPU, money, logs) so the generation stack can be
configured with no notion of a chat group at all. Composed, not inherited:
`FunSettings.studio` is the process-wide `Settings`, read once.
"""

from __future__ import annotations

from pathlib import Path

from ai_studio.config import settings as studio_settings
from ai_studio.config.settings import Settings, get_settings
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class FunSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ---------------------------------------------------------- caps

    max_jobs_per_user_per_day: int = Field(
        default=10,
        ge=0,
        alias="AI_STUDIO_MAX_JOBS_PER_USER_PER_DAY",
        description="How many requests one LINE user may have accepted in an "
        "Asia/Taipei day. 0 disables the cap. Checked before the request is "
        "enqueued, so a refusal does not also spend an LLM conversion. "
        "/himonkey chat messages are excluded -- see "
        "max_chat_messages_per_user_per_day.",
    )
    max_chat_messages_per_user_per_day: int = Field(
        default=50,
        ge=0,
        alias="AI_STUDIO_MAX_CHAT_MESSAGES_PER_USER_PER_DAY",
        description="How many /himonkey messages one LINE user may have "
        "accepted in an Asia/Taipei day. 0 disables the cap. Separate from "
        "max_jobs_per_user_per_day on purpose: a normal chat conversation's "
        "cadence would otherwise exhaust a user's entire daily video/image "
        "allowance too.",
    )
    max_chat_month_usd: float = Field(
        default=15.0,
        ge=0,
        alias="AI_STUDIO_MAX_CHAT_MONTH_USD",
        description="A sub-ceiling on /himonkey's share of the monthly GPU "
        "budget, enforced by pipeline.drain.render_chat against "
        "pipeline.queue.chat_spent_this_month_usd() before it submits -- "
        "a separate mechanism from runtime.budget.MonthlyBudgetGuard, "
        "which only sees whole sessions and knows nothing about chat; the "
        "all-kinds monthly cap still applies on top. Once hit, new chat jobs "
        "stop being claimed for the rest of the month while video/image keep "
        "running. A starting guess, meant to be retuned from real usage.",
    )

    # ---------------------------------------------------------- /短劇

    max_dramas_per_day: int = Field(
        default=3,
        ge=0,
        alias="AI_STUDIO_MAX_DRAMAS_PER_DAY",
        description="How many /短劇 requests the group may have accepted in an "
        "Asia/Taipei day, all users together. A drama is ~15-30 GPU-minutes "
        "(six H3 clips plus eight Flux stills), so the per-user job cap alone "
        "would let one afternoon spend the month. 0 disables the cap.",
    )
    drama_face_repair: bool = Field(
        default=True,
        alias="AI_STUDIO_DRAMA_FACE_REPAIR",
        description="Run the Impact-Pack FaceDetailer pass on /短劇 keyframe "
        "stills when the pod has the nodes. Never on video. Off: plain "
        "image-to-image keyframes.",
    )
    drama_keyframe_denoise: float = Field(
        default=0.55,
        gt=0,
        le=1.0,
        alias="AI_STUDIO_DRAMA_KEYFRAME_DENOISE",
        description="How much of the character sheet a /短劇 keyframe may "
        "repaint: lower keeps the face, higher lets the scene change. "
        "[speculative] 0.55 -- retune from the first real drama's keyframes.",
    )
    drama_keyframe_denoise_wide: float = Field(
        default=0.70,
        gt=0,
        le=1.0,
        alias="AI_STUDIO_DRAMA_KEYFRAME_DENOISE_WIDE",
        description="The keyframe denoise for a shot that opens wide or as a "
        "two-shot: the character sheet is a head-and-shoulders portrait, and "
        "a wide frame repainted at 0.55 keeps the portrait's framing and "
        "ignores the prompt's. [speculative] 0.70.",
    )
    drama_subshots: bool = Field(
        default=True,
        alias="AI_STUDIO_DRAMA_SUBSHOTS",
        description="Ask H3 to cut to a second framing inside the longer /短劇 "
        "clips (its multi-shot prompt). Off: every shot is one held framing "
        "and the timeline has six segments. The hedge for a model that "
        "ignores the cut time under image-to-video, which is unmeasured.",
    )

    # ---------------------------------------------------------- delivery

    public_base_url: str = Field(
        default="http://localhost:8000",
        alias="AI_STUDIO_PUBLIC_BASE_URL",
        description=(
            "The externally reachable HTTPS origin of the always-on service. It "
            "goes into every link the bot replies with, so it must be the public "
            "hostname, not localhost, in production."
        ),
    )
    files_dir: Path = Field(default=Path("files"), alias="AI_STUDIO_FILES_DIR")
    incoming_dir: Path = Field(default=Path("incoming"), alias="AI_STUDIO_INCOMING_DIR")
    files_retention_days: float = Field(
        default=7.0,
        ge=0.0,
        alias="AI_STUDIO_FILES_RETENTION_DAYS",
        description="Delete delivered media and received photos older than this "
        "many days. `gc` and the daily timer enforce it. 0 keeps everything "
        "(and lets the disk fill -- only for a short-lived host).",
    )

    # ---------------------------------------------------------- LINE

    line_channel_secret: SecretStr | None = Field(default=None, alias="LINE_CHANNEL_SECRET")
    line_channel_access_token: SecretStr | None = Field(
        default=None, alias="LINE_CHANNEL_ACCESS_TOKEN"
    )
    line_allowed_group_id: str | None = Field(default=None, alias="LINE_ALLOWED_GROUP_ID")
    # Optional second gate, inside the group. Group membership is not ours to
    # control: anyone an existing member invites can otherwise spend GPU time.
    line_allowed_user_ids: str | None = Field(default=None, alias="LINE_ALLOWED_USER_IDS")

    @property
    def allowed_users(self) -> frozenset[str]:
        """The user allowlist, or empty meaning "any member of the group"."""
        raw = self.line_allowed_user_ids or ""
        return frozenset(u.strip() for u in raw.replace(";", ",").split(",") if u.strip())

    @property
    def studio(self) -> Settings:
        """The GPU/money/log settings this side composes."""
        return get_settings()


_settings: FunSettings | None = None


def get_fun_settings(*, refresh: bool = False) -> FunSettings:
    """Process-wide singleton. `refresh=True` re-reads for tests."""
    global _settings
    if _settings is None or refresh:
        _settings = FunSettings(_env_file=studio_settings.ENV_FILE)
    return _settings
