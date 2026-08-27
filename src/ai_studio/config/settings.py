"""Runtime settings.

Every secret is a `SecretStr`, which redacts itself in `repr()`, in log lines,
and in pydantic validation tracebacks. Secrets load from the environment only —
never from a file that could be committed — so there is no code path that reads
a key out of the repo.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from the environment, and from `.env` in development.

    `.env` is gitignored; `.env.example` ships key names with empty values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ---------------------------------------------------------- RunPod

    runpod_api_key: SecretStr | None = Field(default=None, alias="RUNPOD_API_KEY")
    prompt_mode: str = Field(
        default="raw",
        alias="AI_STUDIO_PROMPT_MODE",
        pattern="^(raw|structured)$",
        description="raw: the user's own words go to the model verbatim, no LLM "
        "call. structured: the LLM rewrites them into the H3 shot schema / a "
        "Flux prompt (prompts/convert.py, prompts/flux.py). The structured code "
        "stays; raw is what the group asked for.",
    )
    network_volume_id: str | None = Field(
        default=None,
        alias="AI_STUDIO_NETWORK_VOLUME_ID",
        description="A RunPod network volume holding the model weights. When set, "
        "every window pod mounts it at /workspace and is placed in its datacenter, "
        "so a cold open is a ComfyUI restart instead of a 68GB download "
        "(measured 2026-08-26: ~15 minutes and $0.18 per open without one).",
    )

    # ---------------------------------------------------------- ComfyUI

    comfy_url: str = Field(
        default="http://127.0.0.1:8188",
        alias="AI_STUDIO_COMFY_URL",
        description=(
            "Base URL of the ComfyUI instance. On RunPod this is the pod proxy "
            "https://<pod-id>-8188.proxy.runpod.net — note that proxy requests "
            "are cut off at ~100s, so only ever poll through it, never block."
        ),
    )
    comfy_timeout_s: float = Field(default=30.0, gt=0, alias="AI_STUDIO_COMFY_TIMEOUT_S")
    comfy_poll_interval_s: float = Field(
        default=5.0, gt=0, alias="AI_STUDIO_COMFY_POLL_INTERVAL_S"
    )
    comfy_job_timeout_s: float = Field(
        default=1800.0,
        gt=0,
        alias="AI_STUDIO_COMFY_JOB_TIMEOUT_S",
        description="Give up on a single clip after this long. H3 at 1280x736 is ~360s.",
    )

    # ---------------------------------------------------------- understanding

    inference_url: str = Field(
        default="http://127.0.0.1:8189",
        alias="AI_STUDIO_INFERENCE_URL",
        description=(
            "Base URL of deploy/inference_server.py, the pod-side process "
            "serving moondream3/Qwen3-Omni-Captioner/Tarsier2. On RunPod this "
            "is the pod proxy https://<pod-id>-8189.proxy.runpod.net -- same "
            "~100s proxy-timeout caveat as AI_STUDIO_COMFY_URL."
        ),
    )
    inference_timeout_s: float = Field(default=30.0, gt=0, alias="AI_STUDIO_INFERENCE_TIMEOUT_S")
    inference_job_timeout_s: float = Field(
        default=300.0,
        gt=0,
        alias="AI_STUDIO_INFERENCE_JOB_TIMEOUT_S",
        description="Give up on one understanding job after this long, "
        "including a cold model load and the GPU hand-off with ComfyUI.",
    )
    max_audio_understand_s: float = Field(
        default=30.0,
        gt=0,
        alias="AI_STUDIO_MAX_AUDIO_UNDERSTAND_S",
        description="Longest /說音 audio clip accepted, matching "
        "Qwen3-Omni-Captioner's own stated ceiling. Checked against LINE's "
        "own reported message duration before the file is even downloaded.",
    )
    max_video_understand_s: float = Field(
        default=120.0,
        gt=0,
        alias="AI_STUDIO_MAX_VIDEO_UNDERSTAND_S",
        description="[speculative] longest /說影 clip accepted -- nothing has "
        "measured what Tarsier2 actually tolerates or costs per second of "
        "dense video understanding on this hardware yet. Generous rather "
        "than tight until benchmarked; tune once measured.",
    )

    # ---------------------------------------------------------- storage

    runs_dir: Path = Field(default=Path("runs"), alias="AI_STUDIO_RUNS_DIR")
    out_dir: Path = Field(default=Path("out"), alias="AI_STUDIO_OUT_DIR")

    s3_endpoint_url: str | None = Field(default=None, alias="AI_STUDIO_S3_ENDPOINT_URL")
    s3_bucket: str | None = Field(default=None, alias="AI_STUDIO_S3_BUCKET")
    s3_access_key_id: SecretStr | None = Field(default=None, alias="AI_STUDIO_S3_ACCESS_KEY_ID")
    s3_secret_access_key: SecretStr | None = Field(
        default=None, alias="AI_STUDIO_S3_SECRET_ACCESS_KEY"
    )
    s3_public_base_url: str | None = Field(default=None, alias="AI_STUDIO_S3_PUBLIC_BASE_URL")

    # ---------------------------------------------------------- guardrails

    max_cost_usd: float = Field(default=5.0, ge=0, alias="AI_STUDIO_MAX_COST_USD")
    max_month_usd: float = Field(
        default=50.0,
        ge=0,
        alias="AI_STUDIO_MAX_MONTH_USD",
        description="Hard calendar-month ceiling across every window. Enforced by "
        "runtime.budget.MonthlyBudgetGuard before a pod is opened, not after.",
    )
    vps_monthly_usd: float = Field(
        default=5.0,
        ge=0,
        alias="AI_STUDIO_VPS_MONTHLY_USD",
        description="The always-on host's own monthly cost, reserved out of "
        "max_month_usd before computing what's left for GPU compute.",
    )

    max_pod_opens_per_day: int = Field(
        default=15,
        ge=0,
        alias="AI_STUDIO_MAX_POD_OPENS_PER_DAY",
        description="How many pods may be created in one Asia/Taipei day. Pods "
        "are opened on demand and reaped minutes after the last render, so a "
        "normal day can legitimately open a dozen. This is the backstop the "
        "monthly guard cannot provide -- a worker crash-looping opens a fresh "
        "pod on every restart and each one is individually within budget.",
    )
    max_jobs_per_user_per_day: int = Field(
        default=10,
        ge=0,
        alias="AI_STUDIO_MAX_JOBS_PER_USER_PER_DAY",
        description="How many requests one LINE user may have accepted in an "
        "Asia/Taipei day. 0 disables the cap. Checked before the request is "
        "enqueued, so a refusal does not also spend an LLM conversion.",
    )

    ffmpeg_bin: str = Field(default="ffmpeg", alias="AI_STUDIO_FFMPEG_BIN")
    ffprobe_bin: str = Field(default="ffprobe", alias="AI_STUDIO_FFPROBE_BIN")

    # ---------------------------------------------------------- LINE (phase 2)

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
        "many days. `ai-studio gc` and the daily timer enforce it. 0 keeps "
        "everything (and lets the disk fill -- only for a short-lived host).",
    )

    llm_endpoint_id: str | None = Field(default=None, alias="AI_STUDIO_LLM_ENDPOINT_ID")
    llm_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="AI_STUDIO_LLM_MODEL")

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

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id


_settings: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Process-wide settings singleton. `refresh=True` re-reads for tests."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
