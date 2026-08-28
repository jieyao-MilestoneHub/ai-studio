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

ENV_FILE: Path | None = Path(__file__).resolve().parents[3] / ".env"
"""The checkout's `.env`, wherever the process was started from. A
service unit or a shell in a subdirectory reads the same credentials.
Read at construction (`get_settings`), so a test can point it elsewhere."""


class Settings(BaseSettings):
    """Read from the environment, and from `.env` in development.

    `.env` is gitignored; `.env.example` ships key names with empty values.
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ---------------------------------------------------------- RunPod

    runpod_api_key: SecretStr | None = Field(default=None, alias="RUNPOD_API_KEY")
    prompt_mode: str = Field(
        default="structured",
        alias="AI_STUDIO_PROMPT_MODE",
        pattern="^(raw|structured)$",
        description="structured (default since 2026-08-27): gpt-oss-20b on the pod "
        "rewrites every request into its model's best input shape before it is "
        "rendered -- the H3 shot schema (📏 26.0 -> 367.6), a Flux natural-language "
        "prompt, or an understanding model's question (prompts/convert.py, "
        "prompts/flux.py, prompts/understanding.py). raw: the user's words go to "
        "the model verbatim, no rewrite. The rewriter is never a serverless "
        "endpoint; see pipeline/pod_llm.py.",
    )
    hf_token: SecretStr | None = Field(
        default=None,
        alias="HF_TOKEN",
        description="A Hugging Face read token, for any gated model repo the pod "
        "downloads. Fed to deploy/pod_setup.sh on stdin by runtime.session.provision, "
        "never written to the pod's disk. Every model currently served is ungated "
        "(Tarsier2, which was, is retired), so it is optional.",
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
            "serving the understanding and chat models. On RunPod this "
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
        description="Longest audio clip the understanding backend accepts; a "
        "caller checks it before downloading the file.",
    )
    max_video_understand_s: float = Field(
        default=120.0,
        gt=0,
        alias="AI_STUDIO_MAX_VIDEO_UNDERSTAND_S",
        description="[speculative] longest video clip the understanding backend "
        "accepts -- nothing has measured what Qwen2.5-VL tolerates or costs per "
        "second of dense video understanding on this hardware yet. Generous "
        "rather than tight until benchmarked; tune once measured.",
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
    ffmpeg_bin: str = Field(default="ffmpeg", alias="AI_STUDIO_FFMPEG_BIN")
    ffprobe_bin: str = Field(default="ffprobe", alias="AI_STUDIO_FFPROBE_BIN")

    # ---------------------------------------------------------- logs / archive

    log_dir: Path = Field(
        default=Path("logs"),
        alias="AI_STUDIO_LOG_DIR",
        description="Where each service writes its JSONL trace (one file per local "
        "day under <log_dir>/<service>/), plus logs/sessions/ and logs/pods/ -- "
        "the records `ai-studio archive` compresses. See core/observability.py.",
    )
    log_level: str = Field(
        default="INFO",
        alias="AI_STUDIO_LOG_LEVEL",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Root level for both sinks. DEBUG adds the per-minute reaper "
        "decisions to the JSONL without touching journald.",
    )
    archive_dir: Path = Field(
        default=Path("archive"),
        alias="AI_STUDIO_ARCHIVE_DIR",
        description="Where the daily `ai-studio archive` writes "
        "<archive_dir>/YYYY-MM-DD/ai-studio-<stamp>.tar.zst + manifest.json. Same "
        "disk by decision (2026-08-28); an off-box push only adds a destination.",
    )
    log_hot_days: float = Field(
        default=30.0,
        ge=0.0,
        alias="AI_STUDIO_LOG_HOT_DAYS",
        description="JSONL logs, session records and pod logs older than this are "
        "deleted from log_dir -- only after they are inside a verified archive.",
    )
    archive_keep_days: float = Field(
        default=365.0,
        ge=0.0,
        alias="AI_STUDIO_ARCHIVE_KEEP_DAYS",
        description="Archives older than this are deleted. 0 keeps every archive.",
    )

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id


_settings: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Process-wide settings singleton. `refresh=True` re-reads for tests."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings(_env_file=ENV_FILE)
    return _settings
