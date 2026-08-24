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
    )

    # ---------------------------------------------------------- RunPod

    runpod_api_key: SecretStr | None = Field(default=None, alias="RUNPOD_API_KEY")

    # ---------------------------------------------------------- ComfyUI

    comfy_url: str = Field(
        default="http://127.0.0.1:8188",
        alias="VIDEOGEN_COMFY_URL",
        description=(
            "Base URL of the ComfyUI instance. On RunPod this is the pod proxy "
            "https://<pod-id>-8188.proxy.runpod.net — note that proxy requests "
            "are cut off at ~100s, so only ever poll through it, never block."
        ),
    )
    comfy_timeout_s: float = Field(default=30.0, gt=0, alias="VIDEOGEN_COMFY_TIMEOUT_S")
    comfy_poll_interval_s: float = Field(
        default=5.0, gt=0, alias="VIDEOGEN_COMFY_POLL_INTERVAL_S"
    )
    comfy_job_timeout_s: float = Field(
        default=1800.0,
        gt=0,
        alias="VIDEOGEN_COMFY_JOB_TIMEOUT_S",
        description="Give up on a single clip after this long. H3 at 1280x736 is ~360s.",
    )

    # ---------------------------------------------------------- storage

    runs_dir: Path = Field(default=Path("runs"), alias="VIDEOGEN_RUNS_DIR")
    out_dir: Path = Field(default=Path("out"), alias="VIDEOGEN_OUT_DIR")

    s3_endpoint_url: str | None = Field(default=None, alias="VIDEOGEN_S3_ENDPOINT_URL")
    s3_bucket: str | None = Field(default=None, alias="VIDEOGEN_S3_BUCKET")
    s3_access_key_id: SecretStr | None = Field(default=None, alias="VIDEOGEN_S3_ACCESS_KEY_ID")
    s3_secret_access_key: SecretStr | None = Field(
        default=None, alias="VIDEOGEN_S3_SECRET_ACCESS_KEY"
    )
    s3_public_base_url: str | None = Field(default=None, alias="VIDEOGEN_S3_PUBLIC_BASE_URL")

    # ---------------------------------------------------------- guardrails

    max_cost_usd: float = Field(default=5.0, ge=0, alias="VIDEOGEN_MAX_COST_USD")

    ffmpeg_bin: str = Field(default="ffmpeg", alias="VIDEOGEN_FFMPEG_BIN")
    ffprobe_bin: str = Field(default="ffprobe", alias="VIDEOGEN_FFPROBE_BIN")

    # ---------------------------------------------------------- LINE (phase 2)

    public_base_url: str = Field(
        default="http://localhost:8000",
        alias="VIDEOGEN_PUBLIC_BASE_URL",
        description=(
            "The externally reachable HTTPS origin of the always-on service. It "
            "goes into every link the bot replies with, so it must be the public "
            "hostname, not localhost, in production."
        ),
    )
    files_dir: Path = Field(default=Path("files"), alias="VIDEOGEN_FILES_DIR")

    llm_endpoint_id: str | None = Field(default=None, alias="VIDEOGEN_LLM_ENDPOINT_ID")
    llm_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="VIDEOGEN_LLM_MODEL")

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
