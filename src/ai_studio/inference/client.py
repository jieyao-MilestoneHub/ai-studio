"""HTTP client for `deploy/inference_server.py`, the pod-side process that
lazily loads/unloads moondream3, Qwen3-Omni-Captioner, and Tarsier2.

Same submit/poll/cancel discipline as `comfy.client.ComfyClient`, and for the
same reason: this also runs behind RunPod's pod proxy, severed by Cloudflare
at ~100 seconds. A warm caption call is fast, but a *cold model load* for a
model this size is not guaranteed to fit in that window, so nothing here may
ever be one blocking call. Unlike `ComfyClient` there is no `download()` --
the result is text, returned inline by `poll_job()`, not a file to fetch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ai_studio.core.errors import ProviderError, ProviderSubmitError


class InferenceClient:
    """Thin async wrapper over the understanding server's endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- readiness

    async def is_ready(self) -> bool:
        """Cheap liveness probe.

        Unlike ComfyUI's readiness (which waits for a node pack to install),
        this server is "ready" the moment the process is up -- models load
        lazily per request, not at startup.
        """
        try:
            response = await self._client.get("/healthz")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    # ---------------------------------------------------------------- submit

    async def submit_job(
        self, modality: str, input_path: Path | str, prompt: str | None = None
    ) -> str:
        """Upload the media file and enqueue a description job. Returns a job id.

        The server accepts the request immediately and does the actual
        model-load-then-infer in a background task, exactly why submit/poll
        is preserved even though a warm call would be fast enough to block on.
        """
        source = Path(input_path)
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise ProviderSubmitError(f"could not read {source}: {exc}") from exc

        try:
            response = await self._client.post(
                "/submit",
                files={"media": (source.name, data)},
                data={"modality": modality, **({"prompt": prompt} if prompt else {})},
            )
        except httpx.HTTPError as exc:
            raise ProviderSubmitError(
                f"could not reach the inference server at {self.base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise ProviderSubmitError(
                f"inference server rejected the job ({response.status_code}): "
                f"{response.text[:500]}"
            )

        payload = response.json()
        job_id = payload.get("job_id")
        if not job_id:
            raise ProviderSubmitError(f"inference server returned no job_id: {payload!r}")
        return str(job_id)

    # ------------------------------------------------------------------ poll

    async def poll_job(self, job_id: str) -> dict[str, Any]:
        """`{"state": "queued"|"running"|"completed"|"failed", "result_text": ..., "error": ...}`."""
        try:
            response = await self._client.get(f"/poll/{job_id}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"GET /poll/{job_id} failed: {exc}") from exc
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    # ---------------------------------------------------------------- cancel

    async def cancel_job(self, job_id: str) -> None:
        """Best-effort. A job that has already finished has nothing to cancel."""
        try:
            await self._client.post(f"/cancel/{job_id}")
        except httpx.HTTPError:
            return None

    # ----------------------------------------------------------- GPU hand-off

    async def unload(self) -> None:
        """Evict whichever model is currently resident, without stopping the
        server process. Called before a ComfyUI generation job runs on the
        same card -- see `comfy.client.ComfyClient.free_memory` for the
        reverse direction."""
        try:
            await self._client.post("/unload")
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not unload the inference server's model: {exc}") from exc
