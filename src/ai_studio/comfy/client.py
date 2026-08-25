"""ComfyUI HTTP API client.

Every call here is short by construction, and that is a hard requirement rather
than a style choice: when ComfyUI runs on a RunPod pod it is reached through
`https://<pod-id>-8188.proxy.runpod.net`, and that proxy sits behind Cloudflare,
which severs any connection held open past roughly 100 seconds with a 524. An
H3 clip takes 2-6 minutes. So there is no version of "just await the render" —
you queue, you poll, you fetch.

ComfyUI's own API is already shaped that way, which is why the provider
protocol (submit / poll / fetch / cancel) maps onto it one-to-one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from ai_studio.core.errors import ProviderError, ProviderSubmitError

# Output collections a video-producing node may populate. ComfyUI has no single
# convention: the built-in savers use "images", VideoHelperSuite writes "gifs",
# and newer video nodes use "videos".
_OUTPUT_KEYS = ("videos", "gifs", "images")

_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".gif")


@dataclass(frozen=True)
class ComfyOutput:
    """One file ComfyUI produced, as named in the history entry."""

    filename: str
    subfolder: str
    type: str

    @property
    def is_video(self) -> bool:
        return self.filename.lower().endswith(_VIDEO_SUFFIXES)


class ComfyClient:
    """Thin async wrapper over the ComfyUI endpoints we actually use."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        client_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
        )

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- readiness

    async def is_ready(self) -> bool:
        """Cheap liveness probe.

        On a fresh RunPod ComfyUI pod this returns False for roughly the first
        four minutes while the container copies itself into /workspace — the
        proxy answers 502 throughout. That is expected, not a failure.
        """
        try:
            response = await self._client.get("/system_stats")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def system_stats(self) -> dict[str, Any]:
        return await self._get_json("/system_stats")

    async def object_info(self, node_class: str | None = None) -> dict[str, Any]:
        """Available node classes. Use it to confirm the H3 nodes are installed."""
        path = "/object_info" if node_class is None else f"/object_info/{node_class}"
        return await self._get_json(path)

    # ---------------------------------------------------------------- submit

    async def queue_prompt(self, graph: dict[str, Any]) -> str:
        """Enqueue a workflow. Returns the `prompt_id` to poll.

        ComfyUI reports graph problems as a 400 with a `node_errors` map, which
        is far more useful than the status code alone — so it is surfaced
        verbatim rather than collapsed into "submit failed".
        """
        try:
            response = await self._client.post(
                "/prompt", json={"prompt": graph, "client_id": self.client_id}
            )
        except httpx.HTTPError as exc:
            raise ProviderSubmitError(f"could not reach ComfyUI at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderSubmitError(
                f"ComfyUI rejected the workflow ({response.status_code}): {response.text[:2000]}"
            )

        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise ProviderSubmitError(f"ComfyUI returned no prompt_id: {payload!r}")

        if node_errors := payload.get("node_errors"):
            raise ProviderSubmitError(f"ComfyUI reported node errors: {node_errors!r}")

        return str(prompt_id)

    # ------------------------------------------------------------------ poll

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        """The history entry for one prompt, or None while it is still queued."""
        payload = await self._get_json(f"/history/{prompt_id}")
        entry = payload.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def queue_position(self, prompt_id: str) -> int | None:
        """Position in the pending queue, or None if it is running or done."""
        payload = await self._get_json("/queue")
        pending = payload.get("queue_pending", [])
        for index, item in enumerate(pending):
            if isinstance(item, list) and len(item) > 1 and str(item[1]) == prompt_id:
                return index
        return None

    @staticmethod
    def outputs_of(entry: dict[str, Any]) -> list[ComfyOutput]:
        """Flatten a history entry's outputs into a file list, videos first."""
        found: list[ComfyOutput] = []
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict):
            return found
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for key in _OUTPUT_KEYS:
                for item in node_output.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append(
                            ComfyOutput(
                                filename=str(item["filename"]),
                                subfolder=str(item.get("subfolder", "")),
                                type=str(item.get("type", "output")),
                            )
                        )
        found.sort(key=lambda o: not o.is_video)
        return found

    @staticmethod
    def status_of(entry: dict[str, Any]) -> tuple[str, str | None]:
        """`(status_str, error)` from a history entry.

        ComfyUI's own vocabulary is `success` / `error`; anything else is still
        running. The error text is dug out of the execution messages, which is
        where the actual cause lives.
        """
        status = entry.get("status")
        if not isinstance(status, dict):
            return "running", None

        completed = status.get("completed")
        status_str = str(status.get("status_str", "")) or ("success" if completed else "running")

        error: str | None = None
        for message in status.get("messages", []) or []:
            if isinstance(message, list) and len(message) > 1 and message[0] == "execution_error":
                detail = message[1]
                if isinstance(detail, dict):
                    error = str(
                        detail.get("exception_message") or detail.get("exception_type") or detail
                    )
        return status_str, error

    # ----------------------------------------------------------------- fetch

    async def download(self, output: ComfyOutput) -> bytes:
        """Fetch one produced file.

        Outputs live on the pod's container disk, which is destroyed when the
        pod is terminated. Download before `pod down`, not after.
        """
        params = {
            "filename": output.filename,
            "subfolder": output.subfolder,
            "type": output.type,
        }
        try:
            response = await self._client.get("/view", params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not download {output.filename}: {exc}") from exc
        return response.content

    # ---------------------------------------------------------------- cancel

    async def interrupt(self) -> None:
        """Stop the *currently executing* prompt. ComfyUI has no per-id cancel."""
        try:
            await self._client.post("/interrupt")
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not interrupt: {exc}") from exc

    # --------------------------------------------------------------- private

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"GET {path} failed: {exc}") from exc
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
