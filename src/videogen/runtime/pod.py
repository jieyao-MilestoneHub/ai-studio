"""RunPod pod lifecycle for the MiniMax H3 ComfyUI host.

Written against the live REST v2 contract (`https://api.runpod.io/v2`,
`openapi.json` version 2.0.0), not from memory.

Almost everything here is a guardrail, because on this platform the expensive
mistakes are all quiet ones:

- **`stop` does not stop billing.** A stopped pod keeps its disk and keeps
  charging for it. `terminate` is the only thing that ends the meter, so
  `PodManager.down()` terminates and `stop` is not exposed at all.
- **Host RAM is not selectable.** It comes with whatever machine you are
  allocated, and MiniMax H3 needs >= 60 GB — a 31 GB host crashed part-way
  through a second consecutive generation. [reported] So the only workable
  policy is: deploy, inspect, and terminate immediately if the host is short.
- **A pod that cannot pull its image bills while it tries.** Over ~5 minutes is
  a warning sign, over ~10 minutes is a lost machine; redeploy rather than wait.
- **Never let RunPod hold a reservation.** Requesting a GPU with no stock can
  create a standing "deploy when available" order that starts billing whenever
  capacity appears, including overnight. This module checks availability first
  and refuses rather than queueing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from videogen.config.settings import get_settings
from videogen.core.errors import InsufficientHostRamError, PodError

API_BASE = "https://api.runpod.io/v2"

MIN_HOST_RAM_GB = 60
"""H3 loads a 32B text encoder and does not release it. Below this it dies."""

IMAGE_PULL_WARN_S = 300.0
IMAGE_PULL_ABANDON_S = 600.0

LICENCE_EXCLUDED_REGIONS = ("US", "EU", "UK", "KR")
"""MiniMax H3's licence excludes the United States, EU, UK, and South Korea.

Placement is a licensing question here, not only a latency one. Note that
Iceland (`EUR-IS-*`) and Norway (`EUR-NO-*`) are EEA but not EU, while Romania
(`EU-RO-1`), Czechia, France, the Netherlands and Sweden are EU.
"""

LICENCE_SAFE_DATACENTERS = (
    "EUR-IS-1", "EUR-IS-2", "EUR-IS-3", "EUR-IS-4",
    "EUR-NO-1", "EUR-NO-2",
    "CA-MTL-1", "CA-MTL-3", "CA-MTL-4",
    "AP-JP-1", "AP-IN-1",
    "OC-AU-1",
)

PREFERRED_GPUS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA A40",
)
"""In preference order. H3 peaks at 15.7-21.9 GB, so 24 GB is ample and the
48 GB cards are only a fallback for when 4090 stock is thin — which it usually
is. A 5090 is *not* in this list: at the time of writing it offers only CUDA
13.1/13.2, and H3's quantised fast paths want CUDA 13.0."""

REQUIRED_CUDA = "13.0"
"""H3's quantisation fast paths assume the CUDA 13 generation. On 12.8 the
strong cards fall back to a slow path. [reported]"""


@dataclass(frozen=True)
class PodStatus:
    """A snapshot of one pod."""

    id: str
    name: str
    status: str
    data_center_id: str | None
    gpu_id: str | None
    cuda_version: str | None
    host_ram_gb: float | None
    cost_per_hr: float | None
    uptime_s: float
    raw: dict[str, Any]

    @property
    def is_running(self) -> bool:
        return self.status.upper() in {"RUNNING", "READY"}

    @property
    def ram_is_sufficient(self) -> bool | None:
        """None when RAM could not be determined — not the same as 'fine'."""
        if self.host_ram_gb is None:
            return None
        return self.host_ram_gb >= MIN_HOST_RAM_GB

    def cost_so_far_usd(self) -> float | None:
        if self.cost_per_hr is None:
            return None
        return round(self.cost_per_hr * self.uptime_s / 3600.0, 4)


class PodManager:
    """Create, inspect, and terminate the ComfyUI pod."""

    def __init__(self, api_key: str | None = None, *, timeout_s: float = 60.0) -> None:
        settings = get_settings()
        key = api_key or (
            settings.runpod_api_key.get_secret_value() if settings.runpod_api_key else None
        )
        if not key:
            raise PodError(
                "RUNPOD_API_KEY is not set. Put it in .env, or export it. "
                "Pod operations cannot run without it."
            )
        self._client = httpx.Client(
            base_url=API_BASE,
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {key}"},
        )

    def __enter__(self) -> PodManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------ capacity

    def find_capacity(
        self,
        *,
        gpus: tuple[str, ...] = PREFERRED_GPUS,
        datacenters: tuple[str, ...] = LICENCE_SAFE_DATACENTERS,
    ) -> tuple[str, str]:
        """Return the first `(gpu_id, datacenter_id)` with reported stock.

        Raises rather than deploying into a queue. A request with no capacity
        can become a standing order that bills the moment stock appears.
        """
        for gpu_id in gpus:
            payload = self._get(f"/catalog/gpu-types/{_quote(gpu_id)}", params={"product": "POD"})
            for entry in payload.get("dataCenters", []) or []:
                dc_id = str(entry.get("id", ""))
                availability = str(entry.get("availability", "NONE")).upper()
                if dc_id in datacenters and availability not in {"NONE", ""}:
                    return gpu_id, dc_id

        raise PodError(
            "no licence-safe datacenter currently reports stock for any of "
            f"{list(gpus)}. Checked {list(datacenters)}. Wait and retry — do not "
            "configure an auto-deploy reservation, which bills whenever capacity "
            "appears, including overnight."
        )

    # --------------------------------------------------------------- create

    def up(
        self,
        *,
        template_id: str,
        name: str = "videogen-comfyui",
        gpu_id: str | None = None,
        data_center_id: str | None = None,
        cuda_version: str = REQUIRED_CUDA,
        enforce_ram: bool = True,
    ) -> PodStatus:
        """Deploy a pod, then verify the host is actually usable.

        If the allocated host has less than `MIN_HOST_RAM_GB`, the pod is
        terminated immediately and `InsufficientHostRamError` is raised. Keeping
        a short host and hoping is how you pay for a run that dies at clip two.
        """
        if gpu_id is None or data_center_id is None:
            gpu_id, data_center_id = self.find_capacity()

        body = {
            "name": name,
            "templateId": template_id,
            "cloud": "SECURE",
            "dataCenterIds": [data_center_id],
            "gpu": {
                "id": gpu_id,
                "count": 1,
                "allowedCudaVersions": [cuda_version],
            },
            "startSsh": True,
            "startJupyter": False,
        }
        created = self._post("/pods", json=body)
        pod_id = str(created.get("id") or "")
        if not pod_id:
            raise PodError(f"pod creation returned no id: {created!r}")

        status = self.status(pod_id)
        if enforce_ram and status.ram_is_sufficient is False:
            self.down(pod_id)
            raise InsufficientHostRamError(
                f"allocated host has {status.host_ram_gb:.0f} GB RAM, below the "
                f"{MIN_HOST_RAM_GB} GB MiniMax H3 needs. Pod {pod_id} was terminated. "
                "Retry to be allocated a different machine."
            )
        return status

    # --------------------------------------------------------------- status

    def status(self, pod_id: str) -> PodStatus:
        return _parse_status(self._get(f"/pods/{pod_id}"))

    def list_pods(self) -> list[PodStatus]:
        payload = self._get("/pods")
        items = payload.get("items", payload if isinstance(payload, list) else [])
        return [_parse_status(p) for p in items if isinstance(p, dict)]

    def health_warnings(self, status: PodStatus) -> list[str]:
        """Operational warnings a human should act on."""
        warnings: list[str] = []

        if not status.is_running and status.uptime_s > IMAGE_PULL_ABANDON_S:
            warnings.append(
                f"pod has been {status.status} for {status.uptime_s / 60:.0f} min. "
                "Past ~10 minutes this is a bad machine, not slow progress — "
                "terminate and redeploy rather than continuing to pay for it."
            )
        elif not status.is_running and status.uptime_s > IMAGE_PULL_WARN_S:
            warnings.append(
                f"pod still {status.status} after {status.uptime_s / 60:.0f} min; "
                "watch it, and terminate if it passes 10."
            )

        if status.ram_is_sufficient is False:
            warnings.append(
                f"host RAM {status.host_ram_gb:.0f} GB is below the {MIN_HOST_RAM_GB} GB "
                "H3 needs; expect a crash on the second consecutive generation."
            )
        elif status.ram_is_sufficient is None:
            warnings.append("host RAM could not be determined from the API response.")

        if status.cuda_version and not status.cuda_version.startswith("13"):
            warnings.append(
                f"host CUDA is {status.cuda_version}; H3's quantised fast paths "
                "assume CUDA 13 and will fall back to a slow path below it."
            )

        return warnings

    # ------------------------------------------------------------ terminate

    def down(self, pod_id: str) -> None:
        """Terminate. Deliberately not 'stop'.

        Stopping keeps the container disk and keeps charging for it. There is no
        `stop()` on this class because every time it would be reached for, the
        right answer is terminate.
        """
        self._post(f"/pods/{pod_id}/action", json={"action": "terminate"})

    # --------------------------------------------------------------- private

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise PodError(f"{method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            raise PodError(f"{method} {path} -> {response.status_code}: {response.text[:1000]}")
        if not response.content:
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {"items": payload}


def _parse_status(payload: dict[str, Any]) -> PodStatus:
    gpu = payload.get("gpu") or {}
    cpu = payload.get("cpu") or {}
    runtime = payload.get("runtime") or {}
    machine = payload.get("machine") or {}

    return PodStatus(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        status=str(payload.get("status", "UNKNOWN")),
        data_center_id=payload.get("dataCenterId"),
        gpu_id=gpu.get("id") if isinstance(gpu, dict) else None,
        cuda_version=payload.get("cudaVersion"),
        host_ram_gb=_ram_gb(cpu, runtime, machine, payload),
        cost_per_hr=_as_float(payload.get("cost", {}).get("perHr"))
        if isinstance(payload.get("cost"), dict)
        else _as_float(payload.get("costPerHr")),
        uptime_s=_uptime_s(payload),
        raw=payload,
    )


def _ram_gb(*sources: dict[str, Any]) -> float | None:
    """Dig host RAM out of whichever field carries it.

    Reported under different keys depending on the object, and absent entirely
    on some responses. Returning None is the honest answer; callers must treat
    'unknown' as 'unverified', not as 'sufficient'.
    """
    keys = ("memoryInGb", "memoryGb", "ramInGb", "memory", "ram")
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = _as_float(source.get(key))
            if value:
                # Some fields report bytes or MB rather than GB.
                if value > 1 << 20:
                    return value / (1 << 30)
                if value > 4096:
                    return value / 1024
                return value
    return None


def _uptime_s(payload: dict[str, Any]) -> float:
    started = payload.get("startedAt") or payload.get("createdAt")
    if not started:
        return 0.0
    try:
        from datetime import datetime

        stamp = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        return max(0.0, time.time() - stamp.timestamp())
    except (ValueError, TypeError):
        return 0.0


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")
