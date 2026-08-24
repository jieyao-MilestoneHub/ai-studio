"""Service-window management: open a pod, serve for a window, always close it.

The window model exists because the fixed cost of a session is ~20 minutes
(boot, weight download, node install) while a clip is ~5 minutes. Opening a pod
per request spends 80% of the money on setup; opening one window a day amortises
it across everything that arrives inside the window.

Three properties this module is built to guarantee, in order of how much a
failure costs:

1. **A pod is never left running.** Every create passes `--terminate-after`, so
   even if this process dies, the machine self-terminates. `close()` is
   idempotent and will hunt for pods by name if it has no state file.
2. **The window never silently fails to open.** 4090 capacity is thin; opening
   retries across a ladder rather than giving up on the first refusal.
3. **Idle time inside the window is not paid for.** A 3.8h window at 50 clips a
   month is mostly idle; `close_if_idle` lets a scheduler reclaim it.

Why this shells out to `runpodctl` rather than using `PodManager`: the CLI path
is the one proven end to end on this project, it is a single static binary, and
it resolves credentials from `~/.runpod/config.toml` without extra wiring — all
of which matter more for an unattended scheduled job than architectural purity.
`PodManager` (REST v2) remains the library path but has **not** been executed
against the live API yet.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from videogen.core.errors import PodError

TEMPLATE_COMFYUI_CUDA13 = "2lv7ev3wfp"
"""Official Runpod "ComfyUI - CUDA 13" template. CUDA 12.8 makes H3's quantised
fast paths fall back to a slow route, so the version is not interchangeable."""

TERMINATE_BUFFER_MIN = 10
"""How far past window end `--terminate-after` is set. The buffer covers a clip
that is mid-render at the bell; the flag is the backstop, not the mechanism."""

# Placement ladder. Ordered by cost, filtered to datacenters the MiniMax H3
# licence permits (it excludes the US, EU, UK and South Korea). Every 24GB entry
# needs low_vram=True: a measured run peaked at 43.3GB in bypass mode.
CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("NVIDIA GeForce RTX 4090", "EUR-IS-2", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "EUR-IS-2", "SECURE"),
    ("NVIDIA GeForce RTX 5090", "EUR-IS-1", "SECURE"),
    ("NVIDIA L40S", "EUR-IS-2", "COMMUNITY"),
)

STATE_FILE = Path("runs/.session.json")


@dataclass(frozen=True)
class Session:
    """A live service window."""

    pod_id: str
    gpu: str
    datacenter: str
    cloud: str
    cost_per_hr: float
    opened_at: str
    window_end: str
    ssh: dict[str, Any] = field(default_factory=dict)

    @property
    def comfy_url(self) -> str:
        return f"https://{self.pod_id}-8188.proxy.runpod.net"

    def elapsed_hours(self, now: datetime | None = None) -> float:
        started = datetime.fromisoformat(self.opened_at)
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - started).total_seconds() / 3600)

    def spent_usd(self, now: datetime | None = None) -> float:
        return round(self.cost_per_hr * self.elapsed_hours(now), 4)

    def past_window(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= datetime.fromisoformat(self.window_end)


# --------------------------------------------------------------------- runpodctl


def _runpodctl(*args: str, timeout_s: float = 180.0) -> dict[str, Any]:
    """Run runpodctl and parse its JSON. Raises `PodError` with its own message."""
    argv = ["runpodctl", *args, "-o", "json"]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise PodError(
            "runpodctl not found on PATH. Install it from "
            "https://github.com/runpod/runpodctl/releases"
        ) from None
    except subprocess.TimeoutExpired:
        raise PodError(f"runpodctl {' '.join(args)} timed out after {timeout_s}s") from None

    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise PodError(f"runpodctl {' '.join(args)} -> unparseable output: {text[:400]}") from None

    if isinstance(payload, dict) and payload.get("error"):
        raise PodError(str(payload["error"]))
    return payload if isinstance(payload, dict) else {"items": payload}


def list_pods() -> list[dict[str, Any]]:
    result = _runpodctl("pod", "list")
    items = result.get("items", result)
    return items if isinstance(items, list) else []


# ------------------------------------------------------------------------ open


def open_session(
    window_end: datetime,
    *,
    name: str = "videogen-window",
    volume_gb: int = 80,
    candidates: tuple[tuple[str, str, str], ...] = CANDIDATES,
) -> Session:
    """Deploy a pod for this window, or raise if nothing is available.

    Deliberately raises rather than queueing: asking Runpod for a GPU with no
    stock can leave a standing order that starts billing whenever capacity turns
    up, including overnight. A failed window is cheap; a forgotten pod is not.
    """
    if existing := find_existing(name):
        raise PodError(
            f"pod {existing['id']} named {name!r} is already running "
            f"(${existing.get('costPerHr')}/hr). Close it before opening a new window."
        )

    terminate_at = (window_end + timedelta(minutes=TERMINATE_BUFFER_MIN)).astimezone(timezone.utc)
    stamp = terminate_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    failures: list[str] = []
    for gpu, datacenter, cloud in candidates:
        try:
            created = _runpodctl(
                "pod", "create",
                "--name", name,
                "--template-id", TEMPLATE_COMFYUI_CUDA13,
                "--gpu-id", gpu,
                "--data-center-ids", datacenter,
                "--cloud-type", cloud,
                "--volume-in-gb", str(volume_gb),
                "--ports", "8188/http,8888/http,22/tcp",
                "--ssh",
                "--min-cuda-version", "13.0",
                # The backstop. If this process dies, the pod still terminates.
                "--terminate-after", stamp,
                timeout_s=300.0,
            )
        except PodError as exc:
            failures.append(f"{gpu} @ {datacenter}/{cloud}: {exc}")
            continue

        pod_id = str(created.get("id") or "")
        if not pod_id:
            failures.append(f"{gpu} @ {datacenter}/{cloud}: no id in response")
            continue

        session = Session(
            pod_id=pod_id,
            gpu=gpu,
            datacenter=datacenter,
            cloud=cloud,
            cost_per_hr=float(created.get("costPerHr") or 0.0),
            opened_at=datetime.now(timezone.utc).isoformat(),
            window_end=window_end.astimezone(timezone.utc).isoformat(),
        )
        save_state(session)
        return session

    raise PodError(
        "no licence-safe capacity for any candidate GPU. Nothing was created and "
        "nothing is billing. Tried:\n  - " + "\n  - ".join(failures)
    )


# ----------------------------------------------------------------------- close


def close_session(*, name: str = "videogen-window") -> list[str]:
    """Terminate the window's pod. Idempotent, and safe with no state file.

    Terminates rather than stops: a stopped pod keeps its disk and keeps
    charging for it. Falls back to matching by name so a lost state file cannot
    strand a billing machine.
    """
    terminated: list[str] = []

    session = load_state()
    targets = {session.pod_id} if session else set()
    targets |= {str(p["id"]) for p in list_pods() if p.get("name") == name and p.get("id")}

    for pod_id in sorted(targets):
        try:
            _runpodctl("pod", "delete", pod_id)
            terminated.append(pod_id)
        except PodError as exc:
            # Keep going: one stuck pod must not stop us terminating the rest.
            terminated.append(f"{pod_id} (FAILED: {exc})")

    clear_state()
    return terminated


def close_if_idle(idle_minutes: int = 20, *, name: str = "videogen-window") -> str:
    """Close early when the window has gone quiet.

    A 3.8h window sized for peak demand is mostly idle at low volume, and idle
    minutes cost exactly as much as working ones. Requires the queue to report
    when it last finished work; see `last_activity_at` in the state file.
    """
    session = load_state()
    if session is None:
        return "no session"

    if session.past_window():
        return f"window over; {close_session(name=name)}"

    state = _read_state_raw()
    last = state.get("last_activity_at") or session.opened_at
    idle = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
    if idle >= idle_minutes:
        return f"idle {idle:.0f}min >= {idle_minutes}; {close_session(name=name)}"
    return f"active ({idle:.0f}min idle, spent ${session.spent_usd():.2f})"


def find_existing(name: str) -> dict[str, Any] | None:
    return next((p for p in list_pods() if p.get("name") == name), None)


# ----------------------------------------------------------------------- state


def save_state(session: Session) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = session.__dict__ | {"last_activity_at": session.opened_at}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state() -> Session | None:
    raw = _read_state_raw()
    if not raw:
        return None
    known = {f for f in Session.__dataclass_fields__}
    return Session(**{k: v for k, v in raw.items() if k in known})


def touch_activity() -> None:
    """Record that work just happened, so the idle timer restarts."""
    raw = _read_state_raw()
    if raw:
        raw["last_activity_at"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def _read_state_raw() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
