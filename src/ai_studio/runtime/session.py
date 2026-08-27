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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from ai_studio.config.settings import get_settings
from ai_studio.core.errors import CostCeilingExceeded, PodError
from ai_studio.pipeline.queue import JobQueue
from ai_studio.runtime import hours
from ai_studio.runtime.budget import MonthlyBudgetGuard, SpendLedger

TEMPLATE_COMFYUI_STANDARD = "cw3nka7d08"
"""Official RunPod "ComfyUI" template, for standard GPUs (RTX 4090, L40, A100,
etc — the only GPUs `CANDIDATES` below ever asks for). Confirmed against
RunPod's own tutorial (docs.runpod.io/tutorials/pods/comfyui) on 2026-08-25:
the id previously hardcoded here (`2lv7ev3wfp`) has since been renamed/rescoped
to "ComfyUI Blackwell Edition" and is now specifically for RTX 5090/B200 — the
wrong template for this project's ladder.

⚠️ Open question, not yet resolved: RunPod's current docs only mention two
image tags, `runpod/comfyui:latest` (standard) and `runpod/comfyui:cuda12.8`
(Blackwell) — no "cuda13" tag is documented anywhere today, which is in
tension with docs/model-h3.md's claim that H3 needs CUDA 13.0 specifically
("on 12.8 strong cards fall back to a slow path"). Verify what CUDA version
this template actually runs (console template detail page, or
`runpodctl template search comfyui`) before trusting H3's turbo path on it.
"""

TERMINATE_BUFFER_MIN = 10
"""How far past window end `--terminate-after` is set. The buffer covers a clip
that is mid-render at the bell; the flag is the backstop, not the mechanism."""


@dataclass(frozen=True)
class Tier:
    """One rung of the placement ladder."""

    gpu: str
    datacenter: str
    cloud: str
    vram_gb: int
    usd_per_hr: float
    wait: bool = False
    """Whether to retry this rung until capacity appears, rather than moving on."""

    @property
    def low_vram(self) -> bool:
        """True when the card cannot hold the model in the sharpest LoRA mode.

        A measured run peaked at **43.3GB**, so anything under 48GB must run the
        turbo node's `low_vram=True`, which its own docs describe as *softer on
        quantized bases*. This is why falling down the ladder is a quality
        change and not only a price change.
        """
        return self.vram_gb < 48

    @property
    def label(self) -> str:
        short = self.gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        return f"{short}/{self.cloud}"

    @property
    def quantisation(self) -> str:
        """fp8 on Ada and newer with room to spare; int8 where VRAM is tight."""
        return "int8" if self.low_vram else "fp8"


# The ladder, price strictly descending. Only licence-permitted datacenters:
# MiniMax H3's licence excludes the US, EU, UK and South Korea, and Iceland is
# EEA but not EU.
#
# Descending on purpose. Grab the best card available *now* and move on
# immediately if it is gone — wall-clock inside a two-hour window is worth more
# than the price difference. Only the cheapest rung is worth waiting for: what
# you eventually get is the least expensive option, and a softer clip beats no
# clip at all.
#
# Prices include the ~$0.014/hr our disk configuration adds, which is measured:
# a pod listed at $0.44/hr reported currentSpendPerHr 0.454.
# The datacenter is not decoration and it is not the same for every rung.
#
# 📏 Checked against /catalog/gpus on 2026-08-25: L40S secure stock exists only
# in EU-NL-1, OC-AU-1, US-NC-1, US-TX-3 and US-TX-4. Of those, H3's licence
# permits only OC-AU-1 -- the Netherlands is EU, the rest are US. There is no
# L40S in Iceland at all, so the earlier reading that "L40S was refused twice"
# was geography, not stock: those two rungs could never have been filled.
#
# The 4090 rungs stay in Iceland (EUR-IS-2), which is where 4090 secure stock
# actually is among licence-safe datacenters (the alternatives the catalog
# offers, EU-CZ-1 and EU-RO-1, are both EU).
CANDIDATES: tuple[Tier, ...] = (
    Tier("NVIDIA L40S", "OC-AU-1", "SECURE", vram_gb=48, usd_per_hr=1.004),
    Tier("NVIDIA L40S", "OC-AU-1", "COMMUNITY", vram_gb=48, usd_per_hr=0.804),
    Tier("NVIDIA GeForce RTX 4090", "EUR-IS-2", "SECURE", vram_gb=24, usd_per_hr=0.754),
    Tier(
        "NVIDIA GeForce RTX 4090", "EUR-IS-2", "COMMUNITY",
        vram_gb=24, usd_per_hr=0.354, wait=True,
    ),
)

def candidates_for_volume(datacenter: str) -> tuple[Tier, ...]:
    """The ladder when a network volume is in play: one rung, in the volume's
    datacenter, on secure cloud, and worth waiting for.

    Network volumes are secure-cloud only and mount only in their own
    datacenter, so the community rungs and OC-AU-1's L40S cannot be used with
    one. That leaves the 4090 secure rung wherever the volume lives -- and
    since waiting for it beats paying another full weight download, `wait`
    is on. The datacenter still has to be licence-safe; a volume created in
    the wrong region would otherwise smuggle a pod into it.
    """
    from ai_studio.runtime.pod import LICENCE_SAFE_DATACENTERS

    if datacenter not in LICENCE_SAFE_DATACENTERS:
        raise PodError(
            f"network volume lives in {datacenter}, which is not licence-safe for H3 "
            f"({', '.join(LICENCE_SAFE_DATACENTERS)})"
        )
    return (
        Tier("NVIDIA GeForce RTX 4090", datacenter, "SECURE", vram_gb=24, usd_per_hr=0.754, wait=True),
    )


def volume_datacenter(volume_id: str) -> str:
    """Where a network volume lives, from RunPod, so nothing here has to be
    told twice and drift."""
    info = _runpodctl("network-volume", "get", volume_id)
    datacenter = str(info.get("dataCenterId") or "")
    if not datacenter:
        raise PodError(f"network volume {volume_id}: no dataCenterId in {info}")
    return datacenter


def placement() -> tuple[tuple[Tier, ...], str | None]:
    """(ladder, network volume id) for this deployment, from settings."""
    volume_id = get_settings().network_volume_id
    if not volume_id:
        return CANDIDATES, None
    return candidates_for_volume(volume_datacenter(volume_id)), volume_id


WAIT_RETRY_INTERVAL_S = 30.0
WAIT_MAX_S = 45 * 60
"""How long the cheapest rung is retried before the window is given up.

Retrying is done here, in our own loop. It is emphatically **not** done by
leaving a request with RunPod: an order that fills when capacity appears would
start billing unattended, including overnight.
"""

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
    tier_label: str = ""
    vram_gb: int = 0
    low_vram: bool = False
    """True when this rung forces the softer LoRA mode. Recorded so a
    disappointing clip can be traced to the card it was rendered on."""
    quantisation: str = "int8"
    ssh: dict[str, Any] = field(default_factory=dict)
    provisioned: bool = False
    """Whether `deploy/pod_setup.sh` has been started on this pod. Set by
    `mark_provisioned`; the worker provisions once and then only waits."""
    last_media_kind: str = ""
    """What the last render was ("image"/"video"), so the reaper can give a
    video pod -- whose model takes 90 s to reload -- a longer grace."""

    @property
    def comfy_url(self) -> str:
        return f"https://{self.pod_id}-8188.proxy.runpod.net"

    @property
    def inference_url(self) -> str:
        """The understanding-model server's proxy URL. Separate process,
        separate port (8189) -- see `deploy/inference_server.py`."""
        return f"https://{self.pod_id}-8189.proxy.runpod.net"

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
    name: str = "ai-studio-window",
    volume_gb: int = 80,
    candidates: tuple[Tier, ...] = CANDIDATES,
    network_volume_id: str | None = None,
) -> Session:
    """Deploy a pod for this window, or raise if nothing is available.

    With `network_volume_id` the pod mounts that volume at /workspace instead
    of a fresh `volume_gb` disk; pair it with `candidates_for_volume(...)`,
    since the volume only mounts in its own datacenter (see `placement()`).

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
    for tier in candidates:
        deadline = time.time() + (min(WAIT_MAX_S, _seconds_left(window_end) / 2) if tier.wait else 0)

        while True:
            try:
                created = _runpodctl(
                    "pod", "create",
                    "--name", name,
                    "--template-id", TEMPLATE_COMFYUI_STANDARD,
                    "--gpu-id", tier.gpu,
                    "--data-center-ids", tier.datacenter,
                    "--cloud-type", tier.cloud,
                    *(
                        ("--network-volume-id", network_volume_id)
                        if network_volume_id
                        else ("--volume-in-gb", str(volume_gb))
                    ),
                    "--ports", "8188/http,8888/http,8189/http,22/tcp",
                    "--ssh",
                    "--min-cuda-version", "13.0",
                    # The backstop. If this process dies, the pod still terminates.
                    "--terminate-after", stamp,
                    timeout_s=300.0,
                )
            except PodError as exc:
                # Only the cheapest rung is worth waiting on; the others give way
                # immediately so the window is not spent queueing.
                if tier.wait and time.time() < deadline:
                    time.sleep(WAIT_RETRY_INTERVAL_S)
                    continue
                failures.append(f"{tier.label} @ {tier.datacenter}: {exc}")
                break

            pod_id = str(created.get("id") or "")
            if not pod_id:
                failures.append(f"{tier.label}: no id in response")
                break

            session = Session(
                pod_id=pod_id,
                gpu=tier.gpu,
                datacenter=tier.datacenter,
                cloud=tier.cloud,
                cost_per_hr=float(created.get("costPerHr") or tier.usd_per_hr),
                opened_at=datetime.now(timezone.utc).isoformat(),
                window_end=window_end.astimezone(timezone.utc).isoformat(),
                tier_label=tier.label,
                vram_gb=tier.vram_gb,
                low_vram=tier.low_vram,
                quantisation=tier.quantisation,
            )
            save_state(session)
            return session

    raise PodError(
        "no licence-safe capacity on any rung. Nothing was created and nothing "
        "is billing. Tried:\n  - " + "\n  - ".join(failures)
    )


def _seconds_left(window_end: datetime) -> float:
    return max(0.0, (window_end.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())


# -------------------------------------------------------------- request-driven


def ensure_pod(
    queue: JobQueue,
    *,
    name: str = "ai-studio-window",
    candidates: tuple[Tier, ...] | None = None,
    now: datetime | None = None,
    max_opens_per_day: int | None = None,
) -> Session:
    """Return a live window: the open one if there is one, else open one if
    business hours allow it.

    This is the request-driven replacement for the 03:00 UTC `open` timer. The
    timer opened a pod whether or not anyone had asked for anything; this
    *creates* one only when a caller has work in hand and business hours
    allow it. Reusing an already-open pod is not gated on the clock at all --
    that pod is billing either way, and the gate exists to stop new spend, not
    to idle spend that already happened. `open_session` itself is unchanged —
    the deploy logic was already the right shape, only the moment it runs has
    moved.

    The lease is the end of business hours, so `--terminate-after` lands at
    13:10 without anybody inventing a second deadline for it.

    Checked in this order, and the order is deliberate:

    0. **An already-open window is reused, not doubled.** A pod that is
       already live is already billing; not using it would only leave
       paid-for GPU-minutes idle. `open_session` would refuse a second pod
       under the same name anyway, but that refusal is an error and this is
       the ordinary case: the second request of the hour.
    1. **Opens per day.** The failure the monthly guard cannot see — a worker
       that crash-loops opens a fresh pod on every restart, and each one is
       individually inside budget. With the reaper closing pods minutes after
       the last render this is a real count, hence the cap of fifteen rather
       than two.
    2. **Monthly budget.** Same guard, same pessimistic worst-rung arithmetic,
       on the path that actually creates pods.

    There is no clock gate any more (see `runtime.hours`). The lease is
    `LEASE_HOURS` from now; the reaper is expected to close the pod long
    before it, and `--terminate-after` lands ten minutes after it.
    """
    live = load_state()
    if live is not None and not live.past_window():
        return live

    settings = get_settings()
    network_volume_id: str | None = None
    if candidates is None:
        candidates, network_volume_id = placement()
    limit = settings.max_pod_opens_per_day if max_opens_per_day is None else max_opens_per_day
    opened = queue.opens_today(since=hours.day_start(now).timestamp())
    if limit and opened >= limit:
        raise CostCeilingExceeded(
            f"{opened} pod(s) already opened today, cap is {limit} "
            "(AI_STUDIO_MAX_POD_OPENS_PER_DAY). Nothing was created and nothing "
            "is billing."
        )

    guard = MonthlyBudgetGuard(
        SpendLedger(),
        cap_usd=settings.max_month_usd,
        vps_monthly_usd=settings.vps_monthly_usd,
    )
    guard.refuse_if_broke(candidates)
    window_end = guard.throttle(
        hours.window_end_for(now),
        now or datetime.now(timezone.utc),
        max(tier.usd_per_hr for tier in candidates),
    )

    session = open_session(
        window_end, name=name, candidates=candidates, network_volume_id=network_volume_id
    )
    # Recorded before the caller does anything else with the session: a pod
    # that exists but was never counted is one the daily cap cannot see.
    queue.record_pod_open(session.pod_id)
    return session


READY_NODE = "MiniMaxH3TurboLoRA"
"""The node whose presence means `deploy/pod_setup.sh` has finished. It is
the last thing the script installs before its own readiness check, and it is
the node the H3 graphs cannot run without."""


def wait_ready(
    session: Session,
    *,
    timeout_s: float = 900.0,
    interval_s: float = 15.0,
    request_timeout_s: float = 10.0,
) -> float:
    """Block until ComfyUI answers on the pod, and return how long that took.

    `deploy/pod_setup.sh` waits for the same endpoint from inside the pod; this
    is the outside view, and it is the one that matters to a caller about to
    submit. The pod answers 502 through the proxy for several minutes while it
    copies itself to /workspace, so a single request proves nothing.

    Every request is given its own short timeout rather than one long one:
    RunPod's proxy is severed by Cloudflare at ~100 seconds, so a request that
    blocks for longer is not patience, it is a hang.

    "Answers" means answers *with the H3 node pack installed*: the stock
    template's ComfyUI comes up in seconds and returns 200 on `/system_stats`
    long before `deploy/pod_setup.sh` has installed the turbo nodes, and a
    submission in that gap is rejected with `missing_node_type` -- observed
    live, and it burned two of a job's three attempts before setup finished.
    So the probe is `/object_info/<node>`, which only lists a node ComfyUI can
    actually run.

    Raises `PodError` on timeout — the pod is up and billing but unusable, and
    that is precisely the state that must not be mistaken for "nothing is
    wrong".
    """
    url = f"{session.comfy_url}/object_info/{READY_NODE}"
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    last = "no attempt made"

    while True:
        try:
            response = httpx.get(url, timeout=request_timeout_s)
            if response.status_code == 200 and READY_NODE in response.json():
                return time.monotonic() - started
            last = (
                f"HTTP {response.status_code}"
                if response.status_code != 200
                else f"ComfyUI is up but {READY_NODE} is not installed yet (pod_setup.sh?)"
            )
        except (httpx.HTTPError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise PodError(
                f"{session.pod_id} did not answer {url} within {timeout_s:.0f}s "
                f"(last: {last}). The pod is running and billing -- close it."
            )
        time.sleep(interval_s)


def wait_understanding_ready(
    session: Session,
    *,
    timeout_s: float = 300.0,
    interval_s: float = 10.0,
    request_timeout_s: float = 10.0,
) -> float:
    """Block until `deploy/inference_server.py` answers on the pod.

    A different readiness signal from `wait_ready`'s, deliberately: that
    server has no node-pack concept and does not need one loaded to be
    "ready" -- moondream3/Qwen3-Omni-Captioner/Tarsier2 load lazily per
    request, not at startup. A plain `/healthz` is the whole check, so this
    is expected to resolve well inside the shorter default timeout above.
    """
    url = f"{session.inference_url}/healthz"
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    last = "no attempt made"

    while True:
        try:
            response = httpx.get(url, timeout=request_timeout_s)
            if response.status_code == 200:
                return time.monotonic() - started
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise PodError(
                f"{session.pod_id} did not answer {url} within {timeout_s:.0f}s "
                f"(last: {last}). The pod is running and billing -- close it."
            )
        time.sleep(interval_s)


# ----------------------------------------------------------------------- close


def close_session(*, name: str = "ai-studio-window") -> list[str]:
    """Terminate the window's pod. Idempotent, and safe with no state file.

    Terminates rather than stops: a stopped pod keeps its disk and keeps
    charging for it. Falls back to matching by name so a lost state file cannot
    strand a billing machine.

    Records the session's cost into the monthly `SpendLedger` here, not in the
    CLI — this is the one place every close path (the scheduled `session
    close`, and `close_if_idle`'s early closes, which are what actually end
    the window on almost every real day given how much headroom the window
    has over typical demand) funnels through. Recording it only where the CLI
    happens to call this would silently starve the monthly budget guard of
    real spend data on the common path, exactly the "wired but not actually
    wired" failure this project has already paid for once (see
    `tests/unit/test_drain_wiring.py`).
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

    if session is not None:
        SpendLedger().record_session(
            session.spent_usd(),
            tier_label=session.tier_label,
            minutes=session.elapsed_hours() * 60,
        )

    clear_state()
    return terminated


IMAGE_IDLE_MINUTES = 5
VIDEO_IDLE_MINUTES = 10
UNDERSTANDING_IDLE_MINUTES = 5
"""How long a quiet pod is kept after its last render, by what it rendered.

Numbers differ because the reloads cost differently: Flux comes back into
VRAM in 📏 ~15 s, H3's 32B text encoder in 📏 60-90 s. `UNDERSTANDING_IDLE_MINUTES`
is `[speculative]` -- nothing has measured a lazy-load cost for moondream3,
Qwen3-Omni-Captioner, or Tarsier2 on this hardware yet, and the three are not
even the same size, so one shared number is a starting point, not a
considered answer. A pod is worth keeping only while the chance of the next
request within the grace, times the reload it would save, beats the idle
minutes -- and in a group chat the next message usually comes within five
minutes or not for hours. The reaper log says how often a pod was closed and
reopened within a few minutes, which is the number that tunes these.
"""


def close_if_idle(
    *,
    image_idle_minutes: int = IMAGE_IDLE_MINUTES,
    video_idle_minutes: int = VIDEO_IDLE_MINUTES,
    understanding_idle_minutes: int = UNDERSTANDING_IDLE_MINUTES,
    hold: bool = False,
    name: str = "ai-studio-window",
) -> str:
    """Close the pod when it has gone quiet; the grace depends on what it
    last rendered. `hold=True` (work is waiting in the queue) never closes:
    a pod with a job about to land on it is not idle, whatever the clock says
    -- closing it there is the one move that costs a cold open *and* the wait.
    """
    session = load_state()
    if session is None:
        return "no session"

    if session.past_window():
        return f"window over; {close_session(name=name)}"

    state = _read_state_raw()
    last = state.get("last_activity_at") or session.opened_at
    idle = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
    overrides = {
        "image": image_idle_minutes,
        "video": video_idle_minutes,
        "image_understand": understanding_idle_minutes,
        "audio_understand": understanding_idle_minutes,
        "video_understand": understanding_idle_minutes,
    }
    grace = overrides.get(str(state.get("last_media_kind") or ""), video_idle_minutes)
    if hold:
        return f"held: work pending ({idle:.0f}min idle, spent ${session.spent_usd():.2f})"
    if idle >= grace:
        return f"idle {idle:.0f}min >= {grace}; {close_session(name=name)}"
    return f"active ({idle:.0f}min idle of {grace}, spent ${session.spent_usd():.2f})"


# ------------------------------------------------------------------ provision

SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
SETUP_SCRIPT = Path("deploy/pod_setup.sh")
INFERENCE_SERVER_SCRIPT = Path("deploy/inference_server.py")
PROVISION_WAIT_S = 600.0
PROVISION_RETRY_S = 15.0


def _ssh(argv: list[str], *, stdin: str, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, input=stdin, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout_s, check=False,
    )


def _ssh_deposit(host: str, port: str, body: str, *, remote_path: str) -> None:
    """Copy one file's content to `remote_path` over SSH, retrying the
    connection the same way `provision` does below (the pod's SSH port comes
    up some seconds after RunPod reports it running). Deposits only -- does
    not execute anything, unlike `provision`'s own SSH call."""
    argv = [
        "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-p", port, f"root@{host}",
        f"cat > {remote_path}",
    ]
    deadline = time.monotonic() + PROVISION_WAIT_S
    last = ""
    while True:
        try:
            proc = _ssh(argv, stdin=body, timeout_s=60.0)
        except subprocess.TimeoutExpired:
            last = "ssh timed out"
        else:
            if proc.returncode == 0:
                return
            last = (proc.stderr or proc.stdout).strip()[-200:]
        if time.monotonic() >= deadline:
            raise PodError(f"could not copy {remote_path} over ssh ({last})")
        time.sleep(PROVISION_RETRY_S)


def provision(
    session: Session,
    *,
    script: Path = SETUP_SCRIPT,
    inference_script: Path = INFERENCE_SERVER_SCRIPT,
) -> None:
    """Start `deploy/pod_setup.sh` on the pod, detached, over SSH.

    This is the step that used to be a person with a terminal. The worker
    opens a pod and then has to wait for it; nobody else is there to run the
    setup, so the worker does -- copies the script up and starts it under
    nohup, so a dropped SSH link (observed live, mid-download) does not kill
    it. `wait_ready` then watches for the node pack, which is the script's
    last act. With the weights on a network volume the script's own fast
    path makes this a ComfyUI restart, about a minute.

    `deploy/inference_server.py` (the understanding-model server) is
    deposited first, as a second file over the same one-file-at-a-time SSH
    transport -- it cannot be embedded inline in `pod_setup.sh` without that
    script becoming unreadable, and `pod_setup.sh`'s own last step starts it
    once it is in place.

    Retries the SSH connection for up to `PROVISION_WAIT_S`: the pod's SSH
    port comes up some seconds after RunPod reports it running.
    """
    info = _runpodctl("ssh", "info", session.pod_id)
    host, port = str(info.get("ip") or ""), str(info.get("port") or "")
    if not host or not port:
        raise PodError(f"{session.pod_id}: no ssh endpoint in {info}")

    _ssh_deposit(
        host, port, inference_script.read_text(encoding="utf-8"),
        remote_path="/workspace/inference_server.py",
    )

    body = script.read_text(encoding="utf-8")
    argv = [
        "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-p", port, f"root@{host}",
        # `;` not `&&`: `cat > f && nohup ... &` backgrounds the cat too, which
        # then races the closing ssh stdin and writes an empty file (observed
        # live, 2026-08-27). Separated, cat drains stdin in the foreground and
        # only the setup run is backgrounded.
        "cat > /workspace/pod_setup.sh; nohup bash /workspace/pod_setup.sh "
        f"{session.vram_gb} > /workspace/setup.log 2>&1 < /dev/null & disown; echo started",
    ]
    deadline = time.monotonic() + PROVISION_WAIT_S
    last = ""
    while True:
        try:
            proc = _ssh(argv, stdin=body, timeout_s=60.0)
        except subprocess.TimeoutExpired:
            last = "ssh timed out"
        else:
            if proc.returncode == 0 and "started" in proc.stdout:
                return
            last = (proc.stderr or proc.stdout).strip()[-200:]
        if time.monotonic() >= deadline:
            raise PodError(
                f"{session.pod_id}: could not start pod_setup.sh over ssh "
                f"({last}). The pod is running and billing -- close it."
            )
        time.sleep(PROVISION_RETRY_S)


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


def touch_activity(media_kind: str | None = None) -> None:
    """Record that work just happened, so the idle timer restarts, and what
    kind it was, so the reaper can pick the right grace."""
    raw = _read_state_raw()
    if raw:
        raw["last_activity_at"] = datetime.now(timezone.utc).isoformat()
        if media_kind:
            raw["last_media_kind"] = media_kind
        STATE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def mark_provisioned() -> None:
    raw = _read_state_raw()
    if raw:
        raw["provisioned"] = True
        STATE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def _read_state_raw() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        # This file is how `session close` finds a running pod. Guessing at a
        # shape we do not recognise risks reporting "nothing is billing".
        raise PodError(f"{STATE_FILE} is not a JSON object: {type(data).__name__}")
    return data
