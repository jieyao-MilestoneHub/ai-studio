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
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from ai_studio import paths
from ai_studio.config.settings import get_settings
from ai_studio.core.errors import CostCeilingExceeded, PodError
from ai_studio.core.observability import utc_now_iso
from ai_studio.runtime import hours
from ai_studio.runtime.budget import MonthlyBudgetGuard, SpendLedger
from ai_studio.runtime.opens import PodOpenLedger

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
        """`<GPU>/<cloud>` as it appears in the ledger, the logs and the status page."""
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

_log = logging.getLogger("ai_studio.session")

STATE_FILE = Path("runs/.session.json")
SESSIONS_LOG_DIR = Path("logs/sessions")
"""Where every closed session's state lands (pod, tier, quantisation, opened/
closed, cost, why it closed). `.session.json` is unlinked at close, so before
2026-08-28 nothing but the pod id in the opens ledger and the cost in the spend ledger
survived a session. Monkeypatched in tests like STATE_FILE."""
PODS_LOG_DIR = Path("logs/pods")
"""Where a pod's own logs (setup.log, inference.log, comfy.log tail, dl-logs)
are pulled to over ssh before the pod is terminated -- they die with it."""
POD_LOG_PULL_TIMEOUT_S = 30.0
POD_LOG_TAIL_BYTES = 5_000_000


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
    last_activity_label: str = ""
    """What the caller said it just did (free text, for the reaper log)."""
    grace_minutes: float | None = None
    """How long the caller wants this pod kept after that activity; None
    means `DEFAULT_GRACE_MINUTES`. The caller knows what it rendered and what
    the next request is likely to be; this side only keeps the clock."""

    @property
    def comfy_url(self) -> str:
        """ComfyUI through RunPod's proxy (port 8188). Requests through it are cut
        at ~100 s, which is why every provider polls instead of blocking.
        """
        return f"https://{self.pod_id}-8188.proxy.runpod.net"

    @property
    def inference_url(self) -> str:
        """The understanding-model server's proxy URL. Separate process,
        separate port (8189) -- see `deploy/inference_server.py`."""
        return f"https://{self.pod_id}-8189.proxy.runpod.net"

    def elapsed_hours(self, now: datetime | None = None) -> float:
        """Hours since the pod opened, for the running cost."""
        started = datetime.fromisoformat(self.opened_at)
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - started).total_seconds() / 3600)

    def spent_usd(self, now: datetime | None = None) -> float:
        """What this session has cost so far at its tier's hourly rate."""
        return round(self.cost_per_hr * self.elapsed_hours(now), 4)

    def past_window(self, now: datetime | None = None) -> bool:
        """True once the lease has ended: the pod is about to terminate itself and
        must not be handed new work.
        """
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
    """Every pod on the account, as RunPod reports them."""
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
    opens: PodOpenLedger | None = None,
) -> Session:
    """Deploy a pod for this window, or raise if nothing is available.

    Every pod created here is counted in the daily open ledger (`opens`),
    manual `session open` included — a pod that exists but was never counted
    is one the daily cap cannot see.

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
                # immediately so the window is not spent queueing. Say so each
                # time: a silent 45-minute retry loop is indistinguishable from
                # a hung process to whoever is watching the terminal.
                if tier.wait and time.time() < deadline:
                    _log.warning(
                        "pod create failed on %s; retrying in %.0fs", tier.label, WAIT_RETRY_INTERVAL_S,
                        extra={"reason": str(exc)[:300], "datacenter": tier.datacenter,
                               "seconds": round(deadline - time.time())},
                    )
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
            # Counted before anything else can go wrong: a pod that exists but
            # was never counted is one the daily cap cannot see.
            (opens or PodOpenLedger()).record(pod_id)
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
    *,
    name: str = "ai-studio-window",
    candidates: tuple[Tier, ...] | None = None,
    now: datetime | None = None,
    max_opens_per_day: int | None = None,
    opens: PodOpenLedger | None = None,
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
    opens = opens or PodOpenLedger()
    opened = opens.count_since(hours.day_start(now).timestamp())
    if limit and opened >= limit:
        _log.warning("refused to open pod", extra={"reason": "daily opens cap", "opened_today": opened, "cap": limit})
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

    _log.info(
        "opening pod", extra={"reason": "queue has work", "opened_today": opened,
                              "tier": candidates[0].label if candidates else None},
    )
    session = open_session(
        window_end, name=name, candidates=candidates, network_volume_id=network_volume_id,
        opens=opens,
    )
    _log.info(
        "pod opened", extra={"pod_id": session.pod_id, "tier": session.tier_label,
                             "usd_per_hr": session.cost_per_hr, "window_end": session.window_end,
                             "datacenter": session.datacenter},
    )
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
    "ready" -- the understanding and chat models load lazily per
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


def close_session(*, name: str = "ai-studio-window", reason: str = "manual") -> list[str]:
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

    if session is not None and session.provisioned:
        # The pod's own logs die with it. Pull them first, bounded, and never
        # let a failure here delay the delete below -- money first.
        try:
            pulled = pull_pod_logs(session, PODS_LOG_DIR / session.pod_id)
            _log.info("pod logs pulled", extra={"pod_id": session.pod_id, "messages": len(pulled)})
        except Exception as exc:  # any ssh/runpodctl trouble; the pod still gets deleted
            _log.warning("pod log pull failed: %s", exc, extra={"pod_id": session.pod_id})

    for pod_id in sorted(targets):
        try:
            _runpodctl("pod", "delete", pod_id)
            terminated.append(pod_id)
            _log.info("pod closed", extra={"pod_id": pod_id, "reason": reason})
        except PodError as exc:
            # Keep going: one stuck pod must not stop us terminating the rest.
            terminated.append(f"{pod_id} (FAILED: {exc})")
            _log.warning("pod delete FAILED: %s", exc, extra={"pod_id": pod_id, "reason": reason})

    if session is not None:
        cost = session.spent_usd()
        minutes = session.elapsed_hours() * 60
        SpendLedger().record_session(cost, tier_label=session.tier_label, minutes=minutes)
        _archive_session(session, reason=reason, terminated=terminated, cost_usd=cost, minutes=minutes)
        _log.info(
            "session closed", extra={"pod_id": session.pod_id, "reason": reason, "minutes": round(minutes, 1),
                                     "cost_usd": round(cost, 4), "tier": session.tier_label},
        )

    clear_state()
    return terminated


def _archive_session(
    session: Session, *, reason: str, terminated: list[str], cost_usd: float, minutes: float
) -> Path | None:
    """Write the session's full state to `SESSIONS_LOG_DIR` before it is
    unlinked. Best-effort: a failure is logged, never raised into the close."""
    try:
        SESSIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = session.opened_at.replace(":", "").replace("-", "")[:15]
        dest = SESSIONS_LOG_DIR / f"{session.pod_id}-{stamp}.json"
        payload = _read_state_raw() | {
            "closed_at": utc_now_iso(), "reason": reason, "terminated": terminated,
            "minutes": round(minutes, 2), "cost_usd": round(cost_usd, 4),
        }
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(dest)
        return dest
    except Exception as exc:
        _log.warning("session record not written: %s", exc, extra={"pod_id": session.pod_id})
        return None


POD_LOG_FILES = ("setup", "inference", "comfy")


def pull_pod_logs(session: Session, dest: Path, *, timeout_s: float = POD_LOG_PULL_TIMEOUT_S) -> list[Path]:
    """Copy the tail of the pod's logs into `dest` over one ssh call.

    One command, one round trip, bounded by `timeout_s` and `POD_LOG_TAIL_BYTES`
    per file: this runs on the close path, where every second is billed and
    nothing may block the delete. Returns the files written."""
    info = _runpodctl("ssh", "info", session.pod_id)
    host, port = str(info.get("ip") or ""), str(info.get("port") or "")
    if not host or not port:
        raise PodError(f"{session.pod_id}: no ssh endpoint in {info}")
    script = "; ".join(
        [f'echo "== {name}"; tail -c {POD_LOG_TAIL_BYTES} /workspace/{name}.log 2>/dev/null' for name in POD_LOG_FILES]
        + ['echo "== dl-logs"; tail -n 200 /workspace/dl-logs/* 2>/dev/null']
    )
    argv = [
        "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-p", port, f"root@{host}", script,
    ]
    proc = _ssh(argv, stdin="", timeout_s=timeout_s)
    if proc.returncode != 0 and not proc.stdout:
        raise PodError(f"pod log pull exited {proc.returncode}: {(proc.stderr or '')[-200:]}")
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    current: str | None = None
    chunks: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines(keepends=True):
        if line.startswith("== "):
            current = line[3:].strip()
            chunks.setdefault(current, [])
            continue
        if current is not None:
            chunks[current].append(line)
    for name, lines in chunks.items():
        if not lines:
            continue
        path = dest / f"{name}.log"
        path.write_text("".join(lines), encoding="utf-8")
        written.append(path)
    (dest / "pulled_at.txt").write_text(utc_now_iso() + "\n", encoding="utf-8")
    return written


DEFAULT_GRACE_MINUTES = 10.0
"""How long a quiet pod is kept after its last activity when the caller
recorded no grace of its own (`touch_activity(grace_minutes=...)`). Ten
minutes covers H3's text-encoder reload (📏 60-90 s) many times over; the
side that knows what it just rendered, and what the next request is likely
to be, passes a better number."""


class ReapDecision(str):
    """What the reaper decided, as the same string it always returned (tests
    and the CLI compare with `in`), plus fields for the JSONL line and for
    the CLI to log only on a *transition* -- the per-minute `held:` text
    was ~65 % of ai-studio's journald volume (📏 2026-08-28)."""

    action: str
    idle_min: float
    grace: float
    spent_usd: float
    pod_id: str | None

    def __new__(
        cls, text: str, *, action: str, idle_min: float = 0.0, grace: float = 0.0,
        spent_usd: float = 0.0, pod_id: str | None = None,
    ) -> ReapDecision:
        obj = super().__new__(cls, text)
        obj.action, obj.idle_min, obj.grace, obj.spent_usd, obj.pod_id = action, idle_min, grace, spent_usd, pod_id
        return obj


def close_if_idle(
    *,
    default_grace_minutes: float = DEFAULT_GRACE_MINUTES,
    hold: bool = False,
    name: str = "ai-studio-window",
) -> ReapDecision:
    """Close the pod when it has gone quiet. The grace is whatever the last
    `touch_activity` recorded, else `default_grace_minutes`. `hold=True`
    (the caller has work waiting) never closes: a pod with a job about to
    land on it is not idle, whatever the clock says -- closing it there is
    the one move that costs a cold open *and* the wait.
    """
    session = load_state()
    if session is None:
        return ReapDecision("no session", action="none")

    if session.past_window():
        return ReapDecision(
            f"window over; {close_session(name=name, reason='window over')}",
            action="closed", pod_id=session.pod_id, spent_usd=session.spent_usd(),
        )

    state = _read_state_raw()
    last = state.get("last_activity_at") or session.opened_at
    idle = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
    recorded = state.get("grace_minutes")
    grace = float(recorded) if isinstance(recorded, int | float) and recorded > 0 else default_grace_minutes
    if hold:
        return ReapDecision(
            f"held: work pending ({idle:.0f}min idle, spent ${session.spent_usd():.2f})",
            action="held", idle_min=idle, grace=grace, spent_usd=session.spent_usd(), pod_id=session.pod_id,
        )
    if idle >= grace:
        return ReapDecision(
            f"idle {idle:.0f}min >= {grace:g}; {close_session(name=name, reason='idle')}",
            action="closed", idle_min=idle, grace=grace, spent_usd=session.spent_usd(), pod_id=session.pod_id,
        )
    return ReapDecision(
        f"active ({idle:.0f}min idle of {grace}, spent ${session.spent_usd():.2f})",
        action="active", idle_min=idle, grace=grace, spent_usd=session.spent_usd(), pod_id=session.pod_id,
    )


# ------------------------------------------------------------------ provision

SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
SETUP_SCRIPT = paths.deploy_script("pod_setup.sh")
INFERENCE_SERVER_SCRIPT = paths.deploy_script("inference_server.py")
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
        f"mkdir -p {remote_path.rsplit('/', 1)[0]} && cat > {remote_path}",
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
    extras: Sequence[Path] = (),
) -> None:
    """Start `deploy/pod_setup.sh` on the pod, detached, over SSH.

    `extras` are the caller's own `*.sh` files, deposited to
    `/workspace/pod_setup.d/` first and run by the setup script's last step,
    best effort -- what a caller wants on the pod that this package has no
    business knowing about.

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
    for extra in extras:
        if extra.suffix != ".sh" or "/" in extra.name or not extra.is_file():
            raise PodError(f"pod_setup.d extension must be an existing *.sh file, got {extra}")
        _ssh_deposit(
            host, port, extra.read_text(encoding="utf-8"),
            remote_path=f"/workspace/pod_setup.d/{extra.name}",
        )

    # The HF token rides as the first stdin line, ahead of the script body,
    # and is read into the environment `nohup` inherits: it never lands on
    # the pod's disk (the script file is the *rest* of stdin) nor in argv
    # (visible in `ps` on both ends). Empty line when unset -- pod_setup.sh
    # itself decides whether that is fatal (it was, while Tarsier2 was gated
    # and not yet cached).
    token = get_settings().hf_token
    body = (token.get_secret_value() if token else "") + "\n" + script.read_text(encoding="utf-8")
    argv = [
        "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-p", port, f"root@{host}",
        # `;` not `&&`: `cat > f && nohup ... &` backgrounds the cat too, which
        # then races the closing ssh stdin and writes an empty file (observed
        # live, 2026-08-27). Separated, cat drains stdin in the foreground and
        # only the setup run is backgrounded.
        "IFS= read -r HF_TOKEN; export HF_TOKEN; "
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
    """The running pod called `name`, if any -- what stops two windows opening."""
    return next((p for p in list_pods() if p.get("name") == name), None)


# ----------------------------------------------------------------------- state


def save_state(session: Session) -> None:
    """Write `runs/.session.json`: the one record of which pod is ours."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = session.__dict__ | {"last_activity_at": session.opened_at}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state() -> Session | None:
    """The open session from `runs/.session.json`, or None. Unknown keys in the
    file (written by an older version) are ignored, not an error.
    """
    raw = _read_state_raw()
    if not raw:
        return None
    known = {f for f in Session.__dataclass_fields__}
    return Session(**{k: v for k, v in raw.items() if k in known})


def touch_activity(label: str | None = None, *, grace_minutes: float | None = None) -> None:
    """Record that work just happened, so the idle timer restarts; optionally
    what it was (for the log) and how long the pod is worth keeping after it
    (the reaper's grace, until the next touch)."""
    raw = _read_state_raw()
    if raw:
        raw["last_activity_at"] = datetime.now(timezone.utc).isoformat()
        if label:
            raw["last_activity_label"] = label
        if grace_minutes is not None:
            raw["grace_minutes"] = float(grace_minutes)
        STATE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def mark_provisioned() -> None:
    """Record that `pod_setup.sh` has been started on this pod, so a restarted
    worker waits for it instead of starting it twice.
    """
    raw = _read_state_raw()
    if raw:
        raw["provisioned"] = True
        STATE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def clear_state() -> None:
    """Forget the session file. Called after the pod is confirmed gone."""
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


REAP_LAST = Path("runs/.reap_last.json")

def log_reap(decision: Any) -> None:
    """DEBUG every minute (JSONL only), INFO only when the action changes.

    The reaper fires every minute and used to be ~65 % of the host's journald
    volume saying `held: work pending` (📏 2026-08-28); the transitions are
    the information, the repeats are not."""
    log = logging.getLogger("ai_studio.reap")
    fields = {
        "action": getattr(decision, "action", None), "idle_min": round(getattr(decision, "idle_min", 0.0), 1),
        "grace": getattr(decision, "grace", None), "spent": round(getattr(decision, "spent_usd", 0.0), 2),
        "pod_id": getattr(decision, "pod_id", None),
    }
    previous = None
    try:
        previous = json.loads(REAP_LAST.read_text(encoding="utf-8")).get("action")
    except Exception:
        previous = None
    if fields["action"] != previous:
        log.info("reap: %s", decision, extra=fields)
        try:
            REAP_LAST.parent.mkdir(parents=True, exist_ok=True)
            REAP_LAST.write_text(json.dumps({"action": fields["action"]}), encoding="utf-8")
        except OSError:
            pass
    else:
        log.debug("reap: %s", decision, extra=fields)
