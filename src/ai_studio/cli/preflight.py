"""The pre-launch checklist: everything provable without spending a GPU-second.

There is no stub in this project, so there is no "prove the chain works locally
first" step. The first real run is simultaneously the implementation's
acceptance, the measurement, and the only affordable attempt — $1.556 of
approved budget against an L40S at $1.004/hr. Forty minutes.

This module is what replaces the stub. It is not "verify less"; it is **prove
everything that can be proved for free, so those forty minutes are spent only
on the part that genuinely needs a GPU**. Nine checks, from PLAN.md Phase 4,
runnable as one command on the VM instead of nine things to remember.

Two rules it follows, both of which are the whole point:

**A check that cannot run is SKIP, never PASS.** "We could not verify this"
must never render as "this is verified" — that is the silent-degradation
failure this codebase is built to refuse, applied to the checklist itself. A
missing credential produces a skip with the reason attached, and the run is not
green.

**Green means green.** `preflight` exits 0 only when all nine PASS. Offline,
several legitimately skip and the exit code says so, because Phase 4 is not
complete offline and pretending otherwise is how the expensive run gets
started too early.

Nothing here creates a pod, and only check 5 sends anything to a real person —
which is why it is opt-in behind a flag rather than merely credential-gated.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ai_studio.config.settings import Settings, get_settings


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    number: int
    name: str
    status: Status
    detail: str

    @property
    def ok(self) -> bool:
        return self.status is Status.PASS


def _skip(number: int, name: str, why: str) -> CheckResult:
    return CheckResult(number, name, Status.SKIP, why)


def _pass(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.PASS, detail)


def _fail(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.FAIL, detail)


SYNTHETIC_SECRET = "preflight-synthetic-channel-secret"
"""Used only where the subject of the check is not the credential itself.

Check 2 exists to prove the *deployed* secret works, so it skips without one.
Check 4 is about the clock, so a synthetic secret there proves what it claims
to prove.
"""


# ------------------------------------------------------------------ helpers


def _test_client(handler: Any = None, files_dir: Path | None = None) -> Any:
    """An in-process app. Raises `ImportError` if the web extra is absent.

    In-process on purpose: these checks must work before the service is up, and
    a check that needs the thing it is checking already running is not a
    pre-launch check.
    """
    from fastapi.testclient import TestClient

    from ai_studio.api.main import create_app
    from ai_studio.pipeline.queue import JobQueue

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    queue = JobQueue(root / "queue.sqlite3")
    app = create_app(
        queue=queue, handler=handler, files_dir=files_dir or (root / "files")
    )
    return TestClient(app), queue, root


def _handler(queue: Any, *, secret: str, clock: Any = None) -> Any:
    from ai_studio.bots.line.reply import NullReplyClient
    from ai_studio.bots.line.webhook import WebhookHandler

    settings = get_settings()
    return WebhookHandler(
        queue,
        NullReplyClient(),
        channel_secret=secret,
        allowed_group_id=settings.line_allowed_group_id or "Cpreflight",
        base_url=settings.public_base_url,
        clock=clock,
    )


def _event(text: str, *, event_id: str, group: str, user: str = "U" + "1" * 32) -> dict[str, Any]:
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": group, "userId": user},
        "message": {"type": "text", "id": "m1", "text": text},
    }


def _signed(events: list[dict[str, Any]], secret: str) -> tuple[bytes, str]:
    from ai_studio.bots.line.verify import sign

    body = json.dumps({"destination": "U" + "0" * 32, "events": events}).encode("utf-8")
    return body, sign(body, secret)


# ------------------------------------------------------------------- checks


def check_offline_suite(*, run: bool = True) -> CheckResult:
    """1. The whole offline toolchain: tests, ruff, import contracts, types."""
    name = "offline suite (pytest, ruff, lint-imports, mypy)"
    if not run:
        return _skip(1, name, "--skip-suite was passed")

    commands = [
        (["uv", "run", "pytest", "tests", "-q", "-m", "not runpod"], "pytest"),
        (["uv", "run", "ruff", "check", "--no-cache", "src", "tests", "examples"], "ruff"),
        (["uv", "run", "lint-imports"], "lint-imports"),
        (["uv", "run", "mypy"], "mypy"),
    ]
    for argv, label in commands:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=900.0, check=False,
            )
        except FileNotFoundError:
            return _skip(1, name, f"{argv[0]} not on PATH")
        except subprocess.TimeoutExpired:
            return _fail(1, name, f"{label} timed out after 900s")
        if proc.returncode != 0:
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
            return _fail(1, name, f"{label} exited {proc.returncode}: {' | '.join(tail)}")
    return _pass(1, name, "pytest, ruff, lint-imports and mypy all clean")


def check_signature_and_dedupe() -> CheckResult:
    """2. A signed event is accepted, a tampered one is not, a replay is not billed.

    Skips without a configured secret rather than falling back to a synthetic
    one: the thing under test here is the credential the service will actually
    run with. If this is broken, nothing a user types in Phase 7 reaches the
    queue at all.
    """
    name = "webhook signature and event dedupe"
    settings = get_settings()
    secret = settings.line_channel_secret
    if secret is None:
        return _skip(2, name, "LINE_CHANNEL_SECRET is not set")

    try:
        client, queue, _ = _test_client()
    except ImportError as exc:
        return _skip(2, name, f"the web extra is not installed: {exc}")

    try:
        value = secret.get_secret_value()
        group = settings.line_allowed_group_id or "Cpreflight"
        client.app.state.handler = _handler(queue, secret=value)

        body, signature = _signed([_event("/影片 preflight", event_id="pf-1", group=group)], value)
        good = client.post("/callback", content=body, headers={"x-line-signature": signature})
        if good.status_code != 200:
            return _fail(2, name, f"a correctly signed event got {good.status_code}")

        tampered = client.post(
            "/callback", content=body + b" ", headers={"x-line-signature": signature}
        )
        if tampered.status_code != 400:
            return _fail(2, name, f"a tampered body got {tampered.status_code}, expected 400")

        client.post("/callback", content=body, headers={"x-line-signature": signature})
        jobs = queue.counts()
        total = sum(jobs.values())
        if total != 1:
            return _fail(2, name, f"a redelivered event produced {total} jobs, expected 1")
        return _pass(2, name, "200 signed, 400 tampered, redelivery deduped")
    finally:
        queue.close()


def check_queue_and_conversion() -> CheckResult:
    """3. A real Chinese request becomes an English prompt.

    Needs the LLM endpoint. Without it the template fallback runs and the
    stored prompt is still Chinese — which is a working code path but not the
    thing this check is for, so it skips.
    """
    name = "queue and LLM conversion"
    settings = get_settings()
    if not settings.llm_endpoint_id:
        return _skip(3, name, "AI_STUDIO_LLM_ENDPOINT_ID is not set")
    if settings.runpod_api_key is None:
        return _skip(3, name, "RUNPOD_API_KEY is not set (the LLM endpoint needs it too)")

    import asyncio

    from ai_studio.llm.endpoint import RunpodLlmClient
    from ai_studio.pipeline.convert_worker import convert_job
    from ai_studio.pipeline.queue import JobQueue

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    queue = JobQueue(root / "queue.sqlite3")
    try:
        job, _ = queue.enqueue(
            "pf-convert", "Cpreflight", "一隻橘貓坐在窗邊看雨",
            user_id="Upreflight", media_kind=_image_kind(),
        )
        client = RunpodLlmClient(
            settings.llm_endpoint_id,
            settings.runpod_api_key.get_secret_value(),
            model=settings.llm_model,
        )
        how = asyncio.run(convert_job(queue, job.id, client))
        stored = queue.by_id(job.id)
        if stored is None or stored.prompt is None:
            return _fail(3, name, "the job was never parsed")
        rendered = str(stored.prompt.get("_rendered", ""))
        if not rendered.isascii():
            return _fail(3, name, f"the prompt is not English ({how}): {rendered[:60]}")
        return _pass(3, name, f"built_by={how}, rendered={rendered[:60]}")
    except Exception as exc:  # a cold endpoint, a bad id, no network
        return _fail(3, name, f"{type(exc).__name__}: {exc}")
    finally:
        queue.close()


def _image_kind() -> Any:
    from ai_studio.core.enums import MediaKind

    return MediaKind.IMAGE


def check_out_of_hours() -> CheckResult:
    """4. A request at any hour is accepted, answered, and opens nothing itself.

    There are no business hours any more; what this pins is that the *web*
    path never creates a pod. Only the worker does, on its own tick, behind
    the budget guard and the daily cap -- so a webhook that is up while the
    worker is down accepts and holds, and nothing bills.
    """
    name = "any hour: accepted, answered, nothing opened by the webhook"
    try:
        client, queue, _ = _test_client()
    except ImportError as exc:
        return _skip(4, name, f"the web extra is not installed: {exc}")

    from ai_studio.runtime import session as sess

    try:
        settings = get_settings()
        group = settings.line_allowed_group_id or "Cpreflight"
        night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        state_before = sess.load_state()

        handler = _handler(queue, secret=SYNTHETIC_SECRET, clock=lambda: night)
        client.app.state.handler = handler
        body, signature = _signed(
            [_event("/影片 preflight", event_id="pf-shut", group=group)], SYNTHETIC_SECRET
        )
        response = client.post("/callback", content=body, headers={"x-line-signature": signature})
        if response.status_code != 200:
            return _fail(4, name, f"the request got {response.status_code}")

        if sum(queue.counts().values()) != 1:
            return _fail(4, name, "the request was refused instead of held for the worker")

        sent = handler.replier.sent
        if not sent:
            return _fail(4, name, "the request was accepted with no reply at all")
        text = sent[-1][1][0]
        if "/q/" not in text:
            return _fail(4, name, f"the reply carries no status-page link: {text[:80]}")

        state_after = sess.load_state()
        if (state_after is not None) != (state_before is not None) or (
            state_after is not None and state_before is not None
            and state_after.pod_id != state_before.pod_id
        ):
            return _fail(4, name, "the webhook path changed the pod session state")
    except Exception as exc:
        return _fail(4, name, f"{type(exc).__name__}: {exc}")
    return _pass(4, name, "accepted and held (1 waiting), reply carries a status link, no pod created")


def check_push(*, send: bool = False) -> CheckResult:
    """5. A real push reaches the real group, and the mention resolves.

    Opt-in behind `--push`, not merely credential-gated: this is the one check
    that sends a message to actual people and spends actual quota. Note how
    many messages it consumes — that number is the open input in PLAN.md §4.
    """
    name = "push into the real group (with mention)"
    settings = get_settings()
    if not send:
        return _skip(5, name, "not attempted; pass --push to send a real message")
    token = settings.line_channel_access_token
    if token is None:
        return _skip(5, name, "LINE_CHANNEL_ACCESS_TOKEN is not set")
    if not settings.line_allowed_group_id:
        return _skip(5, name, "LINE_ALLOWED_GROUP_ID is not set")

    import asyncio

    from ai_studio.bots.line.push import LinePushClient, text_message

    client = LinePushClient(token.get_secret_value())
    # Plain text: real deliveries quote the request message, and preflight
    # has no request to quote.
    message = text_message("preflight check 5: if you can see this, push delivery works.")
    try:
        asyncio.run(client.push(settings.line_allowed_group_id, [message], retry_key="preflight"))
    except Exception as exc:
        return _fail(5, name, f"{type(exc).__name__}: {exc}")
    return _pass(
        5, name,
        "sent 1 text message"
        " -- now read the quota consumed off LINE Official Account Manager",
    )


def check_files_range() -> CheckResult:
    """6. `/files` answers a Range request with 206.

    LINE's video message object requires it, and nothing about the failure says
    so: the object is accepted and the video simply never plays.
    """
    name = "/files answers Range with 206"
    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / "preflight.mp4").write_bytes(bytes(range(256)) * 4)

    try:
        client, queue, _ = _test_client(files_dir=files)
    except ImportError as exc:
        return _skip(6, name, f"the web extra is not installed: {exc}")

    try:
        response = client.get("/files/preflight.mp4", headers={"Range": "bytes=0-99"})
        if response.status_code != 206:
            return _fail(6, name, f"got {response.status_code}, expected 206")
        content_range = response.headers.get("content-range", "")
        if content_range != "bytes 0-99/1024":
            return _fail(6, name, f"Content-Range was {content_range!r}")
        return _pass(6, name, f"206 with {content_range}")
    finally:
        queue.close()


def check_poster() -> CheckResult:
    """7. A poster comes out of ffmpeg under LINE's 1MB preview ceiling.

    Run for real on the host that will do it: the VPS is a 1 GB box, and the
    question is not whether the code is right but whether that machine can
    decode a frame.
    """
    name = "poster generation under the 1MB ceiling"
    from ai_studio import media

    settings = get_settings()
    if media.which(settings.ffmpeg_bin) is None:
        return _skip(7, name, f"{settings.ffmpeg_bin} is not on PATH")

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    made: list[str] = []
    try:
        for label, source, size in (
            ("clip", root / "clip.mp4", "864x480"),
            ("image", root / "flux.png", "1024x1024"),
        ):
            media.run(
                [
                    settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration=1",
                    "-frames:v", "1" if label == "image" else "24",
                    *(["-pix_fmt", "yuv420p"] if label == "clip" else []),
                    str(source),
                ],
                timeout_s=120.0,
            )
            out = media.poster(source, root / f"{label}_poster.jpg")
            made.append(f"{label}={out.stat().st_size // 1024}KB")
    except Exception as exc:
        return _fail(7, name, f"{type(exc).__name__}: {exc}")
    return _pass(7, name, ", ".join(made) + f" (ceiling {media.POSTER_MAX_BYTES // 1024}KB)")


def check_graphs() -> CheckResult:
    """8. All workflows load, bind and validate.

    A malformed graph is otherwise discovered at the moment of submission, on a
    pod that is already billing.
    """
    name = "all ComfyUI graphs load and validate"
    from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, REQUIRED_BINDINGS, Workflow

    targets = [
        (Path("workflows/h3_fl2va_turbo.json"), REQUIRED_BINDINGS),
        (Path("workflows/h3_fl2va_turbo_fp8.json"), REQUIRED_BINDINGS),
        (Path("workflows/h3_i2va_turbo.json"), REQUIRED_BINDINGS),
        (Path("workflows/h3_i2va_turbo_fp8.json"), REQUIRED_BINDINGS),
        (Path("workflows/flux_dev.json"), IMAGE_REQUIRED_BINDINGS),
        (Path("workflows/flux_dev_i2i.json"), IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"}),
    ]
    loaded: list[str] = []
    for path, required in targets:
        if not path.is_file():
            return _skip(8, name, f"{path} not found (run from the repo root)")
        try:
            workflow = Workflow.load(path, required_bindings=required)
        except Exception as exc:
            return _fail(8, name, f"{path.name}: {type(exc).__name__}: {exc}")
        loaded.append(f"{path.name}({len(workflow.bindings)} bindings)")
    return _pass(8, name, ", ".join(loaded))


def check_placement() -> CheckResult:
    """9. Every ladder rung still corresponds to something the catalog offers.

    Read-only; it creates nothing. A dead rung must be found now, not at 11:00
    when all four are refused and the window is lost.
    """
    name = "placement ladder matches the live catalog"
    settings = get_settings()
    if settings.runpod_api_key is None:
        return _skip(9, name, "RUNPOD_API_KEY is not set")

    from ai_studio.runtime import session as sess
    from ai_studio.runtime.pod import LICENCE_SAFE_DATACENTERS, PodManager

    # Same verdicts as `ai-studio pod placement`, deliberately: a rung the
    # catalog does not offer is refused on deploy exactly like one that is
    # merely out of stock, so without this the ladder falls through to a softer
    # GPU for months and it reads as bad luck.
    dead: list[str] = []
    try:
        with PodManager() as manager:
            for tier in sess.CANDIDATES:
                if tier.datacenter not in LICENCE_SAFE_DATACENTERS:
                    dead.append(f"{tier.label}: outside H3's licence")
                    continue
                if manager.verify_placement(tier.gpu, tier.datacenter, cloud=tier.cloud) == (
                    "not-offered"
                ):
                    dead.append(f"{tier.label} @ {tier.datacenter}: never offered here")
    except Exception as exc:
        return _fail(9, name, f"{type(exc).__name__}: {exc}")

    if dead:
        return _fail(9, name, "; ".join(dead))
    return _pass(9, name, f"all {len(sess.CANDIDATES)} rungs licence-safe and offered")


# ------------------------------------------------------------------- runner


def run_all(
    *,
    run_suite: bool = True,
    send_push: bool = False,
    settings: Settings | None = None,
) -> list[CheckResult]:
    """Every check, in PLAN.md's order. Never raises: a crashing check is a FAIL.

    Ordering is PLAN.md's, not cost's, because the list is read as a checklist
    by a person — and a person reading row 5 wants it to be row 5.
    """
    _ = settings or get_settings()
    checks: list[Callable[[], CheckResult]] = [
        lambda: check_offline_suite(run=run_suite),
        check_signature_and_dedupe,
        check_queue_and_conversion,
        check_out_of_hours,
        lambda: check_push(send=send_push),
        check_files_range,
        check_poster,
        check_graphs,
        check_placement,
    ]

    results: list[CheckResult] = []
    for number, check in enumerate(checks, start=1):
        try:
            results.append(check())
        except Exception as exc:  # a check that dies is a check that failed
            results.append(
                _fail(number, f"check {number}", f"raised {type(exc).__name__}: {exc}")
            )
    return results


def summarise(results: list[CheckResult]) -> tuple[bool, str]:
    """`(all nine green, one-line summary)`.

    Green means green: skips do not count. Phase 4's own definition of done is
    nine passes, and a checklist that reports "fine" with three unknowns on it
    is worse than no checklist, because it gets believed.
    """
    tally = {status: sum(1 for r in results if r.status is status) for status in Status}
    green = tally[Status.PASS] == len(results) and bool(results)
    return green, (
        f"{tally[Status.PASS]} passed, {tally[Status.FAIL]} failed, "
        f"{tally[Status.SKIP]} skipped, of {len(results)}"
    )


def stamp(results: list[CheckResult], *, when: datetime) -> str:
    """A record to paste into the run log. Plain ASCII for the Windows console."""
    lines = [f"preflight {when.isoformat(timespec='seconds')}"]
    lines += [f"  {r.number}. [{r.status.value}] {r.name} -- {r.detail}" for r in results]
    green, summary = summarise(results)
    lines.append(f"  => {summary}{'' if green else '  (NOT green)'}")
    return "\n".join(lines)
