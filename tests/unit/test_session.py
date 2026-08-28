"""Service-window lifecycle.

The properties worth testing are all about not losing money: never leave a pod
running, never queue for capacity, and always terminate rather than stop.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ai_studio.config.settings import get_settings
from ai_studio.core.errors import PodError
from ai_studio.runtime import session as sess


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sess, "STATE_FILE", tmp_path / "session.json")
    # close_session() writes logs/sessions/<pod>-<ts>.json and pulls pod logs
    # into logs/pods/ (2026-08-28); un-isolated, the suite left fourteen
    # `p-*.json` records in the operator's real logs/ -- 📏 found by the first
    # `ai-studio archive --dry-run` listing them as members.
    monkeypatch.setattr(sess, "SESSIONS_LOG_DIR", tmp_path / "logs" / "sessions")
    monkeypatch.setattr(sess, "PODS_LOG_DIR", tmp_path / "logs" / "pods")
    # close_session() now also writes a SpendLedger entry on every close; without
    # isolating it too, every test that closes a session writes into the real
    # repo's runs/.spend_ledger.json instead of a throwaway one.
    from ai_studio.runtime.budget import SpendLedger

    monkeypatch.setattr(sess, "SpendLedger", lambda: SpendLedger(tmp_path / "ledger.json"))


def _calls_recorder(responses: list[Any]) -> tuple[list[list[str]], Any]:
    """Fake `_runpodctl` that replays `responses` and records the argv it saw."""
    seen: list[list[str]] = []

    def fake(*args: str, timeout_s: float = 0.0) -> dict[str, Any]:
        seen.append(list(args))
        item = responses.pop(0) if responses else {}
        if isinstance(item, Exception):
            raise item
        return item

    return seen, fake


def _end(minutes: int = 60) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


# ------------------------------------------------------------------------ open


def test_open_always_passes_terminate_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backstop: if this process dies, the pod must still self-terminate."""
    seen, fake = _calls_recorder([{"items": []}, {"id": "pod1", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    sess.open_session(_end(), name="w")

    create = next(c for c in seen if c[:2] == ["pod", "create"])
    assert "--terminate-after" in create
    stamp = create[create.index("--terminate-after") + 1]
    assert stamp.endswith("Z")


def test_terminate_after_is_past_the_window_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip mid-render at the bell must not be killed by the backstop."""
    seen, fake = _calls_recorder([{"items": []}, {"id": "pod1", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    end = _end(60)
    sess.open_session(end, name="w")

    create = next(c for c in seen if c[:2] == ["pod", "create"])
    stamp = create[create.index("--terminate-after") + 1]
    backstop = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert backstop > end
    assert (backstop - end) <= timedelta(minutes=sess.TERMINATE_BUFFER_MIN + 1)


def test_open_walks_the_candidate_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """L40S stock is thin; one refusal must not end the window."""
    seen, fake = _calls_recorder(
        [{"items": []}, PodError("no instances"), {"id": "pod2", "costPerHr": 0.804}]
    )
    monkeypatch.setattr(sess, "_runpodctl", fake)

    s = sess.open_session(_end(), name="w")
    assert s.pod_id == "pod2"
    assert len([c for c in seen if c[:2] == ["pod", "create"]]) == 2


def test_the_ladder_is_strictly_price_descending() -> None:
    """Grab the best card available now; only the cheapest is worth waiting for."""
    rates = [t.usd_per_hr for t in sess.CANDIDATES]
    assert rates == sorted(rates, reverse=True)
    assert len(sess.CANDIDATES) == 4


def test_only_the_cheapest_rung_waits() -> None:
    assert [t.wait for t in sess.CANDIDATES] == [False, False, False, True]


def test_the_ladder_only_uses_licence_permitted_datacenters() -> None:
    """H3's licence excludes the US, EU, UK and South Korea."""
    from ai_studio.runtime.pod import LICENCE_SAFE_DATACENTERS

    assert all(t.datacenter in LICENCE_SAFE_DATACENTERS for t in sess.CANDIDATES)


def test_quality_mode_follows_vram_not_preference() -> None:
    """A measured run peaked at 43.3GB, so under 48GB must go soft."""
    for tier in sess.CANDIDATES:
        assert tier.low_vram == (tier.vram_gb < 48)
        assert tier.quantisation == ("int8" if tier.low_vram else "fp8")


def test_the_serving_tier_is_recorded_on_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 1.004}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    s = sess.open_session(_end(), name="w")
    assert s.tier_label == "L40S/SECURE"
    assert s.vram_gb == 48 and s.low_vram is False and s.quantisation == "fp8"


def test_open_raises_rather_than_queueing_when_nothing_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for absent capacity can leave a billing reservation. Refuse instead."""
    responses: list[Any] = [{"items": []}] + [PodError("no instances")] * 40
    _, fake = _calls_recorder(responses)
    monkeypatch.setattr(sess, "_runpodctl", fake)
    monkeypatch.setattr(sess.time, "sleep", lambda _s: None)
    # A window ending now means the waiting rung gets no wait budget.
    monkeypatch.setattr(sess, "WAIT_MAX_S", 0)

    with pytest.raises(PodError, match="nothing is billing"):
        sess.open_session(_end(0), name="w")


def test_open_refuses_to_double_book(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": [{"id": "old", "name": "w", "costPerHr": 0.34}]}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    with pytest.raises(PodError, match="already running"):
        sess.open_session(_end(), name="w")


# ----------------------------------------------------------------------- close


def test_close_terminates_and_never_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping a pod keeps its disk and keeps charging. Only terminate."""
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(), name="w")

    seen2, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    assert sess.close_session(name="w") == ["p"]

    assert ["pod", "delete", "p"] in seen2
    assert not any("stop" in c for c in seen2)


def test_close_finds_a_pod_by_name_when_state_is_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lost state file must not strand a billing machine."""
    seen, fake = _calls_recorder([{"items": [{"id": "orphan", "name": "w"}]}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    assert sess.close_session(name="w") == ["orphan"]
    assert ["pod", "delete", "orphan"] in seen


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    assert sess.close_session(name="w") == []


def test_close_records_the_session_cost_into_the_monthly_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ledger has to be fed by `close_session()` itself, not by whichever
    CLI command happened to call it -- `close_if_idle` is the path that
    actually ends the window on almost every real day (see the next test).

    `SpendLedger` is already isolated by the autouse `_isolate_state` fixture,
    same as `STATE_FILE` -- read it back through `sess.SpendLedger()` (the
    patched factory) rather than constructing a second, unrelated instance."""
    import json

    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.5}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    raw = sess._read_state_raw()
    raw["opened_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    sess.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")

    _, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    sess.close_session(name="w")

    assert sess.SpendLedger().spent_this_month_usd() == pytest.approx(0.5, abs=0.05)


def test_reap_closing_a_quiet_window_also_records_to_the_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A window sized for peak demand is quiet most days, so `close_if_idle`
    -- not the scheduled `session close` -- is what actually ends most
    windows. If only the explicit close recorded spend, the monthly budget
    guard would see close to $0 spent nearly every month regardless of the
    real bill."""
    import json

    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.354}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    raw = sess._read_state_raw()
    raw["opened_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    raw["last_activity_at"] = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
    sess.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")

    _, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    assert "idle 45min" in sess.close_if_idle(default_grace_minutes=20, name="w")

    assert sess.SpendLedger().spent_this_month_usd() == pytest.approx(0.354, abs=0.05)


def test_close_keeps_going_when_one_delete_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder(
        [{"items": [{"id": "a", "name": "w"}, {"id": "b", "name": "w"}]},
         PodError("boom"), {"deleted": True}]
    )
    monkeypatch.setattr(sess, "_runpodctl", fake)
    result = sess.close_session(name="w")
    assert len(result) == 2
    assert any("FAILED" in r for r in result)


# ------------------------------------------------------------------------ idle


def test_reap_closes_a_quiet_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    # backdate the activity marker
    raw = sess._read_state_raw()
    raw["last_activity_at"] = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
    sess.STATE_FILE.write_text(__import__("json").dumps(raw), encoding="utf-8")

    _, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    assert "idle 45min" in sess.close_if_idle(default_grace_minutes=20, name="w")


def test_reap_leaves_a_busy_window_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")
    sess.touch_activity("video", grace_minutes=20)

    assert "active" in sess.close_if_idle(name="w")


def _backdate_activity(minutes: int, grace: float | None) -> None:
    import json

    raw = sess._read_state_raw()
    raw["last_activity_at"] = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    if grace is not None:
        raw["grace_minutes"] = grace
    sess.STATE_FILE.write_text(json.dumps(raw), encoding="utf-8")


def test_the_grace_the_last_touch_recorded_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller knows what it rendered (Flux reloads in ~15 s, H3's text
    encoder in ~90 s); this side only keeps the clock. Seven idle minutes is
    over a recorded grace of 5 and under one of 10."""
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    _backdate_activity(7, 5)
    _, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    assert "idle 7min >= 5" in sess.close_if_idle(default_grace_minutes=60, name="w")


def test_no_recorded_grace_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    _backdate_activity(7, None)
    assert "active" in sess.close_if_idle(default_grace_minutes=10, name="w")
    _backdate_activity(7, 10)
    assert "active" in sess.close_if_idle(default_grace_minutes=1, name="w")


def test_a_pod_with_work_pending_is_never_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing a pod a job is about to land on costs a cold open *and* the
    wait -- the one move with no upside."""
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")

    _backdate_activity(45, "image")
    seen, fake2 = _calls_recorder([{"items": []}, {"deleted": True}])
    monkeypatch.setattr(sess, "_runpodctl", fake2)
    assert "held" in sess.close_if_idle(hold=True, name="w")
    assert not any(c[:2] == ["pod", "delete"] for c in seen)


def test_spend_tracking_uses_the_real_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.354}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    s = sess.open_session(_end(), name="w")

    later = datetime.fromisoformat(s.opened_at) + timedelta(hours=3.8)
    assert s.spent_usd(later) == pytest.approx(1.345, abs=0.01)


# -------------------------------------------------------------- template id


def test_the_template_id_is_the_official_standard_gpu_one() -> None:
    """This id has already been wrong once. `2lv7ev3wfp` was hardcoded here
    until it was renamed to "ComfyUI Blackwell Edition" and rescoped to RTX
    5090/B200 — the wrong card for every rung on this ladder."""
    assert sess.TEMPLATE_COMFYUI_STANDARD == "cw3nka7d08"


def test_pod_create_actually_sends_that_template_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinning the constant is not enough: a correct constant passed to the
    wrong flag, or not passed at all, fails in exactly the same way and costs
    exactly as much."""
    seen, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    sess.open_session(_end(), name="w")

    create = next(c for c in seen if c[:2] == ["pod", "create"])
    assert "--template-id" in create
    assert create[create.index("--template-id") + 1] == "cw3nka7d08"


def test_every_rung_is_deployed_from_the_same_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ladder falls through on refusal. A rung that reached for a different
    template would be a different machine image, silently."""
    seen, fake = _calls_recorder(
        [{"items": []}, PodError("no instances"), {"id": "p2", "costPerHr": 0.804}]
    )
    monkeypatch.setattr(sess, "_runpodctl", fake)

    sess.open_session(_end(), name="w")

    creates = [c for c in seen if c[:2] == ["pod", "create"]]
    assert len(creates) == 2
    for create in creates:
        assert create[create.index("--template-id") + 1] == sess.TEMPLATE_COMFYUI_STANDARD


def test_the_docs_and_the_skill_name_the_same_template_id() -> None:
    """Three places carry this string. Two of them are prose, which no linter
    reads, and prose is where the stale id survived last time."""
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    for doc in (
        repo / "docs" / "model-h3.md",
        repo / ".claude" / "skills" / "runpod-session" / "SKILL.md",
    ):
        body = doc.read_text(encoding="utf-8")
        assert sess.TEMPLATE_COMFYUI_STANDARD in body, f"{doc.name} does not name it"
        assert "2lv7ev3wfp" not in body.replace("(`2lv7ev3wfp`)", ""), (
            f"{doc.name} still points at the renamed Blackwell template"
        )


# ------------------------------------------------------------ network volume


def test_a_volume_puts_the_ladder_in_its_datacenter_on_secure_cloud_and_waits() -> None:
    """Network volumes are secure-only and datacenter-bound: the L40S/OC-AU-1
    and community rungs cannot mount one, so the ladder becomes the 4090
    secure rung where the volume is -- worth waiting for, because waiting
    beats paying another 68GB download."""
    (tier,) = sess.candidates_for_volume("EUR-IS-1")
    assert tier.datacenter == "EUR-IS-1"
    assert tier.cloud == "SECURE"
    assert tier.wait is True


def test_a_volume_in_a_licence_unsafe_datacenter_is_refused() -> None:
    with pytest.raises(PodError, match="not licence-safe"):
        sess.candidates_for_volume("EU-RO-1")


def test_open_mounts_the_volume_instead_of_a_fresh_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, fake = _calls_recorder([{"items": []}, {"id": "pod9", "costPerHr": 0.754}])
    monkeypatch.setattr(sess, "_runpodctl", fake)

    sess.open_session(
        _end(), name="w", candidates=sess.candidates_for_volume("EUR-IS-1"),
        network_volume_id="vol123",
    )

    create = next(c for c in seen if c[:2] == ["pod", "create"])
    assert create[create.index("--network-volume-id") + 1] == "vol123"
    assert "--volume-in-gb" not in create
    assert create[create.index("--data-center-ids") + 1] == "EUR-IS-1"


def test_placement_reads_the_volume_datacenter_from_runpod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, fake = _calls_recorder([{"id": "vol123", "dataCenterId": "EUR-IS-1"}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    monkeypatch.setenv("AI_STUDIO_NETWORK_VOLUME_ID", "vol123")
    get_settings(refresh=True)
    try:
        candidates, volume_id = sess.placement()
    finally:
        monkeypatch.delenv("AI_STUDIO_NETWORK_VOLUME_ID")
        get_settings(refresh=True)

    assert volume_id == "vol123"
    assert [t.datacenter for t in candidates] == ["EUR-IS-1"]
    assert seen == [["network-volume", "get", "vol123"]]


def test_placement_without_a_volume_is_the_plain_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched at the settings accessor rather than the environment: the
    developer's own .env may carry a volume id, and this test is about the
    code path, not the machine it runs on."""
    from types import SimpleNamespace

    monkeypatch.setattr(sess, "get_settings", lambda: SimpleNamespace(network_volume_id=None))
    assert sess.placement() == (sess.CANDIDATES, None)


# ------------------------------------------------- what a close leaves behind


def _live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, provisioned: bool = True) -> None:
    monkeypatch.setattr(sess, "SESSIONS_LOG_DIR", tmp_path / "logs" / "sessions")
    monkeypatch.setattr(sess, "PODS_LOG_DIR", tmp_path / "logs" / "pods")
    # Relative to now: a fixed window_end turned "held" into "window over"
    # the moment the wall clock passed it (📏 the first evening after it was
    # written). opened 30 min ago, lease 2 h out.
    now = datetime.now(timezone.utc)
    state = {
        "pod_id": "p1", "gpu": "NVIDIA GeForce RTX 4090", "datacenter": "EUR-IS-1", "cloud": "SECURE",
        "cost_per_hr": 0.74, "opened_at": (now - timedelta(minutes=30)).isoformat(),
        "window_end": (now + timedelta(hours=2)).isoformat(), "tier_label": "RTX 4090/SECURE",
        "vram_gb": 24, "low_vram": True, "quantisation": "int8", "ssh": {}, "provisioned": provisioned,
        "last_activity_label": "drama", "grace_minutes": 10.0,
        "last_activity_at": (now - timedelta(minutes=3)).isoformat(),
    }
    sess.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sess.STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def test_close_writes_the_session_record_before_unlinking_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`.session.json` is unlinked at close; until 2026-08-28 the tier,
    quantisation and why-it-closed of every session were lost with it."""
    _live(monkeypatch, tmp_path, provisioned=False)
    _, fake = _calls_recorder([{"id": "p1"}, {}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    monkeypatch.setattr(sess, "list_pods", lambda: [])

    assert sess.close_session(name="w", reason="idle") == ["p1"]

    (record,) = list((tmp_path / "logs" / "sessions").glob("p1-*.json"))
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["reason"] == "idle" and payload["terminated"] == ["p1"]
    assert payload["quantisation"] == "int8" and payload["tier_label"] == "RTX 4090/SECURE"
    assert payload["closed_at"].endswith("+00:00") and payload["minutes"] > 0
    assert not sess.STATE_FILE.exists()


def test_a_failed_pod_log_pull_never_blocks_the_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Money first: the pull is best-effort and runs before the delete, so it
    must not be able to stop it."""
    _live(monkeypatch, tmp_path, provisioned=True)
    calls: list[tuple[str, ...]] = []

    def runpodctl(*args: str, timeout_s: float = 180.0) -> dict:
        calls.append(args)
        if args[:2] == ("ssh", "info"):
            raise sess.PodError("ssh info unavailable")
        return {"id": "p1"}

    monkeypatch.setattr(sess, "_runpodctl", runpodctl)
    monkeypatch.setattr(sess, "list_pods", lambda: [])

    assert sess.close_session(name="w", reason="window over") == ["p1"]
    assert ("pod", "delete", "p1") in calls
    assert not (tmp_path / "logs" / "pods").exists()


def test_pull_pod_logs_splits_one_ssh_reply_into_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    monkeypatch.setattr(sess, "_runpodctl", lambda *a, **k: {"ip": "1.2.3.4", "port": "22"})
    reply = "== setup\n[setup 2026-08-28T00:00:00Z] hi\n== inference\nINFO x\n== comfy\n== dl-logs\nok\n"
    monkeypatch.setattr(
        sess, "_ssh",
        lambda argv, *, stdin, timeout_s: subprocess.CompletedProcess(argv, 0, stdout=reply, stderr=""),
    )
    live = sess.Session.__new__(sess.Session)
    live.__dict__.update(pod_id="p9", provisioned=True)
    written = sess.pull_pod_logs(live, tmp_path / "p9")
    names = sorted(p.name for p in written)
    assert names == ["dl-logs.log", "inference.log", "setup.log"], "an empty section writes no file"
    assert (tmp_path / "p9" / "setup.log").read_text(encoding="utf-8").startswith("[setup ")
    assert (tmp_path / "p9" / "pulled_at.txt").exists()


def test_the_reaper_decision_is_still_the_old_string_with_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _live(monkeypatch, tmp_path, provisioned=False)
    sess.touch_activity("drama", grace_minutes=10_000)
    decision = sess.close_if_idle(name="w", hold=True)
    assert "held: work pending" in decision  # the contract every caller relies on
    assert decision.action == "held" and decision.pod_id == "p1" and decision.grace == 10_000


def test_provision_ships_extras_into_pod_setup_d(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A caller's extension scripts go up before the setup script runs, to
    the directory its last step walks; anything that is not a plain *.sh is
    refused rather than quietly skipped."""
    from pydantic import SecretStr

    deposited: list[tuple[str, str]] = []
    monkeypatch.setattr(sess, "_runpodctl", lambda *a: {"ip": "1.2.3.4", "port": "22"})
    monkeypatch.setattr(sess, "_ssh_deposit", lambda h, p, body, *, remote_path: deposited.append((remote_path, body)))

    class _Proc:
        returncode, stdout, stderr = 0, "started", ""

    monkeypatch.setattr(sess, "_ssh", lambda argv, *, stdin, timeout_s: _Proc())
    monkeypatch.setattr(sess, "get_settings", lambda: type("S", (), {"hf_token": SecretStr("t")})())
    script, server, ext = tmp_path / "pod_setup.sh", tmp_path / "srv.py", tmp_path / "face_repair.sh"
    script.write_text("echo\n", encoding="utf-8")
    server.write_text("print()\n", encoding="utf-8")
    ext.write_text("exit 0\n", encoding="utf-8")
    live = sess.Session.__new__(sess.Session)
    live.__dict__.update(pod_id="p1", vram_gb=24)

    sess.provision(live, script=script, inference_script=server, extras=[ext])
    assert deposited == [("/workspace/inference_server.py", "print()\n"), ("/workspace/pod_setup.d/face_repair.sh", "exit 0\n")]

    with pytest.raises(PodError):
        sess.provision(live, script=script, inference_script=server, extras=[tmp_path / "nope.py"])
