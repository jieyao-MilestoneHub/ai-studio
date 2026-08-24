"""Service-window lifecycle.

The properties worth testing are all about not losing money: never leave a pod
running, never queue for capacity, and always terminate rather than stop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from videogen.core.errors import PodError
from videogen.runtime import session as sess


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sess, "STATE_FILE", tmp_path / "session.json")


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
    """4090 stock is thin; one refusal must not end the window."""
    seen, fake = _calls_recorder(
        [{"items": []}, PodError("no instances"), {"id": "pod2", "costPerHr": 0.74}]
    )
    monkeypatch.setattr(sess, "_runpodctl", fake)

    s = sess.open_session(_end(), name="w")
    assert s.pod_id == "pod2"
    assert len([c for c in seen if c[:2] == ["pod", "create"]]) == 2


def test_open_raises_rather_than_queueing_when_nothing_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for absent capacity can leave a billing reservation. Refuse instead."""
    responses: list[Any] = [{"items": []}] + [PodError("no instances")] * len(sess.CANDIDATES)
    _, fake = _calls_recorder(responses)
    monkeypatch.setattr(sess, "_runpodctl", fake)

    with pytest.raises(PodError, match="nothing is billing"):
        sess.open_session(_end(), name="w")


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
    assert "idle 45min" in sess.close_if_idle(20, name="w")


def test_reap_leaves_a_busy_window_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.34}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    sess.open_session(_end(120), name="w")
    sess.touch_activity()

    assert "active" in sess.close_if_idle(20, name="w")


def test_spend_tracking_uses_the_real_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _calls_recorder([{"items": []}, {"id": "p", "costPerHr": 0.354}])
    monkeypatch.setattr(sess, "_runpodctl", fake)
    s = sess.open_session(_end(), name="w")

    later = datetime.fromisoformat(s.opened_at) + timedelta(hours=3.8)
    assert s.spent_usd(later) == pytest.approx(1.345, abs=0.01)
