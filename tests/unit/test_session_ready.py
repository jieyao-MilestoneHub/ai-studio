"""`wait_ready` must not say yes to a pod whose node pack is not installed.

The stock ComfyUI template answers `/system_stats` within seconds of boot;
`deploy/pod_setup.sh` takes ~15 minutes after that. A worker that submits in
between gets `missing_node_type` and spends a job's attempts on it.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ai_studio.runtime import session as sess


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _session() -> sess.Session:
    return sess.Session(
        pod_id="pod1", gpu="g", datacenter="d", cloud="c", cost_per_hr=0.7,
        opened_at="2026-08-26T14:00:00+00:00", window_end="2026-08-26T16:00:00+00:00",
    )


def test_ready_only_once_the_h3_node_pack_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter([
        httpx.ConnectError("booting"),
        _Response(502, None),
        _Response(200, {}),                       # ComfyUI up, node pack not yet
        _Response(200, {sess.READY_NODE: {}}),    # pod_setup.sh finished
    ])
    urls: list[str] = []

    def fake_get(url: str, timeout: float) -> Any:
        urls.append(url)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(sess.httpx, "get", fake_get)
    monkeypatch.setattr(sess.time, "sleep", lambda s: None)

    sess.wait_ready(_session(), timeout_s=60.0, interval_s=0.0)

    assert len(urls) == 4
    assert urls[0].endswith(f"/object_info/{sess.READY_NODE}")


def test_a_pod_that_never_installs_the_pack_is_reported_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sess.httpx, "get", lambda url, timeout: _Response(200, {}))
    clock = iter([0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(sess.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(sess.time, "sleep", lambda s: None)

    with pytest.raises(sess.PodError, match="not installed yet"):
        sess.wait_ready(_session(), timeout_s=50.0, interval_s=0.0)
