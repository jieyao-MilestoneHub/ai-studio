"""`PodManager.list_pods` / `status`: parsing the real REST v2 shape.

Fixtures are `ListPodsResponse`'s own example from openapi.json (2026-08-29),
trimmed. 📏 the bug these catch: `list_pods` read key "items" (real key
"pods") and always returned [] -- `pod status` reported "no pods, nothing
is billing" while a pod was live and billing. `cost` is a bare number, not
nested under `.perHr`/`costPerHr` (that name belongs to a different tool,
the runpodctl CLI's own JSON -- see `runtime.session._runpodctl`).
"""

from __future__ import annotations

import json

import httpx
import pytest

from ai_studio.runtime.pod import PodManager

# ListPodsResponse's own example, trimmed to the fields _parse_status reads.
LIST_RESPONSE = {
    "pods": [
        {
            "id": "7h9k2m4n6p",
            "name": "pytorch-training",
            "status": "RUNNING",
            "cost": 0.44,
            "dataCenterId": "US-KS-2",
            "cudaVersion": "12.8",
            "gpu": {"count": 1, "id": "NVIDIA GeForce RTX 4090"},
            "createdAt": "2026-06-01T12:00:00Z",
            "startedAt": "2026-06-01T12:02:00Z",
            "runtime": {"uptime": 3600},
        }
    ]
}

SINGLE_RESPONSE = LIST_RESPONSE["pods"][0]  # GET /pods/{id} returns the Pod unwrapped


def _manager(handler) -> PodManager:
    return PodManager("test-key", transport=httpx.MockTransport(handler))


def test_list_pods_reads_the_real_pods_key() -> None:
    manager = _manager(lambda request: httpx.Response(200, content=json.dumps(LIST_RESPONSE)))
    pods = manager.list_pods()
    assert [p.id for p in pods] == ["7h9k2m4n6p"]
    assert pods[0].cost_per_hr == 0.44
    assert pods[0].is_running


def test_list_pods_on_an_empty_account_is_really_empty() -> None:
    manager = _manager(lambda request: httpx.Response(200, content=json.dumps({"pods": []})))
    assert manager.list_pods() == []


def test_single_status_reads_the_unwrapped_pod() -> None:
    manager = _manager(lambda request: httpx.Response(200, content=json.dumps(SINGLE_RESPONSE)))
    status = manager.status("7h9k2m4n6p")
    assert status.cost_per_hr == 0.44 and status.gpu_id == "NVIDIA GeForce RTX 4090"
    # uptime is wall-clock from startedAt, not the schema's own runtime.uptime
    # (that field is not read); cost_so_far scales whatever uptime comes back.
    assert status.cost_so_far_usd() == pytest.approx(status.cost_per_hr * status.uptime_s / 3600.0)


def test_a_pod_that_has_terminated_bills_zero() -> None:
    """The schema's own note: cost is 0.0 when EXITED or TERMINATED."""
    payload = {**SINGLE_RESPONSE, "status": "TERMINATED", "cost": 0.0}
    manager = _manager(lambda request: httpx.Response(200, content=json.dumps({"pods": [payload]})))
    (status,) = manager.list_pods()
    assert status.cost_per_hr == 0.0 and status.cost_so_far_usd() == 0.0
    assert not status.is_running
