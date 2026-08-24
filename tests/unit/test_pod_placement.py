"""Placement checks against the GPU catalog.

These exist because of a bug they would have caught: every ladder rung pinned
EUR-IS-2, and L40S is not offered in Iceland at all. A deploy attempt reports
that as a refusal, indistinguishable from thin stock -- so two rungs looked
merely unlucky for as long as nobody asked the catalog.
"""

from __future__ import annotations

import json

import httpx
import pytest

from videogen.core.errors import PodError
from videogen.runtime.pod import LICENCE_SAFE_DATACENTERS, PodManager
from videogen.runtime.session import CANDIDATES

# What /catalog/gpus returned on 2026-08-25, trimmed to the fields we read.
CATALOG = {
    "NVIDIA L40S": {
        "SECURE": ["EU-NL-1", "OC-AU-1", "US-NC-1", "US-TX-3", "US-TX-4"],
        "COMMUNITY": None,  # the API reports no per-datacenter breakdown
    },
    "NVIDIA GeForce RTX 4090": {
        "SECURE": ["EU-CZ-1", "EU-RO-1", "EUR-IS-2"],
        "COMMUNITY": None,
    },
}


def _handler(request: httpx.Request) -> httpx.Response:
    from urllib.parse import unquote

    gpu = unquote(request.url.path.rsplit("/", 1)[-1])
    params = request.url.params
    body: dict = {"id": gpu, "memory": 48, "price": {"secure": 0.99}}
    # Availability is opt-in, exactly as the real API behaves: without
    # include=AVAILABILITY there is no stock in the response at all.
    if params.get("include") == "AVAILABILITY":
        assert params.get("product"), "the real API 400s on include without product"
        dcs = CATALOG[gpu][params.get("cloud", "SECURE")]
        body["availability"] = "LOW"
        if dcs is not None:
            body["dataCenters"] = [{"id": d, "availability": "LOW"} for d in dcs]
    return httpx.Response(200, content=json.dumps(body))


@pytest.fixture
def manager() -> PodManager:
    return PodManager("test-key", transport=httpx.MockTransport(_handler))


def test_l40s_is_not_offered_in_iceland(manager: PodManager) -> None:
    """The actual bug. This pairing is dead, not unlucky."""
    assert manager.verify_placement("NVIDIA L40S", "EUR-IS-2") == "not-offered"


def test_l40s_has_stock_in_australia(manager: PodManager) -> None:
    assert manager.verify_placement("NVIDIA L40S", "OC-AU-1") == "stock"


def test_4090_has_stock_in_iceland(manager: PodManager) -> None:
    assert manager.verify_placement("NVIDIA GeForce RTX 4090", "EUR-IS-2") == "stock"


def test_community_cloud_is_unverifiable_not_empty(manager: PodManager) -> None:
    """"Cannot tell from here" must not collapse into "nothing there"; one is a
    reason to try anyway, the other a reason not to."""
    assert (
        manager.verify_placement("NVIDIA L40S", "OC-AU-1", cloud="COMMUNITY")
        == "unverifiable"
    )


def test_every_ladder_rung_is_licence_safe_and_actually_offered(
    manager: PodManager,
) -> None:
    """The regression guard. A rung that can never be filled is a silent
    downgrade: the window falls through to a 24GB card and the sharpest quality
    tier is quietly never used."""
    for tier in CANDIDATES:
        assert tier.datacenter in LICENCE_SAFE_DATACENTERS, (
            f"{tier.gpu} in {tier.datacenter} is outside H3's licence"
        )
        verdict = manager.verify_placement(tier.gpu, tier.datacenter, cloud=tier.cloud)
        assert verdict != "not-offered", (
            f"{tier.gpu} is not offered in {tier.datacenter}: this rung can "
            "never be filled, so the ladder silently skips it"
        )


def test_find_capacity_skips_datacenters_outside_the_licence(
    manager: PodManager,
) -> None:
    """L40S has stock in US-TX-3, and it must still never be chosen."""
    gpu, dc = manager.find_capacity(gpus=("NVIDIA L40S",))
    assert dc == "OC-AU-1"
    assert gpu == "NVIDIA L40S"


def test_find_capacity_raises_rather_than_queueing(manager: PodManager) -> None:
    """A request with no capacity must never become a standing order."""
    with pytest.raises(PodError, match="auto-deploy reservation"):
        manager.find_capacity(
            gpus=("NVIDIA L40S",), datacenters=("EUR-IS-1",)
        )
