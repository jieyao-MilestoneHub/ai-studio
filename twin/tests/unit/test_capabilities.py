"""SurfaceCapabilities. SPEC.md §4.3/§6.4. No concrete per-surface instance
lives in twin.core.capabilities itself — see that module's docstring for why
(SPEC.md §11 item H is unvalidated, not decided false)."""

from __future__ import annotations

from twin.core.capabilities import SurfaceCapabilities


def test_surface_capabilities_constructs_with_signal_available() -> None:
    caps = SurfaceCapabilities(surface="rss", exposure_signal_available=True)
    assert caps.exposure_signal_available is True
    assert caps.exposure_signal_note == ""


def test_surface_capabilities_carries_a_note_when_unavailable() -> None:
    caps = SurfaceCapabilities(
        surface="some_platform",
        exposure_signal_available=False,
        exposure_signal_note="platform API does not expose read receipts",
    )
    assert caps.exposure_signal_available is False
    assert caps.exposure_signal_note
