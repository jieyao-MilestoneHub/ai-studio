"""Surface protocol. SPEC.md §6.4. No concrete surface exists yet — this
confirms the Protocol shape itself with a minimal fake, the same way
test_agent_tools.py confirms Tool with real tool instances."""

from __future__ import annotations

from typing import Any

from twin.agent.surface.base import Surface
from twin.core.capabilities import SurfaceCapabilities


class _FakeSurface:
    name = "fake"

    def capabilities(self) -> SurfaceCapabilities:
        return SurfaceCapabilities(surface="fake", exposure_signal_available=True)

    def poll_events(self) -> list[dict[str, Any]]:
        return []

    def send(self, content: str, *, to: str) -> None:
        return None


def test_fake_surface_satisfies_the_protocol() -> None:
    surface: Surface = _FakeSurface()
    assert surface.name == "fake"
    assert surface.capabilities().exposure_signal_available is True
    assert surface.poll_events() == []
    assert surface.send("hi", to="someone") is None
