"""The Surface interface. SPEC.md §6.4: "Surface MUST 為可插拔配接器，核心 runtime
MUST NOT 認識任何特定平台." No concrete surface (e.g. LINE) is built this pass —
that's Phase 11, and gated on SPEC.md §11 item H's still-unvalidated question
about whether LINE's exposure signal is actually obtainable; see
`core.capabilities`'s docstring for why no placeholder answer ships early.
"""

from __future__ import annotations

from typing import Any, Protocol

from twin.core.capabilities import SurfaceCapabilities


class Surface(Protocol):
    name: str

    def capabilities(self) -> SurfaceCapabilities: ...

    def poll_events(self) -> list[dict[str, Any]]: ...

    def send(self, content: str, *, to: str) -> None: ...
