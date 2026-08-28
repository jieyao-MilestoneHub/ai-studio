"""Daily pod-open ledger: the backstop behind the monthly budget guard.

The guard in `runtime.budget` sees dollars; it cannot see the failure a
crash-looping worker produces — a fresh pod on every restart, each one
individually inside budget. This ledger counts *opens*, so
`runtime.session.ensure_pod` can refuse the sixteenth of the day.

Recorded at open rather than at close, because the cap exists to stop a
second pod being created — by which point no close has happened yet. The
spend ledger records the other half at close.

A JSON file beside the spend ledger rather than a table in the request
queue, so that `runtime` depends on nothing above it: the queue belongs to
whoever submits work, and this count is the pod's own business.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEFAULT_OPENS_FILE = Path("runs/.pod_opens.json")

RETAIN_S = 2 * 86_400
"""Entries older than this are dropped on the next write. The only reader is
"how many since local midnight", and two days covers every timezone edge."""


class PodOpenLedger:
    def __init__(self, path: Path | str = DEFAULT_OPENS_FILE) -> None:
        self.path = Path(path)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        opens = data.get("opens", []) if isinstance(data, dict) else []
        return [o for o in opens if isinstance(o, dict)]

    def _write(self, opens: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"opens": opens}, indent=2), encoding="utf-8")

    def record(self, pod_id: str, *, when: float | None = None) -> None:
        """Note that a pod was created. Called by whoever created it."""
        now = time.time() if when is None else when
        opens = [o for o in self._read() if float(o.get("opened_at", 0)) >= now - RETAIN_S]
        opens.append({"pod_id": pod_id, "opened_at": now})
        self._write(opens)

    def count_since(self, since: float) -> int:
        """How many pods were created at or after `since` (a POSIX timestamp)."""
        return sum(1 for o in self._read() if float(o.get("opened_at", 0)) >= since)
