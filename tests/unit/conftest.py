"""Suite-wide isolation of the two money ledgers.

Both default to a path under the operator's `runs/` and both are written
by code paths the suite exercises (`open_session` records an open,
`close_session` records a cost). Un-isolated, the suite wrote 528 fake
opens into the real `runs/.pod_opens.json` on 2026-08-29 and the worker's
daily cap refused every real pod that day. Every test gets throwaway
ledgers, whether or not it knows it touches one.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_money_ledgers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_studio.runtime import budget, opens

    monkeypatch.setattr(opens, "DEFAULT_OPENS_FILE", tmp_path / "pod_opens.json")
    monkeypatch.setattr(budget, "DEFAULT_LEDGER_FILE", tmp_path / "spend_ledger.json")
