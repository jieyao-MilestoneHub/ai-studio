"""The rule import-linter cannot express for an external package: only the
composition root (`fun_workflow.cli`) may reach ai-studio's pod runtime or
its CLI. Everything else takes sessions and providers by injection
(`pipeline.worker.WindowHost`), which is what keeps the request side
testable with no pod."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "fun_workflow"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+ai_studio\.(runtime|cli)\b", re.M)


def test_only_cli_imports_the_pod_runtime() -> None:
    offenders = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if p.relative_to(SRC).parts[0] != "cli" and FORBIDDEN.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], offenders
