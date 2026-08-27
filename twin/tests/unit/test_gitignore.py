"""The twin/.gitignore must actually ignore what SPEC.md §8 guardrail 2 requires.

Adapted from the root repo's tests/unit/test_gitignore.py: a .gitignore that was
edited once and never verified is exactly how a real person's raw data ends up
committed. Asserting the patterns here means a reorganisation cannot quietly
un-ignore data/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TWIN_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = TWIN_ROOT / ".gitignore"

MUST_IGNORE = [
    "data/",
    "adapters/",
    "transcripts/",
    "eval/",
]


def _patterns() -> set[str]:
    text = GITIGNORE.read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gitignore_exists() -> None:
    assert GITIGNORE.is_file(), "twin/ has no .gitignore"


@pytest.mark.parametrize("pattern", MUST_IGNORE)
def test_pattern_is_present(pattern: str) -> None:
    assert pattern in _patterns(), f"{pattern!r} is no longer ignored"
