"""The .gitignore must actually ignore what it claims to.

The classic way an open-source repo leaks client footage or a 21 GB weights file
is a .gitignore that was edited once and never verified. Asserting the patterns
here means a well-meaning reorganisation cannot quietly un-ignore `runs/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"

MUST_IGNORE = [
    "runs/",
    "out/",
    "files/",
    "incoming/",
    "*.mp4",
    "*.wav",
    "*.jpg",
    "*.png",
    "*.safetensors",
    "*.ckpt",
    ".env",
    "*.local.toml",
]


def _patterns() -> set[str]:
    text = GITIGNORE.read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gitignore_exists() -> None:
    assert GITIGNORE.is_file(), "the repo has no .gitignore"


@pytest.mark.parametrize("pattern", MUST_IGNORE)
def test_pattern_is_present(pattern: str) -> None:
    assert pattern in _patterns(), f"{pattern!r} is no longer ignored"


def test_env_example_is_not_ignored() -> None:
    """`.env.*` would swallow the template that documents the key names."""
    assert "!.env.example" in _patterns()


def test_env_example_contains_names_but_no_values() -> None:
    """A committed example file with a real value in it is the leak."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file()

    assignments = [
        line
        for line in example.read_text(encoding="utf-8").splitlines()
        if re.match(r"^[A-Z][A-Z0-9_]*=", line)
    ]
    assert assignments, ".env.example documents no keys"

    for line in assignments:
        name, _, value = line.partition("=")
        # A numeric default (a cost ceiling) is fine; a credential-shaped one is not.
        if value and not re.fullmatch(r"[\d.]+", value):
            raise AssertionError(f".env.example has a non-numeric value for {name}: {line!r}")
