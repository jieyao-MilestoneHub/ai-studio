"""The twin/ sensitive-data guardrail must actually block commits.

SPEC.md guardrail 2 (twin/reference/SPEC.md §8, D32) requires a pre-commit hook
that hard-blocks twin/{data,adapters,transcripts,eval}/ from version control.
This asserts the hook exists, is scoped correctly, and would not accidentally
block anything outside that boundary — a hook that's too broad breaks
ai-studio contributors, one that's too narrow leaks a real person's data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

BLOCKED_SAMPLES = [
    "twin/data/messages_export.json",
    "twin/adapters/run_042/adapter_model.safetensors",
    "twin/transcripts/2026-09-01.txt",
    "twin/eval/out/run_042.jsonl",
]

ALLOWED_SAMPLES = [
    "twin/src/twin/core/fragment.py",
    "twin/reference/SPEC.md",
    "twin/README.md",
    "twin/PLAN.md",
    "src/ai_studio/core/model.py",
    # No twin/ prefix — a repo-root dir of the same name must stay out of scope.
    "data/should_not_match.json",
]


def _twin_guardrail_hook() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            files = hook.get("files", "")
            if "twin" in files and "data" in files:
                return hook
    raise AssertionError("no pre-commit hook found guarding twin/{data,adapters,transcripts,eval}/")


def test_config_exists() -> None:
    assert CONFIG.is_file(), "repo has no .pre-commit-config.yaml"


def test_hook_blocks_sensitive_paths() -> None:
    pattern = re.compile(_twin_guardrail_hook()["files"])
    for path in BLOCKED_SAMPLES:
        assert pattern.match(path), f"{path!r} should be blocked but the hook's files regex does not match it"


def test_hook_does_not_block_unrelated_paths() -> None:
    pattern = re.compile(_twin_guardrail_hook()["files"])
    for path in ALLOWED_SAMPLES:
        assert not pattern.match(path), f"{path!r} should NOT be blocked but the hook's files regex matches it"


def test_hook_always_fails_when_triggered() -> None:
    hook = _twin_guardrail_hook()
    assert hook.get("language") == "fail", (
        "the guardrail hook must use pre-commit's built-in `fail` language so "
        "a match always blocks the commit, with no subprocess/shell to get wrong"
    )
