"""Where the repo's non-Python assets live, independent of the process's cwd.

ComfyUI graphs (`workflows/`) and the pod bootstrap (`deploy/`) are files
in the checkout, not package data: the CLI used to find them by relative
path and error with "run from the repo root" otherwise. Both packages in
this repo are used as editable installs, so the checkout is always there
and `__file__` is the honest way to reach it.
"""

from __future__ import annotations

from pathlib import Path

from ai_studio.core.errors import AIStudioError


def repo_root() -> Path:
    """The checkout that contains this package (`<root>/src/ai_studio/paths.py`)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise AIStudioError(
            f"{root} is not the ai-studio checkout (no pyproject.toml); "
            "ai-studio must be installed editable from the repo"
        )
    return root


def workflow(name: str) -> Path:
    """A ComfyUI graph under `workflows/`, e.g. `workflow("flux_dev.json")`."""
    path = repo_root() / "workflows" / name
    if not path.is_file():
        raise AIStudioError(f"missing workflow {path}")
    return path


def deploy_script(name: str) -> Path:
    """A file under `deploy/` that is shipped to the pod."""
    path = repo_root() / "deploy" / name
    if not path.is_file():
        raise AIStudioError(f"missing deploy script {path}")
    return path
