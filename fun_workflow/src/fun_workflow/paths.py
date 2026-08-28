"""Where this package's own non-Python assets live: the ComfyUI graphs it
adds on top of ai-studio's (`fun_workflow/workflows/`)."""

from __future__ import annotations

from pathlib import Path

from ai_studio.core.errors import AIStudioError


def package_root() -> Path:
    """`<repo>/fun_workflow`, the directory holding this package's assets."""
    return Path(__file__).resolve().parents[2]


def workflow(name: str) -> Path:
    path = package_root() / "workflows" / name
    if not path.is_file():
        raise AIStudioError(f"missing workflow {path}")
    return path
