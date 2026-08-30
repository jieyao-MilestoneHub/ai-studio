"""Where this package's own non-Python assets live: the ComfyUI graphs it
adds on top of ai-studio's (`fun_workflow/workflows/`) and the pod-setup
extensions it ships (`fun_workflow/deploy/pod_setup.d/`)."""

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


def pod_setup_extras() -> list[Path]:
    """This package's `deploy/pod_setup.d/*.sh`: what it wants installed on
    the pod beyond what ai-studio's own setup does (the FaceDetailer nodes
    for drama keyframes). Shipped by the worker at provision time."""
    return sorted((package_root() / "deploy" / "pod_setup.d").glob("*.sh"))
