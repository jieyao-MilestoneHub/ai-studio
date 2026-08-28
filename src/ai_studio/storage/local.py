"""Local-filesystem artifact store.

Sufficient for development, CI, and the whole generation milestone. It cannot
serve a public URL, and says so by returning `None` from `public_url` rather
than inventing a `file://` no delivery channel could ever fetch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ai_studio.core.errors import AIStudioError


class LocalStore:
    """Stores objects under `root`, keyed by relative path."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise AIStudioError(f"unsafe storage key: {key!r}")
        return self.root / key

    def put(self, key: str, source: Path) -> str:
        source = Path(source)
        if not source.is_file():
            raise AIStudioError(f"cannot store missing file: {source}")
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        return self.uri(key)

    def get(self, key: str, dest: Path) -> Path:
        src = self._path(key)
        if not src.is_file():
            raise AIStudioError(f"key not found: {key}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def uri(self, key: str) -> str:
        return self._path(key).resolve().as_uri()

    def local_path(self, key: str) -> Path:
        """Escape hatch for ffmpeg, which wants a path rather than a URI."""
        return self._path(key)

    def public_url(self, key: str) -> str | None:
        return None

    def presign_put(self, key: str, expires_s: int = 900) -> str | None:
        return None
