"""Retention: keep the delivery and incoming directories from growing forever.

A 24/7 group bot writes one mp4/png (plus a jpg poster) to `files/` per
finished request and one jpg to `incoming/` per photo someone sends, and
nothing ever removed either. Space is not the near-term worry -- the files
are 1-3 MB -- but "never cleaned" on an always-on host is a leak with a
date on it, so this prunes by age. Time-based, not size-based: a clip is
useful only while the person who asked still cares, which is days, not
forever, and the status page it backs says as much.

Pure filesystem plus an age: no infrastructure, so it runs and tests with
none. Called by `ai-studio gc` and the daily timer the deploy scripts write.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepResult:
    removed: int
    freed_bytes: int
    kept: int

    def summary(self) -> str:
        return (
            f"removed {self.removed} file(s), freed {self.freed_bytes / 1_048_576:.1f} MB, "
            f"kept {self.kept}"
        )


def sweep_old_files(
    directory: Path | str,
    *,
    max_age_days: float,
    now: float | None = None,
    dry_run: bool = False,
    keep: set[str] | None = None,
) -> SweepResult:
    """Delete files under `directory` whose mtime is older than `max_age_days`.

    Non-recursive and files only: these directories are flat. `keep` is a set
    of absolute path strings never to delete regardless of age -- the caller
    passes any file a still-live request references, so an image-to-video
    photo waiting for its render is not pruned out from under it. A missing
    directory is not an error: nothing to sweep is a clean zero.
    """
    root = Path(directory)
    if not root.is_dir():
        return SweepResult(0, 0, 0)
    now = time.time() if now is None else now
    cutoff = now - max_age_days * 86_400
    protected = keep or set()
    removed = freed = kept = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        if str(entry.resolve()) in protected or entry.stat().st_mtime >= cutoff:
            kept += 1
            continue
        size = entry.stat().st_size
        if not dry_run:
            entry.unlink()
        removed += 1
        freed += size
    return SweepResult(removed, freed, kept)
