"""`files/index.jsonl`: which delivered file belongs to which request.

Delivered media lands under `AI_STUDIO_FILES_DIR` with a random token for a
name (`OD5_XWwNXUU.png`) -- fine for a URL nobody should guess, useless for
"what is this file, who asked for it, when". One appended line per delivery
is the map back: ts, token, job id, kind, path, size, sha256. The archive
carries it; `gc` protects it.

Append-only with `O_APPEND` so the worker and a manual `drain` can both add
lines without clobbering. Never raises into the pipeline: an index that
could not be written is logged, the delivery still happens.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ai_studio.core.observability import utc_now_iso
from ai_studio.storage.base import sha256_file

_log = logging.getLogger("ai_studio.index")

INDEX_NAME = "index.jsonl"


def index_path(files_dir: Path | str) -> Path:
    return Path(files_dir) / INDEX_NAME


def append_delivery(
    files_dir: Path | str,
    *,
    token: str,
    job_id: int,
    kind: str,
    path: Path | str,
    cost_usd: float | None = None,
) -> dict[str, object] | None:
    """Record one delivered file. Returns the line written, or None on failure."""
    target = Path(path)
    try:
        record: dict[str, object] = {
            "ts": utc_now_iso(),
            "token": token,
            "job_id": job_id,
            "kind": kind,
            "path": str(target),
            "bytes": target.stat().st_size if target.is_file() else None,
            "sha256": sha256_file(target) if target.is_file() else None,
        }
        if cost_usd is not None:
            record["cost_usd"] = cost_usd
        dest = index_path(files_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return record
    except Exception as exc:
        _log.warning("delivery index not written for %s: %s", target, exc)
        return None
