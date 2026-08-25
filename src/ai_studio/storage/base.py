"""Artifact storage.

One protocol, several backends. `LocalStore` is enough for development and for
the whole generation milestone; `S3Store` arrives when the finished mp4 has to
be served to LINE, which hard-requires a public HTTPS host that supports HTTP
range requests.

`presign_put` is the reason the protocol has that method at all: it lets an
orchestrator hand a remote worker a single-object, time-bounded write
capability instead of long-lived credentials. A worker that can only PUT one
key for ten minutes is a much smaller blast radius than one holding an access
key in its environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

_HASH_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Streaming SHA-256. Clips are hundreds of MB; do not read them whole."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@runtime_checkable
class ArtifactStore(Protocol):
    """Where generated clips and finished renders live."""

    def put(self, key: str, source: Path) -> str:
        """Store `source` under `key`. Returns a URI addressing the stored object."""
        ...

    def get(self, key: str, dest: Path) -> Path:
        """Materialise `key` as a local file at `dest`. Returns `dest`."""
        ...

    def exists(self, key: str) -> bool: ...

    def uri(self, key: str) -> str:
        """A URI for `key`, whether or not it exists."""
        ...

    def public_url(self, key: str) -> str | None:
        """A publicly fetchable HTTPS URL, or None if this store cannot serve one.

        `None` is the honest answer for local disk, and callers that need a
        public URL (LINE delivery) must treat it as a hard failure rather than
        substituting something that will not resolve.
        """
        ...

    def presign_put(self, key: str, expires_s: int = 900) -> str | None:
        """A single-object upload URL, or None if unsupported."""
        ...
