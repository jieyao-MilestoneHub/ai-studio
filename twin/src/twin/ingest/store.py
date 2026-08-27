"""Fragment persistence. SPEC.md §7.2: all data-artifact paths MUST be URIs
handled through fsspec, MUST NOT be bare local paths — even though today this
only ever resolves to `file://`, until Phase 8 makes R2 real."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import fsspec

from twin.core.fragment import Fragment


def write_fragments_jsonl(fragments: Iterable[Fragment], uri: str) -> int:
    """Overwrites `uri` wholesale — append is deliberately not offered.
    A stored fragment's `split` is a fact about the cutoffs in force at the
    moment it was ingested (SPEC.md §4.8/D21); silently appending fragments
    decided under different cutoffs into the same file would let two
    incompatible split policies coexist undetected."""
    count = 0
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for fragment in fragments:
            f.write(fragment.model_dump_json())
            f.write("\n")
            count += 1
    return count


def read_fragments_jsonl(uri: str) -> Iterator[Fragment]:
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield Fragment.model_validate_json(stripped)
