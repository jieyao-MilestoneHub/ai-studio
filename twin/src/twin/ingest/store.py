"""Fragment and Trajectory persistence. SPEC.md §7.2: all data-artifact paths
MUST be URIs handled through fsspec, MUST NOT be bare local paths — even
though today this only ever resolves to `file://`, until Phase 8 makes R2
real. Trajectory storage lives here too, not in `twin.train`: serialization is
an L1 concern regardless of which layer produced the record (ingest sources,
and later `agent.reflow`'s veto-to-hard-negative flow, both write trajectories
through the normal ingest path — SPEC.md §6.5, data-contract skill rule 5)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import fsspec

from twin.core.enums import SourceClass
from twin.core.fragment import Fragment
from twin.core.trajectory import Trajectory


def write_fragments_jsonl(fragments: Iterable[Fragment], uri: str) -> int:
    """Overwrites `uri` wholesale — append is deliberately not offered.
    A stored fragment's `split` is a fact about the cutoffs in force at the
    moment it was ingested (SPEC.md §4.8/D21); silently appending fragments
    decided under different cutoffs into the same file would let two
    incompatible split policies coexist undetected.

    Refuses to write any `SourceClass.SELF_REPORT` fragment to a non-
    `file://` URI (INTERVIEW.md §6.3/§8 I-D). Interview-transcript blocks
    become `SELF_REPORT` fragment `content` verbatim, by design, to satisfy
    SPEC.md D26's round-trip requirement — so a `SELF_REPORT` fragment isn't
    merely *related to* the transcript, it structurally *is* transcript text
    for storage purposes. Applied to every `SELF_REPORT` fragment
    (questionnaire responses included), not narrowed to "only the ones that
    came from an interview": nothing in `Fragment` currently distinguishes
    which self-report source produced a given fragment, and erring toward
    keeping all self-report content local costs nothing today. This is the
    actual enforcement point for that guardrail — `config.settings.
    Settings.transcript_store_uri` constrains where raw audio/transcript
    *files* may live, but says nothing about `Fragment` records written
    through this function, which is why this check has to live here too."""
    materialized = list(fragments)
    if not uri.startswith("file://"):
        leaked = [f.fragment_id for f in materialized if f.source_class == SourceClass.SELF_REPORT]
        if leaked:
            raise ValueError(
                f"refusing to write {len(leaked)} SELF_REPORT fragment(s) to a non-file:// "
                f"URI ({uri}) — INTERVIEW.md §6.3/§8 I-D: self-report content (which includes "
                f"verbatim interview transcript text, per SPEC.md D26) MUST NOT enter "
                f"cross-cloud storage until a real disposition decision (SPEC.md §11-C) exists"
            )
    count = 0
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for fragment in materialized:
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


def write_trajectories_jsonl(trajectories: Iterable[Trajectory], uri: str) -> int:
    """Same full-overwrite semantics as `write_fragments_jsonl`, same reason."""
    count = 0
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(trajectory.model_dump_json())
            f.write("\n")
            count += 1
    return count


def read_trajectories_jsonl(uri: str) -> Iterator[Trajectory]:
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield Trajectory.model_validate_json(stripped)
