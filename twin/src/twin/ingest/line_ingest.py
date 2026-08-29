"""One real LINE-export ingest, end to end: file -> Fragments -> JSONL store.

twin/PLAN.md Phase 1's "仍待辦 1" — the library pieces (`ingest.sources.line`,
`ingest.fragment`, `ingest.split`, `ingest.store`) all existed; what did not
was a single function that wires them together with the checks PLAN.md's
Phase 1 acceptance list names, so a real run could only be done by hand-
writing Python. `examples/ingest_line_export.py` is the argv driver; this
module holds the logic so it can be tested with a fictional fixture.

Cutoffs (SPEC.md §4.8/D21): `split` is decided here, once, from the cutoffs
in force at ingest, and is read-only after that. The sealed boundary is
`ingest.split.sealed_cutoff_for` (EVAL.md §9's 20% hold-back), never chosen
by hand.

Overwrite policy: `store.write_fragments_jsonl` overwrites wholesale by
design; this layer refuses to do so unless told explicitly, because a real
fragment store is the upstream of a frozen S1 item bank (PLAN.md Phase 2)
and silently replacing it would orphan that bank's `source_fragment_dataset_hash`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import fsspec

from twin.core.enums import Split
from twin.core.fragment import Fragment
from twin.ingest.fragment import fragments_from_line_export
from twin.ingest.split import sealed_cutoff_for
from twin.ingest.store import write_fragments_jsonl


class IngestRefused(RuntimeError):
    """A pre-write check failed; nothing was written."""


@dataclass(frozen=True)
class SplitSummary:
    count: int
    earliest: datetime | None
    latest: datetime | None


@dataclass(frozen=True)
class IngestSummary:
    total: int
    splits: dict[Split, SplitSummary]
    third_party_tagged: int
    train_cutoff: datetime
    sealed_cutoff: datetime
    uri: str

    def render(self) -> str:
        lines = [
            f"wrote {self.total} fragments -> {self.uri}",
            f"train_cutoff={self.train_cutoff.isoformat()}  sealed_cutoff={self.sealed_cutoff.isoformat()}",
        ]
        for split in (Split.TRAIN, Split.HELDOUT, Split.SEALED):
            s = self.splits[split]
            span = f"{s.earliest.isoformat()} .. {s.latest.isoformat()}" if s.earliest and s.latest else "-"
            lines.append(f"  {split.value:8s} {s.count:6d}  {span}")
        pct = (100.0 * self.third_party_tagged / self.total) if self.total else 0.0
        lines.append(f"third_party_spans tagged: {self.third_party_tagged}/{self.total} ({pct:.1f}%)")
        lines.append("event_time missing: 0 (Fragment.event_time is required by schema)")
        return "\n".join(lines)


def summarize(fragments: list[Fragment], *, train_cutoff: datetime, sealed_cutoff: datetime, uri: str) -> IngestSummary:
    splits: dict[Split, SplitSummary] = {}
    for split in (Split.TRAIN, Split.HELDOUT, Split.SEALED):
        times = sorted(_naive(f.event_time.value) for f in fragments if f.split == split)
        splits[split] = SplitSummary(count=len(times), earliest=times[0] if times else None, latest=times[-1] if times else None)
    return IngestSummary(
        total=len(fragments),
        splits=splits,
        third_party_tagged=sum(1 for f in fragments if f.third_party_spans),
        train_cutoff=train_cutoff,
        sealed_cutoff=sealed_cutoff,
        uri=uri,
    )


def _naive(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=None)


def check_split_order(summary: IngestSummary) -> None:
    """PLAN.md Phase 1 acceptance: heldout time is strictly later than train time.
    With cutoff-based splits this can only fail if a fragment's stored
    `event_time` disagrees with the one its split was decided from — i.e. a
    bug, not a data property — so it raises rather than warns."""
    train, heldout = summary.splits[Split.TRAIN], summary.splits[Split.HELDOUT]
    if train.latest and heldout.earliest and heldout.earliest <= train.latest:
        raise IngestRefused(
            f"heldout earliest ({heldout.earliest.isoformat()}) is not after train latest "
            f"({train.latest.isoformat()}) — split/event_time disagree; refusing to keep this store"
        )


@dataclass(frozen=True)
class LineExportSource:
    """One export file's text plus the closed set of display names in it."""

    text: str
    known_senders: list[str]
    label: str = ""


def ingest_line_export(
    text: str,
    *,
    uri: str,
    principal_id: str,
    principal_display_name: str,
    known_senders: list[str],
    train_cutoff: datetime,
    now: datetime,
    sealed_fraction: float = 0.2,
    overwrite: bool = False,
) -> IngestSummary:
    """Single-export convenience over `ingest_line_exports`."""
    return ingest_line_exports(
        [LineExportSource(text=text, known_senders=known_senders)],
        uri=uri,
        principal_id=principal_id,
        principal_display_name=principal_display_name,
        train_cutoff=train_cutoff,
        now=now,
        sealed_fraction=sealed_fraction,
        overwrite=overwrite,
    )


def ingest_line_exports(
    sources: list[LineExportSource],
    *,
    uri: str,
    principal_id: str,
    principal_display_name: str,
    train_cutoff: datetime,
    now: datetime,
    sealed_fraction: float = 0.2,
    overwrite: bool = False,
) -> IngestSummary:
    """Parse, split, tag, write — several chat rooms of ONE principal, merged
    into one store under one set of cutoffs (SPEC.md §4.8/D21: a store holds
    exactly one split policy, which is why this takes all exports at once
    rather than appending). Raises `IngestRefused` (nothing written) when the
    store already exists without `overwrite`, when any source's
    `known_senders` omits the principal, or when the merged result has no
    held-out / no sealed fragment (Phase 2's item bank has nothing to draw
    from — the fix is an earlier `train_cutoff`, not an empty bank)."""
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path) and not overwrite:
        raise IngestRefused(
            f"fragment store already exists at {uri} — write_fragments_jsonl overwrites wholesale, "
            f"and a frozen S1 item bank may reference it. Pass overwrite=True only if you mean it."
        )
    if not sources:
        raise IngestRefused("no export sources given")
    for source in sources:
        if principal_display_name not in source.known_senders:
            raise IngestRefused(
                f"principal_display_name {principal_display_name!r} is not in known_senders "
                f"{source.known_senders!r} (source {source.label or '?'})"
            )

    sealed_cutoff = sealed_cutoff_for(train_cutoff=train_cutoff, now=now, sealed_fraction=sealed_fraction)
    fragments: list[Fragment] = []
    for source in sources:
        fragments.extend(
            fragments_from_line_export(
                source.text,
                principal_id=principal_id,
                principal_display_name=principal_display_name,
                known_senders=source.known_senders,
                train_cutoff=train_cutoff,
                sealed_cutoff=sealed_cutoff,
            )
        )
    fragments.sort(key=lambda f: f.event_time.value)
    summary = summarize(fragments, train_cutoff=train_cutoff, sealed_cutoff=sealed_cutoff, uri=uri)
    check_split_order(summary)
    if summary.splits[Split.HELDOUT].count == 0:
        raise IngestRefused(
            f"0 held-out fragments (train_cutoff={train_cutoff.isoformat()}, "
            f"sealed_cutoff={sealed_cutoff.isoformat()}, now={now.isoformat()}) — the S1 item bank "
            f"is built only from Split.HELDOUT (EVAL.md §3.1). Move train_cutoff earlier."
        )

    if sealed_fraction > 0 and summary.splits[Split.SEALED].count == 0:
        raise IngestRefused(
            f"0 sealed fragments (sealed_cutoff={sealed_cutoff.isoformat()}, now={now.isoformat()}) — "
            f"EVAL.md §9 requires the 20% hold-back to actually contain data; the export ends "
            f"before the sealed window. Pass an earlier --now (e.g. the export's last timestamp)."
        )

    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    written = write_fragments_jsonl(fragments, uri)
    assert written == summary.total
    return summary
