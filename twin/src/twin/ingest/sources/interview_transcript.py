"""Interview transcript ingest. INTERVIEW.md §4 (the four required blocks),
§6 ("逐字稿...掛於記憶層 Period 級"), §7 Q8; SPEC.md §4.9/D23, D26.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from twin.core.enums import Modality, Precision, SourceClass
from twin.core.fragment import Fragment
from twin.ingest.entities import extract_third_party_spans
from twin.ingest.fragment import fragment_from_text_record

# INTERVIEW.md §4: block order and each block's stated duration in minutes.
BLOCK_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
_BLOCK_DURATIONS_MINUTES: dict[str, int] = {"A": 42, "B": 36, "C": 16, "D": 8}

# INTERVIEW.md §4's per-block minutes are a suggested allocation, not a
# guarantee — "訪談員 MAY 在總長不變的前提下於區塊間調整" — so `event_time`
# below is a derived estimate, not a platform-reported timestamp the way
# `fragments_from_line_export`'s `message.sent_at` is. SPEC.md §4.4: a false
# precision is less like the principal than an honestly lower confidence.
_BLOCK_EVENT_TIME_CONFIDENCE = 0.5


def fragments_from_interview_transcript(
    blocks: dict[str, str],
    *,
    principal_id: str,
    session_started_at: datetime,
    known_parties: list[str],
    train_cutoff: datetime,
    sealed_cutoff: datetime,
    ingest_time: datetime | None = None,
) -> Iterator[Fragment]:
    """One Fragment per interview block (INTERVIEW.md §4's A/B/C/D) — the
    coarsest of SPEC.md §4.5's three memory granularities, matching §6's
    "逐字稿...掛於記憶層 Period 級", not per-sentence. `blocks` maps each
    block's label to its already-transcribed text; this function does not
    itself parse a raw transcript into blocks — no interview-transcript
    export format is specified anywhere in INTERVIEW.md to parse against
    (unlike LINE's real export format, which `ingest.sources.line` parses),
    so blocks arriving pre-segmented is the only assumption this function is
    safe to make. Missing blocks are simply skipped (a short/incomplete
    session is INTERVIEW.md §7 Q3's concern, not this function's).

    `event_time` reflects when each block was *spoken* during the known
    session — offset from `session_started_at` by the cumulative duration of
    the blocks that preceded it, per INTERVIEW.md §4's stated per-block
    minutes — never a date the principal recalls *within* their own
    narrative; those stay untouched, verbatim, inside `content` (mirrors how
    `fragments_from_line_export` uses `message.sent_at`, never a date parsed
    out of message text).

    `extract_third_party_spans` runs unconditionally, for every block
    present, with no skip path — this is how INTERVIEW.md §7's Q8 hard
    blocker ("MUST 阻擋：未標註前逐字稿 MUST NOT 進入記憶層") is enforced
    structurally: there is no way to get a `Fragment` out of this function
    that bypassed extraction.

    Raises on any `blocks` key outside `BLOCK_ORDER` — a mislabeled block
    (a typo, an off-by-one) would otherwise vanish silently with no error
    and no test able to catch it, which is exactly the "looks fine but
    isn't" failure shape this codebase's "fail loudly, never silently
    degrade" rule exists to prevent; block B in particular is named in
    INTERVIEW.md §4 as "取得早期硬負例的唯一來源" — losing it silently would
    be expensive. Also raises if `known_parties` is empty: INTERVIEW.md
    §6.3 states an interview "必然涉及第三方" — zero known parties on real
    transcript text would silently produce zero third-party spans despite
    near-certain third-party content, the same "no safe default" reasoning
    `fragments_from_line_export` applies to `principal_display_name`.
    """
    unknown_labels = sorted(set(blocks) - set(BLOCK_ORDER))
    if unknown_labels:
        raise ValueError(
            f"unrecognised interview block label(s) {unknown_labels} — MUST be one of "
            f"{BLOCK_ORDER} (INTERVIEW.md §4); a typo here would otherwise silently drop "
            f"that block's content"
        )
    if not known_parties:
        raise ValueError(
            "known_parties is empty — INTERVIEW.md §6.3: an interview transcript "
            "'必然涉及第三方（家人、同事）'; there is no safe default that still "
            "satisfies §7 Q8's tagging requirement"
        )

    resolved_ingest_time = ingest_time or datetime.now(UTC)
    elapsed = timedelta()
    for label in BLOCK_ORDER:
        duration = timedelta(minutes=_BLOCK_DURATIONS_MINUTES[label])
        if label not in blocks:
            elapsed += duration
            continue
        content = blocks[label]
        spans = extract_third_party_spans(content, known_parties=known_parties)
        yield fragment_from_text_record(
            principal_id=principal_id,
            content=content,
            event_time=session_started_at + elapsed,
            precision=Precision.MINUTE,
            confidence=_BLOCK_EVENT_TIME_CONFIDENCE,
            source_class=SourceClass.SELF_REPORT,
            modality=Modality.TEXT,
            train_cutoff=train_cutoff,
            sealed_cutoff=sealed_cutoff,
            ingest_time=resolved_ingest_time,
            third_party_spans=spans,
        )
        elapsed += duration
