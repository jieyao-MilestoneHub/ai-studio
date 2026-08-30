"""S1 run assembly: self-consistency, bundled judge samples, and reattaching
judge verdicts to their originating baseline after blinding strips it.
EVAL.md §3.2-§3.4, §6.1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import fsspec
from pydantic import BaseModel, ConfigDict

from twin.harness.schema import HarnessError, JudgedItem, RawEvalSample, S1Answer, S1Item
from twin.harness.shard import sample_id

BaselineKey = Literal["T", "B0", "B1", "B2"]


class S1Candidate(BaseModel):
    """One system's free-text answer to one frozen item — the GPU step's
    output, persisted so it can be produced before Wave 2 exists (EVAL.md
    §3.2 step 5 is independent of R2) and paired with R2 later, on a CPU.
    `model` records what produced it (base model revision, or the adapter
    URI for T) so a round's manifest can name its `adapter_hash`."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    label: BaselineKey
    content: str
    model: str
    adapter_hash: str  # core.hashing.adapter_hash of the decrypted adapter for T; "none" for B0/B1/B2
    generated_at: datetime


def write_candidates(candidates: list[S1Candidate], uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(candidate.model_dump_json())
            f.write("\n")


def read_candidates(uri: str) -> list[S1Candidate]:
    out: list[S1Candidate] = []
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                out.append(S1Candidate.model_validate_json(stripped))
    return out


def compute_self_consistency(r1: list[S1Answer], r2: list[S1Answer]) -> float:
    """EVAL.md §3.3: self_consistency = agreement(R1, R2) — a direct string
    comparison, no judge involved: `S1Answer.answer` is contractually
    verbatim one of the item's own options, never free text, so R1-vs-R2
    agreement never needs semantic judgment. Items missing from either wave
    are simply not compared (not an error) — only the two waves' shared
    item_ids feed the ratio."""
    r1_by_item = {answer.item_id: answer.answer for answer in r1}
    r2_by_item = {answer.item_id: answer.answer for answer in r2}
    shared_ids = r1_by_item.keys() & r2_by_item.keys()
    if not shared_ids:
        raise HarnessError("R1 and R2 share no item_ids — nothing to compare (EVAL.md §3.3)")
    matches = sum(1 for item_id in shared_ids if r1_by_item[item_id] == r2_by_item[item_id])
    return matches / len(shared_ids)


class S1SampleIndexEntry(BaseModel):
    """Side-channel mapping a judge-visible `sample_id` back to which item
    and baseline it came from. Kept outside `eval/in/`/`eval/out/` — the
    judge is never handed this file: blinding by design strips baseline
    attribution from everything it reads (`harness.shard.strip_source_label`
    only strips `source_label`, but S1's real attribution — which of
    T/B0/B1/B2 a sample came from — never enters `RawEvalSample`/
    `StrippedSample` at all, precisely so there's nothing there to strip)."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    item_id: str
    baseline: BaselineKey


def write_sample_index(index: list[S1SampleIndexEntry], uri: str) -> None:
    """Persists the index across the process boundary between preparing a
    round (`examples/prepare_s1_eval_round.py`) and scoring it after judging
    completes (`examples/score_s1_eval_round.py`, run later, possibly in a
    different session). Deliberately outside `eval/in/`/`eval/out/` — the
    judge dispatch never reads this path."""
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for entry in index:
            f.write(entry.model_dump_json())
            f.write("\n")


def read_sample_index(uri: str) -> list[S1SampleIndexEntry]:
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        return [S1SampleIndexEntry.model_validate_json(line) for line in f if line.strip()]


def _bundle_content(item: S1Item, *, reference_answer: str, candidate_text: str) -> str:
    """The judge needs both the reference (R2, the principal's real answer)
    and the candidate text to render a match/no_match verdict — S1's rubric
    task is "does this candidate agree with what the principal actually
    said," not a standalone quality judgment of the candidate alone."""
    lines = [f"題目：{item.prompt}"]
    if item.options:
        lines.append("選項：" + "、".join(item.options))
    lines.append(f"本人的真實答案（R2）：{reference_answer}")
    lines.append(f"候選答案：{candidate_text}")
    return "\n".join(lines)


def build_s1_raw_samples(
    *,
    items: list[S1Item],
    r2_answers: dict[str, S1Answer],
    candidate_answers: dict[BaselineKey, dict[str, str]],
) -> tuple[list[RawEvalSample], list[S1SampleIndexEntry]]:
    """One bundled sample per (item, baseline) pair. Parametrized over all
    four `BaselineKey` values so Phase 6 reuses this unchanged once a real
    `T` exists — this pass's only callers pass B0/B1/B2. Every produced
    `RawEvalSample` gets `source_label="twin"` — a fixed routing tag, not a
    literal claim (see `harness.baseline.generate_baseline_samples`'s
    docstring for the full reasoning and its limits)."""
    samples: list[RawEvalSample] = []
    index: list[S1SampleIndexEntry] = []
    for item in items:
        if item.item_id not in r2_answers:
            raise HarnessError(f"no R2 answer recorded for item {item.item_id}")
        reference = r2_answers[item.item_id].answer
        for baseline, answers_by_item in candidate_answers.items():
            if item.item_id not in answers_by_item:
                raise HarnessError(f"no {baseline} candidate answer for item {item.item_id}")
            content = _bundle_content(item, reference_answer=reference, candidate_text=answers_by_item[item.item_id])
            sid = sample_id(content)
            samples.append(RawEvalSample(sample_id=sid, source_label="twin", content=content, suite="s1"))
            index.append(S1SampleIndexEntry(sample_id=sid, item_id=item.item_id, baseline=baseline))
    return samples, index


def regroup_judged_items_by_baseline(
    judged: list[JudgedItem], index: list[S1SampleIndexEntry]
) -> dict[BaselineKey, list[JudgedItem]]:
    """Reconstructs `harness.aggregate.aggregate_s1`'s `judged_by_baseline`
    argument after the judge has returned verdicts blind to which baseline
    each sample came from. Raises rather than silently `.get()`-ing past a
    mismatch — a `sample_id` that shows up in one list but not the other
    means either the judge skipped/duplicated a line, or this index doesn't
    match the run it's being applied to; either way the numbers that follow
    would be wrong in a way nothing downstream could detect."""
    index_by_sample_id = {entry.sample_id: entry for entry in index}
    judged_ids = {item.sample_id for item in judged}
    index_ids = set(index_by_sample_id)

    missing_from_index = judged_ids - index_ids
    if missing_from_index:
        raise HarnessError(
            f"{len(missing_from_index)} judged sample_id(s) have no matching index "
            f"entry: {sorted(missing_from_index)[:5]}"
        )
    missing_from_judged = index_ids - judged_ids
    if missing_from_judged:
        raise HarnessError(
            f"{len(missing_from_judged)} index sample_id(s) were never judged: {sorted(missing_from_judged)[:5]}"
        )

    grouped: dict[BaselineKey, list[JudgedItem]] = {}
    for item in judged:
        baseline = index_by_sample_id[item.sample_id].baseline
        grouped.setdefault(baseline, []).append(item)
    return grouped
