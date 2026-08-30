"""B0/B1/B2 baseline prompt construction and sample generation. EVAL.md §3.4:
the three un-fine-tuned-base-model baselines S1 scores `T` against — B2 in
particular is the project's kill switch ("若 T 未顯著優於 B2，則 LoRA 不應存
在"). None of the three touches a LoRA adapter; only the prompt differs.
"""

from __future__ import annotations

from typing import Literal, Protocol

from twin.core.enums import SourceClass, Split
from twin.harness.schema import HarnessError, RawEvalSample, S1Item
from twin.harness.shard import sample_id
from twin.ingest.store import read_fragments_jsonl

BaselineId = Literal["B0", "B1", "B2"]


class InferenceBackend(Protocol):
    """Free-text completion from a bare, un-fine-tuned base model — a
    different role than `twin.teacher.base.Teacher.generate`'s structured-
    output contract (that's Gemini, generating typed data; this is the
    project's own base model, generating a free-text candidate answer). Real
    binding (`examples/run_baseline_inference.py`, HF `transformers`) is
    deferred to when GPU hardware is available; tests use a fake."""

    def complete(self, prompt: str) -> str: ...


def render_b0_prompt(item: S1Item) -> str:
    """EVAL.md §3.4: B0 = 未微調底模 + 空白 prompt. Nothing but the item
    itself — no persona, no context."""
    lines = [item.prompt]
    if item.options:
        lines.append("選項：" + "、".join(item.options))
    return "\n".join(lines)


def render_b1_prompt(item: S1Item, *, persona_text: str) -> str:
    """EVAL.md §3.4: B1 = 未微調底模 + persona prompt（一段本人自述）."""
    return f"{persona_text}\n\n{render_b0_prompt(item)}"


def render_b2_prompt(item: S1Item, *, transcript_text: str) -> str:
    """EVAL.md §3.4: B2 = 未微調底模 + 訪談 transcript 注入 context — the
    baseline `T` must beat to justify the LoRA existing at all. `transcript_
    text` MUST be reconstructable verbatim from Phase 3 fragments per
    SPEC.md D26; this function only injects it, it does not itself enforce
    that guarantee."""
    return f"{transcript_text}\n\n{render_b0_prompt(item)}"


def _render_prompt(
    item: S1Item, *, baseline: BaselineId, persona_text: str | None, transcript_text: str | None
) -> str:
    if baseline == "B0":
        return render_b0_prompt(item)
    if baseline == "B1":
        if persona_text is None:
            raise HarnessError("baseline B1 requires persona_text (EVAL.md §3.4)")
        return render_b1_prompt(item, persona_text=persona_text)
    if transcript_text is None:
        raise HarnessError("baseline B2 requires transcript_text (EVAL.md §3.4)")
    return render_b2_prompt(item, transcript_text=transcript_text)


def generate_baseline_samples(
    *,
    items: list[S1Item],
    baseline: BaselineId,
    backend: InferenceBackend,
    persona_text: str | None = None,
    transcript_text: str | None = None,
) -> list[RawEvalSample]:
    """Every sample is tagged `source_label="twin"`. S1 never constructs a
    judge-visible "principal" sample at all: R1-vs-R2 self-consistency is a
    direct string comparison (`harness.s1_run.compute_self_consistency`), no
    judge involved, and every baseline this module produces — B0/B1/B2, and
    later `T` — is non-principal by construction (EVAL.md §6.1: "Judge MUST
    不知道受測樣本來自孿生或本人").

    `"twin"` here is a fixed routing tag, not a claim that a given sample IS
    the twin (`RawEvalSample`'s only other legal value is `"principal"`, so
    "not principal" is the only distinction this field can express; B0/B1/B2
    touch no LoRA adapter and are not "孿生" under SPEC.md §2.1's own
    definition — `spec-auditor` flagged this exact gap). This costs nothing
    today: `harness.shard.strip_source_label` deletes the field entirely
    before anything reaches `eval/in/` (`StrippedSample` has no such field
    at all), so the judge never sees it, and real per-baseline attribution
    for S1 runs through the separate, judge-invisible `harness.s1_run.
    S1SampleIndexEntry` side-channel instead. If a future phase (S4, Phase
    12) ever reads `source_label` for real routing rather than as a fixed
    constant, this reasoning MUST be revisited then, not assumed to still
    hold."""
    if baseline == "B1" and persona_text is None:
        raise HarnessError("baseline B1 requires persona_text (EVAL.md §3.4)")
    if baseline == "B2" and transcript_text is None:
        raise HarnessError("baseline B2 requires transcript_text (EVAL.md §3.4)")

    samples: list[RawEvalSample] = []
    for item in items:
        prompt = _render_prompt(item, baseline=baseline, persona_text=persona_text, transcript_text=transcript_text)
        content = backend.complete(prompt)
        samples.append(RawEvalSample(sample_id=sample_id(content), source_label="twin", content=content, suite="s1"))
    return samples


def load_self_report_transcript(fragment_store_uri: str) -> str | None:
    """B2's context: every `SELF_REPORT` fragment's content, verbatim (SPEC.md
    D26), in `event_time` order. Self-report is always `Split.TRAIN`
    (`ingest.split.decide_self_report_split`, 2026-08-30) — T trains on
    exactly this text, B2 reads exactly this text; EVAL.md §3.4's kill
    switch compares the two with the same information. Filtering on TRAIN
    (not "anything non-sealed") also means a future self-report fragment
    that somehow landed in SEALED can never leak in here. Returns None when
    the store has no self-report yet (Phase 3 not run)."""
    fragments = [
        f
        for f in read_fragments_jsonl(fragment_store_uri)
        if f.source_class == SourceClass.SELF_REPORT and f.split == Split.TRAIN
    ]
    if not fragments:
        return None
    fragments.sort(key=lambda f: f.event_time.value)
    return "\n\n".join(f.content for f in fragments)


def default_persona_uri(fragment_store_uri: str) -> str:
    """B1's persona paragraph lives beside the fragment store (e.g.
    `~/twin-data/data/persona.txt`), never at a checkout-relative default —
    a frozen S1 bank was lost once to exactly that (twin/CLAUDE.md)."""
    return fragment_store_uri.rsplit("/", 1)[0] + "/persona.txt"
