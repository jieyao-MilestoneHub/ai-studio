"""Paraphrase augmentation of interview trajectories. SPEC.md §5.2/D9 (the
Teacher as data factory), §4.10 `ground_truth_source=teacher_synthesized`,
D25 (synthesized trajectories MUST NOT reach any EVAL suite — enforced by
`harness.schema.reject_synthesized_for_eval`), D19 (self-report is the
dominant persona source), C1 (no fabricated names/dates/places).

Why paraphrase instead of repeating (user decision 2026-08-30): repeating
the same 30 QA pairs 20× teaches recitation of the exact wording; K
rewrites in the principal's own register teach the *stance* across
phrasings. The Teacher is told to keep every fact, name, number and time
exactly, change only wording, and keep the register (其實／蠻／哈哈, short
clauses). The originals stay in the store as `observed`; variants are
`teacher_synthesized` and carry the same split/exposure/no-negative shape.

One Teacher call per source trajectory (D9: "少次、大批" — K variants per
call, not one call per variant).
"""

from __future__ import annotations

from collections.abc import Iterator

import zhconv
from pydantic import BaseModel, ConfigDict

from twin.core.enums import GroundTruthSource
from twin.core.trajectory import ActionStep, Exposure, Trajectory
from twin.ingest.interview_trajectories import INTERVIEW_SURFACE
from twin.teacher.base import Teacher


class _Variant(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str


class _VariantsPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    variants: list[_Variant]


def _prompt(question: str, answer: str, *, variants: int, style_samples: list[str]) -> str:
    samples = "\n".join(f"- {s}" for s in style_samples[:6])
    return (
        f"下面是一位受訪者在訪談中被問到的問題與他的原始回答。請產生 {variants} 組改寫版本，每組含改寫後的問題與改寫後的回答。\n"
        "硬性規則：\n"
        "1. 回答 MUST 保留原始回答的所有事實、立場、人名、地名、時間、數字，MUST NOT 新增任何原文沒有的資訊或例子，也不可刪掉主要內容。\n"
        "2. 只改措辭、句序、開頭語、口語詞，讓每一組彼此明顯不同。\n"
        "3. 回答 MUST 保持受訪者本人的口吻：台灣華語口語、短句、直白、第一人稱；語氣強度 MUST 與原文一致——"
        "原文沒有的語助詞、感嘆、流行語或俚語（例如「超」「衝一波」「撩落去」「大夥兒」「開搞」「有夠」）MUST NOT 加進去，"
        "也 MUST NOT 變成書面、條列或客服語氣。改寫的重點是換句話說，不是變得更活潑。\n"
        "4. 問題可以換成不同的問法（更口語、換切入角度），但問的是同一件事。\n"
        "5. 全部台灣繁體中文，MUST NOT 出現任何簡體字；原文中的英文名詞（如工具名、公司名）原樣保留。\n\n"
        f"受訪者平常說話的樣本（只供模仿語氣）：\n{samples}\n\n"
        f"原始問題：{question}\n原始回答：{answer}"
    )


def variant_trajectory(source: Trajectory, *, question: str, answer: str) -> Trajectory:
    """One `teacher_synthesized` paraphrase of `source` — shared by the Gemini
    path below and by `examples/import_self_report_variants.py`, which reads
    paraphrases written by Claude Code (the user's 2026-08-30 choice: the
    interviewer session already holds the principal's voice; SPEC.md §5.2
    lets the Teacher implementation be swapped, and this is that swap in
    file form). Text is normalised to Traditional (SPEC.md §5.1)."""
    step = source.steps[0]
    if not (isinstance(step, ActionStep) and step.surface == INTERVIEW_SURFACE):
        raise ValueError(f"trajectory {source.trajectory_id} is not an interview trajectory")
    question = zhconv.convert(question.strip(), "zh-tw") or source.exposure.stimulus
    answer = zhconv.convert(answer.strip(), "zh-tw")
    if not answer or answer == step.content.strip():
        raise ValueError("variant answer is empty or identical to the original")
    return Trajectory(
        principal_id=source.principal_id,
        context_time=source.context_time,
        split=source.split,
        exposure=Exposure(occurred=True, stimulus=question, evidence=source.exposure.evidence),
        observation=f"訪談員：{question}",
        available_tools=list(source.available_tools),
        steps=[ActionStep(surface=INTERVIEW_SURFACE, content=answer)],
        negative_class=source.negative_class,
        ground_truth_source=GroundTruthSource.TEACHER_SYNTHESIZED,
    )


def augment_interview_trajectories(
    trajectories: list[Trajectory], *, teacher: Teacher, variants_per_trajectory: int, style_samples: list[str]
) -> Iterator[Trajectory]:
    """For each source (observed, interview-surface) trajectory yields up to
    `variants_per_trajectory` new trajectories. Variants whose answer is
    empty or identical to the original are dropped, never padded."""
    if variants_per_trajectory < 1:
        raise ValueError("variants_per_trajectory must be >= 1")
    for source in trajectories:
        step = source.steps[0]
        if not (isinstance(step, ActionStep) and step.surface == INTERVIEW_SURFACE):
            raise ValueError(f"trajectory {source.trajectory_id} is not an interview trajectory")
        payload = teacher.generate(
            _prompt(source.exposure.stimulus, step.content, variants=variants_per_trajectory, style_samples=style_samples),
            response_schema=_VariantsPayload,
        )
        seen = {step.content.strip()}
        for variant in payload.variants[:variants_per_trajectory]:
            try:
                built = variant_trajectory(source, question=variant.question, answer=variant.answer)
            except ValueError:
                continue
            content = built.steps[0].content  # type: ignore[union-attr]
            if content in seen:
                continue
            seen.add(content)
            yield built
