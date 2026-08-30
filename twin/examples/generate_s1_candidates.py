#!/usr/bin/env python3
"""GPU step of an S1 round: every requested system answers every frozen item;
answers are written as `harness.s1_run.S1Candidate` JSONL, one file per
label, to --out-dir. Pairing with R2 and sharding for the judge is
examples/prepare_s1_eval_round.py (CPU, later) — split on purpose so this
can run before Wave 2 exists (EVAL.md §3.2 step 5 does not depend on R2).

    uv run python examples/generate_s1_candidates.py --s1-root file:///... --out-dir /tmp/out \\
        --labels B0,B1,B2,T [--persona-file p.txt] [--transcript-file t.txt] \\
        [--adapter-uri s3://twin-checkpoints/<principal>/<run_id>/final] [--consistency-probe 8]

Labels: B0/B1/B2 per EVAL.md §3.4 (B1 needs --persona-file, B2 needs
--transcript-file); T = base + LoRA adapter (downloaded from R2 and
decrypted into a temp dir — TWIN_ADAPTER_ENCRYPTION_KEY required), asked in
the L4 shape it was trained in (`system: available tools: ...` + the item as
the stimulus; the answer is the `content` of the reply tool call,
`twin.agent.decode.reply_content`) — the 2026-08-30 smoke test showed a bare
user prompt just yields raw `<tool_call>` JSON. **No memory store / recall()
yet** (Phase 7 wiring) — recorded in `model` as "<adapter_uri> (no recall)"
so `prepare_s1_eval_round.py` refuses it as T. `T.raw.jsonl` keeps the
undecoded completions for inspection (how many were no_action / non-reply).

--consistency-probe N: SPEC.md §5.1 "選型 MUST 驗證繁簡一致性" — the first N
items are also asked in Simplified Chinese (zhconv) and both answers, plus
whether each answer is written in Traditional script, go to
consistency-<label>.jsonl for a human to read. Not an EVAL suite score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import zhconv
from run_baseline_inference import HFBaselineBackend

from twin.agent.decode import reply_content
from twin.core.hashing import adapter_hash
from twin.harness.baseline import BaselineId, generate_baseline_samples, render_b0_prompt
from twin.harness.item_bank import read_and_verify_item_bank
from twin.harness.s1_run import BaselineKey, S1Candidate, write_candidates
from twin.harness.schema import S1Item
from twin.ingest.trajectories import V1_TOOLS

BASELINES: tuple[BaselineId, ...] = ("B0", "B1", "B2")


def _read(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8") if path else None


def _download_adapter(adapter_uri: str, scratch: str) -> str:
    from twin.train.checkpoint import _download_and_decrypt_directory

    key = os.environ.get("TWIN_ADAPTER_ENCRYPTION_KEY")
    if not key:
        sys.exit("TWIN_ADAPTER_ENCRYPTION_KEY is required to decrypt the adapter (SPEC.md §8)")
    _download_and_decrypt_directory(adapter_uri, scratch, encryption_key=key.encode())
    return scratch


def _is_traditional(text: str) -> bool:
    return zhconv.convert(text, "zh-tw") == text


def _consistency_probe(
    backend: HFBaselineBackend, items: list[S1Item], *, label: str, n: int, out_dir: Path
) -> None:
    rows = []
    for item in items[:n]:
        trad_prompt = render_b0_prompt(item)
        simp_prompt = zhconv.convert(trad_prompt, "zh-cn")
        trad_answer = backend.complete(trad_prompt)
        simp_answer = backend.complete(simp_prompt)
        rows.append(
            {
                "item_id": item.item_id,
                "answer_to_traditional": trad_answer,
                "answer_to_simplified": simp_answer,
                "traditional_output_for_traditional_input": _is_traditional(trad_answer),
                "traditional_output_for_simplified_input": _is_traditional(simp_answer),
            }
        )
    (out_dir / f"consistency-{label}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    trad_ok = sum(r["traditional_output_for_traditional_input"] for r in rows)
    print(f"[{label}] consistency probe: {trad_ok}/{len(rows)} answers to Traditional input are Traditional script")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s1-root", required=True, help="fsspec URI holding item_bank.jsonl + manifest.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--labels", default="B0", help="comma-separated subset of B0,B1,B2,T")
    parser.add_argument("--persona-file", default=None)
    parser.add_argument("--transcript-file", default=None)
    parser.add_argument("--adapter-uri", default=None)
    parser.add_argument("--consistency-probe", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    labels: list[BaselineKey] = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()]  # type: ignore[misc]
    unknown = set(labels) - {"B0", "B1", "B2", "T"}
    if unknown:
        sys.exit(f"unknown label(s) {sorted(unknown)}")
    items, _ = read_and_verify_item_bank(
        bank_uri=f"{args.s1_root}/item_bank.jsonl", manifest_uri=f"{args.s1_root}/manifest.json"
    )
    persona_text = _read(args.persona_file)
    transcript_text = _read(args.transcript_file)
    if "B1" in labels and persona_text is None:
        sys.exit("B1 requires --persona-file (EVAL.md §3.4)")
    if "B2" in labels and transcript_text is None:
        sys.exit("B2 requires --transcript-file (EVAL.md §3.4)")
    if "T" in labels and not args.adapter_uri:
        sys.exit("T requires --adapter-uri")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    import torch

    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    baseline_labels = [lbl for lbl in labels if lbl != "T"]
    if baseline_labels:
        base = HFBaselineBackend(max_new_tokens=args.max_new_tokens)
        for label in baseline_labels:
            samples = generate_baseline_samples(
                items=items,
                baseline=label,  # type: ignore[arg-type]
                backend=base,
                persona_text=persona_text,
                transcript_text=transcript_text,
            )
            candidates = [
                S1Candidate(
                    item_id=item.item_id, label=label, content=s.content, model=base.model_label, adapter_hash="none", generated_at=now
                )
                for item, s in zip(items, samples, strict=True)
            ]
            write_candidates(candidates, (args.out_dir / f"{label}.jsonl").as_uri())
            print(f"[{label}] {len(candidates)} candidates written")
        if args.consistency_probe:
            _consistency_probe(base, items, label="B0", n=args.consistency_probe, out_dir=args.out_dir)
        del base

    if "T" in labels:
        with tempfile.TemporaryDirectory() as scratch:
            adapter_dir = _download_adapter(args.adapter_uri, scratch)
            twin = HFBaselineBackend(adapter_dir=adapter_dir, max_new_tokens=args.max_new_tokens)
            model_label = f"{args.adapter_uri} (no recall)"
            digest = adapter_hash((Path(adapter_dir) / "adapter_model.safetensors").as_uri())
            system = {"role": "system", "content": f"available tools: {', '.join(V1_TOOLS)}"}  # C2: real names injected at inference
            candidates: list[S1Candidate] = []
            raw_rows: list[str] = []
            no_reply = 0
            for item in items:
                raw = twin.complete_messages([system, {"role": "user", "content": render_b0_prompt(item)}])
                content = reply_content(raw)
                if content is None:
                    no_reply += 1
                    content = ""  # no_action / non-reply tool: judge sees an empty answer -> unjudgeable
                candidates.append(
                    S1Candidate(
                        item_id=item.item_id, label="T", content=content, model=model_label, adapter_hash=digest, generated_at=now
                    )
                )
                raw_rows.append(json.dumps({"item_id": item.item_id, "raw": raw}, ensure_ascii=False))
            write_candidates(candidates, (args.out_dir / "T.jsonl").as_uri())
            (args.out_dir / "T.raw.jsonl").write_text("\n".join(raw_rows) + "\n", encoding="utf-8")
            print(f"[T] {len(candidates)} candidates written; {no_reply} had no reply tool call (no_action or other tool)")
            if args.consistency_probe:
                _consistency_probe(twin, items, label="T", n=args.consistency_probe, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
