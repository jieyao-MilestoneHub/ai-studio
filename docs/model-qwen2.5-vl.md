# Qwen2.5-VL-7B-Instruct

Serves `/說影` from `deploy/inference_server.py` since 2026-08-27, replacing
Tarsier2-7b-0115, whose `TarsierForConditionalGeneration` exists only in ByteDance's repo pinned to transformers 4.47 (see `model-tarsier2.md`). Chosen because it **runs** on the stack the pod already has —
transformers 5.16, one RTX 4090 — not because it is the strongest
video model; that trade was made explicitly ("效果更差但可以跑").

| | |
|---|---|
| repo | `Qwen/Qwen2.5-VL-7B-Instruct` — ungated, **Apache-2.0** `[reported]` per the model card |
| disk | 📏 16.6 GB, whole repo, pre-staged by `deploy/pod_setup.sh` |
| VRAM | `[speculative]` ~16–18 GB resident at fp16; measure on first load |
| prompt | the model is instruction-tuned, so `/說影` asks it for Traditional Chinese first, then English (`inference_server.py`) |
| loader | `Qwen2_5_VLForConditionalGeneration + AutoProcessor, qwen-vl-utils[decord] for frame sampling (fps=1, max_pixels=360*420)` — the model card's own snippet, not a guess |
| input cap | `AI_STUDIO_MAX_VIDEO_UNDERSTAND_S` (120 s), unchanged and still `[speculative]` |

Not yet measured: cold load time, resident VRAM, per-call latency, and
Chinese output quality on real group media. Same slot discipline as the
other backends: one model on the card at a time, evicted by the next kind.
