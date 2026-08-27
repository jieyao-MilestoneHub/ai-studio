# Qwen2-Audio-7B-Instruct

Serves `/說音` from `deploy/inference_server.py` since 2026-08-27, replacing
Qwen3-Omni-30B-A3B-Captioner, which does not fit a 24GB card on transformers 5 (fused MoE experts stay fp16 under bitsandbytes and compressed-tensors alike — see `model-qwen3-omni-captioner.md`). Chosen because it **runs** on the stack the pod already has —
transformers 5.16, one RTX 4090 — not because it is the strongest
audio model; that trade was made explicitly ("效果更差但可以跑").

| | |
|---|---|
| repo | `Qwen/Qwen2-Audio-7B-Instruct` — ungated, **Apache-2.0** `[reported]` per the model card |
| disk | 📏 16.8 GB, whole repo, pre-staged by `deploy/pod_setup.sh` |
| VRAM | `[speculative]` ~16–18 GB resident at fp16; measure on first load |
| prompt | the model is instruction-tuned, so `/說音` asks it for Traditional Chinese first, then English (`inference_server.py`) |
| loader | `Qwen2AudioForConditionalGeneration + AutoProcessor, librosa at the feature extractor's sampling rate` — the model card's own snippet, not a guess |
| input cap | `AI_STUDIO_MAX_AUDIO_UNDERSTAND_S` (30 s), unchanged |

Not yet measured: cold load time, resident VRAM, per-call latency, and
Chinese output quality on real group media. Same slot discipline as the
other backends: one model on the card at a time, evicted by the next kind.
