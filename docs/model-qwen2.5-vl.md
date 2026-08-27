# Qwen2.5-VL-7B-Instruct

Serves the **picture half** of `/說影` from `deploy/inference_server.py` since
2026-08-27 — the model samples frames only (`process_vision_info`) and cannot
hear, so `_run_job` follows it with Qwen2-Audio on the ffmpeg-extracted track
and joins the two under 【畫面】/【聲音】 (asked 2026-08-27: 「避免 /說影
只根據畫面」). Replacing
Tarsier2-7b-0115, whose `TarsierForConditionalGeneration` exists only in ByteDance's repo pinned to transformers 4.47 (see `model-tarsier2.md`). Chosen because it **runs** on the stack the pod already has —
transformers 5.16, one RTX 4090 — not because it is the strongest
video model; that trade was made explicitly ("效果更差但可以跑").

| | |
|---|---|
| repo | `Qwen/Qwen2.5-VL-7B-Instruct` — ungated, **Apache-2.0** `[reported]` per the model card |
| disk | 📏 16.6 GB, whole repo, pre-staged by `deploy/pod_setup.sh` |
| VRAM | 📏 17.0 GB resident at fp16 (RTX 4090, 2026-08-27) |
| prompt | the model is instruction-tuned, so `/說影` asks it for Traditional Chinese first, then English (`inference_server.py`) |
| loader | `Qwen2_5_VLForConditionalGeneration + AutoProcessor, qwen-vl-utils[decord] for frame sampling (fps=1, max_pixels=360*420)` — the model card's own snippet, not a guess |
| input cap | `AI_STUDIO_MAX_VIDEO_UNDERSTAND_S` (120 s), unchanged and still `[speculative]` |

📏 First live runs (2026-08-27): cold load 53–69 s; as the picture half of
`/說影` the full two-model answer took 105 s cold (video 69 s + audio swap
35 s + two inferences), $0.025; described the ffmpeg test pattern in Traditional Chinese, then restated it in English as asked (fps=1, max_pixels=360*420 on a 4 s 640×360 clip). Not yet
measured: per-call latency on real group media, and output quality beyond
one synthetic probe. Same slot discipline as the
other backends: one model on the card at a time, evicted by the next kind.
