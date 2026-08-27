# gpt-oss-20b

The chat model behind the LINE bot's `/himonkey` trigger — a plain-text
message in, a plain-text reply out, no attached media, no tool calling in
this cut. Run as a lazily-loaded backend inside `deploy/inference_server.py`,
alongside MiniMax H3, Flux.1-dev, moondream3-preview, Qwen3-Omni-Captioner,
and Tarsier2 on the same pod. See [line-bot.md](line-bot.md) for the trigger
and [architecture.md](architecture.md) for the provider quartet shape
(`core/chat_spec.py`, sibling to the understanding quartet).

Unlike the three understanding models, this one's licence **is** settled:
`openai/gpt-oss-20b` ships Apache 2.0, no geographic restriction reported
anywhere it has been checked. No licence warning box needed here.

All figures below are `[reported]` or `[speculative]` — nothing here has run
on this project's own hardware yet.

## Weights

`[reported]`, native MXFP4 (the model's own shipped quantization for the
MoE expert weights — not something this project chooses or configures):

| component | size | notes |
|---|---|---|
| gpt-oss-20b, MXFP4 | `[reported]` ~16GB | 21B total params, 3.6B active per token (mixture-of-experts); the vendor's own headline claim is "runs on a single 16GB+ GPU" |

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3/Flux/moondream3/Qwen3-Omni-Captioner/Tarsier2 — RTX 4090 24GB minimum | one pod, one card; see the GPU hand-off in [schedule.md](schedule.md) |
| VRAM headroom | `[speculative]` ~8GB against a 24GB ceiling when nothing else is resident | irrelevant to whether it can ever coexist with H3: H3 alone measures 22.1-22.8GB peak on the real RTX 4090 (`pipeline/convert_worker.py`), so gpt-oss-20b can **never** be simultaneously resident with H3 regardless of its own footprint — it evicts/is-evicted exactly like the three understanding models already do |
| CUDA | same pod, same template as H3 | `deploy/inference_server.py` runs inside ComfyUI's own `.venv-cu128` |
| `kernels==0.16.0` | installed by `deploy/pod_setup.sh` | 📏 2026-08-27: without it transformers 5.16 prints "MXFP4 quantization requires the `kernels` package … defaulting to dequantizing the model to bf16" and the load climbs past 23.7GB on the way to ~40GB — an OOM, not a slow path |
| Disk | 📏 41.3GB — the whole repo, into the standard HF cache, pre-staged by `deploy/pod_setup.sh`'s `dl_repo openai/gpt-oss-20b` | `transformers` reads only the three `model-0000*-of-00002.safetensors` shards (13.8GB); `original/` (raw MXFP4 for the reference implementation) and `metal/` (Apple) are kept anyway by decision. Counted in that script's ~238GB disk-headroom check, so a short volume fails at setup rather than inside the first `/himonkey` request |
| Context window | 128k `[reported]` | far more than one `/himonkey` turn plus its rolling history (`JobQueue.recent_chat_turns()`, capped at the last 10 turns) will ever use; token budget is not the constraint on history length, billed GPU-seconds and reply latency are |

## The two unresolved things, and how they get checked

**1. The harmony channel-tag syntax `_final_channel()` assumes.**
gpt-oss-20b's chat template renders the "harmony" response format —
separate analysis/commentary/final channels — even with zero tool calling in
play, and `GptOssChatBackend.infer()` must return only the final channel or
the LINE reply leaks the model's internal chain-of-thought. The exact
`<|channel|>final<|message|>...<|end|>` marker `_final_channel()` looks for
is a best-effort reading of the published format, not yet verified against a
real generation. Check before trusting `/himonkey` in production:

```bash
uv run ai-studio understand <sample text via a CLI harness, once one exists> --provider chat
```

or, until such a harness exists, a manual pod smoke test per the
`runpod-session` skill: send `/himonkey` a real message, inspect the raw
decode in the server log before channel-splitting, and confirm the marker
actually appears where expected.

**2. `GptOssChatBackend.MAX_NEW_TOKENS` (512), unmeasured against a real
token-to-character ratio.** This cap is the primary defense against an
unbounded generation — see `deploy/inference_server.py`'s module docstring
on why `/unload` racing a still-running `infer()` call is the single
highest-severity risk `/himonkey` introduces, since gpt-oss-20b is the first
backend on this pod whose generation length isn't bounded by a fixed input
shape the way a single-image caption is. 512 tokens is a starting guess
sized with headroom above `MAX_OUTPUT_CHARS` (1000); tune it down once a
real token/character ratio and typical reply length are observed.

## Prompting

`/himonkey` never sends a system prompt or rewrites the user's message
first — unlike H3/Flux, there is no LLM-conversion step
(`pipeline/convert_worker.py::convert_job()` routes `MediaKind.CHAT`
straight to `parsed`). gpt-oss-20b receives the user's own words verbatim,
plus (if any) their rolling conversation history assembled host-side by
`JobQueue.recent_chat_turns()` and shipped over as the `history` form field
on `POST /submit` — never persisted on the pod itself, which is ephemeral.
See [line-bot.md](line-bot.md) and `core/chat_spec.py`'s module docstring.

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

*(nothing yet — pending a real run)*

⛔ **Still unmeasured — do not treat any number in this file as ours**

- Actual VRAM peak during a chat load and during generation on this
  hardware.
- Cold load time, and whether it fits inside RunPod's ~100s pod-proxy
  window (the reason `/submit` is fire-and-forget rather than one blocking
  call, same as the three understanding backends).
- Whether the harmony channel-tag syntax `_final_channel()` assumes matches
  a real generation's actual output.
- A real token-to-character ratio, to retune `MAX_NEW_TOKENS` and the
  `AI_STUDIO_MAX_CHAT_MONTH_USD` / `CHAT_IDLE_MINUTES` starting guesses in
  `config/settings.py` / `runtime/session.py` against real usage.
