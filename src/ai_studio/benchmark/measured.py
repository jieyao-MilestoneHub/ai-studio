"""Every number this project has measured on its own hardware, as data.

The docs grade every figure: 📏 measured by us, `[reported]` quoted from
someone else, `[speculative]` inferred (CLAUDE.md, "Number honesty"). Only
the first kind is allowed in here, and every row says when and where it was
measured and which doc records the run. `ai-studio metrics` renders this
table; the docs keep the narrative. If a number is not here it has not been
measured -- do not add one without a run behind it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RTX4090 = "RTX 4090 24 GB"
A40 = "A40 48 GB"
"""RunPod Secure Cloud in both cases."""


@dataclass(frozen=True)
class Measured:
    model: str
    gpu: str
    metric: str
    value: float
    unit: str
    measured_on: str
    """ISO date of the run (YYYY-MM-DD, or YYYY-MM when the doc records only the month)."""
    source: str
    """The doc and section the figure was written down in."""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


MEASURED: tuple[Measured, ...] = (
    # ---- MiniMax H3, RTX 4090, 864x480, int8 + merged LoRA, 6-step turbo, one run each
    Measured("MiniMax H3", RTX4090, "clip 5.17 s (124 frames): submit to mp4", 79, "s", "2026-08-26",
             "docs/model-h3.md, Clip length", "79-100 s across seven production jobs; lower bound listed"),
    Measured("MiniMax H3", RTX4090, "clip 10.1 s (243 frames): submit to mp4", 79, "s", "2026-08-26",
             "docs/model-h3.md, Clip length"),
    Measured("MiniMax H3", RTX4090, "clip 10.1 s (243 frames): VRAM peak", 22.1, "GB", "2026-08-26",
             "docs/model-h3.md, Clip length", "nvidia-smi at 500 ms"),
    Measured("MiniMax H3", RTX4090, "clip 8.0 s (192 frames): submit to mp4", 170, "s", "2026-08-26",
             "docs/model-h3.md, Clip length", "includes ~60 s model reload after a RAM-pressure eviction"),
    Measured("MiniMax H3", RTX4090, "clip 8.0 s (192 frames): VRAM peak", 23.5, "GB", "2026-08-26",
             "docs/model-h3.md, Clip length", "during the reload"),
    Measured("MiniMax H3", RTX4090, "clip 12.25 s (294 frames): submit to mp4", 215, "s", "2026-08-26",
             "docs/model-h3.md, Clip length", "a reload is likely inside this figure"),
    Measured("MiniMax H3", RTX4090, "clip 12.25 s (294 frames): VRAM peak", 23.0, "GB", "2026-08-26",
             "docs/model-h3.md, Clip length"),
    Measured("MiniMax H3", RTX4090, "text-encoder reload after a swap", 60, "s", "2026-08",
             "docs/schedule.md, reaper grace", "60-90 s; lower bound listed"),
    # ---- MiniMax H3, A40, 864x480 / 124 frames, int8 + LoRA bypass (the first measured session)
    Measured("MiniMax H3", A40, "clip 5.17 s (124 frames): VRAM peak", 43.3, "GB", "2026-08",
             "docs/model-h3.md, Measured on our own account", "int8 + LoRA bypass; why sub-48 GB rungs merge the LoRA"),
    Measured("MiniMax H3", A40, "clip 5.17 s (124 frames): sampling, 6-step turbo", 125, "s", "2026-08",
             "docs/model-h3.md, Measured on our own account", "87 -> 40 -> 25 -> 18 s/iter; the first step carries the warm-up"),
    Measured("MiniMax H3", A40, "clip 5.17 s (124 frames): submit to mp4", 300, "s", "2026-08",
             "docs/model-h3.md, Measured on our own account"),
    Measured("MiniMax H3", "any", "weight set on disk, int8 path", 54.7, "GB", "2026-08",
             "docs/model-h3.md, Measured on our own account", "five files"),
    # ---- Flux.1-dev
    Measured("Flux.1-dev", RTX4090, "checkpoint reload after a swap", 15, "s", "2026-08",
             "docs/schedule.md, reaper grace"),
    Measured("Flux.1-dev NSFW LoRA", "any", "adapter on disk", 0.69, "GB", "2026-08",
             "docs/model-flux.md, weights", "687,476,088 bytes per the HF blobs API"),
    # ---- the understanding and chat models, RTX 4090, fp16
    Measured("Qwen2-Audio-7B-Instruct", RTX4090, "VRAM resident, fp16", 17.2, "GB", "2026-08-27",
             "docs/model-qwen2-audio.md"),
    Measured("Qwen2-Audio-7B-Instruct", "any", "repo on disk", 16.8, "GB", "2026-08-27",
             "docs/model-qwen2-audio.md"),
    Measured("Qwen2-Audio-7B-Instruct", RTX4090, "cold load", 27, "s", "2026-08-27",
             "docs/model-qwen2-audio.md, first live run"),
    Measured("Qwen2.5-VL-7B-Instruct", RTX4090, "VRAM resident, fp16", 17.0, "GB", "2026-08-27",
             "docs/model-qwen2.5-vl.md"),
    Measured("Qwen2.5-VL-7B-Instruct", "any", "repo on disk", 16.6, "GB", "2026-08-27",
             "docs/model-qwen2.5-vl.md"),
    Measured("Qwen2.5-VL-7B-Instruct", RTX4090, "cold load", 53, "s", "2026-08-27",
             "docs/model-qwen2.5-vl.md, first live runs", "53-69 s across runs; lower bound listed"),
    Measured("Qwen2.5-VL + Qwen2-Audio", RTX4090, "video description, both models, cold", 105, "s", "2026-08-27",
             "docs/model-qwen2.5-vl.md, first live runs", "video 69 s + audio swap 35 s + two inferences; $0.025 at $0.74/hr"),
    Measured("moondream3-preview", "any", "repo on disk", 47.6, "GB", "2026-08",
             "docs/model-moondream3.md", "four modelv2-* shards are the 18.5 GB actually loaded"),
    Measured("gpt-oss-20b", "any", "repo on disk", 41.3, "GB", "2026-08-27",
             "docs/model-gpt-oss-20b.md", "13.8 GB of safetensors shards are what transformers reads"),
    # ---- the pod itself
    Measured("RunPod pod", "any", "disk overhead, 80 GB volume + 20 GB container", 0.014, "USD/hr", "2026-08",
             "docs/model-h3.md, Measured on our own account", "a $0.74/hr pod bills $0.754"),
)
