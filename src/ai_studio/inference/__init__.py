"""HTTP client for the pod-side understanding-model server.

Sibling to `ai_studio.comfy`, not part of it: moondream3, Qwen2-Audio,
Qwen2.5-VL and gpt-oss-20b are not ComfyUI nodes, so they are served by a small separate
process (`deploy/inference_server.py`) rather than a ComfyUI graph. See
`docs/architecture.md` for how the two servers share one 24GB card.
"""

from __future__ import annotations
