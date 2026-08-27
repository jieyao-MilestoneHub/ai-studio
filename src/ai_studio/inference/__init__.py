"""HTTP client for the pod-side understanding-model server.

Sibling to `ai_studio.comfy`, not part of it: moondream3, Qwen3-Omni-Captioner,
and Tarsier2 are not ComfyUI nodes, so they are served by a small separate
process (`deploy/inference_server.py`) rather than a ComfyUI graph. See
`docs/architecture.md` for how the two servers share one 24GB card.
"""

from __future__ import annotations
