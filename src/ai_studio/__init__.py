"""ai-studio: open-weight video, image and understanding models on rented
GPUs, measured honestly.

This package owns the pod (`runtime`), the models behind one provider
protocol (`providers`, `comfy`, `inference`), the prompt builders
(`prompts`), the money guards (`runtime.budget`, `runtime.opens`) and the
measurements (`benchmark`). It knows nothing about who asked for a render;
the request side is the sibling package `fun_workflow`. See README.md.
"""

__version__ = "0.1.0"
