"""What ai-studio still keeps between a provider and a caller: the one-card
model hand-off (`residency.py`) and the pod-side prompt rewriter presented
as an `LlmClient` (`pod_llm.py`). The request queue, worker loop and drain
that use them live in `fun_workflow.pipeline`.
"""
