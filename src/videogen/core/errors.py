"""Error types.

Discipline, inherited from video-autopilot-kit: **fail loudly, never silently
degrade.** Upstream documents real videos that shipped broken because a lookup
silently fell back to a default — an unknown colour key rendering as white
through three published shorts. Every registry lookup in this codebase raises.

A silent fallback produces a plausible-looking artifact that is wrong. A raise
produces a stack trace. The stack trace is cheaper.
"""

from __future__ import annotations


class VideogenError(Exception):
    """Base class for every error this package raises deliberately."""


# --------------------------------------------------------------- config


class ConfigError(VideogenError):
    """Settings or policy file is missing, malformed, or has an unknown key."""


class UnknownKeyError(VideogenError):
    """A registry lookup missed.

    Raised instead of returning a default. Carries the offending key and the
    valid set so the message is actionable.
    """

    def __init__(self, kind: str, key: object, valid: object = None) -> None:
        msg = f"unknown {kind}: {key!r}"
        if valid is not None:
            msg += f" (valid: {sorted(valid)})"  # type: ignore[call-overload]
        super().__init__(msg)
        self.kind = kind
        self.key = key


# --------------------------------------------------------------- format


class FormatPolicyViolation(VideogenError):
    """A requested delivery format exceeds the upscale or area-retention limit.

    Carries both measured values so the caller can report what to fix, and
    whether a waiver would make it legal.
    """

    def __init__(
        self,
        message: str,
        *,
        upscale: float | None = None,
        area_retained: float | None = None,
        waivable: bool = True,
    ) -> None:
        super().__init__(message)
        self.upscale = upscale
        self.area_retained = area_retained
        self.waivable = waivable


# --------------------------------------------------------------- providers


class ProviderError(VideogenError):
    """Base for clip-provider failures."""


class UnknownProviderError(ProviderError, UnknownKeyError):
    """`get_provider()` was handed a name that is not registered."""

    def __init__(self, key: object, valid: object = None) -> None:
        UnknownKeyError.__init__(self, "provider", key, valid)


class ProviderSubmitError(ProviderError):
    """The backend refused the job."""


class ProviderJobFailed(ProviderError):
    """The job reached a terminal failed state on the backend."""


class ProviderTimeout(ProviderError):
    """A job did not reach a terminal state inside its budget."""


# --------------------------------------------------------------- comfyui


class GraphValidationError(VideogenError):
    """A ComfyUI workflow graph failed validation before submission.

    The load-bearing case is the MiniMax H3 turbo trap: driving the turbo LoRA
    through the stock ``LoraLoaderModelOnly`` node, or sampling with the generic
    ``KSamplerSelect``, produces vertical comb artifacts and banding *while
    running roughly 4x faster* — so it looks like a speedup and reads as a win
    on a benchmark. This error exists to make that impossible to submit.
    """


# --------------------------------------------------------------- gates


class GateFailure(VideogenError):
    """A gate returned FAIL findings while running in strict mode."""


# --------------------------------------------------------------- runtime


class PodError(VideogenError):
    """Pod lifecycle problem — no capacity, unhealthy host, failed readiness."""


class InsufficientHostRamError(PodError):
    """The allocated host has less RAM than MiniMax H3 needs.

    Upstream field report: a 31 GB host crashed part-way through the second
    consecutive generation. The remedy is to terminate and redeploy, not to
    retry on the same machine.
    """


class CostCeilingExceeded(VideogenError):
    """The estimated cost of a run exceeds the configured ceiling.

    Raised *before* any GPU-second is spent. See docs/architecture.md on why
    gate ordering is architectural rather than cosmetic here.
    """
