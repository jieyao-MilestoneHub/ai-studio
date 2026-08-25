"""Provider lookup.

Raises on an unknown name rather than falling back to a default. Falling back
to `stub` because a name was misspelled would produce a run that looks like it
worked and contains synthetic test-pattern footage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_studio.core.errors import UnknownProviderError
from ai_studio.providers.base import ClipProvider, ImageProvider

ProviderFactory = Callable[..., ClipProvider | ImageProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


def get_provider(name: str, **kwargs: Any) -> ClipProvider | ImageProvider:
    _ensure_builtins()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownProviderError(name, _REGISTRY.keys()) from None
    return factory(**kwargs)


_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import built-in providers lazily.

    Deferred so that `ai-studio doctor` and the offline test suite do not pay
    for httpx/boto3 import time, and so a broken optional provider cannot stop
    the CLI from starting.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True

    from ai_studio.providers.stub import StubProvider

    register("stub", StubProvider)

    from ai_studio.providers.comfyui import ComfyUIProvider

    register("comfyui", ComfyUIProvider)

    from ai_studio.providers.flux import FluxComfyUIProvider

    register("flux", FluxComfyUIProvider)
