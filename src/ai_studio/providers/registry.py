"""Provider lookup.

Raises on an unknown name rather than falling back to a default. Falling back
to `stub` because a name was misspelled would produce a run that looks like it
worked and contains synthetic test-pattern footage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_studio.core.errors import UnknownProviderError
from ai_studio.providers.base import (
    ChatProvider,
    ClipProvider,
    ImageProvider,
    UnderstandingProvider,
)

ProviderFactory = Callable[..., ClipProvider | ImageProvider | UnderstandingProvider | ChatProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


def get_provider(
    name: str, **kwargs: Any
) -> ClipProvider | ImageProvider | UnderstandingProvider | ChatProvider:
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

    from ai_studio.providers.stub import StubImageProvider, StubProvider, StubUnderstandingProvider

    register("stub", StubProvider)
    register("stub-understanding", StubUnderstandingProvider)
    register("stub-flux", StubImageProvider)

    from ai_studio.providers.comfyui import ComfyUIProvider

    register("comfyui", ComfyUIProvider)

    from ai_studio.providers.flux import FluxComfyUIProvider

    register("flux", FluxComfyUIProvider)

    from ai_studio.core.enums import MediaKind
    from ai_studio.providers.understanding import UnderstandingProvider

    # Three names, one class: each binds a different `modality` at
    # construction time so `--provider understand-image` (etc) is selectable
    # directly, even though all three share one wire protocol.
    for name, modality in (
        ("understand-image", MediaKind.IMAGE_UNDERSTAND),
        ("understand-audio", MediaKind.AUDIO_UNDERSTAND),
        ("understand-video", MediaKind.VIDEO_UNDERSTAND),
    ):
        register(name, lambda modality=modality, **kw: UnderstandingProvider(modality=modality, **kw))

    from ai_studio.providers.chat import ChatProvider as _ChatProvider

    register("chat", _ChatProvider)
