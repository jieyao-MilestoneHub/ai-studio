"""`pipeline.residency`: the hand-off is decided by what each provider
declares, never by what the job is."""

from __future__ import annotations

import pytest

from ai_studio.core.errors import ProviderError
from ai_studio.pipeline.residency import make_room_for, residency_group_of


class _Provider:
    def __init__(self, group: str, log: list[str], label: str) -> None:
        self.residency_group = group
        self._log, self._label = log, label

    async def evict(self) -> None:
        self._log.append(self._label)


class _Silent:
    residency_group = "comfyui"


async def test_evicts_only_the_other_side_and_each_instance_once() -> None:
    log: list[str] = []
    comfy = _Provider("comfyui", log, "comfy")
    infer = _Provider("inference", log, "infer")
    providers = {"video": comfy, "image": comfy, "chat": infer, "audio": infer}

    await make_room_for(comfy, providers)
    assert log == ["infer"]
    log.clear()
    await make_room_for(infer, providers)
    assert log == ["comfy"]


async def test_a_provider_without_evict_is_tolerated() -> None:
    await make_room_for(_Provider("inference", [], "x"), {"video": _Silent()})


async def test_an_undeclared_provider_is_refused() -> None:
    with pytest.raises(ProviderError):
        residency_group_of(object())
    with pytest.raises(ProviderError):
        await make_room_for(_Provider("inference", [], "x"), {"video": object()})
