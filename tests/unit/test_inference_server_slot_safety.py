"""`deploy/inference_server.py`'s GPU-slot arbitration.

The one part of that standalone file worth testing despite the whole file
being outside the `ai_studio` package and otherwise untested/`[speculative]`:
`_ensure_loaded()`'s exception-safety and `_wait_for_idle()`'s in-flight
guard are what stand between an ordinary failed model load / a slow
`/himonkey` reply and a crashed pod process that takes every modality down
with it -- see that file's module docstring. The three backends' own
model-loading code (`MoondreamBackend.load()` etc.) stays untested here;
this file covers only the new slot-arbitration logic, which is pure Python
and needs no GPU.

Imported by file path rather than via `sys.path`, since `deploy/` is not a
package and is deliberately outside `ai_studio` (the file has no access to
`ai_studio.core`/`ai_studio.config` on the real pod either).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_inference_server() -> ModuleType:
    path = REPO / "deploy" / "inference_server.py"
    spec = importlib.util.spec_from_file_location("inference_server_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


srv = _load_inference_server()


class _FakeBackend:
    """A minimal, GPU-free stand-in for a real `ModelBackend`."""

    def __init__(self, modality: str, *, load_error: Exception | None = None) -> None:
        self.modality = modality
        self.model_id = f"fake-{modality}"
        self.accepts_prompt = True
        self._load_error = load_error
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def infer(self, media_path: object, prompt: object, *, history: object = None) -> str:
        return "ok"


@pytest.fixture(autouse=True)
def _reset_slot():
    """Each test gets a clean slot. `_slot` is process-global module state in
    the real server too -- that is the whole point of this file -- but tests
    must not leak into each other."""
    srv._slot.backend = None
    srv._slot.in_flight = 0
    yield
    srv._slot.backend = None
    srv._slot.in_flight = 0


# ----------------------------------------------------------- A.1(a): loading


@pytest.mark.asyncio
async def test_a_failed_load_leaves_the_slot_empty_and_releases_vram(monkeypatch) -> None:
    """A failed `load()` must not orphan VRAM that poisons the next job of
    any kind, nor leave `_slot.backend` pointing at a half-constructed
    object."""
    released = []
    monkeypatch.setattr(srv, "_release_vram", lambda: released.append(True))
    monkeypatch.setitem(
        srv._BACKENDS, "broken",
        lambda: _FakeBackend("broken", load_error=RuntimeError("OOM during load")),
    )

    with pytest.raises(RuntimeError):
        await srv._ensure_loaded("broken")

    assert srv._slot.backend is None
    assert released == [True]


@pytest.mark.asyncio
async def test_a_successful_load_occupies_the_slot(monkeypatch) -> None:
    monkeypatch.setitem(srv._BACKENDS, "fake", lambda: _FakeBackend("fake"))

    backend = await srv._ensure_loaded("fake")

    assert backend.loaded
    assert srv._slot.backend is backend


@pytest.mark.asyncio
async def test_loading_a_new_modality_evicts_the_current_one(monkeypatch) -> None:
    monkeypatch.setitem(srv._BACKENDS, "a", lambda: _FakeBackend("a"))
    monkeypatch.setitem(srv._BACKENDS, "b", lambda: _FakeBackend("b"))

    first = await srv._ensure_loaded("a")
    second = await srv._ensure_loaded("b")

    assert first.unloaded
    assert srv._slot.backend is second


@pytest.mark.asyncio
async def test_the_same_modality_is_not_reloaded(monkeypatch) -> None:
    monkeypatch.setitem(srv._BACKENDS, "a", lambda: _FakeBackend("a"))

    first = await srv._ensure_loaded("a")
    second = await srv._ensure_loaded("a")

    assert first is second
    assert not first.unloaded


# ---------------------------------------------------- A.1(b): in-flight guard


@pytest.mark.asyncio
async def test_wait_for_idle_returns_immediately_when_nothing_is_running() -> None:
    backend = _FakeBackend("x")
    srv._slot.backend = backend
    srv._slot.in_flight = 0

    await srv._wait_for_idle(backend, timeout_s=5.0)  # must not block


@pytest.mark.asyncio
async def test_wait_for_idle_waits_for_in_flight_to_clear() -> None:
    """The core guarantee: eviction does not proceed while `infer()` is
    still running against the backend being evicted."""
    backend = _FakeBackend("x")
    srv._slot.backend = backend
    srv._slot.in_flight = 1

    async def _clear_after_a_moment() -> None:
        await asyncio.sleep(0.05)
        srv._slot.in_flight = 0

    task = asyncio.create_task(_clear_after_a_moment())
    await srv._wait_for_idle(backend, timeout_s=5.0)
    await task

    assert srv._slot.in_flight == 0


@pytest.mark.asyncio
async def test_wait_for_idle_gives_up_after_the_timeout_and_logs() -> None:
    """The last-resort escape hatch: a truly stuck thread must not wedge the
    pod's GPU hand-off forever, even though unloading anyway is risky."""
    backend = _FakeBackend("x")
    srv._slot.backend = backend
    srv._slot.in_flight = 1  # never cleared

    await srv._wait_for_idle(backend, timeout_s=0.05)  # must return, not hang

    assert srv._slot.in_flight == 1, "the flag is left as-is; only the wait gives up"


@pytest.mark.asyncio
async def test_ensure_loaded_waits_for_the_previous_backend_before_evicting(
    monkeypatch,
) -> None:
    """The exact scenario the in-flight guard exists for: an abandoned
    `infer()` call from a job the caller gave up on must not be torn out
    from under by the *next* job's load."""
    monkeypatch.setitem(srv._BACKENDS, "a", lambda: _FakeBackend("a"))
    monkeypatch.setitem(srv._BACKENDS, "b", lambda: _FakeBackend("b"))

    first = await srv._ensure_loaded("a")
    srv._slot.in_flight = 1  # simulate an abandoned infer() still running

    async def _clear_after_a_moment() -> None:
        await asyncio.sleep(0.05)
        srv._slot.in_flight = 0

    task = asyncio.create_task(_clear_after_a_moment())
    second = await srv._ensure_loaded("b")
    await task

    assert first.unloaded, "eviction must still happen, just after the wait"
    assert srv._slot.backend is second


@pytest.mark.asyncio
async def test_unload_endpoint_also_waits_for_in_flight_work() -> None:
    """`POST /unload` (the GPU hand-off to ComfyUI) is the other caller of
    the same guard -- it must not race a still-running generation either."""
    backend = _FakeBackend("x")
    srv._slot.backend = backend
    srv._slot.in_flight = 1

    async def _clear_after_a_moment() -> None:
        await asyncio.sleep(0.05)
        srv._slot.in_flight = 0

    task = asyncio.create_task(_clear_after_a_moment())
    result = await srv.unload()
    await task

    assert result == {"ok": True}
    assert backend.unloaded
    assert srv._slot.backend is None


# ------------------------------------------------------------------- gpt-oss


@pytest.mark.asyncio
async def test_the_chat_backend_is_registered() -> None:
    assert srv._BACKENDS["chat"] is srv.GptOssChatBackend
    assert srv.GptOssChatBackend.modality == "chat"


def test_final_channel_extracts_only_the_final_harmony_channel() -> None:
    """`infer()` must never leak the model's internal chain-of-thought into
    the LINE reply."""
    raw = (
        "<|channel|>analysis<|message|>thinking...<|end|>"
        "<|channel|>final<|message|>the actual reply<|end|>"
    )
    assert srv._final_channel(raw) == "the actual reply"


def test_final_channel_falls_back_to_the_whole_text_if_no_marker_is_found() -> None:
    """A leaked thinking trace is a worse failure than skipped channel-
    splitting, but returning nothing at all is worse still."""
    assert srv._final_channel("plain reply, no channels") == "plain reply, no channels"


@pytest.mark.asyncio
async def test_a_slow_load_does_not_freeze_the_event_loop(monkeypatch) -> None:
    """The first request against real weights (2026-08-27): moondream3's
    ~60s load ran inline in `_ensure_loaded`, every route stalled for the
    duration, /poll returned nothing, and the client's 30s poll timeout
    failed a job that was fine. A load must run off the loop so /poll and
    /healthz keep answering while it happens."""
    import time

    class _SlowBackend(_FakeBackend):
        def load(self) -> None:
            time.sleep(0.5)  # blocking, like from_pretrained
            super().load()

    monkeypatch.setitem(srv._BACKENDS, "slow", lambda: _SlowBackend("slow"))
    srv._slot.backend = None

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        await srv._ensure_loaded("slow")
    finally:
        beat.cancel()
    # A frozen loop would have let the heartbeat run ~0 times during the load.
    assert ticks >= 5, f"event loop starved during load ({ticks} ticks)"


def test_final_channel_cuts_at_return_and_survives_stripped_special_tokens() -> None:
    """📏 Shapes from the first real gpt-oss-20b generation (2026-08-27)."""
    raw = (
        "<|channel|>analysis<|message|>User wants one sentence.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>我是 AI 語言助手。<|return|>"
    )
    assert srv._final_channel(raw) == "我是 AI 語言助手。"
    stripped = "analysisUser wants one sentence.assistantfinal我是 AI 語言助手。"
    assert srv._final_channel(stripped) == "我是 AI 語言助手。"


def test_final_channel_never_leaks_a_truncated_analysis() -> None:
    """📏 2026-08-27: a generation that spent its whole token budget in the
    analysis channel must not be returned as the reply."""
    truncated = "<|channel|>analysis<|message|>The user writes in Chinese and I should"
    out = srv._final_channel(truncated)
    assert "The user writes" not in out and out


@pytest.mark.asyncio
async def test_eviction_sweeps_vram_before_the_next_load(monkeypatch) -> None:
    """📏 job 86 (2026-08-27): the evicted model's memory was not yet
    collected when the next load ran, and it OOMed. The sweep must run
    between unload() and load(), and again on the /unload route."""
    order: list[str] = []
    monkeypatch.setattr(srv, "_release_vram", lambda: order.append("release"))

    class _Tracing(_FakeBackend):
        def load(self) -> None:
            order.append(f"load:{self.modality}")
            super().load()

        def unload(self) -> None:
            order.append(f"unload:{self.modality}")
            super().unload()

    monkeypatch.setitem(srv._BACKENDS, "a", lambda: _Tracing("a"))
    monkeypatch.setitem(srv._BACKENDS, "b", lambda: _Tracing("b"))
    srv._slot.backend = None
    await srv._ensure_loaded("a")
    await srv._ensure_loaded("b")
    assert order == ["load:a", "unload:a", "release", "load:b"]


@pytest.mark.asyncio
async def test_an_oom_load_is_retried_once_after_a_sweep(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(srv, "_release_vram", lambda: calls.append("release"))
    attempts = {"n": 0}

    class _FlakyOOM(_FakeBackend):
        def load(self) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
            super().load()

    monkeypatch.setitem(srv._BACKENDS, "flaky", lambda: _FlakyOOM("flaky"))
    srv._slot.backend = None
    backend = await srv._ensure_loaded("flaky")
    assert backend.modality == "flaky" and attempts["n"] == 2
    assert calls == ["release"]


@pytest.mark.asyncio
async def test_a_non_oom_load_failure_is_not_retried(monkeypatch) -> None:
    monkeypatch.setattr(srv, "_release_vram", lambda: None)
    attempts = {"n": 0}

    class _Broken(_FakeBackend):
        def load(self) -> None:
            attempts["n"] += 1
            raise ValueError("Unrecognized configuration class")

    monkeypatch.setitem(srv._BACKENDS, "broken2", lambda: _Broken("broken2"))
    srv._slot.backend = None
    with pytest.raises(ValueError):
        await srv._ensure_loaded("broken2")
    assert attempts["n"] == 1
