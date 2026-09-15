"""Test harness for the online duplex app.

The Thinker and the worker gateway are stubbed: these tests are about session
lifecycle - who holds the single model slot, when it is freed, what survives a
bad frame - not about inference. Nothing here loads a model.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest


class StubThinker:
    """Stands in for ``GanderDuplexSession``."""

    def __init__(self) -> None:
        self.closed = False
        self.close_count = 0
        self.interrupted = 0
        self.frames: list[Any] = []

    def talker_state(self) -> dict[str, Any]:
        return {"generation_id": 0}

    def close(self, *, drain_speech: bool = False) -> None:
        self.close_count += 1
        self.closed = True

    # --- explicit-stop drain path ---------------------------------------
    def flush_pending(self, *, unit_capture_start_ms=None) -> tuple:
        return ()

    def should_continue_draining(self, _steps: int) -> bool:
        return False

    def step_silence(self):
        raise AssertionError("stub thinker never steps silence")

    def should_stop_after(self, _event) -> bool:
        return True

    def wait_for_speech(self) -> None:
        return None

    def interrupt_output(self) -> None:
        self.interrupted += 1

    def poll_output(self, _timeout: float) -> None:
        return None

    def drain_outputs(self) -> tuple[Any, ...]:
        return ()

    def enqueue_screen_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def set_media_mode(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class StubProvider:
    def __init__(self, *, fail: bool = False, gate: threading.Event | None = None):
        self.fail = fail
        self.gate = gate
        self.warmups = 0

    async def warmup(self) -> None:
        self.warmups += 1
        if self.gate is not None:
            # Hold warmup open so a test can prove `ready` does not wait on it.
            # Polled rather than blocked: occupying a default-executor thread
            # would starve the app's own asyncio.to_thread calls.
            for _ in range(500):
                if self.gate.is_set():
                    break
                await asyncio.sleep(0.01)
        if self.fail:
            raise RuntimeError("ornith unreachable")


class StubGateway:
    def __init__(self, provider: StubProvider) -> None:
        self.providers = {"stub": provider}
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class StubCoordinator:
    """Stands in for ``TaskToolsRealtimeCoordinator``."""

    def __init__(self, gateway: StubGateway, **_kwargs: Any) -> None:
        self.gateway = gateway
        self.started = 0
        self.model_jobs_stopped = 0
        self.closed = 0
        self._outputs: asyncio.Queue[Any] = asyncio.Queue()

    async def start(self) -> None:
        self.started += 1

    async def stop_model_jobs(self) -> None:
        self.model_jobs_stopped += 1

    async def close(self, *, discard_state: bool = False) -> None:
        self.closed += 1
        await self.gateway.close()

    async def next_output(self) -> Any:
        # Never resolves: the outbound loop just waits.
        return await self._outputs.get()

    def record_pcm16(self, _audio: bytes) -> None:
        return None

    def task_status(self) -> dict[str, Any]:
        return {"tasks": []}

    def remember_media(self, _media: Any) -> None:
        return None

    def model_output(self, _event: Any) -> Any:
        raise AssertionError("stub coordinator emits no model output")

    def observe_frontbrain(self, _event: Any) -> None:
        return None


@dataclass
class StubModel:
    vpm: object = field(default_factory=object)
    resampler: object = field(default_factory=object)


@dataclass
class StubBundle:
    model: StubModel = field(default_factory=StubModel)


@dataclass
class StubParams:
    chunk_ms: int = 1000
    generate_audio: bool = False
    sliding_window_mode: str = "context_no_previous"
    context_max_units: int = 64
    context_previous_max_tokens: int = 0
    decode_mode: str = "sampling"


@dataclass
class Harness:
    app: Any
    provider: StubProvider
    gateway: StubGateway
    thinkers: list[StubThinker]
    coordinators: list[StubCoordinator]
    open_gate: threading.Event | None = None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Build the duplex app with the Thinker and gateway stubbed out."""

    def _build(
        *,
        warmup_fails: bool = False,
        warmup_gate: threading.Event | None = None,
        open_gate: threading.Event | None = None,
        **settings_kwargs: Any,
    ) -> Harness:
        from gander_runtime import online_duplex

        provider = StubProvider(fail=warmup_fails, gate=warmup_gate)
        gateway = StubGateway(provider)
        thinkers: list[StubThinker] = []
        coordinators: list[StubCoordinator] = []

        def fake_build_session(_runtime, **_kwargs):
            if open_gate is not None:
                # Simulate a slow model open so a test can disconnect during it.
                open_gate.wait(5.0)
            thinker = StubThinker()
            thinkers.append(thinker)
            return thinker

        def fake_coordinator(gw, **kwargs):
            coordinator = StubCoordinator(gw, **kwargs)
            coordinators.append(coordinator)
            return coordinator

        monkeypatch.setattr(online_duplex, "_build_session", fake_build_session)
        monkeypatch.setattr(
            online_duplex, "TaskToolsRealtimeCoordinator", fake_coordinator
        )
        monkeypatch.setattr(
            online_duplex, "_prepare_static_prefix", lambda _runtime: None
        )

        app = online_duplex.create_online_duplex_app(
            StubBundle(),
            params=StubParams(),
            gateway_factory=lambda _session_id: gateway,
            provider_name="stub",
            settings=online_duplex.OnlineDuplexSettings(
                **{"reconnect_grace_sec": 0.3, **settings_kwargs}
            ),
            media_dir=tmp_path / "media",
        )
        return Harness(
            app=app,
            provider=provider,
            gateway=gateway,
            thinkers=thinkers,
            coordinators=coordinators,
            open_gate=open_gate,
        )

    return _build
