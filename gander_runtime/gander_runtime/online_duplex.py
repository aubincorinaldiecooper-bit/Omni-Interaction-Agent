from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import secrets
import time
import urllib.parse
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .contracts import MediaRef
from .duplex_bridge import GanderDuplexSession
from .media_mode import (
    CLIENT_VIDEO_MODES,
    VIDEO_SOURCES,
    client_video_allowed,
    estimated_tokens_per_unit,
    resolve_target_mode,
    source_warnings,
    vision_available,
)
from .media_timeline import (
    AUDIO_INPUT_PROTOCOL,
    AudioCaptureTimeline,
    AudioFrameHeader,
)
from .screen import LatestScreenFrameBuffer, ScreenFrameRateGate
from .screen_transport import (
    DecodedScreenFrame,
    ScreenFrameHeader,
    decode_screen_frame,
    persist_screen_frame,
)
from .task_tools_online import TaskToolsRealtimeCoordinator

LOGGER = logging.getLogger(__name__)
_SESSION_ID = re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9_.-]{1,128}$")
ONLINE_TASK_PROTOCOL = "task_tools_v1"
# Longest we wait for the talker to finish draining before cancelling it.
# Teardown holds the single model slot, so it must be bounded.
SPEECH_PUMP_DRAIN_TIMEOUT_SEC = 5.0
# How long a dropped duplex socket keeps its Thinker, its coordinator and
# the model slot, waiting for the same client to come back. The screen
# channel stays open for this long too, so the share is not interrupted.
RECONNECT_GRACE_SEC = 15.0
# A client is not supposed to send anything before `ready`. What it does send
# while the model opens is buffered, so the buffer is bounded: an unauthenticated
# connection already holds the single model slot, and must not also be able to
# grow the server's memory without limit.
STARTUP_BUFFER_MAX_MESSAGES = 32
STARTUP_BUFFER_MAX_BYTES = 4 * 1024 * 1024


class _StartupFlood(Exception):
    """A client sent more before `ready` than the startup buffer will hold."""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse


@dataclass(frozen=True)
class OnlineDuplexSettings:
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    system_prompt: str | None = None
    ref_audio_path: str | None = None
    trailing_silence_sec: float = 20.0
    send_listen_audio: bool = False
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    asr_base_url: str | None = None
    asr_timeout_sec: float = 120.0
    turn_bind_grace_sec: float = 5.0
    # How long a dropped duplex socket keeps its Thinker, its coordinator and
    # the model slot, waiting for the same client to come back.
    reconnect_grace_sec: float = RECONNECT_GRACE_SEC
    # Initial media mode for each session.
    media_mode: Literal["voice", "omni", "auto"] = "voice"
    # Allow clients to switch vision through `media.mode`.
    allow_client_video: bool = False
    client_video_mode: Literal["omni", "auto"] = "omni"
    client_video_sources: tuple[str, ...] = VIDEO_SOURCES
    vision_max_slice_nums: int = 1
    vision_batch_feed: bool = False
    max_screen_frame_bytes: int = 4 * 1024 * 1024
    max_screen_pixels: int = 4096 * 4096
    codex_frame_rate_multiplier: float = 3.0
    codex_screen_history_seconds: float = 8.0
    tool_schemas: tuple[dict[str, Any], ...] = ()
    expose_task_slate_to_model: bool = False
    warm_first_unit: bool = True


@dataclass
class _ActiveSession:
    duplex: GanderDuplexSession
    screen_token: str
    codex_frame_gate: ScreenFrameRateGate
    # Per-session media state, initialized from settings.
    media_mode: str = "voice"
    video_source: str | None = None
    coordinator: Any | None = None
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    # Reconnect grace. `attached` is False while the session is parked with no
    # socket; `resumed` fires when a client re-binds; `slot_released` makes the
    # model slot's release idempotent across the handler and the grace timer.
    resume_token: str = ""
    attached: bool = True
    stopped: bool = False
    slot_released: bool = False
    resumed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Runtime:
    bundle: Any
    params: Any
    settings: OnlineDuplexSettings
    media_dir: Path
    gateway_factory: Callable[[str], Any]
    provider_name: str
    detached_talker: Any | None = None
    prefix_snapshot: Any | None = None
    prefix_cache_status: str = "pending"
    prefix_prepare_seconds: float | None = None
    first_unit_warmup_seconds: float | None = None
    model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sessions: dict[str, _ActiveSession] = field(default_factory=dict)
    # Strong references to in-flight grace timers: a task nobody holds can be
    # garbage-collected while pending, skipping its teardown entirely.
    grace_tasks: set[asyncio.Task[None]] = field(default_factory=set)


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


async def _warm_backbrain_providers(gateway: Any) -> None:
    """Start each worker process before the realtime session is announced ready."""

    for provider in gateway.providers.values():
        await provider.warmup()


async def _screen_media_ref(
    runtime: _Runtime,
    session_id: str,
    header: ScreenFrameHeader,
    payload: bytes,
    decoded: DecodedScreenFrame,
    *,
    kind: Literal["screen", "frame"] = "screen",
) -> MediaRef:
    media = await asyncio.to_thread(
        persist_screen_frame,
        runtime.media_dir,
        session_id,
        header,
        payload,
        decoded,
        kind=kind,
    )
    return media


def _build_session(
    runtime: _Runtime,
    *,
    media_mode: str | None = None,
    screen_frames: LatestScreenFrameBuffer | None = None,
) -> GanderDuplexSession:
    from mcpmft.infer.realtime import DuplexLiveConfig, DuplexLiveSession
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
    from mcpmft.tool_protocol import ensure_lean_task_tools

    tools = ensure_lean_task_tools(runtime.settings.tool_schemas)

    live = DuplexLiveSession(
        runtime.bundle,
        params=runtime.params,
        system_prompt=runtime.settings.system_prompt or GANDER_DUPLEX_SYSTEM_PROMPT,
        ref_audio_path=runtime.settings.ref_audio_path,
        config=DuplexLiveConfig(
            trailing_silence_sec=runtime.settings.trailing_silence_sec,
            send_listen_audio=runtime.settings.send_listen_audio,
            media_mode=(
                media_mode if media_mode is not None else runtime.settings.media_mode
            ),
            max_slice_nums=runtime.settings.vision_max_slice_nums,
            batch_vision_feed=runtime.settings.vision_batch_feed,
        ),
        detached_talker=runtime.detached_talker,
        tools=tools,
        prefix_snapshot=runtime.prefix_snapshot,
    )
    if screen_frames is None:
        # Retain one frame per model unit across the full context window.
        screen_frames = LatestScreenFrameBuffer(
            max_pending_frames=runtime.params.context_max_units
        )
    return GanderDuplexSession(
        live,
        decode_mode=runtime.settings.decode_mode,
        screen_frames=screen_frames,
    )


def _prepare_static_prefix(runtime: _Runtime) -> None:
    """Prefill the process-static system/tool prefix once for all live sessions."""
    from mcpmft.infer.online import OnlineRunner
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
    from mcpmft.tool_protocol import ensure_lean_task_tools

    started = time.perf_counter()
    runner_params = runtime.params
    if runtime.detached_talker is not None:
        runner_params = replace(runner_params, generate_audio=False)
    runner = OnlineRunner(runtime.bundle, runner_params)
    runner.prepare(
        system_prompt=(
            runtime.settings.system_prompt or GANDER_DUPLEX_SYSTEM_PROMPT
        ),
        ref_audio_path=(
            None if runtime.detached_talker is not None else runtime.settings.ref_audio_path
        ),
        tools=ensure_lean_task_tools(runtime.settings.tool_schemas),
    )
    runtime.prefix_snapshot = runner.capture_prefix_snapshot()
    runtime.prefix_prepare_seconds = time.perf_counter() - started
    runtime.prefix_cache_status = "ready"
    LOGGER.info(
        "Prepared static Gander prefix cache: tokens=%d seconds=%.3f",
        runtime.prefix_snapshot.token_count,
        runtime.prefix_prepare_seconds,
    )

    if not runtime.settings.warm_first_unit:
        return
    if runtime.detached_talker is not None:
        started = time.perf_counter()
        runtime.detached_talker.warm_token2wav()
        runtime.first_unit_warmup_seconds = time.perf_counter() - started
        LOGGER.info(
            "Warmed detached Token2wav first chunk: seconds=%.3f",
            runtime.first_unit_warmup_seconds,
        )
        return
    started = time.perf_counter()
    warm_session = _build_session(runtime)
    try:
        event = warm_session.step_silence()
    finally:
        warm_session.close()
    runtime.first_unit_warmup_seconds = time.perf_counter() - started
    LOGGER.info(
        "Warmed first Gander duplex unit: decision=%s seconds=%.3f",
        "listen" if event.is_listen else "speak",
        runtime.first_unit_warmup_seconds,
    )


async def _open_session(
    runtime: _Runtime,
    *,
    media_mode: str | None = None,
    screen_frames: LatestScreenFrameBuffer | None = None,
) -> GanderDuplexSession:
    return await asyncio.to_thread(
        _build_session,
        runtime,
        media_mode=media_mode,
        screen_frames=screen_frames,
    )


def _tool_names(runtime: _Runtime) -> list[str]:
    names = [str(schema.get("name") or "") for schema in runtime.settings.tool_schemas]
    names = [name for name in names if name]
    required = ("task_start", "task_send", "task_resolve")
    return list(dict.fromkeys([*names, *required]))


def _codex_frame_interval_ms(runtime: _Runtime) -> float:
    return runtime.params.chunk_ms / runtime.settings.codex_frame_rate_multiplier


def _codex_frame_rate(runtime: _Runtime) -> float:
    return 1000.0 / _codex_frame_interval_ms(runtime)


def _max_recent_screen_frames(runtime: _Runtime) -> int:
    return max(
        1,
        math.ceil(
            _codex_frame_rate(runtime)
            * runtime.settings.codex_screen_history_seconds
        ),
    )


def _vision_available(runtime: _Runtime) -> bool:
    return vision_available(runtime.bundle.model)


def _client_video_allowed(runtime: _Runtime) -> bool:
    return client_video_allowed(
        vision_ok=_vision_available(runtime),
        media_mode=runtime.settings.media_mode,
        allow_client_video=runtime.settings.allow_client_video,
    )


def _unit_frame_rate(runtime: _Runtime) -> float:
    """Frames per second the client should send: one per model unit."""

    return 1000.0 / runtime.params.chunk_ms


def _client_video_capabilities(runtime: _Runtime) -> dict[str, Any]:
    return {
        "enabled": _client_video_allowed(runtime),
        "vision_available": _vision_available(runtime),
        "sources": list(runtime.settings.client_video_sources),
        "mode": runtime.settings.client_video_mode,
        "recommended_frame_rate": _unit_frame_rate(runtime),
    }


def _screen_channel(runtime: _Runtime, active: _ActiveSession) -> dict[str, Any]:
    """Everything a client needs to start uploading frames on ``/ws/screen``."""

    return {
        "path": "/ws/screen",
        "token": active.screen_token,
        "protocol": "metadata-json+encoded-binary-v1",
        "encodings": ["jpeg", "webp", "png"],
        "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
        "max_pixels": runtime.settings.max_screen_pixels,
        "recommended_frame_rate": _unit_frame_rate(runtime),
        "codex_frame_rate": _codex_frame_rate(runtime),
        "codex_screen_history_seconds": runtime.settings.codex_screen_history_seconds,
    }


def _reject_media_mode(
    runtime: _Runtime,
    *,
    want_video: bool,
    source: Any,
) -> str | None:
    """Return why a ``media.mode`` request is refused, or None to accept it."""

    if not want_video:
        return None
    if not _vision_available(runtime):
        return "the loaded model has no vision tower"
    if not _client_video_allowed(runtime):
        return "client video is disabled"
    if source not in runtime.settings.client_video_sources:
        return f"unsupported video source: {source!r}"
    return None


async def _send_chunk(
    websocket: WebSocket,
    event: Any,
    output_sample_rate: int,
) -> None:
    from mcpmft.infer.detached_talker import (
        PlaybackCancel,
        SpeechSynthesisChunk,
        SpeechSynthesisDone,
        SpeechSynthesisError,
    )
    from mcpmft.infer.realtime import waveform_to_pcm16_bytes

    audio = b""
    if isinstance(event, PlaybackCancel):
        payload = {
            "type": "playback.cancel",
            "generation_id": event.generation_id,
            "cancelled_generation_id": event.cancelled_generation_id,
            "reason": event.reason,
        }
    elif isinstance(event, SpeechSynthesisError):
        payload = {
            "type": "error",
            "message": f"Detached Talker failed: {event.message}",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
        }
    elif isinstance(event, SpeechSynthesisDone):
        payload = {
            "type": "audio.done",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
            "end_of_turn": event.end_of_turn,
            "metrics": event.metrics,
        }
    elif isinstance(event, SpeechSynthesisChunk):
        audio = waveform_to_pcm16_bytes(event.waveform)
        payload = {
            "type": "audio.chunk",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
            "sequence": event.sequence,
            "current_time": event.current_time,
            "end_of_turn": event.end_of_turn,
            "metrics": event.metrics,
        }
    else:
        audio = (
            waveform_to_pcm16_bytes(event.audio_waveform)
            if event.audio_waveform is not None
            else b""
        )
        payload = event.to_payload()
    payload.update(
        audio=bool(audio),
        audio_bytes=len(audio),
        audio_format="pcm16",
        audio_sample_rate=output_sample_rate,
    )
    await websocket.send_text(_json(payload))
    if audio:
        await websocket.send_bytes(audio)


class _WebSocketOutbox:
    """Serialize websocket writes without blocking model ingestion on audio I/O."""

    MODEL_PRIORITY = 0
    AUDIO_PRIORITY = 10

    def __init__(self, websocket: WebSocket, output_sample_rate: int) -> None:
        self.websocket = websocket
        self.output_sample_rate = output_sample_rate
        self.queue: asyncio.PriorityQueue[
            tuple[int, int, float, str, Any, asyncio.Future[None] | None]
        ] = asyncio.PriorityQueue()
        self.sequence = 0
        self.audio_generation_floor = 0
        self.task = asyncio.create_task(
            self._run(), name="gander-duplex-websocket-writer"
        )

    def _raise_if_failed(self) -> None:
        if not self.task.done():
            return
        if self.task.cancelled():
            raise WebSocketDisconnect(code=1006)
        error = self.task.exception()
        if error is not None:
            raise error
        raise WebSocketDisconnect(code=1006)

    async def send_text(
        self,
        payload: dict[str, Any],
        *,
        wait_sent: bool = False,
    ) -> None:
        self._raise_if_failed()
        self.sequence += 1
        sent = asyncio.get_running_loop().create_future() if wait_sent else None
        self.queue.put_nowait(
            (
                self.MODEL_PRIORITY,
                self.sequence,
                time.perf_counter(),
                f"text:{payload.get('type', 'unknown')}",
                payload,
                sent,
            )
        )
        # Yield once so the writer can send latency-sensitive control events first.
        await asyncio.sleep(0)
        if sent is not None:
            await sent

    async def send_event(
        self,
        event: Any,
        *,
        audio: bool = False,
        wait_sent: bool = False,
    ) -> None:
        self._raise_if_failed()
        if type(event).__name__ == "PlaybackCancel":
            self.audio_generation_floor = max(
                self.audio_generation_floor,
                int(event.generation_id),
            )
        self.sequence += 1
        sent = asyncio.get_running_loop().create_future() if wait_sent else None
        self.queue.put_nowait(
            (
                self.AUDIO_PRIORITY if audio else self.MODEL_PRIORITY,
                self.sequence,
                time.perf_counter(),
                f"event:{type(event).__name__}",
                event,
                sent,
            )
        )
        await asyncio.sleep(0)
        if sent is not None:
            await sent

    async def join(self) -> None:
        self._raise_if_failed()
        joined = asyncio.create_task(self.queue.join())
        done, _ = await asyncio.wait(
            (joined, self.task), return_when=asyncio.FIRST_COMPLETED
        )
        if self.task in done and not joined.done():
            joined.cancel()
            await asyncio.gather(joined, return_exceptions=True)
            self._raise_if_failed()
        await joined
        self._raise_if_failed()

    def discard_pending(self) -> None:
        """Drop output queued before a conversation reset."""

        while True:
            try:
                _priority, _sequence, _queued_at, _label, _payload, sent = (
                    self.queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                return
            if sent is not None and not sent.done():
                sent.cancel()
            self.queue.task_done()

    async def close(self) -> None:
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while True:
            try:
                _priority, _sequence, _queued_at, _label, _payload, sent = (
                    self.queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                break
            if sent is not None and not sent.done():
                sent.cancel()
            self.queue.task_done()

    def _stale_audio(self, event: Any) -> bool:
        return (
            type(event).__name__
            in {
                "SpeechSynthesisChunk",
                "SpeechSynthesisDone",
                "SpeechSynthesisError",
            }
            and int(event.generation_id) < self.audio_generation_floor
        )

    async def _run(self) -> None:
        while True:
            _priority, _sequence, queued_at, label, payload, sent = (
                await self.queue.get()
            )
            started = time.perf_counter()
            try:
                if self._stale_audio(payload):
                    LOGGER.debug(
                        "discarding stale %s generation=%s floor=%s",
                        type(payload).__name__,
                        payload.generation_id,
                        self.audio_generation_floor,
                    )
                elif label.startswith("text:"):
                    await self.websocket.send_text(_json(payload))
                else:
                    await _send_chunk(
                        self.websocket, payload, self.output_sample_rate
                    )
                if sent is not None and not sent.done():
                    sent.set_result(None)
            except RuntimeError as exc:
                error = WebSocketDisconnect(code=1006)
                if sent is not None and not sent.done():
                    sent.set_exception(error)
                raise error from exc
            except asyncio.CancelledError:
                if sent is not None and not sent.done():
                    sent.cancel()
                raise
            except Exception as exc:
                if sent is not None and not sent.done():
                    sent.set_exception(exc)
                raise
            finally:
                self.queue.task_done()
            finished = time.perf_counter()
            queue_ms = (started - queued_at) * 1000
            send_ms = (finished - started) * 1000
            if queue_ms >= 100 or send_ms >= 100:
                LOGGER.info(
                    "websocket outbound %s queue_ms=%.1f send_ms=%.1f pending=%d",
                    label,
                    queue_ms,
                    send_ms,
                    self.queue.qsize(),
                )


async def _drain(
    session: GanderDuplexSession,
    emit_model_event: Callable[[Any], Awaitable[None]],
    *,
    pending_unit_capture_start_ms: float | None = None,
) -> None:
    for event in await asyncio.to_thread(
        session.flush_pending,
        unit_capture_start_ms=pending_unit_capture_start_ms,
    ):
        await emit_model_event(event)

    trailing_steps = 0
    while session.should_continue_draining(trailing_steps):
        event = await asyncio.to_thread(session.step_silence)
        await emit_model_event(event)
        trailing_steps += 1
        if session.should_stop_after(event):
            break
    await asyncio.to_thread(session.wait_for_speech)
    await asyncio.to_thread(session.close, drain_speech=True)


def create_online_duplex_app(
    bundle: Any,
    *,
    params: Any,
    gateway_factory: Callable[[str], Any],
    provider_name: str,
    settings: OnlineDuplexSettings | None = None,
    media_dir: str | Path,
    detached_talker: Any | None = None,
):
    """Create the persistent MiniCPM, task-tools and screen endpoints."""

    from mcpmft.infer.web import STATIC_DIR, request_json

    runtime = _Runtime(
        bundle=bundle,
        params=params,
        settings=settings or OnlineDuplexSettings(),
        gateway_factory=gateway_factory,
        provider_name=provider_name,
        media_dir=Path(media_dir).expanduser().resolve(),
        detached_talker=detached_talker,
    )
    if runtime.params.sliding_window_mode not in {
        "context_memory",
        "context_no_previous",
        "context_slate",
    }:
        raise ValueError(
            "task_tools_v1 requires a bounded live context mode "
            "('context_memory', 'context_slate', or 'context_no_previous')"
        )
    if (
        runtime.params.sliding_window_mode == "context_slate"
        and not runtime.settings.expose_task_slate_to_model
    ):
        raise ValueError("context_slate requires task-slate exposure")
    if (
        runtime.params.sliding_window_mode == "context_no_previous"
        and runtime.settings.expose_task_slate_to_model
    ):
        raise ValueError("context_no_previous cannot expose a task slate")
    if runtime.settings.media_mode not in {"voice", "omni", "auto"}:
        raise ValueError(f"unsupported media_mode: {runtime.settings.media_mode!r}")
    if runtime.settings.client_video_mode not in CLIENT_VIDEO_MODES:
        raise ValueError(
            f"unsupported client_video_mode: {runtime.settings.client_video_mode!r}"
        )
    if not runtime.settings.client_video_sources or not set(
        runtime.settings.client_video_sources
    ) <= set(VIDEO_SOURCES):
        raise ValueError(
            f"client_video_sources must be a non-empty subset of {VIDEO_SOURCES}"
        )
    # Client video requires an initialized vision tower.
    if (
        runtime.settings.media_mode != "voice" or runtime.settings.allow_client_video
    ) and not _vision_available(runtime):
        raise ValueError(
            "video input requires model.init_vision=true in the serving configuration"
        )
    if runtime.settings.vision_max_slice_nums <= 0:
        raise ValueError("vision_max_slice_nums must be positive")
    if runtime.settings.max_screen_frame_bytes <= 0:
        raise ValueError("max_screen_frame_bytes must be positive")
    if runtime.settings.max_screen_pixels <= 0:
        raise ValueError("max_screen_pixels must be positive")
    if not 1 <= runtime.settings.codex_frame_rate_multiplier <= 10:
        raise ValueError("codex_frame_rate_multiplier must be between 1 and 10")
    if runtime.settings.codex_screen_history_seconds <= 0:
        raise ValueError("codex_screen_history_seconds must be positive")
    if runtime.params.chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if runtime.settings.asr_timeout_sec <= 0:
        raise ValueError("asr_timeout_sec must be positive")
    if runtime.settings.turn_bind_grace_sec < 0:
        raise ValueError("turn_bind_grace_sec cannot be negative")
    app = FastAPI(title="Gander Online Duplex", version="1.0.0")

    async def prepare_static_prefix() -> None:
        await asyncio.to_thread(_prepare_static_prefix, runtime)

    app.router.add_event_handler("startup", prepare_static_prefix)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/app.css")
    async def app_css() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/video.js")
    async def video_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "video.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/mic-worklet.js")
    async def mic_worklet_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "mic-worklet.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "busy": runtime.model_lock.locked(),
            "active_sessions": list(runtime.sessions),
            # Sessions whose socket dropped and that are inside their
            # reconnect window: the same client can still re-bind to them.
            "resumable_sessions": [
                name
                for name, active in runtime.sessions.items()
                if not active.attached and not active.stopped
            ],
            "session_media_modes": {
                name: active.media_mode
                for name, active in runtime.sessions.items()
            },
            "vision_available": _vision_available(runtime),
            "client_video": _client_video_capabilities(runtime),
            "task_protocol": ONLINE_TASK_PROTOCOL,
            "audio_input_protocol": AUDIO_INPUT_PROTOCOL,
            "input_sample_rate": runtime.settings.input_sample_rate,
            "output_sample_rate": runtime.settings.output_sample_rate,
            "generate_audio": bool(
                runtime.params.generate_audio or runtime.detached_talker is not None
            ),
            "detached_talker": runtime.detached_talker is not None,
            "chunk_ms": runtime.params.chunk_ms,
            "sliding_window_mode": runtime.params.sliding_window_mode,
            "memory_episode_channel": (
                "enabled"
                if runtime.params.sliding_window_mode == "context_memory"
                else "disabled"
            ),
            "context_max_units": runtime.params.context_max_units,
            "context_previous_max_tokens": runtime.params.context_previous_max_tokens,
            "prefix_cache_status": runtime.prefix_cache_status,
            "prefix_cache_tokens": (
                runtime.prefix_snapshot.token_count
                if runtime.prefix_snapshot is not None
                else 0
            ),
            "prefix_prepare_seconds": runtime.prefix_prepare_seconds,
            "first_unit_warmup_seconds": runtime.first_unit_warmup_seconds,
            "task_slate_visible_to_model": (
                runtime.settings.expose_task_slate_to_model
            ),
            "asr_enabled": bool(runtime.settings.asr_base_url),
            "tools": _tool_names(runtime),
        }

    @app.get("/api/asr/health")
    async def asr_health() -> JSONResponse:
        if not runtime.settings.asr_base_url:
            return JSONResponse(
                {"status": "disabled", "message": "ASR upstream is not configured"},
                status_code=503,
            )
        try:
            payload = await asyncio.to_thread(
                request_json,
                f"{runtime.settings.asr_base_url.rstrip('/')}/health",
                timeout=min(runtime.settings.asr_timeout_sec, 2.0),
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"status": "unavailable", "message": str(exc)},
                status_code=502,
            )
        return JSONResponse(payload)

    @app.post("/api/asr/transcribe")
    async def asr_transcribe(request: Request) -> JSONResponse:
        if not runtime.settings.asr_base_url:
            return JSONResponse(
                {"type": "error", "message": "ASR upstream is not configured"},
                status_code=503,
            )
        pcm = await request.body()
        if not pcm or len(pcm) % 2:
            return JSONResponse(
                {"type": "error", "message": "ASR input must be non-empty PCM16"},
                status_code=400,
            )
        max_bytes = runtime.settings.input_sample_rate * 2 * 120
        if len(pcm) > max_bytes:
            return JSONResponse(
                {"type": "error", "message": "ASR input exceeds 120 seconds"},
                status_code=413,
            )
        query = urllib.parse.urlencode(
            {
                "sample_rate": runtime.settings.input_sample_rate,
                "start_ms": request.query_params.get("start_ms", "0"),
                "language": request.query_params.get("language", ""),
            }
        )
        url = f"{runtime.settings.asr_base_url.rstrip('/')}/transcribe?{query}"
        try:
            payload = await asyncio.to_thread(
                request_json,
                url,
                data=pcm,
                headers={"Content-Type": "application/octet-stream"},
                timeout=runtime.settings.asr_timeout_sec,
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"type": "error", "message": str(exc)},
                status_code=502,
            )
        return JSONResponse(payload)

    @app.websocket("/ws/screen")
    async def screen(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = websocket.query_params.get("session_id") or ""
        token = websocket.query_params.get("token") or ""
        active = runtime.sessions.get(session_id)
        if not _SESSION_ID.fullmatch(session_id) or active is None:
            await websocket.send_text(
                _json({"type": "error", "message": "duplex session is not active"})
            )
            await websocket.close(code=1008)
            return
        if active.media_mode == "voice":
            await websocket.send_text(
                _json({"type": "error", "message": "screen input is disabled in voice mode"})
            )
            await websocket.close(code=1008)
            return
        if not token or not secrets.compare_digest(token, active.screen_token):
            await websocket.send_text(
                _json({"type": "error", "message": "invalid screen session token"})
            )
            await websocket.close(code=1008)
            return

        await websocket.send_text(
            _json(
                {
                    "type": "screen.ready",
                    "session_id": session_id,
                    "media_mode": active.media_mode,
                    "video_source": active.video_source,
                    "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
                    "max_pixels": runtime.settings.max_screen_pixels,
                    "recommended_frame_rate": _unit_frame_rate(runtime),
                    "codex_frame_rate": _codex_frame_rate(runtime),
                    "codex_screen_history_seconds": (
                        runtime.settings.codex_screen_history_seconds
                    ),
                }
            )
        )

        async def receive_while_session_active() -> dict[str, Any] | None:
            receive_task = asyncio.create_task(websocket.receive())
            ended_task = asyncio.create_task(active.ended.wait())
            done, pending = await asyncio.wait(
                (receive_task, ended_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if ended_task in done:
                if not receive_task.done():
                    receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                try:
                    await websocket.close()
                except RuntimeError:
                    pass
                return None
            return receive_task.result()

        try:
            while runtime.sessions.get(session_id) is active:
                metadata_message = await receive_while_session_active()
                if metadata_message is None:
                    return
                if metadata_message.get("type") == "websocket.disconnect":
                    return
                frame_id: Any = None
                try:
                    raw_metadata = metadata_message.get("text")
                    if raw_metadata is None:
                        raise ValueError("screen frame metadata must be JSON text")
                    payload = json.loads(raw_metadata)
                    if not isinstance(payload, dict):
                        raise ValueError("screen frame metadata must be an object")
                    header = ScreenFrameHeader.from_payload(payload)
                    frame_id = header.frame_id

                    image_message = await receive_while_session_active()
                    if image_message is None:
                        return
                    if image_message.get("type") == "websocket.disconnect":
                        return
                    image_payload = image_message.get("bytes")
                    if image_payload is None:
                        raise ValueError("screen frame image must be binary")
                    decoded = await asyncio.to_thread(
                        decode_screen_frame,
                        header,
                        image_payload,
                        max_bytes=runtime.settings.max_screen_frame_bytes,
                        max_pixels=runtime.settings.max_screen_pixels,
                    )
                    if runtime.sessions.get(session_id) is not active:
                        return
                    if active.media_mode == "voice":
                        # Ignore a frame completed after the session returned to audio mode.
                        await websocket.send_text(
                            _json(
                                {
                                    "type": "screen.frame.dropped",
                                    "frame_id": header.frame_id,
                                    "reason": "media_mode is voice",
                                }
                            )
                        )
                        continue

                    context_sampled = active.codex_frame_gate.accept(
                        header.captured_at_ms
                    )
                    # Publish to the front brain before ACK; persist off the event loop.
                    active.duplex.enqueue_screen_frame(decoded.frame)
                    await websocket.send_text(
                        _json(
                            {
                                "type": "screen.frame.accepted",
                                "frame_id": header.frame_id,
                                "captured_at_ms": header.captured_at_ms,
                                "width": decoded.width,
                                "height": decoded.height,
                                "asset_id": decoded.asset_id,
                                "context_sampled": context_sampled,
                            }
                        )
                    )
                    if context_sampled:
                        # Tag webcam and screen frames separately for back-brain context.
                        source = header.video_source or active.video_source
                        media = await _screen_media_ref(
                            runtime,
                            session_id,
                            header,
                            image_payload,
                            decoded,
                            kind="frame" if source == "camera" else "screen",
                        )
                        if runtime.sessions.get(session_id) is not active:
                            return
                        coordinator = active.coordinator
                        if coordinator is not None:
                            coordinator.remember_media(media)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    # One malformed or undecodable frame is not a reason to end
                    # the screen channel. The capture is still good and the next
                    # frame usually is too, so report the drop and keep going.
                    LOGGER.warning(
                        "dropping screen frame for %s: %s", session_id, exc
                    )
                    await websocket.send_text(
                        _json(
                            {
                                "type": "screen.frame.dropped",
                                "frame_id": frame_id,
                                "reason": str(exc),
                            }
                        )
                    )
                    continue
        except WebSocketDisconnect:
            return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                await websocket.send_text(
                    _json({"type": "error", "message": str(exc)})
                )
                await websocket.close(code=1003)
            except RuntimeError:
                pass

    @app.websocket("/ws/duplex")
    async def duplex(websocket: WebSocket) -> None:
        await websocket.accept()
        requested_id = websocket.query_params.get("session_id")
        session_id = requested_id or f"duplex_{secrets.token_hex(8)}"
        if not _SESSION_ID.fullmatch(session_id):
            await websocket.send_text(
                _json(
                    {
                        "type": "error",
                        "message": "invalid session_id",
                        "fatal": True,
                    }
                )
            )
            await websocket.close(code=1008)
            return
        # A session whose socket dropped is held briefly. The same client,
        # with the token it was given, re-binds to its own Thinker instead of
        # queuing behind it for the model slot it is already holding.
        resume_token = websocket.query_params.get("resume_token") or ""
        parked = runtime.sessions.get(session_id)
        resuming = bool(
            parked is not None
            and not parked.attached
            and not parked.stopped
            and resume_token
            and parked.resume_token
            and secrets.compare_digest(resume_token, parked.resume_token)
        )
        if not resuming:
            if runtime.model_lock.locked():
                await websocket.send_text(
                    _json(
                        {
                            "type": "error",
                            "message": "model is busy with another session",
                            "fatal": False,
                            "retry": True,
                        }
                    )
                )
                await websocket.close(code=1013)
                return
            await runtime.model_lock.acquire()

        park_for_resume = False
        slot_state = {"released": False}
        session: GanderDuplexSession | None = None
        coordinator: Any | None = None
        active: _ActiveSession | None = None
        open_task: asyncio.Task[GanderDuplexSession] | None = None
        outbound_task: asyncio.Task[None] | None = None
        warmup_task: asyncio.Task[None] | None = None
        receive_watcher: asyncio.Task[dict[str, Any]] | None = None
        pending_messages: deque[dict[str, Any]] = deque()
        pending_bytes = 0
        speech_output_task: asyncio.Task[None] | None = None
        speech_output_stop: asyncio.Event | None = None
        pending_audio_header: AudioFrameHeader | None = None
        audio_timeline = AudioCaptureTimeline(
            sample_rate=runtime.settings.input_sample_rate,
            unit_ms=runtime.params.chunk_ms,
        )
        websocket_outbox = _WebSocketOutbox(
            websocket, runtime.settings.output_sample_rate
        )

        async def receive_message() -> dict[str, Any]:
            """Next client message, oldest first.

            Anything the client sent while the session was still starting was
            buffered by the startup watcher, and is delivered before the
            socket is read again.
            """

            if pending_messages:
                return pending_messages.popleft()
            return await websocket.receive()

        async def send_text(
            payload: dict[str, Any],
            *,
            wait_sent: bool = False,
        ) -> None:
            if payload.get("type") == "error" and "fatal" not in payload:
                # Control-parsing and validation errors leave the Thinker
                # usable, so they must not end the client's session. Only a
                # caller that knows otherwise sets fatal=True.
                payload = {**payload, "fatal": False}
            await websocket_outbox.send_text(payload, wait_sent=wait_sent)

        async def send_model_event(
            event: Any,
            *,
            wait_sent: bool = False,
        ) -> None:
            await websocket_outbox.send_event(event, wait_sent=wait_sent)

        def start_speech_output_pump(
            target: GanderDuplexSession,
        ) -> tuple[asyncio.Event, asyncio.Task[None]]:
            stop = asyncio.Event()

            async def pump() -> None:
                while True:
                    event = await asyncio.to_thread(target.poll_output, 0.05)
                    if event is not None:
                        await websocket_outbox.send_event(
                            event,
                            audio=type(event).__name__
                            in {"SpeechSynthesisChunk", "SpeechSynthesisDone"},
                        )
                        if not stop.is_set():
                            continue
                        # Stop arrived mid-stream. Fall through to drain what
                        # is already buffered rather than following the model
                        # for as long as it keeps producing.
                    if stop.is_set():
                        for remaining in target.drain_outputs():
                            await websocket_outbox.send_event(
                                remaining,
                                audio=(
                                    type(remaining).__name__
                                    in {
                                        "SpeechSynthesisChunk",
                                        "SpeechSynthesisDone",
                                    }
                                ),
                            )
                        return

            return stop, asyncio.create_task(
                pump(), name=f"gander-detached-talker-output-{session_id}"
            )

        async def stop_speech_output_pump() -> None:
            nonlocal speech_output_task, speech_output_stop
            if speech_output_stop is not None:
                speech_output_stop.set()
            task = speech_output_task
            if task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), SPEECH_PUMP_DRAIN_TIMEOUT_SEC
                    )
                except asyncio.TimeoutError:
                    LOGGER.warning(
                        "speech output pump for %s did not drain in %.1fs; "
                        "cancelling",
                        session_id,
                        SPEECH_PUMP_DRAIN_TIMEOUT_SEC,
                    )
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                except Exception:
                    # The pump re-raises whatever the websocket writer stored,
                    # which is any exception type at all. Teardown must not
                    # depend on which one it is.
                    LOGGER.debug(
                        "speech output pump for %s ended in error",
                        session_id,
                        exc_info=True,
                    )
            speech_output_task = None
            speech_output_stop = None

        async def emit_model_event(event: Any) -> None:
            assert coordinator is not None
            output = coordinator.model_output(event)
            await send_model_event(
                event,
                wait_sent=output.delivery_id is not None,
            )
            if output.delivery_id is not None:
                coordinator.acknowledge_output(output)
            coordinator.observe_frontbrain(event)

        async def build_coordinator(
            model_session: GanderDuplexSession,
        ) -> TaskToolsRealtimeCoordinator:
            gateway = runtime.gateway_factory(session_id)
            task_coordinator = TaskToolsRealtimeCoordinator(
                gateway,
                owner_id=session_id,
                session_id=session_id,
                session=model_session,
                provider_name=runtime.provider_name,
                media_dir=runtime.media_dir,
                input_sample_rate=runtime.settings.input_sample_rate,
                max_recent_media=_max_recent_screen_frames(runtime),
                turn_bind_grace_sec=runtime.settings.turn_bind_grace_sec,
                expose_task_slate_to_model=(
                    runtime.settings.expose_task_slate_to_model
                ),
                close_ledger=True,
            )
            await task_coordinator.start()
            # The Brain is warmed in the background once the client is ready;
            # seeing and hearing must not wait on it.
            return task_coordinator

        def release_model_slot() -> None:
            """Free the single Thinker slot, exactly once.

            The guard lives on the session when there is one, so the handler,
            the grace timer and the abandoned-open cleanup cannot each release
            it separately.
            """

            if active is not None:
                if active.slot_released:
                    return
                active.slot_released = True
            else:
                if slot_state["released"]:
                    return
                slot_state["released"] = True
            runtime.model_lock.release()
            LOGGER.info("released model slot for %s", session_id)

        async def close_abandoned_open(task: "asyncio.Task[Any]") -> None:
            """Close a model session whose client left while it was opening.

            The open runs on a worker thread that cancellation cannot reach, so
            the session is still being built after the handler has gone. The
            slot stays held until it exists and has been closed: releasing
            early would let a second session build against the same model and
            leave this one open forever.
            """

            try:
                orphan = await task
            except asyncio.CancelledError:
                orphan = None
            except Exception:
                orphan = None
                LOGGER.warning(
                    "abandoned model open failed for %s", session_id, exc_info=True
                )
            try:
                if orphan is not None and not orphan.closed:
                    await asyncio.to_thread(orphan.close)
                    LOGGER.info("closed abandoned model open for %s", session_id)
            except Exception:
                LOGGER.warning(
                    "closing the abandoned model open for %s failed",
                    session_id,
                    exc_info=True,
                )
            finally:
                release_model_slot()

        async def expire_reconnect_grace(
            held: _ActiveSession,
            detach: Callable[[], Awaitable[None]],
        ) -> None:
            """Hold a parked session, then tear it down if nobody returns.

            The detach runs here rather than in the handler: a handler that
            was cancelled cannot await anything, and the session must still be
            parked and cleaned up.
            """

            try:
                await detach()
            except Exception:
                LOGGER.warning(
                    "detaching %s for resume failed", session_id, exc_info=True
                )
            try:
                await asyncio.wait_for(
                    held.resumed.wait(), runtime.settings.reconnect_grace_sec
                )
                return
            except asyncio.TimeoutError:
                pass
            if held.attached:
                # A client re-bound between the timeout and this check.
                return
            LOGGER.info("reconnect grace expired for %s", session_id)
            try:
                if runtime.sessions.get(session_id) is held:
                    runtime.sessions.pop(session_id, None)
                # Only now does /ws/screen learn the session is over.
                held.ended.set()
                if held.coordinator is not None:
                    await held.coordinator.stop_model_jobs()
            except Exception:
                LOGGER.warning(
                    "grace teardown failed for %s", session_id, exc_info=True
                )
            finally:
                try:
                    if not held.duplex.closed:
                        held.duplex.close()
                finally:
                    if not held.slot_released:
                        held.slot_released = True
                        runtime.model_lock.release()
                        LOGGER.info("released model slot for %s", session_id)
            if held.coordinator is not None:
                try:
                    await held.coordinator.close()
                except Exception:
                    LOGGER.warning(
                        "worker gateway teardown failed for %s",
                        session_id,
                        exc_info=True,
                    )

        try:
            async def run_startup() -> None:
                """Open the Thinker and announce readiness."""

                nonlocal session, active, coordinator, open_task
                nonlocal outbound_task, warmup_task
                nonlocal speech_output_stop, speech_output_task

                if resuming and parked is not None:
                    # Same Thinker, same coordinator, same model slot. Only
                    # the connection is new, so nothing is reloaded and the
                    # conversation carries on where it left off.
                    active = parked
                    session = active.duplex
                    coordinator = active.coordinator
                    active.attached = True
                    active.resumed.set()
                    if runtime.detached_talker is not None:
                        speech_output_stop, speech_output_task = (
                            start_speech_output_pump(session)
                        )
                    LOGGER.info("duplex session resumed: %s", session_id)
                else:
                    open_task = asyncio.create_task(
                        _open_session(runtime),
                        name=f"gander-model-open-{session_id}",
                    )
                    session = await asyncio.shield(open_task)
                    if runtime.detached_talker is not None:
                        speech_output_stop, speech_output_task = (
                            start_speech_output_pump(session)
                        )
                    active = _ActiveSession(
                        duplex=session,
                        screen_token=secrets.token_urlsafe(32),
                        codex_frame_gate=ScreenFrameRateGate(
                            _codex_frame_interval_ms(runtime)
                        ),
                        media_mode=runtime.settings.media_mode,
                        resume_token=secrets.token_urlsafe(32),
                    )
                    runtime.sessions[session_id] = active
                    coordinator = await build_coordinator(session)
                    active.coordinator = coordinator

                async def forward_tool_outputs() -> None:
                    assert coordinator is not None
                    while True:
                        output = await coordinator.next_output()
                        if output.kind == "control":
                            await send_text(
                                output.value,
                                wait_sent=output.delivery_id is not None,
                            )
                        else:
                            await send_model_event(
                                output.value,
                                wait_sent=output.delivery_id is not None,
                            )
                        if output.delivery_id is not None:
                            coordinator.acknowledge_output(output)

                outbound_task = asyncio.create_task(
                    forward_tool_outputs(),
                    name=f"gander-native-tool-output-{session_id}",
                )
                outbound_task.add_done_callback(
                    lambda task: (
                        None
                        if task.cancelled()
                        else LOGGER.error(
                            "native tool output loop failed for %s: %s",
                            session_id,
                            task.exception(),
                        )
                        if task.exception() is not None
                        else None
                    )
                )
                screen_enabled = (
                    active.media_mode != "voice" or _client_video_allowed(runtime)
                )
                await send_text(
                    {
                        "type": "ready",
                        "session_id": session_id,
                        "resume_token": active.resume_token,
                        "resumed": resuming,
                        "reconnect_grace_ms": round(
                            runtime.settings.reconnect_grace_sec * 1000
                        ),
                        "input_sample_rate": runtime.settings.input_sample_rate,
                        "output_sample_rate": runtime.settings.output_sample_rate,
                        "chunk_ms": runtime.params.chunk_ms,
                        "audio_input": {
                            "protocol": AUDIO_INPUT_PROTOCOL,
                            "encoding": "pcm_s16le",
                            "clock": "unix_ms",
                        },
                        "generate_audio": bool(
                            runtime.params.generate_audio
                            or runtime.detached_talker is not None
                        ),
                        "detached_talker": runtime.detached_talker is not None,
                        "generation_id": int(
                            session.talker_state().get("generation_id", 0)
                        ),
                        "sliding_window_mode": runtime.params.sliding_window_mode,
                        "context_max_units": runtime.params.context_max_units,
                        "context_previous_max_tokens": (
                            runtime.params.context_previous_max_tokens
                        ),
                        "transport": "ws",
                        "decode_mode": runtime.settings.decode_mode,
                        "media_mode": active.media_mode,
                        "video_source": active.video_source,
                        "tool_protocol": "native_complete_call_v1",
                        "task_protocol": ONLINE_TASK_PROTOCOL,
                        "turn_bind_grace_ms": round(
                            runtime.settings.turn_bind_grace_sec * 1000
                        ),
                        "tools": _tool_names(runtime),
                        "context_events": [
                            "turn.final",
                            *(
                                ["memory.episode"]
                                if runtime.params.sliding_window_mode == "context_memory"
                                else []
                            ),
                            "task_status",
                            "screen",
                        ],
                        "screen": {
                            "enabled": screen_enabled,
                            "path": "/ws/screen" if screen_enabled else None,
                            "token": active.screen_token if screen_enabled else None,
                            "protocol": "metadata-json+encoded-binary-v1",
                            "encodings": ["jpeg", "webp", "png"],
                            "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
                            "max_pixels": runtime.settings.max_screen_pixels,
                            "vision_max_slice_nums": runtime.settings.vision_max_slice_nums,
                            "vision_batch_feed": runtime.settings.vision_batch_feed,
                            "recommended_frame_rate": _unit_frame_rate(runtime),
                            "codex_frame_rate": _codex_frame_rate(runtime),
                            "codex_screen_history_seconds": (
                                runtime.settings.codex_screen_history_seconds
                            ),
                            "client_video": _client_video_capabilities(runtime),
                        },
                    }
                )

                async def warm_back_brain() -> None:
                    """Warm the Brain behind the session that is already running.

                    Gander is perceptually ready before this finishes. A delegated
                    task that arrives first is not dropped: the provider's worker
                    start is guarded by its own lock, so the task waits for the
                    same warm-up rather than starting a second one.
                    """

                    try:
                        await send_text({"type": "brain.status", "status": "warming"})
                        await _warm_backbrain_providers(coordinator.gateway)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # An unreachable or cold Brain costs delegated work, not
                        # the session: Gander can still see, hear and answer.
                        LOGGER.warning(
                            "back brain warmup failed for %s: %s", session_id, exc
                        )
                        try:
                            await send_text(
                                {
                                    "type": "brain.status",
                                    "status": "error",
                                    "message": str(exc),
                                }
                            )
                        except Exception:
                            LOGGER.debug(
                                "could not report brain warmup failure for %s",
                                session_id,
                                exc_info=True,
                            )
                    else:
                        try:
                            await send_text(
                                {"type": "brain.status", "status": "ready"}
                            )
                        except Exception:
                            LOGGER.debug(
                                "could not report brain readiness for %s",
                                session_id,
                                exc_info=True,
                            )

                if not resuming:
                    # A resumed session already has a warm Brain.
                    warmup_task = asyncio.create_task(
                        warm_back_brain(),
                        name=f"gander-brain-warmup-{session_id}",
                    )

            # Watch the socket while the session starts. A client that
            # gives up during the model open must free the Thinker slot
            # now, not whenever startup happens to finish.
            receive_watcher = asyncio.create_task(
                websocket.receive(),
                name=f"gander-duplex-receive-{session_id}",
            )
            startup = asyncio.create_task(
                run_startup(), name=f"gander-duplex-startup-{session_id}"
            )
            try:
                while not startup.done():
                    await asyncio.wait(
                        {startup, receive_watcher},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not receive_watcher.done():
                        continue
                    early = receive_watcher.result()
                    if early.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect(code=1006)
                    # Anything the client sent while we were starting is
                    # kept in order for the main loop below, up to a bound.
                    payload = early.get("text") or early.get("bytes") or b""
                    pending_bytes += len(payload)
                    if (
                        len(pending_messages) >= STARTUP_BUFFER_MAX_MESSAGES
                        or pending_bytes > STARTUP_BUFFER_MAX_BYTES
                    ):
                        raise _StartupFlood(
                            "too much data sent before the session was ready"
                        )
                    pending_messages.append(early)
                    receive_watcher = asyncio.create_task(
                        websocket.receive(),
                        name=f"gander-duplex-receive-{session_id}",
                    )
            finally:
                if not startup.done():
                    startup.cancel()
                    await asyncio.gather(startup, return_exceptions=True)
                if receive_watcher is not None:
                    # Hand the socket back to the main loop. Cancelling a
                    # receive that has not produced anything consumes nothing,
                    # and a message that landed in the meantime is kept.
                    receive_watcher.cancel()
                    await asyncio.gather(receive_watcher, return_exceptions=True)
                    if not receive_watcher.cancelled():
                        if receive_watcher.exception() is None:
                            pending_messages.append(receive_watcher.result())
                    receive_watcher = None
            # Surface anything startup itself raised.
            await startup


            while True:
                message = await receive_message()
                if message.get("type") == "websocket.disconnect":
                    park_for_resume = True
                    return
                if message.get("bytes") is not None:
                    audio = message["bytes"]
                    await asyncio.to_thread(coordinator.record_pcm16, audio)
                    # Process one unit per model call so provider events run between units.
                    chunk_bytes = int(
                        runtime.settings.input_sample_rate * runtime.params.chunk_ms / 1000 * 2
                    )
                    if pending_audio_header is None:
                        parts = tuple(
                            (
                                audio[offset : offset + chunk_bytes],
                                None,
                            )
                            for offset in range(0, len(audio), chunk_bytes)
                        )
                    else:
                        timed_parts = audio_timeline.split_frame(
                            pending_audio_header,
                            audio,
                            max_part_bytes=chunk_bytes,
                        )
                        pending_audio_header = None
                        parts = tuple(
                            (part.data, part.unit_capture_start_ms)
                            for part in timed_parts
                        )
                    for part, capture_starts in parts:
                        events = await asyncio.to_thread(
                            session.feed_pcm16,
                            part,
                            unit_capture_start_ms=capture_starts,
                        )
                        for event in events:
                            await emit_model_event(event)
                    continue

                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    control = json.loads(raw)
                except json.JSONDecodeError:
                    await send_text(
                        {"type": "error", "message": "invalid json control event"}
                    )
                    continue
                if not isinstance(control, dict):
                    await send_text(
                        {"type": "error", "message": "control event must be an object"}
                    )
                    continue

                event_type = control.get("type")
                if event_type == "audio.frame":
                    if pending_audio_header is not None:
                        await send_text(
                            {
                                "type": "error",
                                "message": (
                                    "audio frame metadata requires a following "
                                    "binary payload"
                                ),
                            }
                        )
                        continue
                    try:
                        pending_audio_header = AudioFrameHeader.from_payload(control)
                    except ValueError as exc:
                        await send_text({"type": "error", "message": str(exc)})
                elif event_type == "ping":
                    await send_text({"type": "pong", "id": control.get("id")})
                elif event_type == "stop":
                    # An explicit Stop ends the session; it is never parked.
                    if active is not None:
                        active.stopped = True
                    await _drain(
                        session,
                        emit_model_event,
                        pending_unit_capture_start_ms=(
                            audio_timeline.pending_unit_capture_start_ms
                        ),
                    )
                    await stop_speech_output_pump()
                    await websocket_outbox.join()
                    await send_text({"type": "session.done"})
                    await websocket_outbox.join()
                    await websocket.close()
                    return
                elif event_type == "reset":
                    pending_audio_header = None
                    audio_timeline.reset()
                    if runtime.detached_talker is not None:
                        # Cancel Talker generation before resetting the conversation.
                        await asyncio.to_thread(session.interrupt_output)
                    await stop_speech_output_pump()
                    if outbound_task is not None:
                        outbound_task.cancel()
                        await asyncio.gather(outbound_task, return_exceptions=True)
                        outbound_task = None
                    websocket_outbox.discard_pending()
                    assert active is not None
                    screen_frames = active.duplex.screen_frames
                    # Keep the video source across reset, but clear captured frames.
                    screen_frames.reset()
                    active.coordinator = None
                    await coordinator.close(discard_state=True)
                    coordinator = None
                    await asyncio.to_thread(session.close)
                    session = await _open_session(
                        runtime,
                        media_mode=active.media_mode,
                        screen_frames=screen_frames,
                    )
                    if runtime.detached_talker is not None:
                        speech_output_stop, speech_output_task = (
                            start_speech_output_pump(session)
                        )
                    coordinator = await build_coordinator(session)
                    active.duplex = session
                    active.coordinator = coordinator
                    outbound_task = asyncio.create_task(
                        forward_tool_outputs(),
                        name=f"gander-native-tool-output-{session_id}",
                    )
                    await send_text(
                        {
                            "type": "reset.done",
                            "generation_id": int(
                                session.talker_state().get("generation_id", 0)
                            ),
                            "media_mode": active.media_mode,
                            "video_source": active.video_source,
                        }
                    )
                elif event_type == "media.mode":
                    assert active is not None
                    want_video = bool(control.get("video"))
                    source = control.get("source") if want_video else None
                    reason = _reject_media_mode(
                        runtime, want_video=want_video, source=source
                    )
                    if reason is not None:
                        # A rejected mode request is nonfatal for the session.
                        await send_text(
                            {"type": "media.mode.rejected", "reason": reason}
                        )
                        continue
                    target = resolve_target_mode(
                        want_video=want_video,
                        client_video_mode=runtime.settings.client_video_mode,
                    )
                    previous = await asyncio.to_thread(
                        active.duplex.set_media_mode, target
                    )
                    active.media_mode = target
                    active.video_source = source
                    warnings = source_warnings(source)
                    if warnings:
                        LOGGER.warning(
                            "session %s enabled %s video: %s",
                            session_id,
                            source,
                            ", ".join(warnings),
                        )
                    await send_text(
                        {
                            "type": "media.mode.done",
                            "video": want_video,
                            "source": source,
                            "media_mode": target,
                            "previous_media_mode": previous,
                            "estimated_tokens_per_unit": estimated_tokens_per_unit(
                                media_mode=target,
                                speak_text_tokens_per_unit=(
                                    runtime.params.speak_text_tokens_per_unit
                                ),
                            ),
                            "warnings": list(warnings),
                            "screen": (
                                _screen_channel(runtime, active)
                                if want_video
                                else None
                            ),
                        }
                    )
                elif event_type == "break":
                    await asyncio.to_thread(session.set_break)
                    await send_text({"type": "break.done"})
                elif event_type == "clear_break":
                    await asyncio.to_thread(session.clear_break)
                    await send_text({"type": "clear_break.done"})
                elif event_type == "tool.response":
                    try:
                        response_event = await coordinator.inject_external_tool_response(
                            control.get("content")
                        )
                    except (RuntimeError, ValueError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    if response_event is None:
                        await send_text({"type": "tool.response.queued"})
                    else:
                        await emit_model_event(response_event)
                elif event_type == "turn.final":
                    try:
                        turn = coordinator.bind_final_turn(control)
                    except (TypeError, ValueError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    await send_text(
                        {
                            "type": "turn.final.accepted",
                            "turn_id": turn.turn_id,
                            "context_revision": turn.context_revision,
                        }
                    )
                elif event_type == "memory.episode":
                    if runtime.params.sliding_window_mode != "context_memory":
                        await send_text(
                            {
                                "type": "error",
                                "message": "memory.episode is disabled for this context mode",
                            }
                        )
                        continue
                    if "entry" not in control:
                        await send_text(
                            {
                                "type": "error",
                                "message": "memory.episode requires entry",
                            }
                        )
                        continue
                    try:
                        changed = await coordinator.inject_memory_episode(
                            control["entry"]
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    await send_text(
                        {
                            "type": "memory.episode.done",
                            "context_changed": changed,
                        }
                    )
                elif event_type == "context":
                    await send_text(
                        {
                            "type": "error",
                            "message": (
                                "task_tools_v1 binds user text only through turn.final"
                            ),
                        }
                    )
                elif event_type == "task_status":
                    status = coordinator.task_status()
                    await send_text({"type": "task_status", "task": status})
                else:
                    await send_text(
                        {
                            "type": "error",
                            "message": f"unsupported control event: {event_type}",
                        }
                    )
        except WebSocketDisconnect:
            LOGGER.info("duplex websocket disconnected: %s", session_id)
            # A socket that dropped without a Stop is a candidate for resume.
            park_for_resume = True
        except _StartupFlood as exc:
            # The client's doing, not a fault here: no traceback, and the
            # session ends rather than being held for a resume.
            LOGGER.warning("duplex startup flood from %s: %s", session_id, exc)
            try:
                await send_text(
                    {"type": "error", "message": str(exc), "fatal": True}
                )
                await websocket_outbox.join()
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception as exc:
            LOGGER.exception("duplex websocket failed: %s", session_id)
            try:
                await send_text(
                    {"type": "error", "message": str(exc), "fatal": True}
                )
                await websocket_outbox.join()
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            async def detach_connection() -> None:
                """Stop everything tied to this socket, keeping the session."""

                if (
                    runtime.detached_talker is not None
                    and session is not None
                    and not session.closed
                ):
                    try:
                        await asyncio.to_thread(session.interrupt_output)
                    except RuntimeError:
                        pass
                await stop_speech_output_pump()
                if warmup_task is not None:
                    warmup_task.cancel()
                    await asyncio.gather(warmup_task, return_exceptions=True)
                if outbound_task is not None:
                    outbound_task.cancel()
                    await asyncio.gather(outbound_task, return_exceptions=True)
                await websocket_outbox.close()

            if (
                park_for_resume
                and active is not None
                and not active.stopped
                and runtime.sessions.get(session_id) is active
            ):
                # The socket dropped and nobody asked to stop. Hold the
                # Thinker, the coordinator and the slot for a short window so
                # the same client can come back to the same conversation, and
                # leave /ws/screen open meanwhile so the share is untouched.
                #
                # Nothing here awaits. If this handler was cancelled rather
                # than disconnected, an await would re-raise immediately and
                # the session would be dropped instead of held; the grace
                # timer does the waiting, and this only records the decision.
                active.attached = False
                active.resumed = asyncio.Event()
                grace_task = asyncio.create_task(
                    expire_reconnect_grace(active, detach_connection),
                    name=f"gander-duplex-grace-{session_id}",
                )
                runtime.grace_tasks.add(grace_task)
                grace_task.add_done_callback(runtime.grace_tasks.discard)
                LOGGER.info(
                    "holding %s for %.1fs for a reconnect",
                    session_id,
                    runtime.settings.reconnect_grace_sec,
                )
                return

            if session is None and open_task is not None and not open_task.cancelled():
                # The client left while the model was still being built on a
                # worker thread. Cancelling the await did not stop that thread,
                # so the slot is handed to a task that waits for the session to
                # exist and closes it. Releasing here would let a second
                # session build against the same model while this one is
                # finished and never closed.
                abandoned = asyncio.create_task(
                    close_abandoned_open(open_task),
                    name=f"gander-abandoned-open-{session_id}",
                )
                runtime.grace_tasks.add(abandoned)
                abandoned.add_done_callback(runtime.grace_tasks.discard)
                LOGGER.info(
                    "holding the model slot for %s until its abandoned open "
                    "finishes",
                    session_id,
                )
                try:
                    await detach_connection()
                except Exception:
                    LOGGER.warning(
                        "detaching the abandoned startup for %s failed",
                        session_id,
                        exc_info=True,
                    )
                return

            try:
                await detach_connection()
                if active is not None:
                    if runtime.sessions.get(session_id) is active:
                        runtime.sessions.pop(session_id, None)
                    active.ended.set()
                if coordinator is not None:
                    # Everything that touches the Thinker stops here. The
                    # gateway's own teardown waits until the slot is free.
                    await coordinator.stop_model_jobs()
            except Exception:
                LOGGER.warning(
                    "teardown before slot release failed for %s",
                    session_id,
                    exc_info=True,
                )
            finally:
                # However the block above ended - cleanly, in error, or
                # cancelled - the Thinker is closed and the slot is freed.
                # Neither call suspends, so nothing can land between them.
                try:
                    if session is not None and not session.closed:
                        session.close()
                finally:
                    release_model_slot()

            if coordinator is not None:
                # Worker/Ornith teardown is HTTP and can be slow. It runs with
                # the slot already free, so Stop followed immediately by Start
                # no longer reports the model busy.
                try:
                    await coordinator.close()
                except Exception:
                    LOGGER.warning(
                        "worker gateway teardown failed for %s",
                        session_id,
                        exc_info=True,
                    )

    return app
