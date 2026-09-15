"""Lifecycle tests for the online duplex endpoints.

Each test names a thing a person can observe: whether Start works after Stop,
whether one bad frame ends their screen share, whether Gander answers before
the Brain is warm.
"""
from __future__ import annotations

import contextlib
import json
import threading

import pytest
from starlette.testclient import TestClient


@contextlib.contextmanager
def connect(client, url: str):
    """Open a websocket, tolerating a handler that finishes before we close.

    The duplex handler returns as soon as the client is gone, so by the time
    the test client sends its own close frame the server task can already be
    finished. Starlette surfaces that as a cancelled portal future, which says
    nothing about the session under test - the assertions in each test, and
    the server's own log lines, are what decide whether it behaved.
    """

    connection = client.websocket_connect(url)
    websocket = connection.__enter__()
    try:
        yield websocket
    finally:
        try:
            connection.__exit__(None, None, None)
        except Exception:
            pass


def _drain_until(ws, wanted: str, limit: int = 20) -> dict:
    """Read frames until one of type ``wanted`` arrives."""

    for _ in range(limit):
        message = ws.receive_json()
        if message.get("type") == wanted:
            return message
    raise AssertionError(f"never received {wanted!r}")


def _settle(ws) -> dict:
    """Wait for `ready` and for the background Brain warmup to finish.

    Warmup reports itself after `ready`, so a test that acts before it settles
    is racing a background writer. Tests that are specifically about warmup
    timing read the frames themselves instead of calling this.
    """

    ready = _drain_until(ws, "ready")
    for _ in range(40):
        message = ws.receive_json()
        if message.get("type") == "brain.status" and message["status"] in {
            "ready",
            "error",
        }:
            return ready
    raise AssertionError("brain warmup never settled")


def test_ready_does_not_wait_for_the_brain(harness):
    """Gander sees and hears before Ornith answers (audit 14)."""

    gate = threading.Event()
    h = harness(warmup_gate=gate)
    try:
        with TestClient(h.app) as client:
            with connect(client, "/ws/duplex?session_id=s1") as ws:
                ready = _drain_until(ws, "ready")
                assert ready["session_id"] == "s1"
                # `ready` arrived while warmup is still blocked on the gate.
                assert not gate.is_set()
                warming = _drain_until(ws, "brain.status")
                assert warming["status"] == "warming"
    finally:
        gate.set()


def test_warmup_failure_still_reaches_ready(harness):
    """An unreachable Brain is reported, not fatal (audit 14)."""

    h = harness(warmup_fails=True)
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _drain_until(ws, "ready")
            statuses = []
            for _ in range(6):
                message = ws.receive_json()
                if message.get("type") == "brain.status":
                    statuses.append(message["status"])
                    if message["status"] in {"ready", "error"}:
                        break
            assert statuses[-1] == "error"
            # The session is still usable: a ping still answers.
            ws.send_text(json.dumps({"type": "ping", "id": 7}))
            pong = _drain_until(ws, "pong")
            assert pong["id"] == 7


def test_control_error_is_recoverable(harness):
    """A bad control event does not end the session (audit 12)."""

    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
            ws.send_text(json.dumps({"type": "nonsense"}))
            error = _drain_until(ws, "error")
            assert error["fatal"] is False
            # Still alive afterwards.
            ws.send_text(json.dumps({"type": "ping", "id": 1}))
            assert _drain_until(ws, "pong")["id"] == 1


def test_malformed_json_control_is_recoverable(harness):
    """Unparseable control text is reported without ending the session."""

    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
            ws.send_text("{not json")
            error = _drain_until(ws, "error")
            assert error["fatal"] is False
            assert "invalid json" in error["message"]


def _read_rejection(client, url: str) -> dict:
    """First frame of a connection the server rejects and then closes.

    The server closes straight after replying, so the connection is unwound by
    hand: letting the context manager close an already-closed socket surfaces
    as a cancelled portal future rather than anything meaningful.
    """

    connection = client.websocket_connect(url)
    websocket = connection.__enter__()
    try:
        return websocket.receive_json()
    finally:
        try:
            connection.__exit__(None, None, None)
        except Exception:
            pass


def test_second_connection_is_told_to_retry(harness):
    """One browser, one Thinker - and the other is asked to wait, not fail."""

    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
            busy = _read_rejection(client, "/ws/duplex?session_id=s2")
            assert busy["type"] == "error"
            assert "busy" in busy["message"]
            # Busy is a retry, not a reason to tear the page down.
            assert busy["fatal"] is False
            assert busy["retry"] is True


def test_invalid_session_id_is_fatal(harness):
    """A malformed session id is the client's bug, and ends the attempt."""

    h = harness()
    with TestClient(h.app) as client:
        rejected = _read_rejection(client, "/ws/duplex?session_id=../escape")
        assert rejected["type"] == "error"
        assert rejected["fatal"] is True


def test_stop_then_immediate_connect_succeeds(harness, caplog):
    """Start straight after Stop must not report the model busy (audit 16)."""

    caplog.set_level("INFO")
    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
        # Second session opens on the slot the first just freed.
        with connect(client, "/ws/duplex?session_id=s2") as ws:
            assert _settle(ws)["session_id"] == "s2"

    releases = [
        record for record in caplog.records
        if "released model slot" in record.getMessage()
    ]
    assert len(releases) == 2, "each session frees the slot exactly once"


def test_slot_is_released_exactly_once(harness, caplog):
    """The single slot is freed once per session, never twice (audit 16)."""

    caplog.set_level("INFO")
    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
    releases = [
        record for record in caplog.records
        if "released model slot for s1" in record.getMessage()
    ]
    assert len(releases) == 1
    assert h.thinkers[0].close_count == 1


def test_health_reports_the_slot_free_after_disconnect(harness):
    """`busy` clears once the session ends, so the client can Start again."""

    h = harness()
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            _settle(ws)
            assert client.get("/health").json()["busy"] is True
        assert client.get("/health").json()["busy"] is False


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _screen_header(frame_id: str) -> dict:
    return {
        "type": "screen.frame",
        "frame_id": frame_id,
        "captured_at_ms": 1_000,
        "encoding": "jpeg",
        "video_source": "screen",
    }


def _open_screen(client, ws_duplex):
    """Attach a screen channel to a duplex session that is already ready."""

    ready = _settle(ws_duplex)
    screen = ready["screen"]
    assert screen["enabled"] is True
    return connect(
        client,
        f"/ws/screen?session_id={ready['session_id']}&token={screen['token']}",
    )


def test_one_bad_frame_does_not_close_the_screen(harness):
    """A malformed frame is dropped; the share survives it (audit 13)."""

    h = harness(media_mode="omni")
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            with _open_screen(client, ws) as screen:
                _drain_until(screen, "screen.ready")

                # Metadata that is not JSON at all.
                screen.send_text("{not json")
                dropped = _drain_until(screen, "screen.frame.dropped")
                assert dropped["frame_id"] is None
                assert dropped["reason"]

                # A header that parses but names an encoding we do not accept.
                bad = _screen_header("f2") | {"encoding": "tiff"}
                screen.send_text(json.dumps(bad))
                dropped = _drain_until(screen, "screen.frame.dropped")
                assert "encoding" in dropped["reason"]

                # The socket is still usable: a good frame is accepted.
                screen.send_text(json.dumps(_screen_header("f3")))
                screen.send_bytes(_jpeg())
                accepted = _drain_until(screen, "screen.frame.accepted")
                assert accepted["frame_id"] == "f3"
                assert accepted["width"] == 32


def test_undecodable_image_is_dropped_not_fatal(harness):
    """Bytes that are not an image drop one frame, not the channel."""

    h = harness(media_mode="omni")
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1") as ws:
            with _open_screen(client, ws) as screen:
                _drain_until(screen, "screen.ready")
                screen.send_text(json.dumps(_screen_header("f1")))
                screen.send_bytes(b"this is not a jpeg")
                dropped = _drain_until(screen, "screen.frame.dropped")
                assert dropped["frame_id"] == "f1"

                screen.send_text(json.dumps(_screen_header("f2")))
                screen.send_bytes(_jpeg())
                assert _drain_until(screen, "screen.frame.accepted")["frame_id"] == "f2"


def test_disconnect_during_startup_frees_the_slot(harness, caplog):
    """A client that gives up while the model opens frees it (audit 15)."""

    caplog.set_level("INFO")
    gate = threading.Event()
    h = harness(open_gate=gate)
    with TestClient(h.app) as client:
        with connect(client, "/ws/duplex?session_id=s1"):
            # Leave immediately, while _open_session is still blocked.
            pass
        gate.set()
        # The slot must come back without a second session having to wait
        # for the abandoned startup to finish on its own.
        with connect(client, "/ws/duplex?session_id=s2") as ws:
            assert _settle(ws)["session_id"] == "s2"
