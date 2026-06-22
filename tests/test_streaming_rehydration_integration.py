"""
Integration tests for Sub-AC 9.2 — Streaming SSE rehydration pipeline.

This module tests that the PII-Guard proxy correctly rehydrates placeholder
tokens in streaming SSE responses from upstream LLMs, using a bounded
look-ahead buffer to handle tokens that span SSE chunk boundaries.

The tests use a mock streaming SSE server that introduces configurable
inter-chunk delays to verify:

  (a) **TTFT preserved** — the first safe token passes through to the client
      before the full response body is received from the upstream.

  (b) **No block-category placeholder** — no ``[CATEGORY_N]`` or
      ``[CATEGORY_N_BLOCKED]`` token appears in any emitted chunk.

  (c) **Lossless round-trip** — the final reassembled response from all
      forwarded chunks matches the original unmasked content exactly.

Test scenarios mandated by Sub-AC 9.2
--------------------------------------
1. **Whole-chunk placeholders** — every placeholder is delivered complete in
   a single SSE chunk (no split).  Rehydration still occurs; no tokens appear.

2. **Split-boundary placeholders** — every placeholder is deliberately split
   across SSE chunk boundaries.  The look-ahead buffer reassembles them
   before rehydration.

3. **Mixed** — a response that contains both split placeholders and unsplit
   placeholders interleaved with plain text.

Architecture
------------
  MockStreamingSSEServer
      A minimal HTTP server that sends SSE events with configurable inter-event
      delays, allowing the test to verify that the proxy forwards the first
      rehydrated chunk before all upstream chunks have arrived.

  PIIGuardProxy (under test)
      Configured with a pre-populated session restoration map and pointing at
      the mock server.

  _SSEStreamReader
      A helper that reads SSE bytes from the proxy, records the time of first
      byte received, collects all chunks, and extracts the text deltas for
      verification.
"""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest

from pii_guard.engine import Engine
from pii_guard.proxy import PIIGuardProxy
from pii_guard.streaming_rehydrator import (
    StreamingSSERehydrator,
    _SSEParser,
    _extract_claude_stream_text,
    _extract_openai_stream_text,
    _extract_gemini_stream_text,
    _inject_claude_stream_text,
    _inject_openai_stream_text,
    _inject_gemini_stream_text,
    _extract_text_from_event,
    _inject_text_into_event,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants and helpers
# ─────────────────────────────────────────────────────────────────────────────

# Placeholder pattern — matches [CATEGORY_N] and [CATEGORY_N_BLOCKED]
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+(?:_BLOCKED)?\]")

#: Delay between SSE events on the mock server (seconds).
#: Large enough to make TTFT clearly measurable, small enough for fast tests.
_EVENT_DELAY_S: float = 0.15

#: TTFT must be this factor smaller than the total stream duration.
#: If delay is 150ms and we have ≥2 events, total is ≥150ms.
#: Client should see first byte in <100ms.
_TTFT_FRACTION: float = 0.80


def _contains_placeholder(text: str) -> bool:
    """Return True if *text* contains any ``[CATEGORY_N]`` placeholder."""
    return bool(_PLACEHOLDER_RE.search(text))


def _extract_all_text_from_sse_bytes(
    raw: bytes,
    provider: str,
) -> str:
    """
    Parse raw SSE bytes and concatenate all text deltas in order.

    Used to reconstruct the full text content from a streaming response
    for final comparison with the expected unmasked content.
    """
    parser = _SSEParser()
    parts: List[str] = []

    for event_type, data, raw_event in parser.feed(raw):
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        text = _extract_text_from_event(obj, provider)
        if text:
            parts.append(text)

    # Flush any remaining partial event
    for event_type, data, raw_event in parser.flush():
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        text = _extract_text_from_event(obj, provider)
        if text:
            parts.append(text)

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Mock streaming SSE upstream server
# ─────────────────────────────────────────────────────────────────────────────

class _SSEEvent:
    """A single SSE event to be sent by the mock server."""
    __slots__ = ("event_type", "data", "delay_before_s")

    def __init__(
        self,
        data: str,
        event_type: Optional[str] = None,
        delay_before_s: float = 0.0,
    ) -> None:
        self.data = data
        self.event_type = event_type
        self.delay_before_s = delay_before_s

    def encode(self) -> bytes:
        """Encode this event to SSE wire format bytes."""
        if self.event_type:
            return f"event: {self.event_type}\ndata: {self.data}\n\n".encode("utf-8")
        return f"data: {self.data}\n\n".encode("utf-8")


class _MockSSEHandler(BaseHTTPRequestHandler):
    """HTTP handler for the mock SSE upstream server."""

    def log_message(self, fmt: str, *args) -> None:
        pass  # suppress access log in tests

    def do_POST(self) -> None:
        # Read and discard the request body
        length = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(length)

        events: List[_SSEEvent] = self.server.events  # type: ignore[attr-defined]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for event in events:
            if event.delay_before_s > 0:
                time.sleep(event.delay_before_s)
            try:
                self.wfile.write(event.encode())
                self.wfile.flush()
            except OSError:
                return  # client disconnected


class MockStreamingSSEServer:
    """
    Minimal mock SSE upstream server.

    Sends a configurable sequence of SSE events with per-event delays.
    Thread-safe via a lock on the ``events`` attribute setter.
    """

    def __init__(self) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), _MockSSEHandler)
        self._server.events = []  # type: ignore[attr-defined]
        host, port = self._server.server_address
        self._host: str = host
        self._port: int = port
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def events(self) -> List[_SSEEvent]:
        return self._server.events  # type: ignore[attr-defined]

    @events.setter
    def events(self, value: List[_SSEEvent]) -> None:
        self._server.events = value  # type: ignore[attr-defined]

    def start(self) -> "MockStreamingSSEServer":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="mock-sse-server",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "MockStreamingSSEServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SSE stream reader helper
# ─────────────────────────────────────────────────────────────────────────────

class _SSEStreamReader:
    """
    Reads all bytes from a streaming HTTP response, recording TTFT.

    Attributes
    ----------
    first_chunk_latency_s:
        Seconds elapsed until the first non-empty chunk arrived.
        ``None`` if no chunks were received.
    all_chunks:
        All raw byte chunks received from the response, in order.
    full_body:
        Concatenation of all chunks.
    """

    def __init__(self) -> None:
        self.first_chunk_latency_s: Optional[float] = None
        self.all_chunks: List[bytes] = []
        self.full_body: bytes = b""

    def read_response(
        self,
        conn: http.client.HTTPConnection,
        method: str,
        path: str,
        body: bytes,
        headers: Dict[str, str],
    ) -> None:
        """Send a request and read the streaming response, recording TTFT."""
        conn.request(method, path, body, headers)
        resp = conn.getresponse()

        start = time.monotonic()
        buf = b""

        while True:
            # Read in small chunks so TTFT can be measured accurately
            chunk = resp.read(512)
            if not chunk:
                break
            if self.first_chunk_latency_s is None:
                self.first_chunk_latency_s = time.monotonic() - start
            self.all_chunks.append(chunk)
            buf += chunk

        self.full_body = buf


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_sse_server():
    """Start a mock SSE upstream server for the duration of a test."""
    with MockStreamingSSEServer() as srv:
        yield srv


def _make_proxy_with_map(
    upstream_url: str,
    restoration_map: Dict[str, str],
) -> PIIGuardProxy:
    """
    Create a PIIGuardProxy with a pre-populated session restoration map.

    Pre-populating avoids the need to route a PII-bearing request through
    the proxy just to populate the session map — which would complicate
    multi-step tests.
    """
    engine = Engine()
    # Inject the restoration map directly into the session map by encoding
    # each (original, token) pair.
    for token, original in restoration_map.items():
        # Derive category and index from token string: "EMAIL_1" → ("EMAIL", 1)
        # For blocked tokens: "API_KEY_1_BLOCKED" → blocked=True
        blocked = token.endswith("_BLOCKED")
        base_token = token[:-len("_BLOCKED")] if blocked else token
        # Split on last underscore to separate category from index
        parts = base_token.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            category = parts[0]
        else:
            category = base_token
        engine.session_map.encode(original, category, blocked=blocked)

    proxy = PIIGuardProxy(
        upstream_url,
        engine=engine,
        unknown_field_action="warn_allow",
        unscannable_action="warn_allow",
        rehydrate_responses=True,
        terminal_restore=False,
    )
    return proxy


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests for StreamingSSERehydrator
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingSSERehydratorUnit:
    """Pure unit tests for StreamingSSERehydrator and helper functions."""

    RMAP: Dict[str, str] = {
        "EMAIL_1": "alice@corp.io",
        "PHONE_1": "010-1234-5678",
        "PERSON_1": "Alice Smith",
        "API_KEY_1_BLOCKED": "sk-secret-key",
    }

    # ── Helper: make SSE event bytes ──────────────────────────────────────────

    @staticmethod
    def _claude_delta_event(text: str, index: int = 0) -> bytes:
        obj = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(obj)}\n\n".encode()

    @staticmethod
    def _claude_stop_event() -> bytes:
        obj = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(obj)}\n\n".encode()

    @staticmethod
    def _openai_delta_event(text: str) -> bytes:
        obj = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()

    @staticmethod
    def _gemini_delta_event(text: str) -> bytes:
        obj = {
            "candidates": [{
                "content": {"parts": [{"text": text}], "role": "model"}
            }]
        }
        return f"data: {json.dumps(obj)}\n\n".encode()

    # ── Provider text extraction helpers ──────────────────────────────────────

    def test_extract_claude_text_from_text_delta_event(self):
        obj = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }
        assert _extract_claude_stream_text(obj) == "hello"

    def test_extract_claude_text_returns_none_for_message_stop(self):
        obj = {"type": "message_stop"}
        assert _extract_claude_stream_text(obj) is None

    def test_extract_claude_text_returns_none_for_non_text_delta(self):
        obj = {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        }
        assert _extract_claude_stream_text(obj) is None

    def test_extract_openai_text_from_delta_content(self):
        obj = {
            "choices": [{"delta": {"content": "world"}}]
        }
        assert _extract_openai_stream_text(obj) == "world"

    def test_extract_openai_text_returns_none_for_empty_content(self):
        obj = {"choices": [{"delta": {"role": "assistant"}}]}
        assert _extract_openai_stream_text(obj) is None

    def test_extract_gemini_text_from_parts(self):
        obj = {
            "candidates": [{
                "content": {"parts": [{"text": "Gemini text"}]}
            }]
        }
        assert _extract_gemini_stream_text(obj) == "Gemini text"

    def test_inject_claude_text(self):
        obj = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "[EMAIL_1]"},
        }
        modified = _inject_claude_stream_text(obj, "alice@corp.io")
        assert modified["delta"]["text"] == "alice@corp.io"
        # Original not mutated
        assert obj["delta"]["text"] == "[EMAIL_1]"

    def test_inject_openai_text(self):
        obj = {
            "choices": [{"delta": {"content": "[PHONE_1]"}}]
        }
        modified = _inject_openai_stream_text(obj, "010-1234-5678")
        assert modified["choices"][0]["delta"]["content"] == "010-1234-5678"
        # Original not mutated
        assert obj["choices"][0]["delta"]["content"] == "[PHONE_1]"

    def test_inject_gemini_text(self):
        obj = {
            "candidates": [{
                "content": {"parts": [{"text": "[PERSON_1]"}]}
            }]
        }
        modified = _inject_gemini_stream_text(obj, "Alice Smith")
        assert modified["candidates"][0]["content"]["parts"][0]["text"] == "Alice Smith"

    # ── Basic rehydrator feed / flush ──────────────────────────────────────────

    def test_single_chunk_plain_text_forwarded_immediately(self):
        """Plain text with no placeholders passes through immediately."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")
        chunk = self._claude_delta_event("Hello world")
        output = rehydrator.feed_chunk(chunk)
        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert text == "Hello world"

    def test_single_chunk_complete_placeholder_rehydrated(self):
        """A complete [EMAIL_1] token in one chunk is rehydrated."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")
        chunk = self._claude_delta_event("Reply to [EMAIL_1] soon.")
        output = rehydrator.feed_chunk(chunk)
        output += rehydrator.flush()
        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert "alice@corp.io" in text
        assert "[EMAIL_1]" not in text

    def test_single_chunk_blocked_token_rehydrated(self):
        """[API_KEY_1_BLOCKED] is also rehydrated (no raw token in output)."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")
        chunk = self._claude_delta_event("Key was [API_KEY_1_BLOCKED] here.")
        output = rehydrator.feed_chunk(chunk)
        output += rehydrator.flush()
        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert "sk-secret-key" in text
        assert "[API_KEY_1_BLOCKED]" not in text

    def test_non_text_events_forwarded_verbatim(self):
        """Claude message_stop events are passed through unchanged."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")
        stop = self._claude_stop_event()
        output = rehydrator.feed_chunk(stop)
        assert b"message_stop" in output

    def test_openai_done_sentinel_forwarded_verbatim(self):
        """OpenAI [DONE] sentinel is forwarded unchanged."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "openai")
        done = b"data: [DONE]\n\n"
        output = rehydrator.feed_chunk(done)
        assert b"[DONE]" in output

    def test_empty_restoration_map_text_forwarded_unchanged(self):
        """With no restoration map, text passes through without rehydration."""
        rehydrator = StreamingSSERehydrator({}, "claude")
        chunk = self._claude_delta_event("Some text with no PII")
        output = rehydrator.feed_chunk(chunk)
        output += rehydrator.flush()
        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert text == "Some text with no PII"

    # ── Split-boundary handling ────────────────────────────────────────────────

    def test_placeholder_split_across_two_chunks(self):
        """
        [EMAIL_1] split between two SSE chunks is reassembled correctly.

        Chunk 1 text: "I will reply to [EMA"
        Chunk 2 text: "IL_1] shortly."
        """
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")

        chunk1 = self._claude_delta_event("I will reply to [EMA")
        chunk2 = self._claude_delta_event("IL_1] shortly.")

        output1 = rehydrator.feed_chunk(chunk1)
        output2 = rehydrator.feed_chunk(chunk2)
        tail = rehydrator.flush()

        combined_output = output1 + output2 + tail
        full_text = _extract_all_text_from_sse_bytes(combined_output, "claude")

        # Final text must contain the rehydrated value
        assert "alice@corp.io" in full_text, (
            f"Expected 'alice@corp.io' in text. Got: {full_text!r}"
        )
        # No placeholder token should appear
        assert not _contains_placeholder(full_text), (
            f"Placeholder token must not appear in output. Got: {full_text!r}"
        )
        # Full content is correct
        assert full_text == "I will reply to alice@corp.io shortly.", (
            f"Full text mismatch. Got: {full_text!r}"
        )

    def test_placeholder_split_one_char_at_a_time(self):
        """[PERSON_1] split one character per chunk is fully reassembled."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")

        token = "[PERSON_1]"
        output = b""
        for char in f"Dear {token}, how are you?":
            output += rehydrator.feed_chunk(self._claude_delta_event(char))
        output += rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert "Alice Smith" in text
        assert not _contains_placeholder(text)
        assert text == "Dear Alice Smith, how are you?"

    def test_openai_split_placeholder(self):
        """OpenAI streaming: placeholder split across chunks is rehydrated."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "openai")

        chunk1 = self._openai_delta_event("Contact [EMA")
        chunk2 = self._openai_delta_event("IL_1].")
        done = b"data: [DONE]\n\n"

        output = b""
        output += rehydrator.feed_chunk(chunk1)
        output += rehydrator.feed_chunk(chunk2)
        output += rehydrator.feed_chunk(done)
        output += rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output, "openai")
        assert "alice@corp.io" in text
        assert not _contains_placeholder(text)

    def test_gemini_split_placeholder(self):
        """Gemini streaming: placeholder split across chunks is rehydrated."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "gemini")

        chunk1 = self._gemini_delta_event("Calling [PHO")
        chunk2 = self._gemini_delta_event("NE_1].")

        output = b""
        output += rehydrator.feed_chunk(chunk1)
        output += rehydrator.feed_chunk(chunk2)
        output += rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output, "gemini")
        assert "010-1234-5678" in text
        assert not _contains_placeholder(text)

    # ── Multiple placeholders ──────────────────────────────────────────────────

    def test_multiple_placeholders_all_rehydrated(self):
        """Multiple placeholders in the same stream are all rehydrated."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")

        output = b""
        output += rehydrator.feed_chunk(self._claude_delta_event("Contact "))
        output += rehydrator.feed_chunk(self._claude_delta_event("[EMAIL_1]"))
        output += rehydrator.feed_chunk(self._claude_delta_event(" or "))
        output += rehydrator.feed_chunk(self._claude_delta_event("[PHONE_1]"))
        output += rehydrator.feed_chunk(self._claude_delta_event("."))
        output += rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert "alice@corp.io" in text
        assert "010-1234-5678" in text
        assert not _contains_placeholder(text)

    def test_blocked_and_regular_tokens_both_rehydrated(self):
        """Both [EMAIL_1] and [API_KEY_1_BLOCKED] are rehydrated in one stream."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")

        output = b""
        output += rehydrator.feed_chunk(
            self._claude_delta_event("Email: [EMAIL_1], key: [API_KEY_1_BLOCKED]")
        )
        output += rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output, "claude")
        assert "alice@corp.io" in text
        assert "sk-secret-key" in text
        assert not _contains_placeholder(text)

    # ── Flush correctness ──────────────────────────────────────────────────────

    def test_flush_emits_tail_content(self):
        """Content held in the look-ahead buffer is emitted on flush()."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")

        # Partial placeholder at end — buffer holds it until flush
        output = rehydrator.feed_chunk(self._claude_delta_event("Hello [EMAIL_1"))
        # Nothing comes after — flush should emit the tail
        tail = rehydrator.flush()

        text = _extract_all_text_from_sse_bytes(output + tail, "claude")
        # The partial "[EMAIL_1" can't be fully matched — it's emitted verbatim
        # (no ']' means it's not a complete placeholder; rehydration won't match it)
        assert "Hello" in text

    def test_flush_multiple_times_safe(self):
        """Calling flush() twice is safe; second call returns empty."""
        rehydrator = StreamingSSERehydrator(self.RMAP, "claude")
        rehydrator.feed_chunk(self._claude_delta_event("some text"))
        rehydrator.flush()
        second_flush = rehydrator.flush()
        # Second flush should not crash and should not produce extra content
        assert isinstance(second_flush, bytes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Unit tests for _SSEParser
# ─────────────────────────────────────────────────────────────────────────────

class TestSSEParser:
    """Unit tests for the internal _SSEParser class."""

    def test_single_event_parsed(self):
        parser = _SSEParser()
        raw = b"data: {\"text\": \"hello\"}\n\n"
        events = list(parser.feed(raw))
        assert len(events) == 1
        event_type, data, raw_event = events[0]
        assert event_type is None
        assert data == '{"text": "hello"}'

    def test_event_with_type_parsed(self):
        parser = _SSEParser()
        raw = b"event: content_block_delta\ndata: {\"x\": 1}\n\n"
        events = list(parser.feed(raw))
        assert len(events) == 1
        event_type, data, raw_event = events[0]
        assert event_type == "content_block_delta"
        assert data == '{"x": 1}'

    def test_multiple_events_in_one_chunk(self):
        parser = _SSEParser()
        raw = b"data: first\n\ndata: second\n\n"
        events = list(parser.feed(raw))
        assert len(events) == 2
        assert events[0][1] == "first"
        assert events[1][1] == "second"

    def test_event_split_across_chunks(self):
        parser = _SSEParser()
        part1 = b"data: he"
        part2 = b"llo\n\n"
        events1 = list(parser.feed(part1))
        events2 = list(parser.feed(part2))
        assert events1 == []  # incomplete — held in buffer
        assert len(events2) == 1
        assert events2[0][1] == "hello"

    def test_done_sentinel_parsed(self):
        parser = _SSEParser()
        raw = b"data: [DONE]\n\n"
        events = list(parser.feed(raw))
        assert len(events) == 1
        assert events[0][1] == "[DONE]"

    def test_flush_emits_remaining_partial_event(self):
        parser = _SSEParser()
        _ = list(parser.feed(b"data: partial"))  # no trailing \n\n
        remaining = list(parser.flush())
        assert len(remaining) == 1
        assert "partial" in remaining[0][1]

    def test_crlf_separator_accepted(self):
        """SSE events separated by \\r\\n\\r\\n are parsed correctly."""
        parser = _SSEParser()
        raw = b"data: first\r\n\r\ndata: second\r\n\r\n"
        events = list(parser.feed(raw))
        assert len(events) == 2
        assert events[0][1] == "first"
        assert events[1][1] == "second"

    def test_empty_feed_yields_nothing(self):
        parser = _SSEParser()
        events = list(parser.feed(b""))
        assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. Integration tests with mock streaming SSE server
# ─────────────────────────────────────────────────────────────────────────────

# Restoration map for integration tests
_INTEGRATION_RMAP: Dict[str, str] = {
    "EMAIL_1": "alice@corp.io",
    "PHONE_1": "010-1234-5678",
    "PERSON_1": "Alice Smith",
    "API_KEY_1_BLOCKED": "sk-top-secret",
}

# Expected original unmasked text for each scenario (used for assertion (c))
_WHOLE_CHUNK_ORIGINAL = (
    "Hello, Alice Smith. Your email alice@corp.io was received. "
    "Call 010-1234-5678 if needed."
)
_SPLIT_CHUNK_ORIGINAL = (
    "Dear alice@corp.io, your key sk-top-secret is active."
)
_MIXED_ORIGINAL = (
    "Hi Alice Smith! Contact alice@corp.io or 010-1234-5678. "
    "The key sk-top-secret should not appear."
)


def _make_claude_text_events(texts: List[str], delay_s: float) -> List[_SSEEvent]:
    """Build a sequence of Claude SSE events with delays."""
    events: List[_SSEEvent] = []
    for i, text in enumerate(texts):
        obj = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
        events.append(_SSEEvent(
            data=json.dumps(obj),
            event_type="content_block_delta",
            delay_before_s=delay_s if i > 0 else 0.0,
        ))
    # Append message_stop
    events.append(_SSEEvent(
        data=json.dumps({"type": "message_stop"}),
        event_type="message_stop",
        delay_before_s=0.0,
    ))
    return events


def _read_streaming_response(
    proxy_base_url: str,
    path: str,
    provider: str,
) -> Tuple[Optional[float], List[bytes], str, float]:
    """
    Send a POST request to the proxy and read the streaming response.

    Returns
    -------
    (first_chunk_latency_s, all_chunks, full_text, total_elapsed_s)
        first_chunk_latency_s: seconds until first non-empty chunk arrived
        all_chunks: list of raw byte chunks received
        full_text: concatenated text deltas from all SSE events
        total_elapsed_s: total time to read the full response
    """
    # Build a minimal request body
    if provider == "claude":
        body = json.dumps({
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode("utf-8")
    elif provider == "openai":
        body = json.dumps({
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode("utf-8")
    else:
        body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "test"}]}]}).encode()

    import urllib.parse
    parsed = urllib.parse.urlparse(proxy_base_url)
    host = parsed.hostname
    port = parsed.port

    conn = http.client.HTTPConnection(host, port, timeout=10)

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }

    start = time.monotonic()
    conn.request("POST", path, body, headers)
    resp = conn.getresponse()

    first_chunk_latency_s: Optional[float] = None
    all_chunks: List[bytes] = []
    buf = b""

    # Use resp.fp.read1() for non-blocking streaming reads.
    # resp.read(n) blocks until n bytes arrive OR EOF — for small SSE events
    # (< n bytes each), this effectively means blocking until EOF, which
    # collapses TTFT to total_elapsed.
    # BufferedReader.read1(n) does at most ONE underlying read() syscall,
    # returning whatever data the OS has available immediately.  This gives
    # true per-chunk delivery for TTFT measurement.
    fp = getattr(resp, "fp", None)
    use_read1 = fp is not None and hasattr(fp, "read1")

    while True:
        if use_read1:
            fp = getattr(resp, "fp", None)  # refresh; may become None after EOF
            if fp is None:
                chunk = b""
            else:
                chunk = fp.read1(512)
        else:
            chunk = resp.read(512)
        if not chunk:
            break
        if first_chunk_latency_s is None:
            first_chunk_latency_s = time.monotonic() - start
        all_chunks.append(chunk)
        buf += chunk

    total_elapsed_s = time.monotonic() - start

    conn.close()
    full_text = _extract_all_text_from_sse_bytes(buf, provider)
    return first_chunk_latency_s, all_chunks, full_text, total_elapsed_s


class TestStreamingIntegration:
    """
    Integration tests for the full proxy streaming SSE rehydration pipeline.

    Uses a mock SSE server to:
    (a) Measure TTFT — first byte arrives before full response is received.
    (b) Assert no block-category placeholder in any emitted chunk.
    (c) Assert final reassembled response matches original unmasked content.

    Three scenarios are covered:
    1. All-whole-chunk: every placeholder delivered whole in one SSE chunk.
    2. All-split: every placeholder split across chunk boundaries.
    3. Mixed: split and unsplit placeholders interleaved with plain text.
    """

    CLAUDE_PATH = "/v1/messages"

    # ── Scenario 1: Every placeholder delivered whole in one chunk ───────────

    def test_whole_chunk_placeholders_rehydrated(self, mock_sse_server):
        """
        SCENARIO 1 — Whole-chunk placeholders.

        Each placeholder is delivered as a complete token in a single SSE event
        with no split.  Rehydration still replaces all tokens; no placeholder
        appears in the forwarded bytes.

        Assertions (b) and (c) verified.
        """
        # Build events: each text segment is one event
        texts = [
            "Hello, [PERSON_1]. ",
            "Your email [EMAIL_1] was received. ",
            "Call [PHONE_1] if needed.",
        ]
        mock_sse_server.events = _make_claude_text_events(texts, delay_s=0.0)

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # ── (b) No placeholder token in any chunk ────────────────────────────
        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw), (
            f"Placeholder token found in forwarded chunks:\n{combined_raw!r}"
        )

        # ── (c) Final text matches original ──────────────────────────────────
        assert full_text == _WHOLE_CHUNK_ORIGINAL, (
            f"Final text mismatch.\nExpected: {_WHOLE_CHUNK_ORIGINAL!r}\n"
            f"Got:      {full_text!r}"
        )

    # ── Scenario 2: Every placeholder split across chunk boundaries ──────────

    def test_split_boundary_placeholders_rehydrated(self, mock_sse_server):
        """
        SCENARIO 2 — Split-boundary placeholders.

        Every placeholder token is deliberately split across SSE chunk
        boundaries.  The look-ahead buffer reassembles each token before
        rehydration, ensuring the correct original value appears in the output.

        Assertions (b) and (c) verified.
        """
        # Split each placeholder at a middle character
        # Original: "Dear alice@corp.io, your key sk-top-secret is active."
        # Masked:   "Dear [EMAIL_1], your key [API_KEY_1_BLOCKED] is active."
        # Split:    "Dear [EMA" | "IL_1], your key [API_KEY_1_" | "BLOCKED] is active."
        texts = [
            "Dear [EMA",
            "IL_1], your key [API_KEY_1_",
            "BLOCKED] is active.",
        ]
        mock_sse_server.events = _make_claude_text_events(texts, delay_s=0.0)

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # ── (b) No placeholder in any chunk ──────────────────────────────────
        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw), (
            f"Placeholder token found in forwarded chunks:\n{combined_raw!r}"
        )

        # ── (c) Final text matches original ──────────────────────────────────
        assert full_text == _SPLIT_CHUNK_ORIGINAL, (
            f"Final text mismatch.\nExpected: {_SPLIT_CHUNK_ORIGINAL!r}\n"
            f"Got:      {full_text!r}"
        )

    # ── Scenario 3: Mixed split and unsplit with plain text ──────────────────

    def test_mixed_split_and_whole_placeholders_rehydrated(self, mock_sse_server):
        """
        SCENARIO 3 — Mixed: split + unsplit placeholders with plain text.

        The response interleaves:
        - Plain text segments (no placeholders)
        - Complete (unsplit) placeholder tokens
        - Placeholder tokens split across chunk boundaries

        All placeholders are rehydrated; the final text matches the original.

        Assertions (b) and (c) verified.
        """
        # Original: "Hi Alice Smith! Contact alice@corp.io or 010-1234-5678. "
        #            "The key sk-top-secret should not appear."
        # Masked:   "Hi [PERSON_1]! Contact [EMAIL_1] or [PHONE_1]. "
        #            "The key [API_KEY_1_BLOCKED] should not appear."
        #
        # Delivery:
        #  chunk 1: "Hi [PERSON_1]! Contact [EMA"       ← [PERSON_1] whole, [EMAIL split
        #  chunk 2: "IL_1] or [PHONE_1]. "               ← EMAIL_1 complete, [PHONE_1] whole
        #  chunk 3: "The key [API_KEY_1_"                ← start of BLOCKED token
        #  chunk 4: "BLOCKED] should not appear."        ← end of BLOCKED token
        texts = [
            "Hi [PERSON_1]! Contact [EMA",
            "IL_1] or [PHONE_1]. ",
            "The key [API_KEY_1_",
            "BLOCKED] should not appear.",
        ]
        mock_sse_server.events = _make_claude_text_events(texts, delay_s=0.0)

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # ── (b) No placeholder in any chunk ──────────────────────────────────
        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw), (
            f"Placeholder token found in forwarded chunks:\n{combined_raw!r}"
        )

        # ── (c) Final text matches original ──────────────────────────────────
        assert full_text == _MIXED_ORIGINAL, (
            f"Final text mismatch.\nExpected: {_MIXED_ORIGINAL!r}\n"
            f"Got:      {full_text!r}"
        )

    # ── Assertion (a): TTFT preserved ────────────────────────────────────────

    def test_ttft_preserved_first_chunk_before_full_response(self, mock_sse_server):
        """
        ASSERTION (a) — TTFT preserved.

        The first safe token passes through to the client before the full
        response body is received from the upstream.

        Setup: mock server sends plain text first event immediately, then waits
        _EVENT_DELAY_S before sending remaining events.  The proxy must forward
        the first rehydrated event within the delay window.
        """
        # Event 1: plain text (no placeholder) — safe immediately, no delay
        # Events 2+: more text with a significant delay before each
        first_text = "Hello, this is plain text. "
        subsequent_texts = [
            "More text here. ",
            "And some more at the end.",
        ]

        events: List[_SSEEvent] = []
        # Event 1: immediate (no delay)
        obj1 = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": first_text},
        }
        events.append(_SSEEvent(data=json.dumps(obj1), event_type="content_block_delta",
                                delay_before_s=0.0))

        # Events 2+: delayed
        for txt in subsequent_texts:
            obj = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": txt},
            }
            events.append(_SSEEvent(data=json.dumps(obj), event_type="content_block_delta",
                                    delay_before_s=_EVENT_DELAY_S))

        events.append(_SSEEvent(
            data=json.dumps({"type": "message_stop"}),
            event_type="message_stop",
            delay_before_s=0.0,
        ))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            first_latency, all_chunks, full_text, total_elapsed = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # ── (a) TTFT < total response time × threshold ────────────────────────
        assert first_latency is not None, "No chunks received"
        assert total_elapsed > 0, "Total elapsed must be positive"

        # With 2 delayed events at _EVENT_DELAY_S each, total ≥ 2 * _EVENT_DELAY_S.
        # First event is immediate (no delay), so first_latency should be well
        # below total_elapsed.
        assert first_latency < total_elapsed * _TTFT_FRACTION, (
            f"TTFT not preserved: first_chunk={first_latency:.3f}s, "
            f"total={total_elapsed:.3f}s (expected < {_TTFT_FRACTION:.0%} of total)"
        )

        # Also verify the first byte arrived well before the first delay:
        # first_latency should be much less than _EVENT_DELAY_S
        assert first_latency < _EVENT_DELAY_S, (
            f"First chunk latency {first_latency:.3f}s >= delay {_EVENT_DELAY_S}s — "
            f"proxy appears to be buffering the full response"
        )

        # ── (c) Content is correct ────────────────────────────────────────────
        expected = first_text + "".join(subsequent_texts)
        assert full_text == expected, (
            f"Content mismatch.\nExpected: {expected!r}\nGot:      {full_text!r}"
        )

    def test_ttft_preserved_with_placeholder_in_first_chunk(self, mock_sse_server):
        """
        TTFT is preserved even when the first chunk contains a complete placeholder.

        The placeholder is rehydrated immediately on arrival; the client sees the
        real value in the first chunk — not after buffering the whole response.
        """
        events = _make_claude_text_events(
            ["[EMAIL_1] is your contact."],
            delay_s=_EVENT_DELAY_S,
        )
        # Add more delayed events so total_elapsed is significantly > first_latency
        more = [
            "Additional text one. ",
            "Additional text two.",
        ]
        for txt in more:
            obj = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": txt},
            }
            events.insert(-1, _SSEEvent(  # insert before message_stop
                data=json.dumps(obj),
                event_type="content_block_delta",
                delay_before_s=_EVENT_DELAY_S,
            ))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            first_latency, all_chunks, full_text, total_elapsed = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # TTFT assertion
        assert first_latency is not None
        assert first_latency < total_elapsed * _TTFT_FRACTION, (
            f"TTFT not preserved: {first_latency:.3f}s vs {total_elapsed:.3f}s"
        )

        # Content assertion
        assert "alice@corp.io" in full_text
        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw)

    # ── Cross-provider streaming ──────────────────────────────────────────────

    def test_openai_streaming_whole_chunk(self, mock_sse_server):
        """
        OpenAI streaming: complete placeholder rehydrated, no token in output.

        Assertions (b) and (c).
        """
        original = "Your contact is alice@corp.io. Key: sk-top-secret."
        # Masked text split into OpenAI SSE events
        texts = [
            "Your contact is [EMAIL_1]. ",
            "Key: [API_KEY_1_BLOCKED].",
        ]
        events: List[_SSEEvent] = []
        for text in texts:
            obj = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            events.append(_SSEEvent(data=json.dumps(obj)))
        events.append(_SSEEvent(data="[DONE]"))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, "/v1/chat/completions", "openai"
            )

        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw), (
            f"Placeholder in OpenAI output:\n{combined_raw!r}"
        )
        assert full_text == original, (
            f"OpenAI text mismatch.\nExpected: {original!r}\nGot: {full_text!r}"
        )

    def test_openai_streaming_split_placeholder(self, mock_sse_server):
        """
        OpenAI streaming: placeholder split across chunks is reassembled
        and rehydrated correctly.

        Assertions (b) and (c).
        """
        original = "Email: alice@corp.io and phone: 010-1234-5678."
        texts = [
            "Email: [EMA",
            "IL_1] and phone: [PHO",
            "NE_1].",
        ]
        events: List[_SSEEvent] = []
        for text in texts:
            obj = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            events.append(_SSEEvent(data=json.dumps(obj)))
        events.append(_SSEEvent(data="[DONE]"))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, "/v1/chat/completions", "openai"
            )

        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw)
        assert full_text == original, (
            f"OpenAI split text mismatch.\nExpected: {original!r}\nGot: {full_text!r}"
        )

    def test_gemini_streaming_whole_chunk(self, mock_sse_server):
        """
        Gemini streaming: complete placeholder rehydrated, no token in output.
        """
        original = "Contact Alice Smith at alice@corp.io."
        texts = [
            "Contact [PERSON_1] at ",
            "[EMAIL_1].",
        ]
        events: List[_SSEEvent] = []
        for text in texts:
            obj = {
                "candidates": [{
                    "content": {"parts": [{"text": text}], "role": "model"}
                }]
            }
            events.append(_SSEEvent(data=json.dumps(obj)))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            gemini_path = "/v1beta/models/gemini-1.5-pro:streamGenerateContent"
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, gemini_path, "gemini"
            )

        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw)
        assert full_text == original, (
            f"Gemini text mismatch.\nExpected: {original!r}\nGot: {full_text!r}"
        )

    def test_gemini_streaming_split_placeholder(self, mock_sse_server):
        """
        Gemini streaming: placeholder split across chunks is reassembled.
        """
        original = "Name: Alice Smith, email: alice@corp.io."
        texts = [
            "Name: [PERSON_",
            "1], email: [EMAIL_",
            "1].",
        ]
        events: List[_SSEEvent] = []
        for text in texts:
            obj = {
                "candidates": [{
                    "content": {"parts": [{"text": text}], "role": "model"}
                }]
            }
            events.append(_SSEEvent(data=json.dumps(obj)))

        mock_sse_server.events = events

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            gemini_path = "/v1beta/models/gemini-1.5-pro:streamGenerateContent"
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, gemini_path, "gemini"
            )

        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw)
        assert full_text == original, (
            f"Gemini split text mismatch.\nExpected: {original!r}\nGot: {full_text!r}"
        )

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_non_streaming_response_not_affected(self, mock_sse_server):
        """
        A non-streaming (buffered JSON) response from the upstream is still
        correctly rehydrated by the existing buffered path (not streaming path).
        """
        # Configure mock to return a plain JSON response (not text/event-stream)
        class _PlainJSONHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args): pass
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                _ = self.rfile.read(length)
                body = json.dumps({
                    "content": [{"type": "text", "text": "Reply to [EMAIL_1]."}]
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), _PlainJSONHandler)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            proxy = _make_proxy_with_map(
                f"http://{host}:{port}", _INTEGRATION_RMAP
            )
            with proxy:
                import urllib.request as ureq
                body = json.dumps({
                    "model": "claude-opus-4-5",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                }).encode("utf-8")
                req = ureq.Request(
                    proxy.base_url + "/v1/messages",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with ureq.urlopen(req, timeout=5) as resp:
                    resp_body = resp.read()

            resp_json = json.loads(resp_body)
            text = resp_json["content"][0]["text"]
            assert "alice@corp.io" in text
            assert "[EMAIL_1]" not in text
        finally:
            server.shutdown()
            thread.join(timeout=3)

    def test_empty_stream_no_crash(self, mock_sse_server):
        """A stream that sends no text events does not crash the proxy."""
        mock_sse_server.events = [
            _SSEEvent(
                data=json.dumps({"type": "message_stop"}),
                event_type="message_stop",
            )
        ]

        proxy = _make_proxy_with_map(mock_sse_server.base_url, _INTEGRATION_RMAP)
        with proxy:
            _, all_chunks, full_text, _ = _read_streaming_response(
                proxy.base_url, self.CLAUDE_PATH, "claude"
            )

        # Should not crash; full_text may be empty or contain only metadata
        assert isinstance(full_text, str)
        combined_raw = b"".join(all_chunks).decode("utf-8", errors="replace")
        assert not _contains_placeholder(combined_raw)


# ─────────────────────────────────────────────────────────────────────────────
# 4. StreamingSSERehydrator — three mandated AC scenarios (pure unit)
# ─────────────────────────────────────────────────────────────────────────────

class TestACMandatedScenarios:
    """
    Direct verification of the three scenarios mandated by Sub-AC 9.2.

    These tests drive StreamingSSERehydrator without a network server to
    verify correctness at the rehydrator level independently of the proxy.
    """

    RMAP: Dict[str, str] = {
        "EMAIL_1": "alice@corp.io",
        "API_KEY_1_BLOCKED": "sk-secret",
        "PERSON_1": "Alice Smith",
    }

    def _claude_chunk(self, text: str) -> bytes:
        obj = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(obj)}\n\n".encode()

    def _get_all_text(self, raw_bytes: bytes) -> str:
        return _extract_all_text_from_sse_bytes(raw_bytes, "claude")

    # ── AC Scenario A: all placeholders whole in single chunks ────────────────

    def test_scenario_whole_chunk_no_placeholder_in_output(self):
        """
        Sub-AC 9.2 — SCENARIO A:
        Every placeholder is delivered whole in one chunk.
        (b) No placeholder in any emitted chunk.
        (c) Final content matches original.
        """
        rh = StreamingSSERehydrator(self.RMAP, "claude")

        # Chunks — each placeholder is complete in its own chunk
        chunks = [
            self._claude_chunk("Hello, "),
            self._claude_chunk("[PERSON_1]"),
            self._claude_chunk(". Email: "),
            self._claude_chunk("[EMAIL_1]"),
            self._claude_chunk(". Key: "),
            self._claude_chunk("[API_KEY_1_BLOCKED]"),
            self._claude_chunk("."),
        ]
        output = b""
        for chunk in chunks:
            output += rh.feed_chunk(chunk)
        output += rh.flush()

        # (b) No placeholder in any forwarded byte
        decoded = output.decode("utf-8", errors="replace")
        assert not _contains_placeholder(decoded), (
            f"Placeholder found in scenario A output:\n{decoded!r}"
        )

        # (c) Final text matches original
        text = self._get_all_text(output)
        expected = "Hello, Alice Smith. Email: alice@corp.io. Key: sk-secret."
        assert text == expected, f"Scenario A: {text!r} != {expected!r}"

    # ── AC Scenario B: all placeholders split across chunk boundaries ─────────

    def test_scenario_split_boundary_no_placeholder_in_output(self):
        """
        Sub-AC 9.2 — SCENARIO B:
        Every placeholder is split across chunk boundaries.
        (b) No placeholder in any emitted chunk.
        (c) Final content matches original.
        """
        rh = StreamingSSERehydrator(self.RMAP, "claude")

        # Split each placeholder at its midpoint
        chunks = [
            self._claude_chunk("Reply to [EMA"),       # [EMAIL_1] split
            self._claude_chunk("IL_1]. Key: [API_"),   # EMAIL_1 closed; API_KEY split
            self._claude_chunk("KEY_1_BLOCKED]. End."), # API_KEY closed
        ]
        output = b""
        for chunk in chunks:
            output += rh.feed_chunk(chunk)
        output += rh.flush()

        # (b) No placeholder in forwarded bytes
        decoded = output.decode("utf-8", errors="replace")
        assert not _contains_placeholder(decoded), (
            f"Placeholder found in scenario B output:\n{decoded!r}"
        )

        # (c) Final text
        text = self._get_all_text(output)
        expected = "Reply to alice@corp.io. Key: sk-secret. End."
        assert text == expected, f"Scenario B: {text!r} != {expected!r}"

    # ── AC Scenario C: mixed split and unsplit with plain text ────────────────

    def test_scenario_mixed_no_placeholder_in_output(self):
        """
        Sub-AC 9.2 — SCENARIO C:
        Mixed response: split and unsplit placeholders with plain text.
        (b) No placeholder in any emitted chunk.
        (c) Final content matches original.
        """
        rh = StreamingSSERehydrator(self.RMAP, "claude")

        # Chunk 1: plain text + start of split placeholder
        # Chunk 2: end of split placeholder + unsplit placeholder + plain text
        # Chunk 3: plain text only
        chunks = [
            self._claude_chunk("Hi [PERSON_1], see [EMA"),  # PERSON_1 whole, EMAIL split
            self._claude_chunk("IL_1] and [API_KEY_1_BLOCKED] now."),  # both closed
            self._claude_chunk(" That's all."),                         # plain text
        ]
        output = b""
        for chunk in chunks:
            output += rh.feed_chunk(chunk)
        output += rh.flush()

        # (b) No placeholder in forwarded bytes
        decoded = output.decode("utf-8", errors="replace")
        assert not _contains_placeholder(decoded), (
            f"Placeholder found in scenario C output:\n{decoded!r}"
        )

        # (c) Final text
        text = self._get_all_text(output)
        expected = (
            "Hi Alice Smith, see alice@corp.io and sk-secret now. That's all."
        )
        assert text == expected, f"Scenario C: {text!r} != {expected!r}"

    # ── AC Assertion (a): TTFT — first safe token before full response ────────

    def test_ttft_first_safe_token_emitted_before_full_response(self):
        """
        Sub-AC 9.2 — ASSERTION (a):
        First safe token passes through before full response received.

        Verified by showing that feed_chunk() on the first event returns
        non-empty output (i.e. the proxy doesn't buffer everything).
        This is a unit-level proof that TTFT is preserved by design.
        """
        rh = StreamingSSERehydrator(self.RMAP, "claude")

        # First chunk: plain text — immediately safe (no placeholder prefix)
        first_chunk = self._claude_chunk("Hello world. ")
        output_first = rh.feed_chunk(first_chunk)

        # The first feed must return content immediately — not wait for more chunks
        first_text = self._get_all_text(output_first)
        assert first_text == "Hello world. ", (
            f"Expected first chunk forwarded immediately. Got: {first_text!r}"
        )
        assert len(output_first) > 0, (
            "First chunk produced no output — proxy is buffering (TTFT not preserved)"
        )

        # Second chunk: more content (simulating a delayed second SSE event)
        second_chunk = self._claude_chunk("More content here.")
        output_second = rh.feed_chunk(second_chunk)
        output_second += rh.flush()

        second_text = self._get_all_text(output_second)
        assert second_text == "More content here."

    def test_ttft_preserved_when_first_chunk_ends_with_partial_placeholder(self):
        """
        When the first chunk ends with a partial placeholder, the plain text
        before the placeholder is still forwarded immediately (TTFT preserved
        for the non-placeholder content).
        """
        rh = StreamingSSERehydrator(self.RMAP, "claude")

        # "Hello [EMA" — "Hello " is safe, "[EMA" is held
        first_chunk = self._claude_chunk("Hello [EMA")
        output_first = rh.feed_chunk(first_chunk)

        # "Hello " should be forwarded immediately; "[EMA" is held
        first_text = self._get_all_text(output_first)
        assert "Hello" in first_text, (
            f"Plain text before partial placeholder must be forwarded immediately. "
            f"Got: {first_text!r}"
        )

        # Complete the placeholder
        second_chunk = self._claude_chunk("IL_1] there.")
        output_second = rh.feed_chunk(second_chunk)
        output_second += rh.flush()

        second_text = self._get_all_text(output_second)
        assert "alice@corp.io" in second_text

        # Overall text is correct
        total_text = self._get_all_text(output_first + output_second)
        assert total_text == "Hello alice@corp.io there."
        assert not _contains_placeholder(
            (output_first + output_second).decode("utf-8", errors="replace")
        )
