"""
PII-Guard streaming SSE rehydration pipeline (Sub-AC 9.2).

Wires the :class:`~pii_guard.streaming_buffer.StreamingLookAheadBuffer`
into the live SSE response stream and restores ``[CATEGORY_N]`` placeholder
tokens to their original PII/secret values before forwarding to the calling
agent, while preserving streaming TTFT.

Architecture
------------
The proxy sends masked outbound requests to the upstream LLM.  The LLM may
echo back placeholder tokens (e.g. ``[EMAIL_1]``, ``[API_KEY_1_BLOCKED]``)
in its streaming response.  This module processes each SSE chunk as it
arrives — without buffering the full response — and rehydrates placeholders
on the fly:

1. Parse the raw SSE bytes into complete SSE events.
2. For each text-bearing event, extract the text delta.
3. Feed the extracted text through :class:`StreamingLookAheadBuffer`.
4. Rehydrate the safe (fully-buffered) prefix using the restoration map.
5. Re-inject the rehydrated text into the SSE event.
6. Forward the modified SSE bytes to the client **immediately**.

TTFT Preservation
-----------------
The look-ahead buffer holds back at most one incomplete placeholder prefix
(≤ ``_MAX_PLACEHOLDER_LEN`` chars ≈ 64 chars).  Plain text that appears
before the first placeholder is emitted on the very first safe flush — the
client receives it without waiting for the full response.

No Block-Category Token Emission
---------------------------------
All extracted text passes through rehydration before being forwarded.  No
``[CATEGORY_N]`` or ``[CATEGORY_N_BLOCKED]`` token ever appears in the bytes
sent downstream to the agent.

Lossless Correctness
--------------------
The concatenation of all forwarded text-delta values equals the original
unmasked content that the LLM intended to produce.

Provider SSE Event Coverage
----------------------------
Claude  (``/v1/messages`` with ``stream: true``):
    ``content_block_delta`` events where ``delta.type == "text_delta"``.
    All other event types (``message_start``, ``message_stop``, etc.) are
    forwarded unchanged.

OpenAI  (``/v1/chat/completions`` with ``stream: true``):
    Streaming chunks with ``choices[*].delta.content``.
    The terminal ``[DONE]`` sentinel is forwarded unchanged.

Gemini  (``streamGenerateContent``):
    Streaming chunks with ``candidates[*].content.parts[*].text``.

Unknown / non-JSON events are forwarded verbatim.

Usage::

    rehydrator = StreamingSSERehydrator(
        restoration_map={"EMAIL_1": "alice@corp.io"},
        provider="claude",
    )

    for raw_chunk in upstream_sse_socket:
        output_bytes = rehydrator.feed_chunk(raw_chunk)
        if output_bytes:
            client_socket.write(output_bytes)

    tail = rehydrator.flush()
    if tail:
        client_socket.write(tail)
"""
from __future__ import annotations

import copy
import json
from typing import Dict, Iterator, List, Optional, Tuple

from .response_rehydrator import _rehydrate_str
from .streaming_buffer import StreamingLookAheadBuffer


# ─────────────────────────────────────────────────────────────────────────────
# Provider-specific text extraction and injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_claude_stream_text(event_obj: dict) -> Optional[str]:
    """
    Extract the text delta from a Claude ``content_block_delta`` event.

    Returns ``None`` for all other Claude event types (``message_start``,
    ``message_stop``, ``content_block_start``, ``content_block_stop``,
    ``message_delta``), which carry no text delta that needs rehydration.
    """
    if event_obj.get("type") != "content_block_delta":
        return None
    delta = event_obj.get("delta")
    if not isinstance(delta, dict):
        return None
    if delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return text if isinstance(text, str) else None


def _inject_claude_stream_text(event_obj: dict, text: str) -> dict:
    """Inject rehydrated *text* into a Claude ``content_block_delta`` event."""
    result = copy.deepcopy(event_obj)
    result["delta"]["text"] = text
    return result


def _extract_openai_stream_text(event_obj: dict) -> Optional[str]:
    """
    Extract the text content from an OpenAI streaming chunk.

    Looks at ``choices[0].delta.content``.  Returns ``None`` if the chunk
    does not carry a text delta (e.g. role-only delta, finish_reason chunk).
    """
    choices = event_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) else None


def _inject_openai_stream_text(event_obj: dict, text: str) -> dict:
    """Inject rehydrated *text* into an OpenAI streaming chunk."""
    result = copy.deepcopy(event_obj)
    result["choices"][0]["delta"]["content"] = text
    return result


def _extract_gemini_stream_text(event_obj: dict) -> Optional[str]:
    """
    Extract the text from a Gemini streaming chunk.

    Looks at ``candidates[0].content.parts[0].text``.
    """
    candidates = event_obj.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    text = parts[0].get("text")
    return text if isinstance(text, str) else None


def _inject_gemini_stream_text(event_obj: dict, text: str) -> dict:
    """Inject rehydrated *text* into a Gemini streaming chunk."""
    result = copy.deepcopy(event_obj)
    result["candidates"][0]["content"]["parts"][0]["text"] = text
    return result


def _extract_text_from_event(
    event_obj: dict,
    provider: Optional[str],
) -> Optional[str]:
    """Dispatch text extraction to the appropriate provider handler."""
    if provider == "claude":
        return _extract_claude_stream_text(event_obj)
    elif provider == "openai":
        return _extract_openai_stream_text(event_obj)
    elif provider == "gemini":
        return _extract_gemini_stream_text(event_obj)
    return None


def _inject_text_into_event(
    event_obj: dict,
    text: str,
    provider: Optional[str],
) -> dict:
    """Dispatch text injection to the appropriate provider handler."""
    if provider == "claude":
        return _inject_claude_stream_text(event_obj, text)
    elif provider == "openai":
        return _inject_openai_stream_text(event_obj, text)
    elif provider == "gemini":
        return _inject_gemini_stream_text(event_obj, text)
    # Unknown provider — return a deep copy unchanged
    return copy.deepcopy(event_obj)


# ─────────────────────────────────────────────────────────────────────────────
# Internal SSE event parser
# ─────────────────────────────────────────────────────────────────────────────

class _SSEParser:
    """
    Accumulates raw bytes from an SSE stream and yields complete SSE events.

    An SSE event is delimited by a double-newline (``\\n\\n`` or
    ``\\r\\n\\r\\n``).  Partial events are held in the internal string buffer
    until the delimiter arrives.

    Yields tuples of ``(event_type, data_content, raw_event_text)`` where:

    * ``event_type`` — value of the ``event:`` line, or ``None`` if absent.
    * ``data_content`` — value of the ``data:`` line(s), concatenated.
    * ``raw_event_text`` — the original raw SSE event string, used for
      pass-through when no modification is needed.
    """

    # Both CRLF and LF double-newlines are valid SSE event separators.
    _SEPARATORS: Tuple[str, ...] = ("\r\n\r\n", "\n\n")

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, raw_bytes: bytes) -> Iterator[Tuple[Optional[str], str, str]]:
        """
        Accept a chunk of raw SSE bytes and yield any complete events found.

        Parameters
        ----------
        raw_bytes:
            Raw bytes from the upstream connection (UTF-8 encoded).

        Yields
        ------
        (event_type, data_content, raw_event_text)
        """
        self._buf += raw_bytes.decode("utf-8", errors="replace")
        yield from self._drain()

    def flush(self) -> Iterator[Tuple[Optional[str], str, str]]:
        """
        Drain any remaining partial event at end-of-stream.

        A partial event (e.g. the last event without a trailing ``\\n\\n``) is
        yielded verbatim.
        """
        # Try to drain any complete events first
        yield from self._drain()
        # Emit any leftover content as a final partial event
        remaining = self._buf.strip()
        if remaining:
            self._buf = ""
            event_type, data = self._parse_event(remaining)
            yield (event_type, data, remaining)

    def _drain(self) -> Iterator[Tuple[Optional[str], str, str]]:
        """Extract and yield all complete SSE events from the internal buffer."""
        while True:
            found = False
            for sep in self._SEPARATORS:
                idx = self._buf.find(sep)
                if idx >= 0:
                    raw_event = self._buf[:idx + len(sep)]
                    self._buf = self._buf[idx + len(sep):]
                    event_type, data = self._parse_event(raw_event)
                    yield (event_type, data, raw_event)
                    found = True
                    break
            if not found:
                break

    @staticmethod
    def _parse_event(raw_event: str) -> Tuple[Optional[str], str]:
        """
        Parse a raw SSE event string into ``(event_type, data_content)``.

        Handles multi-line ``data:`` fields by joining with ``\\n``.
        """
        event_type: Optional[str] = None
        data_parts: List[str] = []

        for line in raw_event.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].strip())
            elif line.startswith(":"):
                pass  # SSE comment — skip
            # Blank lines and other lines are ignored here

        return event_type, "\n".join(data_parts)


# ─────────────────────────────────────────────────────────────────────────────
# StreamingSSERehydrator — public API
# ─────────────────────────────────────────────────────────────────────────────

class StreamingSSERehydrator:
    """
    Streaming SSE rehydration pipeline.

    Wires :class:`~pii_guard.streaming_buffer.StreamingLookAheadBuffer` into
    the live SSE response stream from an upstream LLM and restores
    ``[CATEGORY_N]`` placeholder tokens to their original PII/secret values
    before forwarding to the calling agent.

    Guarantees
    ----------
    * **TTFT preserved** — plain text before the first placeholder is
      forwarded on the first safe emission, without waiting for the full
      response.  The buffer withholds at most one incomplete placeholder
      prefix (≤ 64 chars).
    * **No block-category token emission** — all text passes through
      rehydration before forwarding; no ``[CATEGORY_N]`` or
      ``[CATEGORY_N_BLOCKED]`` token appears in the forwarded SSE bytes.
    * **Lossless** — the concatenation of all text deltas in the forwarded
      events equals the original unmasked content.

    Parameters
    ----------
    restoration_map:
        ``{placeholder_token: original_value}`` mapping from the proxy's
        session state (e.g. ``{"EMAIL_1": "alice@corp.io"}``).
    provider:
        ``"claude"``, ``"openai"``, ``"gemini"``, or ``None``.
    max_buffer_size:
        Maximum characters the look-ahead buffer may hold before
        force-emitting.  Must be ≥ 128 (see
        :data:`~pii_guard.streaming_buffer._MIN_BUFFER_SIZE`).
        Defaults to 512.

    Usage::

        rehydrator = StreamingSSERehydrator(
            restoration_map={"EMAIL_1": "alice@corp.io"},
            provider="claude",
        )
        for raw_chunk in upstream_connection:
            output = rehydrator.feed_chunk(raw_chunk)
            if output:
                client.write(output)
        tail = rehydrator.flush()
        if tail:
            client.write(tail)
    """

    def __init__(
        self,
        restoration_map: Dict[str, str],
        provider: Optional[str],
        max_buffer_size: int = 512,
    ) -> None:
        self._map: Dict[str, str] = restoration_map
        self._provider: Optional[str] = provider
        self._look_ahead: StreamingLookAheadBuffer = StreamingLookAheadBuffer(
            max_buffer_size=max_buffer_size
        )
        self._sse_parser: _SSEParser = _SSEParser()

    # ── Public API ───────────────────────────────────────────────────────────

    def feed_chunk(self, raw_bytes: bytes) -> bytes:
        """
        Process a raw SSE chunk from the upstream and return bytes to forward.

        Extracts text from each complete SSE event, passes it through the
        look-ahead buffer, rehydrates safe text, and re-serializes the event.

        Text-bearing events are emitted only when safe text is available.
        When the buffer is holding an incomplete placeholder prefix, the text
        field of the current event is set to the safe prefix (possibly empty)
        and the remainder is held for the next chunk.

        Non-text events (e.g. Claude ``message_start``, OpenAI ``[DONE]``)
        are forwarded verbatim.

        Parameters
        ----------
        raw_bytes:
            Raw SSE bytes from the upstream LLM connection.

        Returns
        -------
        bytes
            SSE bytes ready to forward to the calling agent.
            May be ``b""`` if the buffer is holding all content pending a
            later chunk (e.g. the chunk ends mid-placeholder).
        """
        output_parts: List[bytes] = []
        for event_type, data, raw_event in self._sse_parser.feed(raw_bytes):
            part = self._process_event(event_type, data, raw_event)
            if part:
                output_parts.append(part)
        return b"".join(output_parts)

    def flush(self) -> bytes:
        """
        Flush remaining content at end-of-stream.

        Drains both the SSE parser and the look-ahead buffer, emitting any
        remaining content as a final synthetic SSE text event.  Must be
        called after the last upstream chunk has been processed.

        Returns
        -------
        bytes
            Any remaining SSE bytes to forward (may be ``b""``).
        """
        output_parts: List[bytes] = []

        # Drain remaining partial SSE events from the parser
        for event_type, data, raw_event in self._sse_parser.flush():
            part = self._process_event(event_type, data, raw_event)
            if part:
                output_parts.append(part)

        # Flush the look-ahead buffer (may hold an incomplete placeholder prefix)
        remaining_text = self._look_ahead.flush()
        if remaining_text:
            rehydrated, _ = _rehydrate_str(remaining_text, self._map)
            if rehydrated:
                output_parts.append(self._make_synthetic_text_event(rehydrated))

        return b"".join(output_parts)

    # ── Internal processing ──────────────────────────────────────────────────

    def _process_event(
        self,
        event_type: Optional[str],
        data: str,
        raw_event: str,
    ) -> bytes:
        """
        Process one complete SSE event and return bytes to forward.

        Text-bearing events are passed through the look-ahead buffer and
        rehydrated.  Non-text events are forwarded verbatim.
        """
        # ── Pass-through: OpenAI [DONE] sentinel ──────────────────────────────
        if data == "[DONE]":
            return raw_event.encode("utf-8")

        # ── Attempt to parse data as JSON ─────────────────────────────────────
        if not data:
            # Empty data line — pass through (e.g. keep-alive)
            return raw_event.encode("utf-8")

        try:
            event_obj = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # Non-JSON data — pass through unchanged
            return raw_event.encode("utf-8")

        if not isinstance(event_obj, dict):
            return raw_event.encode("utf-8")

        # ── Extract text from provider-specific fields ─────────────────────────
        text = _extract_text_from_event(event_obj, self._provider)

        if text is None:
            # Non-text event (message_start, message_stop, etc.) — pass through
            return raw_event.encode("utf-8")

        # ── Feed text through look-ahead buffer ───────────────────────────────
        safe_text = self._look_ahead.feed(text)

        if not safe_text:
            # The buffer is holding all content (incomplete placeholder prefix).
            # Emit this event with an empty text field so the event position in
            # the stream is preserved but no placeholder fragment is leaked.
            modified = _inject_text_into_event(event_obj, "", self._provider)
            serialized = self._serialize_event(event_type, modified)
            # Only forward if there's actual structural content in the event
            # (skip pure empty-text noise events for cleanliness).
            # We DO emit to maintain event ordering for Claude index tracking.
            return serialized

        # ── Rehydrate safe text ───────────────────────────────────────────────
        rehydrated, _ = _rehydrate_str(safe_text, self._map)

        # ── Re-inject rehydrated text and forward ─────────────────────────────
        modified = _inject_text_into_event(event_obj, rehydrated, self._provider)
        return self._serialize_event(event_type, modified)

    def _serialize_event(
        self,
        event_type: Optional[str],
        event_obj: dict,
    ) -> bytes:
        """Serialize an SSE event dict to bytes."""
        data_str = json.dumps(event_obj, ensure_ascii=False)
        if event_type:
            return f"event: {event_type}\ndata: {data_str}\n\n".encode("utf-8")
        return f"data: {data_str}\n\n".encode("utf-8")

    def _make_synthetic_text_event(self, text: str) -> bytes:
        """
        Create a synthetic provider-appropriate SSE text event for tail content
        flushed from the look-ahead buffer at end-of-stream.
        """
        if self._provider == "claude":
            obj = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }
            return (
                f"event: content_block_delta\n"
                f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
            ).encode("utf-8")

        elif self._provider == "openai":
            obj = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

        elif self._provider == "gemini":
            obj = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    }
                }]
            }
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

        else:
            # Generic fallback — use a simple JSON wrapper
            obj = {"text": text}
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
