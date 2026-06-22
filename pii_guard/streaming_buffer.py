"""
PII-Guard streaming look-ahead buffer (Sub-AC 9.1).

Reassembles placeholder tokens that span SSE chunk boundaries without
buffering the full response, preserving streaming TTFT.

Problem Statement
-----------------
When the proxy masks outbound requests it replaces PII values with indexed
placeholder tokens such as ``[EMAIL_1]``, ``[PERSON_2]``, or
``[API_KEY_1_BLOCKED]``.  An LLM may echo those tokens back in its streaming
response.  Because an SSE stream delivers the response as a sequence of small
text chunks, a placeholder might be split across chunk boundaries:

    chunk A: "I will reply to [EMA"
    chunk B: "IL_1] shortly."

Without reassembly, a naïve rehydrator would see incomplete fragments and
fail to restore the original value.

Solution
--------
:class:`StreamingLookAheadBuffer` maintains a sliding window over the
accumulated text.  Each time a new chunk arrives it:

1. Appends the chunk to the internal buffer.
2. Locates the rightmost character sequence that could be the **incomplete
   leading prefix** of a placeholder token (any suffix of the buffer that
   starts with ``[`` followed by uppercase letters, digits, or underscores,
   and that does NOT yet contain the closing ``]``).
3. Emits everything before that position as "safe" text — guaranteed to
   contain no broken placeholders.
4. Retains the potentially incomplete prefix in the buffer until later chunks
   complete it (or show it is not actually a placeholder).

The buffer is **bounded**: when its size exceeds the configured maximum the
oldest content is force-emitted so that the buffer never grows without bound.
At most ``_MAX_PLACEHOLDER_LEN`` characters are retained after a force-emit.

Placeholder Token Grammar
--------------------------
A complete placeholder token matches::

    \\[[A-Z][A-Z0-9_]*_\\d+(?:_BLOCKED)?\\]

Examples::

    [EMAIL_1]           [PHONE_2]           [PERSON_10]
    [API_KEY_1_BLOCKED] [AWS_SECRET_3]      [PRIVATE_KEY_1_BLOCKED]

An *incomplete prefix* is any leading substring of such a token that does not
yet include the closing ``]`` — from a bare ``[`` up to just before the ``]``:

    [   [E   [EM   [EMAIL   [EMAIL_   [EMAIL_1   [EMAIL_1_BLOCKED

Usage::

    buf = StreamingLookAheadBuffer(max_buffer_size=512)

    for raw_chunk in sse_event_stream():
        safe_text = buf.feed(raw_chunk)
        if safe_text:
            yield rehydrate(safe_text)

    # Signal end-of-stream to flush all remaining buffered content
    tail = buf.flush()
    if tail:
        yield rehydrate(tail)
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum theoretical length of a placeholder token:
#   "[" + category (≤ 30 chars) + "_" + serial (≤ 6 digits)
#        + "_BLOCKED" (8 chars) + "]"  = ≤ 47 chars
# We use 64 as a generous upper bound to be safe.
_MAX_PLACEHOLDER_LEN: int = 64

# Minimum valid max_buffer_size: must hold at least two full placeholders
# so that back-to-back tokens near the boundary are handled correctly.
_MIN_BUFFER_SIZE: int = _MAX_PLACEHOLDER_LEN * 2  # 128


# ─────────────────────────────────────────────────────────────────────────────
# Helper: incomplete placeholder prefix detection
# ─────────────────────────────────────────────────────────────────────────────

def _could_be_placeholder_prefix(s: str) -> bool:
    """
    Return ``True`` if *s* is a plausible incomplete leading prefix of a
    placeholder token.

    A placeholder has the form ``[CATEGORY_N]`` or ``[CATEGORY_N_BLOCKED]``.
    A *prefix* is any non-empty leading substring of such a token that does
    NOT include the closing ``]``.

    Rules applied (in order):
    1. Must start with ``'['``.
    2. Must NOT contain a ``']'`` (a closed bracket means the token is
       complete, not incomplete).
    3. The character immediately after ``'['`` (if present) must be an
       uppercase ASCII letter (``[A-Z]``); a digit, lowercase letter, or
       punctuation cannot start a valid category name.
    4. All subsequent characters must be uppercase letters, ASCII digits, or
       underscores — the allowed alphabet of a placeholder body.
    5. The total length must not exceed ``_MAX_PLACEHOLDER_LEN`` (a longer
       prefix cannot be a real placeholder).

    Examples that return ``True``::

        "["          "[E"         "[EM"        "[EMAIL"
        "[EMAIL_"    "[EMAIL_1"   "[EMAIL_10"
        "[EMAIL_1_"  "[EMAIL_1_BLOCKED"
        "[AWS_SECRET_3"  "[PRIVATE_KEY_1_BLOCKED"

    Examples that return ``False``::

        ""                 # empty
        "hello"            # doesn't start with '['
        "[hello"           # lowercase after '['
        "[123"             # digit after '['
        "[EMAIL_1]"        # contains ']' → complete, not incomplete
        "[EMAIL_1_blocked" # lowercase 'b' in body
        "[" + "X" * 64    # exceeds max placeholder length
    """
    if not s:
        return False
    if s[0] != '[':
        return False
    if ']' in s:
        return False  # Already closed — this is a complete (or closed) token
    if len(s) > _MAX_PLACEHOLDER_LEN:
        return False

    body = s[1:]  # everything after the opening '['
    if not body:
        # Just a bare '[' — still could grow into a placeholder
        return True

    # First char of body must be uppercase letter
    if not ('A' <= body[0] <= 'Z'):
        return False

    # All remaining chars must be [A-Z0-9_]
    for ch in body[1:]:
        if not (('A' <= ch <= 'Z') or ('0' <= ch <= '9') or ch == '_'):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core buffer
# ─────────────────────────────────────────────────────────────────────────────

class StreamingLookAheadBuffer:
    """
    Bounded look-ahead buffer for streaming SSE placeholder reassembly.

    Maintains a sliding window of bounded size over incoming token text.
    Detects potential placeholder-token boundaries that may span chunk edges
    and emits safely-reassembled token sequences.

    The buffer guarantees:

    * **Correctness** — no complete placeholder is ever split at emission time.
    * **Boundedness** — the internal buffer never exceeds *max_buffer_size*
      characters regardless of chunk sizes or adversarial input.
    * **TTFT preservation** — the buffer holds back at most one potential
      placeholder prefix (≤ ``_MAX_PLACEHOLDER_LEN`` chars), so streaming
      latency is bounded and the first tokens are delivered as soon as the
      first safe chunk is emitted.
    * **No silent loss** — callers must call :meth:`flush` at stream end to
      retrieve any remaining buffered bytes; the buffer does not auto-discard.

    Parameters
    ----------
    max_buffer_size:
        Maximum number of characters the buffer may hold before force-emitting
        content.  Must be ≥ ``_MIN_BUFFER_SIZE`` (128).  Defaults to 512.

    Raises
    ------
    ValueError
        If *max_buffer_size* is below ``_MIN_BUFFER_SIZE``.

    Example
    -------
    ::

        buf = StreamingLookAheadBuffer()
        for chunk in sse_stream:
            safe = buf.feed(chunk)
            if safe:
                downstream.write(rehydrate(safe))
        tail = buf.flush()
        if tail:
            downstream.write(rehydrate(tail))
    """

    def __init__(self, max_buffer_size: int = 512) -> None:
        if max_buffer_size < _MIN_BUFFER_SIZE:
            raise ValueError(
                f"max_buffer_size must be >= {_MIN_BUFFER_SIZE} "
                f"(got {max_buffer_size})"
            )
        self._max: int = max_buffer_size
        self._buf: str = ""

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def buffer(self) -> str:
        """Current contents of the internal buffer (read-only snapshot)."""
        return self._buf

    @property
    def buffer_size(self) -> int:
        """Current number of characters held in the internal buffer."""
        return len(self._buf)

    # ── Public API ───────────────────────────────────────────────────────────

    def feed(self, chunk: str) -> str:
        """
        Accept *chunk* from the SSE stream and return the safely-emittable
        portion.

        The returned string may be empty if no safe bytes can be emitted yet
        (e.g. the entire buffer ends with what might be the start of a
        placeholder and no subsequent chunk has arrived to confirm or deny it).

        Parameters
        ----------
        chunk:
            Raw text from one SSE data line (already decoded from bytes).

        Returns
        -------
        str
            The portion of the accumulated buffer that is safe to emit.
            May be an empty string if nothing can be safely emitted yet.
        """
        self._buf += chunk
        return self._extract_safe()

    def flush(self) -> str:
        """
        Signal end-of-stream; return and clear all remaining buffered text.

        Must be called after the last chunk has been fed to ensure the tail
        of the stream is not silently discarded.  Any incomplete placeholder
        prefix at the tail is emitted verbatim (no further chunks can complete
        it at this point).

        Returns
        -------
        str
            All remaining buffered text (may be empty if the buffer was
            already empty).
        """
        remaining = self._buf
        self._buf = ""
        return remaining

    def reset(self) -> None:
        """
        Discard all buffered content without emitting it.

        Used on error, abort, or when starting a new SSE stream on the same
        buffer instance.
        """
        self._buf = ""

    # ── Internal mechanics ───────────────────────────────────────────────────

    def _extract_safe(self) -> str:
        """
        Compute and return the safe-to-emit prefix of the internal buffer,
        then update the buffer to contain only the held-back portion.

        **Force-emit path**: when the buffer size exceeds *_max*, we must
        emit content unconditionally to prevent unbounded growth.  We retain
        at most ``_MAX_PLACEHOLDER_LEN`` characters (enough to hold one full
        incomplete placeholder prefix) and emit everything before that.

        **Normal path**: locate the rightmost incomplete placeholder prefix
        (if any) and emit everything that precedes it.
        """
        buf = self._buf

        # ── Force-emit when buffer is oversized ──────────────────────────────
        if len(buf) > self._max:
            # Keep at most _MAX_PLACEHOLDER_LEN chars in the buffer so a
            # potential placeholder at the new tail isn't split mid-token.
            force_boundary = len(buf) - _MAX_PLACEHOLDER_LEN
            safe = buf[:force_boundary]
            self._buf = buf[force_boundary:]
            return safe

        # ── Normal path: find safe boundary ──────────────────────────────────
        safe_end = self._find_safe_boundary(buf)

        if safe_end == 0:
            # The entire buffer might be an incomplete placeholder prefix;
            # nothing can be safely emitted yet.
            return ""

        safe = buf[:safe_end]
        self._buf = buf[safe_end:]
        return safe

    def _find_safe_boundary(self, buf: str) -> int:
        """
        Return the index of the first character that should NOT yet be emitted
        because it is (or might be) part of an incomplete placeholder prefix.

        Scans backward through *buf* looking for a ``'['`` that begins a
        potential incomplete placeholder.  The first such position (counting
        from the right) is the hold boundary.  Everything before it is safe
        to emit.

        If no such position exists the entire buffer is safe and ``len(buf)``
        is returned.

        Algorithm
        ---------
        1. Find the rightmost ``'['`` in *buf*.
        2. Test whether the substring from that ``'['`` to the end of *buf*
           satisfies :func:`_could_be_placeholder_prefix`.
        3. If yes → that position is the hold boundary → return it.
        4. If no → that ``'['`` is not a placeholder start (e.g. it opens a
           markdown link, or a closed token); move the search window left of
           it and repeat from step 1.
        5. If no ``'['`` is found → return ``len(buf)`` (fully safe).
        """
        search_end = len(buf)

        while search_end > 0:
            idx = buf.rfind('[', 0, search_end)
            if idx < 0:
                # No '[' left in the search window — buffer is fully safe.
                break

            candidate = buf[idx:]
            if _could_be_placeholder_prefix(candidate):
                # This '[' begins a potential incomplete placeholder;
                # hold back from this position.
                return idx

            # This '[' is not a valid placeholder start (e.g. already closed,
            # lowercase body, etc.).  Continue scanning to the left.
            search_end = idx

        # Entire buffer is safe to emit.
        return len(buf)
