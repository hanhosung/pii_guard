"""
Unit tests for Sub-AC 9.1 — bounded look-ahead buffer module.

Tests verify that :class:`StreamingLookAheadBuffer` correctly reassembles
placeholder tokens that span SSE chunk boundaries.

Test coverage (matching the five AC-mandated scenarios plus edge cases):

  1. Single-chunk throughput
     ─ An SSE chunk with no placeholder boundaries emits its full content
       immediately; no bytes are held back.

  2. Placeholder split exactly on a chunk boundary
     ─ ``[EMAIL_1]`` split into exactly two chunks reassembles to the
       complete token byte-for-byte; neither half leaks out incomplete.

  3. Placeholder split across three or more consecutive chunks
     ─ A placeholder delivered one character at a time across N ≥ 3 chunks
       is held back until the closing ``']'`` arrives, then the full token
       is emitted in a single safe burst.

  4. Back-to-back placeholders with a split between them
     ─ ``[EMAIL_1][PERSON_2]`` split between the two tokens correctly emits
       the first token as soon as it is complete and holds the second until
       it closes.

  5. Chunk containing only partial placeholder bytes with no safe-emit
     ─ When the entire buffer is a potential placeholder prefix (e.g. the
       only content ever fed is ``"[EMAI"``), ``feed()`` returns an empty
       string and ``buffer_size`` is non-zero.

Additional edge-case and regression tests ensure:
  - Buffer never exceeds its configured maximum size.
  - ``flush()`` always drains the buffer completely.
  - ``reset()`` discards buffered content.
  - Multi-character text between complete tokens emits correctly.
  - ``_could_be_placeholder_prefix`` helper correctly classifies prefixes.
  - ``ValueError`` is raised for an undersized max_buffer_size.
  - A lone ``[`` is held back (it could start a placeholder).
  - A ``[`` followed by lowercase is NOT held back.
"""
from __future__ import annotations

import pytest

from pii_guard.streaming_buffer import (
    StreamingLookAheadBuffer,
    _MAX_PLACEHOLDER_LEN,
    _MIN_BUFFER_SIZE,
    _could_be_placeholder_prefix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: reassemble all output from a sequence of chunks
# ─────────────────────────────────────────────────────────────────────────────

def _feed_all(buf: StreamingLookAheadBuffer, chunks: list) -> str:
    """Feed every chunk into *buf* and collect all emitted output including flush."""
    out = ""
    for chunk in chunks:
        out += buf.feed(chunk)
    out += buf.flush()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. _could_be_placeholder_prefix helper
# ─────────────────────────────────────────────────────────────────────────────

class TestCouldBePlaceholderPrefix:
    """Unit tests for the _could_be_placeholder_prefix predicate."""

    # ── True cases (valid incomplete prefixes) ────────────────────────────────

    def test_bare_open_bracket(self):
        assert _could_be_placeholder_prefix("[") is True

    def test_bracket_plus_one_uppercase(self):
        assert _could_be_placeholder_prefix("[E") is True

    def test_category_partial(self):
        assert _could_be_placeholder_prefix("[EMA") is True

    def test_full_category_word(self):
        assert _could_be_placeholder_prefix("[EMAIL") is True

    def test_category_with_trailing_underscore(self):
        assert _could_be_placeholder_prefix("[EMAIL_") is True

    def test_category_with_partial_number(self):
        assert _could_be_placeholder_prefix("[EMAIL_1") is True

    def test_category_with_multidigit_number(self):
        assert _could_be_placeholder_prefix("[EMAIL_10") is True

    def test_blocked_variant_partial(self):
        assert _could_be_placeholder_prefix("[EMAIL_1_BLOCKED") is True

    def test_compound_category(self):
        assert _could_be_placeholder_prefix("[AWS_SECRET_3") is True

    def test_private_key_blocked_partial(self):
        assert _could_be_placeholder_prefix("[PRIVATE_KEY_1_BLOCKED") is True

    def test_korean_rrn_partial(self):
        assert _could_be_placeholder_prefix("[RRN_1") is True

    def test_api_key_partial(self):
        assert _could_be_placeholder_prefix("[API_KEY_1") is True

    # ── False cases (not valid incomplete prefixes) ──────────────────────────

    def test_empty_string(self):
        assert _could_be_placeholder_prefix("") is False

    def test_no_bracket(self):
        assert _could_be_placeholder_prefix("hello") is False

    def test_complete_token_not_a_prefix(self):
        # A complete token (has closing ']') is NOT an incomplete prefix
        assert _could_be_placeholder_prefix("[EMAIL_1]") is False

    def test_lowercase_after_bracket(self):
        assert _could_be_placeholder_prefix("[hello") is False

    def test_digit_after_bracket(self):
        assert _could_be_placeholder_prefix("[123") is False

    def test_space_after_bracket(self):
        assert _could_be_placeholder_prefix("[ EMAIL") is False

    def test_lowercase_in_body(self):
        assert _could_be_placeholder_prefix("[EMAIL_1_blocked") is False

    def test_special_char_in_body(self):
        assert _could_be_placeholder_prefix("[EMAIL@1") is False

    def test_exceeds_max_length(self):
        # A string longer than _MAX_PLACEHOLDER_LEN cannot be a placeholder prefix
        long_prefix = "[" + "A" * _MAX_PLACEHOLDER_LEN
        assert _could_be_placeholder_prefix(long_prefix) is False

    def test_exactly_at_max_length_is_false(self):
        # _MAX_PLACEHOLDER_LEN includes the '[' so body must be _MAX_PLACEHOLDER_LEN-1
        # len(s) > _MAX_PLACEHOLDER_LEN → False; len(s) == _MAX_PLACEHOLDER_LEN → True
        at_limit = "[" + "A" * (_MAX_PLACEHOLDER_LEN - 1)
        assert len(at_limit) == _MAX_PLACEHOLDER_LEN
        assert _could_be_placeholder_prefix(at_limit) is True

    def test_one_over_max_length_is_false(self):
        over_limit = "[" + "A" * _MAX_PLACEHOLDER_LEN  # len = _MAX_PLACEHOLDER_LEN + 1
        assert _could_be_placeholder_prefix(over_limit) is False

    def test_markdown_link_prefix_false(self):
        # "[link text]" style — lowercase after bracket
        assert _could_be_placeholder_prefix("[link") is False

    def test_closed_token_with_garbage_after(self):
        assert _could_be_placeholder_prefix("[EMAIL_1] ") is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. StreamingLookAheadBuffer — construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBufferConstruction:

    def test_default_max_is_512(self):
        buf = StreamingLookAheadBuffer()
        assert buf._max == 512

    def test_custom_max_accepted(self):
        buf = StreamingLookAheadBuffer(max_buffer_size=_MIN_BUFFER_SIZE)
        assert buf._max == _MIN_BUFFER_SIZE

    def test_undersized_max_raises(self):
        with pytest.raises(ValueError, match="max_buffer_size must be >="):
            StreamingLookAheadBuffer(max_buffer_size=_MIN_BUFFER_SIZE - 1)

    def test_initial_buffer_is_empty(self):
        buf = StreamingLookAheadBuffer()
        assert buf.buffer == ""
        assert buf.buffer_size == 0

    def test_zero_max_raises(self):
        with pytest.raises(ValueError):
            StreamingLookAheadBuffer(max_buffer_size=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AC-mandated scenario 1: Single-chunk throughput
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleChunkThroughput:
    """
    AC scenario 1 — A chunk with no placeholder boundaries (no '[' or only
    closed complete tokens) emits its full content immediately without holding
    anything back.
    """

    def test_plain_text_emits_fully(self):
        buf = StreamingLookAheadBuffer()
        result = buf.feed("Hello, world!")
        assert result == "Hello, world!"
        assert buf.buffer_size == 0

    def test_complete_placeholder_emits_fully(self):
        """A chunk containing a single complete placeholder is safe and emits fully."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("Reply to [EMAIL_1] immediately.")
        assert result == "Reply to [EMAIL_1] immediately."
        assert buf.buffer_size == 0

    def test_multiple_complete_placeholders_emit_fully(self):
        """Multiple complete placeholders in one chunk all emit immediately."""
        buf = StreamingLookAheadBuffer()
        text = "Contact [EMAIL_1] or [PHONE_2] via [PERSON_3]."
        result = buf.feed(text)
        assert result == text
        assert buf.buffer_size == 0

    def test_empty_chunk_emits_empty(self):
        buf = StreamingLookAheadBuffer()
        result = buf.feed("")
        assert result == ""
        assert buf.buffer_size == 0

    def test_whitespace_only_chunk_emits_fully(self):
        buf = StreamingLookAheadBuffer()
        result = buf.feed("   \n\t  ")
        assert result == "   \n\t  "
        assert buf.buffer_size == 0

    def test_chunk_with_closed_bracket_only_emits_fully(self):
        """A ']' without a preceding '[' is plain text, fully safe."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("end of something]")
        assert result == "end of something]"

    def test_markdown_link_in_single_chunk_emits_fully(self):
        """[link text](url) style — not a placeholder, emits fully."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("See [this link](https://example.com) for details.")
        assert result == "See [this link](https://example.com) for details."
        assert buf.buffer_size == 0

    def test_flush_after_single_clean_chunk_returns_empty(self):
        buf = StreamingLookAheadBuffer()
        buf.feed("All safe text here.")
        tail = buf.flush()
        assert tail == ""

    def test_output_is_byte_for_byte_correct_single_chunk(self):
        """Emitted output must exactly equal the input for plain text."""
        buf = StreamingLookAheadBuffer()
        text = "The quick brown fox jumps over the lazy dog."
        emitted = buf.feed(text)
        assert emitted == text

    def test_unicode_text_in_single_chunk(self):
        """Unicode characters outside ASCII are treated as plain text."""
        buf = StreamingLookAheadBuffer()
        text = "안녕하세요 [EMAIL_1] 입니다."
        result = buf.feed(text)
        assert result == text
        assert buf.buffer_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. AC-mandated scenario 2: Placeholder split exactly on a chunk boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaceholderSplitExactlyOnBoundary:
    """
    AC scenario 2 — ``[EMAIL_1]`` is split into exactly two chunks; the split
    falls somewhere between the opening ``[`` and the closing ``]``.

    In each sub-test the emitted output must be byte-for-byte equal to the
    concatenation of all chunks; no content must be lost or duplicated.
    """

    @pytest.mark.parametrize("split_at", [
        1,   # "[" | "EMAIL_1] text"
        2,   # "[E" | "MAIL_1] text"
        3,   # "[EM" | "AIL_1] text"
        4,   # "[EMA" | "IL_1] text"
        5,   # "[EMAI" | "L_1] text"
        6,   # "[EMAIL" | "_1] text"
        7,   # "[EMAIL_" | "1] text"
        8,   # "[EMAIL_1" | "] text"
        9,   # "[EMAIL_1]" | " text"
    ])
    def test_email_split_at_every_position(self, split_at: int):
        """Placeholder split at every character position within the token."""
        full = "[EMAIL_1] text after"
        chunk_a = full[:split_at]
        chunk_b = full[split_at:]

        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, [chunk_a, chunk_b])
        assert total == full, (
            f"split_at={split_at}: expected {full!r}, got {total!r}"
        )

    def test_split_exactly_after_open_bracket(self):
        """Split right after the opening '[' — most minimal prefix case."""
        buf = StreamingLookAheadBuffer()
        emitted_a = buf.feed("prefix [")
        # 'prefix ' is safe; '[' is held back
        assert emitted_a == "prefix "
        assert buf.buffer == "["

        emitted_b = buf.feed("EMAIL_1] suffix")
        # Now we have "[EMAIL_1] suffix" — all safe to emit
        assert emitted_b == "[EMAIL_1] suffix"
        assert buf.buffer_size == 0
        tail = buf.flush()
        assert tail == ""

    def test_split_in_category_middle(self):
        """Split in the middle of the category name: '[EMA' | 'IL_1] rest'."""
        buf = StreamingLookAheadBuffer()
        emitted_a = buf.feed("[EMA")
        assert emitted_a == ""  # entire buffer is a placeholder prefix
        assert buf.buffer == "[EMA"

        emitted_b = buf.feed("IL_1] rest")
        assert "[EMAIL_1]" in emitted_b or emitted_b + buf.flush() == "[EMAIL_1] rest"
        # Full round-trip must be correct
        total = emitted_a + emitted_b + buf.flush()
        assert total == "[EMAIL_1] rest"

    def test_split_after_underscore(self):
        """Split after the underscore: '[EMAIL_' | '1] after'."""
        buf = StreamingLookAheadBuffer()
        emitted_a = buf.feed("[EMAIL_")
        assert emitted_a == ""
        assert buf.buffer == "[EMAIL_"

        emitted_b = buf.feed("1] after")
        total = emitted_a + emitted_b + buf.flush()
        assert total == "[EMAIL_1] after"

    def test_split_after_digit(self):
        """Split after the digit: '[EMAIL_1' | '] suffix'."""
        buf = StreamingLookAheadBuffer()
        emitted_a = buf.feed("[EMAIL_1")
        assert emitted_a == ""
        assert buf.buffer == "[EMAIL_1"

        emitted_b = buf.feed("] suffix")
        total = emitted_a + emitted_b + buf.flush()
        assert total == "[EMAIL_1] suffix"

    def test_split_blocked_variant(self):
        """[API_KEY_1_BLOCKED] split between '_BLOCKED' and ']'."""
        token = "[API_KEY_1_BLOCKED]"
        for split_at in range(1, len(token)):
            buf = StreamingLookAheadBuffer()
            total = _feed_all(buf, [token[:split_at], token[split_at:]])
            assert total == token, (
                f"BLOCKED split_at={split_at}: got {total!r}"
            )

    def test_preamble_before_split_placeholder(self):
        """Text before the split token is emitted on the first chunk."""
        buf = StreamingLookAheadBuffer()
        emitted_a = buf.feed("Hello, ")
        assert emitted_a == "Hello, "  # plain text — safe immediately

        emitted_b = buf.feed("[EMAIL_")
        assert emitted_b == ""  # placeholder prefix held

        emitted_c = buf.feed("1] world")
        total = emitted_a + emitted_b + emitted_c + buf.flush()
        assert total == "Hello, [EMAIL_1] world"

    def test_suffix_after_complete_token_emits_correctly(self):
        """Text after the closing ']' emits with the completed token."""
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, ["[PHONE_", "2] call me"])
        assert total == "[PHONE_2] call me"

    def test_buffer_is_empty_after_split_resolved(self):
        """Once the closing ']' arrives the buffer should be drained."""
        buf = StreamingLookAheadBuffer()
        buf.feed("before [EMAIL_")
        buf.feed("1]")
        assert buf.buffer_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. AC-mandated scenario 3: Placeholder split across 3+ consecutive chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaceholderSplitAcrossThreePlusChunks:
    """
    AC scenario 3 — A single placeholder is delivered one or a few characters
    at a time across three or more consecutive chunks.  The buffer must hold
    back all fragments until the closing ']' arrives, then emit the complete
    token with the correct bytes.
    """

    def test_three_chunk_split(self):
        """[EMAIL_1] in three chunks: '[EMA', 'IL_', '1] tail'."""
        buf = StreamingLookAheadBuffer()

        e1 = buf.feed("[EMA")
        assert e1 == ""
        assert buf.buffer == "[EMA"

        e2 = buf.feed("IL_")
        assert e2 == ""
        assert buf.buffer == "[EMAIL_"

        e3 = buf.feed("1] tail")
        total = e1 + e2 + e3 + buf.flush()
        assert total == "[EMAIL_1] tail"

    def test_character_by_character_split(self):
        """[EMAIL_1] delivered one character at a time across 9 chunks."""
        token = "[EMAIL_1]"
        chunks = list(token)  # ['[', 'E', 'M', 'A', 'I', 'L', '_', '1', ']']
        assert len(chunks) == 9

        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == token

    def test_person_token_four_chunk_split(self):
        """[PERSON_2] split into four chunks."""
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, ["[PER", "SON", "_", "2]"])
        assert total == "[PERSON_2]"

    def test_blocked_token_many_chunks(self):
        """[PRIVATE_KEY_1_BLOCKED] delivered in 5 roughly equal chunks."""
        token = "[PRIVATE_KEY_1_BLOCKED]"
        size = len(token)
        n = 5
        chunk_size = size // n
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n - 1 else size
            chunks.append(token[start:end])
        assert "".join(chunks) == token

        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == token

    def test_three_chunks_with_preamble(self):
        """Text before the placeholder is emitted before the split fragments."""
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("Hello [EMA")
        assert e1 == "Hello "   # preamble is safe; '[EMA' is held
        assert buf.buffer == "[EMA"

        e2 = buf.feed("IL_")
        assert e2 == ""
        assert buf.buffer == "[EMAIL_"

        e3 = buf.feed("1] world")
        total = e1 + e2 + e3 + buf.flush()
        assert total == "Hello [EMAIL_1] world"

    def test_buffer_does_not_exceed_max_during_multi_chunk_split(self):
        """The internal buffer must never exceed max_buffer_size."""
        max_size = _MIN_BUFFER_SIZE * 2
        buf = StreamingLookAheadBuffer(max_buffer_size=max_size)

        # Simulate a short placeholder split across many chunks
        token = "[EMAIL_1]"
        for ch in token:
            buf.feed(ch)
            assert buf.buffer_size <= max_size, (
                f"Buffer exceeded max ({max_size}): size={buf.buffer_size}"
            )

    def test_long_placeholder_multi_chunk(self):
        """A compound category like [AWS_SECRET_KEY_3] split across chunks."""
        token = "[AWS_SECRET_KEY_3]"
        mid = len(token) // 2
        chunks = [token[:3], token[3:mid], token[mid:]]
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == token

    def test_output_byte_for_byte_correct_multi_chunk(self):
        """
        Regardless of how a placeholder is split, the concatenation of all
        emitted text must equal the original input byte-for-byte.
        """
        original = "Start [PERSON_5] middle [PHONE_2] end"
        # Split the original into 7 chunks of varying sizes
        import textwrap
        pieces = []
        i = 0
        sizes = [3, 8, 5, 4, 7, 6, len(original)]  # last size overshoots → clipped
        for s in sizes:
            chunk = original[i:i + s]
            if not chunk:
                break
            pieces.append(chunk)
            i += s
            if i >= len(original):
                break

        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, pieces)
        assert total == original


# ─────────────────────────────────────────────────────────────────────────────
# 6. AC-mandated scenario 4: Back-to-back placeholders with a split between them
# ─────────────────────────────────────────────────────────────────────────────

class TestBackToBackPlaceholdersWithSplit:
    """
    AC scenario 4 — Two placeholders appear consecutively or close together
    and the chunk boundary falls between them.  The first complete token must
    be emitted before the second begins, and neither token must appear
    corrupted.
    """

    def test_adjacent_placeholders_split_between(self):
        """
        '[EMAIL_1][PHONE_2]' split right between the two tokens:
        chunk A = '[EMAIL_1]'
        chunk B = '[PHONE_2]'
        """
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("[EMAIL_1]")
        e2 = buf.feed("[PHONE_2]")
        total = e1 + e2 + buf.flush()
        assert total == "[EMAIL_1][PHONE_2]"

    def test_adjacent_split_inside_second_token(self):
        """
        '[EMAIL_1][PHO' in first chunk, 'NE_2]' in second chunk.
        First token + '[PHO' is partially emitted; '[PHO' is held back.
        """
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("[EMAIL_1][PHO")
        # '[EMAIL_1]' is complete and safe; '[PHO' is a placeholder prefix
        assert "[EMAIL_1]" in e1 or "[EMAIL_1]" in (e1 + buf.buffer)
        assert buf.buffer == "[PHO"

        e2 = buf.feed("NE_2] end")
        total = e1 + e2 + buf.flush()
        assert total == "[EMAIL_1][PHONE_2] end"

    def test_space_between_placeholders_split_on_space(self):
        """
        '[EMAIL_1] [PHO' in first chunk, 'NE_2]' in second.
        The space separates the tokens; first token + space are emitted
        when the first chunk arrives.
        """
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("[EMAIL_1] [PHO")
        # '[EMAIL_1] ' is safe; '[PHO' is held
        assert e1 == "[EMAIL_1] "
        assert buf.buffer == "[PHO"

        e2 = buf.feed("NE_2]")
        total = e1 + e2 + buf.flush()
        assert total == "[EMAIL_1] [PHONE_2]"

    def test_three_placeholders_back_to_back(self):
        """
        '[A_1][B_2][C_3]' split after each token.
        All three tokens must appear in the output byte-for-byte.
        """
        tokens = "[A_1][B_2][C_3]"
        # Split: '[A_1]' | '[B_2]' | '[C_3]'
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, ["[A_1]", "[B_2]", "[C_3]"])
        assert total == tokens

    def test_split_inside_first_of_back_to_back(self):
        """
        '[EMA' | 'IL_1][PERSON_2]' — first token is split, second is complete.
        """
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("[EMA")
        assert e1 == ""
        assert buf.buffer == "[EMA"

        e2 = buf.feed("IL_1][PERSON_2]")
        total = e1 + e2 + buf.flush()
        assert total == "[EMAIL_1][PERSON_2]"

    def test_split_inside_both_back_to_back(self):
        """
        '[EMA' | 'IL_1][PERS' | 'ON_2] rest' — both tokens are split.
        """
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, ["[EMA", "IL_1][PERS", "ON_2] rest"])
        assert total == "[EMAIL_1][PERSON_2] rest"

    def test_many_back_to_back_splits_all_correct(self):
        """
        A sequence of five tokens delivered in many small chunks must
        reconstruct byte-for-byte.
        """
        tokens = "[T1_1][T2_2][T3_3][T4_4][T5_5]"
        # Chunk sizes of 3 characters each
        chunks = [tokens[i:i+3] for i in range(0, len(tokens), 3)]
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == tokens


# ─────────────────────────────────────────────────────────────────────────────
# 7. AC-mandated scenario 5: Chunk with only partial bytes and no safe-emit
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialOnlyChunkNoSafeEmit:
    """
    AC scenario 5 — The entire buffer content is a potential placeholder prefix
    with no character that can be safely emitted yet.  ``feed()`` must return
    an empty string and the buffer must hold the fragment.
    """

    @pytest.mark.parametrize("fragment", [
        "[",
        "[E",
        "[EM",
        "[EMA",
        "[EMAI",
        "[EMAIL",
        "[EMAIL_",
        "[EMAIL_1",
        "[EMAIL_1_",
        "[EMAIL_1_BLOCKED",
        "[AWS_SECRET",
        "[RRN_",
    ])
    def test_partial_placeholder_emits_nothing(self, fragment: str):
        """
        When the entire buffer is a placeholder prefix, feed() returns ''.
        """
        buf = StreamingLookAheadBuffer()
        result = buf.feed(fragment)
        assert result == "", (
            f"Expected empty emit for prefix {fragment!r}, got {result!r}"
        )
        assert buf.buffer == fragment, (
            f"Buffer should hold {fragment!r}, but holds {buf.buffer!r}"
        )

    def test_partial_buffer_never_exceeds_max(self):
        """
        Even when every chunk is a partial prefix and nothing is emitted,
        the buffer must never exceed max_buffer_size.
        """
        buf = StreamingLookAheadBuffer(max_buffer_size=_MIN_BUFFER_SIZE)
        # Feed a long sequence of uppercase letters (looks like a placeholder body)
        # Once the buffer is full it must force-emit to stay bounded.
        result = ""
        for _ in range(20):
            chunk = "[" + "A" * 10  # each chunk pushes the buffer larger
            result += buf.feed(chunk)
            assert buf.buffer_size <= buf._max, (
                f"Buffer exceeded max: size={buf.buffer_size}"
            )

    def test_partial_then_complete(self):
        """
        After feeding a partial prefix with no safe emit, adding the closing
        ']' must emit the whole token.
        """
        buf = StreamingLookAheadBuffer()
        buf.feed("[EMAI")
        buf.feed("L_")
        result = buf.feed("1]")
        total = result + buf.flush()
        assert total == "[EMAIL_1]"

    def test_partial_then_non_placeholder_text(self):
        """
        '[EMAI' followed by 'L ' (note: space makes it clear category ends)
        — the buffer holds '[EMAI' then finds the sequence is not a valid
        placeholder when 'L ' arrives (uppercase then space).

        Since '[EMAIL ' contains no ']', and space is invalid in a placeholder
        body, the entire '[EMAIL ' is actually safe to emit as literal text.
        """
        buf = StreamingLookAheadBuffer()
        e1 = buf.feed("[EMAI")
        assert e1 == ""

        # The space character terminates the potential placeholder body;
        # '[EMAI ' is no longer a valid placeholder prefix (space invalid).
        e2 = buf.feed("L text here")
        total = e1 + e2 + buf.flush()
        # Output must contain the original input unchanged
        assert total == "[EMAIL text here"

    def test_flush_after_only_partial_returns_the_partial(self):
        """
        If the stream ends while the buffer holds a partial prefix, flush()
        must return it verbatim (it's the caller's job to decide whether it
        is an unrestorable token or literal text).
        """
        buf = StreamingLookAheadBuffer()
        buf.feed("[EMAIL_1")  # incomplete placeholder — no ']' yet
        tail = buf.flush()
        assert tail == "[EMAIL_1"
        assert buf.buffer_size == 0

    def test_buffer_cleared_after_flush(self):
        buf = StreamingLookAheadBuffer()
        buf.feed("[PARTIAL")
        buf.flush()
        assert buf.buffer == ""
        assert buf.buffer_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Buffer size invariant (cross-scenario)
# ─────────────────────────────────────────────────────────────────────────────

class TestBufferSizeInvariant:
    """
    The internal buffer must never exceed *max_buffer_size* characters,
    regardless of chunk content or sequence.
    """

    def test_plain_text_oversized_single_chunk(self):
        """
        A single chunk larger than max_buffer_size with no '[' is force-emitted
        and the buffer stays bounded.
        """
        max_size = 128
        buf = StreamingLookAheadBuffer(max_buffer_size=max_size)
        big_chunk = "X" * (max_size + 50)
        result = buf.feed(big_chunk)
        # Some content is emitted; buffer is bounded
        assert len(result) > 0
        assert buf.buffer_size <= max_size

    def test_adversarial_partial_prefix_chunks(self):
        """
        An attacker (or adversarial LLM) that sends many tiny chunks each
        starting with '[A' (looks like the start of a placeholder) must not
        cause the buffer to grow without bound.
        """
        max_size = _MIN_BUFFER_SIZE
        buf = StreamingLookAheadBuffer(max_buffer_size=max_size)
        for _ in range(200):
            buf.feed("[A")
            assert buf.buffer_size <= max_size, (
                f"Buffer exceeded max: {buf.buffer_size}"
            )

    def test_buffer_size_bounded_through_all_scenarios(self):
        """
        Interleave all scenario types; buffer must always stay within max.
        """
        max_size = 256
        buf = StreamingLookAheadBuffer(max_buffer_size=max_size)
        sequences = [
            "Hello world ",           # plain text
            "[EMAIL_",                 # partial
            "1] ",                     # completes EMAIL_1
            "[",                       # lone bracket
            "PHONE_",                  # more partial
            "2]",                      # completes PHONE_2
            "[PERSON_3][ADDRESS_4]",   # two complete tokens
            "X" * 100,                 # large plain text
            "[A" * 5,                  # multiple partials
        ]
        for chunk in sequences:
            buf.feed(chunk)
            assert buf.buffer_size <= max_size

    def test_buffer_completely_drained_by_flush(self):
        """After flush(), buffer_size is always 0."""
        buf = StreamingLookAheadBuffer()
        buf.feed("[EMAIL_1")
        buf.feed("more text")
        buf.flush()
        assert buf.buffer_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Round-trip correctness (all output == all input)
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTripCorrectness:
    """
    For any sequence of chunks, the concatenation of all emitted text
    (including the final flush()) must equal the concatenation of all input
    chunks — byte-for-byte, no content lost, no duplication.
    """

    @pytest.mark.parametrize("original,chunk_sizes", [
        (
            "[EMAIL_1] hello [PERSON_2] world",
            [5, 3, 7, 2, 8, 6],
        ),
        (
            "no placeholders here at all",
            [4, 10, 9, 4],
        ),
        (
            "[A_1][B_2][C_3][D_4]",
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ),
        (
            "[PRIVATE_KEY_1_BLOCKED] secret was blocked",
            [3, 5, 7, 9, 2, 6],
        ),
        (
            "prefix [EMAIL_1][PHONE_",
            # This ends with an incomplete placeholder prefix
            [7, 8, 5],
        ),
    ])
    def test_round_trip_byte_for_byte(self, original: str, chunk_sizes: list):
        """Output == input for the given chunking pattern."""
        # Build chunks from the original string
        chunks = []
        i = 0
        for size in chunk_sizes:
            piece = original[i:i + size]
            if piece:
                chunks.append(piece)
            i += size
        # Append any remainder
        if i < len(original):
            chunks.append(original[i:])

        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == original, (
            f"Round-trip failed: input={original!r}, got={total!r}"
        )

    def test_round_trip_with_unicode(self):
        """Unicode content round-trips correctly."""
        original = "안녕 [EMAIL_1] 테스트 [PHONE_2] 완료"
        chunks = [original[i:i+4] for i in range(0, len(original), 4)]
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == original

    def test_round_trip_single_character_chunks(self):
        """Every character as a separate chunk — the hardest case."""
        original = "hi [EMAIL_1] there [PERSON_2]"
        chunks = list(original)
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == original

    def test_round_trip_large_payload(self):
        """A large payload with many placeholders round-trips correctly."""
        tokens = ["[EMAIL_1]", "[PHONE_2]", "[PERSON_3]", "[ADDRESS_4]", "[DOB_5]"]
        parts = []
        for i, tok in enumerate(tokens):
            parts.append(f"text segment {i} ")
            parts.append(tok)
            parts.append(f" more text {i} ")
        original = "".join(parts)

        # Split into 20-character chunks
        chunks = [original[i:i+20] for i in range(0, len(original), 20)]
        buf = StreamingLookAheadBuffer()
        total = _feed_all(buf, chunks)
        assert total == original


# ─────────────────────────────────────────────────────────────────────────────
# 10. reset() behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestResetBehaviour:

    def test_reset_clears_buffer(self):
        buf = StreamingLookAheadBuffer()
        buf.feed("[PARTIAL")
        buf.reset()
        assert buf.buffer == ""
        assert buf.buffer_size == 0

    def test_reset_allows_fresh_start(self):
        buf = StreamingLookAheadBuffer()
        buf.feed("[EMAIL_1")   # partial
        buf.reset()
        result = buf.feed("fresh content")
        assert result == "fresh content"
        assert buf.buffer_size == 0

    def test_flush_after_reset_returns_empty(self):
        buf = StreamingLookAheadBuffer()
        buf.feed("[PARTIAL")
        buf.reset()
        assert buf.flush() == ""


# ─────────────────────────────────────────────────────────────────────────────
# 11. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_lone_open_bracket_held_back(self):
        """A bare '[' alone must be held (could start a placeholder)."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("[")
        assert result == ""
        assert buf.buffer == "["

    def test_bracket_followed_by_lowercase_not_held(self):
        """'[hello' — lowercase after '[' means it's NOT a placeholder prefix."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("[hello world")
        assert result == "[hello world"
        assert buf.buffer_size == 0

    def test_bracket_followed_by_digit_not_held(self):
        """'[123' — digit after '[' is not a placeholder prefix."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("[123 error code")
        assert result == "[123 error code"
        assert buf.buffer_size == 0

    def test_multiple_open_brackets_rightmost_wins(self):
        """
        '[EMAIL_1] some [EMAI' — the rightmost '[' starts an incomplete
        placeholder; content up to (but not including) that '[' is emitted.
        """
        buf = StreamingLookAheadBuffer()
        result = buf.feed("[EMAIL_1] some [EMAI")
        assert result == "[EMAIL_1] some "
        assert buf.buffer == "[EMAI"

    def test_complete_then_partial_tokens(self):
        """A mix of complete and partial tokens in one chunk."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("[EMAIL_1][PHONE_2][PERSON_")
        # [EMAIL_1] and [PHONE_2] are complete → safe; [PERSON_ is partial → held
        assert "[EMAIL_1]" in result
        assert "[PHONE_2]" in result
        assert "[PERSON_" not in result  # held in buffer
        assert buf.buffer == "[PERSON_"

    def test_closed_bracket_without_open_emits_immediately(self):
        """A ']' with no preceding '[' in the buffer is just text."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("text] more text")
        assert result == "text] more text"

    def test_markdown_bracket_content_not_held(self):
        """[note] style (lowercase after '[') does not hold the buffer."""
        buf = StreamingLookAheadBuffer()
        result = buf.feed("See [note] for details")
        assert result == "See [note] for details"
        assert buf.buffer_size == 0

    def test_multiple_flushes_are_safe(self):
        """Calling flush() multiple times is idempotent after the first."""
        buf = StreamingLookAheadBuffer()
        buf.feed("[EMAIL_1] text")
        buf.flush()
        assert buf.flush() == ""  # second flush is empty, not an error

    def test_feed_after_flush_works(self):
        """Buffer can be reused after flush() for a new stream."""
        buf = StreamingLookAheadBuffer()
        buf.feed("first stream [EMAIL_1]")
        buf.flush()

        # New stream
        result = buf.feed("[PHONE_2] new stream")
        total = result + buf.flush()
        assert total == "[PHONE_2] new stream"

    def test_very_long_plain_text_does_not_hold_anything(self):
        """A very long chunk of plain text (no '[') emits all of it."""
        buf = StreamingLookAheadBuffer()
        big = "A" * 1000
        result = buf.feed(big)
        # All content should be emitted (or held at most _MAX_PLACEHOLDER_LEN)
        # After flush the total must equal big
        total = result + buf.flush()
        assert total == big
