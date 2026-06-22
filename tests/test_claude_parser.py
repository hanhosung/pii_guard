"""
Unit tests for the Claude provider request parser (Sub-AC 1).

These tests verify that:
  1. parse_claude_request extracts the correct field set from real and
     synthetic Claude Messages API payloads.
  2. Every text-bearing location is classified with the correct ScanField.
  3. Masking targets ONLY parsed (text_fields) — structural fields such as
     model, max_tokens, role, id, name, type, tool_use_id, and non-string
     leaves do NOT appear as scan targets.
  4. Unscannable fields (image, base64/url documents) appear in
     unscannable_fields, not in text_fields.
  5. Unknown block types appear in unknown_fields with a coverage_gap_reason.
  6. Nested tool_use.input objects are recursively expanded; non-string leaves
     are excluded.
  7. Edge cases (empty payload, missing keys, None values) do not raise.
  8. ClaudeFieldMap helpers (text_fields, unscannable_fields, unknown_fields,
     get_field) behave correctly.
"""
from __future__ import annotations

import pytest

from pii_guard.providers.claude_parser import (
    ClaudeFieldMap,
    ParsedField,
    ScanField,
    parse_claude_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _locations(fields) -> list[str]:
    """Return just the location strings from a list of ParsedField."""
    return [f.location for f in fields]


def _scan_fields(fields) -> list[str]:
    """Return just the scan_field values from a list of ParsedField."""
    return [f.scan_field for f in fields]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic structural field extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicFieldExtraction:
    """Core extraction of text-bearing fields."""

    def test_empty_payload_returns_empty_map(self):
        fm = parse_claude_request({})
        assert fm.all_fields == []
        assert fm.text_fields == []
        assert fm.unscannable_fields == []
        assert fm.unknown_fields == []

    def test_payload_without_messages_returns_empty_map(self):
        fm = parse_claude_request({"model": "claude-opus-4-5", "max_tokens": 1024})
        assert fm.all_fields == []

    def test_model_captured_not_as_scan_target(self):
        """model field is recorded in ClaudeFieldMap.model but not in text_fields."""
        fm = parse_claude_request({"model": "claude-3-5-sonnet-20241022"})
        assert fm.model == "claude-3-5-sonnet-20241022"
        # model is NOT a text scan target
        assert not any(f.location == "model" for f in fm.text_fields)

    def test_api_version_propagated(self):
        fm = parse_claude_request({}, api_version="2023-06-01")
        assert fm.api_version == "2023-06-01"

    def test_structural_fields_not_in_text_fields(self):
        """max_tokens, stream, temperature, top_p, stop — never scan targets."""
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "stream": True,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        for key in ("model", "max_tokens", "stream", "temperature"):
            assert key not in locs, f"Structural field {key!r} must not be a scan target"


# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPromptParsing:
    """system field as string and TextBlock array."""

    def test_system_string_produces_one_system_prompt_field(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        fm = parse_claude_request(payload)
        sys_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_PROMPT]
        assert len(sys_fields) == 1
        assert sys_fields[0].location == "system"
        assert sys_fields[0].text == "You are a helpful assistant."
        assert sys_fields[0].is_scannable is True

    def test_system_string_text_content_matches(self):
        payload = {"system": "Contact admin@corp.io for help.", "messages": []}
        fm = parse_claude_request(payload)
        sf = fm.get_field("system")
        assert sf is not None
        assert sf.text == "Contact admin@corp.io for help."

    def test_system_block_array_text_blocks_each_produce_field(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": [
                {"type": "text", "text": "Block one."},
                {"type": "text", "text": "Block two."},
            ],
            "messages": [],
        }
        fm = parse_claude_request(payload)
        sys_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_PROMPT]
        assert len(sys_fields) == 2
        assert sys_fields[0].location == "system[0].text"
        assert sys_fields[0].text == "Block one."
        assert sys_fields[1].location == "system[1].text"
        assert sys_fields[1].text == "Block two."

    def test_system_block_array_unknown_block_type(self):
        payload = {
            "system": [
                {"type": "text", "text": "OK"},
                {"type": "future_block", "data": "something"},
            ],
            "messages": [],
        }
        fm = parse_claude_request(payload)
        # One text field for the text block
        sys_text = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_PROMPT]
        assert len(sys_text) == 1
        # One unknown field for the future_block
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert uf.location == "system[1]"
        assert "future_block" in uf.coverage_gap_reason

    def test_empty_system_string_produces_field(self):
        payload = {"system": "", "messages": []}
        fm = parse_claude_request(payload)
        sf = fm.get_field("system")
        assert sf is not None
        assert sf.is_scannable is True
        assert sf.text == ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Message content — string shorthand
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageContentString:
    """message.content as a plain string."""

    def test_string_content_produces_message_text_field(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "Send invoice to bob@example.com"}
            ],
        }
        fm = parse_claude_request(payload)
        msg_fields = [f for f in fm.text_fields if f.scan_field == ScanField.MESSAGE_TEXT]
        assert len(msg_fields) == 1
        assert msg_fields[0].location == "messages[0].content"
        assert msg_fields[0].text == "Send invoice to bob@example.com"

    def test_multi_turn_each_message_produces_field(self):
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Bye"},
            ]
        }
        fm = parse_claude_request(payload)
        msg_fields = [f for f in fm.text_fields if f.scan_field == ScanField.MESSAGE_TEXT]
        assert len(msg_fields) == 3
        locs = _locations(msg_fields)
        assert "messages[0].content" in locs
        assert "messages[1].content" in locs
        assert "messages[2].content" in locs

    def test_role_field_not_a_scan_target(self):
        """role is structural — must not appear in text_fields."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].role" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Message content — text block
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageTextBlock:
    """message.content is a list with type=text blocks."""

    def test_single_text_block_extracted(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reach alice@corp.io"}
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content[0].text"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "Reach alice@corp.io"

    def test_multiple_text_blocks_each_extracted(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First span"},
                        {"type": "text", "text": "Second span"},
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tf = fm.text_fields
        assert len(tf) == 2
        assert tf[0].location == "messages[0].content[0].text"
        assert tf[1].location == "messages[0].content[1].text"

    def test_block_type_field_not_a_scan_target(self):
        """type='text' marker itself is structural, not a masking target."""
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content[0].type" not in locs
        assert "messages[0].content[0]" not in locs

    def test_empty_text_block_still_registered(self):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": ""}]}
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].text == ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_use blocks — input recursively expanded
# ─────────────────────────────────────────────────────────────────────────────

class TestToolUseInputParsing:
    """tool_use.input is recursively walked; non-string leaves excluded."""

    def test_flat_string_values_extracted(self):
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_001",
                            "name": "send_email",
                            "input": {"to": "user@example.com", "subject": "Hello"},
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tu_fields = [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        locs = _locations(tu_fields)
        assert "messages[0].content[0].input.to" in locs
        assert "messages[0].content[0].input.subject" in locs

    def test_nested_dict_recursively_expanded(self):
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_002",
                            "name": "store",
                            "input": {
                                "contact": {
                                    "email": "deep@nested.com",
                                    "phone": "010-1111-2222",
                                }
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        )
        assert "messages[0].content[0].input.contact.email" in locs
        assert "messages[0].content[0].input.contact.phone" in locs

    def test_array_of_strings_each_registered(self):
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_003",
                            "name": "notify",
                            "input": {"emails": ["a@b.com", "c@d.io"]},
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        )
        assert "messages[0].content[0].input.emails[0]" in locs
        assert "messages[0].content[0].input.emails[1]" in locs

    def test_numeric_boolean_none_values_excluded(self):
        """Numbers, booleans, None in tool_use.input are NOT masking targets."""
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_004",
                            "name": "set_params",
                            "input": {
                                "count": 42,
                                "enabled": True,
                                "ratio": 3.14,
                                "nothing": None,
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tu_fields = [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        assert tu_fields == [], (
            "Numeric/boolean/None tool_use.input values must not be scan targets"
        )

    def test_tool_use_id_name_not_scan_targets(self):
        """Structural tool_use fields (id, name, type) are NOT masking targets."""
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_preserve",
                            "name": "my_tool",
                            "input": {"q": "hello"},
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content[0].id" not in locs
        assert "messages[0].content[0].name" not in locs
        assert "messages[0].content[0].type" not in locs

    def test_deeply_nested_tool_input_registered(self):
        """3-level nesting is fully expanded."""
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "deep",
                            "name": "configure",
                            "input": {
                                "level1": {
                                    "level2": {
                                        "level3": {"token": "ghp_secret_value"}
                                    }
                                }
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        )
        assert "messages[0].content[0].input.level1.level2.level3.token" in locs

    def test_mixed_input_string_and_numeric(self):
        """String values in mixed input are extracted; numeric skipped."""
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_mix",
                            "name": "op",
                            "input": {
                                "name": "Alice",
                                "age": 30,
                                "active": False,
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_USE_INPUT]
        )
        assert "messages[0].content[0].input.name" in locs
        assert "messages[0].content[0].input.age" not in locs
        assert "messages[0].content[0].input.active" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 6. tool_result blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResultParsing:
    """tool_result.content as string or TextBlock array."""

    def test_string_content_registered(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_100",
                            "content": "Email sent to alice@example.com",
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_RESULT]
        assert len(tr_fields) == 1
        assert tr_fields[0].location == "messages[0].content[0].content"
        assert tr_fields[0].text == "Email sent to alice@example.com"

    def test_block_array_content_each_text_block_registered(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_101",
                            "content": [
                                {"type": "text", "text": "Phone: 010-5555-1234"},
                                {"type": "text", "text": "Status: OK"},
                            ],
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        tr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.TOOL_RESULT]
        assert len(tr_fields) == 2
        locs = _locations(tr_fields)
        assert "messages[0].content[0].content[0].text" in locs
        assert "messages[0].content[0].content[1].text" in locs

    def test_image_in_tool_result_is_unscannable(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_102",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBORw==",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].scan_field == ScanField.IMAGE
        # Must NOT appear in text_fields
        assert not any(f.scan_field == ScanField.IMAGE for f in fm.text_fields)

    def test_tool_result_without_content_field_ignored(self):
        """tool_result with no content key produces no fields."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_103",
                            # no "content" key
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert fm.all_fields == []

    def test_tool_use_id_not_a_scan_target(self):
        """tool_use_id is structural metadata — never a masking target."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_preserve_me",
                            "content": "Done.",
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content[0].tool_use_id" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 7. Document blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentBlockParsing:
    """document blocks with text/base64/url sources."""

    def test_text_source_registered_as_scannable(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Contact manager@company.io for help.",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        doc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.DOCUMENT_BLOCK]
        assert len(doc_fields) == 1
        assert doc_fields[0].location == "messages[0].content[0].source.data"
        assert doc_fields[0].text == "Contact manager@company.io for help."
        assert doc_fields[0].is_scannable is True

    def test_base64_source_registered_as_unscannable(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjMK...",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert uf.scan_field == ScanField.DOCUMENT_BLOCK
        assert uf.location == "messages[0].content[0].source"
        assert "base64" in uf.coverage_gap_reason
        # Must NOT appear in text_fields
        assert not any(
            f.scan_field == ScanField.DOCUMENT_BLOCK for f in fm.text_fields
        )

    def test_url_source_registered_as_unscannable(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/doc.pdf",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert "url" in uf.coverage_gap_reason

    def test_unknown_document_source_type_is_unknown_field(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "future_source_type",
                                "data": "...",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert "future_source_type" in uf.coverage_gap_reason

    def test_document_missing_source_is_unknown_field(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "document"},  # no source
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unknown_fields) == 1

    def test_document_media_type_not_a_scan_target(self):
        """media_type is structural metadata — not a masking target."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Hello",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content[0].source.media_type" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 8. Image blocks → unscannable
# ─────────────────────────────────────────────────────────────────────────────

class TestImageBlockParsing:
    """image blocks produce unscannable entries, not text entries."""

    def test_image_block_in_unscannable_not_text_fields(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQ==",
                            },
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        # No text scan targets
        assert fm.text_fields == []
        # One unscannable entry
        assert len(fm.unscannable_fields) == 1
        img = fm.unscannable_fields[0]
        assert img.scan_field == ScanField.IMAGE
        assert img.is_unscannable is True
        assert img.is_scannable is False
        assert img.text is None

    def test_image_coverage_gap_reason_set(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                        }
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        img = fm.unscannable_fields[0]
        assert img.coverage_gap_reason is not None
        assert len(img.coverage_gap_reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Unknown block types → unknown field entries
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownBlockTypes:
    """Unrecognized block types appear in unknown_fields with coverage_gap_reason."""

    def test_unknown_content_block_type_registered(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_content_type", "data": "something"},
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert uf.scan_field == ScanField.UNKNOWN
        assert uf.is_unknown is True
        assert uf.is_scannable is False
        assert "future_content_type" in uf.coverage_gap_reason

    def test_unknown_field_not_in_text_fields(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "weird_type", "payload": "data"}],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert fm.text_fields == []
        assert len(fm.unknown_fields) == 1

    def test_mix_known_and_unknown_blocks(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "unknown_type", "data": "X"},
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        assert len(fm.text_fields) == 1
        assert len(fm.unknown_fields) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. Masking targets only parsed (text_fields)
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskingTargetBoundary:
    """
    Critical invariant: text_fields is the exclusive set of locations the
    scrubber may modify.  Every other key in the payload is off-limits.
    """

    def test_only_content_locations_in_text_fields(self):
        """
        A full payload is parsed; every text_field location must be a
        recognized content path (contains .text, .data, .content, or .input).
        """
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 2048,
            "system": "Be helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email: user@example.com"},
                        {
                            "type": "tool_use",
                            "id": "tu_x",
                            "name": "lookup",
                            "input": {"query": "find user@example.com"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_x",
                            "content": "Found user@example.com",
                        }
                    ],
                },
            ],
        }
        fm = parse_claude_request(payload)
        # Every text_field location must reference a content-bearing path
        for pf in fm.text_fields:
            assert any(
                seg in pf.location
                for seg in (".text", ".data", ".content", ".input", "system")
            ), f"Unexpected scan target location: {pf.location!r}"

    def test_no_structural_keys_in_any_field_location(self):
        """
        model, max_tokens, role, id, name, type, tool_use_id, media_type,
        stream, temperature — none appear as standalone scan target locations.
        """
        STRUCTURAL = {
            "model", "max_tokens", "role", "stream", "temperature",
            "top_p", "top_k", "stop_sequences", "metadata",
        }
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "stream": False,
            "temperature": 1.0,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "greet",
                            "input": {"greeting": "hello"},
                        }
                    ],
                }
            ],
        }
        fm = parse_claude_request(payload)
        all_locs = set(_locations(fm.all_fields))
        for key in STRUCTURAL:
            assert key not in all_locs, (
                f"Structural field {key!r} must not appear in the field map"
            )

    def test_original_payload_not_mutated(self):
        """parse_claude_request must not modify the caller's payload dict."""
        import copy
        payload = {
            "model": "claude-3",
            "system": "Email: admin@corp.io",
            "messages": [{"role": "user", "content": "Hi admin@corp.io"}],
        }
        original_copy = copy.deepcopy(payload)
        parse_claude_request(payload)
        assert payload == original_copy


# ─────────────────────────────────────────────────────────────────────────────
# 11. Full mixed payload — real-world synthetic request
# ─────────────────────────────────────────────────────────────────────────────

class TestFullMixedPayload:
    """
    End-to-end parse of a realistic multi-content-type payload.
    Asserts the complete expected field map.
    """

    def _build_payload(self):
        return {
            "model": "claude-opus-4-5",
            "max_tokens": 2048,
            "system": "System: admin@internal.corp",
            "messages": [
                {
                    "role": "user",
                    "content": "User message: user@example.com",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I see you have an email."},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "send_email",
                            "input": {
                                "to": "user@example.com",
                                "subject": "Hello",
                                "body": {"greeting": "Dear user@example.com"},
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [
                                {"type": "text", "text": "Sent to user@example.com."},
                            ],
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Doc body: user@example.com",
                            },
                        },
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                        },
                    ],
                },
            ],
        }

    def test_all_text_fields_present(self):
        fm = parse_claude_request(self._build_payload())
        locs = set(_locations(fm.text_fields))
        expected = {
            "system",
            "messages[0].content",
            "messages[1].content[0].text",
            "messages[1].content[1].input.to",
            "messages[1].content[1].input.subject",
            "messages[1].content[1].input.body.greeting",
            "messages[2].content[0].content[0].text",
            "messages[2].content[1].source.data",
        }
        assert expected == locs, (
            f"Missing: {expected - locs}\nExtra: {locs - expected}"
        )

    def test_unscannable_fields_present(self):
        fm = parse_claude_request(self._build_payload())
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].scan_field == ScanField.IMAGE

    def test_unknown_fields_absent(self):
        fm = parse_claude_request(self._build_payload())
        assert fm.unknown_fields == []

    def test_total_field_count(self):
        fm = parse_claude_request(self._build_payload())
        assert len(fm.text_fields) == 8
        assert len(fm.unscannable_fields) == 1
        assert len(fm.all_fields) == 9

    def test_all_text_fields_have_correct_scan_field_types(self):
        fm = parse_claude_request(self._build_payload())
        expected_types = {
            "system": ScanField.SYSTEM_PROMPT,
            "messages[0].content": ScanField.MESSAGE_TEXT,
            "messages[1].content[0].text": ScanField.MESSAGE_TEXT,
            "messages[1].content[1].input.to": ScanField.TOOL_USE_INPUT,
            "messages[1].content[1].input.subject": ScanField.TOOL_USE_INPUT,
            "messages[1].content[1].input.body.greeting": ScanField.TOOL_USE_INPUT,
            "messages[2].content[0].content[0].text": ScanField.TOOL_RESULT,
            "messages[2].content[1].source.data": ScanField.DOCUMENT_BLOCK,
        }
        for pf in fm.text_fields:
            assert pf.scan_field == expected_types[pf.location], (
                f"Wrong scan_field for {pf.location!r}: "
                f"got {pf.scan_field}, want {expected_types[pf.location]}"
            )

    def test_text_values_match_payload(self):
        payload = self._build_payload()
        fm = parse_claude_request(payload)
        values = {f.location: f.text for f in fm.text_fields}
        assert values["system"] == "System: admin@internal.corp"
        assert values["messages[0].content"] == "User message: user@example.com"
        assert values["messages[1].content[0].text"] == "I see you have an email."
        assert values["messages[1].content[1].input.to"] == "user@example.com"
        assert values["messages[2].content[1].source.data"] == "Doc body: user@example.com"


# ─────────────────────────────────────────────────────────────────────────────
# 12. ClaudeFieldMap helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeFieldMapHelpers:
    """Unit tests for ClaudeFieldMap property and method helpers."""

    def _make_map(self) -> ClaudeFieldMap:
        fm = ClaudeFieldMap()
        fm.all_fields.append(ParsedField(
            location="system",
            scan_field=ScanField.SYSTEM_PROMPT,
            text="Hello",
            is_scannable=True,
        ))
        fm.all_fields.append(ParsedField(
            location="messages[0].content[0]",
            scan_field=ScanField.IMAGE,
            text=None,
            is_scannable=False,
            is_unscannable=True,
            coverage_gap_reason="image block",
        ))
        fm.all_fields.append(ParsedField(
            location="messages[0].content[1]",
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason="unknown type",
        ))
        return fm

    def test_text_fields_returns_only_scannable(self):
        fm = self._make_map()
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "system"

    def test_unscannable_fields_returns_only_unscannable(self):
        fm = self._make_map()
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].location == "messages[0].content[0]"

    def test_unknown_fields_returns_only_unknown(self):
        fm = self._make_map()
        assert len(fm.unknown_fields) == 1
        assert fm.unknown_fields[0].location == "messages[0].content[1]"

    def test_get_field_existing_location(self):
        fm = self._make_map()
        pf = fm.get_field("system")
        assert pf is not None
        assert pf.text == "Hello"

    def test_get_field_missing_location_returns_none(self):
        fm = self._make_map()
        assert fm.get_field("nonexistent.path") is None

    def test_all_fields_count(self):
        fm = self._make_map()
        assert len(fm.all_fields) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 13. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases that must not raise."""

    def test_none_message_content_skipped(self):
        payload = {
            "messages": [
                {"role": "user", "content": None}
            ]
        }
        fm = parse_claude_request(payload)
        assert fm.all_fields == []

    def test_non_dict_message_skipped(self):
        payload = {"messages": ["not a dict"]}
        fm = parse_claude_request(payload)
        assert fm.all_fields == []

    def test_non_list_messages_ignored(self):
        payload = {"messages": "not a list"}
        fm = parse_claude_request(payload)
        assert fm.all_fields == []

    def test_no_model_key_model_is_none(self):
        fm = parse_claude_request({"messages": []})
        assert fm.model is None

    def test_model_integer_gives_none(self):
        """Non-string model values are ignored."""
        fm = parse_claude_request({"model": 42})
        assert fm.model is None

    def test_empty_messages_list(self):
        fm = parse_claude_request({"messages": []})
        assert fm.all_fields == []

    def test_large_multi_message_no_errors(self):
        """Stress test: 20 turns with mixed content types."""
        messages = []
        for i in range(20):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [
                    {"type": "text", "text": f"Turn {i} message text."},
                ],
            })
        payload = {"model": "claude-opus-4-5", "messages": messages}
        fm = parse_claude_request(payload)
        assert len(fm.text_fields) == 20
        assert fm.unknown_fields == []
        assert fm.unscannable_fields == []

    def test_tool_use_with_none_input(self):
        """tool_use.input == None is handled gracefully."""
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "ping", "input": None}
                    ],
                }
            ]
        }
        fm = parse_claude_request(payload)
        # None input → no fields (not None-scannable)
        assert fm.text_fields == []

    def test_system_none_is_unknown(self):
        """system == None triggers an unknown field entry, not an error."""
        payload = {"system": None, "messages": []}
        fm = parse_claude_request(payload)
        assert len(fm.unknown_fields) == 1

    def test_system_integer_is_unknown(self):
        payload = {"system": 42, "messages": []}
        fm = parse_claude_request(payload)
        assert len(fm.unknown_fields) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 14. ScanField enum values
# ─────────────────────────────────────────────────────────────────────────────

class TestScanFieldEnum:
    """Verify ScanField enum values match ontology scan_field strings."""

    def test_system_prompt_value(self):
        assert ScanField.SYSTEM_PROMPT.value == "system_prompt"

    def test_message_text_value(self):
        assert ScanField.MESSAGE_TEXT.value == "message_text"

    def test_tool_use_input_value(self):
        assert ScanField.TOOL_USE_INPUT.value == "tool_use_input"

    def test_tool_result_value(self):
        assert ScanField.TOOL_RESULT.value == "tool_result"

    def test_document_block_value(self):
        assert ScanField.DOCUMENT_BLOCK.value == "document_block"

    def test_image_value(self):
        assert ScanField.IMAGE.value == "image"

    def test_unknown_value(self):
        assert ScanField.UNKNOWN.value == "unknown"

    def test_all_seven_fields_defined(self):
        expected = {
            "system_prompt", "message_text", "tool_use_input",
            "tool_result", "document_block", "image", "unknown",
        }
        actual = {sf.value for sf in ScanField}
        assert actual == expected
