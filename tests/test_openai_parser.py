"""
Unit tests for the OpenAI provider request parser (Sub-AC 2).

These tests verify that:
  1. parse_openai_request extracts the correct field set from real and
     synthetic OpenAI chat-completions payloads.
  2. Every text-bearing location is classified with the correct ScanField.
  3. Masking targets ONLY parsed (text_fields) — structural fields such as
     model, max_tokens, role, id, name, type, tool_call_id, and non-string
     leaves do NOT appear as scan targets.
  4. Unscannable fields (image_url, input_audio, file parts) appear in
     unscannable_fields, not in text_fields.
  5. Unknown part types appear in unknown_fields with a coverage_gap_reason.
  6. Nested tool_call function.arguments JSON is recursively expanded;
     non-string leaves are excluded.
  7. Invalid (non-JSON) tool_call arguments are registered as raw-text scan
     targets with has_json_parse_gap=True.
  8. Edge cases (empty payload, missing keys, None content, empty strings)
     do not raise.
  9. OpenAIFieldMap helpers (text_fields, unscannable_fields, unknown_fields,
     get_field) behave correctly.
  10. All six roles — system, user, assistant, tool, developer, and an unknown
      future role — produce the correct ScanField assignments.
  11. Masking boundary: the returned text_fields enumeration is the exact set
      the scrubber (scrub_openai_request) is permitted to touch; fields not
      listed here must not be modified by the scrubber.
"""
from __future__ import annotations

import json

import pytest

from pii_guard.providers.openai_parser import (
    OpenAIFieldMap,
    ParsedField,
    ScanField,
    parse_openai_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _locations(fields: list) -> list[str]:
    """Return just the location strings from a list of ParsedField."""
    return [f.location for f in fields]


def _scan_fields(fields: list) -> list[ScanField]:
    """Return just the scan_field values from a list of ParsedField."""
    return [f.scan_field for f in fields]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic structural field extraction — empty/minimal payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicFieldExtraction:
    """Core extraction of text-bearing fields; structural fields excluded."""

    def test_empty_payload_returns_empty_map(self):
        fm = parse_openai_request({})
        assert fm.all_fields == []
        assert fm.text_fields == []
        assert fm.unscannable_fields == []
        assert fm.unknown_fields == []

    def test_payload_without_messages_returns_empty_map(self):
        fm = parse_openai_request({"model": "gpt-4o", "max_tokens": 512})
        assert fm.all_fields == []

    def test_empty_messages_list_returns_empty_map(self):
        fm = parse_openai_request({"model": "gpt-4o", "messages": []})
        assert fm.all_fields == []

    def test_model_captured_not_as_scan_target(self):
        """model field is recorded in OpenAIFieldMap.model but not in text_fields."""
        fm = parse_openai_request({"model": "gpt-4o", "messages": []})
        assert fm.model == "gpt-4o"
        assert not any(f.location == "model" for f in fm.text_fields)

    def test_model_none_when_not_string(self):
        """Non-string model value → fm.model is None."""
        fm = parse_openai_request({"model": 42, "messages": []})
        assert fm.model is None

    def test_structural_fields_not_in_text_fields(self):
        """model, max_tokens, stream, temperature — never scan targets."""
        payload = {
            "model": "gpt-4o",
            "max_tokens": 512,
            "stream": True,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        for key in ("model", "max_tokens", "stream", "temperature"):
            assert key not in locs, (
                f"Structural field {key!r} must not be a scan target"
            )

    def test_message_role_not_in_text_fields(self):
        """messages[*].role is structural and must never be a scan target."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].role" not in locs

    def test_non_dict_messages_skipped_gracefully(self):
        """Non-dict entries in messages list must not raise."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                "not a dict",
                {"role": "user", "content": "Hello"},
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        # Only the real dict message's content should be extracted
        assert "messages[1].content" in locs
        # The string entry should not produce any field
        assert not any(loc.startswith("messages[0]") for loc in locs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. System role message parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemMessageParsing:
    """system role messages — content as string and array of parts."""

    def test_system_string_content_produces_one_system_message_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content"
        assert tf[0].scan_field == ScanField.SYSTEM_MESSAGE
        assert tf[0].text == "You are a helpful assistant."
        assert tf[0].is_scannable is True

    def test_system_string_content_correct_text_captured(self):
        text = "Use API key = sk-abc for all requests."
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": text}],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields[0].text == text

    def test_system_array_text_part_produces_system_message_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "System instructions."}],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content[0].text"
        assert tf[0].scan_field == ScanField.SYSTEM_MESSAGE
        assert tf[0].text == "System instructions."

    def test_system_array_multiple_text_parts(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Part one."},
                        {"type": "text", "text": "Part two."},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content[0].text" in locs
        assert "messages[0].content[1].text" in locs
        for f in fm.text_fields:
            assert f.scan_field == ScanField.SYSTEM_MESSAGE

    def test_system_unknown_part_type_produces_unknown_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "future_type", "data": "something"}],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        assert len(fm.unknown_fields) == 1
        assert fm.unknown_fields[0].location == "messages[0].content[0]"
        assert "future_type" in fm.unknown_fields[0].coverage_gap_reason


# ─────────────────────────────────────────────────────────────────────────────
# 3. User role message parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMessageParsing:
    """user role messages — content as string and array of parts."""

    def test_user_string_content_produces_message_text_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "Hello world"

    def test_user_array_text_part_produces_message_text_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email is user@example.com"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content[0].text"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "My email is user@example.com"

    def test_user_image_url_part_produces_unscannable_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x.com/img.png"}},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        uf = fm.unscannable_fields
        assert len(uf) == 1
        assert uf[0].location == "messages[0].content[0]"
        assert uf[0].scan_field == ScanField.IMAGE_URL
        assert uf[0].is_unscannable is True
        assert uf[0].coverage_gap_reason is not None

    def test_user_input_audio_part_produces_unscannable_field(self):
        payload = {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "base64audio", "format": "wav"},
                        },
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        uf = fm.unscannable_fields
        assert len(uf) == 1
        assert uf[0].is_unscannable is True

    def test_user_file_part_produces_unscannable_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_id": "file-abc123"}},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].is_unscannable is True

    def test_user_unknown_part_type_produces_unknown_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "new_future_type", "data": "something"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        uf = fm.unknown_fields
        assert len(uf) == 1
        assert uf[0].is_unknown is True
        assert "new_future_type" in uf[0].coverage_gap_reason

    def test_user_mixed_parts_text_and_image(self):
        """Text part → text_fields; image_url part → unscannable_fields."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "messages[0].content[0].text"
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].location == "messages[0].content[1]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assistant role message parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestAssistantMessageParsing:
    """assistant role — content string, array, refusal parts."""

    def test_assistant_string_content_produces_message_text_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        }
        fm = parse_openai_request(payload)
        asst_fields = [f for f in fm.text_fields if "messages[1]" in f.location]
        assert len(asst_fields) == 1
        assert asst_fields[0].scan_field == ScanField.MESSAGE_TEXT
        assert asst_fields[0].text == "Hello! How can I help?"

    def test_assistant_text_part_array_produces_message_text_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here is the answer."},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content[0].text"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT

    def test_assistant_refusal_part_produces_message_text_field(self):
        """refusal parts carry text that must be scanned."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": "Cannot help with that."},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content[0].refusal"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "Cannot help with that."

    def test_assistant_null_content_produces_no_field(self):
        """content=None (typical for tool_calls-only assistant) must not error."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": "{}"},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        # No content field produced (content is None → skipped)
        content_fields = [f for f in fm.all_fields if "content" in f.location]
        assert content_fields == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tool role message parsing (tool_result)
# ─────────────────────────────────────────────────────────────────────────────

class TestToolMessageParsing:
    """tool role messages — content maps to TOOL_RESULT scan field."""

    def test_tool_string_content_produces_tool_result_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "content": "Result: operation succeeded.",
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content"
        assert tf[0].scan_field == ScanField.TOOL_RESULT
        assert tf[0].text == "Result: operation succeeded."

    def test_tool_string_content_with_pii_correct_text(self):
        """Exact text captured — scanner will detect PII in a later stage."""
        text = "User email: victim@example.com. Status: done."
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "tool", "tool_call_id": "tc_1", "content": text},
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields[0].text == text

    def test_tool_array_text_parts_produce_tool_result_fields(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_002",
                    "content": [
                        {"type": "text", "text": "Line 1"},
                        {"type": "text", "text": "Line 2"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 2
        for f in tf:
            assert f.scan_field == ScanField.TOOL_RESULT

    def test_tool_call_id_not_in_text_fields(self):
        """tool_call_id is structural and must not be a scan target."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "preserve_me",
                    "content": "Some output.",
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].tool_call_id" not in locs

    def test_tool_message_scan_field_is_tool_result(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "tool", "tool_call_id": "x", "content": "Some result."},
            ],
        }
        fm = parse_openai_request(payload)
        sf = _scan_fields(fm.text_fields)
        assert ScanField.TOOL_RESULT in sf


# ─────────────────────────────────────────────────────────────────────────────
# 6. Developer role message parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestDeveloperMessageParsing:
    """developer role messages (o-series models) — mapped to MESSAGE_TEXT."""

    def test_developer_string_content_produces_message_text_field(self):
        payload = {
            "model": "o3",
            "messages": [
                {"role": "developer", "content": "Internal instructions."},
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "messages[0].content"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "Internal instructions."

    def test_developer_array_text_part(self):
        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "developer",
                    "content": [
                        {"type": "text", "text": "Config: use api@dev.com"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# 7. Unknown / future role messages
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownRoleParsing:
    """Unknown / future role names — content classified as MESSAGE_TEXT."""

    def test_unknown_role_string_content_classified_as_message_text(self):
        """An unrecognized role produces MESSAGE_TEXT fields (future-safe)."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "moderator", "content": "Some future role content."},
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# 8. Tool-call arguments parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCallArgumentsParsing:
    """tool_calls[*].function.arguments JSON parsing and recursive expansion."""

    def test_simple_json_arguments_string_leaf_extracted(self):
        args = json.dumps({"to": "recipient@domain.com", "subject": "Hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {"name": "send_email", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        # Both string leaves should be extracted
        assert "messages[0].tool_calls[0].function.arguments.to" in locs
        assert "messages[0].tool_calls[0].function.arguments.subject" in locs

    def test_tool_call_arguments_scan_field_is_tool_call_args(self):
        args = json.dumps({"msg": "Hello contact@domain.com"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {"name": "greet", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        args_fields = [
            f for f in fm.text_fields
            if f.scan_field == ScanField.TOOL_CALL_ARGS
        ]
        assert args_fields

    def test_numeric_boolean_null_leaves_not_in_text_fields(self):
        """Numbers, booleans, and null in arguments must NOT be scan targets."""
        args = json.dumps({
            "count": 42,
            "enabled": True,
            "ratio": 3.14,
            "nothing": None,
        })
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_003",
                            "type": "function",
                            "function": {"name": "configure", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        # No text fields should be produced (no string leaves)
        args_fields = [
            f for f in fm.text_fields
            if "arguments" in f.location
        ]
        assert args_fields == []

    def test_nested_object_arguments_recursively_expanded(self):
        args = json.dumps({
            "contact": {
                "email": "nested@corp.com",
                "phone": "010-7777-8888",
            }
        })
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_004",
                            "type": "function",
                            "function": {"name": "store_contact", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        base = "messages[0].tool_calls[0].function.arguments"
        assert f"{base}.contact.email" in locs
        assert f"{base}.contact.phone" in locs

    def test_array_values_in_arguments_recursively_expanded(self):
        args = json.dumps({"emails": ["a@first.com", "b@second.io", "safe"]})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_005",
                            "type": "function",
                            "function": {"name": "notify_all", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        base = "messages[0].tool_calls[0].function.arguments"
        assert f"{base}.emails[0]" in locs
        assert f"{base}.emails[1]" in locs
        assert f"{base}.emails[2]" in locs

    def test_deeply_nested_arguments_scanned(self):
        args = json.dumps({"config": {"creds": {"level3": {"token": "ghp_abc"}}}})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_006",
                            "type": "function",
                            "function": {"name": "configure", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        base = "messages[0].tool_calls[0].function.arguments"
        loc = f"{base}.config.creds.level3.token"
        assert loc in _locations(fm.text_fields)

    def test_invalid_json_arguments_registered_with_json_parse_gap(self):
        """
        If arguments is not valid JSON, the raw string is a scan target
        and has_json_parse_gap=True.
        """
        raw = "email=victim@corp.com&key=AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_007",
                            "type": "function",
                            "function": {"name": "bad_args", "arguments": raw},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].text == raw
        assert tf[0].has_json_parse_gap is True
        assert tf[0].coverage_gap_reason is not None

    def test_multiple_tool_calls_all_parsed(self):
        """Multiple tool calls in one assistant message are all parsed."""
        args1 = json.dumps({"email": "first@tool.com"})
        args2 = json.dumps({"key": "secret_value"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": args1},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "tool_b", "arguments": args2},
                        },
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].tool_calls[0].function.arguments.email" in locs
        assert "messages[0].tool_calls[1].function.arguments.key" in locs

    def test_tool_call_structural_fields_not_in_text_fields(self):
        """id, type, and function.name are structural and must not be scan targets."""
        args = json.dumps({"q": "hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "struct_id",
                            "type": "function",
                            "function": {"name": "my_function", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        # Structural tool_call fields must be absent
        assert "messages[0].tool_calls[0].id" not in locs
        assert "messages[0].tool_calls[0].type" not in locs
        assert "messages[0].tool_calls[0].function.name" not in locs

    def test_empty_json_object_arguments_produces_no_fields(self):
        """Empty JSON object {} has no string leaves → no scan targets."""
        args = json.dumps({})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_empty",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        args_fields = [f for f in fm.text_fields if "arguments" in f.location]
        assert args_fields == []


# ─────────────────────────────────────────────────────────────────────────────
# 9. Multi-role / multi-message payload
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiRolePayload:
    """Full realistic multi-role conversations produce the correct field set."""

    def test_all_four_roles_correctly_classified(self):
        """system → SYSTEM_MESSAGE, user/assistant → MESSAGE_TEXT, tool → TOOL_RESULT."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User message."},
                {"role": "assistant", "content": "Assistant reply."},
                {"role": "tool", "tool_call_id": "tc1", "content": "Tool result."},
            ],
        }
        fm = parse_openai_request(payload)
        tf = fm.text_fields

        sf_by_loc = {f.location: f.scan_field for f in tf}
        assert sf_by_loc["messages[0].content"] == ScanField.SYSTEM_MESSAGE
        assert sf_by_loc["messages[1].content"] == ScanField.MESSAGE_TEXT
        assert sf_by_loc["messages[2].content"] == ScanField.MESSAGE_TEXT
        assert sf_by_loc["messages[3].content"] == ScanField.TOOL_RESULT

    def test_realistic_four_turn_conversation(self):
        """A realistic 4-turn conversation with tool use."""
        args = json.dumps({"to": "user@example.com", "subject": "Report"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Send to user@example.com"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_i",
                            "type": "function",
                            "function": {"name": "send_email", "arguments": args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc_i",
                    "content": "Email sent successfully.",
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)

        # System message
        assert "messages[0].content" in locs
        # User message
        assert "messages[1].content" in locs
        # Tool-call arguments (string leaves)
        assert "messages[2].tool_calls[0].function.arguments.to" in locs
        assert "messages[2].tool_calls[0].function.arguments.subject" in locs
        # Tool result
        assert "messages[3].content" in locs

    def test_multi_message_pii_in_each_turn(self):
        """Every turn with PII has a text field registered."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "My email: first@a.com"},
                {"role": "assistant", "content": "Got it, first@a.com."},
                {"role": "user", "content": "Also: second@b.io"},
            ],
        }
        fm = parse_openai_request(payload)
        locs = _locations(fm.text_fields)
        assert "messages[0].content" in locs
        assert "messages[1].content" in locs
        assert "messages[2].content" in locs
        assert len(locs) == 3

    def test_field_count_matches_expected(self):
        """Exact count of text fields for a known payload."""
        # 1 system + 1 user + 2 tool-call arg leaves + 1 tool result = 5
        args = json.dumps({"email": "x@y.com", "note": "hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "User message."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tc1", "content": "Tool output."},
            ],
        }
        fm = parse_openai_request(payload)
        assert len(fm.text_fields) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 10. OpenAIFieldMap helper methods
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIFieldMapHelpers:
    """OpenAIFieldMap property and method correctness."""

    def test_text_fields_only_scannable(self):
        """text_fields contains only entries with is_scannable=True."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                        {"type": "future_type", "data": "x"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        for f in fm.text_fields:
            assert f.is_scannable is True

    def test_unscannable_fields_only_unscannable(self):
        """unscannable_fields contains only entries with is_unscannable=True."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        for f in fm.unscannable_fields:
            assert f.is_unscannable is True

    def test_unknown_fields_only_unknown(self):
        """unknown_fields contains only entries with is_unknown=True."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_type", "data": "x"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        for f in fm.unknown_fields:
            assert f.is_unknown is True

    def test_get_field_returns_correct_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        fm = parse_openai_request(payload)
        f = fm.get_field("messages[0].content")
        assert f is not None
        assert f.text == "Hello world"

    def test_get_field_returns_none_for_missing_location(self):
        fm = parse_openai_request({"model": "gpt-4o", "messages": []})
        assert fm.get_field("nonexistent.location") is None

    def test_all_fields_is_superset_of_text_unscannable_unknown(self):
        """all_fields must equal text_fields + unscannable_fields + unknown_fields."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                        {"type": "future_type", "data": "x"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        combined = fm.text_fields + fm.unscannable_fields + fm.unknown_fields
        assert sorted(_locations(fm.all_fields)) == sorted(_locations(combined))

    def test_model_attribute_populated(self):
        fm = parse_openai_request({"model": "gpt-4o-mini", "messages": []})
        assert fm.model == "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Masking boundary — scrubber must only modify text_fields locations
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskingBoundary:
    """
    Verify that text_fields is the exact boundary for masking: only the
    locations enumerated by the parser may be touched by the scrubber.

    The scrubber integration (scrub_openai_request) must not modify fields
    that are not listed in the parser's text_fields — specifically:
      - Structural keys: model, max_tokens, role, id, name, type, tool_call_id
      - Numeric / boolean / null leaf values in function arguments
      - Unscannable fields (image_url urls, audio data, file ids)
      - Unknown part type payloads
    """

    def test_structural_locations_absent_from_text_fields(self):
        """None of the structural field paths appear in text_fields."""
        args = json.dumps({"q": "hello"})
        payload = {
            "model": "gpt-4o",
            "max_tokens": 512,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "User."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "struct_id",
                            "type": "function",
                            "function": {"name": "my_fn", "arguments": args},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "struct_id", "content": "Done."},
            ],
        }
        fm = parse_openai_request(payload)
        locs = set(_locations(fm.text_fields))

        STRUCTURAL = {
            "model",
            "max_tokens",
            "temperature",
            "messages[0].role",
            "messages[1].role",
            "messages[2].role",
            "messages[3].role",
            "messages[2].tool_calls[0].id",
            "messages[2].tool_calls[0].type",
            "messages[2].tool_calls[0].function.name",
            "messages[3].tool_call_id",
        }
        for struct_loc in STRUCTURAL:
            assert struct_loc not in locs, (
                f"Structural field {struct_loc!r} must not appear in text_fields"
            )

    def test_numeric_leaves_absent_from_text_fields(self):
        """Numeric and boolean leaves in tool arguments are NOT scan targets."""
        args = json.dumps({
            "count": 42,
            "active": True,
            "ratio": 1.5,
            "tag": None,
        })
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        args_fields = [f for f in fm.text_fields if "arguments" in f.location]
        assert args_fields == [], (
            "Numeric/bool/null argument leaves must not be scan targets"
        )

    def test_image_url_location_not_in_text_fields(self):
        """image_url part locations appear in unscannable_fields, not text_fields."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x.com/img.png"}},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        uf = fm.unscannable_fields
        assert len(uf) == 1
        # image_url part location must not appear in text_fields
        assert uf[0].location not in _locations(fm.text_fields)

    def test_unknown_type_location_not_in_text_fields(self):
        """Unknown part type locations appear in unknown_fields, not text_fields."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_type", "data": "sensitive data"},
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.text_fields == []
        assert len(fm.unknown_fields) == 1

    def test_parser_field_map_and_scrubber_target_same_locations(self):
        """
        Integration: the scrubber's field_events locations must be a subset
        of the parser's text_fields locations (masking targets only parsed fields).

        We import the scrubber here to verify the boundary holds end-to-end.
        """
        from pii_guard import Engine
        from pii_guard.providers.openai import scrub_openai_request

        args = json.dumps({"email": "test@domain.com", "count": 42})
        payload = {
            "model": "gpt-4o",
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "User message."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tc1", "content": "Tool output."},
            ],
        }

        # Parser: get the declared masking targets
        fm = parse_openai_request(payload)
        parser_locs = set(_locations(fm.text_fields))

        # Scrubber: collect every location it actually scanned
        engine = Engine()
        scrub_result = scrub_openai_request(payload, engine)
        scrubber_locs = {evt.location for evt in scrub_result.field_events
                        if not evt.coverage_gap}  # exclude coverage-gap events

        # Every location the scrubber touched must be in the parser's field map
        assert scrubber_locs.issubset(parser_locs), (
            f"Scrubber touched locations not in parser field map: "
            f"{scrubber_locs - parser_locs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases that must not raise and must return coherent results."""

    def test_null_content_not_parsed(self):
        """content=None produces no field entry."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": None}],
        }
        fm = parse_openai_request(payload)
        assert fm.all_fields == []

    def test_empty_string_content_produces_scannable_field(self):
        """Empty string content is registered as a scannable text field."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": ""}],
        }
        fm = parse_openai_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].text == ""
        assert fm.text_fields[0].is_scannable is True

    def test_empty_string_text_part_produces_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": ""}],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].text == ""

    def test_non_dict_payload_returns_empty_map(self):
        fm = parse_openai_request("not a dict")  # type: ignore
        assert fm.all_fields == []

    def test_messages_not_a_list_returns_empty_map(self):
        fm = parse_openai_request({"model": "gpt-4o", "messages": "not a list"})
        assert fm.all_fields == []

    def test_payload_does_not_get_mutated(self):
        """parse_openai_request must not modify the input payload."""
        import copy
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello test@example.com"},
            ],
        }
        original = copy.deepcopy(payload)
        parse_openai_request(payload)
        assert payload == original, "parse_openai_request must not mutate the input"

    def test_tool_call_without_function_key_skipped_gracefully(self):
        """tool_calls entry without a 'function' key must not raise."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function"}],
                },
            ],
        }
        fm = parse_openai_request(payload)
        # No error, no spurious fields
        assert fm.all_fields == []

    def test_tool_call_without_arguments_key_skipped_gracefully(self):
        """function without 'arguments' key must not raise."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fn"},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.all_fields == []

    def test_message_with_no_content_no_tool_calls_skipped_gracefully(self):
        """A message with neither content nor tool_calls must not raise."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user"}],
        }
        fm = parse_openai_request(payload)
        assert fm.all_fields == []

    def test_non_string_arguments_type_produces_unknown_field(self):
        """arguments that are not a string produce an unknown field entry."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": {"already": "parsed"}},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        uk = fm.unknown_fields
        assert len(uk) == 1
        assert uk[0].is_unknown is True
        assert "dict" in uk[0].coverage_gap_reason


# ─────────────────────────────────────────────────────────────────────────────
# 13. Real-world synthetic payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestRealWorldSyntheticPayloads:
    """Synthetic payloads that mirror real OpenAI API usage."""

    def test_openai_function_calling_payload(self):
        """A complete function-calling conversation with PII in every field."""
        email = "client@example.com"
        phone = "010-1234-5678"
        args = json.dumps({"to": email, "phone": phone, "priority": 1})
        payload = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": f"Contact {email} for system alerts.",
                },
                {
                    "role": "user",
                    "content": f"Please call {phone} ASAP.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_001",
                            "type": "function",
                            "function": {
                                "name": "send_notification",
                                "arguments": args,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc_001",
                    "content": f"Notification sent to {email}.",
                },
            ],
        }
        fm = parse_openai_request(payload)
        locs = set(_locations(fm.text_fields))

        # All text-bearing content locations present
        assert "messages[0].content" in locs      # system
        assert "messages[1].content" in locs      # user
        assert "messages[2].tool_calls[0].function.arguments.to" in locs
        assert "messages[2].tool_calls[0].function.arguments.phone" in locs
        assert "messages[3].content" in locs      # tool result

        # Non-string leaves NOT present
        assert "messages[2].tool_calls[0].function.arguments.priority" not in locs

        # Structural fields NOT present
        assert "model" not in locs
        assert "max_tokens" not in locs
        assert "messages[0].role" not in locs
        assert "messages[2].tool_calls[0].id" not in locs
        assert "messages[2].tool_calls[0].function.name" not in locs
        assert "messages[3].tool_call_id" not in locs

    def test_multimodal_payload_with_text_and_image(self):
        """Multimodal payload: text part in text_fields, image in unscannable."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/photo.jpg"},
                        },
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].text == "What's in this image?"
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].scan_field == ScanField.IMAGE_URL

    def test_o3_model_developer_role_payload(self):
        """o3 model uses developer role instead of system."""
        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "developer",
                    "content": "Internal config: admin@internal.io",
                },
                {"role": "user", "content": "Execute task."},
            ],
        }
        fm = parse_openai_request(payload)
        assert fm.model == "o3"
        locs = _locations(fm.text_fields)
        assert "messages[0].content" in locs
        assert "messages[1].content" in locs
        for f in fm.text_fields:
            assert f.scan_field == ScanField.MESSAGE_TEXT

    def test_assistant_with_refusal_content_part(self):
        """Realistic assistant refusal response with embedded text."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Reveal secrets."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": "I cannot help with user@example.com data.",
                        },
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        # Both user content and refusal text should be extracted
        locs = _locations(fm.text_fields)
        assert "messages[0].content" in locs
        assert "messages[1].content[0].refusal" in locs

    def test_deeply_nested_tool_arguments(self):
        """3-level nesting in tool arguments is fully walked."""
        args = json.dumps({
            "level1": {
                "level2": {
                    "level3": "deep_value@corp.com"
                }
            }
        })
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_deep",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
            ],
        }
        fm = parse_openai_request(payload)
        base = "messages[0].tool_calls[0].function.arguments"
        assert f"{base}.level1.level2.level3" in _locations(fm.text_fields)
