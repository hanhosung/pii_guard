"""
Unit tests for the Gemini provider request parser (Sub-AC 3).

These tests verify that:
  1. parse_gemini_request extracts the correct field set from real and
     synthetic Gemini API payloads.
  2. Every text-bearing location is classified with the correct ScanField.
  3. Masking targets ONLY parsed (text_fields) — structural fields such as
     model, generationConfig, safetySettings, role, name, language, outcome,
     mimeType, and fileUri do NOT appear as scan targets.
  4. Unscannable fields (inlineData, fileData) appear in unscannable_fields,
     not in text_fields.
  5. Unknown part types appear in unknown_fields with a coverage_gap_reason.
  6. functionCall.args and functionResponse.response are recursively expanded;
     non-string leaves (int, float, bool, None) are excluded.
  7. Both camelCase and snake_case field-name variants are handled identically.
  8. systemInstruction as string shorthand and dict-with-parts are both parsed.
  9. Edge cases (empty payload, missing keys, None values) do not raise.
  10. GeminiFieldMap helpers (text_fields, unscannable_fields, unknown_fields,
      get_field) behave correctly.
  11. The original payload is never mutated.
  12. Masking boundary: text_fields is the exact set the scrubber may modify.
"""
from __future__ import annotations

import copy

import pytest

from pii_guard.providers.gemini_parser import (
    GeminiFieldMap,
    ParsedField,
    ScanField,
    parse_gemini_request,
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
# 1. Basic structural field extraction — empty / minimal payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicFieldExtraction:
    """Core extraction of text-bearing fields; structural fields excluded."""

    def test_empty_payload_returns_empty_map(self):
        fm = parse_gemini_request({})
        assert fm.all_fields == []
        assert fm.text_fields == []
        assert fm.unscannable_fields == []
        assert fm.unknown_fields == []

    def test_payload_without_contents_returns_empty_map(self):
        fm = parse_gemini_request({"model": "gemini-2.0-flash"})
        assert fm.all_fields == []

    def test_model_captured_not_as_scan_target(self):
        """model field is recorded in GeminiFieldMap.model but NOT in text_fields."""
        fm = parse_gemini_request({"model": "gemini-2.0-flash", "contents": []})
        assert fm.model == "gemini-2.0-flash"
        assert not any(f.location == "model" for f in fm.text_fields)

    def test_model_none_when_not_string(self):
        """Non-string model value → fm.model is None."""
        fm = parse_gemini_request({"model": 42, "contents": []})
        assert fm.model is None

    def test_api_version_propagated(self):
        fm = parse_gemini_request({}, api_version="v1beta")
        assert fm.api_version == "v1beta"

    def test_structural_top_level_fields_not_in_text_fields(self):
        """generationConfig, safetySettings, tools — never scan targets."""
        payload = {
            "model": "gemini-2.0-flash",
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}],
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        for key in ("model", "generationConfig", "safetySettings", "tools"):
            assert key not in locs, (
                f"Structural field {key!r} must not be a scan target"
            )

    def test_content_role_field_not_in_text_fields(self):
        """contents[*].role is structural and must never be a scan target."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].role" not in locs

    def test_non_dict_payload_returns_empty_map(self):
        fm = parse_gemini_request("not a dict")  # type: ignore
        assert fm.all_fields == []

    def test_empty_contents_list_returns_empty_map(self):
        fm = parse_gemini_request({"model": "gemini-2.0-flash", "contents": []})
        assert fm.all_fields == []

    def test_non_list_contents_ignored(self):
        fm = parse_gemini_request({"contents": "not a list"})
        assert fm.all_fields == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. systemInstruction parsing — string shorthand
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemInstructionStringShorthand:
    """systemInstruction as a plain string (shorthand form)."""

    def test_string_system_instruction_produces_one_field(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": "You are a helpful assistant.",
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 1
        assert si_fields[0].location == "systemInstruction"
        assert si_fields[0].text == "You are a helpful assistant."
        assert si_fields[0].is_scannable is True

    def test_string_system_instruction_text_content_matches(self):
        text = "Contact admin@corp.io for support."
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": text,
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        sf = fm.get_field("systemInstruction")
        assert sf is not None
        assert sf.text == text

    def test_empty_string_system_instruction_produces_field(self):
        payload = {"systemInstruction": "", "contents": []}
        fm = parse_gemini_request(payload)
        sf = fm.get_field("systemInstruction")
        assert sf is not None
        assert sf.is_scannable is True
        assert sf.text == ""

    def test_snake_case_string_system_instruction(self):
        """system_instruction (snake_case) string form is handled identically."""
        payload = {
            "model": "gemini-2.0-flash",
            "system_instruction": "Alert dev@snake.io always.",
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 1
        assert si_fields[0].location == "system_instruction"
        assert "dev@snake.io" in si_fields[0].text


# ─────────────────────────────────────────────────────────────────────────────
# 3. systemInstruction parsing — dict with parts
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemInstructionDictWithParts:
    """systemInstruction as a dict Content object with a parts list."""

    def test_single_text_part_produces_one_field(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": "Always notify admin@corp.io."}]
            },
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 1
        assert si_fields[0].location == "systemInstruction.parts[0].text"
        assert si_fields[0].text == "Always notify admin@corp.io."
        assert si_fields[0].is_scannable is True

    def test_multiple_text_parts_each_produce_a_field(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [
                    {"text": "Part one."},
                    {"text": "Part two."},
                ]
            },
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 2
        locs = _locations(si_fields)
        assert "systemInstruction.parts[0].text" in locs
        assert "systemInstruction.parts[1].text" in locs

    def test_role_field_in_system_instruction_dict_not_a_scan_target(self):
        """role inside systemInstruction dict is structural — not a masking target."""
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "Be helpful."}],
            },
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "systemInstruction.role" not in locs

    def test_missing_parts_key_produces_unknown_field(self):
        """systemInstruction dict without a 'parts' key is an unknown field."""
        payload = {
            "systemInstruction": {"role": "system"},
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert uf.is_unknown is True
        assert "parts" in uf.coverage_gap_reason

    def test_non_list_parts_produces_unknown_field(self):
        """systemInstruction with parts as a non-list is an unknown field."""
        payload = {
            "systemInstruction": {"parts": "not a list"},
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1

    def test_snake_case_system_instruction_dict(self):
        """system_instruction (snake_case) dict form is handled identically."""
        payload = {
            "model": "gemini-2.0-flash",
            "system_instruction": {
                "parts": [{"text": "Contact admin@snake.io for issues."}]
            },
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 1
        assert si_fields[0].location == "system_instruction.parts[0].text"
        assert "admin@snake.io" in si_fields[0].text

    def test_system_instruction_unexpected_type_produces_unknown_field(self):
        """systemInstruction with an unexpected type (int, list, etc.) is unknown."""
        payload = {"systemInstruction": 42, "contents": []}
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert uf.location == "systemInstruction"
        assert "int" in uf.coverage_gap_reason


# ─────────────────────────────────────────────────────────────────────────────
# 4. contents[*].parts[*].text — message text parts
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageTextParts:
    """Regular text parts in contents produce MESSAGE_TEXT fields."""

    def test_single_text_part_produces_message_text_field(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Send to billing@company.com"}]},
            ],
        }
        fm = parse_gemini_request(payload)
        tf = fm.text_fields
        assert len(tf) == 1
        assert tf[0].location == "contents[0].parts[0].text"
        assert tf[0].scan_field == ScanField.MESSAGE_TEXT
        assert tf[0].text == "Send to billing@company.com"
        assert tf[0].is_scannable is True

    def test_multiple_text_parts_each_registered(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "First part"},
                        {"text": "Second part"},
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        tf = fm.text_fields
        assert len(tf) == 2
        locs = _locations(tf)
        assert "contents[0].parts[0].text" in locs
        assert "contents[0].parts[1].text" in locs

    def test_multi_turn_each_content_item_produces_fields(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi there"}]},
                {"role": "user", "parts": [{"text": "Bye"}]},
            ],
        }
        fm = parse_gemini_request(payload)
        tf = fm.text_fields
        assert len(tf) == 3
        locs = _locations(tf)
        assert "contents[0].parts[0].text" in locs
        assert "contents[1].parts[0].text" in locs
        assert "contents[2].parts[0].text" in locs

    def test_empty_text_part_still_registered(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": ""}]}]
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].text == ""

    def test_content_item_without_parts_key_skipped(self):
        """Content items without a 'parts' key are skipped without error."""
        payload = {
            "contents": [
                {"role": "user"},  # no parts key
                {"role": "model", "parts": [{"text": "Hello"}]},
            ]
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "contents[1].parts[0].text"

    def test_non_dict_content_item_skipped(self):
        """Non-dict entries in contents are skipped without error."""
        payload = {
            "contents": [
                "not a dict",
                {"role": "user", "parts": [{"text": "Hi"}]},
            ]
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1

    def test_non_dict_part_skipped(self):
        """Non-dict entries in parts are skipped without error."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        "not a dict part",
                        {"text": "Hello world"},
                    ],
                }
            ]
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "contents[0].parts[1].text"

    def test_message_text_scan_field_value(self):
        payload = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        fm = parse_gemini_request(payload)
        assert fm.text_fields[0].scan_field == ScanField.MESSAGE_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# 5. functionCall.args — function call arguments (JSON object)
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionCallArgsParsing:
    """functionCall.args is a JSON object; all string leaf values are registered."""

    def test_flat_string_values_extracted(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_email",
                                "args": {"to": "user@example.com", "subject": "Hello"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        locs = _locations(fc_fields)
        assert "contents[0].parts[0].functionCall.args.to" in locs
        assert "contents[0].parts[0].functionCall.args.subject" in locs

    def test_string_values_have_correct_text(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "fn",
                                "args": {"email": "target@corp.com"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert len(fc_fields) == 1
        assert fc_fields[0].text == "target@corp.com"

    def test_nested_dict_args_recursively_expanded(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "store_contact",
                                "args": {
                                    "contact": {
                                        "email": "deep@nested.com",
                                        "phone": "010-1111-2222",
                                        "metadata": {"source": "CRM"},
                                    }
                                },
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        )
        base = "contents[0].parts[0].functionCall.args"
        assert f"{base}.contact.email" in locs
        assert f"{base}.contact.phone" in locs
        assert f"{base}.contact.metadata.source" in locs

    def test_array_values_in_args_recursively_expanded(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "notify",
                                "args": {"emails": ["a@b.com", "c@d.io", "safe"]},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        )
        base = "contents[0].parts[0].functionCall.args"
        assert f"{base}.emails[0]" in locs
        assert f"{base}.emails[1]" in locs
        assert f"{base}.emails[2]" in locs

    def test_numeric_boolean_null_values_not_in_text_fields(self):
        """Numbers, booleans, and null in args must NOT be scan targets."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "set_params",
                                "args": {
                                    "count": 42,
                                    "enabled": True,
                                    "ratio": 3.14,
                                    "nothing": None,
                                },
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert fc_fields == [], (
            "Numeric/boolean/None values in functionCall.args must not be scan targets"
        )

    def test_function_call_name_not_a_scan_target(self):
        """functionCall.name is a structural field and must not be a masking target."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "my_function_name",
                                "args": {"q": "hello"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].functionCall.name" not in locs

    def test_deeply_nested_function_call_args(self):
        """3+ level nesting inside functionCall.args is fully walked."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "configure",
                                "args": {
                                    "level1": {
                                        "level2": {
                                            "level3": {"token": "ghp_abc"}
                                        }
                                    }
                                },
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        base = "contents[0].parts[0].functionCall.args"
        locs = _locations(fm.text_fields)
        assert f"{base}.level1.level2.level3.token" in locs

    def test_function_call_snake_case_key(self):
        """function_call (snake_case) is handled identically to functionCall."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "send_email",
                                "args": {"to": "snake@domain.com"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        )
        assert "contents[0].parts[0].function_call.args.to" in locs

    def test_function_call_scan_field_value(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "fn",
                                "args": {"msg": "hello"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert fc_fields
        assert all(f.scan_field == ScanField.FUNCTION_CALL_ARGS for f in fc_fields)

    def test_empty_args_produces_no_fields(self):
        """Empty functionCall.args produces no scan target fields."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "fn", "args": {}}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert fc_fields == []

    def test_mixed_string_and_numeric_args(self):
        """String values in mixed args are extracted; numeric skipped."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "op",
                                "args": {"name": "Alice", "age": 30, "active": False},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        )
        base = "contents[0].parts[0].functionCall.args"
        assert f"{base}.name" in locs
        assert f"{base}.age" not in locs
        assert f"{base}.active" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 6. functionResponse.response — function response values
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionResponseParsing:
    """functionResponse.response is walked recursively for string leaves."""

    def test_flat_string_values_extracted(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup_user",
                                "response": {"email": "user@response.com", "status": "active"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        locs = _locations(fr_fields)
        base = "contents[0].parts[0].functionResponse.response"
        assert f"{base}.email" in locs
        assert f"{base}.status" in locs

    def test_nested_response_recursively_expanded(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_contact",
                                "response": {
                                    "data": {
                                        "email": "resp@nested.com",
                                        "phone": "010-7777-8888",
                                    }
                                },
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        )
        base = "contents[0].parts[0].functionResponse.response"
        assert f"{base}.data.email" in locs
        assert f"{base}.data.phone" in locs

    def test_numeric_values_not_in_text_fields(self):
        """Numeric and boolean response values must NOT be scan targets."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "calc",
                                "response": {"result": 42, "success": True, "error": None},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        assert fr_fields == []

    def test_function_response_name_not_a_scan_target(self):
        """functionResponse.name is structural and must not be a masking target."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_user_info",
                                "response": {"info": "user@test.com"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].functionResponse.name" not in locs

    def test_function_response_snake_case_key(self):
        """function_response (snake_case) is handled identically to functionResponse."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "get_user",
                                "response": {"email": "resp@snake.io"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        )
        assert "contents[0].parts[0].function_response.response.email" in locs

    def test_deeply_nested_function_response(self):
        """3-level nesting in functionResponse.response is fully walked."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {
                                    "level1": {"level2": {"level3": {"contact": "deep@nested.io"}}}
                                },
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        base = "contents[0].parts[0].functionResponse.response"
        locs = _locations(
            [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        )
        assert f"{base}.level1.level2.level3.contact" in locs

    def test_function_response_scan_field_value(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "check",
                                "response": {"result": "Contact admin@corp.io"},
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        fr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        assert fr_fields
        assert all(f.scan_field == ScanField.FUNCTION_RESPONSE for f in fr_fields)


# ─────────────────────────────────────────────────────────────────────────────
# 7. executableCode.code — source code text
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutableCodeParsing:
    """executableCode.code produces an EXECUTABLE_CODE text field."""

    def test_executable_code_code_registered(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": "send_email('admin@code.com', 'Hello')",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        ec_fields = [f for f in fm.text_fields if f.scan_field == ScanField.EXECUTABLE_CODE]
        assert len(ec_fields) == 1
        assert ec_fields[0].location == "contents[0].parts[0].executableCode.code"
        assert ec_fields[0].text == "send_email('admin@code.com', 'Hello')"
        assert ec_fields[0].is_scannable is True

    def test_executable_code_language_field_not_a_scan_target(self):
        """executableCode.language is structural — not a masking target."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"language": "JAVASCRIPT", "code": "const x = 1;"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].executableCode.language" not in locs

    def test_executable_code_snake_case_key(self):
        """executable_code (snake_case) is handled identically to executableCode."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executable_code": {
                                "language": "PYTHON",
                                "code": "contact('user@snake.com')",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        ec_fields = [f for f in fm.text_fields if f.scan_field == ScanField.EXECUTABLE_CODE]
        assert len(ec_fields) == 1
        assert ec_fields[0].location == "contents[0].parts[0].executable_code.code"

    def test_executable_code_scan_field_value(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"language": "PYTHON", "code": "print('hello')"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        ec_fields = [f for f in fm.text_fields if f.scan_field == ScanField.EXECUTABLE_CODE]
        assert ec_fields[0].scan_field == ScanField.EXECUTABLE_CODE

    def test_empty_code_field_still_registered(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"language": "PYTHON", "code": ""}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        ec_fields = [f for f in fm.text_fields if f.scan_field == ScanField.EXECUTABLE_CODE]
        assert len(ec_fields) == 1
        assert ec_fields[0].text == ""


# ─────────────────────────────────────────────────────────────────────────────
# 8. codeExecutionResult.output — code execution output
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExecutionResultParsing:
    """codeExecutionResult.output produces a CODE_EXECUTION_RESULT text field."""

    def test_code_execution_result_output_registered(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "Result: user@output.com logged in",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        cer_fields = [
            f for f in fm.text_fields if f.scan_field == ScanField.CODE_EXECUTION_RESULT
        ]
        assert len(cer_fields) == 1
        assert cer_fields[0].location == "contents[0].parts[0].codeExecutionResult.output"
        assert cer_fields[0].text == "Result: user@output.com logged in"
        assert cer_fields[0].is_scannable is True

    def test_code_execution_result_outcome_not_a_scan_target(self):
        """codeExecutionResult.outcome is structural — not a masking target."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"codeExecutionResult": {"outcome": "OUTCOME_FAILED", "output": "Error"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].codeExecutionResult.outcome" not in locs

    def test_code_execution_result_snake_case_key(self):
        """code_execution_result (snake_case) is handled identically."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "code_execution_result": {
                                "outcome": "OUTCOME_OK",
                                "output": "Logged in as admin@snake.io",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        cer_fields = [
            f for f in fm.text_fields if f.scan_field == ScanField.CODE_EXECUTION_RESULT
        ]
        assert len(cer_fields) == 1
        assert cer_fields[0].location == "contents[0].parts[0].code_execution_result.output"

    def test_code_execution_result_scan_field_value(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "Done"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        cer_fields = [
            f for f in fm.text_fields if f.scan_field == ScanField.CODE_EXECUTION_RESULT
        ]
        assert cer_fields[0].scan_field == ScanField.CODE_EXECUTION_RESULT


# ─────────────────────────────────────────────────────────────────────────────
# 9. inlineData / inline_data — unscannable binary parts
# ─────────────────────────────────────────────────────────────────────────────

class TestInlineDataParsing:
    """inlineData parts produce INLINE_DATA unscannable entries."""

    def test_inline_data_in_unscannable_not_text_fields(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgAB...",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert fm.text_fields == []
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert uf.scan_field == ScanField.INLINE_DATA
        assert uf.is_unscannable is True
        assert uf.is_scannable is False
        assert uf.text is None

    def test_inline_data_location_correct(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "abc"}}],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        uf = fm.unscannable_fields[0]
        assert uf.location == "contents[0].parts[0].inlineData"

    def test_inline_data_coverage_gap_reason_set(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "audio/mp3", "data": "abc"}}],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        uf = fm.unscannable_fields[0]
        assert uf.coverage_gap_reason is not None
        assert len(uf.coverage_gap_reason) > 0

    def test_inline_data_snake_case_key(self):
        """inline_data (snake_case) is handled identically to inlineData."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": "base64encodeddata",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert uf.scan_field == ScanField.INLINE_DATA
        assert uf.location == "contents[0].parts[0].inline_data"

    def test_mime_type_not_a_scan_target(self):
        """inlineData.mimeType is structural and must not be a masking target."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "abc"}}],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].inlineData.mimeType" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 10. fileData / file_data — unscannable external file references
# ─────────────────────────────────────────────────────────────────────────────

class TestFileDataParsing:
    """fileData parts produce FILE_DATA unscannable entries."""

    def test_file_data_in_unscannable_not_text_fields(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "gs://bucket/doc.pdf",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert fm.text_fields == []
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert uf.scan_field == ScanField.FILE_DATA
        assert uf.is_unscannable is True
        assert uf.is_scannable is False
        assert uf.text is None

    def test_file_data_location_correct(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "video/mp4", "fileUri": "gs://bucket/v.mp4"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        uf = fm.unscannable_fields[0]
        assert uf.location == "contents[0].parts[0].fileData"

    def test_file_data_coverage_gap_reason_set(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "text/plain", "fileUri": "gs://b/f.txt"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        uf = fm.unscannable_fields[0]
        assert uf.coverage_gap_reason is not None

    def test_file_data_snake_case_key(self):
        """file_data (snake_case) is handled identically to fileData."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "file_uri": "gs://bucket/doc.pdf",
                            }
                        }
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unscannable_fields) == 1
        uf = fm.unscannable_fields[0]
        assert uf.scan_field == ScanField.FILE_DATA
        assert uf.location == "contents[0].parts[0].file_data"

    def test_file_uri_not_a_scan_target(self):
        """fileData.fileUri is structural — not a masking target."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/f.pdf"}}
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].fileData.fileUri" not in locs


# ─────────────────────────────────────────────────────────────────────────────
# 11. Unknown part types — coverage alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownPartTypes:
    """Unrecognized part types appear in unknown_fields with coverage_gap_reason."""

    def test_unknown_part_type_in_unknown_fields(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"future_part_type": {"data": "something"}}],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert uf.scan_field == ScanField.UNKNOWN
        assert uf.is_unknown is True
        assert uf.is_scannable is False
        assert "future_part_type" in uf.coverage_gap_reason

    def test_unknown_part_not_in_text_fields(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"weird_type": {"value": "xyz"}}],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert fm.text_fields == []
        assert len(fm.unknown_fields) == 1

    def test_unknown_part_in_system_instruction(self):
        """Unknown part type in systemInstruction also triggers a coverage alarm."""
        payload = {
            "systemInstruction": {
                "parts": [
                    {"text": "Be helpful"},
                    {"future_block": {"content": "xyz"}},
                ]
            },
            "contents": [],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert "future_block" in uf.coverage_gap_reason

    def test_mix_known_and_unknown_parts(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Hello"},
                        {"unknown_type": {"data": "X"}},
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1
        assert len(fm.unknown_fields) == 1

    def test_empty_part_dict_produces_unknown_field(self):
        """An empty dict {} in parts has no recognized keys → unknown field."""
        payload = {
            "contents": [
                {"role": "user", "parts": [{}]},
            ]
        }
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1
        uf = fm.unknown_fields[0]
        assert "(empty)" in uf.coverage_gap_reason


# ─────────────────────────────────────────────────────────────────────────────
# 12. Mixed parts — multiple field types in one request
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedParts:
    """Multiple part types in one content item."""

    def test_text_and_inline_data_parts(self):
        """Text part → text_fields; inlineData → unscannable_fields."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Describe this image."},
                        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "contents[0].parts[0].text"
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].scan_field == ScanField.INLINE_DATA

    def test_text_and_function_call_parts(self):
        """text + functionCall in same content item — both extracted."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"text": "I'll call the function."},
                        {"functionCall": {"name": "do_thing", "args": {"param": "value@corp.com"}}},
                    ],
                }
            ],
        }
        fm = parse_gemini_request(payload)
        tf_locs = _locations(fm.text_fields)
        assert "contents[0].parts[0].text" in tf_locs
        assert "contents[0].parts[1].functionCall.args.param" in tf_locs
        assert len(fm.text_fields) == 2

    def test_all_gemini_field_types_in_one_request(self):
        """All supported field types in one payload produce the correct field map."""
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": "System: admin@corp.io"}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "User text"},
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"data": "resp data"},
                            }
                        },
                        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/f.pdf"}},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "send", "args": {"to": "x@y.com"}}},
                        {"executableCode": {"language": "PYTHON", "code": "print('hi')"}},
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "Done"}},
                    ],
                },
            ],
        }
        fm = parse_gemini_request(payload)

        # Text fields
        text_locs = _locations(fm.text_fields)
        assert "systemInstruction.parts[0].text" in text_locs
        assert "contents[0].parts[0].text" in text_locs
        assert "contents[0].parts[1].functionResponse.response.data" in text_locs
        assert "contents[1].parts[0].functionCall.args.to" in text_locs
        assert "contents[1].parts[1].executableCode.code" in text_locs
        assert "contents[1].parts[2].codeExecutionResult.output" in text_locs

        # Unscannable fields
        unscan_locs = _locations(fm.unscannable_fields)
        assert "contents[0].parts[2].inlineData" in unscan_locs
        assert "contents[0].parts[3].fileData" in unscan_locs

        # No unknown fields
        assert fm.unknown_fields == []


# ─────────────────────────────────────────────────────────────────────────────
# 13. snake_case field alias coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestSnakeCaseFieldAliases:
    """All camelCase/snake_case aliases produce identical results."""

    def test_all_snake_case_aliases_in_one_payload(self):
        """snake_case part keys produce the same field map as camelCase equivalents."""
        payload = {
            "model": "gemini-2.0-flash",
            "system_instruction": {"parts": [{"text": "System"}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "User text"},
                        {
                            "function_response": {
                                "name": "fn",
                                "response": {"val": "snake_resp"},
                            }
                        },
                        {"inline_data": {"mime_type": "image/jpeg", "data": "abc"}},
                        {"file_data": {"mime_type": "application/pdf", "file_uri": "gs://b/f.pdf"}},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"function_call": {"name": "fn", "args": {"to": "x@snake.com"}}},
                        {"executable_code": {"language": "PYTHON", "code": "pass"}},
                        {"code_execution_result": {"outcome": "OUTCOME_OK", "output": "Done"}},
                    ],
                },
            ],
        }
        fm = parse_gemini_request(payload)

        # system_instruction
        si_fields = [f for f in fm.text_fields if f.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert len(si_fields) == 1
        assert si_fields[0].location == "system_instruction.parts[0].text"

        # function_response
        fr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        assert any("function_response" in f.location for f in fr_fields)

        # function_call
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert any("function_call" in f.location for f in fc_fields)

        # executable_code
        ec_fields = [f for f in fm.text_fields if f.scan_field == ScanField.EXECUTABLE_CODE]
        assert any("executable_code" in f.location for f in ec_fields)

        # code_execution_result
        cer_fields = [f for f in fm.text_fields if f.scan_field == ScanField.CODE_EXECUTION_RESULT]
        assert any("code_execution_result" in f.location for f in cer_fields)

        # inline_data
        id_fields = [f for f in fm.unscannable_fields if f.scan_field == ScanField.INLINE_DATA]
        assert any("inline_data" in f.location for f in id_fields)

        # file_data
        fd_fields = [f for f in fm.unscannable_fields if f.scan_field == ScanField.FILE_DATA]
        assert any("file_data" in f.location for f in fd_fields)


# ─────────────────────────────────────────────────────────────────────────────
# 14. GeminiFieldMap helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiFieldMapHelpers:
    """Unit tests for GeminiFieldMap property and method helpers."""

    def _make_map(self) -> GeminiFieldMap:
        fm = GeminiFieldMap()
        fm.all_fields.append(ParsedField(
            location="systemInstruction.parts[0].text",
            scan_field=ScanField.SYSTEM_INSTRUCTION,
            text="Hello",
            is_scannable=True,
        ))
        fm.all_fields.append(ParsedField(
            location="contents[0].parts[0].inlineData",
            scan_field=ScanField.INLINE_DATA,
            text=None,
            is_scannable=False,
            is_unscannable=True,
            coverage_gap_reason="binary data",
        ))
        fm.all_fields.append(ParsedField(
            location="contents[0].parts[1]",
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason="unrecognized part",
        ))
        return fm

    def test_text_fields_returns_only_scannable(self):
        fm = self._make_map()
        assert len(fm.text_fields) == 1
        assert fm.text_fields[0].location == "systemInstruction.parts[0].text"

    def test_unscannable_fields_returns_only_unscannable(self):
        fm = self._make_map()
        assert len(fm.unscannable_fields) == 1
        assert fm.unscannable_fields[0].location == "contents[0].parts[0].inlineData"

    def test_unknown_fields_returns_only_unknown(self):
        fm = self._make_map()
        assert len(fm.unknown_fields) == 1
        assert fm.unknown_fields[0].location == "contents[0].parts[1]"

    def test_get_field_existing_location(self):
        fm = self._make_map()
        pf = fm.get_field("systemInstruction.parts[0].text")
        assert pf is not None
        assert pf.text == "Hello"

    def test_get_field_missing_location_returns_none(self):
        fm = self._make_map()
        assert fm.get_field("nonexistent.path") is None

    def test_all_fields_count(self):
        fm = self._make_map()
        assert len(fm.all_fields) == 3

    def test_all_fields_is_superset_of_text_unscannable_unknown(self):
        """all_fields must contain every text + unscannable + unknown field."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Hi"},
                        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                        {"future_type": {"data": "x"}},
                    ],
                },
            ],
        }
        fm = parse_gemini_request(payload)
        combined = fm.text_fields + fm.unscannable_fields + fm.unknown_fields
        assert sorted(_locations(fm.all_fields)) == sorted(_locations(combined))

    def test_model_attribute_populated(self):
        fm = parse_gemini_request({"model": "gemini-2.0-flash", "contents": []})
        assert fm.model == "gemini-2.0-flash"

    def test_model_attribute_none_when_absent(self):
        fm = parse_gemini_request({"contents": []})
        assert fm.model is None


# ─────────────────────────────────────────────────────────────────────────────
# 15. Masking boundary — scrubber must only modify text_fields locations
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskingBoundary:
    """
    Critical invariant: text_fields is the exclusive set of locations the
    scrubber may modify.  Every other key in the payload is off-limits.
    """

    def test_structural_locations_absent_from_text_fields(self):
        """None of the structural field paths appear in text_fields."""
        payload = {
            "model": "gemini-2.0-flash",
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            "systemInstruction": {"parts": [{"text": "Be helpful."}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "My email: user@example.com"},
                        {
                            "functionCall": {
                                "name": "greet",
                                "args": {"greeting": "hello"},
                            }
                        },
                    ],
                },
            ],
        }
        fm = parse_gemini_request(payload)
        locs = set(_locations(fm.text_fields))

        STRUCTURAL = {
            "model",
            "generationConfig",
            "safetySettings",
            "contents[0].role",
            "contents[0].parts[0].text",   # This IS in text_fields, so skip
        }
        for key in ("model", "generationConfig", "safetySettings"):
            assert key not in locs, (
                f"Structural field {key!r} must not appear in text_fields"
            )
        # Role is structural
        assert "contents[0].role" not in locs
        # Function name is structural
        assert "contents[0].parts[1].functionCall.name" not in locs

    def test_all_text_fields_are_content_paths(self):
        """
        Every text_field location must be a recognized content path
        (contains .text, .code, .output, .args, or .response).
        """
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": "System prompt."}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "User message."},
                        {
                            "functionResponse": {
                                "name": "fn",
                                "response": {"result": "some text"},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "fn", "args": {"param": "value"}}},
                        {"executableCode": {"language": "PYTHON", "code": "pass"}},
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "done"}},
                    ],
                },
            ],
        }
        fm = parse_gemini_request(payload)
        for pf in fm.text_fields:
            assert any(
                seg in pf.location
                for seg in (".text", ".code", ".output", ".args", ".response")
            ), f"Unexpected scan target location: {pf.location!r}"

    def test_parser_field_map_and_scrubber_target_same_locations(self):
        """
        Integration check: the scrubber's field_events locations should be a
        subset of the parser's text_fields locations (masking targets only
        parsed fields).
        """
        from pii_guard import Engine
        from pii_guard.providers.gemini import scrub_gemini_request

        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": "Admin: admin@corp.io"}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Contact user@example.com"},
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"email": "found@example.com"},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_email",
                                "args": {"to": "target@domain.com", "count": 1},
                            }
                        },
                        {"executableCode": {"language": "PYTHON", "code": "print('hi')"}},
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "ok"}},
                    ],
                },
            ],
        }

        # Parser: get declared masking targets
        fm = parse_gemini_request(payload)
        parser_locs = set(_locations(fm.text_fields))

        # Scrubber: collect every location it actually scanned
        engine = Engine()
        scrub_result = scrub_gemini_request(payload, engine)
        scrubber_locs = {
            evt.location for evt in scrub_result.field_events
            if not evt.coverage_gap  # exclude coverage-gap events
        }

        # Every location the scrubber touched must be in the parser's field map
        assert scrubber_locs.issubset(parser_locs), (
            f"Scrubber touched locations not in parser field map: "
            f"{scrubber_locs - parser_locs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 16. Original payload not mutated
# ─────────────────────────────────────────────────────────────────────────────

class TestOriginalPayloadNotMutated:
    """parse_gemini_request must not modify the caller's payload dict."""

    def test_parse_does_not_mutate_original(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": "Admin: admin@corp.io"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "user@example.com"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "fn",
                                "args": {"contact": "x@y.com"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "fn",
                                "response": {"result": "x@y.com"},
                            }
                        }
                    ],
                },
            ],
        }
        original_copy = copy.deepcopy(payload)
        parse_gemini_request(payload)
        assert payload == original_copy, "parse_gemini_request must not mutate the input"


# ─────────────────────────────────────────────────────────────────────────────
# 17. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases that must not raise and must return coherent results."""

    def test_empty_payload_does_not_raise(self):
        fm = parse_gemini_request({})
        assert fm.all_fields == []

    def test_no_contents_key(self):
        fm = parse_gemini_request({"model": "gemini-2.0-flash"})
        assert fm.all_fields == []

    def test_empty_parts_list(self):
        fm = parse_gemini_request(
            {"model": "gemini-2.0-flash", "contents": [{"role": "user", "parts": []}]}
        )
        assert fm.all_fields == []

    def test_non_list_parts_skipped(self):
        fm = parse_gemini_request(
            {"contents": [{"role": "user", "parts": "not a list"}]}
        )
        assert fm.all_fields == []

    def test_large_multi_turn_no_errors(self):
        """Stress: 20 turns with mixed part types should produce no errors."""
        messages = []
        for i in range(20):
            messages.append({
                "role": "user" if i % 2 == 0 else "model",
                "parts": [{"text": f"Turn {i} message."}],
            })
        payload = {"model": "gemini-2.0-flash", "contents": messages}
        fm = parse_gemini_request(payload)
        assert len(fm.text_fields) == 20
        assert fm.unknown_fields == []
        assert fm.unscannable_fields == []

    def test_function_call_with_none_args(self):
        """functionCall.args == None → no fields produced (not scannable)."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "ping", "args": None}}
                    ],
                }
            ]
        }
        fm = parse_gemini_request(payload)
        fc_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert fc_fields == []

    def test_function_response_with_none_response(self):
        """functionResponse.response == None → no fields produced."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "fn", "response": None}}
                    ],
                }
            ]
        }
        fm = parse_gemini_request(payload)
        fr_fields = [f for f in fm.text_fields if f.scan_field == ScanField.FUNCTION_RESPONSE]
        assert fr_fields == []

    def test_system_instruction_none_value_is_unknown(self):
        """systemInstruction == None triggers an unknown field entry."""
        payload = {"systemInstruction": None, "contents": []}
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1

    def test_system_instruction_integer_is_unknown(self):
        payload = {"systemInstruction": 42, "contents": []}
        fm = parse_gemini_request(payload)
        assert len(fm.unknown_fields) == 1

    def test_model_integer_gives_none(self):
        """Non-string model values are ignored."""
        fm = parse_gemini_request({"model": 42})
        assert fm.model is None


# ─────────────────────────────────────────────────────────────────────────────
# 18. ScanField enum values
# ─────────────────────────────────────────────────────────────────────────────

class TestScanFieldEnum:
    """Verify ScanField enum values match ontology scan_field strings."""

    def test_system_instruction_value(self):
        assert ScanField.SYSTEM_INSTRUCTION.value == "system_instruction"

    def test_message_text_value(self):
        assert ScanField.MESSAGE_TEXT.value == "message_text"

    def test_function_call_args_value(self):
        assert ScanField.FUNCTION_CALL_ARGS.value == "function_call_args"

    def test_function_response_value(self):
        assert ScanField.FUNCTION_RESPONSE.value == "function_response"

    def test_executable_code_value(self):
        assert ScanField.EXECUTABLE_CODE.value == "executable_code"

    def test_code_execution_result_value(self):
        assert ScanField.CODE_EXECUTION_RESULT.value == "code_execution_result"

    def test_inline_data_value(self):
        assert ScanField.INLINE_DATA.value == "inline_data"

    def test_file_data_value(self):
        assert ScanField.FILE_DATA.value == "file_data"

    def test_unknown_value(self):
        assert ScanField.UNKNOWN.value == "unknown"

    def test_all_nine_fields_defined(self):
        expected = {
            "system_instruction", "message_text", "function_call_args",
            "function_response", "executable_code", "code_execution_result",
            "inline_data", "file_data", "unknown",
        }
        actual = {sf.value for sf in ScanField}
        assert actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# 19. Full mixed payload — real-world synthetic request
# ─────────────────────────────────────────────────────────────────────────────

class TestFullMixedPayload:
    """
    End-to-end parse of a realistic multi-content-type Gemini payload.
    Asserts the complete expected field map for a real-world-like request.
    """

    def _build_payload(self):
        return {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": "System: admin@internal.corp"}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "User message: user@example.com"},
                        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "I see you have an email."},
                        {
                            "functionCall": {
                                "name": "send_email",
                                "args": {
                                    "to": "user@example.com",
                                    "subject": "Hello",
                                    "body": {"greeting": "Dear user@example.com"},
                                },
                            }
                        },
                        {"executableCode": {"language": "PYTHON", "code": "print('done')"}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "send_email",
                                "response": {"status": "sent", "to": "user@example.com"},
                            }
                        },
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "Email sent to user@example.com",
                            }
                        },
                        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/f.pdf"}},
                    ],
                },
            ],
        }

    def test_all_text_fields_present(self):
        fm = parse_gemini_request(self._build_payload())
        locs = set(_locations(fm.text_fields))
        expected = {
            "systemInstruction.parts[0].text",
            "contents[0].parts[0].text",
            "contents[1].parts[0].text",
            "contents[1].parts[1].functionCall.args.to",
            "contents[1].parts[1].functionCall.args.subject",
            "contents[1].parts[1].functionCall.args.body.greeting",
            "contents[1].parts[2].executableCode.code",
            "contents[2].parts[0].functionResponse.response.status",
            "contents[2].parts[0].functionResponse.response.to",
            "contents[2].parts[1].codeExecutionResult.output",
        }
        assert expected == locs, (
            f"Missing: {expected - locs}\nExtra: {locs - expected}"
        )

    def test_unscannable_fields_present(self):
        fm = parse_gemini_request(self._build_payload())
        assert len(fm.unscannable_fields) == 2
        scan_fields = {f.scan_field for f in fm.unscannable_fields}
        assert ScanField.INLINE_DATA in scan_fields
        assert ScanField.FILE_DATA in scan_fields

    def test_unknown_fields_absent(self):
        fm = parse_gemini_request(self._build_payload())
        assert fm.unknown_fields == []

    def test_total_field_count(self):
        fm = parse_gemini_request(self._build_payload())
        assert len(fm.text_fields) == 10
        assert len(fm.unscannable_fields) == 2
        assert len(fm.all_fields) == 12

    def test_all_text_fields_have_correct_scan_field_types(self):
        fm = parse_gemini_request(self._build_payload())
        expected_types = {
            "systemInstruction.parts[0].text": ScanField.SYSTEM_INSTRUCTION,
            "contents[0].parts[0].text": ScanField.MESSAGE_TEXT,
            "contents[1].parts[0].text": ScanField.MESSAGE_TEXT,
            "contents[1].parts[1].functionCall.args.to": ScanField.FUNCTION_CALL_ARGS,
            "contents[1].parts[1].functionCall.args.subject": ScanField.FUNCTION_CALL_ARGS,
            "contents[1].parts[1].functionCall.args.body.greeting": ScanField.FUNCTION_CALL_ARGS,
            "contents[1].parts[2].executableCode.code": ScanField.EXECUTABLE_CODE,
            "contents[2].parts[0].functionResponse.response.status": ScanField.FUNCTION_RESPONSE,
            "contents[2].parts[0].functionResponse.response.to": ScanField.FUNCTION_RESPONSE,
            "contents[2].parts[1].codeExecutionResult.output": ScanField.CODE_EXECUTION_RESULT,
        }
        for pf in fm.text_fields:
            assert pf.scan_field == expected_types[pf.location], (
                f"Wrong scan_field for {pf.location!r}: "
                f"got {pf.scan_field}, want {expected_types[pf.location]}"
            )

    def test_text_values_match_payload(self):
        payload = self._build_payload()
        fm = parse_gemini_request(payload)
        values = {f.location: f.text for f in fm.text_fields}
        assert values["systemInstruction.parts[0].text"] == "System: admin@internal.corp"
        assert values["contents[0].parts[0].text"] == "User message: user@example.com"
        assert values["contents[1].parts[1].functionCall.args.to"] == "user@example.com"
        assert values["contents[1].parts[1].functionCall.args.body.greeting"] == "Dear user@example.com"
        assert values["contents[1].parts[2].executableCode.code"] == "print('done')"
        assert values["contents[2].parts[1].codeExecutionResult.output"] == "Email sent to user@example.com"

    def test_is_scannable_flags_correct(self):
        fm = parse_gemini_request(self._build_payload())
        for pf in fm.text_fields:
            assert pf.is_scannable is True
            assert pf.is_unscannable is False
            assert pf.is_unknown is False
            assert pf.text is not None

    def test_is_unscannable_flags_correct(self):
        fm = parse_gemini_request(self._build_payload())
        for pf in fm.unscannable_fields:
            assert pf.is_scannable is False
            assert pf.is_unscannable is True
            assert pf.is_unknown is False
            assert pf.text is None
            assert pf.coverage_gap_reason is not None
