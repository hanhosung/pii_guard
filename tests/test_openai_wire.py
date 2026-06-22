"""
Integration tests for OpenAI wire format traversal and payload reconstruction.
Sub-AC 2b — every OpenAI chat-completions field type scrubbed with no user config.

Tests assert:
  1.  system role message content (string) is correctly scrubbed.
  2.  user role message content (string) is correctly scrubbed.
  3.  user role message content (array of text parts) is correctly scrubbed.
  4.  assistant role message content (string/array) is correctly scrubbed.
  5.  tool_calls[*].function.arguments JSON is parsed and all string leaf
      values are scanned; sanitized JSON is serialized back correctly.
  6.  tool role message content (string/array) is correctly scrubbed.
  7.  developer role message content is correctly scrubbed.
  8.  refusal content parts inside assistant messages are scanned.
  9.  image_url content parts record a coverage gap and block by default.
  10. input_audio content parts record a coverage gap.
  11. file content parts record a coverage gap.
  12. Unknown content part types raise a coverage alarm (unknown_fields list).
  13. The sanitized payload is structurally valid (roles, ids, names preserved).
  14. Cross-field placeholder consistency: same real value → same placeholder.
  15. No original PII/secret text survives in the sanitized payload.
  16. No user configuration is required — plain Engine() provides full protection.
  17. Multi-message / multi-turn payloads are all scrubbed.
  18. Tool-call arguments that are not valid JSON fall back to raw-string scan.
  19. Numeric / boolean / null values in tool-call arguments pass through unchanged.
  20. The original payload dict is never mutated.
  21. Edge cases: empty payload, missing messages, null content, empty strings.
"""
from __future__ import annotations

import json
import re
import copy

import pytest

from pii_guard import Engine
from pii_guard.providers.openai import (
    FieldScanEvent,
    OpenAIRequestScrubResult,
    ScanField,
    scrub_openai_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fresh_engine() -> Engine:
    """Return a new Engine with no pre-existing session state."""
    return Engine()


def _collect_text_values(payload: dict) -> list[str]:
    """
    Recursively collect every user-visible string value in the payload that
    could carry PII — content strings and text/refusal part texts, plus tool
    call argument strings.  Excludes structural keys (role, type, id, name,
    model, …).
    """
    TEXT_KEYS = {"content", "text", "refusal", "arguments"}
    found: list[str] = []

    def _walk(obj: object) -> None:
        if isinstance(obj, str):
            found.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in TEXT_KEYS:
                    _walk(v)
                elif isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
    return found


def _email_placeholders(text: str) -> list[str]:
    return re.findall(r"\[EMAIL_\d+\]", text)


def _phone_placeholders(text: str) -> list[str]:
    return re.findall(r"\[PHONE_\d+\]", text)


# ─────────────────────────────────────────────────────────────────────────────
# 1. System role message — string content
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemMessage:
    """system role messages — content is a plain string."""

    def test_email_in_system_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Notify admin@corp.io of all alerts."},
                {"role": "user", "content": "Go"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        sys_content = result.sanitized_payload["messages"][0]["content"]
        assert "admin@corp.io" not in sys_content
        assert "[EMAIL_" in sys_content
        assert not result.should_block

    def test_api_key_in_system_blocks(self):
        key = "sk-proj-" + "A" * 48
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Use key={key} for auth."},
                {"role": "user", "content": "Proceed"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        sys_content = result.sanitized_payload["messages"][0]["content"]
        assert key not in sys_content
        assert result.should_block

    def test_clean_system_message_passes_through(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0]["content"] == "You are a helpful assistant."
        assert not result.should_block

    def test_system_message_scan_field_tagged(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Contact support@example.com"},
                {"role": "user", "content": "Ok"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        sys_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.SYSTEM_MESSAGE]
        assert sys_evts, "Expected at least one SYSTEM_MESSAGE scan event"

    def test_rrn_in_system_blocks(self):
        """Korean RRN in system message → blocked (high-risk PII)."""
        rrn = "900505-1234564"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Target RRN: {rrn}"},
                {"role": "user", "content": "Continue"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert rrn not in result.sanitized_payload["messages"][0]["content"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 2. User role message — string content
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMessageString:
    """user role messages — content is a plain string."""

    def test_email_in_user_message_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Send invoice to alice@example.com please."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "alice@example.com" not in content
        assert "[EMAIL_" in content
        assert not result.should_block

    def test_phone_in_user_message_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "My phone is 010-1234-5678"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "010-1234-5678" not in content
        assert "[PHONE_" in content
        assert not result.should_block

    def test_aws_key_in_user_message_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Here is my key: {key}"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["messages"][0]["content"]
        assert result.should_block

    def test_message_text_scan_field_for_user(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello user@example.com"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        user_evts = [e for e in result.field_events
                     if e.scan_field == ScanField.MESSAGE_TEXT]
        assert user_evts


# ─────────────────────────────────────────────────────────────────────────────
# 3. User role message — array of content parts
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMessageArray:
    """user role messages — content is an array of content parts."""

    def test_text_part_email_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Contact bob@domain.com for details."},
                    ],
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        text = result.sanitized_payload["messages"][0]["content"][0]["text"]
        assert "bob@domain.com" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_multiple_text_parts_all_scrubbed(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Email: user@test.com"},
                        {"type": "text", "text": "Phone: 010-5555-6666"},
                    ],
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        parts = result.sanitized_payload["messages"][0]["content"]
        assert "user@test.com" not in parts[0]["text"]
        assert "010-5555-6666" not in parts[1]["text"]
        assert not result.should_block

    def test_image_url_part_coverage_gap(self):
        """image_url parts are unscannable → coverage gap and block by default."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.coverage_gaps, "image_url part should record a coverage gap"
        assert result.should_block, "default unscannable_action=block should block"

    def test_image_url_warn_allow_does_not_block(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                },
            ],
        }
        result = scrub_openai_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps  # gap still recorded
        assert not result.should_block

    def test_image_url_scan_field_tagged(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x.com/img.jpg"}},
                    ],
                },
            ],
        }
        result = scrub_openai_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        img_evts = [e for e in result.field_events if e.scan_field == ScanField.IMAGE_URL]
        assert img_evts

    def test_mixed_text_and_image_text_scanned_image_gaped(self):
        """Text parts are scanned; image_url parts get a coverage gap."""
        email = "mixed@example.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"See {email} for context."},
                        {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                    ],
                },
            ],
        }
        result = scrub_openai_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        text_content = result.sanitized_payload["messages"][0]["content"][0]["text"]
        assert email not in text_content
        assert "[EMAIL_" in text_content
        assert result.coverage_gaps  # image gap present


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assistant role message — string and array content
# ─────────────────────────────────────────────────────────────────────────────

class TestAssistantMessageContent:
    """assistant role — content (no tool_calls)."""

    def test_email_in_assistant_content_string_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Who should I contact?"},
                {"role": "assistant", "content": "Reach out to contact@org.com."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        asst_content = result.sanitized_payload["messages"][1]["content"]
        assert "contact@org.com" not in asst_content
        assert "[EMAIL_" in asst_content
        assert not result.should_block

    def test_api_key_in_assistant_content_blocks(self):
        key = "sk-" + "B" * 48
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the API key?"},
                {"role": "assistant", "content": f"The key is {key}."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        asst_content = result.sanitized_payload["messages"][1]["content"]
        assert key not in asst_content
        assert result.should_block

    def test_assistant_text_part_array_scanned(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Your email was user@example.com."},
                    ],
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        text = result.sanitized_payload["messages"][1]["content"][0]["text"]
        assert "user@example.com" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_refusal_part_scanned(self):
        """refusal parts inside assistant messages have their text scanned."""
        email = "victim@example.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Request"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": f"Cannot send to {email}."},
                    ],
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        refusal_text = result.sanitized_payload["messages"][1]["content"][0]["refusal"]
        assert email not in refusal_text
        assert "[EMAIL_" in refusal_text
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_calls — function.arguments JSON parsed and scanned
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCallArguments:
    """tool_calls[*].function.arguments JSON string is parsed and recursively scanned."""

    def test_email_in_arguments_string_value_masked(self):
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
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args_str = (
            result.sanitized_payload["messages"][0]
            ["tool_calls"][0]["function"]["arguments"]
        )
        san_args = json.loads(san_args_str)
        assert "recipient@domain.com" not in san_args["to"]
        assert "[EMAIL_" in san_args["to"]
        assert san_args["subject"] == "Hello"  # non-PII unchanged
        assert not result.should_block

    def test_api_key_in_arguments_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        args = json.dumps({"access_key": key, "region": "us-east-1"})
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
                            "function": {"name": "call_aws", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args_str = (
            result.sanitized_payload["messages"][0]
            ["tool_calls"][0]["function"]["arguments"]
        )
        san_args = json.loads(san_args_str)
        assert key not in san_args["access_key"]
        assert result.should_block

    def test_nested_object_in_arguments_recursively_scanned(self):
        args = json.dumps({
            "contact": {
                "email": "nested@corp.com",
                "phone": "010-7777-8888",
                "notes": "Important client",
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
                            "id": "call_003",
                            "type": "function",
                            "function": {"name": "store_contact", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        contact = san_args["contact"]
        assert "nested@corp.com" not in contact["email"]
        assert "010-7777-8888" not in contact["phone"]
        assert contact["notes"] == "Important client"  # non-PII unchanged
        assert not result.should_block

    def test_array_values_in_arguments_recursively_scanned(self):
        args = json.dumps({"emails": ["a@first.com", "b@second.io", "safe_value"]})
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
                            "function": {"name": "notify_all", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        emails = san_args["emails"]
        assert "a@first.com" not in emails
        assert "b@second.io" not in emails
        assert "safe_value" in emails
        assert not result.should_block

    def test_numeric_and_boolean_values_unchanged(self):
        """Numbers, booleans, and null in arguments pass through untouched."""
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
                            "id": "call_005",
                            "type": "function",
                            "function": {"name": "set_params", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        assert san_args["count"] == 42
        assert san_args["enabled"] is True
        assert san_args["ratio"] == 3.14
        assert san_args["nothing"] is None
        assert not result.should_block

    def test_deeply_nested_arguments_scanned(self):
        key = "ghp_" + "G" * 40
        args = json.dumps({
            "config": {"creds": {"level3": {"token": key}}}
        })
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
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        token = san_args["config"]["creds"]["level3"]["token"]
        assert key not in token
        assert result.should_block

    def test_tool_call_args_scan_field_tagged(self):
        args = json.dumps({"message": "Hello contact@domain.com"})
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
                            "function": {"name": "greet", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        args_evts = [e for e in result.field_events
                     if e.scan_field == ScanField.TOOL_CALL_ARGS]
        assert args_evts

    def test_invalid_json_arguments_scanned_as_raw_text(self):
        """
        If arguments cannot be parsed as JSON, fall back to raw-text scanning.
        A coverage gap is recorded because the structure couldn't be validated.
        """
        raw = 'email=victim@corp.com&key=AKIAIOSFODNN7EXAMPLE'
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_008",
                            "type": "function",
                            "function": {"name": "bad_args", "arguments": raw},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args_str = (
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        # victim@corp.com should be masked (raw scan caught it)
        assert "victim@corp.com" not in san_args_str
        # Coverage gap recorded because JSON structure could not be validated
        assert result.coverage_gaps

    def test_multiple_tool_calls_all_scanned(self):
        """Multiple tool calls in one assistant message are all scanned."""
        args1 = json.dumps({"email": "first@tool.com"})
        key = "AKIAIOSFODNN7EXAMPLE"
        args2 = json.dumps({"access_key": key})
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
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        calls = result.sanitized_payload["messages"][0]["tool_calls"]
        san1 = json.loads(calls[0]["function"]["arguments"])
        san2 = json.loads(calls[1]["function"]["arguments"])
        assert "first@tool.com" not in san1["email"]
        assert key not in san2["access_key"]
        assert result.should_block  # AWS key → block

    def test_tool_call_id_name_preserved(self):
        """tool_call id, type, and function.name are not modified."""
        args = json.dumps({"q": "hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_preserve_id",
                            "type": "function",
                            "function": {"name": "my_function", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        tc = result.sanitized_payload["messages"][0]["tool_calls"][0]
        assert tc["id"] == "call_preserve_id"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_function"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tool role message — content scrubbed (corresponds to tool_result)
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRoleMessage:
    """tool role messages — content (string or array) maps to TOOL_RESULT."""

    def test_tool_message_string_content_masked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_res_001",
                    "content": "The user email is result@example.com. Done.",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "result@example.com" not in content
        assert "[EMAIL_" in content
        assert not result.should_block

    def test_tool_message_string_content_blocks_on_secret(self):
        key = "sk-ant-api03-" + "T" * 50
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_res_002",
                    "content": f"Retrieved key: {key}",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["messages"][0]["content"]
        assert result.should_block

    def test_tool_message_array_content_scrubbed(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_res_003",
                    "content": [
                        {"type": "text", "text": "Phone: 010-4444-5555"},
                        {"type": "text", "text": "Status: OK"},
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        parts = result.sanitized_payload["messages"][0]["content"]
        assert "010-4444-5555" not in parts[0]["text"]
        assert "[PHONE_" in parts[0]["text"]
        assert parts[1]["text"] == "Status: OK"
        assert not result.should_block

    def test_tool_message_scan_field_tagged_as_tool_result(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_field",
                    "content": "Contact: user@mail.com",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        tr_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.TOOL_RESULT]
        assert tr_evts

    def test_tool_call_id_preserved(self):
        """tool_call_id is structural and must not be altered."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "preserve_this_id",
                    "content": "Some output",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0]["tool_call_id"] == "preserve_this_id"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Developer role message
# ─────────────────────────────────────────────────────────────────────────────

class TestDeveloperMessage:
    """developer role messages (used by some OpenAI models) are scanned."""

    def test_developer_message_email_masked(self):
        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "developer",
                    "content": "Internal contact: dev@internal.io",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "dev@internal.io" not in content
        assert "[EMAIL_" in content
        assert not result.should_block

    def test_developer_message_tagged_as_message_text(self):
        payload = {
            "model": "o3",
            "messages": [
                {"role": "developer", "content": "Config: api@dev.com"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        dev_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.MESSAGE_TEXT]
        assert dev_evts


# ─────────────────────────────────────────────────────────────────────────────
# 8. Unscannable and unknown content parts
# ─────────────────────────────────────────────────────────────────────────────

class TestUnscannableAndUnknownParts:
    """Unscannable content parts record coverage gaps; unknown types alarm."""

    def test_input_audio_coverage_gap(self):
        payload = {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "base64audiodata", "format": "wav"},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.coverage_gaps, "input_audio should record a coverage gap"
        assert result.should_block  # default unscannable_action=block

    def test_input_audio_warn_allow_does_not_block(self):
        payload = {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "base64audio", "format": "mp3"},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps
        assert not result.should_block

    def test_file_part_coverage_gap(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_id": "file-abc123"}},
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.coverage_gaps
        assert result.should_block

    def test_unknown_content_part_type_alarm_and_block(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_part_type", "data": "some_content"},
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.unknown_fields, "Expected unknown field alarm"
        assert any("future_part_type" in u for u in result.unknown_fields)
        assert result.should_block

    def test_unknown_content_part_warn_allow_does_not_block(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "new_future_type", "data": "something"},
                    ],
                }
            ],
        }
        result = scrub_openai_request(
            payload, fresh_engine(), unknown_field_action="warn_allow"
        )
        assert result.unknown_fields
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 9. Structural validity of sanitized payload
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizedPayloadStructure:
    """The sanitized payload must be structurally valid and preserve metadata."""

    def test_top_level_keys_preserved(self):
        payload = {
            "model": "gpt-4o",
            "max_tokens": 512,
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": "Hi"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        sp = result.sanitized_payload
        assert sp["model"] == "gpt-4o"
        assert sp["max_tokens"] == 512
        assert sp["temperature"] == 0.7

    def test_message_roles_preserved(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "tool", "tool_call_id": "tc1", "content": "Done."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "tool"

    def test_content_part_type_fields_preserved(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello alice@example.com"},
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        part = result.sanitized_payload["messages"][0]["content"][0]
        assert part["type"] == "text"

    def test_tool_call_structure_preserved(self):
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
                            "function": {"name": "my_fn", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        tc = result.sanitized_payload["messages"][0]["tool_calls"][0]
        assert tc["id"] == "struct_id"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_fn"
        # arguments is still a string
        assert isinstance(tc["function"]["arguments"], str)

    def test_arguments_remains_valid_json_after_scrub(self):
        """After scrubbing, function.arguments must still be valid JSON."""
        args = json.dumps({
            "email": "test@corp.com",
            "count": 5,
            "tags": ["important", "urgent"],
        })
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_valid",
                            "type": "function",
                            "function": {"name": "task", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args_str = (
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        # Must be valid JSON
        parsed = json.loads(san_args_str)
        assert parsed["count"] == 5
        assert parsed["tags"] == ["important", "urgent"]
        assert "test@corp.com" not in parsed["email"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cross-field placeholder consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFieldConsistency:
    """Same real value → same placeholder across all fields in one request."""

    def test_same_email_same_placeholder_across_roles(self):
        """An email in system + user message gets the same placeholder."""
        email = "shared@example.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Contact {email} for support."},
                {"role": "user", "content": f"Please email {email} now."},
            ],
        }
        engine = fresh_engine()
        result = scrub_openai_request(payload, engine)
        sys_text = result.sanitized_payload["messages"][0]["content"]
        usr_text = result.sanitized_payload["messages"][1]["content"]
        sys_ph = _email_placeholders(sys_text)
        usr_ph = _email_placeholders(usr_text)
        assert sys_ph, "Expected placeholder in system message"
        assert usr_ph, "Expected placeholder in user message"
        assert sys_ph[0] == usr_ph[0], "Same email must produce the same placeholder"

    def test_same_email_same_placeholder_in_message_and_tool_args(self):
        """Email appearing in message text and tool_call arguments → same placeholder."""
        email = "shared_tc@corp.com"
        args = json.dumps({"recipient": email})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Send to {email}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_x",
                            "type": "function",
                            "function": {"name": "notify", "arguments": args},
                        }
                    ],
                },
            ],
        }
        engine = fresh_engine()
        result = scrub_openai_request(payload, engine)
        usr_text = result.sanitized_payload["messages"][0]["content"]
        san_args = json.loads(
            result.sanitized_payload["messages"][1]["tool_calls"][0]["function"]["arguments"]
        )
        usr_ph = _email_placeholders(usr_text)
        args_ph = _email_placeholders(san_args["recipient"])
        assert usr_ph and args_ph
        assert usr_ph[0] == args_ph[0]

    def test_same_email_same_placeholder_in_tool_result(self):
        """Email appearing in tool role content keeps the same placeholder."""
        email = "tool_result@corp.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Info about {email}"},
                {
                    "role": "tool",
                    "tool_call_id": "tc_y",
                    "content": f"User {email} confirmed.",
                },
            ],
        }
        engine = fresh_engine()
        result = scrub_openai_request(payload, engine)
        usr_text = result.sanitized_payload["messages"][0]["content"]
        tool_text = result.sanitized_payload["messages"][1]["content"]
        usr_ph = _email_placeholders(usr_text)
        tool_ph = _email_placeholders(tool_text)
        assert usr_ph and tool_ph
        assert usr_ph[0] == tool_ph[0]

    def test_different_emails_different_placeholders(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "From a@first.com"},
                {"role": "user", "content": "To b@second.io"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        ph_a = _email_placeholders(result.sanitized_payload["messages"][0]["content"])
        ph_b = _email_placeholders(result.sanitized_payload["messages"][1]["content"])
        assert ph_a and ph_b
        assert ph_a[0] != ph_b[0], "Different emails must get different placeholders"


# ─────────────────────────────────────────────────────────────────────────────
# 11. No raw PII/secret text survives in the sanitized payload
# ─────────────────────────────────────────────────────────────────────────────

class TestNoPIISurvivesInPayload:
    """Integration assertion: no raw sensitive value in any text field."""

    def test_full_mixed_payload_all_sanitized(self):
        """Full mixed payload with PII in every field type is fully sanitized."""
        email = "victim@corp.io"
        phone = "010-8888-9999"
        api_key = "sk-" + "X" * 48

        args = json.dumps({"contact_email": email, "backup": phone})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Notify {email} of all issues."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"User phone: {phone}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_full",
                            "type": "function",
                            "function": {"name": "save", "arguments": args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc_full",
                    "content": f"Saved {email}. Key: {api_key}",
                },
            ],
        }

        result = scrub_openai_request(payload, fresh_engine())
        sp = result.sanitized_payload

        # system message
        assert email not in sp["messages"][0]["content"]

        # user text part
        assert phone not in sp["messages"][1]["content"][0]["text"]

        # tool_call arguments
        san_args = json.loads(sp["messages"][2]["tool_calls"][0]["function"]["arguments"])
        assert email not in san_args["contact_email"]
        assert phone not in san_args["backup"]

        # tool role content
        assert email not in sp["messages"][3]["content"]
        assert api_key not in sp["messages"][3]["content"]

        # api_key is a secret → should_block
        assert result.should_block

    def test_no_config_required(self):
        """
        Verify that a plain Engine() with no arguments provides protection
        for all category classes (secret, pii, korean_pii) out of the box.
        """
        rrn = "900505-1234564"           # valid Korean RRN checksum
        card = "4532015112830366"         # Luhn-valid Visa test number
        email = "test@example.com"
        aws_key = "AKIAIOSFODNN7EXAMPLE"

        args = json.dumps({"data": f"Card: {card}, Email: {email}"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"RRN: {rrn}"},
                {"role": "user", "content": f"Key: {aws_key}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_noconf",
                            "type": "function",
                            "function": {"name": "store", "arguments": args},
                        }
                    ],
                },
            ],
        }

        # No policy file, no custom config — plain Engine()
        result = scrub_openai_request(payload, Engine())
        sp = result.sanitized_payload

        assert rrn not in sp["messages"][0]["content"]
        assert aws_key not in sp["messages"][1]["content"]
        san_args = json.loads(sp["messages"][2]["tool_calls"][0]["function"]["arguments"])
        assert card not in san_args["data"]
        assert email not in san_args["data"]
        # secrets → should_block
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 12. Multi-message / multi-turn payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnPayload:
    """Multi-turn conversations are fully scrubbed."""

    def test_pii_across_multiple_turns(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "My email is first@a.com"},
                {"role": "assistant", "content": "Got it, first@a.com noted."},
                {"role": "user", "content": "Also reach second@b.io"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        assert "first@a.com" not in msgs[0]["content"]
        assert "first@a.com" not in msgs[1]["content"]
        assert "second@b.io" not in msgs[2]["content"]
        assert not result.should_block

    def test_same_value_consistent_across_turns(self):
        """Single engine session → same email always gets same placeholder."""
        email = "consistent@x.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Email: {email}"},
                {"role": "assistant", "content": f"Sure, {email} it is."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        ph_u = _email_placeholders(msgs[0]["content"])
        ph_a = _email_placeholders(msgs[1]["content"])
        assert ph_u and ph_a
        assert ph_u[0] == ph_a[0]

    def test_interleaved_roles_all_scrubbed(self):
        """A realistic 4-turn conversation with tool use."""
        key = "AKIAIOSFODNN7EXAMPLE"
        email = "user@example.com"
        args = json.dumps({"to": email, "subject": "Report"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"Send to {email}"},
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
                    "content": f"Sent. AWS key used: {key}",
                },
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        sp = result.sanitized_payload
        san_args = json.loads(sp["messages"][2]["tool_calls"][0]["function"]["arguments"])
        assert email not in san_args["to"]
        assert key not in sp["messages"][3]["content"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 13. Original payload is not mutated
# ─────────────────────────────────────────────────────────────────────────────

class TestOriginalPayloadNotMutated:
    def test_scrub_does_not_mutate_original(self):
        email = "original@check.com"
        args = json.dumps({"contact": email})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Email: {email}"},
                {"role": "user", "content": email},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_mut",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_mut", "content": email},
            ],
        }
        original_copy = copy.deepcopy(payload)
        scrub_openai_request(payload, fresh_engine())
        assert payload == original_copy, "scrub_openai_request must not mutate the input"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_payload_does_not_raise(self):
        result = scrub_openai_request({}, fresh_engine())
        assert result.sanitized_payload == {}
        assert not result.should_block
        assert not result.coverage_gaps

    def test_no_messages_key(self):
        result = scrub_openai_request({"model": "gpt-4o"}, fresh_engine())
        assert result.sanitized_payload["model"] == "gpt-4o"
        assert not result.should_block

    def test_empty_messages_list(self):
        result = scrub_openai_request({"model": "gpt-4o", "messages": []}, fresh_engine())
        assert result.sanitized_payload["messages"] == []
        assert not result.should_block

    def test_null_content_not_scanned(self):
        """Content=None (typical for tool_calls assistant) must not error."""
        args = json.dumps({"q": "hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_null",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0]["content"] is None
        assert not result.should_block

    def test_empty_string_content_not_modified(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": ""}],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0]["content"] == ""
        assert not result.should_block

    def test_empty_tool_call_arguments_object(self):
        args = json.dumps({})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_empty",
                            "type": "function",
                            "function": {"name": "fn", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        ) == {}
        assert not result.should_block

    def test_deeply_nested_pii_in_arguments_scanned(self):
        """3-level nesting inside tool arguments is fully walked."""
        email = "deep@nested.com"
        args = json.dumps({
            "a": {"b": {"c": email}}
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
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        assert email not in san_args["a"]["b"]["c"]
        assert "[EMAIL_" in san_args["a"]["b"]["c"]
        assert not result.should_block

    def test_non_dict_message_in_list_skipped_gracefully(self):
        """Non-dict entries in messages list must not raise."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                "not a dict",
                {"role": "user", "content": "Hi user@test.com"},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0] == "not a dict"
        user_content = result.sanitized_payload["messages"][1]["content"]
        assert "user@test.com" not in user_content
        assert not result.should_block

    def test_jwt_token_in_arguments_blocks(self):
        """JWT-style token in tool arguments is detected and blocked."""
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        args = json.dumps({"token": jwt})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_jwt",
                            "type": "function",
                            "function": {"name": "auth", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        san_args = json.loads(
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        assert jwt not in san_args["token"]
        assert result.should_block

    def test_private_key_pem_in_user_message_blocks(self):
        """PEM private key header in message content is blocked."""
        pem = "-----BEGIN RSA PRIVATE KEY-----"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Here is my key:\n{pem}\nMIIEowIBAAK..."},
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert pem not in result.sanitized_payload["messages"][0]["content"]
        assert result.should_block

    def test_card_number_in_tool_result_blocks(self):
        """Luhn-valid card number in tool role content is blocked."""
        card = "4532015112830366"  # Luhn-valid Visa
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "tc_card",
                    "content": f"Payment card: {card}",
                }
            ],
        }
        result = scrub_openai_request(payload, fresh_engine())
        assert card not in result.sanitized_payload["messages"][0]["content"]
        assert result.should_block
