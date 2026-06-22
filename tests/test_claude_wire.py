"""
Integration tests for Claude wire format traversal and payload reconstruction.
Sub-AC 2a — every Claude-specific field type scrubbed with no user config.

Tests assert:
  1. Every text-bearing field location in the Anthropic Messages API schema is
     correctly detected and scrubbed.
  2. No user configuration is required — a plain Engine() is sufficient.
  3. BLOCK-category content (secrets, high-risk IDs) sets should_block=True.
  4. MASK-category content (email, phone) is replaced with [CAT_N] placeholders
     while should_block remains False.
  5. tool_use.input is recursively scanned (nested dicts, arrays, strings).
  6. tool_result.content handles both string and TextBlock array formats.
  7. document blocks with text source are scanned; base64/url sources record a
     coverage gap.
  8. Image blocks record a coverage gap (unscannable) and block by default.
  9. Unknown content block types raise a coverage alarm (unknown_fields list).
 10. The sanitized payload is a structurally valid Claude request.
 11. Cross-field placeholder consistency: the same real value → same placeholder.
 12. No original PII/secret text survives in the sanitized payload.
"""
from __future__ import annotations

import pytest

from pii_guard import Engine
from pii_guard.providers.claude import (
    ClaudeRequestScrubResult,
    FieldScanEvent,
    ScanField,
    scrub_claude_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fresh_engine() -> Engine:
    """Return a new Engine with no pre-existing session state."""
    return Engine()


def _all_text_in_payload(payload: dict) -> list[str]:
    """
    Walk a sanitized payload and collect every string value that is a content
    field (not a key, not an id/name/type/role).  Used to verify no raw PII
    survives.
    """
    TEXT_KEYS = {"text", "data", "content"}
    found = []

    def _walk(obj):
        if isinstance(obj, str):
            found.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in TEXT_KEYS or k == "input":
                    _walk(v)
                elif isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# 1. System prompt — string form
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPromptString:
    """system is a plain string."""

    def test_email_in_system_string_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": "Always email alice@secret.io for approvals.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert "alice@secret.io" not in result.sanitized_payload["system"]
        assert "[EMAIL_" in result.sanitized_payload["system"]
        assert not result.should_block

    def test_api_key_in_system_string_blocks(self):
        key = "sk-ant-api03-" + "A" * 50
        payload = {
            "model": "claude-opus-4-5",
            "system": f"Use key={key} for auth.",
            "messages": [{"role": "user", "content": "Go"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["system"]
        assert result.should_block

    def test_clean_system_string_passes(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.sanitized_payload["system"] == "You are a helpful assistant."
        assert not result.should_block
        assert result.field_events  # at least one scan event recorded

    def test_system_scan_field_tagged(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": "Contact bob@corp.io",
            "messages": [{"role": "user", "content": "Ok"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        sys_events = [e for e in result.field_events
                      if e.scan_field == ScanField.SYSTEM_PROMPT]
        assert sys_events, "Expected at least one SYSTEM_PROMPT scan event"


# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt — block array form
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPromptBlockArray:
    """system is a list of TextBlock objects."""

    def test_text_block_email_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": [
                {"type": "text", "text": "Contact admin@example.com for help."},
                {"type": "text", "text": "Be polite."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        first_block_text = result.sanitized_payload["system"][0]["text"]
        assert "admin@example.com" not in first_block_text
        assert "[EMAIL_" in first_block_text
        # Second block unchanged
        assert result.sanitized_payload["system"][1]["text"] == "Be polite."
        assert not result.should_block

    def test_multiple_text_blocks_all_scrubbed(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": [
                {"type": "text", "text": "User: 010-1234-5678"},
                {"type": "text", "text": "Email: user@corp.io"},
            ],
            "messages": [{"role": "user", "content": "Proceed"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        texts = [b["text"] for b in result.sanitized_payload["system"]]
        assert "010-1234-5678" not in texts[0]
        assert "user@corp.io" not in texts[1]
        assert not result.should_block

    def test_api_key_in_system_block_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "claude-opus-4-5",
            "system": [{"type": "text", "text": f"Use {key}"}],
            "messages": [{"role": "user", "content": "Run"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["system"][0]["text"]
        assert result.should_block

    def test_unknown_system_block_type_causes_alarm(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": [
                {"type": "text", "text": "Fine"},
                {"type": "future_block", "data": "something"},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert any("future_block" in u for u in result.unknown_fields)
        # Default unknown_field_action="block" → should block
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 3. Message content — string shorthand
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageContentString:
    """message.content is a plain string (API shorthand)."""

    def test_email_in_user_message_string_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "Send invoice to bob@example.com please."}
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "bob@example.com" not in content
        assert "[EMAIL_" in content
        assert not result.should_block

    def test_secret_in_assistant_message_blocks(self):
        key = "sk-" + "x" * 48
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "What's the key?"},
                {"role": "assistant", "content": f"The key is {key}."},
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        asst_content = result.sanitized_payload["messages"][1]["content"]
        assert key not in asst_content
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 4. Message content — text block
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageTextBlock:
    """message.content is a list with type=text blocks."""

    def test_email_in_text_block_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reach me at alice@corp.io anytime."}
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        text = result.sanitized_payload["messages"][0]["content"][0]["text"]
        assert "alice@corp.io" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_secret_in_text_block_blocks(self):
        key = "sk-ant-api03-" + "B" * 50
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Key is {key}"}],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        text = result.sanitized_payload["messages"][0]["content"][0]["text"]
        assert key not in text
        assert result.should_block

    def test_multiple_text_blocks_all_scrubbed(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email: user@example.com"},
                        {"type": "text", "text": "My phone: 010-9999-1234"},
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "user@example.com" not in content[0]["text"]
        assert "010-9999-1234" not in content[1]["text"]
        assert not result.should_block

    def test_scan_field_tagged_as_message_text(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello world"}]},
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        msg_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.MESSAGE_TEXT]
        assert msg_evts


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_use blocks — input recursively scanned
# ─────────────────────────────────────────────────────────────────────────────

class TestToolUseInput:
    """tool_use.input values are recursively scanned."""

    def test_email_in_tool_input_string_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_001",
                            "name": "send_email",
                            "input": {
                                "to": "recipient@domain.com",
                                "subject": "Hello",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        tool_input = result.sanitized_payload["messages"][0]["content"][0]["input"]
        assert "recipient@domain.com" not in tool_input["to"]
        assert "[EMAIL_" in tool_input["to"]
        assert not result.should_block

    def test_api_key_in_tool_input_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_002",
                            "name": "call_aws",
                            "input": {"access_key": key, "action": "list"},
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        tool_input = result.sanitized_payload["messages"][0]["content"][0]["input"]
        assert key not in tool_input["access_key"]
        assert result.should_block

    def test_nested_dict_in_tool_input_scanned(self):
        """Nested dicts inside tool_use.input are recursively scanned."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_003",
                            "name": "store_contact",
                            "input": {
                                "contact": {
                                    "email": "deep@nested.com",
                                    "phone": "010-1111-2222",
                                    "metadata": {"source": "CRM"},
                                }
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        contact = result.sanitized_payload["messages"][0]["content"][0]["input"]["contact"]
        assert "deep@nested.com" not in contact["email"]
        assert "010-1111-2222" not in contact["phone"]
        assert contact["metadata"]["source"] == "CRM"  # non-PII unchanged

    def test_array_of_strings_in_tool_input_scanned(self):
        """Array values inside tool_use.input are recursively scanned."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_004",
                            "name": "notify_users",
                            "input": {
                                "emails": ["a@b.com", "c@d.io", "safe_value"],
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        emails = result.sanitized_payload["messages"][0]["content"][0]["input"]["emails"]
        assert "a@b.com" not in emails
        assert "c@d.io" not in emails
        assert "safe_value" in emails  # non-PII unchanged

    def test_numeric_values_in_tool_input_unchanged(self):
        """Numbers and booleans in tool_use.input pass through untouched."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_005",
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
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        inp = result.sanitized_payload["messages"][0]["content"][0]["input"]
        assert inp["count"] == 42
        assert inp["enabled"] is True
        assert inp["ratio"] == 3.14
        assert inp["nothing"] is None

    def test_scan_field_tagged_as_tool_use_input(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_006",
                            "name": "greet",
                            "input": {"message": "Hello contact@domain.com"},
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        tu_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.TOOL_USE_INPUT]
        assert tu_evts


# ─────────────────────────────────────────────────────────────────────────────
# 6. tool_result blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResult:
    """tool_result.content handles string and TextBlock array."""

    def test_tool_result_string_content_masked(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_100",
                            "content": "User is alice@example.com. Done.",
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        content = result.sanitized_payload["messages"][0]["content"][0]["content"]
        assert "alice@example.com" not in content
        assert "[EMAIL_" in content
        assert not result.should_block

    def test_tool_result_block_array_content_masked(self):
        payload = {
            "model": "claude-opus-4-5",
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
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        rc = result.sanitized_payload["messages"][0]["content"][0]["content"]
        assert "010-5555-1234" not in rc[0]["text"]
        assert "[PHONE_" in rc[0]["text"]
        assert rc[1]["text"] == "Status: OK"
        assert not result.should_block

    def test_secret_in_tool_result_blocks_request(self):
        key = "sk-ant-api03-" + "C" * 50
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_102",
                            "content": f"API key retrieved: {key}",
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["messages"][0]["content"][0]["content"]
        assert result.should_block

    def test_tool_result_scan_field_tagged(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_103",
                            "content": "Email: user@mail.com",
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        tr_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.TOOL_RESULT]
        assert tr_evts

    def test_tool_result_image_block_coverage_gap(self):
        """image block inside tool_result records a coverage gap."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_104",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBORw0KGgoAAAA==",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.coverage_gaps, "Expected coverage gap for image in tool_result"
        # Default unscannable_action="block"
        assert result.should_block

    def test_tool_result_no_content_field_ignored(self):
        """tool_result without content field should not error."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_105",
                            # no "content" key
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert not result.should_block  # no PII found, no coverage gap


# ─────────────────────────────────────────────────────────────────────────────
# 7. Document blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentBlock:
    """document blocks with text/base64/url sources."""

    def test_text_source_document_scanned(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Contact 이영희 at manager@company.io for help.",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        data = result.sanitized_payload["messages"][0]["content"][0]["source"]["data"]
        assert "manager@company.io" not in data
        assert "[EMAIL_" in data
        assert not result.should_block

    def test_text_source_secret_blocks(self):
        key = "hf_" + "Z" * 40
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": f"My HuggingFace token: {key}",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        data = result.sanitized_payload["messages"][0]["content"][0]["source"]["data"]
        assert key not in data
        assert result.should_block

    def test_base64_source_document_coverage_gap(self):
        """base64 document source is unscannable → coverage gap."""
        payload = {
            "model": "claude-opus-4-5",
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
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.coverage_gaps, "Expected coverage gap for base64 document"
        assert result.should_block  # fail-closed default

    def test_url_source_document_coverage_gap(self):
        """URL document source is unscannable → coverage gap."""
        payload = {
            "model": "claude-opus-4-5",
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
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.coverage_gaps
        assert result.should_block

    def test_document_scan_field_tagged(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Plain document text with no PII.",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        doc_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.DOCUMENT_BLOCK]
        assert doc_evts


# ─────────────────────────────────────────────────────────────────────────────
# 8. Image blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestImageBlock:
    """image blocks are not text-scannable → coverage gap + block by default."""

    def test_image_block_coverage_gap_default(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgAB...",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.coverage_gaps, "image block should record a coverage gap"
        assert result.should_block, "default unscannable_action=block should block"

    def test_image_block_warn_allow_mode(self):
        """With unscannable_action='warn_allow' image blocks don't block."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps  # gap still recorded
        assert not result.should_block  # but we don't block

    def test_image_scan_field_tagged(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png",
                                       "data": "abc123"},
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        img_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.IMAGE]
        assert img_evts


# ─────────────────────────────────────────────────────────────────────────────
# 9. Unknown block types → coverage alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownBlockTypes:
    def test_unknown_content_block_type_alarm(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_content_type", "data": "something"},
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.unknown_fields, "Expected unknown field alarm"
        assert any("future_content_type" in u for u in result.unknown_fields)
        assert result.should_block  # default=block

    def test_unknown_block_warn_allow_mode(self):
        """With unknown_field_action='warn_allow' unknown blocks don't block."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "future_block", "data": "something"},
                    ],
                }
            ],
        }
        result = scrub_claude_request(
            payload, fresh_engine(), unknown_field_action="warn_allow"
        )
        assert result.unknown_fields
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 10. Structural validity of sanitized payload
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizedPayloadStructure:
    """The sanitized payload must be structurally valid."""

    def test_all_required_keys_preserved(self):
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "system": "Be helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        sp = result.sanitized_payload
        assert sp["model"] == "claude-opus-4-5"
        assert sp["max_tokens"] == 1024
        assert "system" in sp
        assert "messages" in sp

    def test_message_roles_preserved(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_block_type_fields_preserved(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Contact alice@a.com"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "xyz",
                            "content": "Result: bob@b.com",
                        },
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        blocks = result.sanitized_payload["messages"][0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_result"
        assert blocks[1]["tool_use_id"] == "xyz"

    def test_tool_use_id_and_name_preserved(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_preserve_me",
                            "name": "my_tool",
                            "input": {"q": "hello user@test.com"},
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        tu = result.sanitized_payload["messages"][0]["content"][0]
        assert tu["id"] == "tu_preserve_me"
        assert tu["name"] == "my_tool"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cross-field placeholder consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFieldConsistency:
    """Same real value → same placeholder across all fields in one request."""

    def test_same_email_same_placeholder_across_fields(self):
        """An email appearing in system + message text gets the same placeholder."""
        email = "shared@example.com"
        payload = {
            "model": "claude-opus-4-5",
            "system": f"Contact {email} for support.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please email {email} now."}
                    ],
                }
            ],
        }
        engine = fresh_engine()
        result = scrub_claude_request(payload, engine)
        system_text = result.sanitized_payload["system"]
        msg_text = result.sanitized_payload["messages"][0]["content"][0]["text"]
        # Extract placeholder from system
        import re
        sys_placeholders = re.findall(r"\[EMAIL_\d+\]", system_text)
        msg_placeholders = re.findall(r"\[EMAIL_\d+\]", msg_text)
        assert sys_placeholders, "Expected placeholder in system"
        assert msg_placeholders, "Expected placeholder in message"
        assert sys_placeholders[0] == msg_placeholders[0], (
            "Same email must produce the same placeholder across fields"
        )

    def test_different_emails_different_placeholders(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "From: a@first.com"},
                                {"type": "text", "text": "To: b@second.com"},
                            ],
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        import re
        rc = result.sanitized_payload["messages"][0]["content"][0]["content"]
        ph_a = re.findall(r"\[EMAIL_\d+\]", rc[0]["text"])
        ph_b = re.findall(r"\[EMAIL_\d+\]", rc[1]["text"])
        assert ph_a and ph_b
        assert ph_a[0] != ph_b[0], "Different emails must get different placeholders"


# ─────────────────────────────────────────────────────────────────────────────
# 12. No raw PII/secret survives in sanitized payload
# ─────────────────────────────────────────────────────────────────────────────

class TestNoPIISurvivesInPayload:
    """Integration assertion: no raw sensitive value in any text field."""

    def test_mixed_payload_all_sanitized(self):
        """Full mixed payload with PII in every field type is fully sanitized."""
        email = "victim@corp.io"
        phone = "010-8888-9999"
        api_key = "sk-ant-api03-" + "X" * 50

        payload = {
            "model": "claude-opus-4-5",
            "system": f"Notify {email} of all issues.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"User phone: {phone}",
                        },
                        {
                            "type": "tool_use",
                            "id": "tu_mixed",
                            "name": "save",
                            "input": {
                                "contact_email": email,
                                "api_key": api_key,
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_prev",
                            "content": [
                                {"type": "text", "text": f"Saved {email} OK"},
                            ],
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": f"Phone on file: {phone}",
                            },
                        },
                    ],
                }
            ],
        }

        result = scrub_claude_request(payload, fresh_engine())
        sp = result.sanitized_payload

        # System prompt
        assert email not in sp["system"]

        # Message text block
        assert phone not in sp["messages"][0]["content"][0]["text"]

        # tool_use input
        inp = sp["messages"][0]["content"][1]["input"]
        assert email not in inp["contact_email"]
        assert api_key not in inp["api_key"]

        # tool_result
        rc = sp["messages"][0]["content"][2]["content"]
        assert email not in rc[0]["text"]

        # document
        data = sp["messages"][0]["content"][3]["source"]["data"]
        assert phone not in data

        # api_key is secret → should_block
        assert result.should_block

    def test_no_config_required(self):
        """
        Verify that a plain Engine() with no arguments provides protection
        for all category classes (secret, pii, korean_pii) out of the box.
        """
        rrn = "900505-1234564"          # valid Korean RRN checksum
        card = "4532015112830366"        # Luhn-valid Visa test number
        email = "test@example.com"
        aws_key = "AKIAIOSFODNN7EXAMPLE"

        payload = {
            "model": "claude-opus-4-5",
            "system": f"RRN: {rrn}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Card: {card}"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": f"Email: {email}. Key: {aws_key}",
                        },
                    ],
                }
            ],
        }

        # No policy file, no custom config — plain Engine()
        result = scrub_claude_request(payload, Engine())

        sp = result.sanitized_payload
        # Every sensitive value replaced
        assert rrn not in sp["system"]
        assert card not in sp["messages"][0]["content"][0]["text"]
        content_str = sp["messages"][0]["content"][1]["content"]
        assert email not in content_str
        assert aws_key not in content_str
        # Secrets → should_block
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 13. Multi-turn / multi-message payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnPayload:
    def test_pii_in_multiple_turns_all_scrubbed(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "My email is first@a.com"},
                {"role": "assistant", "content": "Got it, first@a.com noted."},
                {"role": "user", "content": "Also reach second@b.com"},
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        assert "first@a.com" not in msgs[0]["content"]
        assert "first@a.com" not in msgs[1]["content"]
        assert "second@b.com" not in msgs[2]["content"]
        assert not result.should_block

    def test_same_value_consistent_across_turns(self):
        """Single engine session → same email always same placeholder."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": "Email: consistent@x.com"},
                {"role": "assistant", "content": "Sure, consistent@x.com it is."},
            ],
        }
        import re
        result = scrub_claude_request(payload, fresh_engine())
        msgs = result.sanitized_payload["messages"]
        ph_u = re.findall(r"\[EMAIL_\d+\]", msgs[0]["content"])
        ph_a = re.findall(r"\[EMAIL_\d+\]", msgs[1]["content"])
        assert ph_u and ph_a
        assert ph_u[0] == ph_a[0]


# ─────────────────────────────────────────────────────────────────────────────
# 14. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_payload(self):
        """Empty payload should not raise."""
        result = scrub_claude_request({}, fresh_engine())
        assert result.sanitized_payload == {}
        assert not result.should_block
        assert not result.coverage_gaps

    def test_no_messages_key(self):
        result = scrub_claude_request({"model": "claude-3"}, fresh_engine())
        assert result.sanitized_payload["model"] == "claude-3"
        assert not result.should_block

    def test_empty_system_string(self):
        payload = {"model": "claude-3", "system": "", "messages": []}
        result = scrub_claude_request(payload, fresh_engine())
        assert result.sanitized_payload["system"] == ""
        assert not result.should_block

    def test_empty_text_block(self):
        payload = {
            "model": "claude-3",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": ""}]}
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        assert result.sanitized_payload["messages"][0]["content"][0]["text"] == ""

    def test_deep_nesting_in_tool_input(self):
        """Deeply nested tool_use.input is fully scanned."""
        key = "ghp_" + "G" * 40
        payload = {
            "model": "claude-3",
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
                                        "level3": {"token": key}
                                    }
                                }
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, fresh_engine())
        token = (
            result.sanitized_payload["messages"][0]["content"][0]["input"]
            ["level1"]["level2"]["level3"]["token"]
        )
        assert key not in token
        assert result.should_block

    def test_original_payload_not_mutated(self):
        """scrub_claude_request must not modify the original payload dict."""
        import copy
        email = "original@check.com"
        payload = {
            "model": "claude-3",
            "system": f"Email: {email}",
            "messages": [{"role": "user", "content": email}],
        }
        original_copy = copy.deepcopy(payload)
        scrub_claude_request(payload, fresh_engine())
        # Payload should be unchanged
        assert payload == original_copy
