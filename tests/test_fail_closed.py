"""
Sub-AC 3a: Fail-closed on scan failure.

Tests that ANY request whose content cannot be scanned is BLOCKED — the
proxy never forwards a request when the scanner fails.

Failure modes exercised:
  • scanner raises RuntimeError (general engine failure)
  • scanner raises TimeoutError (scan timeout)
  • scanner raises ValueError (bad input / internal assertion)
  • scanner raises MemoryError (OOM — simulated)
  • scanner raises an arbitrary Exception subclass

Per-provider coverage:
  • Claude (scrub_claude_request) — system prompt, message text, tool_use,
    tool_result, document block
  • OpenAI (scrub_openai_request) — system/user/tool roles, tool_call args
  • Gemini (scrub_gemini_request) — systemInstruction, contents, functionCall

Additional invariants verified:
  • should_block=True for every failure mode
  • coverage_gap is recorded so the ledger knows the request was not fully scanned
  • fail_reason is populated on the failing FieldScanEvent
  • unscanned text is not present in sanitized_payload (fail-safe even if caller
    erroneously forwards the blocked payload)
  • A clean payload with a *working* engine passes through normally (control)
  • Partial-failure: if one field fails the whole request is blocked
  • Empty-text fields (which skip the scan) are not marked as failures
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import patch

import pytest

from pii_guard import Engine
from pii_guard.providers.claude import scrub_claude_request, FieldScanEvent as ClaudeFieldScanEvent
from pii_guard.providers.openai import scrub_openai_request, FieldScanEvent as OpenAIFieldScanEvent
from pii_guard.providers.gemini import scrub_gemini_request, FieldScanEvent as GeminiFieldScanEvent


# ─────────────────────────────────────────────────────────────────────────────
# Broken engine helpers
# ─────────────────────────────────────────────────────────────────────────────

class _BrokenEngine(Engine):
    """
    Engine subclass whose scan() always raises a configurable exception.
    Used to simulate scanner failure, timeout, and OOM conditions.
    """

    def __init__(self, exc: Optional[Exception] = None) -> None:
        super().__init__()
        self._exc = exc if exc is not None else RuntimeError("scanner unavailable")

    def scan(self, text: str):  # type: ignore[override]
        raise self._exc


def _broken(exc: Optional[Exception] = None) -> _BrokenEngine:
    """Convenience factory."""
    return _BrokenEngine(exc)


def _fresh() -> Engine:
    """Return a healthy engine (control group)."""
    return Engine()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Claude — fail-closed on scan error
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeFailClosed:
    """Claude wire format: every scan failure mode causes should_block=True."""

    # ── 1a. RuntimeError ─────────────────────────────────────────────────────

    def test_runtime_error_in_user_message_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        result = scrub_claude_request(payload, _broken(RuntimeError("internal crash")))
        assert result.should_block, "RuntimeError → fail-closed (block)"

    def test_runtime_error_records_coverage_gap(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_claude_request(payload, _broken(RuntimeError("crash")))
        assert result.coverage_gaps, "Scan error must be recorded as a coverage gap"

    def test_runtime_error_populates_fail_reason(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_claude_request(payload, _broken(RuntimeError("boom")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert failing, "At least one FieldScanEvent must carry a fail_reason"
        assert "RuntimeError" in failing[0].fail_reason

    def test_unscanned_text_not_in_sanitized_payload(self):
        """Scan error → original text removed from sanitized payload (fail-safe)."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Check alice@example.com"}],
        }
        result = scrub_claude_request(payload, _broken())
        content = result.sanitized_payload["messages"][0]["content"]
        # Original (potentially-PII-bearing) text must not appear
        assert "alice@example.com" not in content
        assert "Check alice@example.com" not in content

    # ── 1b. TimeoutError ─────────────────────────────────────────────────────

    def test_timeout_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Scan this"}],
        }
        result = scrub_claude_request(payload, _broken(TimeoutError("scan timed out")))
        assert result.should_block, "TimeoutError → fail-closed (block)"

    def test_timeout_error_records_coverage_gap(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Scan this"}],
        }
        result = scrub_claude_request(payload, _broken(TimeoutError("timeout")))
        assert result.coverage_gaps

    def test_timeout_error_fail_reason_contains_timeout(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_claude_request(payload, _broken(TimeoutError("scan timed out")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert any("TimeoutError" in e.fail_reason for e in failing)

    # ── 1c. ValueError ───────────────────────────────────────────────────────

    def test_value_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, _broken(ValueError("bad state")))
        assert result.should_block

    # ── 1d. MemoryError ──────────────────────────────────────────────────────

    def test_memory_error_blocks(self):
        """Simulates OOM inside the scanner — must still block."""
        payload = {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "Large payload"}],
        }
        result = scrub_claude_request(payload, _broken(MemoryError("OOM")))
        assert result.should_block, "MemoryError (OOM) → fail-closed (block)"
        assert result.coverage_gaps

    # ── 1e. System prompt ────────────────────────────────────────────────────

    def test_system_prompt_scan_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": "You are an assistant. admin@corp.com is the admin.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, _broken())
        assert result.should_block
        # Original text not in output
        assert "admin@corp.com" not in result.sanitized_payload.get("system", "")

    def test_system_prompt_block_array_scan_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "system": [
                {"type": "text", "text": "Contact admin@secret.io for help."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = scrub_claude_request(payload, _broken())
        assert result.should_block
        sys_texts = [
            b.get("text", "") for b in result.sanitized_payload["system"]
            if isinstance(b, dict)
        ]
        for t in sys_texts:
            assert "admin@secret.io" not in t

    # ── 1f. tool_use input ───────────────────────────────────────────────────

    def test_tool_use_input_scan_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_01",
                            "name": "send_email",
                            "input": {"to": "victim@corp.io", "body": "Hello"},
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, _broken())
        assert result.should_block
        inp = result.sanitized_payload["messages"][0]["content"][0]["input"]
        assert "victim@corp.io" not in str(inp)

    # ── 1g. tool_result ──────────────────────────────────────────────────────

    def test_tool_result_string_scan_error_blocks(self):
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_01",
                            "content": "Result: user@example.com",
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, _broken())
        assert result.should_block
        content = result.sanitized_payload["messages"][0]["content"][0]["content"]
        assert "user@example.com" not in content

    # ── 1h. document block ───────────────────────────────────────────────────

    def test_document_text_source_scan_error_blocks(self):
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
                                "data": "Phone: 010-1234-5678",
                            },
                        }
                    ],
                }
            ],
        }
        result = scrub_claude_request(payload, _broken())
        assert result.should_block
        data = result.sanitized_payload["messages"][0]["content"][0]["source"]["data"]
        assert "010-1234-5678" not in data

    # ── 1i. Partial failure: one clean field + one erroring field ─────────────

    def test_partial_failure_blocks_whole_request(self):
        """
        If the scanner fails on *any* field the entire request must be blocked,
        even if other fields would have been clean.

        Strategy: make the engine raise only on specific text by patching
        the internal scan method so the first call succeeds and the second fails.
        """
        call_count = [0]
        real_engine = _fresh()

        def _flaky_scan(text: str):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("scanner went down mid-request")
            return real_engine.scan(text)

        payload = {
            "model": "claude-opus-4-5",
            "system": "Clean system prompt with no PII.",
            "messages": [
                {"role": "user", "content": "Message that triggers crash"}
            ],
        }

        engine = _fresh()
        with patch.object(engine, "scan", side_effect=_flaky_scan):
            result = scrub_claude_request(payload, engine)

        assert result.should_block, (
            "Partial scanner failure must block the whole request"
        )

    # ── 1j. Control: working engine does NOT block clean content ─────────────

    def test_working_engine_clean_payload_not_blocked(self):
        """Sanity check: a healthy engine + clean payload must NOT be blocked."""
        payload = {
            "model": "claude-opus-4-5",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "What time is it?"}],
        }
        result = scrub_claude_request(payload, _fresh())
        assert not result.should_block
        assert not result.coverage_gaps

    # ── 1k. Empty text fields are not scan-error blocked ─────────────────────

    def test_empty_text_not_marked_as_error(self):
        """
        Empty strings skip the scan call entirely; a broken engine should not
        cause empty-text fields to be marked as coverage gaps.
        """
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": ""}]}
            ],
        }
        # Even with a broken engine, empty text is a no-op (scan never called)
        # so there's no coverage gap *from the empty text* — but if the engine
        # is broken, it won't be called for empty text anyway.
        result = scrub_claude_request(payload, _broken())
        # No coverage gaps because empty text bypassed scan
        assert not result.coverage_gaps
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 2. OpenAI — fail-closed on scan error
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIFailClosed:
    """OpenAI wire format: every scan failure mode causes should_block=True."""

    # ── 2a. RuntimeError ─────────────────────────────────────────────────────

    def test_runtime_error_in_user_message_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        result = scrub_openai_request(payload, _broken(RuntimeError("crash")))
        assert result.should_block

    def test_runtime_error_records_coverage_gap(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_openai_request(payload, _broken(RuntimeError("crash")))
        assert result.coverage_gaps

    def test_runtime_error_populates_fail_reason(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_openai_request(payload, _broken(RuntimeError("boom")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert failing
        assert "RuntimeError" in failing[0].fail_reason

    def test_unscanned_text_not_in_sanitized_payload(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Contact bob@secret.org"}],
        }
        result = scrub_openai_request(payload, _broken())
        content = result.sanitized_payload["messages"][0]["content"]
        assert "bob@secret.org" not in content

    # ── 2b. TimeoutError ─────────────────────────────────────────────────────

    def test_timeout_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Scan this text"}],
        }
        result = scrub_openai_request(payload, _broken(TimeoutError("timed out")))
        assert result.should_block

    def test_timeout_error_fail_reason_contains_timeout(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = scrub_openai_request(payload, _broken(TimeoutError("scan timed out")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert any("TimeoutError" in e.fail_reason for e in failing)

    # ── 2c. ValueError ───────────────────────────────────────────────────────

    def test_value_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
        }
        result = scrub_openai_request(payload, _broken(ValueError("bad input")))
        assert result.should_block

    # ── 2d. MemoryError ──────────────────────────────────────────────────────

    def test_memory_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Large payload text"}],
        }
        result = scrub_openai_request(payload, _broken(MemoryError("OOM")))
        assert result.should_block
        assert result.coverage_gaps

    # ── 2e. System message ───────────────────────────────────────────────────

    def test_system_message_scan_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Admin is admin@corp.io."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = scrub_openai_request(payload, _broken())
        assert result.should_block
        sys_content = result.sanitized_payload["messages"][0]["content"]
        assert "admin@corp.io" not in sys_content

    # ── 2f. Tool call arguments ──────────────────────────────────────────────

    def test_tool_call_arguments_scan_error_blocks(self):
        args = json.dumps({"email": "victim@corp.io", "region": "us-east-1"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {"name": "send_email", "arguments": args},
                        }
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, _broken())
        assert result.should_block
        san_args_str = (
            result.sanitized_payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        # Original text must not survive in the sanitized payload
        assert "victim@corp.io" not in san_args_str

    # ── 2g. Tool role message ────────────────────────────────────────────────

    def test_tool_message_scan_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_01",
                    "content": "User: result@example.com",
                }
            ],
        }
        result = scrub_openai_request(payload, _broken())
        assert result.should_block
        assert "result@example.com" not in result.sanitized_payload["messages"][0]["content"]

    # ── 2h. Partial failure ──────────────────────────────────────────────────

    def test_partial_failure_blocks_whole_request(self):
        call_count = [0]
        real_engine = _fresh()

        def _flaky_scan(text: str):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("scanner went down")
            return real_engine.scan(text)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Clean system message."},
                {"role": "user", "content": "Message that crashes scanner"},
            ],
        }
        engine = _fresh()
        with patch.object(engine, "scan", side_effect=_flaky_scan):
            result = scrub_openai_request(payload, engine)

        assert result.should_block

    # ── 2i. Control: working engine clean payload ────────────────────────────

    def test_working_engine_clean_payload_not_blocked(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What time is it?"},
            ],
        }
        result = scrub_openai_request(payload, _fresh())
        assert not result.should_block
        assert not result.coverage_gaps

    # ── 2j. Empty text not marked as error ───────────────────────────────────

    def test_empty_content_not_scanned_not_error(self):
        """Empty string content skips scan entirely; broken engine irrelevant."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": ""}],
        }
        result = scrub_openai_request(payload, _broken())
        assert not result.coverage_gaps
        assert not result.should_block

    # ── 2k. Array text part scan error ───────────────────────────────────────

    def test_text_part_in_array_scan_error_blocks(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email is user@test.com"},
                    ],
                }
            ],
        }
        result = scrub_openai_request(payload, _broken())
        assert result.should_block
        text = result.sanitized_payload["messages"][0]["content"][0]["text"]
        assert "user@test.com" not in text

    # ── 2l. Multi-message: any failure blocks the entire request ─────────────

    def test_multi_message_one_fail_blocks_all(self):
        """
        A request with three messages where the scanner crashes on the second
        must block the entire request (not just skip that message).
        """
        call_count = [0]
        real_engine = _fresh()

        def _fail_on_second(text: str):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("crash on 2nd call")
            return real_engine.scan(text)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "First message — clean"},
                {"role": "assistant", "content": "Second message — crash here"},
                {"role": "user", "content": "Third message — clean"},
            ],
        }
        engine = _fresh()
        with patch.object(engine, "scan", side_effect=_fail_on_second):
            result = scrub_openai_request(payload, engine)

        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gemini — fail-closed on scan error
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiFailClosed:
    """Gemini wire format: every scan failure mode causes should_block=True."""

    # ── 3a. RuntimeError ─────────────────────────────────────────────────────

    def test_runtime_error_in_contents_blocks(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello world"}]}
            ]
        }
        result = scrub_gemini_request(payload, _broken(RuntimeError("crash")))
        assert result.should_block

    def test_runtime_error_records_coverage_gap(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]}
            ]
        }
        result = scrub_gemini_request(payload, _broken(RuntimeError("crash")))
        assert result.coverage_gaps

    def test_runtime_error_populates_fail_reason(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]}
            ]
        }
        result = scrub_gemini_request(payload, _broken(RuntimeError("boom")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert failing
        assert "RuntimeError" in failing[0].fail_reason

    def test_unscanned_text_not_in_sanitized_payload(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Contact carol@secret.io"}]}
            ]
        }
        result = scrub_gemini_request(payload, _broken())
        text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert "carol@secret.io" not in text

    # ── 3b. TimeoutError ─────────────────────────────────────────────────────

    def test_timeout_error_blocks(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Scan this"}]}]
        }
        result = scrub_gemini_request(payload, _broken(TimeoutError("timed out")))
        assert result.should_block

    def test_timeout_error_fail_reason(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        }
        result = scrub_gemini_request(payload, _broken(TimeoutError("NER timed out")))
        failing = [e for e in result.field_events if e.fail_reason]
        assert any("TimeoutError" in e.fail_reason for e in failing)

    # ── 3c. ValueError ───────────────────────────────────────────────────────

    def test_value_error_blocks(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Test"}]}]
        }
        result = scrub_gemini_request(payload, _broken(ValueError("bad state")))
        assert result.should_block

    # ── 3d. MemoryError ──────────────────────────────────────────────────────

    def test_memory_error_blocks(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Large content"}]}]
        }
        result = scrub_gemini_request(payload, _broken(MemoryError("OOM")))
        assert result.should_block
        assert result.coverage_gaps

    # ── 3e. systemInstruction ────────────────────────────────────────────────

    def test_system_instruction_scan_error_blocks(self):
        payload = {
            "systemInstruction": {
                "parts": [{"text": "Admin is admin@secret.io"}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        si_text = (
            result.sanitized_payload["systemInstruction"]["parts"][0]["text"]
        )
        assert "admin@secret.io" not in si_text

    def test_system_instruction_string_form_scan_error_blocks(self):
        """Shorthand string form for systemInstruction is also protected."""
        payload = {
            "systemInstruction": "Contact secret@example.com for help.",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        si = result.sanitized_payload["systemInstruction"]
        assert "secret@example.com" not in si

    def test_snake_case_system_instruction_scan_error_blocks(self):
        """snake_case alias system_instruction is also protected."""
        payload = {
            "system_instruction": {
                "parts": [{"text": "Contact snake@corp.io"}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        si_text = (
            result.sanitized_payload["system_instruction"]["parts"][0]["text"]
        )
        assert "snake@corp.io" not in si_text

    # ── 3f. functionCall args ────────────────────────────────────────────────

    def test_function_call_args_scan_error_blocks(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_email",
                                "args": {"to": "victim@corp.io"},
                            }
                        }
                    ],
                }
            ]
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        args = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]
        )
        assert "victim@corp.io" not in str(args)

    # ── 3g. functionResponse ─────────────────────────────────────────────────

    def test_function_response_scan_error_blocks(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_user",
                                "response": {"email": "resp@corp.io"},
                            }
                        }
                    ],
                }
            ]
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        resp = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionResponse"]["response"]
        )
        assert "resp@corp.io" not in str(resp)

    # ── 3h. executableCode ───────────────────────────────────────────────────

    def test_executable_code_scan_error_blocks(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": "print('secret_token = abc123')",
                            }
                        }
                    ],
                }
            ]
        }
        result = scrub_gemini_request(payload, _broken())
        assert result.should_block
        code = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["executableCode"]["code"]
        )
        assert "secret_token" not in code

    # ── 3i. Partial failure ──────────────────────────────────────────────────

    def test_partial_failure_blocks_whole_request(self):
        call_count = [0]
        real_engine = _fresh()

        def _flaky_scan(text: str):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("scanner crash")
            return real_engine.scan(text)

        payload = {
            "systemInstruction": {"parts": [{"text": "Clean system text"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Content that crashes scanner"}]}
            ],
        }
        engine = _fresh()
        with patch.object(engine, "scan", side_effect=_flaky_scan):
            result = scrub_gemini_request(payload, engine)

        assert result.should_block

    # ── 3j. Control: working engine clean payload ────────────────────────────

    def test_working_engine_clean_payload_not_blocked(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "What time is it?"}]}
            ]
        }
        result = scrub_gemini_request(payload, _fresh())
        assert not result.should_block
        assert not result.coverage_gaps

    # ── 3k. Empty text not scanned, not error ────────────────────────────────

    def test_empty_text_part_not_scan_error(self):
        """Empty text parts skip the scan call; broken engine is irrelevant."""
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": ""}]}
            ]
        }
        result = scrub_gemini_request(payload, _broken())
        assert not result.coverage_gaps
        assert not result.should_block

    # ── 3l. Multi-content: any failure blocks ────────────────────────────────

    def test_multi_content_one_fail_blocks_all(self):
        call_count = [0]
        real_engine = _fresh()

        def _fail_on_second(text: str):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("crash on 2nd scan")
            return real_engine.scan(text)

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "First message"}]},
                {"role": "model", "parts": [{"text": "Second message — crash"}]},
                {"role": "user", "parts": [{"text": "Third message"}]},
            ]
        }
        engine = _fresh()
        with patch.object(engine, "scan", side_effect=_fail_on_second):
            result = scrub_gemini_request(payload, engine)

        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-provider invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossProviderFailCloseInvariants:
    """
    High-level invariants that must hold for all three providers when a scan
    fails.  Uses parametrize so a single assertion covers Claude, OpenAI, and
    Gemini simultaneously.
    """

    @pytest.mark.parametrize("exc_type,exc_msg", [
        (RuntimeError, "scanner unavailable"),
        (TimeoutError, "scan timed out after 5s"),
        (ValueError,   "invalid scanner state"),
        (MemoryError,  "stage2 NER OOM"),
        (Exception,    "unexpected base exception"),
    ])
    def test_all_providers_block_on_exception(self, exc_type, exc_msg):
        """Every exception type causes should_block=True in all three providers."""
        exc = exc_type(exc_msg)
        engine = _broken(exc)

        # Claude
        claude_result = scrub_claude_request(
            {
                "model": "claude-opus-4-5",
                "messages": [{"role": "user", "content": "Sensitive text"}],
            },
            engine,
        )
        assert claude_result.should_block, (
            f"Claude: {exc_type.__name__} must block but should_block=False"
        )

        # OpenAI
        openai_result = scrub_openai_request(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Sensitive text"}],
            },
            _broken(exc),
        )
        assert openai_result.should_block, (
            f"OpenAI: {exc_type.__name__} must block but should_block=False"
        )

        # Gemini
        gemini_result = scrub_gemini_request(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "Sensitive text"}]}
                ]
            },
            _broken(exc),
        )
        assert gemini_result.should_block, (
            f"Gemini: {exc_type.__name__} must block but should_block=False"
        )

    @pytest.mark.parametrize("exc_type", [
        RuntimeError, TimeoutError, ValueError, MemoryError,
    ])
    def test_all_providers_record_coverage_gap_on_exception(self, exc_type):
        """Every exception type causes at least one coverage gap in all three providers."""
        exc = exc_type("failure")

        claude_result = scrub_claude_request(
            {
                "model": "claude-opus-4-5",
                "messages": [{"role": "user", "content": "Some text"}],
            },
            _broken(exc),
        )
        assert claude_result.coverage_gaps, (
            f"Claude: {exc_type.__name__} must produce a coverage gap"
        )

        openai_result = scrub_openai_request(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Some text"}],
            },
            _broken(exc),
        )
        assert openai_result.coverage_gaps, (
            f"OpenAI: {exc_type.__name__} must produce a coverage gap"
        )

        gemini_result = scrub_gemini_request(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "Some text"}]}
                ]
            },
            _broken(exc),
        )
        assert gemini_result.coverage_gaps, (
            f"Gemini: {exc_type.__name__} must produce a coverage gap"
        )

    @pytest.mark.parametrize("exc_type", [
        RuntimeError, TimeoutError, ValueError,
    ])
    def test_all_providers_populate_fail_reason_on_exception(self, exc_type):
        """Every exception type causes fail_reason to be set on failing events."""
        exc = exc_type("failure detail")

        def _check(result, provider_name):
            failing = [e for e in result.field_events if e.fail_reason]
            assert failing, (
                f"{provider_name}: {exc_type.__name__} must set fail_reason on FieldScanEvent"
            )
            assert any(exc_type.__name__ in e.fail_reason for e in failing), (
                f"{provider_name}: fail_reason must name the exception class"
            )

        _check(
            scrub_claude_request(
                {
                    "model": "claude-opus-4-5",
                    "messages": [{"role": "user", "content": "Text"}],
                },
                _broken(exc),
            ),
            "Claude",
        )
        _check(
            scrub_openai_request(
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Text"}],
                },
                _broken(exc),
            ),
            "OpenAI",
        )
        _check(
            scrub_gemini_request(
                {
                    "contents": [
                        {"role": "user", "parts": [{"text": "Text"}]}
                    ]
                },
                _broken(exc),
            ),
            "Gemini",
        )

    def test_all_providers_clear_unscanned_text(self):
        """
        When the scanner fails, original (potentially PII-bearing) text must
        NOT survive in the sanitized payload for any provider.
        """
        pii_text = "Contact victim@pii.org about 010-1234-5678"

        # Claude
        claude_result = scrub_claude_request(
            {
                "model": "claude-opus-4-5",
                "messages": [{"role": "user", "content": pii_text}],
            },
            _broken(),
        )
        claude_content = claude_result.sanitized_payload["messages"][0]["content"]
        assert "victim@pii.org" not in claude_content
        assert "010-1234-5678" not in claude_content

        # OpenAI
        openai_result = scrub_openai_request(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": pii_text}],
            },
            _broken(),
        )
        openai_content = openai_result.sanitized_payload["messages"][0]["content"]
        assert "victim@pii.org" not in openai_content
        assert "010-1234-5678" not in openai_content

        # Gemini
        gemini_result = scrub_gemini_request(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": pii_text}]}
                ]
            },
            _broken(),
        )
        gemini_text = (
            gemini_result.sanitized_payload["contents"][0]["parts"][0]["text"]
        )
        assert "victim@pii.org" not in gemini_text
        assert "010-1234-5678" not in gemini_text
