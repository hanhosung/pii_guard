"""
Tests for Sub-AC 2c — Inbound LLM-response rehydration with terminal-output
restoration OFF.

These tests verify:

  (a) Agent-facing responses returned by the proxy have ``[CATEGORY_N]``
      placeholder tokens replaced with their original real values.

  (b) Terminal-rendered output retains ``[CATEGORY_N]`` tokens when the
      ``terminal_restore`` flag is OFF (the default).

Architecture
------------
  Client → PIIGuardProxy → MockUpstreamServer (echoes back response with tokens)

The MockUpstreamServer is configured to return a canned response JSON that
contains ``[CATEGORY_N]`` placeholder tokens in its content fields (simulating
the scenario where an LLM echoes back placeholders it received in the request).
The proxy's response post-processor must rehydrate those tokens before returning
the response to the client.

Test organisation
-----------------
  TestResponsePostProcessorUnit
      Pure unit tests for ResponsePostProcessor and the provider-specific
      rehydration functions — no I/O.

  TestProxyResponseRehydration
      Integration tests: an HTTP round-trip through PIIGuardProxy verifying that
      the HTTP response body returned to the agent is rehydrated.

  TestTerminalRestoreFlag
      Tests that confirm ``terminal_restore=False`` (default) keeps tokens in
      ``terminal_text`` while the agent response is always rehydrated.

  TestMultiProviderRehydration
      Verifies rehydration correctness for Claude, OpenAI, and Gemini response
      wire formats.
"""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest

from pii_guard.response_rehydrator import (
    RehydrationResult,
    ResponsePostProcessor,
    _rehydrate_claude_response,
    _rehydrate_gemini_response,
    _rehydrate_openai_response,
    _rehydrate_str,
    _extract_terminal_text_claude,
    _extract_terminal_text_openai,
    _extract_terminal_text_gemini,
)
from pii_guard import Engine
from pii_guard.proxy import PIIGuardProxy


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers and fixtures
# ─────────────────────────────────────────────────────────────────────────────

RESTORATION_MAP_BASIC: Dict[str, str] = {
    "EMAIL_1": "alice@corp.io",
    "PHONE_1": "010-1234-5678",
    "PERSON_1": "Alice Smith",
}


def _post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, bytes]:
    """POST *payload* as JSON to *url*; return (status_code, response_body)."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _contains_placeholder(text: str) -> bool:
    """Return True if *text* contains any ``[CATEGORY_N]`` pattern."""
    return bool(re.search(r"\[[A-Z_]+_\d+(?:_BLOCKED)?\]", text))


# ─────────────────────────────────────────────────────────────────────────────
# Configurable mock upstream: returns a preset response body
# ─────────────────────────────────────────────────────────────────────────────

class _ConfigurableUpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        pass  # suppress access log

    def do_POST(self) -> None:
        # Read and discard the request body
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(content_length)

        # Return the preset response
        resp_bytes = self.server.preset_response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


class ConfigurableUpstreamServer:
    """
    Mock upstream server that returns a configurable response body.
    The response body can be changed between tests via ``.preset_response``.
    """

    def __init__(self) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), _ConfigurableUpstreamHandler)
        self._server.preset_response = b'{"ok": true}'
        _h, _p = self._server.server_address
        self._host = _h
        self._port = _p
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def preset_response(self) -> bytes:
        return self._server.preset_response

    @preset_response.setter
    def preset_response(self, value: bytes) -> None:
        self._server.preset_response = value

    def start(self) -> "ConfigurableUpstreamServer":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="configurable-upstream",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "ConfigurableUpstreamServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


@pytest.fixture()
def upstream():
    with ConfigurableUpstreamServer() as srv:
        yield srv


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: proxy variants
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def proxy(upstream):
    """PIIGuardProxy with rehydration ON and terminal_restore OFF (default)."""
    engine = Engine()
    with PIIGuardProxy(
        upstream.base_url,
        engine=engine,
        unknown_field_action="warn_allow",
        unscannable_action="warn_allow",
        rehydrate_responses=True,
        terminal_restore=False,
    ) as p:
        yield p


@pytest.fixture()
def proxy_terminal_restore(upstream):
    """PIIGuardProxy with rehydration ON and terminal_restore ON."""
    engine = Engine()
    with PIIGuardProxy(
        upstream.base_url,
        engine=engine,
        unknown_field_action="warn_allow",
        unscannable_action="warn_allow",
        rehydrate_responses=True,
        terminal_restore=True,
    ) as p:
        yield p


@pytest.fixture()
def proxy_no_rehydrate(upstream):
    """PIIGuardProxy with rehydration disabled."""
    engine = Engine()
    with PIIGuardProxy(
        upstream.base_url,
        engine=engine,
        unknown_field_action="warn_allow",
        unscannable_action="warn_allow",
        rehydrate_responses=False,
    ) as p:
        yield p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests for ResponsePostProcessor and helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestResponsePostProcessorUnit:
    """Pure unit tests for ResponsePostProcessor — no network I/O."""

    def test_terminal_restore_defaults_to_false(self):
        proc = ResponsePostProcessor()
        assert proc.terminal_restore is False

    def test_terminal_restore_can_be_enabled(self):
        proc = ResponsePostProcessor(terminal_restore=True)
        assert proc.terminal_restore is True

    def test_empty_restoration_map_returns_body_unchanged(self):
        proc = ResponsePostProcessor()
        body = b'{"content": [{"type": "text", "text": "[EMAIL_1]"}]}'
        result = proc.process(body, restoration_map={}, provider="claude")
        assert result.agent_body == body
        assert result.substitution_count == 0
        assert result.was_rehydrated is False

    def test_empty_body_returns_empty(self):
        proc = ResponsePostProcessor()
        result = proc.process(b"", restoration_map=RESTORATION_MAP_BASIC, provider="claude")
        assert result.agent_body == b""
        assert result.was_rehydrated is False

    def test_non_json_body_returned_verbatim(self):
        proc = ResponsePostProcessor()
        body = b"this is not json"
        result = proc.process(body, restoration_map=RESTORATION_MAP_BASIC, provider="claude")
        assert result.agent_body == body
        assert result.was_rehydrated is False

    def test_rehydrate_str_basic(self):
        text = "Reply to [EMAIL_1] about the issue."
        result, count = _rehydrate_str(text, {"EMAIL_1": "alice@corp.io"})
        assert result == "Reply to alice@corp.io about the issue."
        assert count == 1

    def test_rehydrate_str_multiple_tokens(self):
        text = "Contact [EMAIL_1] or [PHONE_1]."
        result, count = _rehydrate_str(text, RESTORATION_MAP_BASIC)
        assert "alice@corp.io" in result
        assert "010-1234-5678" in result
        assert count == 2

    def test_rehydrate_str_unknown_token_unchanged(self):
        text = "Contact [GHOST_99]."
        result, count = _rehydrate_str(text, RESTORATION_MAP_BASIC)
        assert "[GHOST_99]" in result
        assert count == 0

    def test_rehydrate_str_longer_token_wins(self):
        """EMAIL_10 must not be partly replaced by EMAIL_1."""
        rmap = {"EMAIL_1": "alice@corp.io", "EMAIL_10": "ten@corp.io"}
        text = "See [EMAIL_10] and [EMAIL_1]."
        result, count = _rehydrate_str(text, rmap)
        assert "ten@corp.io" in result
        assert "alice@corp.io" in result
        assert "[EMAIL_" not in result
        assert count == 2

    def test_was_rehydrated_true_when_substitution_made(self):
        proc = ResponsePostProcessor()
        body = json.dumps({
            "content": [{"type": "text", "text": "Hello [EMAIL_1]!"}]
        }).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")
        assert result.was_rehydrated is True
        assert result.substitution_count == 1

    def test_was_rehydrated_false_when_no_known_tokens(self):
        proc = ResponsePostProcessor()
        body = json.dumps({
            "content": [{"type": "text", "text": "No PII here."}]
        }).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")
        assert result.was_rehydrated is False
        assert result.substitution_count == 0

    def test_provider_is_recorded_in_result(self):
        proc = ResponsePostProcessor()
        body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        result = proc.process(body, {}, "openai")
        assert result.provider == "openai"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Claude response rehydration
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeResponseRehydration:
    """Unit tests for Claude response structure rehydration."""

    def _make_claude_response(self, text: str) -> Dict[str, Any]:
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        }

    def test_text_block_rehydrated(self):
        response = self._make_claude_response(
            "The email you gave me was [EMAIL_1], right?"
        )
        rehydrated, count = _rehydrate_claude_response(
            response, {"EMAIL_1": "alice@corp.io"}
        )
        assert rehydrated["content"][0]["text"] == "The email you gave me was alice@corp.io, right?"
        assert count == 1

    def test_multiple_text_tokens_rehydrated(self):
        response = self._make_claude_response("Contact [EMAIL_1] at [PHONE_1].")
        rehydrated, count = _rehydrate_claude_response(response, RESTORATION_MAP_BASIC)
        text = rehydrated["content"][0]["text"]
        assert "alice@corp.io" in text
        assert "010-1234-5678" in text
        assert count == 2

    def test_tool_use_input_rehydrated(self):
        response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "send_email",
                    "input": {"to": "[EMAIL_1]", "body": "Hi [PERSON_1]"},
                }
            ],
        }
        rehydrated, count = _rehydrate_claude_response(response, RESTORATION_MAP_BASIC)
        tool_input = rehydrated["content"][0]["input"]
        assert tool_input["to"] == "alice@corp.io"
        assert "Alice Smith" in tool_input["body"]
        assert count == 2

    def test_non_content_fields_unchanged(self):
        response = self._make_claude_response("Hello")
        response["model"] = "claude-opus-4-5"
        response["id"] = "msg_preserve_me"
        rehydrated, _ = _rehydrate_claude_response(response, RESTORATION_MAP_BASIC)
        assert rehydrated["model"] == "claude-opus-4-5"
        assert rehydrated["id"] == "msg_preserve_me"

    def test_no_tokens_in_response_count_is_zero(self):
        response = self._make_claude_response("The sky is blue.")
        _, count = _rehydrate_claude_response(response, RESTORATION_MAP_BASIC)
        assert count == 0

    def test_original_response_not_mutated(self):
        """_rehydrate_claude_response must deep-copy — original is not mutated."""
        response = self._make_claude_response("Reply to [EMAIL_1].")
        original_text = response["content"][0]["text"]
        _rehydrate_claude_response(response, {"EMAIL_1": "alice@corp.io"})
        assert response["content"][0]["text"] == original_text

    def test_terminal_text_extraction_returns_text_field(self):
        response = self._make_claude_response("The email is [EMAIL_1].")
        terminal = _extract_terminal_text_claude(response)
        assert "The email is [EMAIL_1]." in terminal

    def test_processor_claude_agent_body_rehydrated(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_claude_response("Reply to [EMAIL_1] soon.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        agent_json = json.loads(result.agent_body)
        assert "alice@corp.io" in agent_json["content"][0]["text"]
        assert "[EMAIL_1]" not in agent_json["content"][0]["text"]

    def test_processor_claude_terminal_text_keeps_tokens_when_flag_off(self):
        """When terminal_restore=False, terminal_text must retain placeholder tokens."""
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_claude_response("Reply to [EMAIL_1] soon.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        # Agent body: rehydrated
        agent_json = json.loads(result.agent_body)
        assert "alice@corp.io" in agent_json["content"][0]["text"]

        # Terminal text: tokens retained
        assert "[EMAIL_1]" in result.terminal_text
        assert "alice@corp.io" not in result.terminal_text

    def test_processor_claude_terminal_text_rehydrated_when_flag_on(self):
        """When terminal_restore=True, terminal_text also has real values."""
        proc = ResponsePostProcessor(terminal_restore=True)
        response = self._make_claude_response("Reply to [EMAIL_1] soon.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        assert "[EMAIL_1]" not in result.terminal_text
        assert "alice@corp.io" in result.terminal_text


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenAI response rehydration
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIResponseRehydration:
    """Unit tests for OpenAI response structure rehydration."""

    def _make_openai_response(self, content: str) -> Dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "index": 0}
            ],
            "model": "gpt-4o",
        }

    def test_message_content_rehydrated(self):
        response = self._make_openai_response("Your email [EMAIL_1] is on file.")
        rehydrated, count = _rehydrate_openai_response(
            response, {"EMAIL_1": "alice@corp.io"}
        )
        assert rehydrated["choices"][0]["message"]["content"] == "Your email alice@corp.io is on file."
        assert count == 1

    def test_tool_call_arguments_json_string_rehydrated(self):
        args_json = json.dumps({"to": "[EMAIL_1]", "subject": "Hello"})
        response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": args_json,
                                },
                            }
                        ],
                    },
                    "index": 0,
                }
            ],
        }
        rehydrated, count = _rehydrate_openai_response(
            response, {"EMAIL_1": "alice@corp.io"}
        )
        new_args_str = rehydrated["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        new_args = json.loads(new_args_str)
        assert new_args["to"] == "alice@corp.io"
        assert count == 1

    def test_non_choices_fields_unchanged(self):
        response = self._make_openai_response("hi")
        response["id"] = "preserve-me"
        response["model"] = "gpt-4o"
        rehydrated, _ = _rehydrate_openai_response(response, RESTORATION_MAP_BASIC)
        assert rehydrated["id"] == "preserve-me"
        assert rehydrated["model"] == "gpt-4o"

    def test_terminal_text_extraction(self):
        response = self._make_openai_response("Contacting [EMAIL_1] now.")
        terminal = _extract_terminal_text_openai(response)
        assert "[EMAIL_1]" in terminal

    def test_processor_openai_agent_body_rehydrated(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_openai_response("Found [EMAIL_1] in the system.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "openai")

        agent_json = json.loads(result.agent_body)
        content = agent_json["choices"][0]["message"]["content"]
        assert "alice@corp.io" in content
        assert "[EMAIL_1]" not in content

    def test_processor_openai_terminal_text_keeps_tokens(self):
        """terminal_restore=False: terminal_text retains [EMAIL_N]."""
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_openai_response("Found [EMAIL_1] in the system.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "openai")

        assert "[EMAIL_1]" in result.terminal_text
        assert "alice@corp.io" not in result.terminal_text

    def test_processor_openai_terminal_text_rehydrated_when_flag_on(self):
        proc = ResponsePostProcessor(terminal_restore=True)
        response = self._make_openai_response("Found [EMAIL_1] in the system.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "openai")

        assert "alice@corp.io" in result.terminal_text
        assert "[EMAIL_1]" not in result.terminal_text


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gemini response rehydration
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiResponseRehydration:
    """Unit tests for Gemini response structure rehydration."""

    def _make_gemini_response(self, text: str) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    def test_parts_text_rehydrated(self):
        response = self._make_gemini_response("I will contact [EMAIL_1].")
        rehydrated, count = _rehydrate_gemini_response(
            response, {"EMAIL_1": "alice@corp.io"}
        )
        text = rehydrated["candidates"][0]["content"]["parts"][0]["text"]
        assert text == "I will contact alice@corp.io."
        assert count == 1

    def test_function_call_args_rehydrated(self):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "send_message",
                                    "args": {"to": "[EMAIL_1]", "msg": "Hi [PERSON_1]"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        rehydrated, count = _rehydrate_gemini_response(response, RESTORATION_MAP_BASIC)
        fc_args = rehydrated["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
        assert fc_args["to"] == "alice@corp.io"
        assert "Alice Smith" in fc_args["msg"]
        assert count == 2

    def test_terminal_text_extraction(self):
        response = self._make_gemini_response("Calling [PHONE_1] shortly.")
        terminal = _extract_terminal_text_gemini(response)
        assert "[PHONE_1]" in terminal

    def test_processor_gemini_agent_body_rehydrated(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_gemini_response("Reply to [EMAIL_1] confirmed.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "gemini")

        agent_json = json.loads(result.agent_body)
        text = agent_json["candidates"][0]["content"]["parts"][0]["text"]
        assert "alice@corp.io" in text
        assert "[EMAIL_1]" not in text

    def test_processor_gemini_terminal_text_keeps_tokens(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._make_gemini_response("Reply to [EMAIL_1] confirmed.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "gemini")

        assert "[EMAIL_1]" in result.terminal_text
        assert "alice@corp.io" not in result.terminal_text

    def test_processor_gemini_terminal_text_rehydrated_when_flag_on(self):
        proc = ResponsePostProcessor(terminal_restore=True)
        response = self._make_gemini_response("Reply to [EMAIL_1] confirmed.")
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "gemini")

        assert "alice@corp.io" in result.terminal_text
        assert "[EMAIL_1]" not in result.terminal_text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Terminal restore flag tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTerminalRestoreFlag:
    """
    Tests that prove the terminal_restore flag semantics:
      - OFF (default): agent body is rehydrated, terminal_text keeps tokens.
      - ON:            both agent body AND terminal_text are rehydrated.
    """

    def _claude_response_with_email(self, email_token: str) -> Dict[str, Any]:
        return {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": f"I will reply to {email_token} promptly."}],
        }

    def test_flag_off_agent_rehydrated_terminal_retains_tokens(self):
        """
        DEFAULT BEHAVIOUR (terminal_restore=False):
        - Agent-facing response body: real values
        - Terminal text: placeholder tokens
        """
        proc = ResponsePostProcessor(terminal_restore=False)
        response = self._claude_response_with_email("[EMAIL_1]")
        body = json.dumps(response).encode()

        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        # (a) Agent body must contain real value
        agent_json = json.loads(result.agent_body)
        agent_text = agent_json["content"][0]["text"]
        assert "alice@corp.io" in agent_text, (
            f"Agent body must contain real email. Got: {agent_text!r}"
        )
        assert "[EMAIL_1]" not in agent_text, (
            f"Agent body must NOT contain placeholder. Got: {agent_text!r}"
        )

        # (b) Terminal text must retain placeholder
        assert "[EMAIL_1]" in result.terminal_text, (
            f"Terminal text must retain [EMAIL_1] when terminal_restore=False. "
            f"Got: {result.terminal_text!r}"
        )
        assert "alice@corp.io" not in result.terminal_text, (
            f"Terminal text must NOT contain real email when terminal_restore=False. "
            f"Got: {result.terminal_text!r}"
        )

    def test_flag_on_both_agent_and_terminal_rehydrated(self):
        """
        TERMINAL RESTORE ENABLED (terminal_restore=True):
        - Agent-facing response body: real values
        - Terminal text: real values
        """
        proc = ResponsePostProcessor(terminal_restore=True)
        response = self._claude_response_with_email("[EMAIL_1]")
        body = json.dumps(response).encode()

        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        # Agent body rehydrated
        agent_json = json.loads(result.agent_body)
        assert "alice@corp.io" in agent_json["content"][0]["text"]

        # Terminal text also rehydrated
        assert "alice@corp.io" in result.terminal_text
        assert "[EMAIL_1]" not in result.terminal_text

    def test_proxy_terminal_restore_defaults_to_false(self, proxy):
        """The proxy's terminal_restore property defaults to False."""
        assert proxy.terminal_restore is False

    def test_proxy_terminal_restore_can_be_set_true(self, upstream):
        engine = Engine()
        with PIIGuardProxy(
            upstream.base_url,
            engine=engine,
            terminal_restore=True,
        ) as p:
            assert p.terminal_restore is True

    def test_multiple_tokens_all_governed_by_same_flag(self):
        """
        With terminal_restore=False, ALL tokens are kept in terminal_text,
        regardless of category.
        """
        proc = ResponsePostProcessor(terminal_restore=False)
        response = {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Contact [EMAIL_1] or [PERSON_1] via [PHONE_1].",
                }
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, RESTORATION_MAP_BASIC, "claude")

        # Agent body has real values
        agent_json = json.loads(result.agent_body)
        agent_text = agent_json["content"][0]["text"]
        assert "alice@corp.io" in agent_text
        assert "Alice Smith" in agent_text
        assert "010-1234-5678" in agent_text

        # Terminal text keeps all tokens
        assert "[EMAIL_1]" in result.terminal_text
        assert "[PERSON_1]" in result.terminal_text
        assert "[PHONE_1]" in result.terminal_text

    def test_blocked_token_also_kept_in_terminal_when_flag_off(self):
        """_BLOCKED tokens are also retained in terminal_text when flag is OFF."""
        proc = ResponsePostProcessor(terminal_restore=False)
        rmap = {"API_KEY_1_BLOCKED": "sk-secret123"}
        response = {
            "id": "msg_03",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The key [API_KEY_1_BLOCKED] was used."}
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, rmap, "claude")

        # Agent body: rehydrated
        agent_json = json.loads(result.agent_body)
        assert "sk-secret123" in agent_json["content"][0]["text"]

        # Terminal text: token retained
        assert "[API_KEY_1_BLOCKED]" in result.terminal_text


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: proxy performs rehydration in HTTP round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyResponseRehydration:
    """
    Integration tests verifying the full HTTP round-trip:
    Client sends PII-bearing request → proxy masks it → upstream echoes back
    a response containing the same placeholder tokens → proxy rehydrates → client
    receives real values.
    """

    CLAUDE_PATH = "/v1/messages"
    OPENAI_PATH = "/v1/chat/completions"
    GEMINI_PATH = "/v1beta/models/gemini-1.5-pro:generateContent"

    def _send_claude_request(self, proxy, email: str) -> Tuple[int, bytes]:
        """Send a Claude request containing *email* through the proxy."""
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": f"My email is {email}."}],
        }
        return _post_json(proxy.base_url + self.CLAUDE_PATH, payload)

    # ── (a) Agent-facing responses are rehydrated ────────────────────────────

    def test_claude_agent_response_is_rehydrated(self, proxy, upstream):
        """
        SCENARIO: Claude — agent-facing response contains real values.
        The upstream echoes back a response with the [EMAIL_N] token.
        The proxy must rehydrate it before returning to the agent.
        """
        email = "agent@test.com"

        # Phase 1: send request so the session map learns email → EMAIL_1
        proxy.engine.reset_session()
        self._send_claude_request(proxy, email)

        # Now the session map has EMAIL_1 → email
        rmap = proxy.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None, f"Email not in restoration_map: {rmap}"

        # Phase 2: configure upstream to respond with the placeholder
        upstream.preset_response = json.dumps({
            "id": "msg_rehydrate",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"I will reply to [{email_token}] shortly.",
                }
            ],
        }).encode()

        # Phase 3: send another request (body doesn't matter — proxy rehydrates response)
        _, resp_body = self._send_claude_request(proxy, email)

        # Agent sees the rehydrated response
        resp_json = json.loads(resp_body)
        agent_text = resp_json["content"][0]["text"]

        assert email in agent_text, (
            f"Agent-facing response must contain real email {email!r}. "
            f"Got: {agent_text!r}"
        )
        assert f"[{email_token}]" not in agent_text, (
            f"Agent-facing response must NOT contain placeholder [{email_token}]. "
            f"Got: {agent_text!r}"
        )

    def test_openai_agent_response_is_rehydrated(self, proxy, upstream):
        """OpenAI: agent-facing response is rehydrated."""
        email = "openai-agent@test.com"
        proxy.engine.reset_session()

        # Send request to populate session map
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"My contact is {email}."}],
        }
        _post_json(proxy.base_url + self.OPENAI_PATH, payload)

        rmap = proxy.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None

        # Configure upstream to return placeholder in response
        upstream.preset_response = json.dumps({
            "id": "chatcmpl-rehydrate",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"I have stored [{email_token}] for you.",
                    },
                    "index": 0,
                }
            ],
        }).encode()

        _, resp_body = _post_json(proxy.base_url + self.OPENAI_PATH, payload)
        resp_json = json.loads(resp_body)
        content = resp_json["choices"][0]["message"]["content"]

        assert email in content, (
            f"OpenAI agent response must contain real email {email!r}. Got: {content!r}"
        )
        assert f"[{email_token}]" not in content, (
            f"Placeholder [{email_token}] must be removed from agent response. Got: {content!r}"
        )

    def test_gemini_agent_response_is_rehydrated(self, proxy, upstream):
        """Gemini: agent-facing response is rehydrated."""
        email = "gemini-agent@test.com"
        proxy.engine.reset_session()

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"My contact: {email}"}]}
            ]
        }
        _post_json(proxy.base_url + self.GEMINI_PATH, payload)

        rmap = proxy.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None

        upstream.preset_response = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": f"Will contact [{email_token}] soon."}],
                        "role": "model",
                    }
                }
            ]
        }).encode()

        _, resp_body = _post_json(proxy.base_url + self.GEMINI_PATH, payload)
        resp_json = json.loads(resp_body)
        text = resp_json["candidates"][0]["content"]["parts"][0]["text"]

        assert email in text, (
            f"Gemini agent response must contain real email {email!r}. Got: {text!r}"
        )
        assert f"[{email_token}]" not in text, (
            f"Placeholder [{email_token}] must be removed from agent response. Got: {text!r}"
        )

    # ── (b) Terminal text retains tokens when terminal_restore is OFF ─────────

    def test_terminal_text_retains_tokens_when_flag_off(self, proxy, upstream):
        """
        SCENARIO: terminal_restore=False (default).
        The last_rehydration_result.terminal_text must contain [EMAIL_N] tokens.
        """
        email = "terminal-test@example.com"
        proxy.engine.reset_session()

        self._send_claude_request(proxy, email)

        rmap = proxy.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None

        # Configure response with the placeholder token
        upstream.preset_response = json.dumps({
            "id": "msg_terminal",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"I noted [{email_token}] in your message."}
            ],
        }).encode()

        self._send_claude_request(proxy, email)

        rr = proxy.last_rehydration_result
        assert rr is not None, "last_rehydration_result should be set after a request"

        # (b) Terminal text keeps the placeholder
        assert f"[{email_token}]" in rr.terminal_text, (
            f"Terminal text must contain [{email_token}] when terminal_restore=False. "
            f"Got: {rr.terminal_text!r}"
        )
        # Agent body has real value
        agent_json = json.loads(rr.agent_body)
        agent_text = agent_json["content"][0]["text"]
        assert email in agent_text, (
            f"Agent body must contain real email {email!r}. Got: {agent_text!r}"
        )

    def test_terminal_text_rehydrated_when_flag_on(self, proxy_terminal_restore, upstream):
        """
        SCENARIO: terminal_restore=True.
        The last_rehydration_result.terminal_text must also contain real values.
        """
        email = "terminal-restore@example.com"
        proxy_terminal_restore.engine.reset_session()

        # Send request to populate map
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": f"My email is {email}."}],
        }
        _post_json(proxy_terminal_restore.base_url + self.CLAUDE_PATH, payload)

        rmap = proxy_terminal_restore.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None

        upstream.preset_response = json.dumps({
            "id": "msg_tr",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Confirmed [{email_token}] is registered."}
            ],
        }).encode()

        _post_json(proxy_terminal_restore.base_url + self.CLAUDE_PATH, payload)

        rr = proxy_terminal_restore.last_rehydration_result
        assert rr is not None

        # Terminal text is rehydrated when flag is ON
        assert email in rr.terminal_text, (
            f"Terminal text should contain real email when terminal_restore=True. "
            f"Got: {rr.terminal_text!r}"
        )

    # ── rehydrate_responses=False: response passed verbatim ──────────────────

    def test_rehydrate_responses_false_passes_response_verbatim(
        self, proxy_no_rehydrate, upstream
    ):
        """
        When rehydrate_responses=False the proxy must NOT modify the response body.
        The placeholder tokens are returned to the client as-is.
        """
        email = "no-rehydrate@test.com"
        proxy_no_rehydrate.engine.reset_session()

        self._send_claude_request(proxy_no_rehydrate, email)

        rmap = proxy_no_rehydrate.restoration_map
        email_token = next((k for k, v in rmap.items() if v == email), None)
        assert email_token is not None

        response_with_token = json.dumps({
            "id": "msg_no_rh",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"[{email_token}] acknowledged."}
            ],
        })
        upstream.preset_response = response_with_token.encode()

        _, resp_body = self._send_claude_request(proxy_no_rehydrate, email)
        resp_text = resp_body.decode("utf-8")

        # The placeholder should still be in the response (no rehydration)
        assert f"[{email_token}]" in resp_text, (
            f"With rehydrate_responses=False, placeholder [{email_token}] must "
            f"remain in response. Got: {resp_text!r}"
        )
        assert email not in resp_text, (
            f"Real email must NOT appear when rehydrate_responses=False. Got: {resp_text!r}"
        )

    # ── Empty restoration map: response not modified ─────────────────────────

    def test_clean_response_with_no_session_map_passes_through(self, proxy, upstream):
        """
        When the session map is empty (no PII was masked) the proxy returns the
        response verbatim without modification.
        """
        proxy.engine.reset_session()
        response_body = json.dumps({
            "id": "msg_clean",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "2 + 2 = 4."}],
        }).encode()
        upstream.preset_response = response_body

        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        }
        _, resp = _post_json(proxy.base_url + self.CLAUDE_PATH, payload)
        resp_json = json.loads(resp)
        assert resp_json["content"][0]["text"] == "2 + 2 = 4."


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-category and multiple-value rehydration
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossCategoryRehydration:
    """Multiple categories in a single response — all must be rehydrated."""

    def test_email_and_phone_both_rehydrated(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = {
            "id": "msg_multi",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Reach [EMAIL_1] or call [PHONE_1].",
                }
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, RESTORATION_MAP_BASIC, "claude")

        agent_json = json.loads(result.agent_body)
        text = agent_json["content"][0]["text"]

        assert "alice@corp.io" in text
        assert "010-1234-5678" in text
        assert "[EMAIL_1]" not in text
        assert "[PHONE_1]" not in text
        assert result.substitution_count == 2

    def test_email_and_phone_tokens_retained_in_terminal(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = {
            "id": "msg_multi",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reach [EMAIL_1] or call [PHONE_1]."}
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, RESTORATION_MAP_BASIC, "claude")

        assert "[EMAIL_1]" in result.terminal_text
        assert "[PHONE_1]" in result.terminal_text
        assert "alice@corp.io" not in result.terminal_text
        assert "010-1234-5678" not in result.terminal_text

    def test_same_token_appearing_twice_both_replaced(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = {
            "id": "msg_dup",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Send to [EMAIL_1]. Reply to [EMAIL_1].",
                }
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        agent_json = json.loads(result.agent_body)
        text = agent_json["content"][0]["text"]
        assert text.count("alice@corp.io") == 2
        assert "[EMAIL_1]" not in text

    def test_unknown_token_left_unchanged_in_agent_body(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        response = {
            "id": "msg_unk",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Known: [EMAIL_1], unknown: [GHOST_99]."}
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, {"EMAIL_1": "alice@corp.io"}, "claude")

        agent_json = json.loads(result.agent_body)
        text = agent_json["content"][0]["text"]
        assert "alice@corp.io" in text
        assert "[GHOST_99]" in text  # unknown token left intact

    def test_substitution_count_is_accurate(self):
        proc = ResponsePostProcessor(terminal_restore=False)
        # 3 tokens in RESTORATION_MAP_BASIC: EMAIL_1, PHONE_1, PERSON_1
        response = {
            "id": "msg_count",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "[EMAIL_1], [PHONE_1], [PERSON_1] — all found.",
                }
            ],
        }
        body = json.dumps(response).encode()
        result = proc.process(body, RESTORATION_MAP_BASIC, "claude")
        assert result.substitution_count == 3
        assert result.was_rehydrated is True
