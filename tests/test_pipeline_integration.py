"""
Pipeline integration tests for PII-Guard proxy (Sub-AC 2b-ii).

These tests verify the full intercept pipeline end-to-end:
  1. Send an HTTP request containing known PII through the PII-Guard proxy.
  2. Assert the request body captured by the mock upstream contains ONLY
     ``[CATEGORY_N]`` placeholder tokens — no raw PII or secrets.
  3. Confirm the proxy's session mapping store is populated with the correct
     ``placeholder → original`` reverse entries for that request.

Architecture
------------
  Client → PIIGuardProxy (port auto-assigned) → MockUpstreamServer (port auto-assigned)

Both servers run in daemon threads so they are automatically cleaned up when
the test process exits.  Fixtures use ``contextlib.closing`` / ``with`` blocks
to guarantee orderly teardown after each test.

Provider coverage
-----------------
  * Claude   — /v1/messages
  * OpenAI   — /v1/chat/completions
  * Gemini   — /v1beta/models/{model}:generateContent

Scenarios tested
----------------
  Each provider has four core scenarios:
    1. Email (MASK action) → placeholder in forwarded body, mapping populated
    2. Secret / API key (BLOCK action) → 400 returned, nothing forwarded
    3. Mixed PII (email + phone) → both masked in forwarded body
    4. Clean payload → forwarded unchanged (no placeholders, no block)

  Additional cross-provider / integration scenarios:
    5. Same real value → same placeholder (cross-field session consistency)
    6. tool_use / tool_calls / functionCall PII → masked in forwarded body
    7. tool_result / tool role / functionResponse PII → masked
    8. Path routing: unknown path → pass through without scrubbing
    9. Blocked request body is NOT forwarded to upstream
   10. Mapping store accumulates entries across the session
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest

from pii_guard import Engine
from pii_guard.proxy import PIIGuardProxy


# ─────────────────────────────────────────────────────────────────────────────
# Mock upstream server
# ─────────────────────────────────────────────────────────────────────────────

class _MockUpstreamHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP server that records all POST request bodies and returns a
    canned 200 response.  State is shared via the ``MockUpstreamServer``
    instance passed through the ``server`` attribute.
    """

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover
        pass  # suppress access-log noise during tests

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(content_length)
        # Store on the server so the test can inspect it
        self.server.received_requests.append(
            {
                "path": self.path,
                "body": body,
                "headers": dict(self.headers),
            }
        )
        # Return a minimal valid JSON response
        response_body = json.dumps({"id": "mock-response", "ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


class MockUpstreamServer:
    """
    Thread-backed HTTP server that captures POST request bodies.

    Usage::

        with MockUpstreamServer() as upstream:
            # upstream.base_url  → "http://127.0.0.1:<port>"
            # upstream.received_requests  → list of captured requests
            upstream.last_body  → last captured body as bytes
            upstream.last_json  → last captured body parsed as dict
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._server = HTTPServer((host, 0), _MockUpstreamHandler)
        self._server.received_requests: List[Dict[str, Any]] = []
        _h, _p = self._server.server_address
        self._host = _h
        self._port = _p
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def received_requests(self) -> List[Dict[str, Any]]:
        return self._server.received_requests

    @property
    def last_body(self) -> Optional[bytes]:
        return self._server.received_requests[-1]["body"] if self._server.received_requests else None

    @property
    def last_json(self) -> Optional[Dict[str, Any]]:
        body = self.last_body
        return json.loads(body) if body else None

    def reset(self) -> None:
        self._server.received_requests.clear()

    def start(self) -> "MockUpstreamServer":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="mock-upstream",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "MockUpstreamServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    import re
    return bool(re.search(r"\[[A-Z_]+_\d+(?:_BLOCKED)?\]", text))


def _json_text_values(obj: Any) -> List[str]:
    """
    Recursively collect every string value from the JSON structure that could
    carry PII (all string leaf values, regardless of key name).
    """
    results: List[str] = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_json_text_values(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_json_text_values(item))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def upstream():
    """Start a fresh MockUpstreamServer for the test, stop after."""
    with MockUpstreamServer() as srv:
        yield srv


@pytest.fixture()
def proxy(upstream):
    """Start a fresh PIIGuardProxy pointing at *upstream*, stop after."""
    engine = Engine()
    with PIIGuardProxy(
        upstream.base_url,
        engine=engine,
        unknown_field_action="warn_allow",  # keep tests focused on masking, not unknown-block
        unscannable_action="warn_allow",    # ditto
    ) as p:
        yield p


@pytest.fixture()
def strict_proxy(upstream):
    """PIIGuardProxy with default strict settings (block on unknown/unscannable)."""
    engine = Engine()
    with PIIGuardProxy(upstream.base_url, engine=engine) as p:
        yield p


# ─────────────────────────────────────────────────────────────────────────────
# ── Claude pipeline tests (/v1/messages)
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_PATH = "/v1/messages"


class TestClaudePipeline:
    """End-to-end pipeline tests for Claude (Anthropic Messages API)."""

    def test_email_in_system_prompt_masked_in_forwarded_body(self, proxy, upstream):
        """
        A Claude request with an email in the system prompt:
        - The proxy forwards the request (200 from upstream).
        - The forwarded body has [EMAIL_N] instead of the raw email.
        - The mapping store contains email → EMAIL_1 entry.
        """
        email = "alice@example.com"
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "system": f"Always cc {email} on all replies.",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)

        assert status == 200, "Non-blocked request should return 200 from upstream"
        assert upstream.last_json is not None, "Upstream should have received a request"

        forwarded = upstream.last_json
        forwarded_text_values = _json_text_values(forwarded)

        # Raw email must not appear in any string value of the forwarded body
        assert not any(email in v for v in forwarded_text_values), (
            f"Raw email {email!r} must not appear in the forwarded payload. "
            f"Forwarded system: {forwarded.get('system')!r}"
        )

        # A placeholder must appear instead
        assert any(_contains_placeholder(v) for v in forwarded_text_values), (
            "Forwarded payload should contain [EMAIL_N] placeholder"
        )
        # Specifically check the system field
        system_text = forwarded.get("system", "")
        assert "[EMAIL_" in system_text, (
            f"Expected [EMAIL_N] in forwarded system prompt, got: {system_text!r}"
        )

        # Mapping store must contain the reverse entry
        mapping = proxy.restoration_map
        assert any(original == email for original in mapping.values()), (
            f"Mapping store should contain {email!r} as a value. Got: {mapping}"
        )

    def test_api_key_in_message_blocked_not_forwarded(self, proxy, upstream):
        """
        A Claude request containing a secret (API key) must be blocked (400)
        and must NOT be forwarded to the upstream.
        """
        api_key = "sk-ant-api03-" + "A" * 50
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": f"Use key {api_key} for auth."}
            ],
        }

        upstream.reset()
        status, resp_body = _post_json(proxy.base_url + CLAUDE_PATH, payload)

        # Proxy must return 400
        assert status == 400, (
            f"Request with secret should be blocked with 400. Got: {status}"
        )
        # Upstream must NOT have received any request
        assert not upstream.received_requests, (
            "Blocked request must NOT be forwarded to upstream. "
            f"Upstream received: {upstream.received_requests}"
        )
        # Response body should mention blocking
        resp_text = resp_body.decode("utf-8", errors="replace")
        assert "block" in resp_text.lower() or "pii" in resp_text.lower(), (
            f"400 response should indicate blocking. Got: {resp_text!r}"
        )

    def test_mixed_pii_email_and_phone_both_masked(self, proxy, upstream):
        """
        A Claude request with both email and phone in the message text:
        - Both are replaced by placeholders in the forwarded body.
        - Both appear in the mapping store.
        """
        email = "bob@corp.io"
        phone = "010-1234-5678"
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": f"Contact {email} or call {phone}.",
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        msg_text = forwarded["messages"][0]["content"]

        assert email not in msg_text, f"Raw email must not be in forwarded body: {msg_text!r}"
        assert phone not in msg_text, f"Raw phone must not be in forwarded body: {msg_text!r}"
        assert "[EMAIL_" in msg_text, f"Expected [EMAIL_N] placeholder: {msg_text!r}"
        assert "[PHONE_" in msg_text, f"Expected [PHONE_N] placeholder: {msg_text!r}"

        # Both originals in the mapping store
        mapping = proxy.restoration_map
        mapping_values = set(mapping.values())
        assert email in mapping_values, f"Email {email!r} missing from mapping: {mapping}"
        assert phone in mapping_values, f"Phone {phone!r} missing from mapping: {mapping}"

    def test_clean_payload_forwarded_unchanged(self, proxy, upstream):
        """
        A Claude request with no PII or secrets:
        - Forwarded with status 200.
        - Forwarded body contains no placeholders.
        - Mapping store remains empty (no false positives).
        """
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        }

        proxy.engine.reset_session()  # start fresh
        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        forwarded_texts = _json_text_values(forwarded)
        assert not any(_contains_placeholder(v) for v in forwarded_texts), (
            "Clean payload should not have placeholders in forwarded body"
        )
        # Mapping store should be empty (no detections)
        assert proxy.restoration_map == {}, (
            f"Mapping store should be empty for clean payload. Got: {proxy.restoration_map}"
        )

    def test_tool_use_input_pii_masked_in_forwarded_body(self, proxy, upstream):
        """
        A Claude request with PII in a tool_use.input field:
        - The tool input is masked in the forwarded body.
        """
        email = "recipient@domain.com"
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_001",
                            "name": "send_email",
                            "input": {"to": email, "subject": "Test"},
                        }
                    ],
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        tool_input = forwarded["messages"][0]["content"][0]["input"]
        assert email not in tool_input["to"], (
            f"Raw email must not appear in forwarded tool_use input: {tool_input!r}"
        )
        assert "[EMAIL_" in tool_input["to"], (
            f"Expected [EMAIL_N] in forwarded tool_use input: {tool_input!r}"
        )

        # Mapping store
        mapping = proxy.restoration_map
        assert any(v == email for v in mapping.values()), (
            f"Email {email!r} missing from mapping store: {mapping}"
        )

    def test_tool_result_pii_masked_in_forwarded_body(self, proxy, upstream):
        """
        A Claude request with PII in a tool_result.content string:
        - The tool result content is masked in the forwarded body.
        """
        email = "result@example.com"
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_100",
                            "content": f"User email is {email}.",
                        }
                    ],
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        tool_result_content = forwarded["messages"][0]["content"][0]["content"]
        assert email not in tool_result_content, (
            f"Raw email must not appear in forwarded tool_result: {tool_result_content!r}"
        )
        assert "[EMAIL_" in tool_result_content, (
            f"Expected [EMAIL_N] in forwarded tool_result: {tool_result_content!r}"
        )

    def test_cross_field_same_value_same_placeholder(self, proxy, upstream):
        """
        The same email appearing in system prompt and message content must get
        the same placeholder in the forwarded body (session-map consistency).
        """
        import re
        email = "consistent@example.com"
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "system": f"Always contact {email}.",
            "messages": [
                {"role": "user", "content": f"Send to {email} please."}
            ],
        }

        proxy.engine.reset_session()
        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        sys_text = forwarded.get("system", "")
        msg_text = forwarded["messages"][0]["content"]

        sys_placeholders = re.findall(r"\[EMAIL_\d+\]", sys_text)
        msg_placeholders = re.findall(r"\[EMAIL_\d+\]", msg_text)

        assert sys_placeholders, "Expected [EMAIL_N] in forwarded system"
        assert msg_placeholders, "Expected [EMAIL_N] in forwarded message"
        assert sys_placeholders[0] == msg_placeholders[0], (
            "Same email should produce the same placeholder across fields. "
            f"system: {sys_placeholders[0]!r}, message: {msg_placeholders[0]!r}"
        )

    def test_mapping_store_has_correct_reverse_entries(self, proxy, upstream):
        """
        After processing a request, the mapping store contains exactly one entry
        per distinct masked value, with the correct placeholder → original mapping.
        """
        import re
        email = "target@example.com"
        proxy.engine.reset_session()

        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": f"Email: {email}"}
            ],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        msg_text = forwarded["messages"][0]["content"]

        # Find the placeholder that replaced the email
        placeholders = re.findall(r"EMAIL_\d+", msg_text)
        assert placeholders, f"Expected EMAIL_N placeholder in forwarded body: {msg_text!r}"
        placeholder_token = placeholders[0]

        mapping = proxy.restoration_map
        assert placeholder_token in mapping, (
            f"Placeholder {placeholder_token!r} missing from mapping store. "
            f"Store: {mapping}"
        )
        assert mapping[placeholder_token] == email, (
            f"Mapping store should map {placeholder_token!r} → {email!r}. "
            f"Got: {mapping[placeholder_token]!r}"
        )

    def test_forwarded_payload_preserves_structural_fields(self, proxy, upstream):
        """
        The forwarded payload must preserve non-PII structural fields unchanged:
        model, max_tokens, role, type, id, etc.
        """
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 256,
            "system": "You help users.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email is contact@test.com"},
                        {
                            "type": "tool_use",
                            "id": "tu_preserve",
                            "name": "my_tool",
                            "input": {"q": "hi"},
                        },
                    ],
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + CLAUDE_PATH, payload)
        assert status == 200

        fwd = upstream.last_json
        assert fwd["model"] == "claude-opus-4-5"
        assert fwd["max_tokens"] == 256
        assert fwd["messages"][0]["role"] == "user"
        assert fwd["messages"][0]["content"][0]["type"] == "text"
        assert fwd["messages"][0]["content"][1]["type"] == "tool_use"
        assert fwd["messages"][0]["content"][1]["id"] == "tu_preserve"
        assert fwd["messages"][0]["content"][1]["name"] == "my_tool"


# ─────────────────────────────────────────────────────────────────────────────
# ── OpenAI pipeline tests (/v1/chat/completions)
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_PATH = "/v1/chat/completions"


class TestOpenAIPipeline:
    """End-to-end pipeline tests for OpenAI (chat-completions API)."""

    def test_email_in_user_message_masked_in_forwarded_body(self, proxy, upstream):
        """OpenAI: email in user message content is masked in the forwarded body."""
        email = "user@openai-test.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Send the report to {email}."}
            ],
        }

        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        msg_content = forwarded["messages"][0]["content"]

        assert email not in msg_content, (
            f"Raw email must not appear in forwarded body: {msg_content!r}"
        )
        assert "[EMAIL_" in msg_content, (
            f"Expected [EMAIL_N] in forwarded message: {msg_content!r}"
        )

        # Mapping store
        mapping = proxy.restoration_map
        assert any(v == email for v in mapping.values()), (
            f"Email {email!r} missing from mapping store: {mapping}"
        )

    def test_openai_key_in_system_message_blocked(self, proxy, upstream):
        """OpenAI: OpenAI API key in system message is blocked (400), not forwarded."""
        api_key = "sk-" + "x" * 48
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"Use key {api_key} for requests."},
                {"role": "user", "content": "Do it."},
            ],
        }

        upstream.reset()
        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)

        assert status == 400
        assert not upstream.received_requests, (
            "Blocked request must not reach upstream"
        )

    def test_mixed_pii_in_multi_turn_all_masked(self, proxy, upstream):
        """OpenAI: PII across multiple turns is all masked in the forwarded body."""
        email = "multi@turn.com"
        phone = "010-9876-5432"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"My email: {email}"},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": f"Also call me at {phone}"},
            ],
        }

        proxy.engine.reset_session()
        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        msgs = forwarded["messages"]

        assert email not in msgs[0]["content"]
        assert "[EMAIL_" in msgs[0]["content"]
        assert phone not in msgs[2]["content"]
        assert "[PHONE_" in msgs[2]["content"]

        mapping = proxy.restoration_map
        values = set(mapping.values())
        assert email in values
        assert phone in values

    def test_tool_call_arguments_pii_masked(self, proxy, upstream):
        """OpenAI: PII inside tool_calls[*].function.arguments is masked."""
        email = "tool-call@test.com"
        tool_args = json.dumps({"to": email, "subject": "Hello"})
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": tool_args,
                            },
                        }
                    ],
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        fwd_args = forwarded["messages"][0]["tool_calls"][0]["function"]["arguments"]
        parsed_args = json.loads(fwd_args)

        assert email not in parsed_args["to"], (
            f"Raw email must not be in forwarded tool_call args: {parsed_args!r}"
        )
        assert "[EMAIL_" in parsed_args["to"], (
            f"Expected [EMAIL_N] in forwarded tool_call args: {parsed_args!r}"
        )

        mapping = proxy.restoration_map
        assert any(v == email for v in mapping.values()), (
            f"Email {email!r} missing from mapping store: {mapping}"
        )

    def test_tool_role_message_pii_masked(self, proxy, upstream):
        """OpenAI: PII in a tool-role message content is masked."""
        email = "tool-result@example.org"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_xyz",
                    "content": f"Lookup result: {email}",
                }
            ],
        }

        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        content = forwarded["messages"][0]["content"]
        assert email not in content
        assert "[EMAIL_" in content

    def test_clean_openai_payload_passes_through_unmodified(self, proxy, upstream):
        """OpenAI: A clean request reaches upstream unchanged (no placeholders)."""
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Translate this: hello world"}],
        }

        proxy.engine.reset_session()
        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        forwarded_texts = _json_text_values(forwarded)
        assert not any(_contains_placeholder(v) for v in forwarded_texts), (
            "Clean payload should not have placeholders in forwarded body"
        )
        assert proxy.restoration_map == {}

    def test_mapping_store_has_correct_placeholder_entries(self, proxy, upstream):
        """
        OpenAI: Mapping store contains correct placeholder → original entries
        matching what appears in the forwarded body.
        """
        import re
        email = "verify@mapping-store.com"
        proxy.engine.reset_session()

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"My contact is {email}"}],
        }

        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        content = forwarded["messages"][0]["content"]

        # Extract the placeholder from the forwarded body
        token_matches = re.findall(r"EMAIL_\d+", content)
        assert token_matches, f"No EMAIL_N token in forwarded body: {content!r}"
        token = token_matches[0]

        mapping = proxy.restoration_map
        assert token in mapping, (
            f"Token {token!r} not in mapping store. Store: {mapping}"
        )
        assert mapping[token] == email, (
            f"Mapping[{token!r}] should be {email!r}. Got: {mapping[token]!r}"
        )

    def test_developer_role_content_masked(self, proxy, upstream):
        """OpenAI: PII in developer-role messages is masked like user messages."""
        email = "dev@example.com"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": f"Developer note: {email}"},
                {"role": "user", "content": "Proceed."},
            ],
        }

        status, _ = _post_json(proxy.base_url + OPENAI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        dev_content = forwarded["messages"][0]["content"]
        assert email not in dev_content
        assert "[EMAIL_" in dev_content


# ─────────────────────────────────────────────────────────────────────────────
# ── Gemini pipeline tests (/v1beta/models/{model}:generateContent)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_PATH = "/v1beta/models/gemini-1.5-pro:generateContent"


class TestGeminiPipeline:
    """End-to-end pipeline tests for Gemini (generateContent API)."""

    def test_email_in_content_part_masked_in_forwarded_body(self, proxy, upstream):
        """Gemini: email in contents[*].parts[*].text is masked."""
        email = "gemini@example.com"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Contact {email} about this."}],
                }
            ]
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        part_text = forwarded["contents"][0]["parts"][0]["text"]

        assert email not in part_text, (
            f"Raw email must not appear in forwarded Gemini part: {part_text!r}"
        )
        assert "[EMAIL_" in part_text, (
            f"Expected [EMAIL_N] in forwarded Gemini part: {part_text!r}"
        )

        mapping = proxy.restoration_map
        assert any(v == email for v in mapping.values()), (
            f"Email {email!r} missing from mapping store: {mapping}"
        )

    def test_system_instruction_pii_masked(self, proxy, upstream):
        """Gemini: email in systemInstruction is masked."""
        email = "sys@gemini.io"
        payload = {
            "systemInstruction": {
                "parts": [{"text": f"Always cc {email} on responses."}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]}
            ],
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        si_text = forwarded["systemInstruction"]["parts"][0]["text"]
        assert email not in si_text
        assert "[EMAIL_" in si_text

    def test_api_key_in_content_blocked(self, proxy, upstream):
        """Gemini: an API key in content parts is blocked (400), not forwarded."""
        # Use a GCP API key pattern
        api_key = "AIza" + "B" * 35
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Use token {api_key} for auth."}],
                }
            ]
        }

        upstream.reset()
        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)

        assert status == 400
        assert not upstream.received_requests, "Blocked request must not reach upstream"

    def test_function_call_args_pii_masked(self, proxy, upstream):
        """Gemini: PII in functionCall.args string values is masked."""
        email = "func-call@gemini.io"
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_message",
                                "args": {"to": email, "body": "Hi"},
                            }
                        }
                    ],
                }
            ]
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        fc_args = forwarded["contents"][0]["parts"][0]["functionCall"]["args"]
        assert email not in fc_args["to"], (
            f"Raw email must not appear in forwarded functionCall args: {fc_args!r}"
        )
        assert "[EMAIL_" in fc_args["to"], (
            f"Expected [EMAIL_N] in forwarded functionCall args: {fc_args!r}"
        )

    def test_function_response_pii_masked(self, proxy, upstream):
        """Gemini: PII in functionResponse.response string values is masked."""
        email = "func-resp@gemini.io"
        payload = {
            "contents": [
                {
                    "role": "tool",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_user",
                                "response": {"email": email, "name": "Alice"},
                            }
                        }
                    ],
                }
            ]
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        fr_resp = forwarded["contents"][0]["parts"][0]["functionResponse"]["response"]
        assert email not in fr_resp["email"], (
            f"Raw email must not appear in forwarded functionResponse: {fr_resp!r}"
        )
        assert "[EMAIL_" in fr_resp["email"], (
            f"Expected [EMAIL_N] in forwarded functionResponse: {fr_resp!r}"
        )

    def test_clean_gemini_payload_passes_through(self, proxy, upstream):
        """Gemini: A clean request reaches upstream unchanged (no placeholders)."""
        proxy.engine.reset_session()
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "What is the capital of France?"}]}
            ]
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        forwarded_texts = _json_text_values(forwarded)
        assert not any(_contains_placeholder(v) for v in forwarded_texts)
        assert proxy.restoration_map == {}

    def test_snake_case_system_instruction_also_masked(self, proxy, upstream):
        """Gemini: snake_case system_instruction field is also recognised and masked."""
        email = "snake@gemini.io"
        payload = {
            "system_instruction": {
                "parts": [{"text": f"Contact {email}."}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": "OK"}]}
            ],
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        si_text = forwarded["system_instruction"]["parts"][0]["text"]
        assert email not in si_text
        assert "[EMAIL_" in si_text

    def test_mapping_store_has_correct_entries_for_gemini(self, proxy, upstream):
        """Gemini: Mapping store contains correct placeholder → original after request."""
        import re
        email = "gemini-map@check.com"
        proxy.engine.reset_session()

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"My contact: {email}"}]}
            ]
        }

        status, _ = _post_json(proxy.base_url + GEMINI_PATH, payload)
        assert status == 200

        forwarded = upstream.last_json
        part_text = forwarded["contents"][0]["parts"][0]["text"]

        tokens = re.findall(r"EMAIL_\d+", part_text)
        assert tokens, f"No EMAIL_N token in forwarded body: {part_text!r}"
        token = tokens[0]

        mapping = proxy.restoration_map
        assert token in mapping, f"Token {token!r} not in mapping. Store: {mapping}"
        assert mapping[token] == email, (
            f"Mapping[{token!r}] should be {email!r}. Got: {mapping[token]!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── Routing and pass-through tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyRouting:
    """Path routing and pass-through behaviour."""

    def test_unknown_path_passes_through_to_upstream(self, proxy, upstream):
        """
        Requests to unrecognised paths are forwarded unchanged (no scrubbing).
        This ensures the proxy does not accidentally break other API endpoints.
        """
        email = "passthrough@test.com"
        payload = {"data": f"Contact {email}"}

        upstream.reset()
        status, _ = _post_json(proxy.base_url + "/some/unknown/api/endpoint", payload)

        # Upstream should have received the request
        assert status == 200
        assert upstream.last_json is not None
        # The email should NOT have been masked (pass-through path)
        assert upstream.last_json.get("data") == f"Contact {email}", (
            "Unknown path should pass payload through unchanged"
        )

    def test_health_endpoint_returns_200(self, proxy, upstream):
        """GET /health returns 200 OK."""
        req = urllib.request.Request(
            proxy.base_url + "/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

    def test_gemini_v1_path_also_routed(self, proxy, upstream):
        """Gemini v1 path (without 'beta') is also correctly routed."""
        email = "v1@gemini.io"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"Contact {email}"}]}
            ]
        }

        status, _ = _post_json(
            proxy.base_url + "/v1/models/gemini-pro:generateContent",
            payload
        )
        assert status == 200

        forwarded = upstream.last_json
        part_text = forwarded["contents"][0]["parts"][0]["text"]
        assert email not in part_text
        assert "[EMAIL_" in part_text

    def test_blocked_request_not_forwarded_to_upstream(self, strict_proxy, upstream):
        """Blocked request (secret detected) never reaches the upstream server."""
        api_key = "AKIAIOSFODNN7EXAMPLE"  # AWS key pattern
        payload = {
            "model": "claude-opus-4-5",
            "messages": [
                {"role": "user", "content": f"AWS key: {api_key}"}
            ],
        }

        upstream.reset()
        status, _ = _post_json(strict_proxy.base_url + CLAUDE_PATH, payload)

        assert status == 400
        assert not upstream.received_requests, (
            "Blocked request must NOT be forwarded to the upstream server."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── Mapping store accumulation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMappingStoreAccumulation:
    """Verify that the mapping store accumulates entries across multiple requests."""

    def test_mapping_accumulates_across_requests(self, proxy, upstream):
        """
        When the same Engine is reused across requests, the mapping store
        accumulates entries (session-consistent placeholders).
        """
        import re
        proxy.engine.reset_session()

        email1 = "first@example.com"
        email2 = "second@example.com"

        # First request
        _post_json(proxy.base_url + CLAUDE_PATH, {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": f"Email 1: {email1}"}],
        })

        # Second request
        _post_json(proxy.base_url + CLAUDE_PATH, {
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": f"Email 2: {email2}"}],
        })

        mapping = proxy.restoration_map
        values = set(mapping.values())

        assert email1 in values, (
            f"email1 {email1!r} not in accumulated mapping: {mapping}"
        )
        assert email2 in values, (
            f"email2 {email2!r} not in accumulated mapping: {mapping}"
        )

    def test_same_value_always_same_placeholder_across_requests(self, proxy, upstream):
        """
        The same value submitted in two separate requests gets the same placeholder
        (session-map idempotency).
        """
        import re
        proxy.engine.reset_session()
        email = "idempotent@example.com"

        def _extract_placeholder(url: str) -> Optional[str]:
            _post_json(proxy.base_url + url, {
                "model": "claude-opus-4-5",
                "messages": [{"role": "user", "content": f"My email: {email}"}],
            })
            fwd = upstream.last_json
            content = fwd["messages"][0]["content"]
            tokens = re.findall(r"EMAIL_\d+", content)
            return tokens[0] if tokens else None

        ph1 = _extract_placeholder(CLAUDE_PATH)
        ph2 = _extract_placeholder(CLAUDE_PATH)

        assert ph1 is not None
        assert ph2 is not None
        assert ph1 == ph2, (
            "Same email in two separate requests should produce the same placeholder. "
            f"Got: {ph1!r} vs {ph2!r}"
        )

    def test_no_raw_pii_in_any_forwarded_request(self, proxy, upstream):
        """
        Comprehensive check: over multiple requests with different PII types,
        no raw PII appears in any body captured by the upstream mock.
        """
        email = "all-clean@test.org"
        phone = "010-0001-0002"

        requests_payloads = [
            # Claude
            {
                "path": CLAUDE_PATH,
                "payload": {
                    "model": "claude-opus-4-5",
                    "messages": [{"role": "user", "content": f"email: {email}"}],
                },
            },
            # OpenAI
            {
                "path": OPENAI_PATH,
                "payload": {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": f"phone: {phone}"}],
                },
            },
            # Gemini
            {
                "path": GEMINI_PATH,
                "payload": {
                    "contents": [
                        {"role": "user", "parts": [{"text": f"contact {email} or {phone}"}]}
                    ]
                },
            },
        ]

        upstream.reset()
        for item in requests_payloads:
            _post_json(proxy.base_url + item["path"], item["payload"])

        for req_record in upstream.received_requests:
            body = req_record["body"].decode("utf-8")
            assert email not in body, (
                f"Raw email {email!r} found in upstream request body: {body[:200]!r}"
            )
            assert phone not in body, (
                f"Raw phone {phone!r} found in upstream request body: {body[:200]!r}"
            )
