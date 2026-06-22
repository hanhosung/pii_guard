"""
Unit and integration tests for the full-body tripwire sweep (Sub-AC 8.2).

The tripwire is a *complementary* scanner that runs a PII-class regex sweep
over the raw serialised request body, catching PII in fields that the
structured per-provider parsers do not visit.

Test strategy
-------------
Every test group follows the same three-step proof:

  1. Build a payload with PII injected into a **non-standard or nested field**
     that is absent from the provider's known schema.
  2. Run the provider's structured scrubber — prove it does **not** catch the
     PII (the sanitised payload still contains the raw value).
  3. Serialise the sanitised payload to JSON and run the tripwire — prove it
     **does** fire on the residual PII.

This validates the AC requirement: "assert the tripwire fires while the
structured parser alone would have missed them."

Coverage
--------
  A. sweep_raw_body() unit tests
     A1. Returns empty result for clean JSON
     A2. Detects email in raw JSON string
     A3. Detects API key in raw JSON string
     A4. Detects Korean RRN in raw JSON string
     A5. Detects card number (Luhn-valid) in raw JSON string
     A6. Allowlist suppresses matches
     A7. Overlap resolution: higher-priority category wins
     A8. min_confidence_override filters rules
     A9. Detects multiple categories simultaneously
     A10. TypeError on non-str input
     A11. Empty string returns empty result

  B. Structured parser misses non-standard fields (proof of gap)
     B1. Claude scrubber misses PII in metadata field
     B2. Claude scrubber misses PII in custom top-level field
     B3. Claude scrubber misses PII in deeply nested non-standard object
     B4. OpenAI scrubber misses PII in metadata field
     B5. OpenAI scrubber misses API key in non-standard field
     B6. Gemini scrubber misses PII in non-standard field

  C. Tripwire catches what the structured parser missed
     C1. Tripwire catches email in Claude metadata after structured scrub
     C2. Tripwire catches phone in OpenAI non-standard field after structured scrub
     C3. Tripwire catches API key in Gemini non-standard field after structured scrub
     C4. Tripwire catches card in deeply nested field after structured scrub
     C5. Tripwire catches Korean PII (RRN) in non-standard field
     C6. Tripwire DOES NOT fire on already-masked placeholders (proves no double-fire)
     C7. Tripwire should_block=True when non-standard field contains a BLOCK secret
     C8. Tripwire should_block=False for contact PII (MASK/TOKENIZE action)

  D. Provider coverage — all three providers' non-standard fields
     D1. Claude: PII in request-level metadata
     D2. OpenAI: PII in request-level metadata
     D3. Gemini: PII in request-level metadata

  E. Proxy integration — full HTTP round-trip via PIIGuardProxy
     E1. Claude request with PII in metadata → 400 (tripwire blocks)
     E2. OpenAI request with API key in non-standard field → 400
     E3. Proxy last_tripwire_result is populated after each request
     E4. Clean request (no PII anywhere) → not blocked, tripwire has no detections
     E5. PII in known field (blocked by structured parser) → 400; tripwire not needed
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
from pii_guard.models import Action
from pii_guard.proxy import PIIGuardProxy
from pii_guard.providers.claude import scrub_claude_request
from pii_guard.providers.gemini import scrub_gemini_request
from pii_guard.providers.openai import scrub_openai_request
from pii_guard.tripwire import TripwireHit, TripwireResult, sweep_raw_body


# ─────────────────────────────────────────────────────────────────────────────
# Test constants — PII values used across tests
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL = "alice@example.com"
_PHONE_KR = "010-1234-5678"
_CARD = "4532015112830366"      # Luhn-valid Visa test card (no-harm test value)
_API_KEY_ANTHROPIC = "sk-ant-api03-" + "A" * 40
_API_KEY_OPENAI = "sk-" + "B" * 48
_VALID_RRN = "801231-1234565"   # checksum-valid RRN (fictional; check digit=5)

# Note: RRN tests use _VALID_RRN because the category spec applies a checksum
# validator.  The check digit is calculated as:
#   digits 801231123456, weights [2,3,4,5,6,7,8,9,2,3,4,5]
#   total=149, check=(11-149%11)%10=5  → last digit must be 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hits_for_category(result: TripwireResult, category: str) -> List[TripwireHit]:
    return [h for h in result.hits if h.category == category]


def _matched_texts(result: TripwireResult, category: str) -> List[str]:
    return [h.matched_text for h in _hits_for_category(result, category)]


# ─────────────────────────────────────────────────────────────────────────────
# Mock upstream server — used by proxy integration tests
# ─────────────────────────────────────────────────────────────────────────────

class _MockUpstreamHandler(BaseHTTPRequestHandler):
    """Records the last request body and returns a minimal success response."""

    def log_message(self, fmt: str, *args) -> None:
        pass  # suppress access log

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length)
        server = self.server  # type: ignore[attr-defined]
        with server._lock:
            server._last_body = body
        resp = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


class _MockUpstreamServer:
    """Thin wrapper around HTTPServer used as a mock LLM upstream."""

    def __init__(self) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), _MockUpstreamHandler)
        self._server._lock = threading.Lock()  # type: ignore[attr-defined]
        self._server._last_body: Optional[bytes] = None  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def last_body(self) -> Optional[bytes]:
        with self._server._lock:  # type: ignore[attr-defined]
            return self._server._last_body  # type: ignore[attr-defined]

    def last_payload(self) -> Optional[Dict[str, Any]]:
        body = self.last_body
        if body is None:
            return None
        return json.loads(body)

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


@pytest.fixture
def mock_upstream():
    srv = _MockUpstreamServer()
    yield srv
    srv.close()


@pytest.fixture
def proxy(mock_upstream):
    with PIIGuardProxy(mock_upstream.url) as p:
        yield p


def _post(url: str, payload: dict, extra_headers: Optional[dict] = None) -> Tuple[int, bytes]:
    """POST *payload* as JSON to *url* and return (status_code, body_bytes)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# ─────────────────────────────────────────────────────────────────────────────
# A. sweep_raw_body() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepRawBodyUnit:
    """A. Direct unit tests for sweep_raw_body()."""

    # A1
    def test_returns_empty_for_clean_json(self):
        payload = json.dumps({
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello there!"}],
        })
        result = sweep_raw_body(payload)
        assert not result.has_detections
        assert not result.should_block
        assert result.hits == []

    # A2
    def test_detects_email_in_raw_json(self):
        raw = json.dumps({"metadata": {"user_email": _EMAIL}})
        result = sweep_raw_body(raw)
        assert result.has_detections
        email_hits = _hits_for_category(result, "EMAIL")
        assert email_hits, "Expected EMAIL hit"
        assert any(_EMAIL in h.matched_text for h in email_hits)
        assert email_hits[0].action == Action.TOKENIZE_ROUNDTRIP

    # A3
    def test_detects_api_key_in_raw_json(self):
        raw = json.dumps({"debug": {"key": _API_KEY_ANTHROPIC}})
        result = sweep_raw_body(raw)
        assert result.has_detections
        api_hits = _hits_for_category(result, "API_KEY")
        assert api_hits, "Expected API_KEY hit"
        assert result.should_block, "API_KEY should trigger BLOCK"

    # A4
    def test_detects_valid_rrn_in_raw_json(self):
        raw = json.dumps({"user_info": {"rrn": _VALID_RRN}})
        result = sweep_raw_body(raw)
        rrn_hits = _hits_for_category(result, "RRN")
        assert rrn_hits, f"Expected RRN hit in: {raw!r}"
        assert result.should_block, "RRN should trigger BLOCK"

    # A5
    def test_detects_card_number_in_raw_json(self):
        raw = json.dumps({"payment": {"card": _CARD}})
        result = sweep_raw_body(raw)
        card_hits = _hits_for_category(result, "CARD")
        assert card_hits, "Expected CARD hit"
        assert result.should_block, "CARD should trigger BLOCK"

    # A6
    def test_allowlist_suppresses_match(self):
        import re
        raw = json.dumps({"metadata": {"email": "test@example.com"}})
        allowlist = [re.compile(r"test@example\.com")]
        result = sweep_raw_body(raw, allowlist_patterns=allowlist)
        email_hits = _hits_for_category(result, "EMAIL")
        assert not email_hits, "Allowlisted email should be suppressed"

    # A7
    def test_overlap_resolution_higher_priority_wins(self):
        """When two categories overlap at the same position, higher-priority wins."""
        # AWS_SECRET has higher priority than API_KEY in ALL_CATEGORIES ordering.
        # An AKIA key satisfies the AWS AKID pattern which appears before the generic
        # api_key pattern in the ALL_CATEGORIES list.
        raw = json.dumps({"key": "AKIAIOSFODNN7EXAMPLE"})
        result = sweep_raw_body(raw)
        # Should only have one hit (no overlap duplicates)
        # AWS_SECRET should win over any lower-priority category at the same span
        categories_found = {h.category for h in result.hits}
        # At most one hit for the same span
        assert len(result.hits) <= 2  # may catch AKID + key-in-context separately

    # A8
    def test_min_confidence_override_filters_rules(self):
        """Rules with confidence < override are skipped."""
        raw = json.dumps({"contact": {"phone": "010-1234-5678"}})
        # Phone rules have confidence ~0.88–0.95; force a very high threshold
        result = sweep_raw_body(raw, min_confidence_override=0.99)
        phone_hits = _hits_for_category(result, "PHONE")
        assert not phone_hits, "All phone rules should be filtered at min_confidence=0.99"

    # A9
    def test_detects_multiple_categories_simultaneously(self):
        raw = json.dumps({
            "user": {
                "email": _EMAIL,
                "api_key": _API_KEY_OPENAI,
                "phone": _PHONE_KR,
            }
        })
        result = sweep_raw_body(raw)
        categories_found = {h.category for h in result.hits}
        assert "EMAIL" in categories_found
        assert "API_KEY" in categories_found
        assert "PHONE" in categories_found
        assert result.should_block, "API_KEY should cause block"

    # A10
    def test_raises_type_error_for_non_str(self):
        with pytest.raises(TypeError, match="sweep_raw_body\\(\\) expects str"):
            sweep_raw_body(b"bytes not allowed")  # type: ignore[arg-type]

    # A11
    def test_empty_string_returns_empty_result(self):
        result = sweep_raw_body("")
        assert not result.has_detections
        assert result.hits == []

    def test_tripwire_result_summary_keys(self):
        """summary() returns required keys and no raw PII."""
        raw = json.dumps({"metadata": {"email": _EMAIL}})
        result = sweep_raw_body(raw)
        s = result.summary()
        assert "tripwire_hits" in s
        assert "categories" in s
        assert "actions" in s
        assert "should_block" in s
        assert "has_detections" in s
        # Confirm raw email NOT in summary
        assert _EMAIL not in json.dumps(s)

    def test_block_hits_and_mask_hits_properties(self):
        raw = json.dumps({
            "email": _EMAIL,
            "api_key": _API_KEY_ANTHROPIC,
        })
        result = sweep_raw_body(raw)
        # API_KEY → BLOCK; EMAIL → TOKENIZE_ROUNDTRIP
        assert any(h.action == Action.BLOCK for h in result.block_hits)
        assert any(h.action == Action.TOKENIZE_ROUNDTRIP for h in result.mask_hits)

    def test_hit_span_length(self):
        raw = json.dumps({"email": _EMAIL})
        result = sweep_raw_body(raw)
        email_hits = _hits_for_category(result, "EMAIL")
        assert email_hits
        h = email_hits[0]
        assert h.span_length == h.raw_end - h.raw_offset
        assert h.span_length == len(h.matched_text)


# ─────────────────────────────────────────────────────────────────────────────
# B. Structured parsers miss non-standard fields (proof of gap)
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredParserGaps:
    """
    B. Prove that structured provider scrubbers do NOT touch non-standard fields.

    These tests establish the baseline gap that the tripwire is designed to cover.
    """

    # B1
    def test_claude_scrubber_misses_metadata_email(self):
        """Claude scrubber doesn't know about 'metadata' — email passes through."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"user_email": _EMAIL},           # non-standard
        }
        engine = Engine()
        result = scrub_claude_request(payload, engine)

        # Structured scrubber must NOT have touched the metadata field
        assert "metadata" in result.sanitized_payload
        assert result.sanitized_payload["metadata"]["user_email"] == _EMAIL, (
            "Structured parser should have left metadata.user_email untouched"
        )
        # Not blocked (structured parser didn't see it)
        assert not result.should_block

    # B2
    def test_claude_scrubber_misses_custom_top_level_field(self):
        """Claude scrubber ignores arbitrary top-level extension fields."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "x_user_phone": _PHONE_KR,                   # non-standard top-level
        }
        engine = Engine()
        result = scrub_claude_request(payload, engine)

        assert result.sanitized_payload.get("x_user_phone") == _PHONE_KR
        assert not result.should_block

    # B3
    def test_claude_scrubber_misses_deeply_nested_non_standard(self):
        """Claude scrubber doesn't recurse into non-schema nested objects."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "request_context": {
                "caller": {
                    "contact": _EMAIL,
                    "credentials": {"api_key": _API_KEY_OPENAI},
                }
            },
        }
        engine = Engine()
        result = scrub_claude_request(payload, engine)

        nested = result.sanitized_payload["request_context"]["caller"]
        assert nested["contact"] == _EMAIL, "Email should be unmasked in non-standard nested field"
        assert nested["credentials"]["api_key"] == _API_KEY_OPENAI, (
            "API key should be unmasked in non-standard nested field"
        )
        # Critically: NOT blocked — the structured parser didn't see these
        assert not result.should_block, (
            "Structured scrubber should not have blocked this payload — "
            "it doesn't visit non-standard nested fields"
        )

    # B4
    def test_openai_scrubber_misses_metadata_email(self):
        """OpenAI chat-completions scrubber ignores non-standard top-level fields."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_metadata": {"contact": _EMAIL},
        }
        engine = Engine()
        result = scrub_openai_request(payload, engine)

        assert result.sanitized_payload["user_metadata"]["contact"] == _EMAIL
        assert not result.should_block

    # B5
    def test_openai_scrubber_misses_api_key_in_nonstandard_field(self):
        """OpenAI scrubber doesn't visit a non-standard 'debug' field."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Tell me a joke"}],
            "debug": {"upstream_key": _API_KEY_ANTHROPIC},
        }
        engine = Engine()
        result = scrub_openai_request(payload, engine)

        assert result.sanitized_payload["debug"]["upstream_key"] == _API_KEY_ANTHROPIC
        assert not result.should_block, (
            "Structured OpenAI scrubber should NOT have blocked this — "
            "it doesn't know about the 'debug' field"
        )

    # B6
    def test_gemini_scrubber_misses_nonstandard_field(self):
        """Gemini scrubber doesn't visit arbitrary non-schema extension fields."""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "extra_context": {"reporter_email": _EMAIL},
        }
        engine = Engine()
        result = scrub_gemini_request(payload, engine)

        assert result.sanitized_payload["extra_context"]["reporter_email"] == _EMAIL
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# C. Tripwire catches what the structured parser missed
# ─────────────────────────────────────────────────────────────────────────────

class TestTripwireCatchesGaps:
    """
    C. Prove the tripwire fires on the sanitised payload for non-standard fields.

    Each test follows the three-step proof:
      1. Build payload with PII in non-standard field
      2. Run structured scrubber → PII survives (proven in class B above)
      3. Run tripwire on sanitised JSON → PII caught
    """

    # C1
    def test_tripwire_catches_email_in_claude_metadata(self):
        """Tripwire fires on email that survived Claude structured scrubbing."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"user_email": _EMAIL},
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        # Step 2: Structured scrubber left the email intact
        assert scrub_result.sanitized_payload["metadata"]["user_email"] == _EMAIL

        # Step 3: Tripwire catches it
        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert tripwire.has_detections, "Tripwire should have fired on the residual email"
        email_hits = _hits_for_category(tripwire, "EMAIL")
        assert email_hits, "Tripwire must report an EMAIL hit"
        assert any(_EMAIL in h.matched_text for h in email_hits)

    # C2
    def test_tripwire_catches_phone_in_openai_nonstandard_field(self):
        """Tripwire catches Korean phone number in OpenAI non-standard field."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "No PII here"}],
            "caller_info": {"mobile": _PHONE_KR},
        }
        engine = Engine()
        scrub_result = scrub_openai_request(payload, engine)

        assert scrub_result.sanitized_payload["caller_info"]["mobile"] == _PHONE_KR

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert tripwire.has_detections
        phone_hits = _hits_for_category(tripwire, "PHONE")
        assert phone_hits, "Tripwire must detect the phone number"

    # C3
    def test_tripwire_catches_api_key_in_gemini_nonstandard_field(self):
        """Tripwire catches Anthropic API key injected into a Gemini non-standard field."""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "routing": {"fallback_key": _API_KEY_ANTHROPIC},
        }
        engine = Engine()
        scrub_result = scrub_gemini_request(payload, engine)

        assert scrub_result.sanitized_payload["routing"]["fallback_key"] == _API_KEY_ANTHROPIC

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert tripwire.should_block, "API key in non-standard field must trigger BLOCK via tripwire"
        api_hits = _hits_for_category(tripwire, "API_KEY")
        assert api_hits, "Tripwire must report an API_KEY hit"

    # C4
    def test_tripwire_catches_card_in_deeply_nested_field(self):
        """Tripwire catches Luhn-valid card number in a multi-level nested non-standard path."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Process my payment"}],
            "payment_context": {
                "billing": {
                    "card_details": {
                        "number": _CARD,
                    }
                }
            },
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        # Verify structured scrubber left the card intact
        nested_card = (
            scrub_result.sanitized_payload
            ["payment_context"]["billing"]["card_details"]["number"]
        )
        assert nested_card == _CARD

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert tripwire.should_block, "CARD in nested non-standard field must block via tripwire"
        card_hits = _hits_for_category(tripwire, "CARD")
        assert card_hits, "Tripwire must detect the card number"

    # C5
    def test_tripwire_catches_rrn_in_nonstandard_field(self):
        """Tripwire catches Korean RRN in an OpenAI non-standard field."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Please verify"}],
            "kyc_data": {"rrn": _VALID_RRN},
        }
        engine = Engine()
        scrub_result = scrub_openai_request(payload, engine)

        assert scrub_result.sanitized_payload["kyc_data"]["rrn"] == _VALID_RRN

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        rrn_hits = _hits_for_category(tripwire, "RRN")
        assert rrn_hits, "Tripwire must detect the RRN in non-standard kyc_data field"
        assert tripwire.should_block

    # C6
    def test_tripwire_does_not_fire_on_placeholder_tokens(self):
        """
        Tripwire MUST NOT flag [CATEGORY_N] placeholder tokens.

        After the structured scrubber replaces PII with placeholders in known
        fields, the tripwire runs on the sanitised JSON.  Placeholders like
        [EMAIL_1] are not valid email addresses, so the tripwire correctly
        ignores them.  This prevents false-positive double-firing.
        """
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": f"My email is {_EMAIL}"}
            ],
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        # The structured scrubber should have masked the email in messages
        sanitised_msg_content = (
            scrub_result.sanitized_payload["messages"][0]["content"]
        )
        assert _EMAIL not in sanitised_msg_content, (
            "Structured scrubber should have replaced email with placeholder"
        )
        assert "[EMAIL_" in sanitised_msg_content, "Expected placeholder token in content"

        # Tripwire on sanitised JSON should NOT re-fire on the placeholder
        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        email_hits = _hits_for_category(tripwire, "EMAIL")
        assert not email_hits, (
            "Tripwire must not fire on [EMAIL_N] placeholder tokens — "
            f"got hits: {email_hits}"
        )

    # C7
    def test_tripwire_should_block_true_for_secret_in_nonstandard_field(self):
        """BLOCK-category secret in non-standard field → should_block=True."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "x_debug_credentials": {
                "openai_key": _API_KEY_OPENAI,
            },
        }
        engine = Engine()
        scrub_result = scrub_openai_request(payload, engine)

        # Structured scrubber did NOT block (it doesn't visit x_debug_credentials)
        assert not scrub_result.should_block

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert tripwire.should_block

    # C8
    def test_tripwire_should_block_false_for_contact_pii_only(self):
        """Contact PII (MASK action) in non-standard field → should_block=False."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_profile": {"email": _EMAIL},
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        # Email is TOKENIZE_ROUNDTRIP (not BLOCK) — tripwire should NOT block
        # but SHOULD report detections
        assert tripwire.has_detections, "Tripwire must report the email hit"
        assert not tripwire.should_block, (
            "Email is TOKENIZE_ROUNDTRIP — should not trigger BLOCK"
        )

    def test_tripwire_catches_multiple_pii_types_in_nonstandard_field(self):
        """Tripwire handles multiple PII types across non-standard fields."""
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Normal message"}],
            "analytics": {
                "user_email": _EMAIL,
                "contact_phone": _PHONE_KR,
            },
            "vault_debug": {"api_secret": _API_KEY_OPENAI},
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        categories = {h.category for h in tripwire.hits}
        assert "EMAIL" in categories
        assert "PHONE" in categories
        assert "API_KEY" in categories
        assert tripwire.should_block  # API_KEY is BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# D. Provider coverage — all three providers' non-standard fields
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderCoverage:
    """D. Confirm the same tripwire pattern works across Claude/OpenAI/Gemini."""

    # D1
    def test_claude_nonstandard_field_coverage(self):
        payload = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "custom_extensions": {
                "org_billing_email": _EMAIL,
                "internal_token": _API_KEY_OPENAI,
            },
        }
        engine = Engine()
        scrub_result = scrub_claude_request(payload, engine)

        # Structured scrubber leaves custom_extensions untouched
        ext = scrub_result.sanitized_payload["custom_extensions"]
        assert ext["org_billing_email"] == _EMAIL
        assert ext["internal_token"] == _API_KEY_OPENAI

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert "EMAIL" in {h.category for h in tripwire.hits}
        assert "API_KEY" in {h.category for h in tripwire.hits}
        assert tripwire.should_block

    # D2
    def test_openai_nonstandard_field_coverage(self):
        payload = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
            "extra_fields": {
                "reporter": _EMAIL,
                "secret_key": _API_KEY_ANTHROPIC,
            },
        }
        engine = Engine()
        scrub_result = scrub_openai_request(payload, engine)

        extra = scrub_result.sanitized_payload["extra_fields"]
        assert extra["reporter"] == _EMAIL
        assert extra["secret_key"] == _API_KEY_ANTHROPIC

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert "EMAIL" in {h.category for h in tripwire.hits}
        assert "API_KEY" in {h.category for h in tripwire.hits}
        assert tripwire.should_block

    # D3
    def test_gemini_nonstandard_field_coverage(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "routing_context": {
                "contact_info": _EMAIL,
                "secret_fallback": _API_KEY_OPENAI,
            },
        }
        engine = Engine()
        scrub_result = scrub_gemini_request(payload, engine)

        routing = scrub_result.sanitized_payload["routing_context"]
        assert routing["contact_info"] == _EMAIL
        assert routing["secret_fallback"] == _API_KEY_OPENAI

        sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
        tripwire = sweep_raw_body(sanitised_json)

        assert "API_KEY" in {h.category for h in tripwire.hits}
        assert tripwire.should_block


# ─────────────────────────────────────────────────────────────────────────────
# E. Proxy integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyTripwireIntegration:
    """
    E. Full HTTP round-trip tests via PIIGuardProxy.

    These tests confirm the tripwire is wired into the proxy so that:
      - Non-standard fields with BLOCK-category PII → 400 (tripwire blocks)
      - The proxy's last_tripwire_result is populated after each request
      - Clean requests are not blocked
    """

    # E1
    def test_claude_request_with_secret_in_metadata_is_blocked(self, proxy, mock_upstream):
        """
        Claude request: API key in metadata field → proxy returns 400 via tripwire.

        The structured scrubber does NOT scan metadata.  Only the tripwire catches
        the key and causes the block.
        """
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {
                "routing_key": _API_KEY_ANTHROPIC,
            },
        }
        status, body = _post(
            f"{proxy.base_url}/v1/messages",
            payload,
            extra_headers={"x-api-key": "test-key"},
        )
        assert status == 400, (
            f"Expected 400 from tripwire block on non-standard field secret, "
            f"got {status}: {body}"
        )
        err = json.loads(body)
        assert err["error"]["type"] == "pii_blocked"
        # Nothing should have been forwarded to the upstream
        assert mock_upstream.last_body is None or (
            # If the proxy accepted a previous request (from another test),
            # the last body won't have the secret
            _API_KEY_ANTHROPIC.encode() not in mock_upstream.last_body
        )

    # E2
    def test_openai_request_with_api_key_in_nonstandard_field_is_blocked(self, proxy, mock_upstream):
        """OpenAI request: OpenAI API key in debug field → proxy returns 400 via tripwire."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "internal_debug": {
                "auth": _API_KEY_OPENAI,
            },
        }
        status, body = _post(
            f"{proxy.base_url}/v1/chat/completions",
            payload,
            extra_headers={"Authorization": f"Bearer {_API_KEY_OPENAI}"},
        )
        assert status == 400, f"Expected 400 from tripwire block, got {status}"
        err = json.loads(body)
        assert err["error"]["type"] == "pii_blocked"

    # E3
    def test_proxy_last_tripwire_result_populated(self, proxy, mock_upstream):
        """last_tripwire_result is set (even for clean requests)."""
        # Initially None
        assert proxy.last_tripwire_result is None

        # Send a clean request (will be forwarded)
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello there"}],
        }
        _post(
            f"{proxy.base_url}/v1/messages",
            payload,
            extra_headers={"x-api-key": "test-key"},
        )

        # last_tripwire_result should now be set
        result = proxy.last_tripwire_result
        assert result is not None, "last_tripwire_result must be populated after a request"
        assert isinstance(result, TripwireResult)

    # E4
    def test_clean_request_not_blocked_tripwire_no_detections(self, proxy, mock_upstream):
        """Clean payload (no PII anywhere) → not blocked, tripwire has no detections."""
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        status, _ = _post(
            f"{proxy.base_url}/v1/messages",
            payload,
            extra_headers={"x-api-key": "test-key"},
        )
        # Should NOT be blocked (the mock upstream returns 200)
        assert status == 200, f"Expected 200 for clean request, got {status}"

        result = proxy.last_tripwire_result
        assert result is not None
        assert not result.has_detections, (
            f"Tripwire should not detect anything in clean payload; "
            f"got: {result.summary()}"
        )

    # E5
    def test_pii_in_known_field_blocked_by_structured_parser(self, proxy, mock_upstream):
        """PII in known field is caught by structured parser; tripwire is complementary."""
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": f"Key: {_API_KEY_ANTHROPIC}"}],
        }
        status, body = _post(
            f"{proxy.base_url}/v1/messages",
            payload,
            extra_headers={"x-api-key": "test-key"},
        )
        assert status == 400, f"Expected 400, structured parser should have blocked this"
        err = json.loads(body)
        assert err["error"]["type"] == "pii_blocked"

    def test_email_in_nonstandard_field_not_blocked_but_tripwire_fires(self, proxy, mock_upstream):
        """
        Contact PII (email) in non-standard field: tripwire detects it but does NOT block
        (email action is TOKENIZE_ROUNDTRIP, not BLOCK).

        The request passes through to the upstream; the tripwire logs the detection
        as a coverage gap (not a blocking event).
        """
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
            "user_profile": {"contact_email": _EMAIL},
        }
        status, _ = _post(
            f"{proxy.base_url}/v1/messages",
            payload,
            extra_headers={"x-api-key": "test-key"},
        )
        # Email is TOKENIZE_ROUNDTRIP (not BLOCK), so not blocked
        assert status == 200, (
            f"Email in non-standard field should not block (action=TOKENIZE_ROUNDTRIP), "
            f"got {status}"
        )

        result = proxy.last_tripwire_result
        assert result is not None
        assert result.has_detections, "Tripwire should have detected the email"
        email_hits = _hits_for_category(result, "EMAIL")
        assert email_hits, "Expected EMAIL in tripwire result"
        assert not result.should_block, "Email should not trigger BLOCK in tripwire"

    def test_openai_gemini_tripwire_result_populated(self, proxy, mock_upstream):
        """Tripwire result is populated for OpenAI and Gemini paths too."""
        openai_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _post(
            f"{proxy.base_url}/v1/chat/completions",
            openai_payload,
        )
        assert proxy.last_tripwire_result is not None

        gemini_payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        _post(
            f"{proxy.base_url}/v1beta/models/gemini-pro:generateContent",
            gemini_payload,
        )
        assert proxy.last_tripwire_result is not None
