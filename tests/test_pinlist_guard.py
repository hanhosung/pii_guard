"""
Tests for PII-Guard pin-list mutation guard (Sub-AC 5d-i).

Scenarios covered
-----------------
A. **Unit — PinListMutationGuard.check()**

   A1. Agent-sourced mutations always return AGENT_MUTATION_BLOCKED.
   A2. Agent-sourced mutations never reach the approval check.
   A3. Out-of-band mutations without approval return PIN_LIST_NOT_APPROVED.
   A4. Out-of-band mutations with approval are allowed.
   A5. Error dict format is correct (type / message / source keys present).
   A6. String-form source value ("agent") is also blocked.
   A7. No state change occurs on a blocked mutation.
   A8. classify_source() maps bool flag → MutationSource correctly.
   A9. DEFAULT_GUARD singleton is a PinListMutationGuard instance.

B. **Unit — MutationResult dataclass**

   B1. allowed=True result has no error_type/message.
   B2. allowed=False result has structured error fields.
   B3. as_error_dict() returns the canonical error dict shape.

C. **Proxy HTTP endpoint — agent-path mutations always blocked**

   C1. POST /pii-guard/control/pin-list → 403 AGENT_MUTATION_BLOCKED.
   C2. Error response body is valid JSON with the correct error type.
   C3. No body is required in the request — guard fires before body read.
   C4. Query-string variants are also blocked (path stripping).
   C5. Alternate path spellings are all blocked.
   C6. The proxy's upstream URL is NOT contacted (no state forwarded).
   C7. Normal LLM paths (e.g. /v1/messages) are unaffected.
   C8. GET /pii-guard/control/pin-list is not intercepted (only POST).

D. **PolicyLoader integration — out-of-band file mutations**

   D1. Pin-list change without approval → old list retained (guard blocks).
   D2. Pin-list change with approval → new list accepted (guard allows).
   D3. No pin-list change → guard is not invoked (hash unchanged).
   D4. Guard-blocked changes do not persist any new entries.
   D5. Guard logs a warning on block (smoke-check via caplog).

E. **No-state-change invariant**

   E1. After a blocked proxy mutation, proxy engine pin-list state unchanged.
   E2. After a blocked PolicyLoader mutation, config pin-list unchanged.
   E3. Multiple consecutive blocked mutations do not accumulate state.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from pii_guard.pinlist_guard import (
    AGENT_MUTATION_BLOCKED,
    CONTROL_PIN_LIST_PATH,
    DEFAULT_GUARD,
    MutationResult,
    MutationSource,
    PIN_LIST_NOT_APPROVED,
    PinListMutationGuard,
    classify_source,
)
from pii_guard.policy import (
    PolicyLoader,
    PinListEntry,
    _hash_pin_list,
)
from pii_guard.proxy import PIIGuardProxy


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str, delay: float = 0.02) -> None:
    """Write *content* to *path* and bump mtime so reload_if_changed fires."""
    path.write_text(content, encoding="utf-8")
    t = time.time() + delay
    os.utime(str(path), (t, t))


def _post(url: str, body: bytes = b"{}") -> urllib.request.Request:
    """Return a POST Request to *url* with *body*."""
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return req


# ─────────────────────────────────────────────────────────────────────────────
# A. Unit — PinListMutationGuard.check()
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardCheckAgent:
    """A1-A2: Agent-sourced mutations are always blocked."""

    def test_agent_source_blocked(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT)
        assert not result.allowed, "Agent-sourced mutation must be blocked"

    def test_agent_error_type_is_correct_constant(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT)
        assert result.error_type == AGENT_MUTATION_BLOCKED

    def test_agent_source_field(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT)
        assert result.source == "agent"

    def test_agent_blocked_regardless_of_approved_flag_true(self):
        """A2: agent is blocked even if approved=True is mistakenly passed."""
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT, approved=True)
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED

    def test_agent_blocked_regardless_of_approved_flag_false(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT, approved=False)
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED

    def test_agent_error_message_is_not_empty(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT)
        assert result.error_message
        assert len(result.error_message) > 20

    def test_string_source_agent_also_blocked(self):
        """A6: The guard also handles the string value 'agent'."""
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.AGENT.value)  # type: ignore[arg-type]
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED


class TestGuardCheckOutOfBand:
    """A3-A4: Out-of-band mutations gate on approval."""

    def test_out_of_band_without_approval_blocked(self):
        """A3: Out-of-band change without approval is blocked."""
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND, approved=False)
        assert not result.allowed
        assert result.error_type == PIN_LIST_NOT_APPROVED

    def test_out_of_band_source_field(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND, approved=False)
        assert result.source == "out_of_band"

    def test_out_of_band_without_approval_default(self):
        """approved defaults to False, so out-of-band is blocked by default."""
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND)
        assert not result.allowed

    def test_out_of_band_with_approval_allowed(self):
        """A4: Out-of-band change with explicit approval is accepted."""
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND, approved=True)
        assert result.allowed

    def test_out_of_band_approved_no_error_fields(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND, approved=True)
        assert result.error_type is None
        assert result.error_message is None

    def test_out_of_band_approved_source_field(self):
        guard = PinListMutationGuard()
        result = guard.check(MutationSource.OUT_OF_BAND, approved=True)
        assert result.source == "out_of_band"


class TestGuardStateIsolation:
    """A7: Blocked mutations produce no side effects on the guard instance."""

    def test_multiple_blocked_agent_calls_no_accumulation(self):
        """A7/E3: Repeated blocked mutations leave the guard unchanged."""
        guard = PinListMutationGuard()
        for _ in range(10):
            result = guard.check(MutationSource.AGENT)
            assert not result.allowed
            assert result.error_type == AGENT_MUTATION_BLOCKED
        # Guard has no internal state to corrupt — idempotent
        final = guard.check(MutationSource.AGENT)
        assert not final.allowed

    def test_blocked_then_approved_out_of_band(self):
        """Blocking then approving an out-of-band mutation works correctly."""
        guard = PinListMutationGuard()
        r_blocked = guard.check(MutationSource.AGENT)
        assert not r_blocked.allowed
        r_allowed = guard.check(MutationSource.OUT_OF_BAND, approved=True)
        assert r_allowed.allowed


class TestClassifySourceHelper:
    """A8: classify_source() maps bool flag to MutationSource."""

    def test_true_maps_to_agent(self):
        assert classify_source(True) == MutationSource.AGENT

    def test_false_maps_to_out_of_band(self):
        assert classify_source(False) == MutationSource.OUT_OF_BAND

    def test_classify_source_used_with_guard(self):
        guard = PinListMutationGuard()
        result = guard.check(classify_source(True))
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED

        result2 = guard.check(classify_source(False), approved=True)
        assert result2.allowed


class TestDefaultGuardSingleton:
    """A9: DEFAULT_GUARD is a ready-to-use singleton."""

    def test_default_guard_is_instance(self):
        assert isinstance(DEFAULT_GUARD, PinListMutationGuard)

    def test_default_guard_blocks_agent(self):
        result = DEFAULT_GUARD.check(MutationSource.AGENT)
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# B. Unit — MutationResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestMutationResult:
    """B1-B3: MutationResult structure and as_error_dict() format."""

    def test_allowed_result_fields(self):
        """B1: Allowed result has allowed=True, no error fields."""
        result = MutationResult(allowed=True, source="out_of_band")
        assert result.allowed is True
        assert result.error_type is None
        assert result.error_message is None

    def test_blocked_result_fields(self):
        """B2: Blocked result has allowed=False, error fields set."""
        result = MutationResult(
            allowed=False,
            source="agent",
            error_type=AGENT_MUTATION_BLOCKED,
            error_message="test message",
        )
        assert result.allowed is False
        assert result.error_type == AGENT_MUTATION_BLOCKED
        assert result.error_message == "test message"

    def test_as_error_dict_top_level_key(self):
        """B3: as_error_dict() returns dict with 'error' as top-level key."""
        result = MutationResult(
            allowed=False,
            source="agent",
            error_type=AGENT_MUTATION_BLOCKED,
            error_message="blocked",
        )
        d = result.as_error_dict()
        assert "error" in d

    def test_as_error_dict_type_field(self):
        result = MutationResult(
            allowed=False, source="agent",
            error_type=AGENT_MUTATION_BLOCKED, error_message="x"
        )
        assert result.as_error_dict()["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_as_error_dict_source_field(self):
        result = MutationResult(
            allowed=False, source="agent",
            error_type=AGENT_MUTATION_BLOCKED, error_message="x"
        )
        assert result.as_error_dict()["error"]["source"] == "agent"

    def test_as_error_dict_message_field(self):
        result = MutationResult(
            allowed=False, source="agent",
            error_type=AGENT_MUTATION_BLOCKED, error_message="test msg"
        )
        assert result.as_error_dict()["error"]["message"] == "test msg"

    def test_as_error_dict_json_serializable(self):
        """B3: The dict must be JSON-serialisable without error."""
        result = MutationResult(
            allowed=False, source="agent",
            error_type=AGENT_MUTATION_BLOCKED, error_message="blocked"
        )
        serialized = json.dumps(result.as_error_dict())
        reparsed = json.loads(serialized)
        assert reparsed["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_as_error_dict_fallback_when_no_error_type(self):
        """as_error_dict() falls back to UNKNOWN_ERROR if error_type is None."""
        result = MutationResult(allowed=False, source="agent")
        d = result.as_error_dict()
        assert d["error"]["type"] == "UNKNOWN_ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# C. Proxy HTTP endpoint — agent-path mutations always blocked
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def proxy_with_fake_upstream(tmp_path):
    """
    Start a real upstream echo server and a PII-Guard proxy pointing at it.

    The echo server records whether it received any request so tests can
    assert the upstream is NOT contacted when the guard fires.
    """
    # ── Upstream echo server ─────────────────────────────────────────────────
    _upstream_requests = []

    class _EchoHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length)
            _upstream_requests.append(body)
            resp = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):  # pragma: no cover
            pass

    upstream_server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    upstream_port = upstream_server.server_address[1]
    upstream_thread = threading.Thread(
        target=upstream_server.serve_forever, daemon=True
    )
    upstream_thread.start()

    # ── PII-Guard proxy ───────────────────────────────────────────────────────
    proxy = PIIGuardProxy(f"http://127.0.0.1:{upstream_port}")
    proxy.start()

    yield proxy, _upstream_requests

    proxy.stop()
    upstream_server.shutdown()


class TestProxyPinListEndpointBlocked:
    """C1-C8: Proxy always blocks agent-sourced pin-list mutation requests."""

    def _post_to_proxy(self, proxy: PIIGuardProxy, path: str, body: bytes = b"{}"):
        """Helper: POST to *path* on the proxy; return (status_code, response_dict)."""
        url = f"{proxy.base_url}{path}"
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_post_control_pin_list_returns_403(self, proxy_with_fake_upstream):
        """C1: POST to the canonical control path returns HTTP 403."""
        proxy, _ = proxy_with_fake_upstream
        status, _ = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH)
        assert status == 403

    def test_post_control_pin_list_error_type(self, proxy_with_fake_upstream):
        """C2: Response body has the correct AGENT_MUTATION_BLOCKED error type."""
        proxy, _ = proxy_with_fake_upstream
        _, body = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH)
        assert "error" in body
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_post_control_pin_list_error_source_is_agent(self, proxy_with_fake_upstream):
        """C2: Source field is 'agent'."""
        proxy, _ = proxy_with_fake_upstream
        _, body = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH)
        assert body["error"]["source"] == "agent"

    def test_post_control_pin_list_error_message_present(self, proxy_with_fake_upstream):
        """C2: Error message is present and non-empty."""
        proxy, _ = proxy_with_fake_upstream
        _, body = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH)
        assert body["error"]["message"]
        assert len(body["error"]["message"]) > 10

    def test_post_with_empty_body_still_blocked(self, proxy_with_fake_upstream):
        """C3: Guard fires before body is read — empty body is also blocked."""
        proxy, _ = proxy_with_fake_upstream
        status, body = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH, body=b"")
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_post_with_valid_json_body_still_blocked(self, proxy_with_fake_upstream):
        """C3: Sending a well-formed pin-list payload does not bypass the guard."""
        proxy, _ = proxy_with_fake_upstream
        payload = json.dumps({
            "pin_list": [{"hash": "sha256:abc", "category": "EMAIL", "action": "allow"}],
            "pin_list_approved": True,
        }).encode()
        status, body = self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH, body=payload)
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_query_string_variant_blocked(self, proxy_with_fake_upstream):
        """C4: Path with query string is also blocked (path stripping)."""
        proxy, _ = proxy_with_fake_upstream
        status, body = self._post_to_proxy(
            proxy, f"{CONTROL_PIN_LIST_PATH}?approved=true"
        )
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_alternate_path_pinlist_blocked(self, proxy_with_fake_upstream):
        """C5: /pii-guard/control/pinlist (no hyphen) is also blocked."""
        proxy, _ = proxy_with_fake_upstream
        status, body = self._post_to_proxy(proxy, "/pii-guard/control/pinlist")
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_alternate_path_pin_underscore_blocked(self, proxy_with_fake_upstream):
        """C5: /pii-guard/control/pin_list (underscore) is also blocked."""
        proxy, _ = proxy_with_fake_upstream
        status, body = self._post_to_proxy(proxy, "/pii-guard/control/pin_list")
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_alternate_path_piiguard_blocked(self, proxy_with_fake_upstream):
        """C5: /piiguard/control/pin-list (no dash in package prefix) is blocked."""
        proxy, _ = proxy_with_fake_upstream
        status, body = self._post_to_proxy(proxy, "/piiguard/control/pin-list")
        assert status == 403
        assert body["error"]["type"] == AGENT_MUTATION_BLOCKED

    def test_upstream_not_contacted_on_block(self, proxy_with_fake_upstream):
        """C6: The upstream echo server does NOT receive any request."""
        proxy, upstream_requests = proxy_with_fake_upstream
        before_count = len(upstream_requests)
        self._post_to_proxy(proxy, CONTROL_PIN_LIST_PATH)
        # Give any async forwarding a moment to (not) arrive
        time.sleep(0.05)
        assert len(upstream_requests) == before_count, (
            "Guard must not forward the request to the upstream"
        )

    def test_normal_llm_path_unaffected(self, proxy_with_fake_upstream):
        """C7: Normal LLM paths (e.g. /v1/messages) are NOT intercepted by the guard."""
        proxy, _ = proxy_with_fake_upstream
        # The proxy will try to scrub this; since body is valid JSON it will
        # attempt forwarding (upstream will return 200).  We just need to
        # confirm it's NOT returned as AGENT_MUTATION_BLOCKED.
        status, body = self._post_to_proxy(
            proxy, "/v1/messages",
            body=json.dumps({"model": "claude-3-5-sonnet-20241022", "messages": []}).encode()
        )
        # Status can be 200 (upstream echo) or 400 (blocked PII) — but never 403
        # with AGENT_MUTATION_BLOCKED.
        assert status != 403 or body.get("error", {}).get("type") != AGENT_MUTATION_BLOCKED

    def test_get_control_endpoint_not_intercepted(self, proxy_with_fake_upstream):
        """C8: GET to the control path is not intercepted (guard only fires on POST)."""
        proxy, _ = proxy_with_fake_upstream
        url = f"{proxy.base_url}{CONTROL_PIN_LIST_PATH}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        # 405 is the proxy's default for unrecognized GETs — NOT 403
        assert status != 403


# ─────────────────────────────────────────────────────────────────────────────
# D. PolicyLoader integration — out-of-band file mutations
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyLoaderPinListGuardIntegration:
    """D1-D5: PolicyLoader uses PinListMutationGuard for file-based changes."""

    def test_pin_list_change_without_approval_blocked_by_guard(self, tmp_path):
        """D1/D4: Pin-list change without approval retains old list."""
        p = tmp_path / "policy.yaml"
        # First load: one approved pin-list entry
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:original\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))
        assert len(loader.config.pin_list) == 1
        assert loader.config.pin_list[0].hash == "sha256:original"

        # Second load: pin-list changed but NOT approved
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:agent_injected\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: false\n"
        ))
        loader.reload_if_changed()

        # Guard must have blocked this → old entry retained
        assert loader.config.pin_list[0].hash == "sha256:original", (
            "Guard must retain old pin-list when pin_list_approved is false"
        )

    def test_pin_list_change_without_approval_no_new_entries(self, tmp_path):
        """D4: No new pin-list entries leak through when guard blocks."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:original\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        # Attempt to add a second entry without approval
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:original\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "  - hash: sha256:new_injected\n"
            "    category: PHONE\n"
            "    action: allow\n"
            "pin_list_approved: false\n"
        ))
        loader.reload_if_changed()

        # Still only one entry (original)
        assert len(loader.config.pin_list) == 1
        assert loader.config.pin_list[0].hash == "sha256:original"

    def test_pin_list_change_with_approval_accepted_by_guard(self, tmp_path):
        """D2: Pin-list change with approval is accepted."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:old\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        # Second load: different entry WITH approval
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:user_approved_new\n"
            "    category: PHONE\n"
            "    action: block\n"
            "pin_list_approved: true\n"
        ))
        loader.reload_if_changed()

        assert loader.config.pin_list[0].hash == "sha256:user_approved_new"
        assert loader.config.pin_list[0].category == "PHONE"

    def test_no_pin_list_change_guard_not_invoked(self, tmp_path):
        """D3: When the pin-list hash is unchanged, the guard path is skipped."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "fail_mode: open\n"
            "pin_list:\n"
            "  - hash: sha256:stable\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        # Change only non-pin-list fields
        _write(p, (
            "fail_mode: closed\n"
            "pin_list:\n"
            "  - hash: sha256:stable\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader.reload_if_changed()

        # fail_mode updated; pin_list unchanged
        assert loader.config.fail_mode == "closed"
        assert loader.config.pin_list[0].hash == "sha256:stable"

    def test_guard_block_emits_warning(self, tmp_path, caplog):
        """D5: Guard block emits a WARNING-level log entry."""
        import logging
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:original\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:changed\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: false\n"
        ))

        with caplog.at_level(logging.WARNING):
            loader.reload_if_changed()

        # At least one warning mentioning "pin" or "block" should be present
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "pin" in w.lower() or "block" in w.lower() or "guard" in w.lower()
            for w in warnings
        ), f"Expected a pin-list guard warning, got: {warnings}"


# ─────────────────────────────────────────────────────────────────────────────
# E. No-state-change invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestNoStateChange:
    """E1-E3: Blocked mutations leave all state identical to before the attempt."""

    def test_proxy_engine_state_unchanged_after_blocked_mutation(
        self, proxy_with_fake_upstream
    ):
        """E1: After a blocked proxy mutation, engine restoration map is unchanged."""
        proxy, _ = proxy_with_fake_upstream
        before_map = dict(proxy.restoration_map)

        # Attempt agent mutation (blocked)
        url = f"{proxy.base_url}{CONTROL_PIN_LIST_PATH}"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError:
            pass

        after_map = dict(proxy.restoration_map)
        assert before_map == after_map, (
            "Engine restoration map must not change after a blocked mutation"
        )

    def test_policy_loader_pin_list_unchanged_after_blocked_mutation(self, tmp_path):
        """E2: PolicyLoader pin-list stays the same after a blocked file mutation."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:sentinel\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))
        original_pin_list = list(loader.config.pin_list)
        original_hash = _hash_pin_list(original_pin_list)

        # Attempt unapproved change
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:attacker_entry\n"
            "    category: API_KEY\n"
            "    action: allow\n"
            "pin_list_approved: false\n"
        ))
        loader.reload_if_changed()

        current_pin_list = loader.config.pin_list
        current_hash = _hash_pin_list(current_pin_list)

        assert current_hash == original_hash, (
            "Pin-list hash must not change after a blocked mutation"
        )
        assert len(current_pin_list) == 1
        assert current_pin_list[0].hash == "sha256:sentinel"

    def test_consecutive_blocked_mutations_no_accumulation(self, tmp_path):
        """E3: Multiple consecutive blocked mutations never accumulate entries."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:base\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        for i in range(5):
            _write(p, (
                f"pin_list:\n"
                f"  - hash: sha256:attempt{i}\n"
                f"    category: EMAIL\n"
                f"    action: allow\n"
                f"pin_list_approved: false\n"
            ))
            loader.reload_if_changed()

        # After 5 blocked attempts, the pin-list is still just the original entry
        assert len(loader.config.pin_list) == 1
        assert loader.config.pin_list[0].hash == "sha256:base"


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge-case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and boundary conditions for the pin-list guard."""

    def test_constant_agent_mutation_blocked_value(self):
        """The AGENT_MUTATION_BLOCKED constant has the expected string value."""
        assert AGENT_MUTATION_BLOCKED == "AGENT_MUTATION_BLOCKED"

    def test_constant_pin_list_not_approved_value(self):
        """The PIN_LIST_NOT_APPROVED constant has the expected string value."""
        assert PIN_LIST_NOT_APPROVED == "PIN_LIST_NOT_APPROVED"

    def test_control_path_constant_starts_with_slash(self):
        """CONTROL_PIN_LIST_PATH starts with a forward slash."""
        assert CONTROL_PIN_LIST_PATH.startswith("/")

    def test_mutation_source_enum_values(self):
        """MutationSource enum members have the expected string values."""
        assert MutationSource.AGENT.value == "agent"
        assert MutationSource.OUT_OF_BAND.value == "out_of_band"

    def test_guard_is_stateless_across_instances(self):
        """Different guard instances behave identically — no shared mutable state."""
        g1 = PinListMutationGuard()
        g2 = PinListMutationGuard()
        r1 = g1.check(MutationSource.AGENT)
        r2 = g2.check(MutationSource.AGENT)
        assert r1.allowed == r2.allowed
        assert r1.error_type == r2.error_type

    def test_pin_list_empty_stays_approved(self, tmp_path):
        """
        An empty pin-list (the default) has a vacuously approved state.
        A reload that adds a new pin-list without approval should retain
        the empty list.
        """
        p = tmp_path / "policy.yaml"
        # No pin-list initially
        _write(p, "fail_mode: closed\n")
        loader = PolicyLoader(str(p))
        assert loader.config.pin_list == []

        # Add pin-list without approval
        _write(p, (
            "fail_mode: closed\n"
            "pin_list:\n"
            "  - hash: sha256:injected\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: false\n"
        ))
        loader.reload_if_changed()

        # Empty list retained (guard blocks the change)
        assert loader.config.pin_list == []

    def test_guard_import_from_top_level_package(self):
        """PinListMutationGuard is importable from the pii_guard package."""
        import pii_guard
        assert hasattr(pii_guard, "PinListMutationGuard")
        assert hasattr(pii_guard, "AGENT_MUTATION_BLOCKED")
        assert hasattr(pii_guard, "MutationSource")

    def test_proxy_has_pin_list_guard_attribute(self):
        """PIIGuardProxy has a _pin_list_guard attribute."""
        proxy = PIIGuardProxy("http://127.0.0.1:9999")
        assert hasattr(proxy, "_pin_list_guard")
        assert isinstance(proxy._pin_list_guard, PinListMutationGuard)
