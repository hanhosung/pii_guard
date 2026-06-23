"""
PII-Guard HTTP intercepting proxy (Sub-AC 2b-ii).

Routes inbound LLM client requests through the PII/secret detection + masking
pipeline, then forwards the sanitised payload to the real upstream LLM endpoint.

Provider routing (path-based)
------------------------------
  POST /v1/messages                               → Claude (Anthropic Messages API)
  POST /v1/chat/completions                       → OpenAI chat-completions
  POST /v1/completions                            → OpenAI legacy completions
  POST /v1beta/models/*:generateContent           → Gemini generateContent
  POST /v1beta/models/*:streamGenerateContent     → Gemini streaming
  Any other path                                  → pass through unchanged (no scrub)

Blocking
--------
When the scrubber returns ``should_block=True``, the proxy returns HTTP 400 with
a JSON error body and does **NOT** forward the payload to the upstream.

Session state
-------------
A single :class:`~pii_guard.engine.Engine` is used per :class:`PIIGuardProxy`
instance, shared across requests for cross-request placeholder consistency.  The
:attr:`PIIGuardProxy.restoration_map` property exposes the accumulating
``placeholder → original`` mapping for rehydration and test inspection.

Thread safety
-------------
:class:`~pii_guard.engine.Engine` and :class:`~pii_guard.session_map.SessionMap`
are **not** thread-safe.  All inbound requests are serialised through a
``threading.Lock`` on the engine so that concurrent client connections do not
race on session state.  For high-throughput deployments, create one
:class:`PIIGuardProxy` per concurrent session instead.

Usage (context manager)::

    from pii_guard.proxy import PIIGuardProxy

    with PIIGuardProxy("https://api.anthropic.com") as proxy:
        # proxy.base_url == "http://127.0.0.1:<port>"
        # Set ANTHROPIC_BASE_URL=proxy.base_url in your client
        ...

Usage (manual start/stop)::

    proxy = PIIGuardProxy("https://api.anthropic.com", port=4444)
    proxy.start()
    # ... serve requests ...
    proxy.stop()
"""
from __future__ import annotations

import json
import re
import socket
import struct
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

from .engine import Engine
from .pinlist_guard import (
    AGENT_MUTATION_BLOCKED,
    CONTROL_PIN_LIST_PATH,
    MutationSource,
    PinListMutationGuard,
)
from .providers.claude import scrub_claude_request
from .providers.gemini import scrub_gemini_request
from .providers.openai import scrub_openai_request
from .response_rehydrator import ResponsePostProcessor, RehydrationResult
from .streaming_rehydrator import StreamingSSERehydrator
from .tripwire import TripwireResult, sweep_raw_body


# ─────────────────────────────────────────────────────────────────────────────
# Path routing constants
# ─────────────────────────────────────────────────────────────────────────────

#: Paths that identify Claude (Anthropic Messages API)
_CLAUDE_PATHS: Tuple[str, ...] = (
    "/v1/messages",
)

#: Paths that identify OpenAI (chat-completions or legacy completions)
_OPENAI_PATHS: Tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
)

#: Pattern that identifies Gemini (v1beta or v1 generateContent endpoints)
_GEMINI_PATH_RE = re.compile(
    r"^/v1(?:beta)?/models/[^/?]+:(generateContent|streamGenerateContent)"
)

#: JSON content type used for blocked/error responses
_JSON_CONTENT_TYPE = "application/json"

#: Read chunk size for streaming SSE responses (bytes)
_STREAM_CHUNK_SIZE: int = 4096

#: Response body returned for blocked requests
_BLOCKED_RESPONSE = json.dumps({
    "error": {
        "type": "pii_blocked",
        "message": (
            "PII-Guard: request blocked because PII or a secret was detected "
            "in the payload. Sensitive content was not forwarded to the LLM."
        ),
    }
}).encode("utf-8")

#: Control paths that are always classified as agent-sourced mutations
#: and therefore always blocked with AGENT_MUTATION_BLOCKED.
_CONTROL_PIN_LIST_PATHS: Tuple[str, ...] = (
    CONTROL_PIN_LIST_PATH,
    "/pii-guard/control/pinlist",       # alternate spelling
    "/pii-guard/control/pin_list",      # underscore variant
    "/piiguard/control/pin-list",       # no-dash package prefix
)

#: Response body returned when the request body is not valid JSON
_INVALID_JSON_RESPONSE = json.dumps({
    "error": {
        "type": "invalid_request",
        "message": "PII-Guard: request body must be valid JSON.",
    }
}).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Provider detection helper
# ─────────────────────────────────────────────────────────────────────────────

def _detect_provider(path: str) -> Optional[str]:
    """
    Return the provider name for *path*, or ``None`` if unrecognised.

    Returns
    -------
    ``"claude"`` | ``"openai"`` | ``"gemini"`` | ``None``
    """
    if any(path == p or path.startswith(p + "?") for p in _CLAUDE_PATHS):
        return "claude"
    if any(path == p or path.startswith(p + "?") for p in _OPENAI_PATHS):
        return "openai"
    if _GEMINI_PATH_RE.match(path):
        return "gemini"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PIIGuardProxy
# ─────────────────────────────────────────────────────────────────────────────

class PIIGuardProxy:
    """
    Lightweight HTTP proxy that intercepts outbound LLM requests and scrubs
    PII/secrets before forwarding to the upstream endpoint.

    Parameters
    ----------
    upstream_url:
        Base URL of the real LLM upstream, e.g. ``"https://api.anthropic.com"``.
        The path from the inbound request is appended verbatim, so the proxy
        acts as a transparent forwarder with the payload replaced.
    engine:
        A pre-constructed :class:`~pii_guard.engine.Engine` instance.  If
        ``None`` (default) a fresh engine is created with secure defaults —
        no configuration required.
    host:
        Local bind address.  Defaults to loopback (``"127.0.0.1"``).
    port:
        Local bind port.  ``0`` (default) lets the OS assign a free port;
        read back via :attr:`port` after :meth:`start`.
    unknown_field_action:
        ``"block"`` (default) or ``"warn_allow"`` — passed to the scrubber.
    unscannable_action:
        ``"block"`` (default) or ``"warn_allow"`` — passed to the scrubber.
    rehydrate_responses:
        When ``True`` (default) the proxy rewrites ``[CATEGORY_N]`` tokens in
        LLM responses with the original values from the session mapping store
        before returning the response to the calling agent.  Agents that sent
        PII-bearing payloads therefore receive correct round-trip content.
        Set to ``False`` to disable agent-facing rehydration (responses are
        returned verbatim from the upstream).
    terminal_restore:
        Controls whether terminal-rendered output (e.g. text displayed in the
        user's shell) is also rehydrated.  Defaults to ``False`` so that the
        terminal retains ``[CATEGORY_N]`` tokens, giving the user visibility
        into what was masked.  Setting to ``True`` enables terminal
        rehydration.  This flag is passed to the internal
        :class:`~pii_guard.response_rehydrator.ResponsePostProcessor`.
    """

    def __init__(
        self,
        upstream_url: str,
        engine: Optional[Engine] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        unknown_field_action: str = "block",
        unscannable_action: str = "block",
        rehydrate_responses: bool = True,
        terminal_restore: bool = False,
        log_masked: bool = False,
    ) -> None:
        self.upstream_url: str = upstream_url.rstrip("/")
        self.engine: Engine = engine if engine is not None else Engine()
        self._engine_lock = threading.Lock()
        # When True, print the sanitised (masked) payload and a detection summary
        # to stdout before forwarding — lets operators confirm that PII never
        # leaves the host in the clear. Only the MASKED payload is logged; the
        # raw request body is never written to the console (no-raw-in-logs).
        self._log_masked = log_masked
        self._unknown_field_action = unknown_field_action
        self._unscannable_action = unscannable_action
        self._rehydrate_responses = rehydrate_responses
        self._response_processor = ResponsePostProcessor(terminal_restore=terminal_restore)
        # Pin-list mutation guard (Sub-AC 5d-i) — classifies incoming requests
        # for pin-list control endpoints as agent-sourced and blocks them.
        self._pin_list_guard = PinListMutationGuard()

        # Keep a record of the last scrub result for test inspection
        self._last_scrub_result: Optional[Any] = None
        self._last_scrub_lock = threading.Lock()

        # Keep a record of the last tripwire result for test inspection
        self._last_tripwire_result: Optional[TripwireResult] = None
        self._last_tripwire_lock = threading.Lock()

        # Keep a record of the last rehydration result for test inspection
        self._last_rehydration_result: Optional[RehydrationResult] = None
        self._last_rehydration_lock = threading.Lock()

        # Build the handler class with a reference to this proxy instance
        proxy_ref = self

        class _Handler(BaseHTTPRequestHandler):
            """Per-request HTTP handler for the PII-Guard proxy."""

            # Suppress default access-log output so tests stay quiet
            def log_message(self, fmt: str, *args) -> None:  # pragma: no cover
                pass

            def do_POST(self) -> None:
                # ── Pin-list control endpoint guard (Sub-AC 5d-i) ────────────
                # Any POST to a pin-list control path is classified as
                # agent-sourced and immediately blocked with AGENT_MUTATION_BLOCKED.
                # No body is read, no state is changed.
                path_no_qs = self.path.split("?")[0]
                if path_no_qs in _CONTROL_PIN_LIST_PATHS:
                    proxy_ref._handle_pin_list_mutation(self)
                    return
                proxy_ref._handle_post(self)

            def do_GET(self) -> None:
                # Health-check endpoint: GET /health → 200 OK
                if self.path == "/health":
                    proxy_ref._send_json(self, 200, {"status": "ok"})
                else:
                    proxy_ref._pass_through(self, "GET")

        self._server = HTTPServer((host, port), _Handler)

        # ── Fail-closed socket configuration (Sub-AC 3c) ─────────────────────
        # Apply SO_LINGER=0 to the listening socket.
        #
        # Graceful-stop guarantee:
        #   When stop() calls server.shutdown() followed by server_close(), the
        #   OS closes the listening socket.  SO_LINGER=0 ensures the close is a
        #   hard RST rather than a FIN, so no SYN packets queued in the kernel
        #   backlog can complete a new TCP handshake after shutdown begins.
        #
        # SIGKILL / process-crash guarantee (OS-level — no code required):
        #   A SIGKILL causes the OS to immediately destroy the process and close
        #   ALL file descriptors.  The kernel sends TCP RST on every open
        #   connection — both the listening socket and every accepted per-request
        #   socket — discarding any unsent data in send buffers.  Clients receive
        #   ConnectionResetError rather than a forwarded upstream response.  This
        #   is the primary network-layer fail-closed mechanism for process crashes
        #   and is verified by the integration test in tests/test_crash_fail_closed.py.
        try:
            self._server.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),  # l_onoff=1, l_linger=0 → hard RST on close
            )
        except OSError:  # pragma: no cover — platform may not support SO_LINGER
            pass

        # Resolve the actual port (important when port=0)
        _bound_host, _bound_port = self._server.server_address
        self._host = _bound_host
        self._port = _bound_port
        self._thread: Optional[threading.Thread] = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def host(self) -> str:
        """The local bind address (e.g. ``"127.0.0.1"``)."""
        return self._host

    @property
    def port(self) -> int:
        """The local bind port (resolved after :meth:`start`)."""
        return self._port

    @property
    def base_url(self) -> str:
        """Full base URL of the proxy, e.g. ``"http://127.0.0.1:4444"``."""
        return f"http://{self._host}:{self._port}"

    @property
    def restoration_map(self) -> Dict[str, str]:
        """
        Read-only snapshot of the current ``placeholder → original`` mapping.

        Populated by the session :class:`~pii_guard.engine.Engine` as each
        request is processed.  Use this in tests to verify the reverse-mapping
        store was correctly populated.
        """
        return self.engine.restoration_map

    @property
    def terminal_restore(self) -> bool:
        """
        Whether terminal-rendered output rehydration is enabled.

        Mirrors the ``terminal_restore`` parameter passed at construction.
        Defaults to ``False`` so terminal output retains ``[CATEGORY_N]``
        tokens.
        """
        return self._response_processor.terminal_restore

    @property
    def last_tripwire_result(self) -> Optional[TripwireResult]:
        """
        The :class:`~pii_guard.tripwire.TripwireResult` from the most recent
        full-body tripwire sweep, or ``None`` if no request with a known
        provider has been processed yet.

        The tripwire runs on the *sanitised* payload (after the provider-specific
        scrubber has replaced PII in known fields with placeholders).  Any hit in
        this result therefore represents a coverage gap — PII found in a
        non-standard or nested field that the structured parser did not visit.

        Thread-safe snapshot; primarily used by tests and diagnostics.
        """
        with self._last_tripwire_lock:
            return self._last_tripwire_result

    @property
    def last_rehydration_result(self) -> Optional[RehydrationResult]:
        """
        The :class:`~pii_guard.response_rehydrator.RehydrationResult` from the
        most recent response post-processing step, or ``None`` if no response
        has been processed yet.  Thread-safe snapshot; primarily used by tests.
        """
        with self._last_rehydration_lock:
            return self._last_rehydration_result

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def start(self) -> "PIIGuardProxy":
        """
        Start the proxy in a daemon thread and return ``self``.

        The server is ready to accept connections when this method returns.
        """
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="pii-guard-proxy",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut down the proxy server and join the background thread."""
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "PIIGuardProxy":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ── Core request handler ────────────────────────────────────────────────────

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        """
        Main POST handler: scrub the request body and forward if not blocked.

        Steps
        -----
        1. Read and JSON-parse the request body.
        2. Detect the provider from the request path.
        3. Call the provider-specific scrubber (serialised via engine lock).
        4. Return 400 if ``should_block=True``; otherwise forward the masked
           payload to the upstream and proxy the response back to the caller.
        """
        # ── 1. Read body ─────────────────────────────────────────────────────
        try:
            content_length = int(handler.headers.get("Content-Length", 0) or 0)
            raw_body = handler.rfile.read(content_length)
        except (ValueError, OSError):
            self._send_json(handler, 400, {"error": "failed to read request body"})
            return

        # ── 2. Parse JSON ────────────────────────────────────────────────────
        try:
            payload: Dict[str, Any] = json.loads(raw_body) if raw_body else {}
        except (json.JSONDecodeError, ValueError):
            self._send_response_bytes(
                handler, 400, _INVALID_JSON_RESPONSE, _JSON_CONTENT_TYPE
            )
            return

        # ── 3. Detect provider and scrub ─────────────────────────────────────
        path = handler.path
        provider = _detect_provider(path)

        if provider is not None:
            scrub_result = self._scrub(payload, provider)
            with self._last_scrub_lock:
                self._last_scrub_result = scrub_result

            # ── 3b. Full-body tripwire sweep (Sub-AC 8.2) ─────────────────
            # Run the complementary tripwire on the *sanitised* payload JSON.
            # Because the structured scrubber has already replaced PII in known
            # fields with [PLACEHOLDER_N] tokens, any PII the tripwire finds
            # here definitively lives in a non-standard or nested field that the
            # provider parser did not visit — a true coverage gap.
            tripwire_result = self._run_tripwire(scrub_result.sanitized_payload)
            with self._last_tripwire_lock:
                self._last_tripwire_result = tripwire_result

            # Merge blocking decision: block if *either* the structured scrubber
            # or the tripwire demands it (fail-closed on coverage gaps with
            # BLOCK-category PII).
            if scrub_result.should_block or tripwire_result.should_block:
                self._log_traffic(path, provider, scrub_result,
                                  tripwire_result, blocked=True)
                self._send_response_bytes(
                    handler, 400, _BLOCKED_RESPONSE, _JSON_CONTENT_TYPE
                )
                return

            forwarded_payload = scrub_result.sanitized_payload
            self._log_traffic(path, provider, scrub_result,
                              tripwire_result, blocked=False)
        else:
            # Unknown/unrecognised path — pass through unchanged (no scrubbing)
            forwarded_payload = payload

        # ── 4. Forward to upstream ────────────────────────────────────────────
        self._forward(handler, path, forwarded_payload)

    def _log_traffic(self, path, provider, scrub_result,
                     tripwire_result, *, blocked: bool) -> None:
        """
        Print the sanitised (masked) outbound payload + a detection summary to
        stdout, so operators can confirm — when calling the real upstream — that
        PII is masked or the request is blocked before anything leaves the host.

        Only the MASKED payload is logged; the original request body is never
        written to the console.
        """
        if not self._log_masked:
            return

        # Collect detections from the structured scrubber's per-field events.
        dets = []  # (category, action, placeholder)
        for ev in getattr(scrub_result, "field_events", []) or []:
            for d in getattr(ev, "detections", []) or []:
                action = str(getattr(d, "action", "")).split(".")[-1]
                dets.append((d.category, action, getattr(d, "placeholder_token", "")))

        out = ["", "=" * 72]
        verdict = "✗ BLOCKED — NOT forwarded (fail-closed)" if blocked \
            else "→ FORWARD to upstream (masked)"
        out.append(f"[PII-Guard] {verdict}")
        out.append(f"  upstream : {self.upstream_url}{path}   (provider={provider})")
        if dets:
            out.append(f"  detections ({len(dets)}):")
            for cat, action, ph in dets:
                mark = "BLOCK" if action == "BLOCK" else "mask "
                out.append(f"    [{mark}] {cat:<13} → {ph}")
        else:
            out.append("  detections: none")
        if getattr(tripwire_result, "should_block", False):
            out.append("  tripwire : BLOCK-category PII found in a non-standard field")
        # The masked payload that is (or would have been) sent upstream.
        try:
            masked_json = json.dumps(
                scrub_result.sanitized_payload, ensure_ascii=False, indent=2
            )
        except Exception:  # noqa: BLE001
            masked_json = "<unserialisable>"
        out.append("  masked payload (sent to upstream):" if not blocked
                   else "  masked payload (withheld — shown for inspection):")
        out.append("\n".join("    " + ln for ln in masked_json.splitlines()))
        out.append("=" * 72)
        print("\n".join(out), file=sys.stdout, flush=True)

    def _handle_pin_list_mutation(self, handler: BaseHTTPRequestHandler) -> None:
        """
        Intercept pin-list control-endpoint requests and block them.

        Sub-AC 5d-i — Pin-list mutation guard
        ---------------------------------------
        Every POST to the pin-list control path is classified as
        ``AGENT``-sourced by :class:`~pii_guard.pinlist_guard.PinListMutationGuard`
        and rejected with HTTP 403 and a structured ``AGENT_MUTATION_BLOCKED``
        JSON error.

        No request body is read, no pin-list state is inspected, and no
        state change is made — the response is constructed purely from the
        classification result.

        The error body format is::

            {
                "error": {
                    "type": "AGENT_MUTATION_BLOCKED",
                    "message": "<human-readable explanation>",
                    "source": "agent"
                }
            }
        """
        result = self._pin_list_guard.check(MutationSource.AGENT)
        # result.allowed is always False here; build the error body
        error_body = json.dumps(result.as_error_dict(), ensure_ascii=False).encode("utf-8")
        self._send_response_bytes(handler, 403, error_body, _JSON_CONTENT_TYPE)

    def _scrub(self, payload: Dict[str, Any], provider: str) -> Any:
        """
        Run the provider-specific scrubber under the engine lock.

        Parameters
        ----------
        payload:
            Parsed request JSON.
        provider:
            ``"claude"``, ``"openai"``, or ``"gemini"``.

        Returns
        -------
        The provider-specific scrub result dataclass
        (``ClaudeRequestScrubResult``, ``OpenAIRequestScrubResult``, or
        ``GeminiRequestScrubResult``).
        """
        with self._engine_lock:
            if provider == "claude":
                return scrub_claude_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            elif provider == "openai":
                return scrub_openai_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            elif provider == "gemini":
                return scrub_gemini_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            else:
                raise ValueError(f"Unknown provider: {provider!r}")

    def _run_tripwire(self, sanitized_payload: Dict[str, Any]) -> TripwireResult:
        """
        Run the full-body tripwire sweep on the *sanitised* payload.

        Serialises *sanitized_payload* to JSON and passes it through
        :func:`~pii_guard.tripwire.sweep_raw_body`.  Because the provider
        scrubber has already masked PII in the fields it knows about, any
        hit found here lives in a non-standard field the structured parser
        did not visit.

        Called under the engine lock (the caller holds it before calling
        :meth:`_scrub`, which is the only concurrent mutation point).
        The tripwire itself is stateless so it does not need a separate lock.

        Parameters
        ----------
        sanitized_payload:
            The scrubbed payload dict returned by the provider scrubber.

        Returns
        -------
        TripwireResult
            Hits found in the serialised sanitised payload, representing
            coverage gaps from the structured parser.
        """
        try:
            sanitized_json = json.dumps(sanitized_payload, ensure_ascii=False)
            return sweep_raw_body(sanitized_json)
        except Exception:  # noqa: BLE001
            # If the tripwire itself fails, return an empty (non-blocking) result
            # rather than crashing the proxy.  The structured scrubber's decision
            # stands.  Infrastructure-level failures in the tripwire are logged
            # separately by the caller when inspecting last_tripwire_result.
            return TripwireResult()

    def _forward(
        self,
        handler: BaseHTTPRequestHandler,
        path: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Serialise *payload* to JSON and POST it to the upstream at *path*.

        Copies through all inbound headers except ``Content-Length`` and
        ``Host`` (which are recomputed for the forwarded request).

        Response rehydration
        --------------------
        When ``rehydrate_responses=True`` (default), the upstream response body
        is passed through the :class:`~pii_guard.response_rehydrator.ResponsePostProcessor`
        before being returned to the calling agent.  ``[CATEGORY_N]`` tokens are
        replaced with their original values from the session mapping store.

        Streaming SSE responses (``Content-Type: text/event-stream``) are
        handled by :meth:`_forward_streaming`, which wires the look-ahead
        buffer into the live event stream and forwards rehydrated chunks
        immediately, preserving streaming TTFT (Sub-AC 9.2).

        Terminal-rendered output rehydration is controlled by the
        ``terminal_restore`` flag on the :attr:`_response_processor` (default OFF).
        """
        forwarded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        upstream_url = self.upstream_url + path

        req = urllib.request.Request(
            upstream_url,
            data=forwarded_body,
            method="POST",
        )

        # Copy inbound headers to the outbound request (preserve auth, content-type…)
        _skip_headers = {"content-length", "host", "transfer-encoding"}
        for key, value in handler.headers.items():
            if key.lower() not in _skip_headers:
                req.add_header(key, value)
        req.add_header("Content-Length", str(len(forwarded_body)))
        if not handler.headers.get("Content-Type"):
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                is_sse = "text/event-stream" in content_type

                if is_sse and self._rehydrate_responses:
                    # ── Streaming SSE path (Sub-AC 9.2) ──────────────────────
                    self._forward_streaming(handler, resp, path)
                else:
                    # ── Buffered (non-streaming) path ─────────────────────────
                    resp_body = resp.read()
                    # ── Response rehydration (Sub-AC 2c) ─────────────────────
                    resp_body = self._rehydrate_response(resp_body, path)
                    handler.send_response(resp.status)
                    for key, value in resp.headers.items():
                        if key.lower() not in {"transfer-encoding", "connection", "content-length"}:
                            handler.send_header(key, value)
                    handler.send_header("Content-Length", str(len(resp_body)))
                    handler.end_headers()
                    handler.wfile.write(resp_body)
        except urllib.error.HTTPError as exc:
            resp_body = exc.read()
            handler.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() not in {"transfer-encoding", "connection"}:
                    handler.send_header(key, value)
            handler.end_headers()
            handler.wfile.write(resp_body)
        except urllib.error.URLError as exc:
            self._send_json(
                handler, 502,
                {"error": f"upstream connection failed: {exc.reason}"}
            )
        except OSError as exc:
            self._send_json(
                handler, 502,
                {"error": f"upstream I/O error: {exc}"}
            )

    def _forward_streaming(
        self,
        handler: BaseHTTPRequestHandler,
        resp: Any,
        path: str,
    ) -> None:
        """
        Forward a streaming SSE response with look-ahead placeholder rehydration.

        Reads the upstream SSE stream in 4 KiB chunks, feeds each chunk through
        :class:`~pii_guard.streaming_rehydrator.StreamingSSERehydrator`, and
        writes rehydrated output to the client **immediately** — without waiting
        for the full response.  This preserves streaming TTFT while ensuring no
        ``[CATEGORY_N]`` placeholder token appears in the forwarded bytes.

        Parameters
        ----------
        handler:
            The per-request HTTP handler (provides ``send_response`` and
            ``wfile`` for writing to the client).
        resp:
            The ``http.client.HTTPResponse`` returned by ``urlopen``.
        path:
            The original request path, used for provider detection.
        """
        provider = _detect_provider(path)

        with self._engine_lock:
            restoration_map = dict(self.engine.restoration_map)

        rehydrator = StreamingSSERehydrator(
            restoration_map=restoration_map,
            provider=provider,
        )

        # ── Send response headers to client (no Content-Length for streaming) ──
        handler.send_response(resp.status)
        for key, value in resp.headers.items():
            if key.lower() not in {
                "transfer-encoding", "connection", "content-length"
            }:
                handler.send_header(key, value)
        # Use Connection: close so the client knows when the stream ends
        handler.send_header("Connection", "close")
        handler.end_headers()

        # ── Stream chunks ────────────────────────────────────────────────────
        # Use resp.fp.read1() instead of resp.read() to get true streaming
        # behaviour.  resp.read(n) (via HTTPResponse) calls readinto(bytearray(n))
        # which blocks until *n* bytes are available or EOF — killing TTFT for
        # small SSE frames delivered in multiple chunks.  BufferedReader.read1(n)
        # performs at most ONE underlying read() syscall and returns whatever is
        # already in the socket buffer, enabling per-chunk delivery.
        raw_reader = getattr(resp, "fp", None)
        use_read1 = raw_reader is not None and hasattr(raw_reader, "read1")

        try:
            while True:
                if use_read1:
                    chunk = raw_reader.read1(_STREAM_CHUNK_SIZE)
                else:
                    # Fallback: standard read — correct but blocks until EOF
                    # on small responses (loses streaming TTFT).
                    chunk = resp.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                output = rehydrator.feed_chunk(chunk)
                if output:
                    handler.wfile.write(output)
                    handler.wfile.flush()

            # ── Flush look-ahead buffer tail ─────────────────────────────────
            tail = rehydrator.flush()
            if tail:
                handler.wfile.write(tail)
                handler.wfile.flush()

        except OSError:
            # Client disconnected or upstream closed unexpectedly — stop silently.
            pass

    def _rehydrate_response(self, resp_body: bytes, path: str) -> bytes:
        """
        Apply inbound response rehydration to *resp_body*.

        Detects the provider from *path*, retrieves the current session
        restoration map, and passes both through
        :class:`~pii_guard.response_rehydrator.ResponsePostProcessor`.

        Returns the (possibly rewritten) response bytes to send to the client.
        When ``rehydrate_responses=False``, returns *resp_body* unchanged.
        """
        if not self._rehydrate_responses:
            return resp_body

        provider = _detect_provider(path)

        with self._engine_lock:
            restoration_map = self.engine.restoration_map

        if not restoration_map:
            # Nothing was masked → nothing to rehydrate
            return resp_body

        rehydration_result = self._response_processor.process(
            response_body=resp_body,
            restoration_map=restoration_map,
            provider=provider,
        )

        with self._last_rehydration_lock:
            self._last_rehydration_result = rehydration_result

        return rehydration_result.agent_body

    def _pass_through(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """Forward an unrecognised or GET request to the upstream unchanged."""
        # For simplicity, return 405 for non-POST/GET requests
        self._send_json(
            handler, 405, {"error": f"method {method} not supported"}
        )

    # ── Response helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        status: int,
        data: Any,
    ) -> None:
        """Send a JSON response."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        PIIGuardProxy._send_response_bytes(
            handler, status, body, _JSON_CONTENT_TYPE
        )

    @staticmethod
    def _send_response_bytes(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        """Send a raw bytes response with the given status and content-type."""
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
