"""
Sub-AC 3c: Proxy process crash fail-closed.

Integration tests that verify the proxy's network-layer behaviour when the
proxy process is killed (SIGKILL) while a client request is in-flight.

Fail-closed guarantee
---------------------
When the proxy process is killed:

1. In-flight connections:
   The OS immediately closes all TCP sockets with TCP RST, discarding any
   data in kernel send buffers.  Clients receive a connection-reset error
   (ConnectionResetError / urllib.error.URLError), never a forwarded upstream
   response.

2. Subsequent connections:
   The proxy's listening socket is closed by the OS.  New connection attempts
   to the same port fail with ConnectionRefusedError.

This test spawns the proxy as a real subprocess (via ``piiguard serve``), uses
a local mock upstream that deliberately delays responses so the proxy is still
in-flight when killed, then asserts connection-error semantics.

Test matrix
-----------
* test_sigkill_inflight_request_gets_connection_error
    Client's in-flight HTTP POST to proxy raises a connection error after SIGKILL.
    The client must NOT receive any upstream response body.

* test_sigkill_subsequent_connections_get_refused
    After the proxy process is dead, new TCP connection attempts to the proxy
    port fail with a connection-refused / OS error rather than succeeding.

* test_sigkill_no_buffered_response_delivered
    Even if the proxy held a partial response in a kernel send-buffer, SIGKILL
    discards it: the client reads zero bytes of upstream response body.

* test_sigterm_closes_connections
    SIGTERM is handled by the proxy's signal handler; connections are also
    closed without completing in-flight requests.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

#: Root of the pii_guard project (one level above this tests/ directory).
_PROJECT_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Mock slow upstream helpers
# ─────────────────────────────────────────────────────────────────────────────

class _SlowUpstreamServer:
    """
    A local HTTP server that deliberately delays POST responses.

    Used to hold the proxy's upstream connection open long enough for the
    integration test to kill the proxy while the request is in-flight.

    Parameters
    ----------
    delay:
        Seconds to sleep before writing the response (default: 15).
    request_received:
        Optional :class:`threading.Event` set as soon as the first POST is
        received, before the delay.  Used to synchronise the test: once this
        event fires, the proxy has already forwarded the client's request to
        this server, so killing the proxy will interrupt an in-flight round-trip.
    """

    def __init__(
        self,
        delay: float = 15.0,
        request_received: Optional[threading.Event] = None,
    ) -> None:
        _delay = delay
        _event = request_received

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                pass  # keep test output clean

            def do_POST(self) -> None:
                # Signal that the proxy's forwarded request arrived
                if _event is not None:
                    _event.set()
                # Hold the connection open to keep the proxy in-flight
                time.sleep(_delay)
                body = json.dumps({"ok": True}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Proxy was killed; the connection was RST'd.  Swallow.
                    pass

        class _QuietHTTPServer(HTTPServer):
            """Suppress handle_error tracebacks caused by proxy crash RSTs."""

            def handle_error(self, request, client_address) -> None:
                pass  # silence "connection reset by peer" tracebacks in tests

        self._server = _QuietHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self) -> None:
        self._server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess proxy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_proxy(upstream_url: str, startup_timeout: float = 15.0):
    """
    Spawn the PII-Guard proxy as a child process via ``python -m pii_guard serve``.

    The subprocess writes ``READY <port>\\n`` to stdout once it is accepting
    connections; this function reads that line and returns ``(proc, port)``.

    The ``PYTHONPATH`` environment variable is set to ``_PROJECT_ROOT`` so the
    subprocess can import ``pii_guard`` regardless of whether the package is
    installed in the current environment.

    Parameters
    ----------
    upstream_url:
        Full URL of the upstream the proxy should forward to.
    startup_timeout:
        Maximum seconds to wait for the "READY" signal before giving up.

    Returns
    -------
    tuple[subprocess.Popen, int]
        The running proxy process and the port it is listening on.

    Raises
    ------
    RuntimeError
        If the subprocess does not emit a valid "READY <port>" line within
        *startup_timeout* seconds.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "pii_guard",
            "serve",
            "--upstream-url", upstream_url,
            "--port", "0",          # OS-assigned ephemeral port
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Read "READY {port}\n" from the subprocess stdout in a thread so we can
    # apply a timeout without blocking the test runner indefinitely.
    port_holder: list = [None]
    error_holder: list = [None]
    ready_event = threading.Event()

    def _reader() -> None:
        try:
            line = proc.stdout.readline().decode().strip()
            if line.startswith("READY "):
                port_holder[0] = int(line.split()[1])
            else:
                error_holder[0] = (
                    f"Proxy subprocess emitted unexpected startup line: {line!r}"
                )
        except Exception as exc:
            error_holder[0] = f"Reading proxy stdout failed: {exc}"
        finally:
            ready_event.set()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    reader.join(timeout=startup_timeout)

    if port_holder[0] is None:
        # Timeout or error — kill subprocess before raising
        try:
            proc.kill()
            proc.wait(timeout=3)
        except OSError:
            pass
        stderr_output = ""
        try:
            proc.stderr.close()
        except OSError:
            pass
        raise RuntimeError(
            error_holder[0]
            or f"Proxy subprocess did not emit 'READY' within {startup_timeout}s.\n"
               f"stderr: {stderr_output}"
        )

    return proc, port_holder[0]


def _kill_proxy(proc: subprocess.Popen, wait_timeout: float = 5.0) -> None:
    """Send SIGKILL to *proc* and wait for it to terminate."""
    try:
        proc.kill()  # SIGKILL — immediate, unblockable
    except (ProcessLookupError, OSError):
        pass  # already dead
    try:
        proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        pass  # shouldn't happen after SIGKILL


def _send_proxy_request(proxy_port: int):
    """
    POST a minimal Claude-format request to the proxy.

    Returns the response body bytes on success.
    Raises :class:`urllib.error.URLError` or :class:`OSError` on failure.
    """
    payload = json.dumps({
        "model": "claude-opus-4-5",
        "messages": [{"role": "user", "content": "hello from integration test"}],
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyCrashFailClosed:
    """
    Network-layer fail-closed behaviour when the proxy process is killed.

    Each test spawns a real subprocess proxy and a local slow-upstream mock,
    asserts the desired failure semantics, then cleans up.
    """

    # ── 1. In-flight request → connection error ───────────────────────────────

    def test_sigkill_inflight_request_gets_connection_error(self):
        """
        Killing the proxy with SIGKILL while a client POST is in-flight must
        cause the client to receive a connection-error exception, not a
        successful HTTP response from the upstream.

        Steps
        -----
        1. Start a slow upstream (15 s delay).
        2. Spawn proxy subprocess.
        3. Send client request in a background thread.
        4. Wait until the upstream confirms it received the forwarded request
           (proxy has scrubbed + forwarded — it is now waiting on the upstream).
        5. Kill the proxy process (SIGKILL).
        6. Assert the client thread raised a connection error and did NOT
           receive any response body.
        """
        request_received = threading.Event()
        upstream = _SlowUpstreamServer(delay=15.0, request_received=request_received)
        proc: Optional[subprocess.Popen] = None

        try:
            proc, proxy_port = _spawn_proxy(upstream.url)

            # Prepare to collect the client outcome
            response_body_holder: list = []
            exception_holder: list = []
            client_done = threading.Event()

            def _client() -> None:
                try:
                    body = _send_proxy_request(proxy_port)
                    response_body_holder.append(body)
                except Exception as exc:
                    exception_holder.append(exc)
                finally:
                    client_done.set()

            client_thread = threading.Thread(target=_client, daemon=True)
            client_thread.start()

            # Wait until the upstream has received the forwarded request.
            # At this point the proxy is blocked waiting for the upstream response.
            received = request_received.wait(timeout=12.0)
            assert received, (
                "Upstream never received the forwarded request — the proxy may "
                "have failed to start or crashed before forwarding."
            )

            # Kill the proxy process
            _kill_proxy(proc)
            proc = None  # prevent double-kill in finally

            # The client must get a connection error quickly
            finished = client_done.wait(timeout=10.0)
            assert finished, "Client thread did not complete within timeout after proxy kill"

            # Primary assertion: the client must NOT have received a response
            assert not response_body_holder, (
                "FAIL-CLOSED VIOLATION: client received a complete upstream response "
                "after the proxy was killed with SIGKILL.  The proxy must NOT "
                "pass through upstream responses after a process crash."
            )

            # Secondary assertion: the client must have gotten a connection error
            assert exception_holder, (
                "Client did not raise any exception after proxy was killed — "
                "expected ConnectionResetError or URLError."
            )

            exc = exception_holder[0]
            assert isinstance(exc, (
                urllib.error.URLError,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                OSError,
            )), (
                f"Expected a network-layer connection error, got: {type(exc).__name__}: {exc}"
            )

        finally:
            if proc is not None:
                _kill_proxy(proc)
            upstream.shutdown()

    # ── 2. Subsequent connections refused after crash ─────────────────────────

    def test_sigkill_subsequent_connections_get_refused(self):
        """
        After the proxy process is killed, the OS frees the listening socket.
        New TCP connection attempts to the same port must fail with
        ConnectionRefusedError (or a similar OS error), not succeed.
        """
        upstream = _SlowUpstreamServer(delay=0.0)
        proc: Optional[subprocess.Popen] = None

        try:
            proc, proxy_port = _spawn_proxy(upstream.url)

            # Verify proxy is alive (health check)
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{proxy_port}/health", timeout=5
                )
            except urllib.error.HTTPError:
                pass  # 4xx is fine — proxy is alive
            # Any other error means proxy is not ready; propagate

            # Kill the proxy
            _kill_proxy(proc)
            proc = None

            # Give the OS a brief moment to reclaim the port
            time.sleep(0.25)

            # Attempt a raw TCP connection to the dead proxy's port
            with pytest.raises((ConnectionRefusedError, OSError)) as exc_info:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3.0)
                try:
                    s.connect(("127.0.0.1", proxy_port))
                    # If connect surprisingly succeeded, close and fail the test
                    s.close()
                    pytest.fail(
                        f"TCP connection to killed proxy on port {proxy_port} "
                        "succeeded — should have been refused."
                    )
                finally:
                    try:
                        s.close()
                    except OSError:
                        pass

            # Either ConnectionRefusedError or a generic OSError (e.g. ECONNRESET)
            # counts — the key invariant is that the connection was not accepted.
            assert exc_info.value is not None

        finally:
            if proc is not None:
                _kill_proxy(proc)
            upstream.shutdown()

    # ── 3. No buffered upstream response delivered ────────────────────────────

    def test_sigkill_no_upstream_response_body_delivered(self):
        """
        Even if the upstream had written a response into the kernel's send
        buffer before the proxy was killed, SIGKILL discards unsent data by
        issuing TCP RST.  The client must read zero bytes of upstream response.

        This test uses a two-phase upstream: it sends the response header
        immediately but pauses before the body.  We kill the proxy between the
        header flush and body write.  The client must get an error and must NOT
        receive any parseable response body.
        """
        # Phase-1: upstream signals header sent; phase-2: body after delay.
        header_sent = threading.Event()
        delay_done = threading.Event()

        class _TwoPhaseHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.dumps({"id": "msg_pii_guard_test"}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    # Flush headers — proxy may buffer this, but we signal here
                    self.wfile.flush()
                except (OSError, BrokenPipeError):
                    return
                header_sent.set()
                # Hold before body — proxy should be killed in this window
                delay_done.wait(timeout=20.0)
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class _QuietHTTPServer(HTTPServer):
            def handle_error(self, request, client_address) -> None:
                pass

        server = _QuietHTTPServer(("127.0.0.1", 0), _TwoPhaseHandler)
        upstream_thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        upstream_thread.start()
        upstream_url = f"http://127.0.0.1:{server.server_address[1]}"

        proc: Optional[subprocess.Popen] = None

        try:
            proc, proxy_port = _spawn_proxy(upstream_url)

            response_body_holder: list = []
            exception_holder: list = []
            client_done = threading.Event()

            def _client() -> None:
                try:
                    body = _send_proxy_request(proxy_port)
                    response_body_holder.append(body)
                except Exception as exc:
                    exception_holder.append(exc)
                finally:
                    client_done.set()

            client_thread = threading.Thread(target=_client, daemon=True)
            client_thread.start()

            # Wait until upstream has flushed response headers (proxy is now
            # waiting for the body before it can forward the complete response)
            received = header_sent.wait(timeout=12.0)
            assert received, "Two-phase upstream never signalled header sent"

            # Kill the proxy; release the upstream delay so its thread can exit
            _kill_proxy(proc)
            proc = None
            delay_done.set()

            # Client must get an error
            finished = client_done.wait(timeout=10.0)
            assert finished, "Client thread did not complete after proxy kill"

            assert not response_body_holder, (
                "FAIL-CLOSED VIOLATION: client received a response body after "
                "proxy was killed mid-response stream."
            )
            assert exception_holder, (
                "Client must raise a network error when proxy is killed "
                "before body is delivered."
            )

        finally:
            delay_done.set()  # unblock upstream thread in case of failure
            if proc is not None:
                _kill_proxy(proc)
            server.shutdown()

    # ── 4. SIGTERM also closes connections ────────────────────────────────────

    def test_sigterm_closes_inflight_connection(self):
        """
        SIGTERM triggers the proxy's graceful-shutdown signal handler.
        In-flight connections must be closed without completing the upstream
        round-trip — the client should receive a connection error.

        Note: Python's HTTPServer.shutdown() waits for the current request to
        finish serving.  Since the upstream delays 15 s and SIGTERM is sent
        almost immediately, the request cannot complete: the proxy process exits
        (via sys.exit in the signal handler) before the upstream responds.
        The OS then RSTs the proxy→client connection.
        """
        request_received = threading.Event()
        upstream = _SlowUpstreamServer(delay=15.0, request_received=request_received)
        proc: Optional[subprocess.Popen] = None

        try:
            proc, proxy_port = _spawn_proxy(upstream.url)

            response_body_holder: list = []
            exception_holder: list = []
            client_done = threading.Event()

            def _client() -> None:
                try:
                    body = _send_proxy_request(proxy_port)
                    response_body_holder.append(body)
                except Exception as exc:
                    exception_holder.append(exc)
                finally:
                    client_done.set()

            client_thread = threading.Thread(target=_client, daemon=True)
            client_thread.start()

            # Wait until upstream received the forwarded request
            received = request_received.wait(timeout=12.0)
            assert received, (
                "Upstream never received forwarded request for SIGTERM test"
            )

            # Send SIGTERM to the proxy process (graceful shutdown signal)
            try:
                proc.send_signal(signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            proc.wait(timeout=10.0)
            proc = None

            # Client must get a connection error
            finished = client_done.wait(timeout=10.0)
            assert finished, "Client thread did not complete after SIGTERM"

            assert not response_body_holder, (
                "FAIL-CLOSED VIOLATION: client received a response body after "
                "proxy received SIGTERM."
            )
            assert exception_holder, (
                "Client must raise a network error when proxy is killed via SIGTERM "
                "before the upstream round-trip completes."
            )

        finally:
            if proc is not None:
                _kill_proxy(proc)
            upstream.shutdown()

    # ── 5. Control: live proxy forwards clean request normally ────────────────

    def test_live_proxy_forwards_request_successfully(self):
        """
        Control test: when the proxy is alive and the upstream responds quickly,
        the client receives a successful response.  This verifies the test
        infrastructure itself is working correctly.
        """
        upstream = _SlowUpstreamServer(delay=0.0)  # respond immediately
        proc: Optional[subprocess.Popen] = None

        try:
            proc, proxy_port = _spawn_proxy(upstream.url)

            response_body = _send_proxy_request(proxy_port)

            assert response_body, "Expected a non-empty response body"
            parsed = json.loads(response_body)
            assert parsed.get("ok") is True, (
                f"Unexpected upstream response: {parsed}"
            )

        finally:
            if proc is not None:
                _kill_proxy(proc)
            upstream.shutdown()
