"""
PII-Guard egress-lockdown integration tests — Sub-AC 6b-ii.

Two test tiers are defined in this file:

TestCLIUnit  (no sudo, runs in any environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests that verify the CLI argument parser, output formatting, and
exit-code behaviour.  All pfctl subprocess calls are intercepted with
``unittest.mock``.  These run anywhere, without root.

TestEgressLockdownIntegration  (requires root + real network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live pf(4) integration tests that:

1.  Verify baseline TCP connectivity to an LLM provider endpoint exists.
2.  Call ``PfManager.enable()`` — loads the piiguard pf anchor via
    ``sudo pfctl -a piiguard -f -``.
3.  Assert that a direct TCP connection attempt to the provider endpoint is
    now **refused / times out** (pf drops SYN packets for blocked IPs).
4.  Call ``PfManager.disable()`` — flushes the anchor.
5.  Assert that TCP connectivity is **restored**.

A parallel test class (``TestCLIEgressIntegration``) exercises the same
lifecycle through the ``piiguard egress enable / disable`` CLI commands.

Running
-------
Unit tests only (no root needed)::

    pytest tests/test_egress_lockdown_integration.py -v -m "not integration"

Full integration suite (root + network required)::

    sudo pytest tests/test_egress_lockdown_integration.py -v -m integration

Or run everything and let integration tests auto-skip when not root::

    pytest tests/test_egress_lockdown_integration.py -v
"""
from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, call, patch

import pytest

# Repository root (parent of tests/) — used as cwd for subprocess CLI invocations
# so tests are location-independent rather than tied to an absolute install path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pii_guard.cli import (
    build_parser,
    cmd_egress_disable,
    cmd_egress_enable,
    cmd_egress_status,
    main,
)
from pii_guard.pf_manager import (
    ALL_PROVIDER_IP_RANGES,
    ANCHOR_TABLE_NAME,
    ANTHROPIC_IP_RANGES,
    DEFAULT_ANCHOR_NAME,
    PfManager,
    PfRuleError,
    collect_all_cidrs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ok_proc(stdout: str = "") -> subprocess.CompletedProcess:
    """Return a CompletedProcess with returncode=0."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _make_fail_proc(returncode: int = 1, stderr: str = "pfctl error") -> subprocess.CompletedProcess:
    """Return a CompletedProcess with a non-zero returncode."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def _try_connect_tcp(
    host: str,
    port: int,
    timeout: float,
    via_ip: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Attempt a TCP connection to *host*:*port* with the given *timeout* (seconds).

    Parameters
    ----------
    host:
        Hostname or IP address to connect to.
    port:
        TCP destination port.
    timeout:
        Socket timeout in seconds.  When pf blocks with ``block out quick``
        (silent drop), the SYN packet is discarded and the socket times out
        after *timeout* seconds.
    via_ip:
        If provided, connect to this pre-resolved IP (bypass DNS).

    Returns
    -------
    (success, error_description)
        ``(True, "")`` on TCP handshake success.
        ``(False, "<reason>")`` on timeout, refusal, or OS error.
    """
    target = via_ip if via_ip else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True, ""
    except socket.timeout:
        return False, f"timeout after {timeout}s (packet likely dropped by pf)"
    except ConnectionRefusedError:
        return False, "connection refused (RST)"
    except OSError as exc:
        return False, f"OSError({exc.errno}): {exc.strerror}"


def _resolve_to_ip(hostname: str) -> Optional[str]:
    """
    Resolve a hostname to its first IPv4 address.  Returns None on failure.
    """
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        return None


def _ip_in_cidr_list(ip_str: str, cidrs: List[str]) -> bool:
    """Return True if *ip_str* falls within any CIDR in *cidrs*."""
    try:
        ip = ipaddress.ip_address(ip_str)
        for cidr in cidrs:
            if ip in ipaddress.ip_network(cidr, strict=False):
                return True
    except ValueError:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Constants for integration tests
# ─────────────────────────────────────────────────────────────────────────────

#: Primary LLM provider endpoint used for connectivity tests.
_LLM_HOST = "api.anthropic.com"
_LLM_PORT = 443

#: Generous timeout for baseline/restoration checks (normal connection ≤ ~1s).
_TIMEOUT_ALLOWED = 10.0

#: Short timeout for blocked-connection checks.
#: When pf drops SYN packets, socket.timeout fires after this many seconds.
#: Must be long enough to guarantee pf had a chance to drop — 4s is safe.
_TIMEOUT_BLOCKED = 4.0

#: All CIDRs that the pf anchor will block (all providers combined).
_ALL_BLOCKED_CIDRS: List[str] = collect_all_cidrs(ALL_PROVIDER_IP_RANGES)


# ─────────────────────────────────────────────────────────────────────────────
# pytest markers
# ─────────────────────────────────────────────────────────────────────────────

# Mark integration tests so they can be selected / excluded:
#   pytest -m integration          → only integration tests
#   pytest -m "not integration"    → skip integration tests
pytestmark_integration = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────────────
# TestCLIUnit — mocked pfctl, no sudo
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIArgumentParsing:
    """Verify the CLI argument parser structure and help text."""

    def test_parser_requires_subcommand(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_egress_enable(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable"])
        assert args.command == "egress"
        assert args.egress_command == "enable"

    def test_parser_accepts_egress_disable(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "disable"])
        assert args.command == "egress"
        assert args.egress_command == "disable"

    def test_parser_accepts_egress_status(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "status"])
        assert args.command == "egress"
        assert args.egress_command == "status"

    def test_egress_enable_has_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable", "--verbose"])
        assert args.verbose is True

    def test_egress_enable_verbose_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable"])
        assert args.verbose is False

    def test_egress_enable_accepts_anchor_name(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable", "--anchor-name", "myanchor"])
        assert args.anchor_name == "myanchor"

    def test_egress_disable_accepts_anchor_name(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "disable", "--anchor-name", "myanchor"])
        assert args.anchor_name == "myanchor"

    def test_egress_status_accepts_anchor_name(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "status", "--anchor-name", "myanchor"])
        assert args.anchor_name == "myanchor"

    def test_egress_enable_anchor_name_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable"])
        assert args.anchor_name is None

    def test_version_flag(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_enable_func_is_cmd_egress_enable(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "enable"])
        assert args.func is cmd_egress_enable

    def test_disable_func_is_cmd_egress_disable(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "disable"])
        assert args.func is cmd_egress_disable

    def test_status_func_is_cmd_egress_status(self):
        parser = build_parser()
        args = parser.parse_args(["egress", "status"])
        assert args.func is cmd_egress_status


class TestCLIEgressEnableUnit:
    """Unit tests for cmd_egress_enable() with mocked PfManager."""

    def _run_enable(self, argv=None, env=None, mock_mgr=None):
        """Run main() with given argv, optionally patching PfManager."""
        argv = argv or ["egress", "enable"]
        mock_mgr = mock_mgr or MagicMock(spec=PfManager)
        mock_mgr.status.return_value = "block out quick proto tcp to <piiguard_llm_ips> port { 80 443 }"
        with patch("pii_guard.cli.PfManager", return_value=mock_mgr):
            rc = main(argv)
        return rc, mock_mgr

    def test_enable_returns_exit_code_0_on_success(self):
        rc, _ = self._run_enable()
        assert rc == 0

    def test_enable_calls_pf_manager_enable(self):
        rc, mock_mgr = self._run_enable()
        mock_mgr.enable.assert_called_once()

    def test_enable_with_custom_anchor_name(self):
        rc, mock_mgr = self._run_enable(["egress", "enable", "--anchor-name", "myanchor"])
        # PfManager should be instantiated with the custom anchor name
        assert rc == 0
        mock_mgr.enable.assert_called_once()

    def test_enable_pf_rule_error_returns_exit_code_1(self):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.enable.side_effect = PfRuleError("pfctl failed: permission denied")
        rc, _ = self._run_enable(mock_mgr=mock_mgr)
        assert rc == 1

    def test_enable_pf_not_found_returns_exit_code_3(self):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.enable.side_effect = PfRuleError("pfctl not found at /sbin/pfctl")
        rc, _ = self._run_enable(mock_mgr=mock_mgr)
        assert rc == 3

    def test_enable_value_error_returns_exit_code_2(self):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.enable.side_effect = ValueError("ip_ranges produced no CIDRs")
        rc, _ = self._run_enable(mock_mgr=mock_mgr)
        assert rc == 2

    def test_enable_verbose_calls_status(self):
        rc, mock_mgr = self._run_enable(["egress", "enable", "--verbose"])
        assert rc == 0
        mock_mgr.status.assert_called_once()

    def test_enable_non_verbose_does_not_call_status(self):
        rc, mock_mgr = self._run_enable(["egress", "enable"])
        assert rc == 0
        mock_mgr.status.assert_not_called()

    def test_enable_prints_enabled_message(self, capsys):
        self._run_enable()
        out = capsys.readouterr().out
        assert "enabled" in out.lower()

    def test_enable_error_prints_to_stderr(self, capsys):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.enable.side_effect = PfRuleError("pfctl failed: permission denied")
        self._run_enable(mock_mgr=mock_mgr)
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_enable_hints_about_sudo_on_permission_error(self, capsys):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.enable.side_effect = PfRuleError("permission denied")
        self._run_enable(mock_mgr=mock_mgr)
        err = capsys.readouterr().err
        assert "sudo" in err.lower()

    def test_enable_anchor_name_env_var_used(self, monkeypatch):
        """PIIGUARD_PF_ANCHOR env var should be forwarded to PfManager as anchor_name."""
        monkeypatch.setenv("PIIGUARD_PF_ANCHOR", "env_anchor")
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.status.return_value = ""
        # Patch PfManager class so we can capture constructor keyword args
        with patch("pii_guard.cli.PfManager", return_value=mock_mgr) as mock_cls:
            rc = main(["egress", "enable"])
        assert rc == 0
        # The CLI resolves env var and passes it as anchor_name to PfManager
        _, kwargs = mock_cls.call_args
        assert kwargs.get("anchor_name") == "env_anchor"


class TestCLIEgressDisableUnit:
    """Unit tests for cmd_egress_disable() with mocked PfManager."""

    def _run_disable(self, argv=None, mock_mgr=None):
        argv = argv or ["egress", "disable"]
        mock_mgr = mock_mgr or MagicMock(spec=PfManager)
        with patch("pii_guard.cli.PfManager", return_value=mock_mgr):
            rc = main(argv)
        return rc, mock_mgr

    def test_disable_returns_exit_code_0_on_success(self):
        rc, _ = self._run_disable()
        assert rc == 0

    def test_disable_calls_pf_manager_disable(self):
        rc, mock_mgr = self._run_disable()
        mock_mgr.disable.assert_called_once()

    def test_disable_pf_rule_error_returns_exit_code_1(self):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.disable.side_effect = PfRuleError("pfctl flush failed")
        rc, _ = self._run_disable(mock_mgr=mock_mgr)
        assert rc == 1

    def test_disable_prints_disabled_message(self, capsys):
        self._run_disable()
        out = capsys.readouterr().out
        assert "disabled" in out.lower()

    def test_disable_with_custom_anchor_name(self):
        rc, mock_mgr = self._run_disable(["egress", "disable", "--anchor-name", "test_a"])
        assert rc == 0
        mock_mgr.disable.assert_called_once()


class TestCLIEgressStatusUnit:
    """Unit tests for cmd_egress_status() with mocked PfManager."""

    def _run_status(self, argv=None, status_return=None):
        argv = argv or ["egress", "status"]
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.status.return_value = status_return
        with patch("pii_guard.cli.PfManager", return_value=mock_mgr):
            rc = main(argv)
        return rc, mock_mgr

    def test_status_returns_exit_code_0_when_rules_present(self):
        rc, _ = self._run_status(status_return="block out quick proto tcp to <piiguard_llm_ips>")
        assert rc == 0

    def test_status_returns_exit_code_0_when_empty(self):
        rc, _ = self._run_status(status_return="")
        assert rc == 0

    def test_status_returns_exit_code_1_when_none(self):
        rc, _ = self._run_status(status_return=None)
        assert rc == 1

    def test_status_calls_pf_manager_status(self):
        rc, mock_mgr = self._run_status(status_return="rules")
        mock_mgr.status.assert_called_once()

    def test_status_prints_rules_when_present(self, capsys):
        self._run_status(status_return="block out quick proto tcp to <piiguard_llm_ips>")
        out = capsys.readouterr().out
        assert "block out" in out

    def test_status_prints_not_active_when_empty(self, capsys):
        self._run_status(status_return="")
        out = capsys.readouterr().out
        assert "not active" in out.lower()

    def test_status_prints_active_when_rules_present(self, capsys):
        self._run_status(status_return="block out quick")
        out = capsys.readouterr().out
        assert "active" in out.lower()

    def test_status_warns_on_none(self, capsys):
        self._run_status(status_return=None)
        err = capsys.readouterr().err
        assert "warning" in err.lower() or "could not" in err.lower()


class TestCLIMainDispatch:
    """Tests for the main() entry point dispatch logic."""

    def test_main_returns_int(self):
        mock_mgr = MagicMock(spec=PfManager)
        mock_mgr.status.return_value = "rules"
        with patch("pii_guard.cli.PfManager", return_value=mock_mgr):
            rc = main(["egress", "enable"])
        assert isinstance(rc, int)

    def test_main_with_no_args_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_main_with_unknown_command_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["unknown_cmd"])
        assert exc_info.value.code != 0

    def test_module_runnable_as_main(self):
        """python3 -m pii_guard.cli --help should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        assert "piiguard" in result.stdout.lower() or "pii-guard" in result.stdout.lower()

    def test_cli_help_mentions_egress(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert "egress" in result.stdout

    def test_egress_enable_help_mentions_sudo(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "egress", "enable", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        assert "sudo" in result.stdout.lower() or "root" in result.stdout.lower()

    def test_egress_disable_help_is_displayable(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "egress", "disable", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0

    def test_egress_status_help_is_displayable(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "egress", "status", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity helper tests (always run — no sudo)
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectivityHelpers:
    """Unit tests for the shared _try_connect_tcp and CIDR helpers."""

    def test_ip_in_cidr_list_match(self):
        cidrs = ["104.18.0.0/16"]
        assert _ip_in_cidr_list("104.18.5.10", cidrs) is True

    def test_ip_in_cidr_list_no_match(self):
        cidrs = ["104.18.0.0/16"]
        assert _ip_in_cidr_list("8.8.8.8", cidrs) is False

    def test_ip_in_cidr_list_multiple_cidrs(self):
        cidrs = ["104.18.0.0/16", "172.64.0.0/13"]
        assert _ip_in_cidr_list("172.64.10.1", cidrs) is True
        assert _ip_in_cidr_list("1.2.3.4", cidrs) is False

    def test_ip_in_cidr_list_invalid_ip_returns_false(self):
        assert _ip_in_cidr_list("not_an_ip", ["10.0.0.0/8"]) is False

    def test_all_blocked_cidrs_non_empty(self):
        assert len(_ALL_BLOCKED_CIDRS) > 0

    def test_all_blocked_cidrs_contains_anthropic(self):
        # At least one Anthropic CIDR should be in the combined list
        assert any(c in _ALL_BLOCKED_CIDRS for c in ANTHROPIC_IP_RANGES)

    def test_try_connect_tcp_loopback_no_listener(self):
        """Connection to a closed loopback port should fail."""
        ok, err = _try_connect_tcp("127.0.0.1", 19999, timeout=1.0)
        # Either connection refused (RST from loopback) or timeout
        assert not ok
        assert err  # error description is non-empty

    def test_try_connect_tcp_loopback_open_port(self):
        """Connection to a listening loopback server should succeed."""
        import threading
        import time
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _accept():
            try:
                conn, _ = server.accept()
                conn.close()
            except OSError:
                pass
            finally:
                server.close()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()
        time.sleep(0.05)  # give the server a moment to be ready

        ok, err = _try_connect_tcp("127.0.0.1", port, timeout=2.0)
        t.join(timeout=2.0)
        assert ok, f"Expected loopback connection to succeed, got: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# TestEgressLockdownIntegration — real pf + network (requires root)
# ─────────────────────────────────────────────────────────────────────────────

def _require_root():
    """Skip the calling test if not running as root."""
    if os.getuid() != 0:
        pytest.skip(
            "Egress-lockdown integration tests require root. "
            "Run with: sudo pytest tests/test_egress_lockdown_integration.py -v -m integration"
        )


def _require_pfctl():
    """Skip the calling test if pfctl is not available."""
    pfctl_path = os.environ.get("PIIGUARD_PFCTL", "/sbin/pfctl")
    if not os.path.exists(pfctl_path):
        pytest.skip(
            f"pfctl not found at {pfctl_path!r} — "
            "this test requires macOS with pf(4)"
        )


def _require_network(host: str, port: int, timeout: float) -> str:
    """
    Skip the calling test if *host*:*port* is not reachable.

    Returns the resolved IP address on success.
    """
    ip = _resolve_to_ip(host)
    if ip is None:
        pytest.skip(
            f"DNS resolution of {host!r} failed — "
            "integration test requires real network access"
        )
    ok, err = _try_connect_tcp(host, port, timeout=timeout, via_ip=ip)
    if not ok:
        pytest.skip(
            f"No baseline connectivity to {host}:{port} (IP {ip}) — "
            f"got: {err}. Integration test requires real network access."
        )
    return ip


@pytest.mark.integration
class TestEgressLockdownIntegration:
    """
    Live pf(4) egress-lockdown integration tests.

    Requirements
    ------------
    - macOS with pf(4) available at /sbin/pfctl
    - Root privileges (run with: sudo pytest)
    - Real network access to api.anthropic.com:443

    What is tested
    --------------
    For each test:
    1.  Pre-flight: verify baseline TCP connectivity to api.anthropic.com:443
        exists (skip if not).
    2.  ``PfManager.enable()`` loads the piiguard anchor via pfctl.
    3.  A direct TCP SYN to the same endpoint is dropped by pf → socket.timeout.
    4.  ``PfManager.disable()`` flushes the anchor.
    5.  TCP connectivity is restored.
    """

    # ── pf block semantics note ────────────────────────────────────────────────
    # ``block out quick`` drops the SYN packet silently.  The socket retries
    # until socket.settimeout fires — typically within _TIMEOUT_BLOCKED seconds.
    # ``block return out quick`` would send a TCP RST (immediate ECONNREFUSED),
    # but we use the default block-drop so we test the timeout path.

    def _setup_checks(self):
        """Run root / pfctl / network pre-flight and return resolved IP."""
        _require_root()
        _require_pfctl()
        return _require_network(_LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED)

    def test_enable_blocks_direct_tcp_to_anthropic(self):
        """
        enable() → direct TCP to api.anthropic.com blocked → disable() → restored.
        """
        resolved_ip = self._setup_checks()

        mgr = PfManager()
        try:
            # Step 2: Enable lockdown
            mgr.enable()
            assert mgr.is_enabled, "is_enabled must be True after enable()"

            # Step 3: Direct TCP to the LLM endpoint must now be blocked
            ok_blocked, err_blocked = _try_connect_tcp(
                _LLM_HOST, _LLM_PORT, _TIMEOUT_BLOCKED, via_ip=resolved_ip
            )
            assert not ok_blocked, (
                f"Expected TCP connection to {_LLM_HOST}:{_LLM_PORT} "
                f"(IP {resolved_ip}) to be BLOCKED by pf after enable(), "
                f"but it SUCCEEDED.\n"
                f"This means the resolved IP ({resolved_ip}) may be outside "
                f"the blocked CIDR ranges. Check ALL_PROVIDER_IP_RANGES."
            )

        finally:
            # Step 4: Always disable — even if assertions above fail
            mgr.disable()

        # Step 5: Connectivity restored
        assert not mgr.is_enabled, "is_enabled must be False after disable()"
        ok_after, err_after = _try_connect_tcp(
            _LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED, via_ip=resolved_ip
        )
        assert ok_after, (
            f"Expected TCP connectivity to {_LLM_HOST}:{_LLM_PORT} "
            f"to be RESTORED after disable(), but got: {err_after}"
        )

    def test_context_manager_blocks_and_restores(self):
        """
        PfManager used as context manager — blocks inside, restores on exit.
        """
        resolved_ip = self._setup_checks()

        ok_before, _ = _try_connect_tcp(
            _LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED, via_ip=resolved_ip
        )
        assert ok_before, "Pre-flight: baseline connectivity required"

        inside_blocked: list = []

        with PfManager() as mgr:
            ok, _ = _try_connect_tcp(
                _LLM_HOST, _LLM_PORT, _TIMEOUT_BLOCKED, via_ip=resolved_ip
            )
            inside_blocked.append(ok)

        assert inside_blocked[0] is False, (
            "Expected connection to be blocked inside 'with PfManager()' block"
        )
        assert not mgr.is_enabled, "is_enabled must be False after context manager exit"

        ok_after, err_after = _try_connect_tcp(
            _LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED, via_ip=resolved_ip
        )
        assert ok_after, f"Connectivity not restored after context manager exit: {err_after}"

    def test_disable_is_idempotent_after_double_call(self):
        """
        Calling disable() twice must not raise and must leave the anchor clean.
        """
        _require_root()
        _require_pfctl()

        mgr = PfManager()
        mgr.enable()
        mgr.disable()
        # Second disable — must not raise
        mgr.disable()
        assert not mgr.is_enabled

    def test_enable_is_safe_to_re_enable(self):
        """
        Calling enable() twice re-loads the anchor cleanly (idempotent).
        """
        resolved_ip = self._setup_checks()

        mgr = PfManager()
        try:
            mgr.enable()
            mgr.enable()   # second enable — should overwrite anchor cleanly
            assert mgr.is_enabled

            ok_blocked, _ = _try_connect_tcp(
                _LLM_HOST, _LLM_PORT, _TIMEOUT_BLOCKED, via_ip=resolved_ip
            )
            assert not ok_blocked, "Expected connection to be blocked after double enable()"

        finally:
            mgr.disable()

    def test_anchor_status_reflects_enabled_state(self):
        """
        PfManager.status() returns non-empty rules while enabled, empty after disable.
        """
        _require_root()
        _require_pfctl()

        mgr = PfManager()
        try:
            mgr.enable()
            rules_while_enabled = mgr.status()
            assert rules_while_enabled is not None
            assert "block" in (rules_while_enabled or "").lower(), (
                f"Expected 'block' in anchor rules while enabled, got: {rules_while_enabled!r}"
            )
        finally:
            mgr.disable()


# ─────────────────────────────────────────────────────────────────────────────
# TestCLIEgressIntegration — CLI subprocess path, real pf + network
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestCLIEgressIntegration:
    """
    Integration tests that exercise the ``piiguard egress enable / disable``
    CLI commands via subprocess, then verify network-level blocking behavior.

    Requirements: root + macOS pf(4) + real network.
    """

    _MODULE_ARGS = [sys.executable, "-m", "pii_guard.cli"]
    _CWD = _REPO_ROOT

    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        """Run the piiguard CLI as a subprocess."""
        return subprocess.run(
            self._MODULE_ARGS + list(args),
            capture_output=True,
            text=True,
            cwd=self._CWD,
        )

    def _setup_checks(self) -> str:
        _require_root()
        _require_pfctl()
        return _require_network(_LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED)

    def test_cli_enable_exits_zero(self):
        _require_root()
        _require_pfctl()
        try:
            result = self._cli("egress", "enable")
            assert result.returncode == 0, (
                f"Expected exit 0, got {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        finally:
            self._cli("egress", "disable")

    def test_cli_enable_blocks_and_disable_restores(self):
        """
        Full CLI cycle:
        1. ``python3 -m pii_guard.cli egress enable``  → exit 0
        2. TCP to api.anthropic.com:443 is blocked
        3. ``python3 -m pii_guard.cli egress disable`` → exit 0
        4. TCP connectivity is restored
        """
        resolved_ip = self._setup_checks()

        try:
            # Step 1: Enable via CLI
            enable_result = self._cli("egress", "enable")
            assert enable_result.returncode == 0, (
                f"'piiguard egress enable' failed (exit {enable_result.returncode}):\n"
                f"stdout: {enable_result.stdout}\nstderr: {enable_result.stderr}"
            )

            # Step 2: Verify blocking
            ok_blocked, err_blocked = _try_connect_tcp(
                _LLM_HOST, _LLM_PORT, _TIMEOUT_BLOCKED, via_ip=resolved_ip
            )
            assert not ok_blocked, (
                f"Expected connection to be BLOCKED after 'piiguard egress enable', "
                f"but it succeeded."
            )

        finally:
            # Step 3: Disable via CLI
            disable_result = self._cli("egress", "disable")
            assert disable_result.returncode == 0, (
                f"'piiguard egress disable' failed (exit {disable_result.returncode}):\n"
                f"stdout: {disable_result.stdout}\nstderr: {disable_result.stderr}"
            )

        # Step 4: Verify restoration
        ok_after, err_after = _try_connect_tcp(
            _LLM_HOST, _LLM_PORT, _TIMEOUT_ALLOWED, via_ip=resolved_ip
        )
        assert ok_after, (
            f"Expected connectivity RESTORED after 'piiguard egress disable', "
            f"but got: {err_after}"
        )

    def test_cli_enable_stdout_mentions_enabled(self):
        _require_root()
        _require_pfctl()
        try:
            result = self._cli("egress", "enable")
            assert "enabled" in result.stdout.lower(), (
                f"Expected 'enabled' in stdout, got: {result.stdout!r}"
            )
        finally:
            self._cli("egress", "disable")

    def test_cli_disable_stdout_mentions_disabled(self):
        _require_root()
        _require_pfctl()
        self._cli("egress", "enable")
        result = self._cli("egress", "disable")
        assert "disabled" in result.stdout.lower(), (
            f"Expected 'disabled' in stdout, got: {result.stdout!r}"
        )

    def test_cli_status_shows_active_after_enable(self):
        _require_root()
        _require_pfctl()
        try:
            self._cli("egress", "enable")
            status_result = self._cli("egress", "status")
            assert status_result.returncode == 0
            combined = status_result.stdout + status_result.stderr
            assert "active" in combined.lower() or "block" in combined.lower(), (
                f"Expected 'active' or 'block' in status output: {combined!r}"
            )
        finally:
            self._cli("egress", "disable")

    def test_cli_status_shows_not_active_after_disable(self):
        _require_root()
        _require_pfctl()
        self._cli("egress", "enable")
        self._cli("egress", "disable")
        status_result = self._cli("egress", "status")
        combined = status_result.stdout + status_result.stderr
        assert "not active" in combined.lower() or "no rules" in combined.lower(), (
            f"Expected 'not active' in status output after disable: {combined!r}"
        )

    def test_cli_enable_verbose_prints_block_rule(self):
        _require_root()
        _require_pfctl()
        try:
            result = self._cli("egress", "enable", "--verbose")
            assert result.returncode == 0
            assert "block" in result.stdout.lower(), (
                f"Expected 'block' in verbose output: {result.stdout!r}"
            )
        finally:
            self._cli("egress", "disable")
