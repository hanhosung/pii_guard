"""
Unit tests for pii_guard.pf_manager — Sub-AC 6b-i.

Verifies the pf anchor rule manager without requiring root privileges or a
real pf stack.  All pfctl subprocess calls are intercepted via
``unittest.mock``.

Test strategy
-------------
TestBuildTableDefinition
    Pure-logic tests for ``build_table_definition()``:
    syntax correctness, CIDR ordering, persist flag, empty input rejection.

TestBuildBlockRule
    Pure-logic tests for ``build_block_rule()``:
    correct keyword sequence, table name interpolation, port list formatting.

TestBuildAnchorRules
    Integration-level tests for ``build_anchor_rules()``:
    header comment present, table definition included, block rule included,
    combined CIDR deduplication, empty-ranges rejection.

TestCommandBuilders
    Verify the argv-list builders for each pfctl operation:
    load, flush-rules, flush-table, show-rules — anchor name injection,
    sudo prefix, flag correctness.

TestCollectAllCidrs
    Tests for the CIDR flattening / deduplication helper.

TestPfManagerEnable
    PfManager.enable() mocked tests:
    correct pfctl command called, rules text piped to stdin,
    anchor name appears in command, ``is_enabled`` flag set.

TestPfManagerDisable
    PfManager.disable() mocked tests:
    flush-rules command called, flush-table command called,
    ``is_enabled`` flag cleared, teardown called even on exception.

TestPfManagerStatus
    PfManager.status() mocked tests:
    show-rules command called, output returned, None on error.

TestPfManagerContextManager
    Context-manager (``with PfManager()``) tests:
    enable called on entry, disable called on exit, disable called even
    when body raises.

TestAnchorNameResolution
    Anchor name comes from constructor arg, then PIIGUARD_PF_ANCHOR env var,
    then DEFAULT_ANCHOR_NAME.

TestProviderCoverage
    Structural checks that all three LLM provider families have non-empty
    IP range lists and that ALL_PROVIDER_IP_RANGES includes all three.

TestPfRuleError
    PfRuleError is raised on non-zero pfctl exit when check=True;
    suppressed when check_output=False.

Run with:   pytest tests/test_pf_manager.py -v
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from pii_guard.pf_manager import (
    ALL_PROVIDER_IP_RANGES,
    ANCHOR_TABLE_NAME,
    ANTHROPIC_IP_RANGES,
    BLOCKED_PORTS,
    DEFAULT_ANCHOR_NAME,
    DEFAULT_PFCTL_PATH,
    DEFAULT_SUDO_PATH,
    GOOGLE_IP_RANGES,
    OPENAI_IP_RANGES,
    PfManager,
    PfRuleError,
    build_anchor_rules,
    build_block_rule,
    build_flush_rules_command,
    build_flush_table_command,
    build_load_command,
    build_show_rules_command,
    build_table_definition,
    collect_all_cidrs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CIDRS: List[str] = ["192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"]
SAMPLE_IP_RANGES: Dict[str, List[str]] = {
    "providerA": ["10.0.0.0/8", "172.16.0.0/12"],
    "providerB": ["192.168.0.0/16"],
}


def _make_ok_result(stdout: str = "") -> subprocess.CompletedProcess:
    """Return a fake CompletedProcess with returncode=0."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=""
    )


def _make_fail_result(returncode: int = 1, stderr: str = "pfctl error") -> subprocess.CompletedProcess:
    """Return a fake CompletedProcess with a non-zero returncode."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestCollectAllCidrs
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectAllCidrs:
    """Tests for collect_all_cidrs() — provider-keyed dict → flat list."""

    def test_single_provider_returns_its_cidrs(self):
        cidrs = collect_all_cidrs({"a": ["1.0.0.0/8"]})
        assert cidrs == ["1.0.0.0/8"]

    def test_multi_provider_cidrs_are_flattened(self):
        cidrs = collect_all_cidrs({
            "a": ["1.0.0.0/8", "2.0.0.0/8"],
            "b": ["3.0.0.0/8"],
        })
        assert cidrs == ["1.0.0.0/8", "2.0.0.0/8", "3.0.0.0/8"]

    def test_duplicate_cidrs_are_removed(self):
        cidrs = collect_all_cidrs({
            "a": ["1.0.0.0/8"],
            "b": ["1.0.0.0/8", "2.0.0.0/8"],
        })
        # 1.0.0.0/8 appears only once
        assert cidrs.count("1.0.0.0/8") == 1
        assert "2.0.0.0/8" in cidrs

    def test_empty_provider_skipped(self):
        cidrs = collect_all_cidrs({"empty": [], "a": ["1.0.0.0/8"]})
        assert cidrs == ["1.0.0.0/8"]

    def test_empty_dict_returns_empty_list(self):
        assert collect_all_cidrs({}) == []

    def test_whitespace_stripped_from_cidrs(self):
        cidrs = collect_all_cidrs({"a": ["  1.0.0.0/8  ", " 2.0.0.0/8"]})
        assert "1.0.0.0/8" in cidrs
        assert "2.0.0.0/8" in cidrs

    def test_insertion_order_preserved_for_first_provider(self):
        cidrs = collect_all_cidrs({"a": ["1.0.0.0/8", "2.0.0.0/8", "3.0.0.0/8"]})
        assert cidrs == ["1.0.0.0/8", "2.0.0.0/8", "3.0.0.0/8"]


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildTableDefinition
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTableDefinition:
    """Tests for build_table_definition() — pf table syntax."""

    def test_output_starts_with_table_keyword(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert result.startswith("table <")

    def test_output_contains_default_table_name(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert f"<{ANCHOR_TABLE_NAME}>" in result

    def test_custom_table_name_used(self):
        result = build_table_definition(SAMPLE_CIDRS, table_name="my_ips")
        assert "<my_ips>" in result
        assert f"<{ANCHOR_TABLE_NAME}>" not in result

    def test_output_contains_persist_keyword(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert "persist" in result

    def test_all_cidrs_appear_in_output(self):
        result = build_table_definition(SAMPLE_CIDRS)
        for cidr in SAMPLE_CIDRS:
            assert cidr in result, f"CIDR {cidr} missing from table definition"

    def test_opening_brace_present(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert "{" in result

    def test_closing_brace_present(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert "}" in result

    def test_single_cidr_valid(self):
        result = build_table_definition(["10.0.0.0/8"])
        assert "10.0.0.0/8" in result
        assert "persist" in result

    def test_many_cidrs_all_present(self):
        cidrs = [f"10.{i}.0.0/16" for i in range(10)]
        result = build_table_definition(cidrs)
        for cidr in cidrs:
            assert cidr in result

    def test_empty_cidrs_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            build_table_definition([])

    def test_output_is_multiline_for_multiple_cidrs(self):
        result = build_table_definition(SAMPLE_CIDRS)
        assert "\n" in result

    def test_commas_separate_entries(self):
        """Each CIDR except the last must be followed by a comma somewhere on its line."""
        result = build_table_definition(SAMPLE_CIDRS)
        lines_with_cidrs = [l for l in result.splitlines() if any(c in l for c in SAMPLE_CIDRS[:-1])]
        for line in lines_with_cidrs:
            assert "," in line, f"Expected comma on line: {line!r}"

    def test_table_name_in_angle_brackets(self):
        """Table name must be wrapped in angle brackets as required by pf syntax."""
        result = build_table_definition(SAMPLE_CIDRS)
        # Must contain '<name>' not just 'name'
        assert f"<{ANCHOR_TABLE_NAME}>" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildBlockRule
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBlockRule:
    """Tests for build_block_rule() — pf block statement syntax."""

    def test_starts_with_block_keyword(self):
        result = build_block_rule()
        assert result.startswith("block")

    def test_contains_out_direction(self):
        result = build_block_rule()
        assert " out " in result

    def test_contains_quick_keyword(self):
        result = build_block_rule()
        assert " quick " in result

    def test_contains_proto_tcp(self):
        result = build_block_rule()
        assert "proto tcp" in result

    def test_contains_to_keyword(self):
        result = build_block_rule()
        assert " to " in result

    def test_contains_default_table_name_in_angle_brackets(self):
        result = build_block_rule()
        assert f"<{ANCHOR_TABLE_NAME}>" in result

    def test_custom_table_name_used(self):
        result = build_block_rule(table_name="custom_tbl")
        assert "<custom_tbl>" in result

    def test_contains_port_keyword(self):
        result = build_block_rule()
        assert "port" in result

    def test_default_ports_present(self):
        result = build_block_rule()
        for port in BLOCKED_PORTS:
            assert str(port) in result, f"Port {port} missing from block rule"

    def test_custom_ports_used(self):
        result = build_block_rule(ports=[8080, 8443])
        assert "8080" in result
        assert "8443" in result

    def test_ports_enclosed_in_braces(self):
        result = build_block_rule()
        assert "{" in result
        assert "}" in result

    def test_single_port_valid(self):
        result = build_block_rule(ports=[443])
        assert "443" in result

    def test_empty_ports_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            build_block_rule(ports=[])

    def test_keyword_order_block_out_quick_proto_tcp_to(self):
        """Verify the pf keyword order is syntactically valid (block out quick proto tcp to)."""
        import re
        result = build_block_rule()
        # Use whole-word search to avoid matching "to" inside "proto" or "out" inside "output"
        def _pos(word):
            m = re.search(r'\b' + re.escape(word) + r'\b', result)
            assert m is not None, f"keyword {word!r} not found in rule: {result!r}"
            return m.start()
        assert _pos("block") < _pos("out")
        assert _pos("out") < _pos("quick")
        assert _pos("quick") < _pos("proto")
        assert _pos("proto") < _pos("tcp")
        # "to" follows "tcp" — use search from the position after "proto tcp"
        tcp_end = result.index("tcp") + 3
        remaining = result[tcp_end:]
        assert " to " in remaining, (
            f"'to' keyword must follow 'tcp' in rule: {result!r}"
        )

    def test_single_line_output(self):
        result = build_block_rule()
        assert "\n" not in result.rstrip("\n")


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildAnchorRules
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAnchorRules:
    """Integration-level tests for build_anchor_rules()."""

    def test_contains_header_comment(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert result.startswith("#")

    def test_header_mentions_anchor_name(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES, anchor_name="piiguard")
        assert "piiguard" in result

    def test_default_anchor_name_in_header(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert DEFAULT_ANCHOR_NAME in result

    def test_contains_table_definition(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert "table <" in result
        assert "persist" in result

    def test_contains_block_rule(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert "block out" in result

    def test_all_provider_cidrs_present(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        for cidrs in SAMPLE_IP_RANGES.values():
            for cidr in cidrs:
                assert cidr in result, f"CIDR {cidr} missing from anchor rules"

    def test_duplicate_cidrs_deduplicated(self):
        ip_ranges = {
            "a": ["1.0.0.0/8"],
            "b": ["1.0.0.0/8"],
        }
        result = build_anchor_rules(ip_ranges)
        assert result.count("1.0.0.0/8") == 1

    def test_empty_ip_ranges_raises_value_error(self):
        with pytest.raises(ValueError):
            build_anchor_rules({})

    def test_all_providers_empty_lists_raises_value_error(self):
        with pytest.raises(ValueError):
            build_anchor_rules({"a": [], "b": []})

    def test_custom_table_name_propagated(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES, table_name="custom_t")
        assert "<custom_t>" in result
        # Table name in both table definition and block rule
        assert result.count("<custom_t>") >= 2

    def test_custom_ports_propagated(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES, ports=[8443])
        assert "8443" in result

    def test_output_ends_with_newline(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert result.endswith("\n")

    def test_table_definition_before_block_rule(self):
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        table_pos = result.index("table <")
        block_pos = result.index("block out")
        assert table_pos < block_pos, "table definition must precede block rule"

    def test_all_provider_ip_ranges_constant_is_valid(self):
        """The built-in ALL_PROVIDER_IP_RANGES must produce a valid ruleset."""
        result = build_anchor_rules(ALL_PROVIDER_IP_RANGES)
        assert "table <" in result
        assert "block out" in result
        # Check at least one known Anthropic CIDR is present
        assert any(cidr in result for cidr in ANTHROPIC_IP_RANGES)

    def test_no_management_comment_about_editing(self):
        """Header must warn that the file is auto-generated."""
        result = build_anchor_rules(SAMPLE_IP_RANGES)
        assert "do not edit" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TestCommandBuilders
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandBuilders:
    """Verify the argv-list builders for each pfctl operation."""

    # ── build_load_command ───────────────────────────────────────────────────

    def test_load_command_starts_with_sudo(self):
        cmd = build_load_command("piiguard")
        assert cmd[0] == DEFAULT_SUDO_PATH

    def test_load_command_second_element_is_pfctl(self):
        cmd = build_load_command("piiguard")
        assert cmd[1] == DEFAULT_PFCTL_PATH

    def test_load_command_contains_anchor_flag(self):
        cmd = build_load_command("piiguard")
        assert "-a" in cmd

    def test_load_command_anchor_name_follows_a_flag(self):
        cmd = build_load_command("piiguard")
        idx = cmd.index("-a")
        assert cmd[idx + 1] == "piiguard"

    def test_load_command_has_f_minus_for_stdin(self):
        """pfctl -f - means read rules from stdin."""
        cmd = build_load_command("piiguard")
        assert "-f" in cmd
        assert "-" in cmd

    def test_load_command_f_and_dash_are_adjacent(self):
        cmd = build_load_command("piiguard")
        idx = cmd.index("-f")
        assert cmd[idx + 1] == "-"

    def test_load_command_custom_anchor_name(self):
        cmd = build_load_command("my_anchor")
        assert "my_anchor" in cmd

    def test_load_command_returns_list_of_strings(self):
        cmd = build_load_command("piiguard")
        assert isinstance(cmd, list)
        assert all(isinstance(s, str) for s in cmd)

    # ── build_flush_rules_command ────────────────────────────────────────────

    def test_flush_rules_command_starts_with_sudo(self):
        cmd = build_flush_rules_command("piiguard")
        assert cmd[0] == DEFAULT_SUDO_PATH

    def test_flush_rules_command_contains_pfctl(self):
        cmd = build_flush_rules_command("piiguard")
        assert DEFAULT_PFCTL_PATH in cmd

    def test_flush_rules_command_contains_anchor_flag(self):
        cmd = build_flush_rules_command("piiguard")
        assert "-a" in cmd

    def test_flush_rules_command_anchor_name_follows_a_flag(self):
        cmd = build_flush_rules_command("piiguard")
        idx = cmd.index("-a")
        assert cmd[idx + 1] == "piiguard"

    def test_flush_rules_command_contains_capital_f_rules(self):
        """pfctl -F rules flushes all filter rules in the anchor."""
        cmd = build_flush_rules_command("piiguard")
        assert "-F" in cmd
        assert "rules" in cmd

    def test_flush_rules_command_f_rules_adjacent(self):
        cmd = build_flush_rules_command("piiguard")
        idx = cmd.index("-F")
        assert cmd[idx + 1] == "rules"

    def test_flush_rules_command_custom_anchor(self):
        cmd = build_flush_rules_command("test_anchor")
        assert "test_anchor" in cmd

    # ── build_flush_table_command ────────────────────────────────────────────

    def test_flush_table_command_starts_with_sudo(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        assert cmd[0] == DEFAULT_SUDO_PATH

    def test_flush_table_command_contains_pfctl(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        assert DEFAULT_PFCTL_PATH in cmd

    def test_flush_table_command_contains_anchor_flag(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        assert "-a" in cmd

    def test_flush_table_command_anchor_name_follows_a_flag(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        idx = cmd.index("-a")
        assert cmd[idx + 1] == "piiguard"

    def test_flush_table_command_contains_uppercase_t_flush(self):
        """pfctl -T flush destroys all entries in the named table."""
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        assert "-T" in cmd
        assert "flush" in cmd

    def test_flush_table_command_t_flush_adjacent(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        idx = cmd.index("-T")
        assert cmd[idx + 1] == "flush"

    def test_flush_table_command_contains_lowercase_t_table_name_flag(self):
        """pfctl -t <name> selects the table."""
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        assert "-t" in cmd

    def test_flush_table_command_table_name_follows_t_flag(self):
        cmd = build_flush_table_command("piiguard", ANCHOR_TABLE_NAME)
        idx = cmd.index("-t")
        assert cmd[idx + 1] == ANCHOR_TABLE_NAME

    def test_flush_table_command_custom_table_name(self):
        cmd = build_flush_table_command("piiguard", "my_table")
        assert "my_table" in cmd

    # ── build_show_rules_command ─────────────────────────────────────────────

    def test_show_rules_command_starts_with_sudo(self):
        cmd = build_show_rules_command("piiguard")
        assert cmd[0] == DEFAULT_SUDO_PATH

    def test_show_rules_command_contains_pfctl(self):
        cmd = build_show_rules_command("piiguard")
        assert DEFAULT_PFCTL_PATH in cmd

    def test_show_rules_command_contains_anchor_flag(self):
        cmd = build_show_rules_command("piiguard")
        assert "-a" in cmd

    def test_show_rules_command_anchor_name_follows_a_flag(self):
        cmd = build_show_rules_command("piiguard")
        idx = cmd.index("-a")
        assert cmd[idx + 1] == "piiguard"

    def test_show_rules_command_contains_lowercase_s_rules(self):
        """pfctl -s rules shows the current ruleset."""
        cmd = build_show_rules_command("piiguard")
        assert "-s" in cmd
        assert "rules" in cmd

    def test_show_rules_command_s_rules_adjacent(self):
        cmd = build_show_rules_command("piiguard")
        idx = cmd.index("-s")
        assert cmd[idx + 1] == "rules"

    # ── Env var overrides ────────────────────────────────────────────────────

    def test_pfctl_path_overridden_by_env_var(self, monkeypatch):
        monkeypatch.setenv("PIIGUARD_PFCTL", "/usr/local/sbin/pfctl")
        cmd = build_load_command("piiguard")
        assert "/usr/local/sbin/pfctl" in cmd

    def test_sudo_path_overridden_by_env_var(self, monkeypatch):
        monkeypatch.setenv("PIIGUARD_SUDO", "/usr/local/bin/sudo")
        cmd = build_load_command("piiguard")
        assert "/usr/local/bin/sudo" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# TestPfManagerEnable
# ─────────────────────────────────────────────────────────────────────────────

class TestPfManagerEnable:
    """PfManager.enable() — mocked pfctl subprocess."""

    def _make_mgr(self, ip_ranges=None, anchor_name=None, **kw) -> PfManager:
        return PfManager(
            ip_ranges=ip_ranges or SAMPLE_IP_RANGES,
            anchor_name=anchor_name or "piiguard",
            **kw,
        )

    def test_enable_calls_run_pfctl_once(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        assert mock_run.call_count == 1

    def test_enable_passes_rules_text_as_input(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            expected_rules = mgr.build_rules()
            mgr.enable()
        _, kwargs = mock_run.call_args
        assert kwargs.get("input_text") == expected_rules

    def test_enable_command_contains_anchor_name(self):
        mgr = self._make_mgr(anchor_name="test_anchor")
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        cmd_arg = mock_run.call_args[0][0]   # first positional arg = cmd list
        assert "test_anchor" in cmd_arg

    def test_enable_command_contains_sudo(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        cmd_arg = mock_run.call_args[0][0]
        assert DEFAULT_SUDO_PATH in cmd_arg

    def test_enable_command_contains_pfctl(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        cmd_arg = mock_run.call_args[0][0]
        assert DEFAULT_PFCTL_PATH in cmd_arg

    def test_enable_command_contains_f_minus_stdin_flag(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        cmd_arg = mock_run.call_args[0][0]
        assert "-f" in cmd_arg
        assert "-" in cmd_arg

    def test_enable_sets_is_enabled_true(self):
        mgr = self._make_mgr()
        assert not mgr.is_enabled
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()):
            mgr.enable()
        assert mgr.is_enabled

    def test_enable_rules_contain_block_out(self):
        mgr = self._make_mgr()
        captured: list = []
        def _capture(cmd, *, input_text=None, check=True):
            captured.append(input_text)
            return _make_ok_result()
        with patch.object(mgr, "_run_pfctl", side_effect=_capture):
            mgr.enable()
        assert captured
        assert "block out" in captured[0]

    def test_enable_rules_contain_table_definition(self):
        mgr = self._make_mgr()
        captured: list = []
        def _capture(cmd, *, input_text=None, check=True):
            captured.append(input_text)
            return _make_ok_result()
        with patch.object(mgr, "_run_pfctl", side_effect=_capture):
            mgr.enable()
        assert "table <" in captured[0]
        assert "persist" in captured[0]

    def test_enable_rules_contain_all_provider_cidrs(self):
        mgr = self._make_mgr()
        captured: list = []
        def _capture(cmd, *, input_text=None, check=True):
            captured.append(input_text)
            return _make_ok_result()
        with patch.object(mgr, "_run_pfctl", side_effect=_capture):
            mgr.enable()
        rules_text = captured[0]
        for cidrs in SAMPLE_IP_RANGES.values():
            for cidr in cidrs:
                assert cidr in rules_text, f"CIDR {cidr} missing from piped rules"

    def test_enable_with_default_ip_ranges(self):
        """Default IP ranges (all three providers) produce a valid ruleset."""
        mgr = PfManager(anchor_name="piiguard")
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.enable()
        assert mock_run.call_count == 1
        _, kwargs = mock_run.call_args
        rules = kwargs["input_text"]
        assert "block out" in rules
        # At least one Anthropic CIDR present
        assert any(cidr in rules for cidr in ANTHROPIC_IP_RANGES)
        # At least one OpenAI CIDR present
        assert any(cidr in rules for cidr in OPENAI_IP_RANGES)
        # At least one Google CIDR present
        assert any(cidr in rules for cidr in GOOGLE_IP_RANGES)

    def test_enable_with_empty_ip_ranges_raises(self):
        mgr = PfManager(ip_ranges={}, anchor_name="piiguard")
        with pytest.raises(ValueError):
            mgr.enable()
        assert not mgr.is_enabled


# ─────────────────────────────────────────────────────────────────────────────
# TestPfManagerDisable
# ─────────────────────────────────────────────────────────────────────────────

class TestPfManagerDisable:
    """PfManager.disable() — teardown behavior."""

    def _enabled_mgr(self) -> PfManager:
        """Return a manager with is_enabled=True (skip actual enable)."""
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        mgr._enabled = True
        return mgr

    def test_disable_calls_run_pfctl_twice(self):
        mgr = self._enabled_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        assert mock_run.call_count == 2

    def test_disable_first_call_is_flush_rules(self):
        mgr = self._enabled_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        first_cmd = mock_run.call_args_list[0][0][0]  # first call, positional arg
        # Must contain -F rules
        assert "-F" in first_cmd
        assert "rules" in first_cmd

    def test_disable_second_call_is_flush_table(self):
        mgr = self._enabled_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        second_cmd = mock_run.call_args_list[1][0][0]
        # Must contain -T flush -t <table_name>
        assert "-T" in second_cmd
        assert "flush" in second_cmd
        assert "-t" in second_cmd
        assert ANCHOR_TABLE_NAME in second_cmd

    def test_disable_sets_is_enabled_false(self):
        mgr = self._enabled_mgr()
        assert mgr.is_enabled
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()):
            mgr.disable()
        assert not mgr.is_enabled

    def test_disable_anchor_name_in_both_commands(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="test_anchor")
        mgr._enabled = True
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            assert "test_anchor" in cmd, f"anchor name missing from command: {cmd}"

    def test_disable_clears_flag_even_when_pfctl_fails(self):
        """is_enabled is cleared even if pfctl calls fail."""
        mgr = self._enabled_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_fail_result()):
            # check_output=False by default for disable calls
            mgr.disable()
        assert not mgr.is_enabled

    def test_disable_called_twice_does_not_raise(self):
        """Double-disable must not raise (idempotent teardown)."""
        mgr = self._enabled_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()):
            mgr.disable()
            mgr.disable()  # second call — flag already False

    def test_disable_on_never_enabled_manager_runs_cleanup_anyway(self):
        """disable() should attempt cleanup even if enable() was never called."""
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        assert not mgr.is_enabled
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        # Cleanup runs regardless
        assert mock_run.call_count == 2

    def test_disable_flush_table_uses_correct_table_name(self):
        mgr = PfManager(
            ip_ranges=SAMPLE_IP_RANGES,
            anchor_name="piiguard",
            table_name="my_table",
        )
        mgr._enabled = True
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.disable()
        second_cmd = mock_run.call_args_list[1][0][0]
        assert "my_table" in second_cmd

    def test_disable_passes_check_false_to_run_pfctl(self):
        """disable() uses check=False so partial teardown does not raise."""
        mgr = self._enabled_mgr()
        check_values: list = []
        def _capture(cmd, *, input_text=None, check=True):
            check_values.append(check)
            return _make_ok_result()
        with patch.object(mgr, "_run_pfctl", side_effect=_capture):
            mgr.disable()
        # Both calls should be check=False for graceful teardown
        assert all(not c for c in check_values), (
            f"disable() calls should use check=False, got: {check_values}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestPfManagerStatus
# ─────────────────────────────────────────────────────────────────────────────

class TestPfManagerStatus:
    """PfManager.status() — read-only pfctl query."""

    def _make_mgr(self) -> PfManager:
        return PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")

    def test_status_calls_run_pfctl_once(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result("some rules")) as mock_run:
            mgr.status()
        assert mock_run.call_count == 1

    def test_status_command_contains_show_rules_flags(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.status()
        cmd = mock_run.call_args[0][0]
        assert "-s" in cmd
        assert "rules" in cmd

    def test_status_command_contains_anchor_name(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()) as mock_run:
            mgr.status()
        cmd = mock_run.call_args[0][0]
        assert "piiguard" in cmd

    def test_status_returns_pfctl_stdout(self):
        mgr = self._make_mgr()
        fake_output = "block out quick proto tcp to <piiguard_llm_ips> port { 443 }"
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result(fake_output)):
            result = mgr.status()
        assert result == fake_output

    def test_status_returns_none_on_error(self):
        mgr = self._make_mgr()
        def _raise(cmd, *, input_text=None, check=True):
            raise PfRuleError("permission denied")
        with patch.object(mgr, "_run_pfctl", side_effect=_raise):
            result = mgr.status()
        assert result is None

    def test_status_returns_empty_string_when_no_rules(self):
        mgr = self._make_mgr()
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result("")):
            result = mgr.status()
        assert result == ""

    def test_status_does_not_change_is_enabled_flag(self):
        mgr = self._make_mgr()
        mgr._enabled = True
        with patch.object(mgr, "_run_pfctl", return_value=_make_ok_result()):
            mgr.status()
        assert mgr.is_enabled


# ─────────────────────────────────────────────────────────────────────────────
# TestPfManagerContextManager
# ─────────────────────────────────────────────────────────────────────────────

class TestPfManagerContextManager:
    """Context-manager (with PfManager()) enable/disable lifecycle."""

    def test_enable_called_on_enter(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch.object(mgr, "enable") as mock_enable, \
             patch.object(mgr, "disable"):
            with mgr:
                pass
            assert mock_enable.call_count == 1

    def test_disable_called_on_exit(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch.object(mgr, "enable"), \
             patch.object(mgr, "disable") as mock_disable:
            with mgr:
                pass
            assert mock_disable.call_count == 1

    def test_disable_called_even_when_body_raises(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch.object(mgr, "enable"), \
             patch.object(mgr, "disable") as mock_disable:
            with pytest.raises(RuntimeError):
                with mgr:
                    raise RuntimeError("test body exception")
            assert mock_disable.call_count == 1

    def test_context_manager_returns_manager_instance(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch.object(mgr, "enable"), \
             patch.object(mgr, "disable"):
            with mgr as ctx:
                assert ctx is mgr

    def test_context_manager_does_not_suppress_body_exception(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch.object(mgr, "enable"), \
             patch.object(mgr, "disable"):
            with pytest.raises(ValueError, match="propagated"):
                with mgr:
                    raise ValueError("propagated")

    def test_full_enable_disable_cycle_via_context_manager(self):
        """Full cycle: enable loads rules, disable flushes them — both mocked."""
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        call_log: list = []
        def _log(cmd, *, input_text=None, check=True):
            call_log.append(cmd)
            return _make_ok_result()
        with patch.object(mgr, "_run_pfctl", side_effect=_log):
            with mgr:
                # After enable, one call logged (load)
                assert len(call_log) == 1
                assert "-f" in call_log[0]  # load command
            # After disable, three total calls (load + flush-rules + flush-table)
            assert len(call_log) == 3
            assert "-F" in call_log[1]   # flush-rules
            assert "-T" in call_log[2]   # flush-table


# ─────────────────────────────────────────────────────────────────────────────
# TestAnchorNameResolution
# ─────────────────────────────────────────────────────────────────────────────

class TestAnchorNameResolution:
    """Anchor name source priority: constructor arg > env var > default."""

    def test_explicit_anchor_name_in_constructor(self):
        mgr = PfManager(anchor_name="explicit_name")
        assert mgr.anchor_name == "explicit_name"

    def test_env_var_used_when_no_constructor_arg(self, monkeypatch):
        monkeypatch.setenv("PIIGUARD_PF_ANCHOR", "env_anchor")
        mgr = PfManager()
        assert mgr.anchor_name == "env_anchor"

    def test_default_anchor_name_when_no_arg_and_no_env(self, monkeypatch):
        monkeypatch.delenv("PIIGUARD_PF_ANCHOR", raising=False)
        mgr = PfManager()
        assert mgr.anchor_name == DEFAULT_ANCHOR_NAME

    def test_constructor_arg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("PIIGUARD_PF_ANCHOR", "env_anchor")
        mgr = PfManager(anchor_name="constructor_wins")
        assert mgr.anchor_name == "constructor_wins"

    def test_default_anchor_name_constant_is_piiguard(self):
        assert DEFAULT_ANCHOR_NAME == "piiguard"

    def test_anchor_name_appears_in_repr(self):
        mgr = PfManager(anchor_name="my_anchor")
        assert "my_anchor" in repr(mgr)


# ─────────────────────────────────────────────────────────────────────────────
# TestProviderCoverage
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderCoverage:
    """Structural checks on LLM provider IP range constants."""

    def test_anthropic_ip_ranges_non_empty(self):
        assert len(ANTHROPIC_IP_RANGES) > 0

    def test_openai_ip_ranges_non_empty(self):
        assert len(OPENAI_IP_RANGES) > 0

    def test_google_ip_ranges_non_empty(self):
        assert len(GOOGLE_IP_RANGES) > 0

    def test_all_provider_ip_ranges_has_anthropic(self):
        assert "anthropic" in ALL_PROVIDER_IP_RANGES

    def test_all_provider_ip_ranges_has_openai(self):
        assert "openai" in ALL_PROVIDER_IP_RANGES

    def test_all_provider_ip_ranges_has_google(self):
        assert "google" in ALL_PROVIDER_IP_RANGES

    def test_all_three_provider_families_present(self):
        required = {"anthropic", "openai", "google"}
        assert required.issubset(set(ALL_PROVIDER_IP_RANGES.keys()))

    def test_anthropic_ranges_are_valid_cidr_format(self):
        for cidr in ANTHROPIC_IP_RANGES:
            assert "/" in cidr, f"CIDR {cidr!r} missing slash notation"
            prefix, bits = cidr.rsplit("/", 1)
            assert bits.isdigit(), f"CIDR prefix length {bits!r} not a digit"
            assert 0 <= int(bits) <= 32

    def test_openai_ranges_are_valid_cidr_format(self):
        for cidr in OPENAI_IP_RANGES:
            assert "/" in cidr
            _, bits = cidr.rsplit("/", 1)
            assert bits.isdigit()
            assert 0 <= int(bits) <= 32

    def test_google_ranges_are_valid_cidr_format(self):
        for cidr in GOOGLE_IP_RANGES:
            assert "/" in cidr
            _, bits = cidr.rsplit("/", 1)
            assert bits.isdigit()
            assert 0 <= int(bits) <= 32

    def test_blocked_ports_includes_443(self):
        assert 443 in BLOCKED_PORTS

    def test_blocked_ports_includes_80(self):
        assert 80 in BLOCKED_PORTS

    def test_anchor_table_name_no_spaces(self):
        assert " " not in ANCHOR_TABLE_NAME

    def test_anchor_table_name_non_empty(self):
        assert ANCHOR_TABLE_NAME


# ─────────────────────────────────────────────────────────────────────────────
# TestPfRuleError
# ─────────────────────────────────────────────────────────────────────────────

class TestPfRuleError:
    """PfRuleError raised on non-zero pfctl exit when check=True."""

    def test_pf_rule_error_is_runtime_error(self):
        assert issubclass(PfRuleError, RuntimeError)

    def test_run_pfctl_raises_pf_rule_error_on_nonzero_exit(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch("subprocess.run", return_value=_make_fail_result(returncode=1, stderr="permission denied")):
            with pytest.raises(PfRuleError, match="permission denied"):
                mgr._run_pfctl(["sudo", "pfctl", "-a", "piiguard", "-f", "-"], check=True)

    def test_run_pfctl_no_raise_when_check_false(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch("subprocess.run", return_value=_make_fail_result(returncode=1)):
            # Should not raise
            result = mgr._run_pfctl(
                ["sudo", "pfctl", "-a", "piiguard", "-F", "rules"],
                check=False,
            )
        assert result.returncode == 1

    def test_run_pfctl_raises_pf_rule_error_on_file_not_found(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard")
        with patch("subprocess.run", side_effect=FileNotFoundError("pfctl not found")):
            with pytest.raises(PfRuleError, match="not found"):
                mgr._run_pfctl(["sudo", "/nonexistent/pfctl", "-a", "piiguard", "-f", "-"])

    def test_enable_propagates_pf_rule_error_when_pfctl_fails(self):
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard", check_output=True)
        with patch.object(mgr, "_run_pfctl", side_effect=PfRuleError("pfctl failed")):
            with pytest.raises(PfRuleError):
                mgr.enable()
        assert not mgr.is_enabled

    def test_disable_with_check_output_false_does_not_raise_on_pfctl_fail(self):
        """check_output=False suppresses teardown errors."""
        mgr = PfManager(ip_ranges=SAMPLE_IP_RANGES, anchor_name="piiguard", check_output=False)
        mgr._enabled = True
        with patch("subprocess.run", return_value=_make_fail_result(returncode=1)):
            # Should not raise even though pfctl "fails"
            mgr.disable()
        assert not mgr.is_enabled


# ─────────────────────────────────────────────────────────────────────────────
# TestPfManagerRepr
# ─────────────────────────────────────────────────────────────────────────────

class TestPfManagerRepr:
    """PfManager.__repr__ contains useful fields."""

    def test_repr_contains_anchor_name(self):
        mgr = PfManager(anchor_name="myanchor")
        assert "myanchor" in repr(mgr)

    def test_repr_contains_table_name(self):
        mgr = PfManager(table_name="my_tbl")
        assert "my_tbl" in repr(mgr)

    def test_repr_contains_ports(self):
        mgr = PfManager(ports=[443])
        assert "443" in repr(mgr)

    def test_repr_contains_enabled_state(self):
        mgr = PfManager()
        assert "enabled=False" in repr(mgr)
        mgr._enabled = True
        assert "enabled=True" in repr(mgr)
