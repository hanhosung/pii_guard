"""
Unit tests for the PII-Guard protection-boundary declaration module.

Sub-AC 6c — Protection-boundary declaration tests
--------------------------------------------------
Tests assert that:

1.  get_protection_boundary() returns a BoundaryReport with the expected
    structure for both ``cooperative_gateway`` and ``egress_lockdown`` modes.

2.  Defended/undefended items correctly reflect the current enforcement tier:
    - cooperative_gateway: env-var injection scope only
    - egress_lockdown: pf(4) network-layer scope (superset of cooperative)

3.  The report contains NO false-assurance claims:
    - 'undefended' is always non-empty
    - 'bypass_paths' is always non-empty
    - Key limitations are explicitly stated
    - threat_actor_model declares root/kernel actors as out of scope
    - assurance_statement uses bounded, not unconditional, language

4.  The CLI ``piiguard boundary`` command returns exit 0 and produces output.

5.  The ``--json`` flag emits valid JSON with all required keys.

6.  Invalid mode argument returns exit 2.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

# Repository root (parent of tests/) — used as cwd for subprocess CLI invocations
# so tests are location-independent rather than tied to an absolute install path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pii_guard.boundary import (
    BoundaryItem,
    BoundaryReport,
    EnforcementTier,
    get_protection_boundary,
    print_boundary_report,
)
from pii_guard.cli import build_parser, cmd_boundary, main


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _descriptions(items: list) -> list:
    """Return the description strings from a list of BoundaryItem."""
    return [item.description for item in items]


# ─────────────────────────────────────────────────────────────────────────────
# TestBoundaryReportStructure — core API contract
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryReportStructure:
    """Verify the structure of BoundaryReport for each enforcement tier."""

    def test_cooperative_returns_boundary_report(self):
        report = get_protection_boundary("cooperative_gateway")
        assert isinstance(report, BoundaryReport)

    def test_lockdown_returns_boundary_report(self):
        report = get_protection_boundary("egress_lockdown")
        assert isinstance(report, BoundaryReport)

    def test_cooperative_tier_field(self):
        report = get_protection_boundary("cooperative_gateway")
        assert report.enforcement_tier == "cooperative_gateway"

    def test_lockdown_tier_field(self):
        report = get_protection_boundary("egress_lockdown")
        assert report.enforcement_tier == "egress_lockdown"

    def test_default_tier_is_cooperative(self):
        report = get_protection_boundary()
        assert report.enforcement_tier == "cooperative_gateway"

    def test_enum_cooperative_accepted(self):
        report = get_protection_boundary(EnforcementTier.COOPERATIVE_GATEWAY)
        assert report.enforcement_tier == "cooperative_gateway"

    def test_enum_lockdown_accepted(self):
        report = get_protection_boundary(EnforcementTier.EGRESS_LOCKDOWN)
        assert report.enforcement_tier == "egress_lockdown"

    def test_custom_proxy_url_reflected(self):
        report = get_protection_boundary(proxy_url="http://127.0.0.1:9999")
        assert report.proxy_url == "http://127.0.0.1:9999"

    def test_default_proxy_url(self):
        report = get_protection_boundary()
        assert "127.0.0.1" in report.proxy_url

    def test_generated_at_set(self):
        report = get_protection_boundary()
        assert report.generated_at  # non-empty string
        assert "Z" in report.generated_at or "T" in report.generated_at

    def test_custom_generated_at_accepted(self):
        ts = "2026-01-01T00:00:00Z"
        report = get_protection_boundary(generated_at=ts)
        assert report.generated_at == ts

    def test_invalid_tier_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown enforcement_tier"):
            get_protection_boundary("invalid_mode")

    def test_boundary_item_fields_present(self):
        report = get_protection_boundary()
        for item in report.defended + report.undefended:
            assert item.description
            assert item.detail
            assert item.category


# ─────────────────────────────────────────────────────────────────────────────
# TestDefendedItemsCorrect — mode vs. defended scope
# ─────────────────────────────────────────────────────────────────────────────

class TestDefendedItemsCorrect:
    """
    Assert defended items correctly reflect the enforcement tier.

    The cooperative tier covers env-var-injected processes.
    The egress_lockdown tier is a superset: includes all cooperative items
    PLUS network-layer blocking items.
    """

    def test_cooperative_defended_non_empty(self):
        report = get_protection_boundary("cooperative_gateway")
        assert len(report.defended) > 0

    def test_lockdown_defended_non_empty(self):
        report = get_protection_boundary("egress_lockdown")
        assert len(report.defended) > 0

    def test_cooperative_defends_ouroboros_spawned_processes(self):
        report = get_protection_boundary("cooperative_gateway")
        descs = _descriptions(report.defended)
        assert any("ouroboros" in d.lower() for d in descs), (
            f"Expected ouroboros processes in defended items; got: {descs}"
        )

    def test_cooperative_defends_sdk_env_var_tools(self):
        report = get_protection_boundary("cooperative_gateway")
        descs = _descriptions(report.defended)
        assert any("sdk" in d.lower() or "env" in d.lower() or "env-var" in d.lower()
                   for d in descs), (
            f"Expected SDK env-var tools in defended items; got: {descs}"
        )

    def test_cooperative_defends_claude_openai_gemini_protocols(self):
        report = get_protection_boundary("cooperative_gateway")
        categories = [item.category for item in report.defended]
        assert "protocol" in categories, (
            "Expected a 'protocol' category in defended items"
        )

    def test_lockdown_defends_pf_network_blocking(self):
        """egress_lockdown must include a network-layer pf(4) blocking item."""
        report = get_protection_boundary("egress_lockdown")
        descs = _descriptions(report.defended)
        assert any(
            "pf" in d.lower() or "tcp" in d.lower() or "cidr" in d.lower()
            or "firewall" in d.lower() or "network" in d.lower()
            for d in descs
        ), f"Expected pf/TCP/network blocking in lockdown defended items; got: {descs}"

    def test_lockdown_defends_more_than_cooperative(self):
        """Egress-lockdown should have at least as many defended items as cooperative."""
        coop = get_protection_boundary("cooperative_gateway")
        lock = get_protection_boundary("egress_lockdown")
        assert len(lock.defended) >= len(coop.defended), (
            f"Expected lockdown to defend ≥ cooperative; "
            f"coop={len(coop.defended)}, lockdown={len(lock.defended)}"
        )

    def test_lockdown_includes_hard_coded_url_bypass_fix(self):
        """Egress-lockdown must defend against clients that hard-code base URLs."""
        report = get_protection_boundary("egress_lockdown")
        descs = _descriptions(report.defended)
        assert any(
            "hard-cod" in d.lower() or "bypass" in d.lower() or "ignore" in d.lower()
            for d in descs
        ), f"Expected hard-coded URL handling in lockdown defended items; got: {descs}"


# ─────────────────────────────────────────────────────────────────────────────
# TestNoFalseAssurance — core honest-threat-model contract
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFalseAssurance:
    """
    Assert the report makes no false assurance claims.

    Key properties:
    - 'undefended' is always non-empty (there are always limitations)
    - 'bypass_paths' is always non-empty
    - Root/kernel actors are always in undefended
    - threat_actor_model declares root/kernel as out of scope
    - assurance_statement is bounded, not unconditional
    - The report never claims to protect against root-level actors
    """

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_undefended_always_non_empty(self, tier):
        report = get_protection_boundary(tier)
        assert len(report.undefended) > 0, (
            f"[{tier}] undefended must be non-empty — report must declare limitations"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_bypass_paths_always_non_empty(self, tier):
        report = get_protection_boundary(tier)
        assert len(report.bypass_paths) > 0, (
            f"[{tier}] bypass_paths must be non-empty — bypass paths must be declared"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_root_actors_always_in_undefended(self, tier):
        """Root/kernel actors must appear in undefended items for both tiers."""
        report = get_protection_boundary(tier)
        actor_items = [item for item in report.undefended if item.category == "actor"]
        assert actor_items, (
            f"[{tier}] Expected at least one 'actor' category in undefended; "
            f"got categories: {[i.category for i in report.undefended]}"
        )
        actor_descs = [item.description for item in actor_items]
        assert any(
            "root" in d.lower() or "kernel" in d.lower() or "sudo" in d.lower()
            for d in actor_descs
        ), (
            f"[{tier}] Expected root/kernel actor in undefended actor items; "
            f"got: {actor_descs}"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_threat_actor_model_declares_root_out_of_scope(self, tier):
        """threat_actor_model must explicitly exclude root/kernel actors."""
        report = get_protection_boundary(tier)
        model = report.threat_actor_model.lower()
        assert "root" in model or "kernel" in model, (
            f"[{tier}] threat_actor_model must declare root/kernel as out of scope"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_assurance_statement_is_bounded(self, tier):
        """
        The assurance statement must be hedged / bounded, not unconditional.

        It should NOT claim 'all traffic is protected' or similar absolute
        guarantees.  It SHOULD contain 'does not' / 'NOT' / 'limited to'
        language.
        """
        report = get_protection_boundary(tier)
        stmt = report.assurance_statement.lower()
        bounded_phrases = [
            "does not", "not guarantee", "not protect", "limited to",
            "not ", "outside the", "cannot", "only",
        ]
        assert any(phrase in stmt for phrase in bounded_phrases), (
            f"[{tier}] assurance_statement must use bounded language; "
            f"got: {report.assurance_statement[:200]!r}"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_threat_actor_model_non_empty(self, tier):
        report = get_protection_boundary(tier)
        assert report.threat_actor_model.strip(), (
            f"[{tier}] threat_actor_model must be non-empty"
        )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_assurance_statement_non_empty(self, tier):
        report = get_protection_boundary(tier)
        assert report.assurance_statement.strip(), (
            f"[{tier}] assurance_statement must be non-empty"
        )

    def test_cooperative_undefended_includes_outside_ouroboros(self):
        """Cooperative mode must declare non-ouroboros processes as undefended."""
        report = get_protection_boundary("cooperative_gateway")
        descs = _descriptions(report.undefended)
        assert any(
            "outside" in d.lower() or "without" in d.lower() or "not" in d.lower()
            for d in descs
        ), f"Expected 'outside' / 'without' / 'not' in cooperative undefended: {descs}"

    def test_cooperative_undefended_includes_hard_coded_url(self):
        """Cooperative mode must declare hard-coded base-URL clients as undefended."""
        report = get_protection_boundary("cooperative_gateway")
        descs = _descriptions(report.undefended)
        assert any(
            "hard-cod" in d.lower() or "hard_cod" in d.lower()
            for d in descs
        ), f"Expected hard-coded URL in cooperative undefended: {descs}"

    def test_cooperative_bypass_mentions_hard_coded_url(self):
        """Cooperative bypass paths must mention hard-coded base URL as a bypass."""
        report = get_protection_boundary("cooperative_gateway")
        bypass_text = " ".join(report.bypass_paths).lower()
        assert "hard-cod" in bypass_text or "base url" in bypass_text.lower(), (
            f"Expected hard-coded base URL in cooperative bypass_paths: {report.bypass_paths}"
        )

    def test_lockdown_undefended_includes_non_standard_port(self):
        """Egress-lockdown must declare non-standard ports as undefended."""
        report = get_protection_boundary("egress_lockdown")
        descs = _descriptions(report.undefended)
        assert any("port" in d.lower() for d in descs), (
            f"Expected non-standard port in lockdown undefended: {descs}"
        )

    def test_lockdown_undefended_includes_cidr_drift(self):
        """Egress-lockdown must declare CIDR drift as a coverage gap."""
        report = get_protection_boundary("egress_lockdown")
        descs = _descriptions(report.undefended)
        assert any(
            "cidr" in d.lower() or "ip range" in d.lower() or "static" in d.lower()
            for d in descs
        ), f"Expected CIDR/IP range limitation in lockdown undefended: {descs}"

    def test_no_tier_claims_total_protection(self):
        """No tier should claim total / complete / absolute protection."""
        for tier in ["cooperative_gateway", "egress_lockdown"]:
            report = get_protection_boundary(tier)
            stmt = report.assurance_statement.lower()
            forbidden = [
                "completely protect", "fully protect", "all traffic is protected",
                "no bypass", "cannot be bypassed", "absolute protection",
            ]
            for phrase in forbidden:
                assert phrase not in stmt, (
                    f"[{tier}] assurance_statement contains false-assurance phrase "
                    f"'{phrase}': {report.assurance_statement[:300]!r}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TestBoundaryDiffers — mode distinction
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryDiffers:
    """
    Assert that the two tiers produce meaningfully different reports, confirming
    the report reflects the current mode rather than returning the same content.
    """

    def test_cooperative_and_lockdown_differ(self):
        coop = get_protection_boundary("cooperative_gateway")
        lock = get_protection_boundary("egress_lockdown")
        # Defended lists must differ (lockdown has more items)
        assert coop.defended != lock.defended

    def test_cooperative_and_lockdown_bypass_differ(self):
        coop = get_protection_boundary("cooperative_gateway")
        lock = get_protection_boundary("egress_lockdown")
        assert coop.bypass_paths != lock.bypass_paths

    def test_cooperative_and_lockdown_undefended_differ(self):
        coop = get_protection_boundary("cooperative_gateway")
        lock = get_protection_boundary("egress_lockdown")
        assert coop.undefended != lock.undefended

    def test_cooperative_assurance_differs_from_lockdown(self):
        coop = get_protection_boundary("cooperative_gateway")
        lock = get_protection_boundary("egress_lockdown")
        assert coop.assurance_statement != lock.assurance_statement


# ─────────────────────────────────────────────────────────────────────────────
# TestAsDictJson — serialisation contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAsDictJson:
    """Verify as_dict() and as_json() produce valid, complete output."""

    def test_as_dict_returns_dict(self):
        report = get_protection_boundary()
        assert isinstance(report.as_dict(), dict)

    def test_as_dict_has_required_keys(self):
        report = get_protection_boundary()
        d = report.as_dict()
        required_keys = {
            "enforcement_tier", "proxy_url", "defended", "undefended",
            "bypass_paths", "threat_actor_model", "assurance_statement",
            "generated_at",
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - d.keys()}"
        )

    def test_as_dict_defended_contains_category_description(self):
        report = get_protection_boundary()
        d = report.as_dict()
        for item in d["defended"]:
            assert "description" in item
            assert "category" in item

    def test_as_dict_undefended_contains_category_description(self):
        report = get_protection_boundary()
        d = report.as_dict()
        for item in d["undefended"]:
            assert "description" in item
            assert "category" in item

    def test_as_json_returns_valid_json(self):
        report = get_protection_boundary()
        text = report.as_json()
        parsed = json.loads(text)  # must not raise
        assert isinstance(parsed, dict)

    def test_as_json_lockdown_tier_reflected(self):
        report = get_protection_boundary("egress_lockdown")
        parsed = json.loads(report.as_json())
        assert parsed["enforcement_tier"] == "egress_lockdown"

    def test_as_json_bypass_paths_is_list_of_strings(self):
        report = get_protection_boundary()
        parsed = json.loads(report.as_json())
        assert isinstance(parsed["bypass_paths"], list)
        assert all(isinstance(p, str) for p in parsed["bypass_paths"])


# ─────────────────────────────────────────────────────────────────────────────
# TestPrintBoundaryReport — renderer
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintBoundaryReport:
    """Verify the human-readable renderer produces expected output."""

    def test_render_cooperative_contains_tier(self):
        report = get_protection_boundary("cooperative_gateway")
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "cooperative_gateway" in out

    def test_render_lockdown_contains_tier(self):
        report = get_protection_boundary("egress_lockdown")
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "egress_lockdown" in out

    def test_render_contains_defended_section(self):
        report = get_protection_boundary()
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "DEFENDED" in out

    def test_render_contains_not_defended_section(self):
        report = get_protection_boundary()
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "NOT DEFENDED" in out

    def test_render_contains_bypass_section(self):
        report = get_protection_boundary()
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "BYPASS" in out

    def test_render_contains_assurance_section(self):
        report = get_protection_boundary()
        buf = io.StringIO()
        print_boundary_report(report, stream=buf)
        out = buf.getvalue()
        assert "Assurance" in out or "assurance" in out

    def test_render_verbose_includes_detail(self):
        report = get_protection_boundary()
        buf = io.StringIO()
        print_boundary_report(report, stream=buf, verbose=True)
        out = buf.getvalue()
        # Detailed text is longer than the first BoundaryItem description
        first_detail = report.defended[0].detail.split("\n")[0][:30]
        assert first_detail in out

    def test_render_non_verbose_shorter_than_verbose(self):
        report = get_protection_boundary()
        buf_short = io.StringIO()
        buf_long = io.StringIO()
        print_boundary_report(report, stream=buf_short, verbose=False)
        print_boundary_report(report, stream=buf_long, verbose=True)
        assert len(buf_long.getvalue()) > len(buf_short.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# TestCLIBoundaryCommand — CLI surface
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIBoundaryCommand:
    """Unit tests for the 'piiguard boundary' CLI command."""

    def _run(self, argv, capsys=None):
        """Run main() and return the exit code."""
        return main(argv)

    def test_boundary_command_exits_zero(self, capsys):
        rc = main(["boundary"])
        assert rc == 0

    def test_boundary_cooperative_exits_zero(self, capsys):
        rc = main(["boundary", "--mode", "cooperative_gateway"])
        assert rc == 0

    def test_boundary_lockdown_exits_zero(self, capsys):
        rc = main(["boundary", "--mode", "egress_lockdown"])
        assert rc == 0

    def test_boundary_default_prints_cooperative(self, capsys):
        main(["boundary"])
        out = capsys.readouterr().out
        assert "cooperative_gateway" in out

    def test_boundary_lockdown_mode_prints_lockdown(self, capsys):
        main(["boundary", "--mode", "egress_lockdown"])
        out = capsys.readouterr().out
        assert "egress_lockdown" in out

    def test_boundary_json_flag_emits_json(self, capsys):
        main(["boundary", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)  # must not raise
        assert parsed["enforcement_tier"] == "cooperative_gateway"

    def test_boundary_json_lockdown_emits_correct_tier(self, capsys):
        main(["boundary", "--mode", "egress_lockdown", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["enforcement_tier"] == "egress_lockdown"

    def test_boundary_json_has_undefended(self, capsys):
        main(["boundary", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "undefended" in parsed
        assert len(parsed["undefended"]) > 0

    def test_boundary_json_has_bypass_paths(self, capsys):
        main(["boundary", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "bypass_paths" in parsed
        assert len(parsed["bypass_paths"]) > 0

    def test_boundary_verbose_flag_includes_detail(self, capsys):
        main(["boundary", "--verbose"])
        out = capsys.readouterr().out
        # verbose mode should include detail text which is longer
        assert len(out) > 500  # very conservative lower bound

    def test_boundary_custom_proxy_url(self, capsys):
        main(["boundary", "--proxy-url", "http://127.0.0.1:9876"])
        out = capsys.readouterr().out
        assert "9876" in out

    def test_boundary_parser_has_mode_flag(self):
        parser = build_parser()
        args = parser.parse_args(["boundary", "--mode", "egress_lockdown"])
        assert args.mode == "egress_lockdown"

    def test_boundary_parser_has_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["boundary", "--json"])
        assert args.json is True

    def test_boundary_parser_has_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["boundary", "--verbose"])
        assert args.verbose is True

    def test_boundary_parser_default_mode_is_cooperative(self):
        parser = build_parser()
        args = parser.parse_args(["boundary"])
        assert args.mode == "cooperative_gateway"

    def test_boundary_help_mentions_enforcement_tier(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "boundary", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "cooperative_gateway" in combined or "egress_lockdown" in combined

    def test_boundary_help_mentions_defended(self):
        result = subprocess.run(
            [sys.executable, "-m", "pii_guard.cli", "boundary", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        combined = result.stdout.lower() + result.stderr.lower()
        assert "defend" in combined or "protect" in combined or "boundary" in combined


class TestCLIBoundaryCommandInvalidMode:
    """Test invalid --mode value handling."""

    def test_invalid_mode_rejected_by_parser(self):
        """argparse should reject unknown mode values."""
        with pytest.raises(SystemExit) as exc_info:
            main(["boundary", "--mode", "invalid_mode"])
        # argparse exits with code 2 for argument errors
        assert exc_info.value.code == 2

    def test_invalid_mode_direct_function_returns_2(self):
        """cmd_boundary with an invalid mode set through mock args returns 2."""
        import argparse
        args = argparse.Namespace(
            mode="invalid_mode",
            proxy_url="http://127.0.0.1:4444",
            json=False,
            verbose=False,
        )
        rc = cmd_boundary(args)
        assert rc == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestCLISubprocess — subprocess invocation
# ─────────────────────────────────────────────────────────────────────────────

class TestCLISubprocess:
    """Run the CLI as a subprocess to verify end-to-end behaviour."""

    _CWD = _REPO_ROOT

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pii_guard.cli"] + list(args),
            capture_output=True,
            text=True,
            cwd=self._CWD,
        )

    def test_subprocess_boundary_exits_zero(self):
        result = self._cli("boundary")
        assert result.returncode == 0, (
            f"Expected exit 0; got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

    def test_subprocess_boundary_stdout_non_empty(self):
        result = self._cli("boundary")
        assert result.stdout.strip(), "boundary command produced no output"

    def test_subprocess_boundary_json_exits_zero(self):
        result = self._cli("boundary", "--json")
        assert result.returncode == 0

    def test_subprocess_boundary_json_is_valid_json(self):
        result = self._cli("boundary", "--json")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, dict)

    def test_subprocess_boundary_json_has_all_required_keys(self):
        result = self._cli("boundary", "--json")
        parsed = json.loads(result.stdout)
        required = {
            "enforcement_tier", "proxy_url", "defended", "undefended",
            "bypass_paths", "threat_actor_model", "assurance_statement",
            "generated_at",
        }
        assert required.issubset(parsed.keys()), (
            f"Missing keys in JSON output: {required - parsed.keys()}"
        )

    def test_subprocess_boundary_lockdown_json(self):
        result = self._cli("boundary", "--mode", "egress_lockdown", "--json")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["enforcement_tier"] == "egress_lockdown"
        assert len(parsed["defended"]) > 0
        assert len(parsed["undefended"]) > 0
        assert len(parsed["bypass_paths"]) > 0

    def test_subprocess_boundary_stderr_clean(self):
        result = self._cli("boundary")
        assert not result.stderr.strip(), (
            f"Unexpected stderr output: {result.stderr!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestAllCategoryTypes — category coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestAllCategoryTypes:
    """Verify that both tiers use expected category labels."""

    _EXPECTED_CATEGORIES = {"process", "protocol", "port", "actor", "coverage"}

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_all_items_have_known_category(self, tier):
        report = get_protection_boundary(tier)
        all_items = report.defended + report.undefended
        for item in all_items:
            assert item.category in self._EXPECTED_CATEGORIES, (
                f"[{tier}] Unknown category {item.category!r} in item "
                f"{item.description!r}"
            )

    @pytest.mark.parametrize("tier", ["cooperative_gateway", "egress_lockdown"])
    def test_process_category_present_in_both_defended_and_undefended(self, tier):
        report = get_protection_boundary(tier)
        defended_cats = {i.category for i in report.defended}
        undefended_cats = {i.category for i in report.undefended}
        assert "process" in defended_cats, f"[{tier}] No 'process' in defended"
        assert "actor" in undefended_cats, f"[{tier}] No 'actor' in undefended"
