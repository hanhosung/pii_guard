"""
Unit tests for pii_guard.launcher — Sub-AC 6a.

Verifies that the ProcessLauncher correctly injects PII-Guard proxy
environment variables (ANTHROPIC_BASE_URL / OPENAI_BASE_URL / GEMINI_BASE_URL
and equivalents) into every child process spawned by ouroboros workflows.

Test strategy
-------------
* ``TestBuildProxyEnv``  — pure-logic tests for the standalone helper.
* ``TestProcessLauncherConfig`` — construction / URL-inference behaviour.
* ``TestChildProcessEnvInjection`` — spawn *real* subprocesses and assert that
  env vars are present and point to the proxy address.
* ``TestEnvPatchContextManager`` — verify os.environ is patched inside the
  context and fully restored on exit.
* ``TestOverrideSemantics`` — ensure pre-existing vars are respected by default
  and overridden when allow_override=True.
* ``TestProviderCoverage`` — structural checks that all three provider families
  are represented.

Run with:   pytest tests/test_launcher.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict

import pytest

from pii_guard.launcher import (
    ALL_PROXY_ENV_VARS,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    PROVIDER_ENV_VARS,
    ProcessLauncher,
    build_proxy_env,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

#: Well-known test proxy URL used across all assertions.
TEST_PROXY_URL = "http://127.0.0.1:4444"

#: Alternative proxy URL used to test "existing var not overridden" behaviour.
CUSTOM_PROXY_URL = "http://10.0.0.1:9999"

#: All env var names that must be injected by a default launcher.
EXPECTED_VARS = [
    "ANTHROPIC_BASE_URL",    # Anthropic / Claude SDK
    "OPENAI_BASE_URL",       # OpenAI SDK v1+
    "OPENAI_API_BASE",       # OpenAI SDK <v1 / LangChain
    "GEMINI_BASE_URL",       # Ouroboros convention
    "GOOGLE_GENAI_BASE_URL", # google-genai SDK
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _child_env_snapshot(launcher: ProcessLauncher) -> Dict[str, str]:
    """
    Spawn a subprocess via *launcher.run()* and return a dict of all
    EXPECTED_VARS from the child's os.environ.

    Uses JSON so the comparison is unambiguous.
    """
    script = (
        "import os, json; "
        f"keys = {json.dumps(EXPECTED_VARS)}; "
        "print(json.dumps({k: os.environ.get(k) for k in keys}))"
    )
    result = launcher.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"child process failed: stderr={result.stderr!r}"
    )
    return json.loads(result.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildProxyEnv — standalone helper
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildProxyEnv:
    """Pure-logic tests for build_proxy_env()."""

    def test_injects_all_expected_vars_into_empty_env(self):
        env = build_proxy_env(TEST_PROXY_URL, base_env={})
        for var in EXPECTED_VARS:
            assert env[var] == TEST_PROXY_URL, f"{var} should be {TEST_PROXY_URL}"

    def test_all_proxy_env_vars_constant_matches_expected(self):
        for var in EXPECTED_VARS:
            assert var in ALL_PROXY_ENV_VARS, f"{var} missing from ALL_PROXY_ENV_VARS"

    def test_returns_new_dict_does_not_mutate_base(self):
        base = {}
        env = build_proxy_env(TEST_PROXY_URL, base_env=base)
        assert base == {}           # original untouched
        assert env != base          # returned dict is new

    def test_does_not_override_pre_existing_vars_by_default(self):
        base = {"ANTHROPIC_BASE_URL": CUSTOM_PROXY_URL}
        env = build_proxy_env(TEST_PROXY_URL, base_env=base)
        assert env["ANTHROPIC_BASE_URL"] == CUSTOM_PROXY_URL  # unchanged

    def test_overrides_pre_existing_vars_when_requested(self):
        base = {"ANTHROPIC_BASE_URL": CUSTOM_PROXY_URL}
        env = build_proxy_env(TEST_PROXY_URL, base_env=base, allow_override=True)
        assert env["ANTHROPIC_BASE_URL"] == TEST_PROXY_URL   # overwritten

    def test_preserves_unrelated_env_vars(self):
        base = {"MY_APP_FLAG": "1", "PATH": "/usr/bin"}
        env = build_proxy_env(TEST_PROXY_URL, base_env=base)
        assert env["MY_APP_FLAG"] == "1"
        assert env["PATH"] == "/usr/bin"

    def test_defaults_to_os_environ_when_base_env_is_none(self):
        """When base_env=None, existing os.environ vars are respected."""
        # Run without base_env — should not raise
        env = build_proxy_env(TEST_PROXY_URL)
        for var in ALL_PROXY_ENV_VARS:
            assert var in env

    def test_all_injected_values_are_the_proxy_url(self):
        env = build_proxy_env(TEST_PROXY_URL, base_env={})
        for var in ALL_PROXY_ENV_VARS:
            assert env[var] == TEST_PROXY_URL


# ─────────────────────────────────────────────────────────────────────────────
# TestProcessLauncherConfig — construction and URL inference
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessLauncherConfig:
    """Tests for ProcessLauncher construction and proxy URL resolution."""

    def test_explicit_proxy_url_stored(self):
        launcher = ProcessLauncher(TEST_PROXY_URL)
        assert launcher.proxy_base_url == TEST_PROXY_URL

    def test_default_url_uses_default_constants(self):
        """When no URL given and PIIGUARD_HOST/PORT not set, use defaults."""
        saved_host = os.environ.pop("PIIGUARD_HOST", None)
        saved_port = os.environ.pop("PIIGUARD_PORT", None)
        try:
            launcher = ProcessLauncher()
            expected = f"http://{DEFAULT_PROXY_HOST}:{DEFAULT_PROXY_PORT}"
            assert launcher.proxy_base_url == expected
        finally:
            if saved_host is not None:
                os.environ["PIIGUARD_HOST"] = saved_host
            if saved_port is not None:
                os.environ["PIIGUARD_PORT"] = saved_port

    def test_default_url_reads_piiguard_host_env(self):
        saved = os.environ.pop("PIIGUARD_HOST", None)
        try:
            os.environ["PIIGUARD_HOST"] = "192.168.1.100"
            launcher = ProcessLauncher()
            assert "192.168.1.100" in launcher.proxy_base_url
        finally:
            if saved is not None:
                os.environ["PIIGUARD_HOST"] = saved
            else:
                os.environ.pop("PIIGUARD_HOST", None)

    def test_default_url_reads_piiguard_port_env(self):
        saved = os.environ.pop("PIIGUARD_PORT", None)
        try:
            os.environ["PIIGUARD_PORT"] = "7777"
            launcher = ProcessLauncher()
            assert ":7777" in launcher.proxy_base_url
        finally:
            if saved is not None:
                os.environ["PIIGUARD_PORT"] = saved
            else:
                os.environ.pop("PIIGUARD_PORT", None)

    def test_default_url_reads_both_host_and_port_env(self):
        saved_host = os.environ.pop("PIIGUARD_HOST", None)
        saved_port = os.environ.pop("PIIGUARD_PORT", None)
        try:
            os.environ["PIIGUARD_HOST"] = "10.0.0.1"
            os.environ["PIIGUARD_PORT"] = "9999"
            launcher = ProcessLauncher()
            assert launcher.proxy_base_url == "http://10.0.0.1:9999"
        finally:
            for key, val in [("PIIGUARD_HOST", saved_host), ("PIIGUARD_PORT", saved_port)]:
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)

    def test_allow_override_defaults_to_false(self):
        launcher = ProcessLauncher(TEST_PROXY_URL)
        assert launcher.allow_override is False

    def test_allow_override_can_be_set(self):
        launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=True)
        assert launcher.allow_override is True

    def test_repr_contains_proxy_url(self):
        launcher = ProcessLauncher(TEST_PROXY_URL)
        assert TEST_PROXY_URL in repr(launcher)

    def test_all_vars_property_matches_constant(self):
        launcher = ProcessLauncher(TEST_PROXY_URL)
        assert launcher.all_vars == ALL_PROXY_ENV_VARS

    def test_provider_vars_property_returns_copy(self):
        launcher = ProcessLauncher(TEST_PROXY_URL)
        pv = launcher.provider_vars
        assert isinstance(pv, dict)
        assert set(pv.keys()) == set(PROVIDER_ENV_VARS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# TestChildProcessEnvInjection — real subprocess spawning
# ─────────────────────────────────────────────────────────────────────────────

class TestChildProcessEnvInjection:
    """
    Integration tests: spawn real subprocesses and assert injected env vars
    are present and point to the proxy address.
    """

    def test_child_sees_anthropic_base_url(self):
        """ANTHROPIC_BASE_URL is injected into child process environment."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ANTHROPIC_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            env={}  # start from empty env so we know exactly what's there
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL

    def test_child_sees_openai_base_url(self):
        """OPENAI_BASE_URL is injected into child process environment."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('OPENAI_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            env={},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL

    def test_child_sees_openai_api_base(self):
        """OPENAI_API_BASE (legacy) is injected into child process environment."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('OPENAI_API_BASE', 'MISSING'))"],
            capture_output=True, text=True,
            env={},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL

    def test_child_sees_gemini_base_url(self):
        """GEMINI_BASE_URL is injected into child process environment."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('GEMINI_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            env={},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL

    def test_child_sees_google_genai_base_url(self):
        """GOOGLE_GENAI_BASE_URL is injected into child process environment."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('GOOGLE_GENAI_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            env={},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL

    def test_all_proxy_vars_present_and_correct_in_child(self):
        """
        Core assertion: every provider env var is present in the child
        process and its value equals the configured proxy address.
        """
        launcher = ProcessLauncher(TEST_PROXY_URL)
        snapshot = _child_env_snapshot(launcher)

        for var in EXPECTED_VARS:
            assert snapshot.get(var) == TEST_PROXY_URL, (
                f"{var} in child process should be {TEST_PROXY_URL!r}, "
                f"got {snapshot.get(var)!r}"
            )

    def test_popen_wrapper_injects_env(self):
        """ProcessLauncher.popen() injects proxy env vars (Popen API)."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        proc = launcher.popen(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ANTHROPIC_BASE_URL', 'MISSING'))"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={},
        )
        stdout, _ = proc.communicate()
        assert proc.returncode == 0
        assert stdout.strip() == TEST_PROXY_URL

    def test_check_output_wrapper_injects_env(self):
        """ProcessLauncher.check_output() injects proxy env vars."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        output = launcher.check_output(
            [sys.executable, "-c",
             "import os; print(os.environ.get('OPENAI_BASE_URL', 'MISSING'))"],
            text=True, env={},
        )
        assert output.strip() == TEST_PROXY_URL

    def test_child_proxy_url_contains_host_and_port(self):
        """Proxy URL injected into child has the expected host:port form."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        snapshot = _child_env_snapshot(launcher)
        for var in EXPECTED_VARS:
            val = snapshot.get(var, "")
            assert "127.0.0.1" in val, f"{var}={val!r} should contain host"
            assert "4444" in val, f"{var}={val!r} should contain port"

    def test_child_inherits_non_proxy_env_vars_via_launcher(self):
        """Non-proxy env vars in the base env survive into the child."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        sentinel_key = "PIIGUARD_TEST_SENTINEL"
        sentinel_val = "hello_from_parent"
        result = launcher.run(
            [sys.executable, "-c",
             f"import os; print(os.environ.get('{sentinel_key}', 'MISSING'))"],
            capture_output=True, text=True,
            env={sentinel_key: sentinel_val},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == sentinel_val

    @pytest.mark.parametrize("var", EXPECTED_VARS)
    def test_each_provider_var_injected_parametrized(self, var):
        """Parametrised: each expected var is individually confirmed in child."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        result = launcher.run(
            [sys.executable, "-c",
             f"import os; print(os.environ.get('{var}', 'MISSING'))"],
            capture_output=True, text=True, env={},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL, (
            f"{var} should be {TEST_PROXY_URL!r} in child, "
            f"got {result.stdout.strip()!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestEnvPatchContextManager — os.environ patching
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvPatchContextManager:
    """Tests for ProcessLauncher.env_patch() context manager."""

    def test_vars_set_inside_context(self):
        """All proxy vars are set in os.environ inside the context."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        # Ensure vars are absent before entry
        for var in ALL_PROXY_ENV_VARS:
            os.environ.pop(var, None)

        try:
            with launcher.env_patch():
                for var in ALL_PROXY_ENV_VARS:
                    assert os.environ.get(var) == TEST_PROXY_URL, (
                        f"{var} not set inside env_patch() context"
                    )
        finally:
            for var in ALL_PROXY_ENV_VARS:
                os.environ.pop(var, None)

    def test_vars_restored_after_context(self):
        """All proxy vars are restored to their original values after context exit."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        # Ensure vars are absent before entry
        for var in ALL_PROXY_ENV_VARS:
            os.environ.pop(var, None)

        try:
            with launcher.env_patch():
                pass
            # After exit: vars should be gone again (they were absent before)
            for var in ALL_PROXY_ENV_VARS:
                assert os.environ.get(var) is None, (
                    f"{var} should be unset after env_patch() exits"
                )
        finally:
            for var in ALL_PROXY_ENV_VARS:
                os.environ.pop(var, None)

    def test_pre_existing_var_restored_after_context(self):
        """A var that was set before the context is restored to its original value."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        pre_existing = "http://pre-existing.proxy:1234"
        os.environ["ANTHROPIC_BASE_URL"] = pre_existing

        try:
            with launcher.env_patch():
                # Inside: default allow_override=False, so pre-existing NOT overwritten
                assert os.environ["ANTHROPIC_BASE_URL"] == pre_existing
            # After: original value restored
            assert os.environ["ANTHROPIC_BASE_URL"] == pre_existing
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_context_manager_restores_even_on_exception(self):
        """env_patch() restores os.environ even when the body raises."""
        launcher = ProcessLauncher(TEST_PROXY_URL)
        for var in ALL_PROXY_ENV_VARS:
            os.environ.pop(var, None)

        try:
            with pytest.raises(RuntimeError):
                with launcher.env_patch():
                    raise RuntimeError("test exception inside context")

            # Vars should be restored despite the exception
            for var in ALL_PROXY_ENV_VARS:
                assert os.environ.get(var) is None, (
                    f"{var} not restored after exception in env_patch()"
                )
        finally:
            for var in ALL_PROXY_ENV_VARS:
                os.environ.pop(var, None)

    def test_subprocess_spawned_inside_context_inherits_vars(self):
        """
        A subprocess.run() call made inside env_patch() inherits the proxy
        vars via os.environ (no launcher.run() needed).
        """
        launcher = ProcessLauncher(TEST_PROXY_URL)
        for var in ALL_PROXY_ENV_VARS:
            os.environ.pop(var, None)

        try:
            with launcher.env_patch():
                result = subprocess.run(
                    [sys.executable, "-c",
                     "import os; print(os.environ.get('ANTHROPIC_BASE_URL', 'MISSING'))"],
                    capture_output=True, text=True,
                    # No explicit env= → inherits current os.environ
                )
            assert result.returncode == 0
            assert result.stdout.strip() == TEST_PROXY_URL
        finally:
            for var in ALL_PROXY_ENV_VARS:
                os.environ.pop(var, None)


# ─────────────────────────────────────────────────────────────────────────────
# TestOverrideSemantics — allow_override=True/False behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestOverrideSemantics:
    """Tests for the allow_override flag on ProcessLauncher and build_proxy_env."""

    def test_existing_var_not_overridden_by_default(self):
        """By default, a pre-existing proxy var in the child env is left alone."""
        launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=False)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ANTHROPIC_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            # Pass ANTHROPIC_BASE_URL already set to a different URL
            env={"ANTHROPIC_BASE_URL": CUSTOM_PROXY_URL},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == CUSTOM_PROXY_URL  # custom value preserved

    def test_existing_var_overridden_when_allow_override_true(self):
        """With allow_override=True, pre-existing proxy vars are overwritten."""
        launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=True)
        result = launcher.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ANTHROPIC_BASE_URL', 'MISSING'))"],
            capture_output=True, text=True,
            env={"ANTHROPIC_BASE_URL": CUSTOM_PROXY_URL},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == TEST_PROXY_URL   # overwritten

    def test_all_vars_injected_into_fresh_env_regardless_of_override_flag(self):
        """When no pre-existing vars, all vars are injected whether or not override is set."""
        for allow_override in (True, False):
            launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=allow_override)
            snapshot = _child_env_snapshot(launcher)
            # _child_env_snapshot uses launcher.run without explicit env=,
            # but we just check all EXPECTED_VARS are present
            for var in EXPECTED_VARS:
                assert snapshot.get(var) is not None, (
                    f"{var} missing from child env with allow_override={allow_override}"
                )

    def test_env_patch_does_not_override_by_default(self):
        """env_patch() respects allow_override=False for pre-existing vars."""
        pre_existing = "http://existing.proxy:5555"
        os.environ["OPENAI_BASE_URL"] = pre_existing
        launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=False)
        try:
            with launcher.env_patch():
                assert os.environ["OPENAI_BASE_URL"] == pre_existing
        finally:
            os.environ.pop("OPENAI_BASE_URL", None)

    def test_env_patch_overrides_when_allow_override_true(self):
        """env_patch() overwrites pre-existing vars when allow_override=True."""
        pre_existing = "http://existing.proxy:5555"
        os.environ["OPENAI_BASE_URL"] = pre_existing
        launcher = ProcessLauncher(TEST_PROXY_URL, allow_override=True)
        try:
            with launcher.env_patch():
                assert os.environ["OPENAI_BASE_URL"] == TEST_PROXY_URL
        finally:
            os.environ.pop("OPENAI_BASE_URL", None)


# ─────────────────────────────────────────────────────────────────────────────
# TestProviderCoverage — structural / ontology checks
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderCoverage:
    """Ensure all three provider families are represented in the constants."""

    def test_anthropic_provider_vars_present(self):
        assert "anthropic" in PROVIDER_ENV_VARS
        assert "ANTHROPIC_BASE_URL" in PROVIDER_ENV_VARS["anthropic"]

    def test_openai_provider_vars_present(self):
        assert "openai" in PROVIDER_ENV_VARS
        assert "OPENAI_BASE_URL" in PROVIDER_ENV_VARS["openai"]
        assert "OPENAI_API_BASE" in PROVIDER_ENV_VARS["openai"]

    def test_gemini_provider_vars_present(self):
        assert "gemini" in PROVIDER_ENV_VARS
        assert "GEMINI_BASE_URL" in PROVIDER_ENV_VARS["gemini"]
        assert "GOOGLE_GENAI_BASE_URL" in PROVIDER_ENV_VARS["gemini"]

    def test_all_proxy_env_vars_covers_all_providers(self):
        """ALL_PROXY_ENV_VARS must include at least one var from each provider."""
        for provider, vars_list in PROVIDER_ENV_VARS.items():
            assert any(v in ALL_PROXY_ENV_VARS for v in vars_list), (
                f"Provider '{provider}' has no vars in ALL_PROXY_ENV_VARS"
            )

    def test_three_provider_families_covered(self):
        """Exactly the three required providers are declared."""
        required = {"anthropic", "openai", "gemini"}
        assert required.issubset(set(PROVIDER_ENV_VARS.keys()))

    def test_all_proxy_env_vars_non_empty(self):
        assert len(ALL_PROXY_ENV_VARS) > 0

    def test_no_duplicate_vars_in_all_proxy_env_vars(self):
        assert len(ALL_PROXY_ENV_VARS) == len(set(ALL_PROXY_ENV_VARS))
