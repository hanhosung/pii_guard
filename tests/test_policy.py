"""
Tests for the PII-Guard policy config loader (Sub-AC 5a).

Scenarios covered
-----------------
1. **Valid load** — all YAML knobs are parsed into the correct PolicyConfig fields.
2. **Schema-invalid fallback** — load a valid file first, replace with an invalid
   one, reload → retains last-valid config, returns False.
3. **Missing-file fallback** — loader constructed with non-existent path →
   SECURE_DEFAULTS are returned immediately.
4. **Empty file** — treated as empty policy, defaults apply.
5. **File deletion after valid load** → reverts to SECURE_DEFAULTS on next reload.
6. **Hot-reload mtime detection** — reload_if_changed() is a no-op when the file
   has not changed, and reloads when it has.
7. **Per-category overrides** — action, mask_style, min_confidence, stage2_fail_action
   are parsed correctly, including partial specs (only overriding some fields).
8. **Allowlist** — both shorthand string and dict forms, compiled regex, bad regex
   raises ValueError.
9. **Pin-list** — valid hashes loaded; changes without approval retain old pin-list
   and warn; changes with approval accepted.
10. **Channel overrides** — per-channel knobs parsed and accessible via channel_setting.
11. **Watcher thread** — start/stop lifecycle.
12. **load_policy() convenience** — wraps PolicyLoader for one-shot use.
13. **SECURE_DEFAULTS** — verify the singleton has the expected secure values.
14. **Unknown top-level keys** — warned but not fatal.
15. **YAML parse error** — retains last-valid config.
16. **Type-error fields** — e.g., memory_budget_mb as a non-int, rehydrate as a
    non-bool, etc.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from pii_guard.policy import (
    SECURE_DEFAULTS,
    AllowlistEntry,
    CategoryPolicy,
    ChannelOverride,
    PinListEntry,
    PolicyConfig,
    PolicyLoader,
    _parse_and_validate,
    _hash_pin_list,
    load_policy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: write a policy YAML file and force a new mtime
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str, delay: float = 0.02) -> None:
    """Write *content* to *path* and ensure the OS updates mtime."""
    path.write_text(content, encoding="utf-8")
    # Advance mtime by at least 10 ms so stat() sees a change
    t = time.time() + delay
    os.utime(str(path), (t, t))


# ─────────────────────────────────────────────────────────────────────────────
# 13. SECURE_DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSecureDefaults:
    """The built-in defaults must be maximally restrictive."""

    def test_fail_mode_is_closed(self):
        assert SECURE_DEFAULTS.fail_mode == "closed"

    def test_on_content_failure_is_block(self):
        assert SECURE_DEFAULTS.on_content_failure == "block"

    def test_on_infra_failure_is_degrade_to_stage1(self):
        assert SECURE_DEFAULTS.on_infra_failure == "degrade_to_stage1"

    def test_stage2_fail_action_is_mask_known_only(self):
        assert SECURE_DEFAULTS.stage2_fail_action == "mask_known_only"

    def test_unscannable_action_is_block(self):
        assert SECURE_DEFAULTS.unscannable_action == "block"

    def test_rehydrate_is_true(self):
        assert SECURE_DEFAULTS.rehydrate is True

    def test_memory_budget_is_1024(self):
        assert SECURE_DEFAULTS.memory_budget_mb == 1024

    def test_no_categories_override(self):
        assert SECURE_DEFAULTS.categories == {}

    def test_no_allowlist(self):
        assert SECURE_DEFAULTS.allowlist == []

    def test_no_pin_list(self):
        assert SECURE_DEFAULTS.pin_list == []

    def test_pin_list_approved_true_for_empty_list(self):
        # Empty pin-list → approval is vacuously true
        assert SECURE_DEFAULTS.pin_list_approved is True

    def test_no_channel_overrides(self):
        assert SECURE_DEFAULTS.channel_overrides == {}

    def test_source_is_built_in(self):
        assert "built-in" in SECURE_DEFAULTS.source


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing-file fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingFileFallback:
    """Loader with a non-existent path must return SECURE_DEFAULTS."""

    def test_nonexistent_path_returns_secure_defaults(self, tmp_path):
        policy_file = tmp_path / "nonexistent.yaml"
        loader = PolicyLoader(str(policy_file))

        config = loader.config
        assert config.fail_mode == "closed"
        assert config.on_content_failure == "block"
        assert config.unscannable_action == "block"
        assert config.rehydrate is True
        assert config.categories == {}
        assert config.allowlist == []

    def test_none_path_returns_secure_defaults(self):
        loader = PolicyLoader(None)
        config = loader.config
        assert config.fail_mode == "closed"
        assert config is SECURE_DEFAULTS

    def test_no_path_given_returns_secure_defaults_via_load_policy(self):
        config = load_policy(None)
        assert config.fail_mode == "closed"
        assert config is SECURE_DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Valid load
# ─────────────────────────────────────────────────────────────────────────────

class TestValidLoad:
    """Full round-trip: write a YAML file, load, verify each knob."""

    # Use plain (no backslash) regex patterns to avoid YAML double-quoted
    # escape sequence restrictions — tests cover pattern/label parsing,
    # not regex semantics.
    FULL_YAML = "\n".join([
        'version: "1"',
        "",
        "fail_mode: open",
        "on_content_failure: warn_allow",
        "on_infra_failure: block",
        "stage2_fail_action: open",
        "unscannable_action: warn_allow",
        "rehydrate: false",
        "memory_budget_mb: 512",
        "",
        "categories:",
        "  EMAIL:",
        "    action: allow",
        "    mask_style: partial",
        "    min_confidence: 0.95",
        "  API_KEY:",
        "    action: block",
        "    min_confidence: 0.99",
        "  PERSON:",
        "    stage2_fail_action: block",
        "",
        "allowlist:",
        "  - pattern: test-fixture@example[.]com",
        "    label: CI fixture email",
        "  - noreply@internal[.]example",
        "",
        "pin_list:",
        "  - hash: sha256:aabbcc",
        "    category: EMAIL",
        "    action: allow",
        "    label: relay address",
        "pin_list_approved: true",
        "",
        "channel_overrides:",
        "  cli:",
        "    unscannable_action: warn_allow",
        "    stage2_fail_action: block",
        "  ouroboros:",
        "    fail_mode: closed",
        "    on_content_failure: block",
        "",
    ])

    def test_fail_mode_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.fail_mode == "open"

    def test_on_content_failure_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.on_content_failure == "warn_allow"

    def test_on_infra_failure_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.on_infra_failure == "block"

    def test_stage2_fail_action_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.stage2_fail_action == "open"

    def test_unscannable_action_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.unscannable_action == "warn_allow"

    def test_rehydrate_false(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.rehydrate is False

    def test_memory_budget_mb(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.memory_budget_mb == 512

    def test_category_email_action_allow(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        cp = config.categories.get("EMAIL")
        assert cp is not None
        assert cp.action == "allow"

    def test_category_email_mask_style(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.categories["EMAIL"].mask_style == "partial"

    def test_category_email_min_confidence(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.categories["EMAIL"].min_confidence == pytest.approx(0.95)

    def test_category_api_key_action(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.categories["API_KEY"].action == "block"

    def test_category_person_partial_spec(self, tmp_path):
        """PERSON only overrides stage2_fail_action; action/mask_style are None."""
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        cp = config.categories.get("PERSON")
        assert cp is not None
        assert cp.stage2_fail_action == "block"
        assert cp.action is None   # not overridden
        assert cp.mask_style is None

    def test_allowlist_length(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert len(config.allowlist) == 2

    def test_allowlist_first_entry_label(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.allowlist[0].label == "CI fixture email"

    def test_allowlist_first_entry_compiled(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        patterns = config.allowlist_patterns()
        assert len(patterns) == 2
        # Pattern uses [.] instead of \. to avoid YAML double-quote escaping
        assert patterns[0].search("test-fixture@example.com")

    def test_allowlist_shorthand_string(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        # Second entry was a bare unquoted YAML string
        assert config.allowlist[1].label == ""
        assert config.allowlist[1].pattern == "noreply@internal[.]example"

    def test_pin_list_loaded(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert len(config.pin_list) == 1
        entry = config.pin_list[0]
        assert entry.hash == "sha256:aabbcc"
        assert entry.category == "EMAIL"
        assert entry.action == "allow"
        assert entry.label == "relay address"

    def test_pin_list_approved_true(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.pin_list_approved is True

    def test_channel_override_cli_unscannable(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("cli", "unscannable_action") == "warn_allow"

    def test_channel_override_cli_stage2(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("cli", "stage2_fail_action") == "block"

    def test_channel_override_ouroboros_fail_mode(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("ouroboros", "fail_mode") == "closed"

    def test_channel_override_missing_channel_returns_none(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("unknown_channel", "fail_mode") is None

    def test_source_is_file_path(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        assert str(p) in config.source

    def test_get_category_policy_helper(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, self.FULL_YAML)
        config = PolicyLoader(str(p)).config
        cp = config.get_category_policy("EMAIL")
        assert cp is not None
        assert cp.action == "allow"
        assert config.get_category_policy("NONEXISTENT") is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Empty file
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_file_returns_secure_defaults(self, tmp_path):
        p = tmp_path / "empty.yaml"
        _write(p, "")
        config = PolicyLoader(str(p)).config
        # Secure defaults apply when file is empty
        assert config.fail_mode == "closed"
        assert config.on_content_failure == "block"
        assert config.categories == {}

    def test_only_version_key(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, 'version: "1"\n')
        config = PolicyLoader(str(p)).config
        assert config.fail_mode == "closed"
        assert config.memory_budget_mb == 1024


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema-invalid fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaInvalidFallback:
    """
    Load a valid file first, then replace with invalid YAML, verify that
    the last-valid config is retained and reload_if_changed returns False.
    """

    def test_invalid_fail_mode_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        # Overwrite with invalid value
        _write(p, "fail_mode: totally_wrong\n")
        result = loader.reload_if_changed()

        assert result is False, "reload_if_changed() should return False on schema error"
        assert loader.config.fail_mode == "open", (
            "Last-valid config must be retained after schema error"
        )

    def test_invalid_memory_budget_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "memory_budget_mb: 512\n")
        loader = PolicyLoader(str(p))
        assert loader.config.memory_budget_mb == 512

        _write(p, "memory_budget_mb: 64\n")  # < 128 minimum
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.memory_budget_mb == 512

    def test_invalid_rehydrate_type_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "rehydrate: false\n")
        loader = PolicyLoader(str(p))
        assert loader.config.rehydrate is False

        _write(p, 'rehydrate: "yes"\n')  # should be bool, not string
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.rehydrate is False

    def test_invalid_category_action_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\ncategories:\n  EMAIL:\n    action: allow\n")
        loader = PolicyLoader(str(p))
        assert loader.config.categories["EMAIL"].action == "allow"

        _write(p, "fail_mode: open\ncategories:\n  EMAIL:\n    action: INVALID\n")
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.categories["EMAIL"].action == "allow"

    def test_bad_allowlist_regex_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        # Invalid regex in allowlist: '[unclosed' is missing the closing bracket
        # YAML single-quoted to avoid double-quote escape issues
        _write(p, "fail_mode: open\nallowlist:\n  - '[unclosed'\n")
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.fail_mode == "open"

    def test_yaml_parse_error_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        # Syntactically invalid YAML
        _write(p, "fail_mode: {unclosed_brace\n")
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.fail_mode == "open"

    def test_categories_not_a_mapping_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        _write(p, "fail_mode: open\ncategories:\n  - EMAIL\n  - PHONE\n")
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.fail_mode == "open"

    def test_allowlist_not_a_list_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        _write(p, "fail_mode: open\nallowlist: not_a_list\n")
        result = loader.reload_if_changed()

        assert result is False

    def test_pin_list_invalid_action_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        _write(p, (
            "fail_mode: open\n"
            "pin_list:\n"
            "  - hash: sha256:abc\n"
            "    category: EMAIL\n"
            "    action: INVALID_ACTION\n"
            "pin_list_approved: true\n"
        ))
        result = loader.reload_if_changed()

        assert result is False

    def test_channel_override_not_a_mapping_retains_last_valid(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        _write(p, "fail_mode: open\nchannel_overrides:\n  cli: not_a_dict\n")
        result = loader.reload_if_changed()

        assert result is False

    def test_memory_budget_as_bool_retains_last_valid(self, tmp_path):
        """Booleans are ints in Python — ensure we reject them for memory_budget_mb."""
        p = tmp_path / "policy.yaml"
        _write(p, "memory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))

        _write(p, "memory_budget_mb: true\n")
        result = loader.reload_if_changed()

        assert result is False
        assert loader.config.memory_budget_mb == 256


# ─────────────────────────────────────────────────────────────────────────────
# 5. File deletion after valid load
# ─────────────────────────────────────────────────────────────────────────────

class TestFileDeletion:
    def test_file_deleted_reverts_to_secure_defaults(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        # Delete the file
        p.unlink()
        result = loader.reload_if_changed()

        assert result is True
        assert loader.config.fail_mode == "closed", (
            "After file deletion, must revert to SECURE_DEFAULTS (fail_mode=closed)"
        )
        assert loader.config is SECURE_DEFAULTS

    def test_file_deleted_then_recreated_reloads(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        p.unlink()
        loader.reload_if_changed()
        assert loader.config.fail_mode == "closed"

        # Re-create the file
        _write(p, "fail_mode: open\non_content_failure: warn_allow\n")
        result = loader.reload_if_changed()

        assert result is True
        assert loader.config.fail_mode == "open"
        assert loader.config.on_content_failure == "warn_allow"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hot-reload mtime detection
# ─────────────────────────────────────────────────────────────────────────────

class TestHotReload:
    def test_reload_returns_false_when_file_unchanged(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        # Immediately reload without touching the file
        result = loader.reload_if_changed()
        assert result is False

    def test_reload_returns_true_when_file_changed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        # Modify the file (advancing mtime)
        _write(p, "fail_mode: closed\n")
        result = loader.reload_if_changed()

        assert result is True
        assert loader.config.fail_mode == "closed"

    def test_hot_reload_updates_all_fields(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\nmemory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"
        assert loader.config.memory_budget_mb == 256

        _write(p, "fail_mode: closed\nmemory_budget_mb: 768\nrehydrate: false\n")
        loader.reload_if_changed()

        assert loader.config.fail_mode == "closed"
        assert loader.config.memory_budget_mb == 768
        assert loader.config.rehydrate is False

    def test_reload_is_none_path_safe(self):
        loader = PolicyLoader(None)
        result = loader.reload_if_changed()
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-category overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoryOverrides:
    def test_action_only(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "categories:\n  EMAIL:\n    action: allow\n")
        config = PolicyLoader(str(p)).config
        cp = config.categories["EMAIL"]
        assert cp.action == "allow"
        assert cp.mask_style is None
        assert cp.min_confidence is None

    def test_full_category_spec(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "categories:\n"
            "  CARD:\n"
            "    action: block\n"
            "    mask_style: format_preserving\n"
            "    min_confidence: 0.85\n"
            "    stage2_fail_action: block\n"
        ))
        config = PolicyLoader(str(p)).config
        cp = config.categories["CARD"]
        assert cp.action == "block"
        assert cp.mask_style == "format_preserving"
        assert cp.min_confidence == pytest.approx(0.85)
        assert cp.stage2_fail_action == "block"

    def test_min_confidence_as_int(self, tmp_path):
        """YAML integers should be accepted for min_confidence."""
        p = tmp_path / "policy.yaml"
        _write(p, "categories:\n  EMAIL:\n    min_confidence: 1\n")
        config = PolicyLoader(str(p)).config
        assert config.categories["EMAIL"].min_confidence == pytest.approx(1.0)

    def test_multiple_categories(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "categories:\n"
            "  EMAIL:\n    action: allow\n"
            "  API_KEY:\n    action: block\n"
            "  PHONE:\n    action: tokenize_roundtrip\n"
        ))
        config = PolicyLoader(str(p)).config
        assert config.categories["EMAIL"].action == "allow"
        assert config.categories["API_KEY"].action == "block"
        assert config.categories["PHONE"].action == "tokenize_roundtrip"

    def test_invalid_mask_style_raises(self, tmp_path):
        """Bad mask_style inside a category → ValueError at parse time."""
        raw = "categories:\n  EMAIL:\n    mask_style: invisible\n"
        with pytest.raises(ValueError, match="mask_style"):
            _parse_and_validate(raw, "<test>")

    def test_out_of_range_min_confidence_raises(self, tmp_path):
        raw = "categories:\n  EMAIL:\n    min_confidence: 1.5\n"
        with pytest.raises(ValueError, match="min_confidence"):
            _parse_and_validate(raw, "<test>")

    def test_negative_min_confidence_raises(self, tmp_path):
        raw = "categories:\n  EMAIL:\n    min_confidence: -0.1\n"
        with pytest.raises(ValueError, match="min_confidence"):
            _parse_and_validate(raw, "<test>")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Allowlist
# ─────────────────────────────────────────────────────────────────────────────

class TestAllowlist:
    def test_string_shorthand(self, tmp_path):
        p = tmp_path / "policy.yaml"
        # Use YAML unquoted or single-quoted to avoid double-quoted escape issues
        _write(p, "allowlist:\n  - test@corp[.]io\n")
        config = PolicyLoader(str(p)).config
        assert len(config.allowlist) == 1
        assert config.allowlist[0].pattern == "test@corp[.]io"
        assert config.allowlist[0].label == ""

    def test_dict_form_with_label(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "allowlist:\n"
            "  - pattern: noreply@example[.]com\n"
            "    label: internal relay\n"
        ))
        config = PolicyLoader(str(p)).config
        assert config.allowlist[0].label == "internal relay"

    def test_compiled_pattern_matches(self, tmp_path):
        p = tmp_path / "policy.yaml"
        # Use a simple pattern to avoid YAML escape pitfalls
        _write(p, "allowlist:\n  - pattern: test@corp[.]io\n")
        config = PolicyLoader(str(p)).config
        patterns = config.allowlist_patterns()
        assert patterns[0].search("test@corp.io")
        assert not patterns[0].search("other@example.com")

    def test_invalid_regex_raises_value_error(self):
        raw = "allowlist:\n  - '[invalid regex'\n"
        with pytest.raises(ValueError, match="invalid regex"):
            _parse_and_validate(raw, "<test>")

    def test_missing_pattern_key_raises(self):
        raw = "allowlist:\n  - label: oops\n"
        with pytest.raises(ValueError, match="pattern"):
            _parse_and_validate(raw, "<test>")

    def test_allowlist_entry_not_string_or_dict_raises(self):
        raw = "allowlist:\n  - 42\n"
        with pytest.raises(ValueError):
            _parse_and_validate(raw, "<test>")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Pin-list
# ─────────────────────────────────────────────────────────────────────────────

class TestPinList:
    def test_valid_pin_list_loaded(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:abc123\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "    label: relay\n"
            "pin_list_approved: true\n"
        ))
        config = PolicyLoader(str(p)).config
        assert len(config.pin_list) == 1
        e = config.pin_list[0]
        assert e.hash == "sha256:abc123"
        assert e.category == "EMAIL"
        assert e.action == "allow"
        assert e.label == "relay"

    def test_pin_list_change_without_approval_retains_old(self, tmp_path):
        """
        If pin-list changes but pin_list_approved is False (or absent), the
        old pin-list is retained and a warning is logged.
        """
        p = tmp_path / "policy.yaml"
        # First load: pin-list with one entry, approved
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

        # Second load: pin-list changed, but pin_list_approved NOT set / false
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:new_entry\n"
            "    category: EMAIL\n"
            "    action: block\n"
            "pin_list_approved: false\n"
        ))
        loader.reload_if_changed()

        # Old pin-list must be retained
        assert loader.config.pin_list[0].hash == "sha256:original"

    def test_pin_list_change_with_approval_accepted(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:old\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            "pin_list_approved: true\n"
        ))
        loader = PolicyLoader(str(p))

        _write(p, (
            "pin_list:\n"
            "  - hash: sha256:new_approved\n"
            "    category: PHONE\n"
            "    action: block\n"
            "pin_list_approved: true\n"
        ))
        loader.reload_if_changed()

        assert loader.config.pin_list[0].hash == "sha256:new_approved"
        assert loader.config.pin_list[0].category == "PHONE"

    def test_hash_pin_list_stable(self):
        entries = [
            PinListEntry(hash="abc", category="EMAIL", action="allow"),
            PinListEntry(hash="def", category="PHONE", action="block"),
        ]
        h1 = _hash_pin_list(entries)
        h2 = _hash_pin_list(entries[::-1])  # reversed order
        assert h1 == h2, "Hash must be order-independent"

    def test_empty_pin_list_hash(self):
        assert _hash_pin_list([]) == _hash_pin_list([])

    def test_pin_list_invalid_action(self):
        raw = (
            "pin_list:\n"
            "  - hash: abc\n"
            "    category: EMAIL\n"
            "    action: NOPE\n"
            "pin_list_approved: true\n"
        )
        with pytest.raises(ValueError, match="action"):
            _parse_and_validate(raw, "<test>")

    def test_pin_list_missing_hash(self):
        raw = (
            "pin_list:\n"
            "  - category: EMAIL\n"
            "    action: allow\n"
        )
        with pytest.raises(ValueError, match="hash"):
            _parse_and_validate(raw, "<test>")

    def test_pin_list_missing_category(self):
        raw = (
            "pin_list:\n"
            "  - hash: abc\n"
            "    action: allow\n"
        )
        with pytest.raises(ValueError, match="category"):
            _parse_and_validate(raw, "<test>")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Channel overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelOverrides:
    def test_partial_channel_override(self, tmp_path):
        """Only one field overridden — others are None (use global setting)."""
        p = tmp_path / "policy.yaml"
        _write(p, (
            "channel_overrides:\n"
            "  cli:\n"
            "    unscannable_action: warn_allow\n"
        ))
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("cli", "unscannable_action") == "warn_allow"
        assert config.channel_setting("cli", "fail_mode") is None

    def test_multiple_channel_overrides(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, (
            "channel_overrides:\n"
            "  cli:\n"
            "    unscannable_action: warn_allow\n"
            "  ouroboros:\n"
            "    fail_mode: closed\n"
            "    stage2_fail_action: block\n"
        ))
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("cli", "unscannable_action") == "warn_allow"
        assert config.channel_setting("ouroboros", "fail_mode") == "closed"
        assert config.channel_setting("ouroboros", "stage2_fail_action") == "block"

    def test_invalid_channel_unscannable_action_raises(self):
        raw = "channel_overrides:\n  cli:\n    unscannable_action: invalid\n"
        with pytest.raises(ValueError):
            _parse_and_validate(raw, "<test>")

    def test_invalid_channel_fail_mode_raises(self):
        raw = "channel_overrides:\n  ouroboros:\n    fail_mode: maybe\n"
        with pytest.raises(ValueError):
            _parse_and_validate(raw, "<test>")

    def test_unknown_channel_setting_returns_none(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "channel_overrides:\n  cli:\n    unscannable_action: warn_allow\n")
        config = PolicyLoader(str(p)).config
        assert config.channel_setting("cli", "nonexistent_field") is None


# ─────────────────────────────────────────────────────────────────────────────
# 11. Watcher thread
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherThread:
    def test_watcher_starts_and_stops(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.start_watcher(interval=0.05)
        assert loader._watcher_thread is not None
        assert loader._watcher_thread.is_alive()
        loader.stop_watcher()
        # Thread should have stopped
        assert not loader._watcher_thread.is_alive() if loader._watcher_thread else True

    def test_watcher_picks_up_file_change(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.start_watcher(interval=0.05)

        try:
            assert loader.config.fail_mode == "open"
            _write(p, "fail_mode: closed\n")

            # Wait for the watcher to pick up the change (up to 1 second)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if loader.config.fail_mode == "closed":
                    break
                time.sleep(0.05)

            assert loader.config.fail_mode == "closed", (
                "Watcher thread should have picked up the file change"
            )
        finally:
            loader.stop_watcher()

    def test_double_start_is_noop(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.start_watcher(interval=0.05)
        first_thread = loader._watcher_thread
        loader.start_watcher(interval=0.05)  # should not create a new thread
        assert loader._watcher_thread is first_thread
        loader.stop_watcher()

    def test_stop_without_start_is_safe(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.stop_watcher()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 12. load_policy() convenience function
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPolicyConvenience:
    def test_none_path_returns_secure_defaults(self):
        config = load_policy(None)
        assert config is SECURE_DEFAULTS

    def test_existing_file_parsed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        config = load_policy(str(p))
        assert config.fail_mode == "open"

    def test_missing_file_returns_secure_defaults(self, tmp_path):
        config = load_policy(str(tmp_path / "nonexistent.yaml"))
        assert config.fail_mode == "closed"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Unknown top-level keys
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownKeys:
    def test_unknown_key_warned_not_fatal(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\nmy_custom_field: 42\n")
        config = PolicyLoader(str(p)).config
        # Should load successfully
        assert config.fail_mode == "open"

    def test_unknown_key_produces_warning(self):
        raw = "fail_mode: open\nmy_custom_field: 42\n"
        config, warnings = _parse_and_validate(raw, "<test>")
        assert any("my_custom_field" in w for w in warnings)

    def test_version_key_not_warned(self):
        raw = 'version: "1"\nfail_mode: open\n'
        config, warnings = _parse_and_validate(raw, "<test>")
        assert not any("version" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Secure-by-default: deleting the file must not open the gateway
# ─────────────────────────────────────────────────────────────────────────────

class TestSecureByDefault:
    """
    Regression: deleting the policy file must NEVER leave the system in an
    unprotected (open) state.  SECURE_DEFAULTS are maximally restrictive.
    """

    def test_no_policy_file_blocks_by_default(self):
        loader = PolicyLoader(None)
        cfg = loader.config
        # Fail-closed
        assert cfg.fail_mode == "closed"
        assert cfg.on_content_failure == "block"
        # Unscannable → blocked
        assert cfg.unscannable_action == "block"
        # No allowlist holes
        assert cfg.allowlist == []
        # No pin-list bypasses
        assert cfg.pin_list == []

    def test_deleted_file_restores_fail_closed(self, tmp_path):
        p = tmp_path / "policy.yaml"
        # Start with an open (non-default) policy
        _write(p, "fail_mode: open\non_content_failure: warn_allow\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        p.unlink()
        loader.reload_if_changed()

        cfg = loader.config
        assert cfg.fail_mode == "closed"
        assert cfg.on_content_failure == "block"
        assert cfg.unscannable_action == "block"
