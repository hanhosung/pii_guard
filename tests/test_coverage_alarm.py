"""
Unit tests for coverage alarm emission and action enforcement — Sub-AC 3b.

Design: tests inject **pre-built** FieldDelta / VersionDelta objects and
assert that:
  1. A CoverageAlarmEvent is emitted for each non-empty delta.
  2. The resulting action (block vs. warn) matches the active mode.
  3. Clean (empty) deltas produce no alarm.
  4. The aggregate should_block on CoverageAlarmResult is the OR of
     individual alarms.
  5. CoverageAlarmEvent fields carry correct metadata.
  6. Both FieldDelta and VersionDelta produce well-formed events.
  7. Mixed deltas (field + version) in one call are all processed.
  8. as_log_dict() never exposes source_delta or raw user data.
  9. Unknown/unexpected delta types do not raise.
 10. apply_coverage_alarm_policy and emit_coverage_alarms are consistent.

No live schema-comparison logic is called.  All delta objects are hand-built.
"""
from __future__ import annotations

import pytest

from pii_guard.providers.coverage_alarm import (
    CoverageAlarmEvent,
    CoverageAlarmResult,
    apply_coverage_alarm_policy,
    emit_coverage_alarms,
)
from pii_guard.providers.schema_coverage import FieldDelta, VersionDelta


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — pre-built delta factories (no live schema-comparison logic)
# ─────────────────────────────────────────────────────────────────────────────

def _field_delta(
    provider: str = "claude",
    path: str = "",
    extra_keys: frozenset = frozenset({"x_novel_field"}),
    known_keys: frozenset = frozenset({"model", "messages"}),
) -> FieldDelta:
    """Build a FieldDelta with a non-empty extra_keys set by default."""
    actual_keys = known_keys | extra_keys
    return FieldDelta(
        provider=provider,
        path=path,
        extra_keys=extra_keys,
        known_keys=known_keys,
        actual_keys=actual_keys,
    )


def _clean_field_delta(
    provider: str = "claude",
    path: str = "",
) -> FieldDelta:
    """Build a FieldDelta with no extra keys (clean / no-alarm)."""
    known = frozenset({"model", "messages"})
    return FieldDelta(
        provider=provider,
        path=path,
        extra_keys=frozenset(),
        known_keys=known,
        actual_keys=known,
    )


def _version_delta(
    provider: str = "claude",
    declared_version: str = "2099-01-01",
    is_future: bool = True,
    is_unknown: bool = True,
    location: str = "header:anthropic-version",
) -> VersionDelta:
    """Build a VersionDelta for an unknown version by default."""
    return VersionDelta(
        provider=provider,
        declared_version=declared_version,
        known_versions=("2023-06-01",),
        is_future=is_future,
        is_unknown=is_unknown,
        location=location,
    )


def _known_version_delta(provider: str = "claude") -> VersionDelta:
    """Build a VersionDelta for a known (good) version — should produce no alarm."""
    return VersionDelta(
        provider=provider,
        declared_version="2023-06-01",
        known_versions=("2023-06-01",),
        is_future=False,
        is_unknown=False,
        location="header:anthropic-version",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic alarm emission — FieldDelta
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldDeltaAlarmEmission:
    """Non-empty FieldDelta → exactly one CoverageAlarmEvent emitted."""

    def test_single_field_delta_emits_one_alarm(self):
        delta = _field_delta(extra_keys=frozenset({"x_novel"}))
        alarms = emit_coverage_alarms([delta], unknown_field_action="block")
        assert len(alarms) == 1

    def test_alarm_type_is_unknown_fields(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.alarm_type == "unknown_fields"

    def test_alarm_provider_matches_delta(self):
        delta = _field_delta(provider="openai")
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.provider == "openai"

    def test_alarm_path_matches_delta(self):
        delta = _field_delta(path="messages[0].content[1]")
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.path == "messages[0].content[1]"

    def test_alarm_extra_keys_match_delta(self):
        keys = frozenset({"x_field_a", "x_field_b"})
        delta = _field_delta(extra_keys=keys)
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.extra_keys == keys

    def test_field_alarm_has_no_declared_version(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.declared_version is None
        assert alarm.is_future is False

    def test_source_delta_is_stored(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.source_delta is delta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic alarm emission — VersionDelta
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionDeltaAlarmEmission:
    """Unknown VersionDelta → exactly one CoverageAlarmEvent emitted."""

    def test_unknown_version_emits_one_alarm(self):
        delta = _version_delta(is_unknown=True)
        alarms = emit_coverage_alarms([delta], unknown_field_action="block")
        assert len(alarms) == 1

    def test_alarm_type_is_unknown_version(self):
        delta = _version_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.alarm_type == "unknown_version"

    def test_alarm_provider_matches_delta(self):
        delta = _version_delta(provider="gemini")
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.provider == "gemini"

    def test_alarm_path_is_version_location(self):
        delta = _version_delta(location="path:/v2beta/")
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.path == "path:/v2beta/"

    def test_alarm_declared_version_matches_delta(self):
        delta = _version_delta(declared_version="2099-01-01")
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.declared_version == "2099-01-01"

    def test_alarm_is_future_matches_delta(self):
        delta = _version_delta(is_future=True)
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.is_future is True

    def test_alarm_is_future_false_when_not_future(self):
        delta = _version_delta(is_future=False, is_unknown=True)
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.is_future is False

    def test_version_alarm_extra_keys_is_empty(self):
        delta = _version_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.extra_keys == frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action enforcement — STRICT mode ("block")
# ─────────────────────────────────────────────────────────────────────────────

class TestStrictModeBlocking:
    """In block mode every emitted alarm must have should_block=True."""

    def test_field_alarm_should_block_true_in_strict(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.should_block is True

    def test_version_alarm_should_block_true_in_strict(self):
        delta = _version_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.should_block is True

    def test_result_should_block_true_when_field_delta(self):
        delta = _field_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert result.should_block is True

    def test_result_should_block_true_when_version_delta(self):
        delta = _version_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert result.should_block is True

    def test_result_should_block_true_when_multiple_deltas(self):
        deltas = [
            _field_delta(provider="claude", path=""),
            _version_delta(provider="claude"),
        ]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert result.should_block is True

    def test_default_action_is_block(self):
        """unknown_field_action defaults to 'block' (strict by default)."""
        delta = _field_delta()
        result = apply_coverage_alarm_policy([delta])  # no explicit action arg
        assert result.should_block is True
        assert result.unknown_field_action == "block"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Action enforcement — PERMISSIVE mode ("warn_allow")
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissiveModeWarnAllow:
    """In warn_allow mode alarms are emitted but should_block=False throughout."""

    def test_field_alarm_should_block_false_in_permissive(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="warn_allow")[0]
        assert alarm.should_block is False

    def test_version_alarm_should_block_false_in_permissive(self):
        delta = _version_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="warn_allow")[0]
        assert alarm.should_block is False

    def test_result_should_block_false_when_field_delta(self):
        delta = _field_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="warn_allow")
        assert result.should_block is False

    def test_result_should_block_false_when_version_delta(self):
        delta = _version_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="warn_allow")
        assert result.should_block is False

    def test_alarms_still_emitted_in_permissive(self):
        """Alarms must be emitted even when warn_allow — they go to the ledger."""
        delta = _field_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="warn_allow")
        assert len(result.alarms) == 1

    def test_result_action_recorded(self):
        delta = _field_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="warn_allow")
        assert result.unknown_field_action == "warn_allow"

    def test_multiple_deltas_all_warn_in_permissive(self):
        deltas = [_field_delta(), _version_delta()]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="warn_allow")
        assert result.should_block is False
        assert len(result.alarms) == 2
        assert all(not a.should_block for a in result.alarms)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clean deltas produce no alarm (no false positives)
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanDeltasNoAlarm:
    """Empty extra_keys / known version → no alarm emitted."""

    def test_empty_field_delta_no_alarm_strict(self):
        delta = _clean_field_delta()
        alarms = emit_coverage_alarms([delta], unknown_field_action="block")
        assert alarms == []

    def test_empty_field_delta_no_alarm_permissive(self):
        delta = _clean_field_delta()
        alarms = emit_coverage_alarms([delta], unknown_field_action="warn_allow")
        assert alarms == []

    def test_known_version_delta_no_alarm(self):
        delta = _known_version_delta()
        alarms = emit_coverage_alarms([delta], unknown_field_action="block")
        assert alarms == []

    def test_known_version_no_block(self):
        delta = _known_version_delta()
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert result.should_block is False
        assert result.alarms == []

    def test_empty_delta_list_no_alarm(self):
        result = apply_coverage_alarm_policy([], unknown_field_action="block")
        assert result.alarms == []
        assert result.should_block is False

    def test_mixed_clean_and_noisy_produces_only_noisy_alarm(self):
        """One clean + one noisy delta → exactly one alarm from the noisy one."""
        deltas = [
            _clean_field_delta(provider="openai", path=""),
            _field_delta(provider="openai", path="messages[0]"),
        ]
        alarms = emit_coverage_alarms(deltas, unknown_field_action="block")
        assert len(alarms) == 1
        assert alarms[0].path == "messages[0]"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mixed delta types in one call
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedDeltaTypes:
    """FieldDelta and VersionDelta together are both consumed correctly."""

    def test_one_field_one_version_both_emit(self):
        deltas: list = [
            _field_delta(provider="gemini", path=""),
            _version_delta(provider="gemini"),
        ]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert len(result.alarms) == 2
        alarm_types = {a.alarm_type for a in result.alarms}
        assert alarm_types == {"unknown_fields", "unknown_version"}

    def test_field_and_version_both_block_in_strict(self):
        deltas: list = [_field_delta(), _version_delta()]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert result.should_block is True
        assert all(a.should_block for a in result.alarms)

    def test_field_and_version_no_block_in_permissive(self):
        deltas: list = [_field_delta(), _version_delta()]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="warn_allow")
        assert result.should_block is False

    def test_multiple_fields_multiple_alarms(self):
        deltas: list = [
            _field_delta(provider="openai", path=""),
            _field_delta(provider="openai", path="messages[0]"),
            _field_delta(provider="openai", path="messages[0].content[2]"),
        ]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert len(result.alarms) == 3
        assert result.should_block is True

    def test_providers_across_claude_openai_gemini(self):
        """Alarms carry the correct provider string from their delta."""
        deltas: list = [
            _field_delta(provider="claude"),
            _field_delta(provider="openai"),
            _field_delta(provider="gemini"),
        ]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="warn_allow")
        providers = {a.provider for a in result.alarms}
        assert providers == {"claude", "openai", "gemini"}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Aggregate should_block semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateShouldBlock:
    """should_block on CoverageAlarmResult is the OR of individual alarms."""

    def test_no_alarms_no_block(self):
        result = apply_coverage_alarm_policy([], unknown_field_action="block")
        assert result.should_block is False

    def test_single_alarm_blocking(self):
        result = apply_coverage_alarm_policy([_field_delta()], unknown_field_action="block")
        assert result.should_block is True

    def test_single_alarm_warn_allow(self):
        result = apply_coverage_alarm_policy([_field_delta()], unknown_field_action="warn_allow")
        assert result.should_block is False

    def test_clean_plus_clean_no_block(self):
        """All clean deltas → no alarms → no block even in strict mode."""
        deltas: list = [_clean_field_delta(), _known_version_delta()]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert result.should_block is False
        assert result.alarms == []

    def test_many_warn_allow_plus_one_block_is_still_block(self):
        """
        If some alarms are warn_allow but one is blocking, the aggregate
        should_block is True.  (This scenario arises when unknown_field_action
        is 'block' — all alarms in a single call share the same mode.)
        """
        # In practice all alarms in one call share the same action, but we
        # test the OR semantics explicitly by constructing the result manually.
        alarms = [
            CoverageAlarmEvent(
                alarm_type="unknown_fields",
                provider="claude",
                path="",
                extra_keys=frozenset({"x"}),
                declared_version=None,
                is_future=False,
                should_block=False,
            ),
            CoverageAlarmEvent(
                alarm_type="unknown_version",
                provider="claude",
                path="header:anthropic-version",
                extra_keys=frozenset(),
                declared_version="2099-01-01",
                is_future=True,
                should_block=True,
            ),
        ]
        result = CoverageAlarmResult(
            alarms=alarms,
            should_block=any(a.should_block for a in alarms),
            unknown_field_action="block",
        )
        assert result.should_block is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. as_log_dict() — ledger safety
# ─────────────────────────────────────────────────────────────────────────────

class TestAsLogDict:
    """as_log_dict() must not expose source_delta or anything unsafe."""

    def test_field_alarm_log_dict_has_required_keys(self):
        delta = _field_delta(
            provider="claude",
            path="messages[0]",
            extra_keys=frozenset({"x_novel"}),
        )
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        assert d["alarm_type"] == "unknown_fields"
        assert d["provider"] == "claude"
        assert d["path"] == "messages[0]"
        assert d["should_block"] is True
        assert "x_novel" in d["extra_keys"]

    def test_version_alarm_log_dict_has_required_keys(self):
        delta = _version_delta(
            provider="openai",
            declared_version="v9",
            is_future=True,
            location="path:/v9/",
        )
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        assert d["alarm_type"] == "unknown_version"
        assert d["provider"] == "openai"
        assert d["path"] == "path:/v9/"
        assert d["declared_version"] == "v9"
        assert d["is_future"] is True
        assert d["should_block"] is True

    def test_log_dict_does_not_contain_source_delta(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        assert "source_delta" not in d

    def test_log_dict_field_alarm_no_version_keys(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        assert "declared_version" not in d
        assert "is_future" not in d

    def test_log_dict_version_alarm_no_extra_keys_field(self):
        """Version alarms have no extra_keys — the key should be absent."""
        delta = _version_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        # extra_keys is empty frozenset — absent from log dict
        assert "extra_keys" not in d

    def test_log_dict_warn_allow_mode(self):
        delta = _field_delta()
        alarm = emit_coverage_alarms([delta], unknown_field_action="warn_allow")[0]
        d = alarm.as_log_dict()
        assert d["should_block"] is False

    def test_extra_keys_sorted_for_determinism(self):
        delta = _field_delta(extra_keys=frozenset({"z_field", "a_field", "m_field"}))
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        d = alarm.as_log_dict()
        assert d["extra_keys"] == sorted(["z_field", "a_field", "m_field"])


# ─────────────────────────────────────────────────────────────────────────────
# 9. Unknown delta types — forward-compatibility robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownDeltaTypeRobustness:
    """Objects that are neither FieldDelta nor VersionDelta are silently skipped."""

    def test_non_delta_object_does_not_raise(self):
        """Passing an unexpected object must not raise an exception."""
        # Simulate a future delta type the module doesn't know about yet
        weird_delta = object()
        alarms = emit_coverage_alarms(
            [weird_delta],  # type: ignore[list-item]
            unknown_field_action="block",
        )
        assert alarms == []

    def test_non_delta_mixed_with_real_delta(self):
        """Real deltas in the same list still produce alarms despite the bad object."""
        weird_delta = {"not": "a real delta"}
        real_delta = _field_delta()
        alarms = emit_coverage_alarms(
            [weird_delta, real_delta],  # type: ignore[list-item]
            unknown_field_action="block",
        )
        assert len(alarms) == 1
        assert alarms[0].alarm_type == "unknown_fields"

    def test_none_in_delta_list_skipped(self):
        alarms = emit_coverage_alarms(
            [None],  # type: ignore[list-item]
            unknown_field_action="block",
        )
        assert alarms == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. emit_coverage_alarms and apply_coverage_alarm_policy are consistent
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistency:
    """The two public functions agree on alarms and should_block."""

    def test_emit_and_apply_same_alarms_strict(self):
        deltas: list = [_field_delta(), _version_delta()]
        alarms = emit_coverage_alarms(deltas, unknown_field_action="block")
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        assert len(alarms) == len(result.alarms)
        # Same alarm_type+provider+path pairs
        emit_keys = {(a.alarm_type, a.provider, a.path) for a in alarms}
        apply_keys = {(a.alarm_type, a.provider, a.path) for a in result.alarms}
        assert emit_keys == apply_keys

    def test_emit_and_apply_same_alarms_permissive(self):
        deltas: list = [_field_delta(), _version_delta()]
        alarms = emit_coverage_alarms(deltas, unknown_field_action="warn_allow")
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="warn_allow")
        assert len(alarms) == len(result.alarms)

    def test_apply_should_block_is_or_of_individual_alarms(self):
        deltas: list = [_field_delta(path=""), _field_delta(path="messages[0]")]
        result = apply_coverage_alarm_policy(deltas, unknown_field_action="block")
        expected_block = any(a.should_block for a in result.alarms)
        assert result.should_block == expected_block


# ─────────────────────────────────────────────────────────────────────────────
# 11. Provider coverage — all three supported providers
# ─────────────────────────────────────────────────────────────────────────────

class TestAllProviders:
    """Coverage alarms work identically for claude, openai, and gemini."""

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_field_alarm_emitted_for_provider(self, provider: str):
        delta = _field_delta(provider=provider)
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert len(result.alarms) == 1
        assert result.alarms[0].provider == provider
        assert result.should_block is True

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_version_alarm_emitted_for_provider(self, provider: str):
        delta = _version_delta(provider=provider)
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert len(result.alarms) == 1
        assert result.alarms[0].provider == provider
        assert result.should_block is True

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_clean_delta_no_alarm_for_provider(self, provider: str):
        delta = _clean_field_delta(provider=provider)
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert result.alarms == []
        assert result.should_block is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. Structural / path variety
# ─────────────────────────────────────────────────────────────────────────────

class TestPathVariety:
    """Alarms at various structural depths carry the correct path."""

    @pytest.mark.parametrize("path", [
        "",
        "messages[0]",
        "messages[3].content[1]",
        "tools[0]",
        "tool_choice",
        "contents[0].parts[2]",
        "systemInstruction",
        "generationConfig",
    ])
    def test_alarm_path_preserved(self, path: str):
        delta = _field_delta(path=path)
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.path == path


# ─────────────────────────────────────────────────────────────────────────────
# 13. Edge cases — multiple extra keys in one FieldDelta
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleExtraKeysInOneDelta:
    """One FieldDelta with several extra keys → one alarm listing all keys."""

    def test_one_alarm_per_delta_not_per_key(self):
        keys = frozenset({"a", "b", "c", "d"})
        delta = _field_delta(extra_keys=keys)
        alarms = emit_coverage_alarms([delta], unknown_field_action="block")
        # One alarm, not four
        assert len(alarms) == 1

    def test_all_extra_keys_in_single_alarm(self):
        keys = frozenset({"foo", "bar", "baz"})
        delta = _field_delta(extra_keys=keys)
        alarm = emit_coverage_alarms([delta], unknown_field_action="block")[0]
        assert alarm.extra_keys == keys


# ─────────────────────────────────────────────────────────────────────────────
# 14. Version delta — unrecognised non-future version
# ─────────────────────────────────────────────────────────────────────────────

class TestUnrecognisedNonFutureVersion:
    """An unrecognised version that doesn't look 'future' still raises an alarm."""

    def test_unrecognised_malformed_version_alarm(self):
        delta = VersionDelta(
            provider="claude",
            declared_version="not-a-date",
            known_versions=("2023-06-01",),
            is_future=False,
            is_unknown=True,        # unknown but not future (malformed)
            location="header:anthropic-version",
        )
        result = apply_coverage_alarm_policy([delta], unknown_field_action="block")
        assert len(result.alarms) == 1
        assert result.alarms[0].is_future is False
        assert result.should_block is True

    def test_unrecognised_non_future_warn_allow(self):
        delta = VersionDelta(
            provider="openai",
            declared_version="v0",          # lower than v1 — not future, but unknown
            known_versions=("v1",),
            is_future=False,
            is_unknown=True,
            location="path:/v0/",
        )
        result = apply_coverage_alarm_policy([delta], unknown_field_action="warn_allow")
        assert len(result.alarms) == 1
        assert result.should_block is False
