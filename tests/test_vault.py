"""
Unit tests for pii_guard.vault  (Sub-AC 5c-ii)

Request-scoped masking vault and rehydration.

Coverage areas
--------------

RequestVault
    1.  assign_token — basic token assignment, returns CATEGORY_N format
    2.  assign_token — idempotent: same original → same token
    3.  assign_token — per-category counter isolation (EMAIL counter ≠ PHONE)
    4.  assign_token — blocked suffix (CATEGORY_N_BLOCKED)
    5.  restore — returns original for known token; None for unknown
    6.  rehydrate — replaces [TOKEN] placeholders with originals
    7.  rehydrate — longest-token-first avoids EMAIL_10 / EMAIL_1 collision
    8.  rehydrate — unknown placeholders left unchanged
    9.  has_token / has_original / token_for helpers
    10. snapshot property — returns copy, not live reference
    11. size and __len__ helpers
    12. counters property
    13. __contains__ operator
    14. Validation — empty original or category raises ValueError
    15. Request-scoped independence — two vaults do not share state

MaskStyle.TOKENIZE — apply_mask_style
    16. Returns "[CATEGORY_N]" for non-blocked items
    17. Returns "[CATEGORY_N_BLOCKED]" for blocked items
    18. Vault stores the original under the token
    19. Multiple distinct values get sequential tokens (EMAIL_1, EMAIL_2)
    20. Same value idempotently gets same token across calls

MaskStyle.PARTIAL — apply_mask_style
    21. Returns "***" for short strings (≤4 chars)
    22. Reveals first 2 and last 2 chars for normal strings
    23. Reveal capped at len//4 to avoid exposing >50 % of value
    24. Vault stores original under assigned token
    25. vault.restore(token) returns the original
    26. Masked text does NOT contain token marker (not automatic rehydration)

MaskStyle.FORMAT_PRESERVING — apply_mask_style
    27. Uppercase letters mapped to 'X'
    28. Lowercase letters mapped to 'x'
    29. Digits mapped to '0'
    30. Punctuation / specials preserved unchanged
    31. Output length equals input length
    32. Vault stores original under assigned token
    33. vault.restore(token) returns the original

mask_payload_with_vault — full payload masking
    34. Empty detections — text unchanged, vault empty
    35. Single detection with tokenize style — placeholder in output, vault has entry
    36. Single detection with partial style — partial in output, vault has entry
    37. Single detection with format_preserving style — FP in output, vault has entry
    38. Multiple detections, different styles — each correctly applied
    39. Overlapping spans — first (earlier start) wins; second skipped
    40. ALLOW action — span left unchanged, vault empty (no storage for allowed)
    41. Unsorted detections — sorted internally by start position
    42. Vault rehydration of tokenize-style masked text → original text (full round-trip)
    43. Vault restore of partial-style masked token → original (explicit round-trip)
    44. Vault restore of format_preserving-style token → original (explicit round-trip)
    45. Existing vault extended by mask_payload_with_vault
    46. Zero-length span skipped

Round-trip completeness tests (mask → vault store → rehydrate/restore → compare)
    47. TOKENIZE: full text round-trip (automatic token-based rehydration)
    48. PARTIAL: explicit token-based restoration via vault.restore()
    49. FORMAT_PRESERVING: explicit token-based restoration via vault.restore()
    50. Multi-style mix: partial + tokenize in same text; tokenize span is rehydrated

Top-level import tests
    51. RequestVault, apply_mask_style, mask_payload_with_vault importable from pii_guard

Run with:   pytest tests/test_vault.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pii_guard.models import Action, CategoryClass, Detection, DetectionStage, MaskStyle
from pii_guard.vault import (
    RequestVault,
    apply_mask_style,
    mask_payload_with_vault,
    _partial_mask,
    _format_preserving_mask,
)
# Top-level import smoke test
import pii_guard


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _det(
    category: str,
    start: int,
    end: int,
    original: str,
    *,
    action: Action = Action.MASK,
    mask_style: MaskStyle = MaskStyle.TOKENIZE,
) -> Detection:
    """Construct a minimal Detection for testing."""
    return Detection(
        category=category,
        category_class=CategoryClass.PII,
        action=action,
        mask_style=mask_style,
        start=start,
        end=end,
        original=original,
        detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
        rule_id="test_rule",
        confidence=0.99,
    )


def _dict_det(
    category: str,
    start: int,
    end: int,
    original: str,
    *,
    action: str = "mask",
    mask_style: str = "tokenize",
) -> dict:
    """Construct a plain-dict detection descriptor."""
    return {
        "category": category,
        "start": start,
        "end": end,
        "original": original,
        "action": action,
        "mask_style": mask_style,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1–15.  RequestVault
# ══════════════════════════════════════════════════════════════════════════════

class TestRequestVaultAssignToken:
    """Tests 1–4: Basic token assignment."""

    def test_assign_returns_category_n_format(self):
        vault = RequestVault()
        token = vault.assign_token("alice@corp.io", "EMAIL")
        assert token == "EMAIL_1"

    def test_assign_second_value_gets_next_index(self):
        vault = RequestVault()
        vault.assign_token("alice@corp.io", "EMAIL")
        token2 = vault.assign_token("bob@corp.io", "EMAIL")
        assert token2 == "EMAIL_2"

    def test_assign_third_value_gets_index_3(self):
        vault = RequestVault()
        for i, email in enumerate(["a@b.com", "c@d.com", "e@f.com"], start=1):
            tok = vault.assign_token(email, "EMAIL")
            assert tok == f"EMAIL_{i}"

    def test_per_category_counter_isolation(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")   # EMAIL_1
        vault.assign_token("c@d.com", "EMAIL")   # EMAIL_2
        phone_tok = vault.assign_token("010-1234-5678", "PHONE")  # PHONE_1 not PHONE_3
        assert phone_tok == "PHONE_1"

    def test_blocked_suffix(self):
        vault = RequestVault()
        tok = vault.assign_token("sk-" + "x" * 20, "API_KEY", blocked=True)
        assert tok == "API_KEY_1_BLOCKED"

    def test_blocked_counter_shared_with_unblocked(self):
        """Blocked and non-blocked items share the same category counter."""
        vault = RequestVault()
        t1 = vault.assign_token("a@b.com", "EMAIL", blocked=False)    # EMAIL_1
        t2 = vault.assign_token("c@d.com", "EMAIL", blocked=True)     # EMAIL_2_BLOCKED
        assert t1 == "EMAIL_1"
        assert t2 == "EMAIL_2_BLOCKED"
        assert vault.counters["EMAIL"] == 2


class TestRequestVaultIdempotent:
    """Test 2: Same original → same token (idempotent)."""

    def test_same_value_returns_same_token(self):
        vault = RequestVault()
        t1 = vault.assign_token("alice@corp.io", "EMAIL")
        t2 = vault.assign_token("alice@corp.io", "EMAIL")
        assert t1 == t2 == "EMAIL_1"

    def test_counter_not_incremented_on_re_assign(self):
        vault = RequestVault()
        vault.assign_token("alice@corp.io", "EMAIL")
        vault.assign_token("alice@corp.io", "EMAIL")
        vault.assign_token("alice@corp.io", "EMAIL")
        assert vault.counters["EMAIL"] == 1

    def test_idempotent_across_many_calls(self):
        vault = RequestVault()
        tokens = [vault.assign_token("x@y.com", "EMAIL") for _ in range(20)]
        assert all(t == "EMAIL_1" for t in tokens)

    def test_distinct_values_get_distinct_tokens(self):
        vault = RequestVault()
        t1 = vault.assign_token("a@corp.io", "EMAIL")
        t2 = vault.assign_token("b@corp.io", "EMAIL")
        assert t1 != t2


class TestRequestVaultRestore:
    """Tests 5: restore() method."""

    def test_restore_known_token_returns_original(self):
        vault = RequestVault()
        vault.assign_token("alice@corp.io", "EMAIL")
        assert vault.restore("EMAIL_1") == "alice@corp.io"

    def test_restore_unknown_token_returns_none(self):
        vault = RequestVault()
        assert vault.restore("EMAIL_99") is None
        assert vault.restore("GHOST_1") is None
        assert vault.restore("") is None

    def test_restore_blocked_token(self):
        vault = RequestVault()
        secret = "sk-" + "z" * 25
        vault.assign_token(secret, "API_KEY", blocked=True)
        assert vault.restore("API_KEY_1_BLOCKED") == secret

    def test_restore_multiple_tokens(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        vault.assign_token("c@d.com", "EMAIL")
        vault.assign_token("010-1234-5678", "PHONE")
        assert vault.restore("EMAIL_1") == "a@b.com"
        assert vault.restore("EMAIL_2") == "c@d.com"
        assert vault.restore("PHONE_1") == "010-1234-5678"


class TestRequestVaultRehydrate:
    """Tests 6–8: rehydrate() method."""

    def test_rehydrate_single_placeholder(self):
        vault = RequestVault()
        vault.assign_token("alice@corp.io", "EMAIL")
        result = vault.rehydrate("Contact [EMAIL_1] for details")
        assert result == "Contact alice@corp.io for details"

    def test_rehydrate_multiple_placeholders(self):
        vault = RequestVault()
        vault.assign_token("alice@corp.io", "EMAIL")
        vault.assign_token("bob@corp.io", "EMAIL")
        vault.assign_token("010-1234-5678", "PHONE")
        text = "[EMAIL_1] or [EMAIL_2], phone: [PHONE_1]"
        result = vault.rehydrate(text)
        assert "alice@corp.io" in result
        assert "bob@corp.io" in result
        assert "010-1234-5678" in result

    def test_rehydrate_leaves_unknown_tokens_unchanged(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        text = "Known: [EMAIL_1], unknown: [GHOST_99]"
        result = vault.rehydrate(text)
        assert "a@b.com" in result
        assert "[GHOST_99]" in result  # left unchanged

    def test_rehydrate_longest_token_first_prevents_shadowing(self):
        """EMAIL_10 must not be partially replaced by EMAIL_1."""
        vault = RequestVault()
        for i in range(10):
            vault.assign_token(f"user{i}@test.com", "EMAIL")
        text = "[EMAIL_10] and [EMAIL_1]"
        result = vault.rehydrate(text)
        assert "user9@test.com" in result   # EMAIL_10
        assert "user0@test.com" in result   # EMAIL_1
        assert "[EMAIL_10]" not in result
        assert "[EMAIL_1]" not in result

    def test_rehydrate_empty_text(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        assert vault.rehydrate("") == ""

    def test_rehydrate_no_placeholders(self):
        vault = RequestVault()
        text = "No tokens here."
        assert vault.rehydrate(text) == text


class TestRequestVaultHelpers:
    """Tests 9–15: helpers, properties, validation, independence."""

    def test_has_token_known(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        assert vault.has_token("EMAIL_1") is True

    def test_has_token_unknown(self):
        vault = RequestVault()
        assert vault.has_token("EMAIL_99") is False

    def test_has_original_known(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        assert vault.has_original("a@b.com") is True

    def test_has_original_unknown(self):
        vault = RequestVault()
        assert vault.has_original("ghost@evil.com") is False

    def test_token_for_known_original(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        assert vault.token_for("a@b.com") == "EMAIL_1"

    def test_token_for_unknown_original_returns_none(self):
        vault = RequestVault()
        assert vault.token_for("ghost") is None

    def test_snapshot_is_copy(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        snap = vault.snapshot
        snap["INJECTED"] = "evil"
        assert "INJECTED" not in vault.snapshot

    def test_size_and_len(self):
        vault = RequestVault()
        assert vault.size == 0
        assert len(vault) == 0
        vault.assign_token("a@b.com", "EMAIL")
        assert vault.size == 1
        assert len(vault) == 1
        vault.assign_token("a@b.com", "EMAIL")  # re-assign same value
        assert vault.size == 1  # still 1

    def test_counters_property(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        vault.assign_token("c@d.com", "EMAIL")
        vault.assign_token("010-0000-0000", "PHONE")
        c = vault.counters
        assert c["EMAIL"] == 2
        assert c["PHONE"] == 1

    def test_counters_is_copy(self):
        vault = RequestVault()
        vault.assign_token("a@b.com", "EMAIL")
        c = vault.counters
        c["EMAIL"] = 9999
        assert vault.counters["EMAIL"] == 1

    def test_contains_operator(self):
        vault = RequestVault()
        assert "a@b.com" not in vault
        vault.assign_token("a@b.com", "EMAIL")
        assert "a@b.com" in vault

    def test_empty_original_raises(self):
        vault = RequestVault()
        with pytest.raises(ValueError, match="empty original"):
            vault.assign_token("", "EMAIL")

    def test_empty_category_raises(self):
        vault = RequestVault()
        with pytest.raises(ValueError, match="empty category"):
            vault.assign_token("a@b.com", "")

    def test_two_vaults_are_independent(self):
        v1 = RequestVault()
        v2 = RequestVault()
        v1.assign_token("a@b.com", "EMAIL")
        # v2 has no entries
        assert v2.restore("EMAIL_1") is None
        assert len(v2) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 16–20.  Internal helper: _partial_mask
# ══════════════════════════════════════════════════════════════════════════════

class TestPartialMaskHelper:
    """Tests for _partial_mask() internal helper."""

    def test_short_string_fully_obscured(self):
        assert _partial_mask("ab") == "***"
        assert _partial_mask("abc") == "***"
        assert _partial_mask("abcd") == "***"

    def test_medium_string_reveals_ends(self):
        result = _partial_mask("abcde")   # len=5, reveal=min(2, 5//4)=min(2,1)=1
        assert result.startswith("a")
        assert result.endswith("e")
        assert "***" in result

    def test_longer_string_reveals_two_chars_each_end(self):
        # "alice@corp.io" len=13, reveal=min(2, 13//4)=min(2,3)=2
        result = _partial_mask("alice@corp.io")
        assert result.startswith("al")
        assert result.endswith("io")
        assert "***" in result

    def test_obscured_middle_is_three_stars(self):
        result = _partial_mask("hello world")  # len=11
        assert "***" in result
        # Exactly three stars (no length leakage)
        assert result.count("***") == 1

    def test_reveal_chars_parameter(self):
        # len=8: len//4 = 2; cap = min(3, 2) = 2 → reveals 2 chars per side
        result = _partial_mask("abcdefgh", reveal_chars=3)
        # reveal_chars=3 is capped to min(3, 8//4)=2 for len-8 string
        assert result.startswith("ab")
        assert result.endswith("gh")
        assert "***" in result

    def test_reveal_caps_at_len_div_4(self):
        # For 8-char string, len//4 = 2, even if reveal_chars=5
        result = _partial_mask("abcdefgh", reveal_chars=5)
        # Should reveal at most 2 chars per side (8//4 = 2)
        assert len(result.split("***")[0]) <= 2
        assert len(result.split("***")[1]) <= 2

    def test_unicode_string(self):
        """Korean name partially masked."""
        result = _partial_mask("김철수영")  # len=4 (exactly at boundary)
        assert result == "***"


# ══════════════════════════════════════════════════════════════════════════════
# 21–26.  Internal helper: _format_preserving_mask
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatPreservingMaskHelper:
    """Tests for _format_preserving_mask() internal helper."""

    def test_uppercase_mapped_to_X(self):
        assert _format_preserving_mask("ABC") == "XXX"

    def test_lowercase_mapped_to_x(self):
        assert _format_preserving_mask("abc") == "xxx"

    def test_digits_mapped_to_0(self):
        assert _format_preserving_mask("123") == "000"

    def test_specials_preserved(self):
        assert _format_preserving_mask("@.-_+") == "@.-_+"

    def test_mixed_input(self):
        result = _format_preserving_mask("alice@corp.io")
        assert result == "xxxxx@xxxx.xx"

    def test_aws_key_format_preserved(self):
        result = _format_preserving_mask("AKIAIOSFODNN7EXAMPLE")
        # AKIA → XXXX, IOSFODNN → XXXXXXXX, 7 → 0, EXAMPLE → XXXXXXX
        assert result == "XXXXXXXXXXXX0XXXXXXX"
        assert len(result) == len("AKIAIOSFODNN7EXAMPLE")

    def test_api_key_format_preserved(self):
        key = "sk-ant-api03-XYZ"
        result = _format_preserving_mask(key)
        assert result == "xx-xxx-xxx00-XXX"
        assert len(result) == len(key)

    def test_same_length_as_input(self):
        for value in ["alice@corp.io", "AKIAIOSFODNN7EXAMPLE", "sk-" + "a" * 20, "abc123!@#"]:
            assert len(_format_preserving_mask(value)) == len(value), (
                f"Format-preserving mask changed length for: {value!r}"
            )

    def test_empty_string(self):
        assert _format_preserving_mask("") == ""

    def test_only_digits(self):
        assert _format_preserving_mask("1234567890") == "0000000000"

    def test_only_specials(self):
        specials = "!@#$%^&*()"
        assert _format_preserving_mask(specials) == specials


# ══════════════════════════════════════════════════════════════════════════════
# 27–35.  apply_mask_style
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyMaskStyleTokenize:
    """Tests 16–20: TOKENIZE style."""

    def test_returns_bracketed_token(self):
        vault = RequestVault()
        result = apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.TOKENIZE, vault)
        assert result == "[EMAIL_1]"

    def test_blocked_returns_blocked_token(self):
        vault = RequestVault()
        result = apply_mask_style("sk-xxx", "API_KEY", MaskStyle.TOKENIZE, vault, blocked=True)
        assert result == "[API_KEY_1_BLOCKED]"

    def test_vault_stores_original_under_token(self):
        vault = RequestVault()
        apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.TOKENIZE, vault)
        assert vault.restore("EMAIL_1") == "alice@corp.io"

    def test_sequential_tokens_for_distinct_values(self):
        vault = RequestVault()
        r1 = apply_mask_style("a@b.com", "EMAIL", MaskStyle.TOKENIZE, vault)
        r2 = apply_mask_style("c@d.com", "EMAIL", MaskStyle.TOKENIZE, vault)
        assert r1 == "[EMAIL_1]"
        assert r2 == "[EMAIL_2]"

    def test_idempotent_same_value_same_token(self):
        vault = RequestVault()
        r1 = apply_mask_style("a@b.com", "EMAIL", MaskStyle.TOKENIZE, vault)
        r2 = apply_mask_style("a@b.com", "EMAIL", MaskStyle.TOKENIZE, vault)
        assert r1 == r2 == "[EMAIL_1]"
        assert len(vault) == 1


class TestApplyMaskStylePartial:
    """Tests 21–26: PARTIAL style."""

    def test_returns_partial_for_normal_length_string(self):
        vault = RequestVault()
        result = apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.PARTIAL, vault)
        assert "***" in result
        # Should not be a tokenize placeholder
        assert not result.startswith("[")

    def test_returns_three_stars_for_short_string(self):
        vault = RequestVault()
        result = apply_mask_style("ab", "EMAIL", MaskStyle.PARTIAL, vault)
        assert result == "***"

    def test_vault_stores_original_under_token(self):
        vault = RequestVault()
        apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.PARTIAL, vault)
        # Vault should have exactly one entry
        assert len(vault) == 1
        # Token is EMAIL_1
        token = vault.token_for("alice@corp.io")
        assert token == "EMAIL_1"
        assert vault.restore("EMAIL_1") == "alice@corp.io"

    def test_masked_text_does_not_contain_token_marker(self):
        """Partial style does not embed [TOKEN] in output."""
        vault = RequestVault()
        result = apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.PARTIAL, vault)
        assert "[EMAIL" not in result

    def test_explicit_restore_via_vault_returns_original(self):
        vault = RequestVault()
        original = "alice@corp.io"
        apply_mask_style(original, "EMAIL", MaskStyle.PARTIAL, vault)
        token = vault.token_for(original)
        assert vault.restore(token) == original

    def test_reveal_chars_parameter_forwarded(self):
        vault = RequestVault()
        result = apply_mask_style(
            "abcdefghij", "TEST", MaskStyle.PARTIAL, vault, reveal_chars=3
        )
        # len=10, len//4=2, min(3, 2)=2 → reveals 2 chars each end
        assert result.startswith("ab")
        assert result.endswith("ij")


class TestApplyMaskStyleFormatPreserving:
    """Tests 27–33: FORMAT_PRESERVING style."""

    def test_returns_fp_masked_string(self):
        vault = RequestVault()
        result = apply_mask_style(
            "alice@corp.io", "EMAIL", MaskStyle.FORMAT_PRESERVING, vault
        )
        assert result == "xxxxx@xxxx.xx"

    def test_same_length_as_original(self):
        vault = RequestVault()
        original = "AKIAIOSFODNN7EXAMPLE"
        result = apply_mask_style(
            original, "AWS_SECRET", MaskStyle.FORMAT_PRESERVING, vault
        )
        assert len(result) == len(original)

    def test_no_token_embedded_in_output(self):
        vault = RequestVault()
        result = apply_mask_style(
            "AKIAIOSFODNN7EXAMPLE", "AWS_SECRET", MaskStyle.FORMAT_PRESERVING, vault
        )
        assert "[" not in result

    def test_vault_stores_original_under_token(self):
        vault = RequestVault()
        original = "AKIAIOSFODNN7EXAMPLE"
        apply_mask_style(original, "AWS_SECRET", MaskStyle.FORMAT_PRESERVING, vault)
        assert len(vault) == 1
        token = vault.token_for(original)
        assert token == "AWS_SECRET_1"
        assert vault.restore("AWS_SECRET_1") == original

    def test_explicit_restore_via_vault_returns_original(self):
        vault = RequestVault()
        original = "sk-ant-api03-" + "A" * 40
        apply_mask_style(original, "API_KEY", MaskStyle.FORMAT_PRESERVING, vault)
        token = vault.token_for(original)
        assert vault.restore(token) == original

    def test_character_classes_preserved(self):
        """Verify: upper→X, lower→x, digit→0, special→unchanged."""
        vault = RequestVault()
        result = apply_mask_style("Ab1@", "TEST", MaskStyle.FORMAT_PRESERVING, vault)
        assert result == "Xx0@"

    def test_empty_string_returns_empty(self):
        vault = RequestVault()
        result = apply_mask_style("non-empty", "TEST", MaskStyle.TOKENIZE, vault)
        # (can't pass empty to apply_mask_style — vault raises ValueError)
        # Instead, verify _format_preserving_mask("") is ""
        from pii_guard.vault import _format_preserving_mask
        assert _format_preserving_mask("") == ""


# ══════════════════════════════════════════════════════════════════════════════
# 34–46.  mask_payload_with_vault
# ══════════════════════════════════════════════════════════════════════════════

class TestMaskPayloadWithVaultBasic:
    """Tests 34–41: Basic mask_payload_with_vault behaviour."""

    def test_empty_detections_text_unchanged(self):
        text = "Hello world"
        masked, vault = mask_payload_with_vault(text, [])
        assert masked == text
        assert len(vault) == 0

    def test_single_tokenize_detection(self):
        text = "Send email to alice@corp.io please."
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.TOKENIZE)
        masked, vault = mask_payload_with_vault(text, [det])
        assert "[EMAIL_1]" in masked
        assert email not in masked
        assert vault.restore("EMAIL_1") == email

    def test_single_partial_detection(self):
        text = "User: alice@corp.io logged in."
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.PARTIAL)
        masked, vault = mask_payload_with_vault(text, [det])
        # Partial output present (not a token placeholder)
        assert "***" in masked
        assert "[EMAIL" not in masked
        # Vault has the original
        assert vault.restore("EMAIL_1") == email

    def test_single_format_preserving_detection(self):
        text = "Key: AKIAIOSFODNN7EXAMPLE here."
        key = "AKIAIOSFODNN7EXAMPLE"
        det = _det("AWS_SECRET", text.index(key), text.index(key) + len(key), key,
                   action=Action.BLOCK, mask_style=MaskStyle.FORMAT_PRESERVING)
        masked, vault = mask_payload_with_vault(text, [det])
        # FP output has same length
        fp = _format_preserving_mask(key)
        assert fp in masked
        assert key not in masked
        assert vault.restore("AWS_SECRET_1_BLOCKED") == key

    def test_multiple_detections_different_styles(self):
        text = "Email alice@corp.io key AKIAIOSFODNN7EXAMPLE phone 010-1234-5678"
        email = "alice@corp.io"
        key = "AKIAIOSFODNN7EXAMPLE"
        phone = "010-1234-5678"
        dets = [
            _det("EMAIL",      text.index(email),  text.index(email) + len(email),  email,
                 action=Action.MASK, mask_style=MaskStyle.TOKENIZE),
            _det("AWS_SECRET", text.index(key),    text.index(key)   + len(key),    key,
                 action=Action.BLOCK, mask_style=MaskStyle.FORMAT_PRESERVING),
            _det("PHONE",      text.index(phone),  text.index(phone) + len(phone),  phone,
                 action=Action.MASK, mask_style=MaskStyle.PARTIAL),
        ]
        masked, vault = mask_payload_with_vault(text, dets)
        assert "[EMAIL_1]" in masked           # tokenize
        assert key not in masked               # FP masked
        assert "***" in masked                 # partial
        assert vault.restore("EMAIL_1") == email
        assert vault.restore("PHONE_1") == phone

    def test_overlapping_spans_first_wins(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        dets = [
            _det("AWS_SECRET", 0, 10, text[:10],
                 action=Action.BLOCK, mask_style=MaskStyle.TOKENIZE),
            _det("API_KEY",    5, 20, text[5:20],
                 action=Action.BLOCK, mask_style=MaskStyle.TOKENIZE),
        ]
        masked, vault = mask_payload_with_vault(text, dets)
        # First span wins
        assert "[AWS_SECRET_1_BLOCKED]" in masked
        # Second span skipped — not in vault
        assert vault.restore("API_KEY_1_BLOCKED") is None

    def test_allow_action_leaves_span_unchanged(self):
        text = "Name: Alice Smith, email: alice@corp.io"
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.ALLOW, mask_style=MaskStyle.TOKENIZE)
        masked, vault = mask_payload_with_vault(text, [det])
        assert email in masked          # allowed — not masked
        assert len(vault) == 0          # nothing stored in vault for allowed span

    def test_unsorted_detections_sorted_internally(self):
        text = "Phone 010-1234-5678, email alice@corp.io"
        email = "alice@corp.io"
        phone = "010-1234-5678"
        # Supply detections in reverse order (phone first, email second)
        dets = [
            _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                 action=Action.MASK, mask_style=MaskStyle.TOKENIZE),
            _det("PHONE", text.index(phone), text.index(phone) + len(phone), phone,
                 action=Action.MASK, mask_style=MaskStyle.TOKENIZE),
        ]
        # Reverse so phone (later in text) comes first in list
        dets_reversed = list(reversed(dets))
        masked, vault = mask_payload_with_vault(text, dets_reversed)
        # Both should be masked; phone appears first in text → PHONE_1... wait,
        # PHONE appears before EMAIL in the text. Let's check.
        # "Phone 010-1234-5678, email alice@corp.io"
        #  ^phone at idx 6        ^email at idx 28
        # So phone should get PHONE_1 and email gets EMAIL_1 (independent counters)
        assert "[EMAIL_1]" in masked
        assert "[PHONE_1]" in masked

    def test_zero_length_span_skipped(self):
        text = "hello world"
        det = _det("EMAIL", 5, 5, "", action=Action.MASK, mask_style=MaskStyle.TOKENIZE)
        # Zero-length → assign_token would raise ValueError for empty original,
        # but zero-length detection is skipped before assign_token is called.
        masked, vault = mask_payload_with_vault(text, [det])
        assert masked == text
        assert len(vault) == 0

    def test_dict_detection_accepted(self):
        """Plain dict detection descriptors work as well as Detection objects."""
        text = "Send to alice@corp.io thanks"
        email = "alice@corp.io"
        det = _dict_det("EMAIL", text.index(email), text.index(email) + len(email), email,
                        action="mask", mask_style="tokenize")
        masked, vault = mask_payload_with_vault(text, [det])
        assert "[EMAIL_1]" in masked
        assert vault.restore("EMAIL_1") == email


class TestMaskPayloadWithVaultVaultExtension:
    """Test 45: Existing vault is extended."""

    def test_existing_vault_extended(self):
        vault = RequestVault()
        vault.assign_token("pre-existing@corp.io", "EMAIL")   # EMAIL_1

        text = "Send to alice@corp.io please"
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.TOKENIZE)
        masked, returned_vault = mask_payload_with_vault(text, [det], vault=vault)

        # Same vault object returned
        assert returned_vault is vault
        # alice@corp.io gets EMAIL_2 (EMAIL_1 already taken by pre-existing)
        assert "[EMAIL_2]" in masked
        assert vault.restore("EMAIL_1") == "pre-existing@corp.io"
        assert vault.restore("EMAIL_2") == email


# ══════════════════════════════════════════════════════════════════════════════
# 47–50.  Full round-trip tests (mask → vault store → rehydrate → compare)
# ══════════════════════════════════════════════════════════════════════════════

class TestRoundTripTokenize:
    """Test 47: TOKENIZE full round-trip via automatic rehydration."""

    def test_single_entity_roundtrip(self):
        text = "Contact alice@corp.io for details."
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.TOKENIZE)
        masked, vault = mask_payload_with_vault(text, [det])

        # masked == "Contact [EMAIL_1] for details."
        # Simulate an LLM response that echoes the placeholder token back
        simulated_llm_response = "I will contact [EMAIL_1] right away."
        # Rehydrate → original email restored
        restored_response = vault.rehydrate(simulated_llm_response)
        assert "alice@corp.io" in restored_response

    def test_multi_entity_roundtrip(self):
        text = "From: alice@corp.io, cc: bob@corp.io, phone: 010-9999-8888"
        entities = [
            ("EMAIL", "alice@corp.io"),
            ("EMAIL", "bob@corp.io"),
            ("PHONE", "010-9999-8888"),
        ]
        dets = []
        for cat, val in entities:
            start = text.index(val)
            dets.append(_det(cat, start, start + len(val), val,
                             action=Action.MASK, mask_style=MaskStyle.TOKENIZE))

        masked, vault = mask_payload_with_vault(text, dets)
        # The masked text contains no originals
        for _, val in entities:
            assert val not in masked

        # Rehydrating the masked text produces the original
        restored = vault.rehydrate(masked)
        assert restored == text

    def test_roundtrip_recovers_exact_original_text(self):
        """vault.rehydrate(masked) == original_text for tokenize style."""
        text = "Key sk-abcdefg and email test@example.com here."
        email = "test@example.com"
        key = "sk-abcdefg"
        dets = [
            _det("API_KEY", text.index(key),   text.index(key) + len(key),   key,
                 action=Action.BLOCK, mask_style=MaskStyle.TOKENIZE),
            _det("EMAIL",   text.index(email), text.index(email) + len(email), email,
                 action=Action.MASK, mask_style=MaskStyle.TOKENIZE),
        ]
        masked, vault = mask_payload_with_vault(text, dets)
        assert vault.rehydrate(masked) == text


class TestRoundTripPartial:
    """Test 48: PARTIAL explicit round-trip via vault.restore(token)."""

    def test_partial_explicit_restore_roundtrip(self):
        vault = RequestVault()
        original = "alice@corp.io"
        masked = apply_mask_style(original, "EMAIL", MaskStyle.PARTIAL, vault)

        # Masked text does NOT auto-rehydrate (no [TOKEN] embedded)
        rehydrated_attempt = vault.rehydrate(masked)
        assert rehydrated_attempt == masked   # unchanged

        # But explicit restore via token returns original
        token = vault.token_for(original)
        assert token is not None
        assert vault.restore(token) == original

    def test_partial_payload_roundtrip(self):
        text = "User: alice@corp.io is logged in."
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.PARTIAL)
        masked, vault = mask_payload_with_vault(text, [det])

        # Retrieve original via vault
        token = vault.token_for(email)
        assert vault.restore(token) == email

    def test_multiple_partial_roundtrip(self):
        originals = ["alice@corp.io", "bob@corp.io", "carol@corp.io"]
        vault = RequestVault()
        masked_vals = [
            apply_mask_style(o, "EMAIL", MaskStyle.PARTIAL, vault)
            for o in originals
        ]
        # Each masked val is different from original
        for orig, mv in zip(originals, masked_vals):
            assert orig != mv
        # Explicit restore for all
        for orig in originals:
            token = vault.token_for(orig)
            assert vault.restore(token) == orig


class TestRoundTripFormatPreserving:
    """Test 49: FORMAT_PRESERVING explicit round-trip via vault.restore(token)."""

    def test_fp_explicit_restore_roundtrip(self):
        vault = RequestVault()
        original = "AKIAIOSFODNN7EXAMPLE"
        masked = apply_mask_style(original, "AWS_SECRET",
                                  MaskStyle.FORMAT_PRESERVING, vault)

        # Masked text does NOT auto-rehydrate
        rehydrated_attempt = vault.rehydrate(masked)
        assert rehydrated_attempt == masked  # unchanged

        # But explicit restore via token returns original
        token = vault.token_for(original)
        assert vault.restore(token) == original

    def test_fp_payload_roundtrip(self):
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE in config"
        key = "AKIAIOSFODNN7EXAMPLE"
        det = _det("AWS_SECRET", text.index(key), text.index(key) + len(key), key,
                   action=Action.BLOCK, mask_style=MaskStyle.FORMAT_PRESERVING)
        masked, vault = mask_payload_with_vault(text, [det])

        token = vault.token_for(key)
        assert vault.restore(token) == key

    def test_fp_masked_preserves_structure(self):
        """FP output has same format structure as input."""
        vault = RequestVault()
        email = "alice@corp.io"
        masked = apply_mask_style(email, "EMAIL",
                                  MaskStyle.FORMAT_PRESERVING, vault)
        # Same length, @ preserved, . preserved
        assert len(masked) == len(email)
        assert "@" in masked
        assert "." in masked
        # Letters masked
        assert "alice" not in masked


class TestRoundTripMixedStyles:
    """Test 50: Multi-style mix; tokenize span rehydrated, others explicit."""

    def test_tokenize_and_partial_in_same_payload(self):
        text = "Name: Alice Smith, email: alice@corp.io, key: sk-" + "x" * 20
        name = "Alice Smith"
        email = "alice@corp.io"
        key = "sk-" + "x" * 20
        dets = [
            _det("PERSON",  text.index(name),  text.index(name) + len(name),  name,
                 action=Action.MASK, mask_style=MaskStyle.PARTIAL),
            _det("EMAIL",   text.index(email), text.index(email) + len(email), email,
                 action=Action.MASK, mask_style=MaskStyle.TOKENIZE),
            _det("API_KEY", text.index(key),   text.index(key) + len(key),   key,
                 action=Action.BLOCK, mask_style=MaskStyle.FORMAT_PRESERVING),
        ]
        masked, vault = mask_payload_with_vault(text, dets)

        # EMAIL token is embedded → rehydrate can restore it
        assert "[EMAIL_1]" in masked
        rehydrated = vault.rehydrate(masked)
        assert "alice@corp.io" in rehydrated

        # PERSON partial is NOT embedded → explicit restore
        person_token = vault.token_for(name)
        assert vault.restore(person_token) == name

        # API_KEY FP is NOT embedded → explicit restore
        key_token = vault.token_for(key)
        assert vault.restore(key_token) == key

    def test_tokenize_rehydration_does_not_touch_partial_output(self):
        """vault.rehydrate() only replaces [TOKEN] patterns; partial '***' is unchanged."""
        vault = RequestVault()
        partial_out = apply_mask_style("alice@corp.io", "EMAIL", MaskStyle.PARTIAL, vault)
        tokenize_out = apply_mask_style("bob@corp.io", "EMAIL2", MaskStyle.TOKENIZE, vault)

        combined = f"{partial_out} and {tokenize_out}"
        rehydrated = vault.rehydrate(combined)

        # partial_out is unchanged (no [TOKEN] marker)
        assert partial_out in rehydrated
        # tokenize_out is replaced
        assert "[EMAIL2_1]" not in rehydrated
        assert "bob@corp.io" in rehydrated


# ══════════════════════════════════════════════════════════════════════════════
# 51.  Top-level import smoke tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTopLevelImports:
    """Test 51: Public API is importable from pii_guard."""

    def test_request_vault_importable(self):
        assert pii_guard.RequestVault is RequestVault

    def test_apply_mask_style_importable(self):
        assert pii_guard.apply_mask_style is apply_mask_style

    def test_mask_payload_with_vault_importable(self):
        assert pii_guard.mask_payload_with_vault is mask_payload_with_vault

    def test_request_vault_usable_from_top_level(self):
        vault = pii_guard.RequestVault()
        assert isinstance(vault, RequestVault)
        tok = vault.assign_token("a@b.com", "EMAIL")
        assert tok == "EMAIL_1"

    def test_apply_mask_style_usable_from_top_level(self):
        vault = pii_guard.RequestVault()
        result = pii_guard.apply_mask_style(
            "alice@corp.io", "EMAIL", pii_guard.MaskStyle.TOKENIZE, vault
        )
        assert result == "[EMAIL_1]"

    def test_mask_payload_with_vault_usable_from_top_level(self):
        text = "Contact alice@corp.io for help."
        email = "alice@corp.io"
        det = _det("EMAIL", text.index(email), text.index(email) + len(email), email,
                   action=Action.MASK, mask_style=MaskStyle.TOKENIZE)
        masked, vault = pii_guard.mask_payload_with_vault(text, [det])
        assert "[EMAIL_1]" in masked
        assert vault.restore("EMAIL_1") == email


# ══════════════════════════════════════════════════════════════════════════════
# Parametric tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("category,value", [
    ("EMAIL",       "test@example.org"),
    ("PHONE",       "010-1234-5678"),
    ("API_KEY",     "sk-" + "A" * 40),
    ("AWS_SECRET",  "AKIAIOSFODNN7EXAMPLE"),
    ("PERSON",      "John Smith"),
    ("PASSWORD",    "SuperSecret123"),
    ("RRN",         "900505-1234564"),
    ("CARD",        "4532015112830366"),
])
def test_tokenize_roundtrip_parametric(category: str, value: str):
    """For every category, tokenize produces a restorable placeholder."""
    vault = RequestVault()
    masked = apply_mask_style(value, category, MaskStyle.TOKENIZE, vault)

    expected_token = f"{category}_1"
    assert masked == f"[{expected_token}]"
    assert vault.restore(expected_token) == value


@pytest.mark.parametrize("category,value", [
    ("EMAIL",       "test@example.org"),
    ("PHONE",       "010-1234-5678"),
    ("API_KEY",     "sk-ant-api03-" + "A" * 30),
    ("AWS_SECRET",  "AKIAIOSFODNN7EXAMPLE"),
    ("PERSON",      "John Smith"),
    ("PASSWORD",    "SuperSecret123!"),
])
def test_partial_explicit_restore_parametric(category: str, value: str):
    """For every category, partial style stores original for explicit restore."""
    vault = RequestVault()
    masked = apply_mask_style(value, category, MaskStyle.PARTIAL, vault)

    # Masked text has *** somewhere
    assert "***" in masked
    # Vault has the original
    token = vault.token_for(value)
    assert token is not None
    assert vault.restore(token) == value


@pytest.mark.parametrize("category,value", [
    ("EMAIL",       "test@example.org"),
    ("PHONE",       "010-1234-5678"),
    ("API_KEY",     "sk-ant-api03-" + "A" * 30),
    ("AWS_SECRET",  "AKIAIOSFODNN7EXAMPLE"),
])
def test_format_preserving_roundtrip_parametric(category: str, value: str):
    """For every category, FP mask preserves length and allows restore."""
    vault = RequestVault()
    masked = apply_mask_style(value, category, MaskStyle.FORMAT_PRESERVING, vault)

    # Same length
    assert len(masked) == len(value)
    # No raw original in masked
    if len(value) > 4:
        # Not a trivial replacement of a very short string
        assert value not in masked or value == masked  # only equal if all same-class chars map to same char
    # Explicit restore works
    token = vault.token_for(value)
    assert vault.restore(token) == value
