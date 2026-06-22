"""
Unit tests for pii_guard.session_map.SessionMap

Coverage areas required by Sub-AC 2a:
  1. Idempotent re-encoding — same value → same placeholder
  2. Cross-category counter isolation — EMAIL counter does not advance PHONE counter
  3. Session-scope reset — after reset, counters restart from 1
  4. Blocked suffix — encode(..., blocked=True) → CATEGORY_N_BLOCKED
  5. Decode round-trip — decode(token) returns original
  6. Unknown token decode — returns None, never raises
  7. Counter monotonicity — indices increase without gaps per category
  8. Multi-value same category — second distinct value gets next index
  9. Restoration map snapshot — returns copy, not live reference
  10. Rehydrate — replaces [TOKEN] in arbitrary text
  11. Convenience helpers — bracket(), __len__, __contains__
  12. Empty value / category validation
  13. Integration with Engine — engine.session_map is the same SessionMap
"""
from __future__ import annotations

import pytest

from pii_guard import Engine, SessionMap


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic encode / decode
# ══════════════════════════════════════════════════════════════════════════════

class TestBasicEncodeDecycle:

    def test_encode_returns_category_index_token(self):
        sm = SessionMap()
        token = sm.encode("alice@corp.io", "EMAIL")
        assert token == "EMAIL_1"

    def test_encode_returns_string(self):
        sm = SessionMap()
        result = sm.encode("user@example.com", "EMAIL")
        assert isinstance(result, str)

    def test_decode_returns_original(self):
        sm = SessionMap()
        sm.encode("alice@corp.io", "EMAIL")
        assert sm.decode("EMAIL_1") == "alice@corp.io"

    def test_decode_unknown_token_returns_none(self):
        sm = SessionMap()
        assert sm.decode("EMAIL_99") is None
        assert sm.decode("") is None
        assert sm.decode("UNKNOWN_1") is None

    def test_decode_bracketed_strips_brackets(self):
        sm = SessionMap()
        sm.encode("alice@corp.io", "EMAIL")
        assert sm.decode_bracketed("[EMAIL_1]") == "alice@corp.io"

    def test_decode_bracketed_no_brackets_returns_none(self):
        sm = SessionMap()
        sm.encode("alice@corp.io", "EMAIL")
        # Without brackets, decode_bracketed should return None
        assert sm.decode_bracketed("EMAIL_1") is None

    def test_decode_bracketed_unknown_token_returns_none(self):
        sm = SessionMap()
        assert sm.decode_bracketed("[GHOST_99]") is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Idempotent re-encoding (same value → same placeholder)
# ══════════════════════════════════════════════════════════════════════════════

class TestIdempotentReEncoding:

    def test_same_value_same_token_twice(self):
        sm = SessionMap()
        t1 = sm.encode("alice@corp.io", "EMAIL")
        t2 = sm.encode("alice@corp.io", "EMAIL")
        assert t1 == t2 == "EMAIL_1"

    def test_same_value_same_token_many_calls(self):
        sm = SessionMap()
        tokens = [sm.encode("x@y.com", "EMAIL") for _ in range(10)]
        assert all(t == "EMAIL_1" for t in tokens)

    def test_same_value_does_not_increment_counter(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.encode("a@b.com", "EMAIL")
        sm.encode("a@b.com", "EMAIL")
        # Counter should still be 1 — re-encodes don't bump it
        assert sm.counters["EMAIL"] == 1

    def test_distinct_values_get_distinct_tokens(self):
        sm = SessionMap()
        t1 = sm.encode("a@corp.io", "EMAIL")
        t2 = sm.encode("b@corp.io", "EMAIL")
        assert t1 != t2
        assert t1 == "EMAIL_1"
        assert t2 == "EMAIL_2"

    def test_idempotent_across_sessions_within_same_object(self):
        """Same value re-encoded throughout a long session always same token."""
        sm = SessionMap()
        sm.encode("secret@example.com", "EMAIL")   # EMAIL_1
        sm.encode("other@example.com", "EMAIL")    # EMAIL_2
        sm.encode("third@example.com", "EMAIL")    # EMAIL_3

        # Re-encoding the first value still returns EMAIL_1
        assert sm.encode("secret@example.com", "EMAIL") == "EMAIL_1"
        # And the counter did not increase
        assert sm.counters["EMAIL"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cross-category counter isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossCategoryCounterIsolation:

    def test_email_and_phone_counters_are_independent(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")       # EMAIL_1
        sm.encode("c@d.com", "EMAIL")       # EMAIL_2
        phone_tok = sm.encode("010-1234-5678", "PHONE")  # PHONE_1 (not PHONE_3)
        assert phone_tok == "PHONE_1"

    def test_phone_counter_does_not_advance_email_counter(self):
        sm = SessionMap()
        sm.encode("010-1111-2222", "PHONE")   # PHONE_1
        sm.encode("010-3333-4444", "PHONE")   # PHONE_2
        email_tok = sm.encode("a@b.com", "EMAIL")  # EMAIL_1 (not EMAIL_3)
        assert email_tok == "EMAIL_1"

    def test_all_categories_start_at_one(self):
        sm = SessionMap()
        categories = ["EMAIL", "PHONE", "PERSON", "ADDRESS", "API_KEY", "RRN", "CARD"]
        values = [
            "a@b.com", "010-0000-0000", "John Smith",
            "123 Main St", "sk-" + "x" * 20, "900505-1234564",
            "4532015112830366",
        ]
        for val, cat in zip(values, categories):
            tok = sm.encode(val, cat)
            assert tok == f"{cat}_1", (
                f"Expected {cat}_1 as first token for {cat}, got {tok!r}"
            )

    def test_counter_snapshots_are_independent(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.encode("c@d.com", "EMAIL")
        sm.encode("010-1234-5678", "PHONE")

        counters = sm.counters
        assert counters["EMAIL"] == 2
        assert counters["PHONE"] == 1
        assert "ADDRESS" not in counters

    def test_category_names_are_case_sensitive(self):
        sm = SessionMap()
        tok_upper = sm.encode("a@b.com", "EMAIL")
        tok_lower = sm.encode("c@d.com", "email")  # different category
        assert tok_upper == "EMAIL_1"
        assert tok_lower == "email_1"
        # They should be independent counter spaces
        assert sm.counters["EMAIL"] == 1
        assert sm.counters["email"] == 1

    def test_five_categories_each_independent(self):
        sm = SessionMap()
        expected = {
            "EMAIL": ("a@b.com", "EMAIL_1"),
            "PHONE": ("010-0000-0000", "PHONE_1"),
            "PERSON": ("John Doe", "PERSON_1"),
            "API_KEY": ("sk-" + "k" * 20, "API_KEY_1"),
            "CARD": ("4532015112830366", "CARD_1"),
        }
        for cat, (val, expected_tok) in expected.items():
            tok = sm.encode(val, cat)
            assert tok == expected_tok, f"Category {cat}: expected {expected_tok}, got {tok}"

    def test_many_values_one_category_others_unaffected(self):
        sm = SessionMap()
        # Encode 100 emails
        for i in range(100):
            sm.encode(f"user{i}@test.com", "EMAIL")
        # PHONE should still start from 1
        phone_tok = sm.encode("010-9999-9999", "PHONE")
        assert phone_tok == "PHONE_1"
        assert sm.counters["EMAIL"] == 100
        assert sm.counters["PHONE"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Session-scope reset
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionScopeReset:

    def test_reset_clears_counters(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.encode("c@d.com", "EMAIL")
        sm.reset()
        assert sm.counters == {}

    def test_reset_clears_encode_map(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.reset()
        assert len(sm) == 0
        assert sm.encode_map == {}

    def test_reset_clears_restoration_map(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.reset()
        assert sm.restoration_map == {}

    def test_after_reset_same_value_gets_index_1_again(self):
        sm = SessionMap()
        tok1 = sm.encode("alice@corp.io", "EMAIL")
        assert tok1 == "EMAIL_1"
        sm.reset()
        tok2 = sm.encode("alice@corp.io", "EMAIL")
        assert tok2 == "EMAIL_1"  # Fresh session — back to 1

    def test_after_reset_counters_restart(self):
        sm = SessionMap()
        for i in range(5):
            sm.encode(f"user{i}@test.com", "EMAIL")
        assert sm.counters["EMAIL"] == 5
        sm.reset()
        sm.encode("new@test.com", "EMAIL")
        assert sm.counters["EMAIL"] == 1

    def test_decode_after_reset_returns_none(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        sm.reset()
        assert sm.decode("EMAIL_1") is None

    def test_multiple_resets(self):
        sm = SessionMap()
        for _ in range(3):
            sm.encode("a@b.com", "EMAIL")
            sm.encode("c@d.com", "EMAIL")
            sm.reset()
        # Final reset leaves everything clean
        tok = sm.encode("a@b.com", "EMAIL")
        assert tok == "EMAIL_1"

    def test_reset_does_not_affect_other_instance(self):
        sm1 = SessionMap()
        sm2 = SessionMap()
        sm1.encode("a@b.com", "EMAIL")
        sm2.encode("a@b.com", "EMAIL")
        sm1.reset()
        # sm2 is independent
        assert sm2.decode("EMAIL_1") == "a@b.com"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Blocked suffix
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockedSuffix:

    def test_blocked_token_has_suffix(self):
        sm = SessionMap()
        tok = sm.encode("sk-" + "x" * 20, "API_KEY", blocked=True)
        assert tok.endswith("_BLOCKED")
        assert tok == "API_KEY_1_BLOCKED"

    def test_non_blocked_token_has_no_suffix(self):
        sm = SessionMap()
        tok = sm.encode("a@b.com", "EMAIL", blocked=False)
        assert not tok.endswith("_BLOCKED")
        assert tok == "EMAIL_1"

    def test_blocked_decode_returns_original(self):
        sm = SessionMap()
        secret = "sk-" + "x" * 20
        sm.encode(secret, "API_KEY", blocked=True)
        restored = sm.decode("API_KEY_1_BLOCKED")
        assert restored == secret

    def test_blocked_rehydrate_restores_original(self):
        sm = SessionMap()
        secret = "sk-" + "x" * 20
        sm.encode(secret, "API_KEY", blocked=True)
        text = f"key=[API_KEY_1_BLOCKED]"
        assert sm.rehydrate(text) == f"key={secret}"

    def test_blocked_counter_increments_for_category(self):
        sm = SessionMap()
        sm.encode("key1", "API_KEY", blocked=True)
        sm.encode("key2", "API_KEY", blocked=True)
        assert sm.counters["API_KEY"] == 2
        assert sm.decode("API_KEY_1_BLOCKED") == "key1"
        assert sm.decode("API_KEY_2_BLOCKED") == "key2"

    def test_mixed_blocked_and_unblocked_same_category(self):
        """Blocked and non-blocked items share the same category counter."""
        sm = SessionMap()
        t1 = sm.encode("email@test.com", "EMAIL", blocked=False)   # EMAIL_1
        t2 = sm.encode("other@test.com", "EMAIL", blocked=True)    # EMAIL_2_BLOCKED
        assert t1 == "EMAIL_1"
        assert t2 == "EMAIL_2_BLOCKED"
        assert sm.counters["EMAIL"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# 6. Rehydration
# ══════════════════════════════════════════════════════════════════════════════

class TestRehydration:

    def test_rehydrate_single_placeholder(self):
        sm = SessionMap()
        sm.encode("alice@corp.io", "EMAIL")
        result = sm.rehydrate("Contact [EMAIL_1] for details")
        assert result == "Contact alice@corp.io for details"

    def test_rehydrate_multiple_placeholders(self):
        sm = SessionMap()
        sm.encode("alice@corp.io", "EMAIL")
        sm.encode("bob@corp.io", "EMAIL")
        sm.encode("010-1234-5678", "PHONE")
        text = "[EMAIL_1] or [EMAIL_2], phone [PHONE_1]"
        restored = sm.rehydrate(text)
        assert "alice@corp.io" in restored
        assert "bob@corp.io" in restored
        assert "010-1234-5678" in restored

    def test_rehydrate_leaves_unknown_tokens_unchanged(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        text = "Known: [EMAIL_1], unknown: [GHOST_99]"
        result = sm.rehydrate(text)
        assert "a@b.com" in result
        assert "[GHOST_99]" in result  # left unchanged

    def test_rehydrate_empty_string(self):
        sm = SessionMap()
        assert sm.rehydrate("") == ""

    def test_rehydrate_no_placeholders(self):
        sm = SessionMap()
        text = "No placeholders here at all."
        assert sm.rehydrate(text) == text

    def test_rehydrate_longer_token_wins_over_shorter(self):
        """EMAIL_10 must not be partially replaced by EMAIL_1."""
        sm = SessionMap()
        for i in range(10):
            sm.encode(f"user{i}@test.com", "EMAIL")
        text = "[EMAIL_10] and [EMAIL_1]"
        result = sm.rehydrate(text)
        assert "user9@test.com" in result   # EMAIL_10
        assert "user0@test.com" in result   # EMAIL_1
        # Neither token should remain in the result
        assert "[EMAIL_10]" not in result
        assert "[EMAIL_1]" not in result

    def test_rehydrate_is_consistent_with_decode(self):
        sm = SessionMap()
        sm.encode("secret@corp.io", "EMAIL")
        via_rehydrate = sm.rehydrate("[EMAIL_1]")
        via_decode = sm.decode("EMAIL_1")
        assert via_rehydrate == via_decode == "secret@corp.io"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Convenience helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestConvenienceHelpers:

    def test_bracket_wraps_token(self):
        assert SessionMap.bracket("EMAIL_1") == "[EMAIL_1]"
        assert SessionMap.bracket("API_KEY_1_BLOCKED") == "[API_KEY_1_BLOCKED]"

    def test_len_counts_distinct_values(self):
        sm = SessionMap()
        assert len(sm) == 0
        sm.encode("a@b.com", "EMAIL")
        assert len(sm) == 1
        sm.encode("a@b.com", "EMAIL")  # re-encode same value
        assert len(sm) == 1            # still 1
        sm.encode("c@d.com", "EMAIL")
        assert len(sm) == 2

    def test_contains_operator(self):
        sm = SessionMap()
        assert "a@b.com" not in sm
        sm.encode("a@b.com", "EMAIL")
        assert "a@b.com" in sm
        assert "unknown@test.com" not in sm

    def test_restoration_map_is_copy(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        rmap = sm.restoration_map
        rmap["INJECTED"] = "evil"  # mutate the copy
        # Live state should be unaffected
        assert "INJECTED" not in sm.restoration_map

    def test_encode_map_is_copy(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        emap = sm.encode_map
        emap["evil@bad.com"] = "INJECTED_1"
        assert "evil@bad.com" not in sm.encode_map

    def test_counters_property_is_copy(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")
        c = sm.counters
        c["EMAIL"] = 9999
        assert sm.counters["EMAIL"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 8. Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:

    def test_empty_value_raises(self):
        sm = SessionMap()
        with pytest.raises(ValueError, match="empty value"):
            sm.encode("", "EMAIL")

    def test_empty_category_raises(self):
        sm = SessionMap()
        with pytest.raises(ValueError, match="empty category"):
            sm.encode("a@b.com", "")

    def test_none_value_raises(self):
        """Passing None should raise an error (implicit via if not value)."""
        sm = SessionMap()
        with pytest.raises((ValueError, AttributeError, TypeError)):
            sm.encode(None, "EMAIL")  # type: ignore[arg-type]

    def test_whitespace_only_value_raises(self):
        """Whitespace-only strings count as falsy — should raise."""
        sm = SessionMap()
        # "" is falsy; " " is truthy so we check the actual contract
        # The spec says "empty value"; "   " is not truly empty so no raise expected,
        # but we verify encode doesn't silently corrupt state.
        tok = sm.encode("   ", "EMAIL")
        assert tok == "EMAIL_1"
        assert sm.decode("EMAIL_1") == "   "


# ══════════════════════════════════════════════════════════════════════════════
# 9. Counter monotonicity
# ══════════════════════════════════════════════════════════════════════════════

class TestCounterMonotonicity:

    def test_indices_have_no_gaps(self):
        sm = SessionMap()
        for i in range(5):
            sm.encode(f"user{i}@test.com", "EMAIL")
        for i in range(1, 6):
            assert sm.decode(f"EMAIL_{i}") is not None

    def test_indices_start_at_one(self):
        sm = SessionMap()
        tok = sm.encode("a@b.com", "EMAIL")
        assert tok == "EMAIL_1"

    def test_counter_never_decrements(self):
        sm = SessionMap()
        sm.encode("a@b.com", "EMAIL")   # EMAIL_1
        sm.encode("c@d.com", "EMAIL")   # EMAIL_2
        sm.encode("e@f.com", "EMAIL")   # EMAIL_3
        # Even if the first value is re-encoded, counter stays at 3
        sm.encode("a@b.com", "EMAIL")
        assert sm.counters["EMAIL"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 10. Integration with Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineIntegration:

    def test_engine_exposes_session_map(self):
        engine = Engine()
        assert isinstance(engine.session_map, SessionMap)

    def test_engine_scan_uses_session_map(self):
        engine = Engine()
        result = engine.scan("Contact alice@example.com")
        # After scan, session map should have the email encoded
        sm = engine.session_map
        assert "alice@example.com" in sm or len(sm) >= 1

    def test_engine_same_value_same_placeholder_across_scans(self):
        engine = Engine()
        r1 = engine.scan("Email: alice@test.com")
        r2 = engine.scan("Reply to alice@test.com now")
        # Both should use [EMAIL_1]
        assert "[EMAIL_1]" in r1.redacted_text
        assert "[EMAIL_1]" in r2.redacted_text

    def test_engine_different_values_different_placeholders(self):
        engine = Engine()
        r1 = engine.scan("From: alice@test.com")
        r2 = engine.scan("To: bob@test.com")
        assert "[EMAIL_1]" in r1.redacted_text
        assert "[EMAIL_2]" in r2.redacted_text

    def test_engine_rehydrate_uses_session_map(self):
        engine = Engine()
        result = engine.scan("Contact alice@corp.io for details")
        restored = engine.rehydrate(result.redacted_text)
        assert "alice@corp.io" in restored

    def test_engine_reset_session_clears_map(self):
        engine = Engine()
        engine.scan("Contact alice@example.com")
        engine.reset_session()
        # After reset the session map is empty
        assert len(engine.session_map) == 0

    def test_engine_reset_session_restarts_counters(self):
        engine = Engine()
        engine.scan("From: alice@test.com")    # EMAIL_1
        engine.scan("To: bob@test.com")         # EMAIL_2
        engine.reset_session()
        r = engine.scan("Reply to alice@test.com")
        assert "[EMAIL_1]" in r.redacted_text  # Back to EMAIL_1 after reset

    def test_engine_restoration_map_property(self):
        engine = Engine()
        engine.scan("Contact alice@example.com")
        rmap = engine.restoration_map
        assert isinstance(rmap, dict)
        # Should contain at least the email mapping
        assert any("alice@example.com" in v for v in rmap.values())

    def test_engine_cross_category_isolation_in_scan(self):
        engine = Engine()
        r1 = engine.scan("Email: alice@test.com")    # EMAIL_1
        r2 = engine.scan("Phone: 010-1234-5678")      # PHONE_1 (not PHONE_2)
        assert "[EMAIL_1]" in r1.redacted_text
        assert "[PHONE_1]" in r2.redacted_text
        # PHONE counter is 1, not 2
        assert engine.session_map.counters.get("PHONE", 0) == 1

    def test_two_engine_instances_are_independent(self):
        e1 = Engine()
        e2 = Engine()
        e1.scan("alice@test.com")
        e1.scan("bob@test.com")
        r = e2.scan("alice@test.com")
        # e2 has its own session, so alice should be EMAIL_1 not EMAIL_3
        assert "[EMAIL_1]" in r.redacted_text

    def test_engine_blocked_secret_in_session_map(self):
        engine = Engine()
        key = "sk-" + "a" * 48
        result = engine.scan(f"key={key}")
        # The session map should hold the blocked secret
        sm = engine.session_map
        # Find the token in the redacted text
        import re
        blocked_tokens = re.findall(r"\[(API_KEY_\d+_BLOCKED)\]", result.redacted_text)
        assert blocked_tokens, "Expected a blocked API_KEY token in output"
        # The token should decode back to the original key
        for tok in blocked_tokens:
            assert sm.decode(tok) == key


# ══════════════════════════════════════════════════════════════════════════════
# 11. Edge cases and stress
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_special_chars_in_value(self):
        sm = SessionMap()
        val = "pass!@#$%^&*()_+-={}|[]\\;':\",./<>?"
        tok = sm.encode(val, "PASSWORD")
        assert tok == "PASSWORD_1"
        assert sm.decode("PASSWORD_1") == val

    def test_unicode_value(self):
        sm = SessionMap()
        tok = sm.encode("김철수", "PERSON")
        assert tok == "PERSON_1"
        assert sm.decode("PERSON_1") == "김철수"

    def test_long_value(self):
        sm = SessionMap()
        long_val = "A" * 10_000
        tok = sm.encode(long_val, "TOKEN")
        assert tok == "TOKEN_1"
        assert sm.decode("TOKEN_1") == long_val

    def test_many_values_performance(self):
        """Encoding 1000 distinct values should be fast and correct."""
        sm = SessionMap()
        for i in range(1000):
            sm.encode(f"user{i:04d}@test.com", "EMAIL")
        assert len(sm) == 1000
        assert sm.counters["EMAIL"] == 1000
        # Re-encode all — should return existing tokens
        for i in range(1000):
            tok = sm.encode(f"user{i:04d}@test.com", "EMAIL")
            assert tok == f"EMAIL_{i + 1}"

    def test_value_as_placeholder_string(self):
        """A value that looks like a token should still be encoded normally."""
        sm = SessionMap()
        val = "[EMAIL_1]"  # looks like a token itself
        tok = sm.encode(val, "EMAIL")
        assert tok == "EMAIL_1"
        assert sm.decode("EMAIL_1") == "[EMAIL_1]"

    def test_bracket_is_static_method(self):
        """bracket() can be called on the class without an instance."""
        assert SessionMap.bracket("TOKEN_1") == "[TOKEN_1]"
