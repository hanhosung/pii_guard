"""
Unit tests for pii_guard.masker.maskPayload  (Sub-AC 2b-i)

Coverage requirements
---------------------
1.  Single entity — span replaced, mapping has exactly one entry, no extras.
2.  Multiple entities of the same category — per-category counter increments
    (EMAIL_1, EMAIL_2, …); mapping has one entry per entity.
3.  Multiple entities of different categories — per-category counters are
    isolated (EMAIL_1 and PHONE_1 for distinct categories).
4.  Overlapping entities — first (earlier start) entity wins; overlapping
    entity is absent from the mapping store (no extras).
5.  No entities — text unchanged, mapping is empty dict.
6.  Entity supplied as Detection object (not a plain dict).
7.  Entity supplied as plain dict.
8.  Text without any matches for given spans — sanity check.
9.  Entity at the very start of text.
10. Entity at the very end of text.
11. Adjacent (non-overlapping) entities processed correctly.
12. Category counter is isolated — one category's counter does not advance
    another.
13. Reverse-mapping keys are bare tokens (no brackets), values are originals.
14. Masked text contains bracketed placeholder ``[CATEGORY_N]``.
15. Original value is NOT present in masked text.
16. Mapping store has no extras beyond the replaced entities.
17. Mapping store entries equal the number of replaced (non-overlapping) entities.
18. Zero-length / inverted spans are silently skipped.
19. Unicode entity values work correctly.
20. Multiple different categories assigned independent monotonic indices.

Run with:   pytest tests/test_masker.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pii_guard.masker import maskPayload
from pii_guard import maskPayload as maskPayload_toplevel  # export smoke-test
from pii_guard.models import (
    Action,
    CategoryClass,
    Detection,
    DetectionStage,
    MaskStyle,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build minimal Detection objects and plain dicts
# ──────────────────────────────────────────────────────────────────────────────

def _det(category: str, start: int, end: int, original: str) -> Detection:
    """Construct a Detection for use as a pre-built entity."""
    return Detection(
        category=category,
        category_class=CategoryClass.PII,
        action=Action.MASK,
        mask_style=MaskStyle.TOKENIZE,
        start=start,
        end=end,
        original=original,
        detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
        rule_id="test_rule",
        confidence=0.99,
    )


def _dict_entity(category: str, start: int, end: int, original: str) -> dict:
    """Construct a plain-dict entity descriptor."""
    return {
        "category": category,
        "start": start,
        "end": end,
        "original": original,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Custom minimal object (attribute-based, not a Detection)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MinimalEntity:
    category: str
    start: int
    end: int
    original: str


# ══════════════════════════════════════════════════════════════════════════════
# 1. Empty entity list
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyEntities:

    def test_empty_list_returns_text_unchanged(self):
        text = "Hello, world! alice@example.com"
        masked, mapping = maskPayload(text, [])
        assert masked == text

    def test_empty_list_returns_empty_mapping(self):
        masked, mapping = maskPayload("some text", [])
        assert mapping == {}

    def test_empty_text_empty_list(self):
        masked, mapping = maskPayload("", [])
        assert masked == ""
        assert mapping == {}

    def test_empty_text_with_entity_list(self):
        # Entities pointing into empty text — zero-length span, no replacement
        masked, mapping = maskPayload("", [_dict_entity("EMAIL", 0, 0, "")])
        assert masked == ""
        assert mapping == {}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Single entity (Detection object)
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleEntityDetectionObject:
    TEXT = "Send report to alice@example.com please."
    EMAIL = "alice@example.com"
    START = TEXT.index(EMAIL)
    END = START + len(EMAIL)

    def test_masked_text_contains_placeholder(self):
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        masked, _ = maskPayload(self.TEXT, [entity])
        assert "[EMAIL_1]" in masked

    def test_original_not_in_masked_text(self):
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        masked, _ = maskPayload(self.TEXT, [entity])
        assert self.EMAIL not in masked

    def test_mapping_has_exactly_one_entry(self):
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert len(mapping) == 1

    def test_mapping_key_is_bare_token_no_brackets(self):
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert "EMAIL_1" in mapping
        assert "[EMAIL_1]" not in mapping

    def test_mapping_value_is_original(self):
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert mapping["EMAIL_1"] == self.EMAIL

    def test_text_structure_preserved(self):
        """Non-entity text parts are preserved verbatim."""
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        masked, _ = maskPayload(self.TEXT, [entity])
        assert masked.startswith("Send report to ")
        assert masked.endswith(" please.")

    def test_no_extras_in_mapping(self):
        """Mapping must contain no keys beyond the single replaced entity."""
        entity = _det("EMAIL", self.START, self.END, self.EMAIL)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert set(mapping.keys()) == {"EMAIL_1"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Single entity (plain dict)
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleEntityDict:
    TEXT = "Call us at 010-1234-5678 for support."
    PHONE = "010-1234-5678"
    START = TEXT.index(PHONE)
    END = START + len(PHONE)

    def test_placeholder_in_masked_text(self):
        entity = _dict_entity("PHONE", self.START, self.END, self.PHONE)
        masked, _ = maskPayload(self.TEXT, [entity])
        assert "[PHONE_1]" in masked

    def test_original_absent_from_masked_text(self):
        entity = _dict_entity("PHONE", self.START, self.END, self.PHONE)
        masked, _ = maskPayload(self.TEXT, [entity])
        assert self.PHONE not in masked

    def test_mapping_key_and_value(self):
        entity = _dict_entity("PHONE", self.START, self.END, self.PHONE)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert mapping == {"PHONE_1": self.PHONE}

    def test_no_extras(self):
        entity = _dict_entity("PHONE", self.START, self.END, self.PHONE)
        _, mapping = maskPayload(self.TEXT, [entity])
        assert len(mapping) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multiple entities — same category (counter increments per category)
# ══════════════════════════════════════════════════════════════════════════════

class TestMultipleEntitiesSameCategory:
    TEXT = "From alice@corp.io, cc bob@corp.io, bcc carol@corp.io."
    EMAILS = ["alice@corp.io", "bob@corp.io", "carol@corp.io"]

    def _entities(self):
        entities = []
        pos = 0
        for email in self.EMAILS:
            start = self.TEXT.index(email, pos)
            end = start + len(email)
            entities.append(_det("EMAIL", start, end, email))
            pos = end
        return entities

    def test_three_email_placeholders(self):
        masked, _ = maskPayload(self.TEXT, self._entities())
        assert "[EMAIL_1]" in masked
        assert "[EMAIL_2]" in masked
        assert "[EMAIL_3]" in masked

    def test_originals_absent(self):
        masked, _ = maskPayload(self.TEXT, self._entities())
        for email in self.EMAILS:
            assert email not in masked

    def test_mapping_has_three_entries(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert len(mapping) == 3

    def test_mapping_keys_are_email_1_2_3(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert set(mapping.keys()) == {"EMAIL_1", "EMAIL_2", "EMAIL_3"}

    def test_mapping_values_match_originals(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert mapping["EMAIL_1"] == "alice@corp.io"
        assert mapping["EMAIL_2"] == "bob@corp.io"
        assert mapping["EMAIL_3"] == "carol@corp.io"

    def test_no_extras_in_mapping(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert set(mapping.keys()) == {"EMAIL_1", "EMAIL_2", "EMAIL_3"}

    def test_dict_entities_same_result(self):
        """Plain dict entities produce the same result as Detection objects."""
        pos = 0
        dict_entities = []
        for email in self.EMAILS:
            start = self.TEXT.index(email, pos)
            end = start + len(email)
            dict_entities.append(_dict_entity("EMAIL", start, end, email))
            pos = end
        masked, mapping = maskPayload(self.TEXT, dict_entities)
        assert "[EMAIL_1]" in masked
        assert mapping["EMAIL_1"] == "alice@corp.io"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Multiple entities — different categories (per-category counter isolation)
# ══════════════════════════════════════════════════════════════════════════════

class TestMultipleEntitiesDifferentCategories:
    TEXT = "Contact alice@corp.io or call 010-9999-0000."
    EMAIL = "alice@corp.io"
    PHONE = "010-9999-0000"
    EMAIL_START = TEXT.index(EMAIL)
    PHONE_START = TEXT.index(PHONE)

    def _entities(self):
        return [
            _det("EMAIL", self.EMAIL_START, self.EMAIL_START + len(self.EMAIL), self.EMAIL),
            _det("PHONE", self.PHONE_START, self.PHONE_START + len(self.PHONE), self.PHONE),
        ]

    def test_email_gets_index_1(self):
        masked, _ = maskPayload(self.TEXT, self._entities())
        assert "[EMAIL_1]" in masked

    def test_phone_gets_index_1_independently(self):
        """PHONE counter starts at 1 regardless of how many emails came before."""
        masked, _ = maskPayload(self.TEXT, self._entities())
        assert "[PHONE_1]" in masked

    def test_originals_absent(self):
        masked, _ = maskPayload(self.TEXT, self._entities())
        assert self.EMAIL not in masked
        assert self.PHONE not in masked

    def test_mapping_has_two_entries(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert len(mapping) == 2

    def test_mapping_keys(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert set(mapping.keys()) == {"EMAIL_1", "PHONE_1"}

    def test_mapping_values(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert mapping["EMAIL_1"] == self.EMAIL
        assert mapping["PHONE_1"] == self.PHONE

    def test_no_extras(self):
        _, mapping = maskPayload(self.TEXT, self._entities())
        assert len(mapping) == 2

    def test_three_categories_all_start_at_1(self):
        """EMAIL_1, PHONE_1, PERSON_1 — all three counters are independent."""
        text = "alice@corp.io called 010-9999-0000 on behalf of Name: John Smith."
        email_s = text.index("alice@corp.io")
        phone_s = text.index("010-9999-0000")
        person_s = text.index("John Smith")
        entities = [
            _det("EMAIL",  email_s,  email_s + len("alice@corp.io"), "alice@corp.io"),
            _det("PHONE",  phone_s,  phone_s + len("010-9999-0000"), "010-9999-0000"),
            _det("PERSON", person_s, person_s + len("John Smith"),   "John Smith"),
        ]
        masked, mapping = maskPayload(text, entities)
        assert "[EMAIL_1]" in masked
        assert "[PHONE_1]" in masked
        assert "[PERSON_1]" in masked
        assert set(mapping.keys()) == {"EMAIL_1", "PHONE_1", "PERSON_1"}

    def test_second_of_same_category_gets_index_2(self):
        """Two emails → EMAIL_1, EMAIL_2; single phone → PHONE_1."""
        text = "From a@b.com cc c@d.com phone 010-1111-2222"
        e1s = text.index("a@b.com")
        e2s = text.index("c@d.com")
        ps  = text.index("010-1111-2222")
        entities = [
            _det("EMAIL", e1s, e1s + len("a@b.com"),        "a@b.com"),
            _det("EMAIL", e2s, e2s + len("c@d.com"),        "c@d.com"),
            _det("PHONE", ps,  ps  + len("010-1111-2222"), "010-1111-2222"),
        ]
        masked, mapping = maskPayload(text, entities)
        assert "[EMAIL_1]" in masked
        assert "[EMAIL_2]" in masked
        assert "[PHONE_1]" in masked
        assert set(mapping.keys()) == {"EMAIL_1", "EMAIL_2", "PHONE_1"}
        assert mapping["EMAIL_1"] == "a@b.com"
        assert mapping["EMAIL_2"] == "c@d.com"
        assert mapping["PHONE_1"] == "010-1111-2222"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Overlapping spans — first span wins, overlapping entity skipped
# ══════════════════════════════════════════════════════════════════════════════

class TestOverlappingEntities:
    TEXT = "AKIAIOSFODNN7EXAMPLE is an AWS key"
    VALUE = "AKIAIOSFODNN7EXAMPLE"
    START = 0
    END = len(VALUE)

    def test_first_entity_wins(self):
        """First entity is kept; second entity overlapping it is skipped."""
        entities = [
            _det("AWS_SECRET", 0, 10, self.VALUE[:10]),   # wins
            _det("API_KEY",    5, 20, self.VALUE[5:20]),  # overlaps → skipped
        ]
        masked, mapping = maskPayload(self.TEXT, entities)
        assert "[AWS_SECRET_1]" in masked
        assert "[API_KEY" not in masked

    def test_overlapping_entity_absent_from_mapping(self):
        """The skipped entity must NOT appear in the mapping store (no extras)."""
        entities = [
            _det("AWS_SECRET", 0, 10, self.VALUE[:10]),
            _det("API_KEY",    5, 20, self.VALUE[5:20]),
        ]
        _, mapping = maskPayload(self.TEXT, entities)
        assert set(mapping.keys()) == {"AWS_SECRET_1"}

    def test_exact_same_span_two_categories(self):
        """Same start/end with two different categories — first in list wins."""
        entities = [
            _det("CARD",    0, 16, self.VALUE[:16]),
            _det("API_KEY", 0, 16, self.VALUE[:16]),  # same span → skipped
        ]
        _, mapping = maskPayload(self.TEXT, entities)
        assert len(mapping) == 1
        assert "CARD_1" in mapping

    def test_subset_span_skipped(self):
        """An entity whose span is fully contained within another is skipped."""
        entities = [
            _det("TOKEN", 0, 20, self.VALUE),       # full span
            _det("EMAIL", 3, 10, self.VALUE[3:10]), # subset → skipped
        ]
        _, mapping = maskPayload(self.TEXT, entities)
        assert set(mapping.keys()) == {"TOKEN_1"}

    def test_non_overlapping_adjacent_both_kept(self):
        """Adjacent (touching but not overlapping) entities are both kept."""
        # TEXT: "AB|CD" — split at position 2
        text = "ABCD"
        entities = [
            _det("ALPHA", 0, 2, "AB"),
            _det("BETA",  2, 4, "CD"),
        ]
        masked, mapping = maskPayload(text, entities)
        assert masked == "[ALPHA_1][BETA_1]"
        assert set(mapping.keys()) == {"ALPHA_1", "BETA_1"}

    def test_entities_supplied_unsorted_still_correct(self):
        """Entities given in reverse order are sorted internally by start."""
        text = "Email alice@test.com phone 010-0000-0000"
        email = "alice@test.com"
        phone = "010-0000-0000"
        e_start = text.index(email)
        p_start = text.index(phone)
        # Intentionally supply phone first, email second
        entities = [
            _det("PHONE", p_start, p_start + len(phone), phone),
            _det("EMAIL", e_start, e_start + len(email), email),
        ]
        masked, mapping = maskPayload(text, entities)
        # Both should be replaced; EMAIL appears first in text so gets EMAIL_1
        assert "[EMAIL_1]" in masked
        assert "[PHONE_1]" in masked
        assert email not in masked
        assert phone not in masked
        assert set(mapping.keys()) == {"EMAIL_1", "PHONE_1"}


# ══════════════════════════════════════════════════════════════════════════════
# 7. Entity at text boundaries
# ══════════════════════════════════════════════════════════════════════════════

class TestTextBoundaries:

    def test_entity_at_very_start(self):
        text = "alice@corp.io is the contact"
        entity = _det("EMAIL", 0, len("alice@corp.io"), "alice@corp.io")
        masked, mapping = maskPayload(text, [entity])
        assert masked.startswith("[EMAIL_1]")
        assert mapping == {"EMAIL_1": "alice@corp.io"}

    def test_entity_at_very_end(self):
        text = "Contact us at alice@corp.io"
        email = "alice@corp.io"
        start = len(text) - len(email)
        entity = _det("EMAIL", start, len(text), email)
        masked, mapping = maskPayload(text, [entity])
        assert masked.endswith("[EMAIL_1]")
        assert mapping == {"EMAIL_1": email}

    def test_entity_spans_entire_text(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        entity = _det("AWS_SECRET", 0, len(text), text)
        masked, mapping = maskPayload(text, [entity])
        assert masked == "[AWS_SECRET_1]"
        assert mapping == {"AWS_SECRET_1": text}

    def test_text_with_only_non_entity_content(self):
        text = "Nothing to redact here."
        # Entity pointing to text that exists but caller labels it EMAIL
        entity = _det("EMAIL", 0, 7, "Nothing")
        masked, mapping = maskPayload(text, [entity])
        assert "[EMAIL_1]" in masked
        assert "Nothing" not in masked[:8]
        assert mapping == {"EMAIL_1": "Nothing"}


# ══════════════════════════════════════════════════════════════════════════════
# 8. Zero-length and inverted spans
# ══════════════════════════════════════════════════════════════════════════════

class TestInvalidSpans:

    def test_zero_length_span_skipped(self):
        """A span where start == end is silently skipped."""
        text = "hello world"
        entity = _dict_entity("EMAIL", 5, 5, "")
        masked, mapping = maskPayload(text, [entity])
        assert masked == text
        assert mapping == {}

    def test_inverted_span_skipped(self):
        """A span where end < start is silently skipped."""
        text = "hello world"
        entity = _dict_entity("EMAIL", 7, 3, "test")
        masked, mapping = maskPayload(text, [entity])
        assert masked == text
        assert mapping == {}

    def test_zero_length_among_valid_only_valid_replaced(self):
        """Zero-length spans are skipped; valid ones are still processed."""
        text = "alice@corp.io here"
        email = "alice@corp.io"
        start = 0
        entities = [
            _dict_entity("GHOST", 5, 5, ""),            # zero-length → skip
            _dict_entity("EMAIL", start, len(email), email),  # valid → replace
        ]
        masked, mapping = maskPayload(text, entities)
        assert "[EMAIL_1]" in masked
        assert set(mapping.keys()) == {"EMAIL_1"}


# ══════════════════════════════════════════════════════════════════════════════
# 9. Unicode entities
# ══════════════════════════════════════════════════════════════════════════════

class TestUnicodeEntities:

    def test_korean_name_entity(self):
        text = "성명: 김철수 씨"
        name = "김철수"
        start = text.index(name)
        entity = _det("PERSON", start, start + len(name), name)
        masked, mapping = maskPayload(text, [entity])
        assert "[PERSON_1]" in masked
        assert name not in masked
        assert mapping == {"PERSON_1": name}

    def test_mixed_unicode_and_ascii(self):
        text = "이름: 이영희, email: park@test.com"
        name = "이영희"
        email = "park@test.com"
        ns = text.index(name)
        es = text.index(email)
        entities = [
            _det("PERSON", ns, ns + len(name),   name),
            _det("EMAIL",  es, es + len(email),  email),
        ]
        masked, mapping = maskPayload(text, entities)
        assert "[PERSON_1]" in masked
        assert "[EMAIL_1]" in masked
        assert name not in masked
        assert email not in masked
        assert set(mapping.keys()) == {"PERSON_1", "EMAIL_1"}
        assert mapping["PERSON_1"] == name
        assert mapping["EMAIL_1"] == email


# ══════════════════════════════════════════════════════════════════════════════
# 10. Minimal attribute-based entity object (not a Detection)
# ══════════════════════════════════════════════════════════════════════════════

class TestMinimalEntityObject:

    def test_custom_dataclass_entity(self):
        text = "key = sk-ant-api03-" + "A" * 50
        value = "sk-ant-api03-" + "A" * 50
        start = text.index(value)
        entity = MinimalEntity("API_KEY", start, start + len(value), value)
        masked, mapping = maskPayload(text, [entity])
        assert "[API_KEY_1]" in masked
        assert value not in masked
        assert mapping == {"API_KEY_1": value}


# ══════════════════════════════════════════════════════════════════════════════
# 11. Monotonically increasing per-category indices, no gaps
# ══════════════════════════════════════════════════════════════════════════════

class TestCounterMonotonicity:

    def test_indices_have_no_gaps(self):
        text = "a@b.com, c@d.com, e@f.com, g@h.com, i@j.com"
        emails = ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com"]
        entities = []
        pos = 0
        for em in emails:
            s = text.index(em, pos)
            entities.append(_det("EMAIL", s, s + len(em), em))
            pos = s + len(em)
        masked, mapping = maskPayload(text, entities)
        for i, em in enumerate(emails, start=1):
            key = f"EMAIL_{i}"
            assert key in mapping, f"Expected {key} in mapping"
            assert mapping[key] == em
        assert len(mapping) == 5

    def test_indices_start_at_1_each_call(self):
        """Each fresh call to maskPayload starts counters from 1."""
        text = "alice@corp.io is here"
        email = "alice@corp.io"
        s = text.index(email)
        entity = _det("EMAIL", s, s + len(email), email)

        _, map1 = maskPayload(text, [entity])
        _, map2 = maskPayload(text, [entity])

        # Both calls produce EMAIL_1 (fresh counter each call)
        assert "EMAIL_1" in map1
        assert "EMAIL_1" in map2

    def test_each_call_is_independent(self):
        """Two calls do not share counter state."""
        text = "alice@corp.io"
        entity = _det("EMAIL", 0, len(text), text)

        _, map1 = maskPayload(text, [entity])
        # Simulate "second email" by calling again with same entity
        _, map2 = maskPayload(text, [entity])

        assert map1 == {"EMAIL_1": text}
        assert map2 == {"EMAIL_1": text}


# ══════════════════════════════════════════════════════════════════════════════
# 12. Reverse-mapping correctness — no extras, exact entries
# ══════════════════════════════════════════════════════════════════════════════

class TestMappingCorrectness:

    def test_mapping_is_exact_no_extras_single(self):
        text = "secret = hunter2pass"
        value = "hunter2pass"
        s = text.index(value)
        entity = _det("PASSWORD", s, s + len(value), value)
        _, mapping = maskPayload(text, [entity])
        # Exactly one entry
        assert len(mapping) == 1
        assert mapping == {"PASSWORD_1": value}

    def test_mapping_is_exact_no_extras_multi(self):
        text = "a@b.com and c@d.com"
        entities = [
            _det("EMAIL", 0, 7,    "a@b.com"),
            _det("EMAIL", 12, 19,  "c@d.com"),
        ]
        _, mapping = maskPayload(text, entities)
        assert len(mapping) == 2
        assert mapping == {"EMAIL_1": "a@b.com", "EMAIL_2": "c@d.com"}

    def test_mapping_keys_are_bare_no_brackets(self):
        text = "test@example.com"
        entity = _det("EMAIL", 0, len(text), text)
        _, mapping = maskPayload(text, [entity])
        for key in mapping:
            assert not key.startswith("["), f"Key {key!r} should not have brackets"
            assert not key.endswith("]"),   f"Key {key!r} should not have brackets"

    def test_masked_text_has_brackets(self):
        text = "test@example.com"
        entity = _det("EMAIL", 0, len(text), text)
        masked, mapping = maskPayload(text, [entity])
        for key in mapping:
            assert f"[{key}]" in masked, f"[{key}] should be present in masked text"

    def test_mapping_value_equals_original_field(self):
        """Mapping value is taken from entity.original, not text[start:end]."""
        text = "visit example.com/path?q=1"
        # Deliberately provide a mismatched original (unusual but tests the contract)
        entity = _dict_entity("URL", 6, 17, "example.com")  # text[6:17] = "example.com"
        _, mapping = maskPayload(text, [entity])
        assert mapping["URL_1"] == "example.com"

    def test_skipped_overlap_absent_from_mapping(self):
        """An overlapping (skipped) entity does not appear in the mapping."""
        text = "AKIAIOSFODNN7EXAMPLE"
        entities = [
            _det("AWS_SECRET", 0, 20, text),   # kept
            _det("API_KEY",    0, 15, text[:15]),  # overlaps → skipped
        ]
        _, mapping = maskPayload(text, entities)
        assert "API_KEY_1" not in mapping
        assert len(mapping) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 13. Roundtrip: masked text + mapping → original text
# ══════════════════════════════════════════════════════════════════════════════

class TestMaskedTextRoundtrip:

    def test_roundtrip_single_entity(self):
        text = "Contact alice@corp.io for help."
        email = "alice@corp.io"
        s = text.index(email)
        entity = _det("EMAIL", s, s + len(email), email)

        masked, mapping = maskPayload(text, [entity])
        # Restore by replacing [TOKEN] with original
        restored = masked
        for token, original in mapping.items():
            restored = restored.replace(f"[{token}]", original)
        assert restored == text

    def test_roundtrip_multi_entity(self):
        text = "From alice@corp.io cc bob@corp.io call 010-9999-8888"
        entities = [
            _det("EMAIL", text.index("alice@corp.io"), text.index("alice@corp.io") + 13, "alice@corp.io"),
            _det("EMAIL", text.index("bob@corp.io"),   text.index("bob@corp.io")   + 11, "bob@corp.io"),
            _det("PHONE", text.index("010-9999-8888"), text.index("010-9999-8888") + 13, "010-9999-8888"),
        ]
        masked, mapping = maskPayload(text, entities)
        restored = masked
        for token, original in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
            restored = restored.replace(f"[{token}]", original)
        assert restored == text


# ══════════════════════════════════════════════════════════════════════════════
# 14. Top-level import smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestTopLevelImport:

    def test_top_level_export_is_same_function(self):
        """pii_guard.maskPayload is the same callable as pii_guard.masker.maskPayload."""
        from pii_guard.masker import maskPayload as direct
        assert maskPayload_toplevel is direct

    def test_top_level_basic_usage(self):
        text = "api_key = sk-" + "x" * 30
        value = "sk-" + "x" * 30
        s = text.index(value)
        entity = _dict_entity("API_KEY", s, s + len(value), value)
        masked, mapping = maskPayload_toplevel(text, [entity])
        assert "[API_KEY_1]" in masked
        assert mapping["API_KEY_1"] == value


# ══════════════════════════════════════════════════════════════════════════════
# 15. Parametric purity — each call is independent (no shared state)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n_calls", [2, 5, 10])
def test_repeated_calls_produce_same_result(n_calls: int):
    """Repeated calls with the same args always produce the same output."""
    text = "alice@test.com is the contact"
    email = "alice@test.com"
    s = text.index(email)
    entity = _det("EMAIL", s, s + len(email), email)

    results = [maskPayload(text, [entity]) for _ in range(n_calls)]
    masked_set   = {r[0] for r in results}
    mapping_keys = [set(r[1].keys()) for r in results]

    assert len(masked_set) == 1,              "All calls must produce the same masked text"
    assert all(k == {"EMAIL_1"} for k in mapping_keys), \
        "All calls must produce mapping {EMAIL_1: ...}"


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
def test_parametric_category_placeholders(category: str, value: str):
    """For every category, placeholder = [CATEGORY_1] and mapping is exact."""
    text = f"data: {value} end"
    s = text.index(value)
    entity = _dict_entity(category, s, s + len(value), value)
    masked, mapping = maskPayload(text, [entity])

    expected_token  = f"{category}_1"
    expected_masked = f"data: [{expected_token}] end"

    assert masked == expected_masked, (
        f"[{category}] masked text mismatch: {masked!r} != {expected_masked!r}"
    )
    assert mapping == {expected_token: value}, (
        f"[{category}] mapping mismatch: {mapping!r}"
    )
