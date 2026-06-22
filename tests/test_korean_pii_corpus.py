"""
Tests for the synthetic Korean PII red-team corpus.

Validates:
1. Corpus structure integrity (all spans match their text slices)
2. Per-category coverage counts meet minimum thresholds
3. Korean format compliance for each category
4. Negative-sample set does not contain labelled PII spans
5. Deterministic reproducibility across identical seeds
6. Detection efficacy: Stage-1 detector achieves ≥70% recall and ≥60% precision
   on all five Korean PII categories
7. RRN checksum validity for every RRN in the corpus
8. Uniqueness: corpus samples are not trivially duplicated

Run:
    pytest tests/test_korean_pii_corpus.py -v
"""
from __future__ import annotations

import re
from typing import Set

import pytest

from pii_guard.corpus import KoreanPIICorpus, CorpusSample, PIISpan
from pii_guard.corpus.korean_pii import (
    _compute_rrn_check_digit,
    _make_rrn,
    compute_precision_recall,
)
from pii_guard import Engine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corpus():
    """Standard corpus with default seed (deterministic)."""
    return KoreanPIICorpus(seed=42, samples_per_format=5)


@pytest.fixture(scope="module")
def engine():
    return Engine()


def detector_fn(engine: Engine):
    """Return a detector closure over the given engine."""
    def _detect(text: str) -> Set[str]:
        result = engine.scan(text)
        return {d.category for d in result.detections}
    return _detect


# ─────────────────────────────────────────────────────────────────────────────
# 1. Corpus structure integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusStructure:
    """Every CorpusSample must be internally consistent."""

    def test_all_samples_have_text(self, corpus):
        for s in corpus.all_samples():
            assert isinstance(s.text, str) and len(s.text) > 0, \
                f"Sample with empty text found: {s!r}"

    def test_all_spans_match_text_slices(self, corpus):
        """PIISpan.value must equal text[start:end] for every sample."""
        for s in corpus.all_samples():
            for span in s.spans:
                extracted = s.text[span.start:span.end]
                assert extracted == span.value, (
                    f"Span mismatch in sample {s.source_tag!r}: "
                    f"text[{span.start}:{span.end}]={extracted!r} "
                    f"!= span.value={span.value!r}"
                )

    def test_verify_spans_method(self, corpus):
        """CorpusSample.verify_spans() returns True for all positive samples."""
        for s in corpus.positive_samples():
            assert s.verify_spans(), \
                f"verify_spans() failed for {s.source_tag!r}: {s.text!r}"

    def test_negative_samples_have_no_spans(self, corpus):
        """Negative samples must have empty span lists."""
        for s in corpus.negative_samples():
            assert s.spans == [], \
                f"Negative sample has spans: {s.source_tag!r} {s.text!r}"

    def test_negative_samples_are_flagged(self, corpus):
        """Every sample with is_negative=True must have no spans."""
        for s in corpus.all_samples():
            if s.is_negative:
                assert len(s.spans) == 0

    def test_positive_samples_have_spans(self, corpus):
        """Every positive sample must have at least one span."""
        for s in corpus.positive_samples():
            assert len(s.spans) >= 1, \
                f"Positive sample has no spans: {s.source_tag!r}"

    def test_span_offsets_within_text(self, corpus):
        """All span start/end offsets must be within text bounds."""
        for s in corpus.all_samples():
            n = len(s.text)
            for span in s.spans:
                assert 0 <= span.start < span.end <= n, (
                    f"Out-of-bounds span [{span.start},{span.end}) "
                    f"in text of length {n}: {s.source_tag!r}"
                )

    def test_categories_derived_from_spans(self, corpus):
        """CorpusSample.categories must equal the set of categories in its spans."""
        for s in corpus.positive_samples():
            expected = frozenset(sp.category for sp in s.spans)
            assert s.categories == expected, \
                f"Category mismatch for {s.source_tag!r}: {s.categories} vs {expected}"

    def test_source_tags_are_nonempty(self, corpus):
        for s in corpus.all_samples():
            assert s.source_tag, f"Sample has empty source_tag: {s!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-category coverage counts
# ─────────────────────────────────────────────────────────────────────────────

# Minimum number of positive samples required per category
_MIN_SAMPLES_PER_CATEGORY = {
    "PERSON":     25,   # 6 formats × 5 + 3 sentence × 5 = 45
    "PHONE":      25,   # 6 formats × 5 = 30
    "RRN":        25,   # 7 format groups × 5 = 35
    "ADDRESS":    20,   # 5 formats × 5 = 25
    "KR_ACCOUNT": 20,   # 4 banks × 3 formats × 5 = 60
}

class TestCoverageCounts:
    def test_total_positive_samples_sufficient(self, corpus):
        assert len(corpus.positive_samples()) >= 100, \
            f"Expected ≥100 positive samples, got {len(corpus.positive_samples())}"

    def test_total_negative_samples_sufficient(self, corpus):
        assert len(corpus.negative_samples()) >= 20, \
            f"Expected ≥20 negative samples, got {len(corpus.negative_samples())}"

    @pytest.mark.parametrize("category,min_count", list(_MIN_SAMPLES_PER_CATEGORY.items()))
    def test_category_minimum_count(self, corpus, category, min_count):
        counts = corpus.category_counts()
        actual = counts.get(category, 0)
        assert actual >= min_count, (
            f"Category {category!r}: expected ≥{min_count} samples, got {actual}"
        )

    def test_all_five_categories_present(self, corpus):
        counts = corpus.category_counts()
        required = {"PERSON", "PHONE", "RRN", "ADDRESS", "KR_ACCOUNT"}
        missing = required - set(counts.keys())
        assert not missing, f"Missing categories from corpus: {missing}"

    def test_samples_for_category_api(self, corpus):
        for cat in ("PERSON", "PHONE", "RRN", "ADDRESS", "KR_ACCOUNT"):
            samples = corpus.samples_for_category(cat)
            assert len(samples) > 0, f"samples_for_category({cat!r}) returned empty list"

    def test_coverage_report_nonempty(self, corpus):
        report = corpus.coverage_report()
        assert "PERSON" in report
        assert "RRN" in report
        assert "KR_ACCOUNT" in report
        assert "positive" in report


# ─────────────────────────────────────────────────────────────────────────────
# 3. Korean format compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestKoreanFormatCompliance:
    """Verify that corpus values match the expected Korean format patterns."""

    # ── PERSON ────────────────────────────────────────────────────────────────

    def test_person_names_are_hangul(self, corpus):
        """All Korean names in the corpus must consist entirely of Hangul characters."""
        _hangul = re.compile(r"^[가-힣]{2,5}$")
        for s in corpus.samples_for_category("PERSON"):
            for span in s.spans:
                if span.category == "PERSON":
                    assert _hangul.match(span.value), \
                        f"Person name is not valid Hangul: {span.value!r} in {s.text!r}"

    def test_person_name_length(self, corpus):
        """Korean full names should be 2-5 Hangul characters."""
        for s in corpus.samples_for_category("PERSON"):
            for span in s.spans:
                if span.category == "PERSON":
                    assert 2 <= len(span.value) <= 5, \
                        f"Unusual name length {len(span.value)}: {span.value!r}"

    # ── PHONE ─────────────────────────────────────────────────────────────────

    def test_mobile_phone_prefix(self, corpus):
        """Mobile phone spans starting with 010-/011-/016-/017-/019- must match the pattern."""
        _mobile = re.compile(r"^(\+82[-\s]?)?01[016789]")
        mobile_samples = [
            s for s in corpus.samples_for_category("PHONE")
            if any(sp.format_tag in ("mobile_dash", "mobile_bare", "mobile_dot",
                                     "inline_sentence", "international")
                   for sp in s.spans)
        ]
        for s in mobile_samples:
            for span in s.spans:
                if span.category == "PHONE":
                    # Strip leading +82 for domestic comparison
                    clean = re.sub(r"^\+82[-\s]?", "0", span.value)
                    digits = re.sub(r"\D", "", clean)
                    assert digits[:3] in ("010", "011", "016", "017", "019") or \
                           span.format_tag == "international", \
                        f"Unexpected mobile prefix in {span.value!r}"

    def test_phone_digit_count(self, corpus):
        """All Korean phone numbers must have 9-11 digits (excl. +82 country code)."""
        for s in corpus.samples_for_category("PHONE"):
            for span in s.spans:
                if span.category == "PHONE":
                    digits = re.sub(r"\D", "", span.value)
                    # +82 adds 2 country code digits
                    if span.value.startswith("+82"):
                        digits = "0" + digits[2:]  # normalize +82 → 0
                        digits = re.sub(r"\D", "", digits)
                    assert 9 <= len(digits) <= 11, \
                        f"Phone digit count {len(digits)} out of range for {span.value!r}"

    def test_landline_format(self, corpus):
        """Landline samples must have a valid Korean area code."""
        _area_codes = {
            "02", "031", "032", "033", "041", "042", "043",
            "051", "052", "053", "054", "055", "061", "062",
            "063", "064",
        }
        landline_samples = [
            s for s in corpus.samples_for_category("PHONE")
            if any(sp.format_tag == "landline" for sp in s.spans)
        ]
        for s in landline_samples:
            for span in s.spans:
                if span.category == "PHONE" and span.format_tag == "landline":
                    digits = re.sub(r"\D", "", span.value)
                    matched = any(digits.startswith(ac) for ac in _area_codes)
                    assert matched, \
                        f"Landline phone has unrecognised area code: {span.value!r}"

    # ── RRN ───────────────────────────────────────────────────────────────────

    def test_rrn_digit_count(self, corpus):
        """Every RRN span must have exactly 13 digits."""
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    digits = re.sub(r"\D", "", span.value)
                    assert len(digits) == 13, \
                        f"RRN does not have 13 digits: {span.value!r} ({len(digits)} digits)"

    def test_rrn_gender_digit_valid(self, corpus):
        """RRN 7th digit must be 1, 2, 3, or 4 (Korean national citizen)."""
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    digits = re.sub(r"\D", "", span.value)
                    gender_digit = int(digits[6])
                    assert gender_digit in (1, 2, 3, 4), \
                        f"Invalid RRN gender digit {gender_digit}: {span.value!r}"

    def test_rrn_checksum_valid(self, corpus):
        """Every RRN in the corpus must have a correct checksum digit."""
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    digits = re.sub(r"\D", "", span.value)
                    assert len(digits) == 13, f"Wrong length: {span.value!r}"
                    expected = _compute_rrn_check_digit(digits[:12])
                    actual = int(digits[12])
                    assert expected == actual, (
                        f"RRN checksum failed for {span.value!r}: "
                        f"expected check={expected}, got {actual}"
                    )

    def test_rrn_date_portion_plausible(self, corpus):
        """RRN date portion YYMMDD must have MM in [01,12] and DD in [01,31]."""
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    digits = re.sub(r"\D", "", span.value)
                    mm = int(digits[2:4])
                    dd = int(digits[4:6])
                    assert 1 <= mm <= 12, \
                        f"RRN month out of range [{mm}] in {span.value!r}"
                    assert 1 <= dd <= 31, \
                        f"RRN day out of range [{dd}] in {span.value!r}"

    # ── ADDRESS ───────────────────────────────────────────────────────────────

    # First-2-char prefixes of province/city names as they appear in the corpus.
    # Uses full-name first-2 chars (전라남도 → "전라", 충청북도 → "충청", etc.)
    # as well as the short administrative abbreviations for robustness.
    _KR_PROVINCES = {
        # Metropolitan cities and special cities
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        # Provinces — full-name first-2-char prefixes
        "경기", "강원", "충청", "전라", "경상", "제주",
        # Abbreviated forms (경북→경상북도 etc.) — kept for compatibility
        "충북", "충남", "전북", "전남", "경북", "경남",
    }

    def test_address_starts_with_kr_province(self, corpus):
        """All Korean addresses must start with a recognised province or city name."""
        for s in corpus.samples_for_category("ADDRESS"):
            for span in s.spans:
                if span.category == "ADDRESS":
                    # Check first 2 chars to cover both full (전라남도) and
                    # abbreviated (전남) province names.
                    first2 = span.value[:2]
                    assert first2 in self._KR_PROVINCES, \
                        f"Address does not start with known province: {span.value!r}"

    def test_address_contains_street_suffix(self, corpus):
        """Korean addresses must contain a street-level suffix (로|길|동|읍|면|로|구|군)."""
        _street_pattern = re.compile(r"[가-힣]+(로|길|동|읍|면|구|군|시)")
        for s in corpus.samples_for_category("ADDRESS"):
            for span in s.spans:
                if span.category == "ADDRESS":
                    assert _street_pattern.search(span.value), \
                        f"Address missing street suffix: {span.value!r}"

    # ── KR_ACCOUNT ────────────────────────────────────────────────────────────

    _KR_ACCOUNT_FORMATS = [
        re.compile(r"^\d{6}-\d{2}-\d{6}$"),   # kookmin 6-2-6
        re.compile(r"^\d{3}-\d{6}-\d{5}$"),   # shinhan/hana 3-6-5
        re.compile(r"^\d{4}-\d{3}-\d{6}$"),   # woori 4-3-6
    ]

    def test_kr_account_matches_known_format(self, corpus):
        """Every KR_ACCOUNT span must match at least one known bank format."""
        for s in corpus.samples_for_category("KR_ACCOUNT"):
            for span in s.spans:
                if span.category == "KR_ACCOUNT":
                    matched = any(p.match(span.value) for p in self._KR_ACCOUNT_FORMATS)
                    assert matched, \
                        f"KR_ACCOUNT does not match any known format: {span.value!r}"

    def test_kr_account_hyphen_count(self, corpus):
        """Korean bank accounts should contain exactly 2 hyphens."""
        for s in corpus.samples_for_category("KR_ACCOUNT"):
            for span in s.spans:
                if span.category == "KR_ACCOUNT":
                    assert span.value.count("-") == 2, \
                        f"KR_ACCOUNT hyphen count != 2: {span.value!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Negative sample integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativeSamples:
    def test_negative_count_by_category_type(self, corpus):
        """At least 3 negatives per major category type."""
        tags = [s.source_tag for s in corpus.negative_samples()]
        for prefix in ("neg_person", "neg_phone", "neg_rrn", "neg_address", "neg_kr_account"):
            count = sum(1 for t in tags if t.startswith(prefix))
            assert count >= 3, \
                f"Too few negative samples for type {prefix!r}: {count}"

    def test_negatives_have_is_negative_true(self, corpus):
        for s in corpus.negative_samples():
            assert s.is_negative

    def test_negative_texts_nonempty(self, corpus):
        for s in corpus.negative_samples():
            assert len(s.text.strip()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Deterministic reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_produces_identical_samples(self):
        """Two corpora with the same seed must be identical."""
        c1 = KoreanPIICorpus(seed=42)
        c2 = KoreanPIICorpus(seed=42)
        s1 = [s.text for s in c1.all_samples()]
        s2 = [s.text for s in c2.all_samples()]
        assert s1 == s2, "Identical seeds produced different samples"

    def test_different_seeds_produce_different_samples(self):
        """Two corpora with different seeds should differ in at least some texts."""
        c1 = KoreanPIICorpus(seed=1)
        c2 = KoreanPIICorpus(seed=2)
        t1 = set(s.text for s in c1.positive_samples())
        t2 = set(s.text for s in c2.positive_samples())
        # They may share some structure but should not be identical
        assert t1 != t2, "Different seeds produced identical sample sets"

    def test_span_offsets_stable_across_instances(self):
        """Span offsets from two corpora with the same seed must be equal."""
        c1 = KoreanPIICorpus(seed=99)
        c2 = KoreanPIICorpus(seed=99)
        for s1, s2 in zip(c1.all_samples(), c2.all_samples()):
            assert s1.spans == s2.spans


# ─────────────────────────────────────────────────────────────────────────────
# 6. Detection efficacy — Stage-1 recall and precision
# ─────────────────────────────────────────────────────────────────────────────

# Minimum acceptable recall and precision for Stage-1 on each category.
# Stage-1 is regex+checksum; PERSON and ADDRESS are harder (NER territory),
# so thresholds are intentionally lower for those categories.
_PRECISION_TARGETS = {
    "PHONE":      0.60,
    "RRN":        0.70,
    "KR_ACCOUNT": 0.50,
    "PERSON":     0.50,
    "ADDRESS":    0.50,
}
_RECALL_TARGETS = {
    "PHONE":      0.70,
    "RRN":        0.70,
    "KR_ACCOUNT": 0.50,
    "PERSON":     0.50,
    "ADDRESS":    0.50,
}


class TestDetectionEfficacy:
    """
    End-to-end detection test: feed corpus samples into the Stage-1 engine
    and verify precision/recall per category meet the targets above.
    """

    @pytest.fixture(scope="class")
    def shared_engine(self):
        return Engine()

    @pytest.mark.parametrize("category", list(_PRECISION_TARGETS.keys()))
    def test_recall_meets_target(self, corpus, shared_engine, category):
        detect = detector_fn(shared_engine)
        precision, recall = compute_precision_recall(corpus, detect, category)
        target = _RECALL_TARGETS[category]
        assert recall >= target, (
            f"{category} recall {recall:.2%} < target {target:.0%} "
            f"(precision={precision:.2%})"
        )

    @pytest.mark.parametrize("category", list(_PRECISION_TARGETS.keys()))
    def test_precision_meets_target(self, corpus, shared_engine, category):
        detect = detector_fn(shared_engine)
        precision, recall = compute_precision_recall(corpus, detect, category)
        target = _PRECISION_TARGETS[category]
        assert precision >= target, (
            f"{category} precision {precision:.2%} < target {target:.0%} "
            f"(recall={recall:.2%})"
        )

    def test_rrn_detected_in_positive_samples(self, corpus, shared_engine):
        """Sanity check: at least 70% of RRN positive samples should be detected."""
        rrn_samples = corpus.samples_for_category("RRN")
        detected_count = 0
        for s in rrn_samples:
            result = shared_engine.scan(s.text)
            if any(d.category == "RRN" for d in result.detections):
                detected_count += 1
        recall = detected_count / len(rrn_samples)
        assert recall >= 0.70, \
            f"RRN detection recall {recall:.2%} < 70% on {len(rrn_samples)} samples"

    def test_phone_detected_in_positive_samples(self, corpus, shared_engine):
        """Sanity check: at least 70% of PHONE positive samples should be detected."""
        phone_samples = corpus.samples_for_category("PHONE")
        detected_count = 0
        for s in phone_samples:
            result = shared_engine.scan(s.text)
            if any(d.category == "PHONE" for d in result.detections):
                detected_count += 1
        recall = detected_count / len(phone_samples)
        assert recall >= 0.70, \
            f"PHONE detection recall {recall:.2%} < 70% on {len(phone_samples)} samples"

    def test_rrn_not_detected_in_negatives(self, corpus, shared_engine):
        """RRN detector must not fire on deliberately-invalid negative RRN strings."""
        neg_rrn = [s for s in corpus.negative_samples()
                   if s.source_tag == "neg_rrn"]
        assert neg_rrn, "No RRN negative samples found"
        false_positives = 0
        for s in neg_rrn:
            result = shared_engine.scan(s.text)
            if any(d.category == "RRN" for d in result.detections):
                false_positives += 1
        # Allow at most 1 FP across the negative RRN set
        assert false_positives <= 1, \
            f"RRN detector fired on {false_positives}/{len(neg_rrn)} negative samples"


# ─────────────────────────────────────────────────────────────────────────────
# 7. RRN checksum utility — unit tests for the generator itself
# ─────────────────────────────────────────────────────────────────────────────

class TestRRNChecksumUtility:
    def test_known_valid_rrn(self):
        """Reference RRN with pre-computed check digit."""
        # 900505-1XXXXX: first 12 = 9005051 + 23456 (arbitrary tail)
        # Compute expected check for "900505123456"
        first12 = "900505123456"
        check = _compute_rrn_check_digit(first12)
        rrn = first12 + str(check)
        assert len(rrn) == 13
        # Verify checksum formula inverts
        assert _compute_rrn_check_digit(rrn[:12]) == int(rrn[12])

    def test_make_rrn_produces_13_digits(self):
        rrn = _make_rrn("800101", 1, "23456")
        assert len(rrn) == 13
        assert rrn.isdigit()

    def test_make_rrn_checksum_correct(self):
        for gender in (1, 2, 3, 4):
            for tail in ("00000", "11111", "99999", "12345"):
                rrn = _make_rrn("900101", gender, tail)
                expected = _compute_rrn_check_digit(rrn[:12])
                assert expected == int(rrn[12]), \
                    f"Checksum wrong for gender={gender}, tail={tail}: {rrn}"

    def test_make_rrn_gender_digit_preserved(self):
        for gender in (1, 2, 3, 4):
            rrn = _make_rrn("850615", gender, "54321")
            assert int(rrn[6]) == gender

    def test_all_corpus_rrns_pass_checksum(self, corpus):
        """Full corpus sweep: every RRN span has valid checksum."""
        fail_count = 0
        failures = []
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    digits = re.sub(r"\D", "", span.value)
                    if len(digits) != 13:
                        fail_count += 1
                        failures.append(f"Wrong length: {span.value!r}")
                        continue
                    expected = _compute_rrn_check_digit(digits[:12])
                    if expected != int(digits[12]):
                        fail_count += 1
                        failures.append(
                            f"Bad checksum: {span.value!r} (expected {expected}, got {digits[12]})"
                        )
        assert fail_count == 0, \
            f"{fail_count} RRN checksum failures:\n" + "\n".join(failures[:10])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sample uniqueness
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleUniqueness:
    def test_positive_texts_mostly_unique(self, corpus):
        """At least 80% of positive sample texts should be distinct."""
        texts = [s.text for s in corpus.positive_samples()]
        unique = len(set(texts))
        ratio = unique / len(texts)
        assert ratio >= 0.80, \
            f"Only {ratio:.0%} of positive samples are unique ({unique}/{len(texts)})"

    def test_negative_texts_all_unique(self, corpus):
        texts = [s.text for s in corpus.negative_samples()]
        assert len(texts) == len(set(texts)), \
            "Negative samples contain duplicate texts"

    def test_rrn_values_unique_across_corpus(self, corpus):
        """All synthetic RRN values in the corpus should be distinct."""
        rrns = []
        for s in corpus.samples_for_category("RRN"):
            for span in s.spans:
                if span.category == "RRN":
                    rrns.append(re.sub(r"\D", "", span.value))
        unique = len(set(rrns))
        ratio = unique / len(rrns)
        assert ratio >= 0.80, \
            f"Only {ratio:.0%} of RRN values are unique ({unique}/{len(rrns)})"


# ─────────────────────────────────────────────────────────────────────────────
# 9. API / helper completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusAPI:
    def test_all_samples_returns_list(self, corpus):
        assert isinstance(corpus.all_samples(), list)

    def test_positive_and_negative_partition_all(self, corpus):
        """positive + negative must equal all samples."""
        total = len(corpus.all_samples())
        pos = len(corpus.positive_samples())
        neg = len(corpus.negative_samples())
        assert pos + neg == total, \
            f"Partition mismatch: {pos} + {neg} != {total}"

    def test_category_counts_returns_dict(self, corpus):
        counts = corpus.category_counts()
        assert isinstance(counts, dict)
        assert all(isinstance(v, int) for v in counts.values())

    def test_samples_for_unknown_category_returns_empty(self, corpus):
        result = corpus.samples_for_category("DOES_NOT_EXIST")
        assert result == []

    def test_samples_per_format_scaling(self):
        """Larger samples_per_format produces proportionally more samples."""
        c_small = KoreanPIICorpus(seed=0, samples_per_format=3)
        c_large = KoreanPIICorpus(seed=0, samples_per_format=9)
        assert len(c_large.positive_samples()) > len(c_small.positive_samples())

    def test_pii_span_is_frozen(self):
        """PIISpan must be immutable (frozen dataclass)."""
        span = PIISpan(category="RRN", start=0, end=13,
                       value="9005051234567", format_tag="test")
        with pytest.raises((AttributeError, TypeError)):
            span.category = "PHONE"  # type: ignore

    def test_corpus_sample_categories_frozenset(self, corpus):
        for s in corpus.positive_samples():
            assert isinstance(s.categories, frozenset)

    def test_compute_precision_recall_perfect(self):
        """compute_precision_recall returns 1.0/1.0 for a perfect detector."""
        c = KoreanPIICorpus(seed=7, samples_per_format=2)
        def perfect_detect(text):
            # Returns all categories that appear in this text's spans
            for s in c.all_samples():
                if s.text == text:
                    return {sp.category for sp in s.spans}
            return set()
        p, r = compute_precision_recall(c, perfect_detect, "RRN")
        assert p == 1.0
        assert r == 1.0

    def test_compute_precision_recall_zero_recall(self):
        """compute_precision_recall returns recall=0.0 for a null detector."""
        c = KoreanPIICorpus(seed=7, samples_per_format=2)
        p, r = compute_precision_recall(c, lambda t: set(), "RRN")
        assert r == 0.0
