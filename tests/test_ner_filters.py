"""
Tests for Stage-2 NER false-positive suppression (negative proximity, Phase 1).

Verifies that code tokens / acronyms / blobs / common nouns are dropped, while
real Korean names, organizations, addresses, and TitleCase English names pass.
"""
from __future__ import annotations

import pytest

from pii_guard.stage2.ner_filters import is_ner_false_positive


# ── Should be SUPPRESSED (true over-masking) ──────────────────────────────────
@pytest.mark.parametrize("cat,text", [
    ("PERSON", "API_KEY"),            # identifier (underscore)
    ("ORGANIZATION", "DB_PASSWORD"),  # identifier
    ("ORGANIZATION", "send_email(to='x@y.com"),  # code call
    ("ORGANIZATION", "example.com"),  # domain
    ("ORGANIZATION", "AWS"),          # acronym
    ("ORGANIZATION", "LGTM"),         # acronym
    ("PERSON", "JWT"),                # acronym
    ("ORGANIZATION", "NDA"),          # acronym
    ("ORGANIZATION", "MIIBOgIBAAJBAKj34GkxFh"),  # base64 blob
    ("PERSON", "주석"),               # common noun (deny-list)
    ("PERSON", "리턴값"),             # common noun
    ("PERSON", "수익자"),             # common noun
    ("PERSON", "여권번호"),           # label noun
    ("ADDRESS", "생년월일"),          # label noun mis-tagged as address
    ("PERSON", "rotate"),             # english common word
    ("PERSON", "admin"),              # english common word
])
def test_suppressed(cat, text):
    assert is_ner_false_positive(cat, text) is True


# ── Should PASS (real PII — must not be filtered) ─────────────────────────────
@pytest.mark.parametrize("cat,text", [
    ("PERSON", "김민수"),             # Korean name
    ("PERSON", "홍길동"),
    ("PERSON", "John Smith"),         # TitleCase English name (real PII)
    ("PERSON", "Mike Brown"),
    ("ORGANIZATION", "삼성전자"),     # Korean org
    ("ORGANIZATION", "국민은행"),
    ("ORGANIZATION", "한빛무역"),
    ("ADDRESS", "서울 강남구 테헤란로 123"),  # address
    ("ADDRESS", "부산 해운대구 우동"),
])
def test_passes(cat, text):
    assert is_ner_false_positive(cat, text) is False


def test_non_ner_category_never_filtered():
    # Stage-1 categories pass through unchanged.
    assert is_ner_false_positive("AWS_SECRET", "AKIAIOSFODNN7EXAMPLE") is False
    assert is_ner_false_positive("EMAIL", "a@b.com") is False
