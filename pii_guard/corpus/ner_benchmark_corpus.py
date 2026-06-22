"""
pii_guard/corpus/ner_benchmark_corpus.py

NER Benchmark Corpus — extends KoreanPIICorpus with ORGANIZATION samples.

This corpus is specifically designed for evaluating Stage-2 NER (Presidio +
ko_core_news_sm) on the three entity types the NER engine is responsible for:
PERSON, ADDRESS, and ORGANIZATION.

The parent KoreanPIICorpus already provides PERSON and ADDRESS samples (the
regex/checksum coverage and the NER gap variants).  This subclass adds:

  * ORGANIZATION positive samples — well-known Korean company and institution
    names in five surface forms that ko_core_news_sm reliably classifies as OG.
  * Clean NER negative samples — texts that mention generic organisational
    concepts (회사, 기업, 학교) without embedding specific named entities, so
    false-positive counts for PERSON / ADDRESS / ORGANIZATION are unambiguous.

Design notes
------------
The negative samples in the parent corpus include strings like
"삼성전자 신제품이 출시되었습니다." which were labelled is_negative=True for
the PERSON detection test (no person name present) but do contain a real
organisation name.  The NER benchmark therefore uses a *separate* NER-clean
negative set (``source_tag="neg_ner_clean"``) to measure ORGANIZATION precision
without inflating false-positive counts.

Usage::

    from pii_guard.corpus.ner_benchmark_corpus import NERBenchmarkCorpus

    corpus = NERBenchmarkCorpus(seed=42, samples_per_format=5)
    org_samples = corpus.samples_for_category("ORGANIZATION")
    neg_ner = corpus.ner_clean_negatives()  # true NER negatives
"""
from __future__ import annotations

import random
from typing import List

from .korean_pii import (
    CorpusSample,
    KoreanPIICorpus,
    PIISpan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Organisation name fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Well-known Korean organisations (companies, institutions, universities, banks).
# Selected because ko_core_news_sm reliably tags them as OG in sentence context.
_ORGANIZATIONS: List[str] = [
    # Technology companies
    "삼성전자",
    "현대자동차",
    "LG전자",
    "카카오",
    "네이버",
    "SK하이닉스",
    "롯데그룹",
    "한화그룹",
    # Universities
    "서울대학교",
    "연세대학교",
    "고려대학교",
    "한국과학기술원",
    # Banks / public entities
    "한국은행",
    "대한항공",
    "한국전력공사",
]

# Sentence templates embedding an organisation name.
# {org} placeholder is replaced with the sampled organisation.
_ORG_SENTENCE_TEMPLATES: List[str] = [
    "{org} 직원이 방문했습니다.",          # "Samsung Electronics employee visited."
    "{org}에서 연락이 왔습니다.",          # "A call came from Hyundai."
    "{org} 관계자가 확인했습니다.",        # "Kakao representative confirmed."
    "{org}의 공지사항을 확인해 주세요.",   # "Please check Naver's notice."
    "{org} 측에서 자료를 요청했습니다.",   # "LG requested the materials."
]


# ─────────────────────────────────────────────────────────────────────────────
# NER-clean negative fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Texts that contain generic organisational nouns (회사, 기업, 학교) but NO
# specific named entity that NER should recognise.  Used as true negatives for
# PERSON / ADDRESS / ORGANIZATION precision measurement.
_NER_CLEAN_NEGATIVES: List[str] = [
    # Generic org terms — no proper name
    "회사에 연락하세요.",
    "기업 투자를 검토합니다.",
    "학교에서 배웠습니다.",
    "은행 업무를 처리해 주세요.",
    "직원이 안내해 드립니다.",
    "기관에 문의해 주세요.",
    "단체 할인이 가능합니다.",
    # Generic person terms — no proper name
    "담당자에게 연락 주세요.",
    "고객분께서 요청하셨습니다.",
    "신청인의 서류가 필요합니다.",
    # Generic location terms — no specific address
    "사무실 위치를 알려주세요.",
    "지점에 방문하시면 됩니다.",
    "건물 내부 안내를 확인하세요.",
    # Technical / neutral texts
    "오늘 회의는 오후 3시에 시작합니다.",
    "프로젝트 완료 예정일을 확인해 주세요.",
    "파이썬 버전 3.11을 사용하세요.",
    "문서를 첨부해 주시기 바랍니다.",
    "절차에 따라 진행해 주세요.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Corpus class
# ─────────────────────────────────────────────────────────────────────────────

class NERBenchmarkCorpus(KoreanPIICorpus):
    """
    Extended Korean PII corpus for NER precision/recall benchmarking.

    Inherits all samples from :class:`~pii_guard.corpus.KoreanPIICorpus` and
    adds ORGANIZATION positive samples plus NER-clean negative samples.

    Parameters
    ----------
    seed:
        Random seed (default 42 — canonical regression fixture).
    samples_per_format:
        Samples generated per format variant per category.  Default 5.
    """

    def _build(self) -> None:
        # Build parent corpus (PERSON, PHONE, RRN, ADDRESS, KR_ACCOUNT)
        super()._build()
        # Add NER-specific additions
        self._build_organization_samples()
        self._build_ner_clean_negatives()

    # ── ORGANIZATION ─────────────────────────────────────────────────────────

    def _build_organization_samples(self) -> None:
        """
        Generate ORGANIZATION positive samples in five surface forms.

        Format 1: "소속: <org>"              — affiliation label
        Format 2: "<org> 직원이 …"            — employee sentence
        Format 3: "<org>에서 연락이 왔습니다."  — contact-from
        Format 4: "회사: <org>"              — company label
        Format 5: "<org> 관계자가 확인했습니다."— representative sentence

        Each format generates ``samples_per_format`` samples using randomly
        chosen organisations from ``_ORGANIZATIONS``.
        """
        # Format 1: "소속: <org>"
        for _ in range(self._n):
            org = self._rng.choice(_ORGANIZATIONS)
            text = f"소속: {org}"
            start = len("소속: ")
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ORGANIZATION", start, start + len(org), org,
                               "affiliation_label")],
                source_tag="org_affiliation_label",
            ))

        # Format 2: "<org> 직원이 방문했습니다."
        for _ in range(self._n):
            org = self._rng.choice(_ORGANIZATIONS)
            text = f"{org} 직원이 방문했습니다."
            start = 0
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ORGANIZATION", start, start + len(org), org,
                               "employee_sentence")],
                source_tag="org_employee_sentence",
            ))

        # Format 3: "<org>에서 연락이 왔습니다."
        for _ in range(self._n):
            org = self._rng.choice(_ORGANIZATIONS)
            text = f"{org}에서 연락이 왔습니다."
            start = 0
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ORGANIZATION", start, start + len(org), org,
                               "contact_from")],
                source_tag="org_contact_from",
            ))

        # Format 4: "회사: <org>"
        for _ in range(self._n):
            org = self._rng.choice(_ORGANIZATIONS)
            text = f"회사: {org}"
            start = len("회사: ")
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ORGANIZATION", start, start + len(org), org,
                               "company_label")],
                source_tag="org_company_label",
            ))

        # Format 5: "<org> 관계자가 확인했습니다."
        for _ in range(self._n):
            org = self._rng.choice(_ORGANIZATIONS)
            text = f"{org} 관계자가 확인했습니다."
            start = 0
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ORGANIZATION", start, start + len(org), org,
                               "representative")],
                source_tag="org_representative",
            ))

    # ── NER-clean negatives ───────────────────────────────────────────────────

    def _build_ner_clean_negatives(self) -> None:
        """
        Add NER-clean negative samples (no named entity of any type).

        These texts are used as the ground-truth negatives in NER precision
        measurement because they contain no specific person names, location
        addresses, or organisation names that NER should fire on.
        """
        for text in _NER_CLEAN_NEGATIVES:
            self._samples.append(CorpusSample(
                text=text,
                spans=[],
                is_negative=True,
                source_tag="neg_ner_clean",
            ))

    # ── Public helpers ────────────────────────────────────────────────────────

    def ner_clean_negatives(self) -> list:
        """Return only the NER-clean negative samples added by this class."""
        return [s for s in self._samples
                if s.is_negative and s.source_tag == "neg_ner_clean"]

    def ner_owned_categories(self) -> list:
        """Return categories that Stage-2 NER is responsible for detecting."""
        return ["PERSON", "ADDRESS", "ORGANIZATION"]
