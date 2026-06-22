"""
Synthetic Korean PII red-team corpus.

Generates labelled fixture samples covering:
  - PERSON       (Korean names: 2-4 Hangul characters, surname-first)
  - PHONE        (Korean mobile 010-XXXX-XXXX, landline 02-XXXX-XXXX, etc.)
  - RRN          (주민등록번호 YYMMDD-NNNNNNN with valid checksum)
  - ADDRESS      (Korean postal addresses: province + city + street + number)
  - KR_ACCOUNT   (Korean bank account numbers in Kookmin/Shinhan/Woori formats)

Design goals
------------
* Purely synthetic — no real individuals.  All RRNs and account numbers are
  algorithmically generated to satisfy their respective checksum/format
  constraints but are NOT registered with any authority.
* Deterministic by default: the default seed produces the same samples on
  every run so test suites are reproducible.
* Ground-truth spans: every sample carries a list of PIISpan objects that
  record the exact character offsets and category of each PII item embedded
  in the sample text, enabling precision/recall measurement against any
  detector.
* Format variety: each category is represented across multiple realistic
  surface forms (with/without dashes, with label prefixes, bare inline, in
  sentence context, mixed Korean/English) to stress-test detector robustness.
* Negative samples: the corpus also includes non-PII texts that must NOT
  trigger false positives for each category.

Usage::

    from pii_guard.corpus import KoreanPIICorpus

    corpus = KoreanPIICorpus()
    samples = corpus.all_samples()          # all positive + negative samples
    positives = corpus.positive_samples()   # only samples with PII
    negatives = corpus.negative_samples()   # only clean samples

    # Iterate with metadata
    for sample in corpus.samples_for_category("RRN"):
        print(sample.text, sample.spans)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PIISpan:
    """Ground-truth annotation for one PII item embedded in a text."""
    category: str        # e.g. "RRN", "PERSON", "PHONE"
    start: int           # inclusive char offset in sample.text
    end: int             # exclusive char offset in sample.text
    value: str           # the exact substring text[start:end]
    format_tag: str = "" # human-readable format descriptor, e.g. "dash", "bare"


@dataclass
class CorpusSample:
    """
    A single labelled text sample.

    Attributes
    ----------
    text:
        The sample string (may contain 0..N PII items).
    spans:
        Ground-truth PIISpan list; empty for negative (clean) samples.
    categories:
        Set of category names present in this sample.
    is_negative:
        True when the sample is intentionally clean (no PII); the detector
        must NOT fire on it.
    source_tag:
        Short descriptor of the fixture that generated this sample.
    """
    text: str
    spans: List[PIISpan] = field(default_factory=list)
    categories: FrozenSet[str] = field(default_factory=frozenset)
    is_negative: bool = False
    source_tag: str = ""

    def __post_init__(self):
        # Derive categories from spans when not supplied explicitly
        if not self.categories and self.spans:
            object.__setattr__(
                self,
                "categories",
                frozenset(s.category for s in self.spans),
            )

    def verify_spans(self) -> bool:
        """Return True if every span's value matches text[start:end]."""
        return all(self.text[s.start:s.end] == s.value for s in self.spans)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — checksum generators
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rrn_check_digit(first12: str) -> int:
    """
    Given the first 12 digits of a Korean RRN, compute the 13th check digit.

    Algorithm: weights [2,3,4,5,6,7,8,9,2,3,4,5], sum mod 11, (11-sum%11)%10.
    """
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(d) * w for d, w in zip(first12, weights))
    return (11 - (total % 11)) % 10


def _make_rrn(birth_yymmdd: str, gender_century: int, tail5: str) -> str:
    """
    Construct a syntactically valid Korean RRN with correct checksum.

    Parameters
    ----------
    birth_yymmdd:
        6-digit birth date string, e.g. "900505".
    gender_century:
        7th digit: 1=male born 1900s, 2=female born 1900s,
                   3=male born 2000s, 4=female born 2000s.
    tail5:
        5 arbitrary digits (positions 8-12).  Position 13 is computed.
    """
    assert len(birth_yymmdd) == 6
    assert gender_century in (1, 2, 3, 4)
    assert len(tail5) == 5
    first12 = birth_yymmdd + str(gender_century) + tail5
    check = _compute_rrn_check_digit(first12)
    return first12[:6] + str(gender_century) + tail5 + str(check)


def _compute_biz_check_digit(first9: str) -> int:
    """Korean business registration number check digit."""
    weights = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    total = sum(int(first9[i]) * weights[i] for i in range(9))
    total += (int(first9[8]) * 5) // 10
    return (10 - (total % 10)) % 10


# ─────────────────────────────────────────────────────────────────────────────
# Fixture tables
# ─────────────────────────────────────────────────────────────────────────────

# Korean surnames (성) — most common 30
_SURNAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
]

# Korean given-name syllables (단음절 and 이음절)
_GIVEN_1 = [
    "민", "준", "서", "예", "지", "현", "수", "진", "영", "혜",
    "유", "재", "성", "광", "도", "철", "용", "태", "인", "승",
]
_GIVEN_2 = [
    "민준", "서연", "하은", "지수", "지원", "예린", "현우", "수빈",
    "태양", "하늘", "도윤", "채원", "지호", "민지", "은서", "아린",
    "재민", "윤서", "나연", "혜원",
]

# Korean provinces/cities for addresses
_PROVINCES = [
    ("서울특별시", "강남구", "테헤란로"),
    ("서울특별시", "서초구", "반포대로"),
    ("서울특별시", "마포구", "홍대입구로"),
    ("부산광역시", "해운대구", "해운대해변로"),
    ("부산광역시", "중구", "광복로"),
    ("대구광역시", "중구", "동성로"),
    ("인천광역시", "남동구", "인하로"),
    ("광주광역시", "서구", "상무중앙로"),
    ("대전광역시", "유성구", "대학로"),
    ("경기도", "수원시", "정조로"),
    ("경기도", "성남시", "분당로"),
    ("경기도", "고양시", "일산서구로"),
    ("강원도", "춘천시", "춘천로"),
    ("충청북도", "청주시", "직지대로"),
    ("전라남도", "여수시", "웅천로"),
]

# Korean bank account formats: (bank_name, format_fn)
def _kookmin_account(rng: random.Random) -> str:
    """국민은행 형식: XXXXXX-XX-XXXXXX"""
    return f"{rng.randint(100000,999999):06d}-{rng.randint(10,99):02d}-{rng.randint(100000,999999):06d}"

def _shinhan_account(rng: random.Random) -> str:
    """신한은행 형식: XXX-XXXXXX-XXXXX"""
    return f"{rng.randint(100,999):03d}-{rng.randint(100000,999999):06d}-{rng.randint(10000,99999):05d}"

def _woori_account(rng: random.Random) -> str:
    """우리은행 형식: XXXX-XXX-XXXXXX"""
    return f"{rng.randint(1000,9999):04d}-{rng.randint(100,999):03d}-{rng.randint(100000,999999):06d}"

def _hana_account(rng: random.Random) -> str:
    """하나은행 형식: XXX-XXXXXX-XXXXX (same as shinhan pattern)"""
    return f"{rng.randint(100,999):03d}-{rng.randint(100000,999999):06d}-{rng.randint(10000,99999):05d}"

_ACCOUNT_MAKERS = [
    ("kookmin",  _kookmin_account,  "6-2-6"),
    ("shinhan",  _shinhan_account,  "3-6-5"),
    ("woori",    _woori_account,    "4-3-6"),
    ("hana",     _hana_account,     "3-6-5"),
]

# Korean mobile area codes
_KR_MOBILE_PREFIXES = ["010", "011", "016", "017", "019"]
_KR_LANDLINE_AREA   = ["02", "031", "032", "033", "041", "042", "043",
                        "051", "052", "053", "054", "055", "061", "062",
                        "063", "064"]


# ─────────────────────────────────────────────────────────────────────────────
# Corpus generator
# ─────────────────────────────────────────────────────────────────────────────

class KoreanPIICorpus:
    """
    Synthetic Korean PII corpus for detection validation.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.  Default 42 gives the canonical
        regression fixture set used in CI.
    samples_per_format:
        How many samples to generate per format variant within each category.
        Default 5 gives ≥ 5 samples × N format variants per category.
    """

    def __init__(self, seed: int = 42, samples_per_format: int = 5):
        self._rng = random.Random(seed)
        self._n = samples_per_format
        self._samples: List[CorpusSample] = []
        self._build()

    # ── public API ────────────────────────────────────────────────────────────

    def all_samples(self) -> List[CorpusSample]:
        """Return all corpus samples (positive + negative)."""
        return list(self._samples)

    def positive_samples(self) -> List[CorpusSample]:
        """Return only samples that contain at least one PII span."""
        return [s for s in self._samples if not s.is_negative]

    def negative_samples(self) -> List[CorpusSample]:
        """Return only clean (no-PII) samples that must not trigger detectors."""
        return [s for s in self._samples if s.is_negative]

    def samples_for_category(self, category: str) -> List[CorpusSample]:
        """Return positive samples that contain spans of the given category."""
        return [s for s in self._samples
                if not s.is_negative and category in s.categories]

    def category_counts(self) -> Dict[str, int]:
        """Return {category: number_of_positive_samples} for all categories."""
        counts: Dict[str, int] = {}
        for s in self.positive_samples():
            for cat in s.categories:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def coverage_report(self) -> str:
        """Human-readable category × format coverage summary."""
        lines = ["Korean PII Corpus — Coverage Report", "=" * 40]
        counts = self.category_counts()
        total_pos = len(self.positive_samples())
        total_neg = len(self.negative_samples())
        for cat, n in sorted(counts.items()):
            lines.append(f"  {cat:<16} {n:>4} positive samples")
        lines.append("-" * 40)
        lines.append(f"  Total positive: {total_pos}")
        lines.append(f"  Total negative: {total_neg}")
        lines.append(f"  Grand total:    {total_pos + total_neg}")
        return "\n".join(lines)

    # ── builders ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self._build_person_samples()
        self._build_phone_samples()
        self._build_rrn_samples()
        self._build_address_samples()
        self._build_kr_account_samples()
        self._build_negative_samples()

    # ── PERSON ────────────────────────────────────────────────────────────────

    def _random_name(self) -> str:
        """Generate a random Korean full name (성+이름)."""
        surname = self._rng.choice(_SURNAMES)
        if self._rng.random() < 0.5:
            given = self._rng.choice(_GIVEN_1)
        else:
            given = self._rng.choice(_GIVEN_2)
        return surname + given

    def _make_person_sample(self, label: str, name: str, context: str) -> CorpusSample:
        """
        Build a CorpusSample embedding ``name`` after ``label:`` in ``context``.
        ``context`` must contain exactly one ``{name}`` placeholder.
        """
        text = context.format(name=name)
        start = text.index(name)
        end = start + len(name)
        span = PIISpan(category="PERSON", start=start, end=end, value=name,
                       format_tag=label)
        return CorpusSample(text=text, spans=[span], source_tag=f"person_{label}")

    def _build_person_samples(self) -> None:
        # Format 1: "성명: <name>"
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "성명_label", name, f"성명: {name}")
            self._samples.append(s)

        # Format 2: "이름: <name>"
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "이름_label", name, f"이름: {name}")
            self._samples.append(s)

        # Format 3: "담당자: <name>"
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "담당자_label", name, f"담당자: {name}")
            self._samples.append(s)

        # Format 4: "신청인: <name>" (applicant)
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "신청인_label", name, f"신청인: {name}")
            self._samples.append(s)

        # Format 5: "작성자: <name>" (author)
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "작성자_label", name, f"작성자: {name}")
            self._samples.append(s)

        # Format 6: English label "Name: <name>"
        for _ in range(self._n):
            name = self._random_name()
            s = self._make_person_sample(
                "name_en_label", name, f"Name: {name}")
            self._samples.append(s)

        # Format 7: sentence context
        contexts = [
            "고객 {name}님께서 요청하셨습니다.",
            "담당자는 {name}입니다.",
            "{name} 고객의 계좌를 확인해 주세요.",
        ]
        for ctx in contexts:
            for _ in range(self._n):
                name = self._random_name()
                text = ctx.format(name=name)
                start = text.index(name)
                end = start + len(name)
                span = PIISpan(category="PERSON", start=start, end=end,
                               value=name, format_tag="sentence_kr")
                self._samples.append(CorpusSample(
                    text=text, spans=[span], source_tag="person_sentence"))

    # ── PHONE ─────────────────────────────────────────────────────────────────

    def _random_mobile(self, dashes: bool = True) -> str:
        prefix = self._rng.choice(_KR_MOBILE_PREFIXES)
        mid = f"{self._rng.randint(1000, 9999):04d}"
        last = f"{self._rng.randint(1000, 9999):04d}"
        if dashes:
            return f"{prefix}-{mid}-{last}"
        else:
            return f"{prefix}{mid}{last}"

    def _random_landline(self) -> str:
        area = self._rng.choice(_KR_LANDLINE_AREA)
        # Seoul (02) has 8-digit subscriber, others 7 or 8
        if area == "02":
            sub1 = f"{self._rng.randint(100, 9999)}"
            sub2 = f"{self._rng.randint(1000, 9999):04d}"
        else:
            sub1 = f"{self._rng.randint(100, 999):03d}"
            sub2 = f"{self._rng.randint(1000, 9999):04d}"
        return f"{area}-{sub1}-{sub2}"

    def _build_phone_samples(self) -> None:
        # Format 1: mobile with dashes
        for _ in range(self._n):
            phone = self._random_mobile(dashes=True)
            text = f"연락처: {phone}"
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "mobile_dash")],
                source_tag="phone_mobile_dash",
            ))

        # Format 2: mobile without dashes
        for _ in range(self._n):
            phone = self._random_mobile(dashes=False)
            text = f"전화번호: {phone}"
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "mobile_bare")],
                source_tag="phone_mobile_bare",
            ))

        # Format 3: landline
        for _ in range(self._n):
            phone = self._random_landline()
            text = f"대표번호: {phone}"
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "landline")],
                source_tag="phone_landline",
            ))

        # Format 4: international format +82
        for _ in range(self._n):
            mid = f"{self._rng.randint(1000, 9999):04d}"
            last = f"{self._rng.randint(1000, 9999):04d}"
            phone = f"+82-10-{mid}-{last}"
            text = f"해외연락처: {phone}"
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "international")],
                source_tag="phone_international",
            ))

        # Format 5: mobile with dots
        for _ in range(self._n):
            prefix = "010"
            mid = f"{self._rng.randint(1000, 9999):04d}"
            last = f"{self._rng.randint(1000, 9999):04d}"
            phone = f"{prefix}.{mid}.{last}"
            text = f"Phone: {phone}"
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "mobile_dot")],
                source_tag="phone_mobile_dot",
            ))

        # Format 6: inline in sentence
        for _ in range(self._n):
            phone = self._random_mobile(dashes=True)
            text = f"{phone}으로 문자 주세요."
            start = text.index(phone)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("PHONE", start, start + len(phone), phone, "inline_sentence")],
                source_tag="phone_inline_sentence",
            ))

    # ── RRN ───────────────────────────────────────────────────────────────────

    def _random_rrn(self, gender_century: Optional[int] = None) -> str:
        """Generate a syntactically valid RRN with correct checksum."""
        # Random birth year 1960-2005
        year = self._rng.randint(60, 99)       # 1960-1999
        month = self._rng.randint(1, 12)
        day = self._rng.randint(1, 28)
        birth = f"{year:02d}{month:02d}{day:02d}"
        if gender_century is None:
            gender_century = self._rng.choice([1, 2])  # 1900s citizens
        tail5 = "".join(str(self._rng.randint(0, 9)) for _ in range(5))
        return _make_rrn(birth, gender_century, tail5)

    def _build_rrn_samples(self) -> None:
        # Format 1: with dash "성별: YYMMDD-NNNNNNN"
        for _ in range(self._n):
            rrn = self._random_rrn()
            rrn_display = rrn[:6] + "-" + rrn[6:]  # insert dash
            text = f"주민등록번호: {rrn_display}"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "dash")],
                source_tag="rrn_dash",
            ))

        # Format 2: without dash
        for _ in range(self._n):
            rrn = self._random_rrn()
            text = f"주민번호: {rrn}"
            start = text.index(rrn)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn), rrn, "bare")],
                source_tag="rrn_bare",
            ))

        # Format 3: male born 1900s (gender digit 1)
        for _ in range(self._n):
            rrn = self._random_rrn(gender_century=1)
            rrn_display = rrn[:6] + "-" + rrn[6:]
            text = f"주민등록번호: {rrn_display}"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "gender1_dash")],
                source_tag="rrn_gender1",
            ))

        # Format 4: female born 1900s (gender digit 2)
        for _ in range(self._n):
            rrn = self._random_rrn(gender_century=2)
            rrn_display = rrn[:6] + "-" + rrn[6:]
            text = f"주민등록번호: {rrn_display}"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "gender2_dash")],
                source_tag="rrn_gender2",
            ))

        # Format 5: male born 2000s (gender digit 3)
        for _ in range(self._n):
            year = self._rng.randint(0, 9)   # 2000-2009
            month = self._rng.randint(1, 12)
            day = self._rng.randint(1, 28)
            birth = f"0{year:01d}{month:02d}{day:02d}"
            tail5 = "".join(str(self._rng.randint(0, 9)) for _ in range(5))
            rrn = _make_rrn(birth, 3, tail5)
            rrn_display = rrn[:6] + "-" + rrn[6:]
            text = f"주민번호: {rrn_display}"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "gender3_2000s")],
                source_tag="rrn_gender3",
            ))

        # Format 6: inline in form context
        for _ in range(self._n):
            rrn = self._random_rrn()
            rrn_display = rrn[:6] + "-" + rrn[6:]
            text = f"신청자 주민번호: {rrn_display} / 연락처 기재"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "inline_form")],
                source_tag="rrn_inline_form",
            ))

        # Format 7: space-separated variant (YYMMDD NNNNNNN)
        for _ in range(self._n):
            rrn = self._random_rrn()
            rrn_display = rrn[:6] + " " + rrn[6:]
            text = f"주민번호 {rrn_display}"
            start = text.index(rrn_display)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("RRN", start, start + len(rrn_display), rrn_display, "space_sep")],
                source_tag="rrn_space",
            ))

    # ── ADDRESS ───────────────────────────────────────────────────────────────

    def _random_address(self) -> Tuple[str, str]:
        """Return (full_address_str, format_tag)."""
        prov, city, street = self._rng.choice(_PROVINCES)
        num = self._rng.randint(1, 999)
        detail = self._rng.choice(["", f" {self._rng.randint(100, 999)}호",
                                   f" {self._rng.randint(1,30)}층"])
        addr = f"{prov} {city} {street} {num}{detail}".strip()
        return addr, f"{prov[:2]}"

    def _build_address_samples(self) -> None:
        # Format 1: bare address
        for _ in range(self._n):
            addr, tag = self._random_address()
            text = addr
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ADDRESS", 0, len(addr), addr, f"bare_{tag}")],
                source_tag="address_bare",
            ))

        # Format 2: labelled "주소: <addr>"
        for _ in range(self._n):
            addr, tag = self._random_address()
            text = f"주소: {addr}"
            start = text.index(addr)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ADDRESS", start, start + len(addr), addr, f"label_{tag}")],
                source_tag="address_labeled",
            ))

        # Format 3: "거주지: <addr>"
        for _ in range(self._n):
            addr, tag = self._random_address()
            text = f"거주지: {addr}"
            start = text.index(addr)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ADDRESS", start, start + len(addr), addr, f"residence_{tag}")],
                source_tag="address_residence",
            ))

        # Format 4: "Address: <addr>" English label
        for _ in range(self._n):
            addr, tag = self._random_address()
            text = f"Address: {addr}"
            start = text.index(addr)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ADDRESS", start, start + len(addr), addr, f"en_label_{tag}")],
                source_tag="address_en_label",
            ))

        # Format 5: mixed context
        for _ in range(self._n):
            addr, tag = self._random_address()
            text = f"배송지는 {addr}입니다."
            start = text.index(addr)
            self._samples.append(CorpusSample(
                text=text,
                spans=[PIISpan("ADDRESS", start, start + len(addr), addr, f"delivery_{tag}")],
                source_tag="address_delivery",
            ))

    # ── KR_ACCOUNT ────────────────────────────────────────────────────────────

    def _build_kr_account_samples(self) -> None:
        # One set per bank format
        for bank_name, maker_fn, fmt_tag in _ACCOUNT_MAKERS:
            # Format A: labelled "계좌번호: <acct>"
            for _ in range(self._n):
                acct = maker_fn(self._rng)
                text = f"계좌번호: {acct}"
                start = text.index(acct)
                self._samples.append(CorpusSample(
                    text=text,
                    spans=[PIISpan("KR_ACCOUNT", start, start + len(acct), acct,
                                   f"{bank_name}_kr_label")],
                    source_tag=f"kr_acct_{bank_name}_kr_label",
                ))

            # Format B: English label "Account No: <acct>"
            for _ in range(self._n):
                acct = maker_fn(self._rng)
                text = f"Account No: {acct}"
                start = text.index(acct)
                self._samples.append(CorpusSample(
                    text=text,
                    spans=[PIISpan("KR_ACCOUNT", start, start + len(acct), acct,
                                   f"{bank_name}_en_label")],
                    source_tag=f"kr_acct_{bank_name}_en_label",
                ))

            # Format C: bare inline "계좌 <acct>로 입금"
            for _ in range(self._n):
                acct = maker_fn(self._rng)
                text = f"계좌 {acct}로 입금해 주세요."
                start = text.index(acct)
                self._samples.append(CorpusSample(
                    text=text,
                    spans=[PIISpan("KR_ACCOUNT", start, start + len(acct), acct,
                                   f"{bank_name}_inline")],
                    source_tag=f"kr_acct_{bank_name}_inline",
                ))

    # ── NEGATIVE SAMPLES ─────────────────────────────────────────────────────

    def _build_negative_samples(self) -> None:
        """
        Clean texts that must NOT trigger any of the 5 Korean PII categories.
        Organised by the category they superficially resemble.
        """
        # --- PERSON negatives: place names, common nouns ----------------------
        person_negatives = [
            "서울역에서 만나요.",                       # city name, not person
            "경복궁 근처에 있습니다.",                   # landmark
            "국민은행 앱을 설치해 주세요.",               # brand name
            "삼성전자 신제품이 출시되었습니다.",           # company
            "고객센터로 연락해 주세요.",                  # generic noun
        ]
        for t in person_negatives:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_person"))

        # --- PHONE negatives: short numbers, zip codes, product codes ---------
        phone_negatives = [
            "주문번호: 12345",                          # 5-digit order ID
            "버전: 2.10.1234",                          # version string
            "제품코드: 0201234",                         # product code prefix 02
            "방 번호 123호",                             # room number
            "1588-1234 고객센터",                        # 4+4 service numbers (no 010/0XX leading)
            "ZIP code: 12345",                          # US zip
        ]
        for t in phone_negatives:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_phone"))

        # --- RRN negatives: 13-digit numbers that fail checksum or format -----
        rrn_negatives = [
            "발주번호: 9005051234567",                   # 13 digits but gender digit = 5 (FOREIGN_REG range)
            "주문: 1234567890123",                       # 13 digits, no valid date prefix
            "결제번호 900505-9234567",                    # gender digit 9 — invalid
            "코드: 000000-0000000",                      # all zeros — checksum fails
            "일련번호: 900101-0123450",                   # gender digit 0 — invalid
        ]
        for t in rrn_negatives:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_rrn"))

        # --- ADDRESS negatives: partial/ambiguous location strings ------------
        address_negatives = [
            "부산행 기차를 탔습니다.",                    # city mention without street
            "서울 날씨가 좋네요.",                        # no street address
            "경기 결과를 확인하세요.",                    # 경기 as sports game, not province
            "강남역 2번 출구로 오세요.",                  # landmark, not postal address
            "마포구 소식지",                              # district reference, no street
        ]
        for t in address_negatives:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_address"))

        # --- KR_ACCOUNT negatives: product IDs, invoice numbers ---------------
        acct_negatives = [
            "주문번호: 2024-03-12345",                   # date-like, not account
            "모델명: KR-1234-56789",                     # short model code
            "인보이스: 001-00001-00001",                  # too short
            "일련번호: 12-34-567890",                    # wrong segment count
            "코드번호: 1000-100-100000",                  # woori-like but context is "코드번호"
        ]
        for t in acct_negatives:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_kr_account"))

        # --- Mixed clean texts with no PII ------------------------------------
        clean_texts = [
            "오늘 회의는 오후 3시에 시작합니다.",
            "프로젝트 완료 예정일은 2025년 12월입니다.",
            "신제품 가격은 99,000원입니다.",
            "파이썬 버전 3.11을 사용하세요.",
            "API 문서를 참조하세요: https://docs.example.com",
            "회사 이메일 형식은 이름@회사.com입니다.",  # generic description, no real email
            "10자리 회원번호를 입력하세요.",
            "계좌이체 수수료는 500원입니다.",
            "비밀번호는 8자 이상이어야 합니다.",
            "주민등록번호 형식: YYMMDD-NNNNNNN",        # pattern description, not actual PII
        ]
        for t in clean_texts:
            self._samples.append(CorpusSample(
                text=t, spans=[], is_negative=True,
                source_tag="neg_clean"))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers for test assertions
# ─────────────────────────────────────────────────────────────────────────────

def compute_precision_recall(
    corpus: KoreanPIICorpus,
    detector_fn,  # callable(text: str) -> set[str] of detected category names
    category: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Compute macro precision and recall of ``detector_fn`` over the corpus.

    Parameters
    ----------
    corpus:
        The KoreanPIICorpus instance.
    detector_fn:
        A callable that takes a text string and returns a set of category
        names detected in it (e.g. {"RRN", "PHONE"}).
    category:
        If given, restrict evaluation to samples of this category.

    Returns
    -------
    (precision, recall) as floats in [0, 1].
    """
    tp = fp = fn = 0

    samples = corpus.all_samples()
    if category:
        samples = [s for s in samples
                   if (category in s.categories) or s.is_negative]

    for sample in samples:
        detected = detector_fn(sample.text)
        true_cats = sample.categories if not sample.is_negative else frozenset()

        if category:
            det_hit = category in detected
            true_hit = category in true_cats
            if det_hit and true_hit:
                tp += 1
            elif det_hit and not true_hit:
                fp += 1
            elif not det_hit and true_hit:
                fn += 1
        else:
            for cat in true_cats:
                if cat in detected:
                    tp += 1
                else:
                    fn += 1
            for cat in detected:
                if cat not in true_cats:
                    fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall
