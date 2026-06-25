"""
Category definitions: patterns, actions, confidence thresholds, and mask styles
for every PII and secret category supported by PII-Guard Stage 1.

Block categories  (action=BLOCK)  — high-risk, never leave the host:
  API_KEY, AWS_SECRET, GCP_KEY, TOKEN, PRIVATE_KEY, PASSWORD,
  RRN, FOREIGN_REG, PASSPORT, DRIVER_LICENSE, CARD

Mask categories (action=TOKENIZE_ROUNDTRIP) — contact/context PII,
placeholders go out, originals are rehydrated on inbound responses:
  EMAIL, PHONE, KR_ACCOUNT, BIZ_NO, PERSON, ADDRESS

Each entry is a list of (rule_id, compiled_pattern, confidence, post_validator_fn).
post_validator_fn may be None or a callable(match_str)->bool for checksum/Luhn etc.
"""
from __future__ import annotations

import re
from typing import Callable, List, NamedTuple, Optional

from .models import Action, CategoryClass, DetectionStage, MaskStyle


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / validators
# ──────────────────────────────────────────────────────────────────────────────

def _luhn_valid(number: str) -> bool:
    """Return True when the digit string passes the Luhn algorithm."""
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _rrn_checksum(raw: str) -> bool:
    """Korean RRN 13-digit Luhn-style checksum."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) != 13:
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(d) * w for d, w in zip(digits, weights))
    check = (11 - (total % 11)) % 10
    return check == int(digits[12])


def _kr_biz_checksum(raw: str) -> bool:
    """Korean business registration number checksum."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) != 10:
        return False
    weights = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    total += int(digits[8]) * 5 // 10
    check = (10 - (total % 10)) % 10
    return check == int(digits[9])


# ──────────────────────────────────────────────────────────────────────────────
# Rule descriptor
# ──────────────────────────────────────────────────────────────────────────────

class PatternRule(NamedTuple):
    rule_id: str
    pattern: re.Pattern
    confidence: float
    validator: Optional[Callable[[str], bool]] = None  # extra checksum/Luhn


# ──────────────────────────────────────────────────────────────────────────────
# Category spec
# ──────────────────────────────────────────────────────────────────────────────

class CategorySpec(NamedTuple):
    category: str
    category_class: CategoryClass
    action: Action
    mask_style: MaskStyle
    min_confidence: float
    rules: List[PatternRule]
    detection_stage: DetectionStage = DetectionStage.STAGE1_REGEX_CHECKSUM


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

# ── EMAIL ─────────────────────────────────────────────────────────────────────
_EMAIL_PATTERN = re.compile(
    r"""(?<![/@\w])              # not preceded by @, /, or word char
    [a-zA-Z0-9._%+\-]+          # local part
    @
    [a-zA-Z0-9.\-]+             # domain
    \.
    [a-zA-Z]{2,}                # TLD
    (?!\.[a-zA-Z])              # not followed by another dot+alpha (avoid .tar.gz etc)
    """,
    re.VERBOSE,
)

EMAIL = CategorySpec(
    category="EMAIL",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.90,
    rules=[
        PatternRule("email_rfc5322", _EMAIL_PATTERN, 0.95),
    ],
)

# ── PHONE ─────────────────────────────────────────────────────────────────────
_PHONE_RULES = [
    # Korean mobile 010-XXXX-XXXX or 010XXXXXXXX
    PatternRule(
        "phone_kr_mobile",
        re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),
        0.95,
    ),
    # Korean landline 02-XXXX-XXXX, 0XX-XXXX-XXXX
    PatternRule(
        "phone_kr_landline",
        re.compile(r"(?<!\d)0[2-9]\d{0,1}[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),
        0.88,
    ),
    # International +XX XX XXXX XXXX
    PatternRule(
        "phone_intl",
        re.compile(r"\+\d{1,3}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{4}(?!\d)"),
        0.85,
    ),
    # US/CA (NXX) NXX-XXXX
    PatternRule(
        "phone_us",
        re.compile(r"(?<!\d)\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
        0.80,
    ),
]

PHONE = CategorySpec(
    category="PHONE",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=_PHONE_RULES,
)

# ── PERSON ─────────────────────────────────────────────────────────────────────
# Stage-1 heuristics only — NER handles this better in Stage 2
_PERSON_RULES = [
    # Explicit label patterns: "Name: John Smith", "담당자: 김철수"
    PatternRule(
        "person_labeled_en",
        re.compile(
            r"(?:(?:full\s+)?name|patient|author|signed?\s*by|prepared\s*by|contact)"
            r"\s*[:：]\s*"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            re.IGNORECASE,
        ),
        0.85,
    ),
    PatternRule(
        "person_labeled_kr",
        re.compile(
            r"(?:성명|이름|담당자|작성자|신청인|피해자|원고|피고)\s*[:：]\s*"
            r"([가-힣]{2,5})",
        ),
        0.90,
    ),
]

PERSON = CategorySpec(
    category="PERSON",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=_PERSON_RULES,
)

# ── ADDRESS ────────────────────────────────────────────────────────────────────
_ADDRESS_RULES = [
    # Korean address: starts with province/city (서울, 경기, 부산 etc.)
    PatternRule(
        "address_kr",
        re.compile(
            r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남"
            r"|전북|전남|경북|경남|제주)(?:특별시|광역시|특별자치시|도)?"
            r"\s*[가-힣\d\s\-\.]+(?:구|군|시)\s*[가-힣\d\s\-\.]+(?:동|읍|면|로|길)"
            r"\s*[\d\-]+",
        ),
        0.85,
    ),
    # US street address: 123 Main St, City, ST 12345
    PatternRule(
        "address_us",
        re.compile(
            r"\b\d{1,5}\s+[A-Za-z0-9\s]{3,40}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)"
            r"\.?(?:\s*,\s*[A-Za-z\s]+)?(?:\s*,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)?",
            re.IGNORECASE,
        ),
        0.75,
    ),
]

ADDRESS = CategorySpec(
    category="ADDRESS",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.75,
    rules=_ADDRESS_RULES,
)

# ── ORGANIZATION — Korean/general organization names ──────────────────────────
# This is a Stage-2-only category: Stage-1 regex cannot reliably detect
# organization names in unstructured text (too many false positives).
# Stage-2 NER (KoreanNEREngine / ko_core_news_sm via Presidio) detects OG labels.
# No Stage-1 rules are defined — the empty rules list means Stage-1 scanning
# skips this category entirely.  Policy and CATEGORY_MAP lookups still work.
ORGANIZATION = CategorySpec(
    category="ORGANIZATION",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.70,
    rules=[],  # Stage-2 NER only; no regex rules
    detection_stage=DetectionStage.STAGE2_NER,
)

# ── IP_ADDRESS — server topology (IPv4) ───────────────────────────────────────
# Strict octet validation (each 0-255); lookarounds prevent matching inside a
# longer dotted run. Masks internal AND public IPv4 so server/client network
# identifiers don't leak to an external LLM during log analysis.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_PATTERN = re.compile(
    rf"(?<![\d.])(?:{_OCTET}\.){{3}}{_OCTET}(?![\d.])"
)
IP_ADDRESS = CategorySpec(
    category="IP_ADDRESS",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=[PatternRule("ipv4", _IPV4_PATTERN, 0.85)],
)

# ── HOSTNAME — internal server hostnames (FQDN with internal TLD) ──────────────
# Conservative: only FQDNs ending in an internal-network TLD (internal/local/
# corp/lan/intranet/cluster.local) — public domains (gmail.com, api.anthropic.com)
# are deliberately NOT matched so legitimate external references stay intact.
_HOSTNAME_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+"
    r"(?:internal|local|corp|lan|intranet)\b",
    re.IGNORECASE,
)
HOSTNAME = CategorySpec(
    category="HOSTNAME",
    category_class=CategoryClass.PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=[PatternRule("hostname_internal", _HOSTNAME_PATTERN, 0.85)],
)

# ── RRN — Korean Resident Registration Number ─────────────────────────────────
# Format: YYMMDD-NNNNNNN   (7th digit 1-4 = Korean national, 5-8 = foreigner)
# RRN uses digits 1-4 as 7th (block both but classify separately)
_RRN_PATTERN = re.compile(
    r"""(?<!\d)
    [0-9]{6}              # birth date YYMMDD
    [-\s]?
    [1-4]                 # gender/century digit for Korean nationals (1-4)
    [0-9]{6}              # remaining 6 digits
    (?!\d)
    """,
    re.VERBOSE,
)

RRN = CategorySpec(
    category="RRN",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.85,
    rules=[
        PatternRule("rrn_kr", _RRN_PATTERN, 0.90, _rrn_checksum),
    ],
)

# ── FOREIGN_REG — Korean Foreign Registration Number ─────────────────────────
# Format: YYMMDD-NNNNNNN  (7th digit 5-8)
_FOREIGN_REG_PATTERN = re.compile(
    r"""(?<!\d)
    [0-9]{6}[-\s]?[5-8][0-9]{6}
    (?!\d)
    """,
    re.VERBOSE,
)

FOREIGN_REG = CategorySpec(
    category="FOREIGN_REG",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.85,
    rules=[
        PatternRule("foreign_reg_kr", _FOREIGN_REG_PATTERN, 0.88),
    ],
)

# ── BIZ_NO — Korean Business Registration Number ─────────────────────────────
# Format: XXX-XX-XXXXX  (10 digits)
_BIZ_NO_PATTERN = re.compile(
    r"(?<!\d)\d{3}[-]\d{2}[-]\d{5}(?!\d)",
)

BIZ_NO = CategorySpec(
    category="BIZ_NO",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.85,
    rules=[
        PatternRule("biz_no_kr", _BIZ_NO_PATTERN, 0.90, _kr_biz_checksum),
    ],
)

# ── KR_ACCOUNT — Korean Bank Account ─────────────────────────────────────────
# Various formats: 10-14 digit numbers, often hyphenated
_KR_ACCOUNT_RULES = [
    # Kookmin: 000000-00-000000 (6-2-6)
    PatternRule(
        "kr_acct_kookmin",
        re.compile(r"(?<!\d)\d{6}-\d{2}-\d{6}(?!\d)"),
        0.88,
    ),
    # Shinhan / Hana: XXX-XXXXXX-XXXXX (3-6-5) or similar
    PatternRule(
        "kr_acct_3_6_5",
        re.compile(r"(?<!\d)\d{3}-\d{6}-\d{5}(?!\d)"),
        0.85,
    ),
    # Woori: XXXX-XXX-XXXXXX (4-3-6)
    PatternRule(
        "kr_acct_4_3_6",
        re.compile(r"(?<!\d)\d{4}-\d{3}-\d{6}(?!\d)"),
        0.85,
    ),
    # Generic 10-14 digit bare account numbers when labelled
    PatternRule(
        "kr_acct_labeled",
        re.compile(
            r"(?:계좌\s*번호|account\s*(?:no\.?|number))\s*[:：]?\s*(\d[\d\-\s]{8,15}\d)",
            re.IGNORECASE,
        ),
        0.80,
    ),
]

KR_ACCOUNT = CategorySpec(
    category="KR_ACCOUNT",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.TOKENIZE_ROUNDTRIP,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=_KR_ACCOUNT_RULES,
)

# ── PASSPORT ──────────────────────────────────────────────────────────────────
_PASSPORT_RULES = [
    # Korean e-passport: M/S/F + 8 digits (new) or letter+digit combos
    # 경계로 (?![A-Za-z0-9])를 쓴다 — (?!\w)는 한국어 조사("M12345678를")를 \w로 보고 매치를 깨므로,
    # 알파벳/숫자로만 확장 차단하고 한글 조사·구두점은 허용(조사 인접 미검출 버그 수정).
    PatternRule(
        "passport_kr",
        re.compile(r"(?<![A-Za-z0-9])[MSF][A-Z]?\d{7,8}(?![A-Za-z0-9])"),
        0.85,
    ),
    # US passport: letter + 8 digits (9 total)
    PatternRule(
        "passport_us",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{8}(?![A-Za-z0-9])"),
        0.80,
    ),
    # Generic ICAO: 1-2 letters + 6-9 digits
    PatternRule(
        "passport_generic",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d{6,9}(?![A-Za-z0-9])"),
        0.75,
    ),
    # Labelled: "passport: M12345678"
    PatternRule(
        "passport_labeled",
        re.compile(
            r"(?:passport|여권)\s*(?:no\.?|number|번호)?\s*[:：]\s*([A-Z0-9]{7,10})",
            re.IGNORECASE,
        ),
        0.90,
    ),
]

PASSPORT = CategorySpec(
    category="PASSPORT",
    category_class=CategoryClass.KOREAN_PII,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.75,
    rules=_PASSPORT_RULES,
)

# ── DRIVER_LICENSE ────────────────────────────────────────────────────────────
_DL_RULES = [
    # Korean DL: 2(year) + 2(region) + 6(serial) + 2(check) = 12 digits
    PatternRule(
        "dl_kr",
        re.compile(r"(?<!\d)\d{2}[-\s]?\d{2}[-\s]?\d{6}[-\s]?\d{2}(?!\d)"),
        0.85,
    ),
    # Labeled: "운전면허 번호: ..." or "driver license: ..."
    PatternRule(
        "dl_labeled",
        re.compile(
            r"(?:driver['\s]?s?\s+licen[sc]e|운전\s*면허)\s*(?:no\.?|number|번호)?\s*[:：]\s*"
            r"([A-Z0-9][\d\-\s]{8,16}[A-Z0-9])",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # US format: varies by state, common pattern XX-XXX-XXXX
    PatternRule(
        "dl_us",
        re.compile(r"(?<!\w)[A-Z]\d{3}[-\s]\d{3}[-\s]\d{4}(?!\w)"),
        0.75,
    ),
]

DRIVER_LICENSE = CategorySpec(
    category="DRIVER_LICENSE",
    category_class=CategoryClass.PII,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.75,
    rules=_DL_RULES,
)

# ── CARD — Credit/Debit/Prepaid ───────────────────────────────────────────────
_CARD_PATTERN = re.compile(
    r"""(?<!\d)
    (?:
        4\d{3}                          # Visa
      | 5[1-5]\d{2}                     # Mastercard
      | 2(?:2[2-9]\d|[3-6]\d{2}|7[01]\d|720)  # Mastercard 2xxx
      | 3[47]\d{2}                      # Amex
      | 3(?:0[0-5]|[68]\d)\d           # Diners
      | 6(?:011|5\d{2})                 # Discover
      | (?:2131|1800|35\d{3})           # JCB
    )
    (?:[-\s]?\d{4}){2}
    [-\s]?\d{3,4}
    (?!\d)
    """,
    re.VERBOSE,
)

CARD = CategorySpec(
    category="CARD",
    category_class=CategoryClass.PII,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.90,
    rules=[
        PatternRule("card_pan", _CARD_PATTERN, 0.90, _luhn_valid),
    ],
)

# ── API_KEY ───────────────────────────────────────────────────────────────────
_API_KEY_RULES = [
    # Anthropic Claude
    PatternRule(
        "apikey_anthropic",
        re.compile(r"sk-ant-(?:api\d{2}-)?[a-zA-Z0-9\-_]{40,}"),
        0.98,
    ),
    # OpenAI sk- keys (classic and project)
    PatternRule(
        "apikey_openai",
        re.compile(r"sk-(?:proj-)?[a-zA-Z0-9\-_]{20,}"),
        0.95,
    ),
    # GitHub PAT (classic and fine-grained). 본체 길이를 {36,}→{20,}로 완화 —
    # ghp_ 등 접두가 매우 특이해 오탐 위험이 낮고, 길이가 짧은 변형/만료 토큰도 잡는다.
    PatternRule(
        "apikey_github",
        re.compile(r"(?:ghp|gho|ghu|ghs|ghr|github_pat)_[a-zA-Z0-9]{20,}"),
        0.97,
    ),
    # Stripe live/test keys
    PatternRule(
        "apikey_stripe",
        re.compile(r"(?:sk|pk|rk)_(?:live|test)_[a-zA-Z0-9]{24,}"),
        0.97,
    ),
    # Generic "api_key = " assignment patterns
    PatternRule(
        "apikey_assignment",
        re.compile(
            r"""(?:api[_\-]?key|apikey|api[_\-]?secret|app[_\-]?secret)\s*
            (?:=|:)\s*
            ['"]?([a-zA-Z0-9\-_\.]{20,})['"]?""",
            re.VERBOSE | re.IGNORECASE,
        ),
        0.85,
    ),
    # Hugging Face tokens
    PatternRule(
        "apikey_hf",
        re.compile(r"hf_[a-zA-Z0-9]{36,}"),
        0.97,
    ),
    # Generic high-entropy hex API keys (≥32 hex chars) with context
    PatternRule(
        "apikey_hex32",
        re.compile(
            r"""(?:key|token|secret|api)[_\-]?\s*[:=]\s*[\'\"]{0,1}([0-9a-fA-F]{32,64})[\'\"]{0,1}""",
            re.IGNORECASE,
        ),
        0.80,
    ),
]

API_KEY = CategorySpec(
    category="API_KEY",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=_API_KEY_RULES,
)

# ── AWS_SECRET ────────────────────────────────────────────────────────────────
_AWS_RULES = [
    # AWS Access Key ID — non-capturing group so the full 20-char key is returned
    PatternRule(
        "aws_akid",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        0.98,
    ),
    # AWS Secret Access Key (40-char base64url after context keyword)
    PatternRule(
        "aws_secret_key",
        re.compile(
            r"(?:aws[_\-]?secret[_\-]?access[_\-]?key|AWS_SECRET_ACCESS_KEY)\s*"
            r"(?:=|:)\s*['\"\\]?([A-Za-z0-9/+=]{40})['\"\\]?",
            re.IGNORECASE,
        ),
        0.97,
    ),
    # AWS session token (very long base64)
    PatternRule(
        "aws_session_token",
        re.compile(
            r"(?:aws[_\-]?session[_\-]?token|AWS_SESSION_TOKEN)\s*(?:=|:)\s*"
            r"['\"\\]?([A-Za-z0-9/+=]{100,})['\"\\]?",
            re.IGNORECASE,
        ),
        0.97,
    ),
]

AWS_SECRET = CategorySpec(
    category="AWS_SECRET",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.90,
    rules=_AWS_RULES,
)

# ── GCP_KEY ───────────────────────────────────────────────────────────────────
_GCP_RULES = [
    # GCP API Key
    PatternRule(
        "gcp_api_key",
        re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
        0.97,
    ),
    # GCP OAuth2 client secrets
    PatternRule(
        "gcp_oauth_client",
        re.compile(r"(?:GOCSPX|GOCSB)-[a-zA-Z0-9\-_]{28,}"),
        0.96,
    ),
    # Service account JSON hint: "private_key_id" in JSON context
    PatternRule(
        "gcp_sa_key_id",
        re.compile(
            r'"private_key_id"\s*:\s*"([a-f0-9]{40})"',
            re.IGNORECASE,
        ),
        0.97,
    ),
]

GCP_KEY = CategorySpec(
    category="GCP_KEY",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.90,
    rules=_GCP_RULES,
)

# ── TOKEN ─────────────────────────────────────────────────────────────────────
_TOKEN_RULES = [
    # JWT: header(eyJ…) + 2 base64url segments. 2번째 세그먼트의 eyJ 강제를 제거 —
    # 실제 JWT 페이로드는 eyJ로 시작하지만, 변형/난독 토큰도 'eyJ헤더.X.X' 형태면 잡도록 완화.
    # 헤더는 eyJ+8자 이상, 나머지 두 세그먼트는 3자 이상으로 최소 길이를 둬 오탐을 억제.
    PatternRule(
        "token_jwt",
        re.compile(
            r"eyJ[a-zA-Z0-9\-_]{8,}\.[a-zA-Z0-9\-_]{3,}\.[a-zA-Z0-9\-_]{3,}"
        ),
        0.97,
    ),
    # Bearer token in Authorization header value
    PatternRule(
        "token_bearer",
        re.compile(
            r'(?i)(?:authorization|auth)\s*[:=]\s*["\']?Bearer\s+([a-zA-Z0-9\-_.~+/]{20,})',
        ),
        0.92,
    ),
    # Generic token assignment with high entropy
    PatternRule(
        "token_assignment",
        re.compile(
            r"""(?:access[_\-]?token|refresh[_\-]?token|auth[_\-]?token|bearer[_\-]?token|id[_\-]?token)
            \s*(?:=|:)\s*['"]?([a-zA-Z0-9\-_.~+/=%]{30,})['"]?""",
            re.VERBOSE | re.IGNORECASE,
        ),
        0.88,
    ),
    # Slack tokens
    PatternRule(
        "token_slack",
        re.compile(r"xox[baprs]-[a-zA-Z0-9\-]{10,}"),
        0.97,
    ),
    # Twilio Account SID / Auth Token
    PatternRule(
        "token_twilio",
        re.compile(r"SK[a-f0-9]{32}|AC[a-f0-9]{32}"),
        0.95,
    ),
]

TOKEN = CategorySpec(
    category="TOKEN",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.85,
    rules=_TOKEN_RULES,
)

# ── PRIVATE_KEY ───────────────────────────────────────────────────────────────
_PRIVATE_KEY_RULES = [
    # PEM header lines
    PatternRule(
        "privkey_pem",
        re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+|ENCRYPTED\s+|PGP\s+)?"
            r"PRIVATE\s+KEY(?:\s+BLOCK)?-----",
        ),
        0.99,
    ),
    # Single-line base64 private key in env/config
    PatternRule(
        "privkey_env",
        re.compile(
            r"(?:PRIVATE[_\-]?KEY|private[_\-]?key)\s*(?:=|:)\s*"
            r"'?\"?(?:-----BEGIN[^'\"]{20,}|[A-Za-z0-9+/]{200,}={0,2})'?\"?",
            re.IGNORECASE,
        ),
        0.92,
    ),
]

PRIVATE_KEY = CategorySpec(
    category="PRIVATE_KEY",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.90,
    rules=_PRIVATE_KEY_RULES,
)

# ── PASSWORD ──────────────────────────────────────────────────────────────────
_PASSWORD_RULES = [
    # JSON/YAML/ENV assignment patterns. 키워드에 `<prefix>_pass[word|wd]` 형태를 추가 —
    # DB_PASS=, temporary_pass:, user_passwd= 같은 접두형 비밀번호 라벨도 잡는다.
    # 'pass' 단독은 제외(passport/passenger 오탐 방지) — 접두 '_pass'일 때만 인정.
    PatternRule(
        "password_assignment",
        re.compile(
            r"""(?:password|passwd|passphrase|secret|pwd|[a-z0-9]+_pass(?:word|wd)?)\s*
            (?:=|:)\s*
            ['"]?(?!\s*$)(?!\s*\*)([^\s'",;|&>]{6,})['"]?""",
            re.VERBOSE | re.IGNORECASE,
        ),
        0.82,
    ),
    # URL with embedded password: scheme://user:password@host
    PatternRule(
        "password_url",
        re.compile(
            r"(?:https?|ftp|postgresql|mysql|mongodb|redis|amqp)"
            r"://[^:@\s]+:([^@\s/]{6,})@[^\s]+",
            re.IGNORECASE,
        ),
        0.92,
    ),
]

PASSWORD = CategorySpec(
    category="PASSWORD",
    category_class=CategoryClass.SECRET,
    action=Action.BLOCK,
    mask_style=MaskStyle.TOKENIZE,
    min_confidence=0.80,
    rules=_PASSWORD_RULES,
)


# ══════════════════════════════════════════════════════════════════════════════
# Master registry — order matters: more specific rules first
# ══════════════════════════════════════════════════════════════════════════════

# Ordered: secrets first (higher priority), then high-risk PII, then contact PII
ALL_CATEGORIES: List[CategorySpec] = [
    # Secrets
    PRIVATE_KEY,
    AWS_SECRET,
    GCP_KEY,
    API_KEY,
    TOKEN,
    PASSWORD,
    # High-risk PII (block)
    RRN,
    FOREIGN_REG,
    CARD,
    PASSPORT,
    DRIVER_LICENSE,
    # Contact / context PII (mask/tokenize)
    EMAIL,
    PHONE,
    BIZ_NO,
    KR_ACCOUNT,
    PERSON,
    ADDRESS,
    # Server topology / network identifiers (mask)
    IP_ADDRESS,
    HOSTNAME,
    # Stage-2-only Korean categories (no Stage-1 rules; NER detection only)
    ORGANIZATION,
]

CATEGORY_MAP = {spec.category: spec for spec in ALL_CATEGORIES}
