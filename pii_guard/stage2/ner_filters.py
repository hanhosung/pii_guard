"""
pii_guard/stage2/ner_filters.py

Negative-proximity / NER false-positive suppression (PROXIMITY_DESIGN.md Phase 1).

Stage-2 NER (spaCy ko) over-extracts on code/technical text and on a handful of
common Korean nouns, mislabelling them PERSON / ORGANIZATION / ADDRESS. The
efficacy validation (validation/EXTERNAL_LLM_TEST_2026-06-23_claude.md) traced 36 true over-masking
FPs, concentrated in:

  - **code / technical tokens** — ``API_KEY``, ``DB_PASSWORD``, ``send_email(...)``,
    ``example.com``, base64 blobs, all-caps acronyms (``AWS``, ``LGTM``, ``JWT``).
  - **common Korean nouns** — ``주석``(comment), ``리턴값``(return value),
    ``수익자``(beneficiary), ``여권번호``(passport-no label), etc.

This module is a **deterministic post-filter** on NER detections: it only ever
**drops** a detection (never adds), so it cannot reduce recall on real entities —
it raises precision by removing things that are provably not Korean person /
organization / address spans.

Design guarantees
-----------------
- **Deterministic / auditable** — pure rules, no model. Consistent with the
  rule-based hybrid (requirements DR-1 / DR-2).
- **Recall-safe** — TitleCase ASCII names (``John Smith``) are NOT filtered, so
  real English personal names still pass; only identifier-shaped / acronym /
  blob spans and an explicit common-noun deny-list are removed.
"""
from __future__ import annotations

import os
import re

# ── Runtime config via env (propagates into the spawned Stage-2 subprocess) ──
# The policy ``proximity.ner_filter_enabled`` / ``ner_extra_stopwords`` are
# pushed to these env vars by Engine/serve before the NER worker is spawned;
# the worker (separate process) inherits them and reads them here at call time.
_ENV_DISABLE = "PIIGUARD_NER_FILTER_OFF"
_ENV_EXTRA = "PIIGUARD_NER_EXTRA_STOPWORDS"


def _runtime_disabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "on", "yes")


def _extra_stopwords() -> frozenset:
    raw = os.environ.get(_ENV_EXTRA, "")
    return frozenset(w.strip() for w in raw.split(",") if w.strip())

# Spans that NER classifies as PERSON/ORG/ADDRESS but are common Korean nouns,
# never real entities. Compared after stripping spaces. Keep CONSERVATIVE —
# every entry must be a word that can never be a real name/org/address.
KOREAN_NER_STOPWORDS = frozenset({
    "주석", "리턴값", "로깅", "수익자", "여권번호", "문진표", "생년월일",
    "네고", "유선", "프라이빗", "비밀번호", "비번", "암호", "담당자",
    "수신자", "발신자", "참조", "첨부", "본문", "제목", "회신",
})

# Lowercase ASCII common words that NER mislabels as person/org.
ENGLISH_NER_STOPWORDS = frozenset({
    "rotate", "admin", "root", "user", "test", "null", "true", "false",
    "config", "debug", "release", "build", "deploy", "commit",
})

# Identifier / code-token characters — a real Korean PERSON/ORG/ADDRESS span
# never contains these.
_CODE_CHARS = set("_()=;{}[]<>@/\\|&%$#`")

_DOMAIN_RE = re.compile(r"\.(com|net|org|io|kr|co|ai|dev|gov|edu)\b", re.IGNORECASE)
_ACRONYM_RE = re.compile(r"[A-Z]{2,5}")
_ASCII_BLOB_RE = re.compile(r"[A-Za-z0-9+/=]+")
_HANGUL_RE = re.compile(r"[가-힣]")

# Categories produced by the Korean NER engine that this filter applies to.
_NER_CATEGORIES = frozenset({"PERSON", "ADDRESS", "ORGANIZATION"})


def is_ner_false_positive(category: str, text: str) -> bool:
    """
    Return True if *text* detected as *category* by Stage-2 NER should be dropped
    as a false positive (code token / acronym / blob / common noun).

    Only applies to NER categories; other categories pass through unchanged.
    """
    if category not in _NER_CATEGORIES:
        return False
    if _runtime_disabled():
        return False  # policy disabled the negative filter entirely

    t = text.strip()
    if not t:
        return True

    no_space = t.replace(" ", "")

    # 1. Deny-lists (common nouns / code keywords) — built-in + policy-supplied.
    if no_space in KOREAN_NER_STOPWORDS or no_space in _extra_stopwords():
        return True
    if t.lower() in ENGLISH_NER_STOPWORDS:
        return True

    # 2. Identifier / code-token shape (e.g. API_KEY, send_email(...), a=b).
    if any(ch in t for ch in _CODE_CHARS):
        return True

    # 3. Domain-like (example.com).
    if _DOMAIN_RE.search(t):
        return True

    # 4. All-caps ASCII acronym (AWS, LGTM, JWT, NDA, HS).
    if _ACRONYM_RE.fullmatch(t):
        return True

    # 5. Long ASCII alphanumeric blob with no Hangul (base64 / hash fragments).
    #    Real names contain Hangul or a space; this only catches keyless blobs.
    if len(no_space) >= 16 and not _HANGUL_RE.search(t) and _ASCII_BLOB_RE.fullmatch(no_space):
        return True

    return False
