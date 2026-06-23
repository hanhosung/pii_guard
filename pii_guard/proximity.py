"""
pii_guard/proximity.py

Positive proximity (context-gated) detection — PROXIMITY_DESIGN.md Phase 2.

Some real PII has an **ambiguous shape** that the Stage-1 regex rules deliberately
do NOT match, to avoid a false-positive explosion (e.g. a bare ``123-456-789012``
could be an order number, a courier id, …). The efficacy validation
(validation/EFFICACY_REPORT.md) showed these slip through:

  - **KR_ACCOUNT** — non-standard ``3-3-6`` and Kakao/Toss ``4-2-7`` formats.
  - **BIZ_NO** — hyphen-less 10-digit business registration numbers.
  - **PASSWORD** — Korean label ("비밀번호: …"), where only the English
    ``password=`` keyword was recognised before.

This module promotes such an ambiguous value to a real detection **only when a
trigger keyword is nearby** (a bank name / "입금"·"계좌" for accounts,
"사업자" for biz-no, "비밀번호"·"암호" for passwords). That keeps recall up
without re-introducing the FPs — promote *only when context confirms*.

Properties: deterministic, regex-based, prompt-injection-immune, auditable (the
matched trigger is recorded in ``rule_id``). Consistent with requirements DR-2.
"""
from __future__ import annotations

import re
from typing import Callable, List, NamedTuple, Optional, Tuple

from .categories import _kr_biz_checksum
from .models import Action, CategoryClass, Detection, DetectionStage, MaskStyle

# ── Trigger vocabularies ──────────────────────────────────────────────────────
_BANKS = (
    "국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "카카오",
    "토스뱅크", "토스", "케이뱅크", "SC", "씨티", "산업", "수협", "새마을",
    "신협", "우체국", "대구", "부산", "경남", "광주", "전북", "제주",
)
_ACCOUNT_VERBS = (
    "계좌", "입금", "이체", "송금", "환불", "예금주", "수령", "받을", "보낼", "보내",
)
_ACCOUNT_TRIGGERS = _BANKS + _ACCOUNT_VERBS
_BIZ_TRIGGERS = ("사업자",)


class ContextRule(NamedTuple):
    category: str
    category_class: CategoryClass
    action: Action
    mask_style: MaskStyle
    value_pattern: re.Pattern            # group(1) = PII value if grouped, else whole match
    triggers: Tuple[str, ...]            # at least one must appear within ±window of the value
    window: int
    confidence: float
    rule_id: str
    validator: Optional[Callable[[str], bool]] = None  # extra checksum on normalized value


CONTEXT_RULES: Tuple[ContextRule, ...] = (
    # KR_ACCOUNT — non-standard 3-3-6 (e.g. 123-456-789012)
    ContextRule(
        "KR_ACCOUNT", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
        re.compile(r"(?<!\d)(\d{3}-\d{3}-\d{6})(?!\d)"),
        _ACCOUNT_TRIGGERS, 25, 0.70, "prox_kr_acct_336",
    ),
    # KR_ACCOUNT — Kakao/Toss style 4-2-7 (e.g. 3333-01-1234567)
    ContextRule(
        "KR_ACCOUNT", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
        re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{7})(?!\d)"),
        _ACCOUNT_TRIGGERS, 25, 0.70, "prox_kr_acct_427",
    ),
    # BIZ_NO — hyphen-less 10 digits, gated by "사업자" + valid checksum
    ContextRule(
        "BIZ_NO", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
        re.compile(r"(?<!\d)(\d{10})(?!\d)"),
        _BIZ_TRIGGERS, 20, 0.85, "prox_biz_bare10", validator=_kr_biz_checksum,
    ),
    # PASSWORD — Korean label (비밀번호 / 비번 / 암호) : value
    ContextRule(
        "PASSWORD", CategoryClass.SECRET, Action.BLOCK, MaskStyle.TOKENIZE,
        re.compile(r"(?:비밀번호|비번|암호)\s*[:=]?\s*([^\s,，.。!?'\"]{4,40})"),
        (), 0, 0.85, "prox_password_kr",
    ),
)


def _norm(v: str) -> str:
    return v.replace("-", "").replace(" ", "")


def scan(text: str) -> List[Detection]:
    """Return context-gated proximity detections for *text* (may be empty)."""
    if not text:
        return []
    out: List[Detection] = []
    for rule in CONTEXT_RULES:
        for m in rule.value_pattern.finditer(text):
            if m.groups():
                start, end, value = m.start(1), m.end(1), m.group(1)
            else:
                start, end, value = m.start(), m.end(), m.group()

            # proximity gate
            trig_hit = None
            if rule.triggers:
                window = text[max(0, start - rule.window): end + rule.window]
                trig_hit = next((t for t in rule.triggers if t in window), None)
                if trig_hit is None:
                    continue

            # optional checksum
            if rule.validator is not None and not rule.validator(_norm(value)):
                continue

            rid = rule.rule_id + (f"+{trig_hit}" if trig_hit else "")
            out.append(Detection(
                category=rule.category,
                category_class=rule.category_class,
                action=rule.action,
                mask_style=rule.mask_style,
                start=start,
                end=end,
                original=value,
                detection_stage=DetectionStage.STAGE1_PROXIMITY,
                rule_id=rid,
                confidence=rule.confidence,
            ))
    return out


def merge(base: List[Detection], extra: List[Detection]) -> List[Detection]:
    """
    Merge *extra* (proximity) detections into *base*.

    Overlap policy (per detection d in extra):
      - if an existing detection *contains* d → keep the existing one, drop d;
      - if d *strictly contains* existing detection(s) → d is the more complete
        interpretation, so those sub-matches are removed and d is added
        (e.g. account ``3333-02-7654321`` subsumes a spurious phone match
        ``02-7654321`` that the Stage-1 regex carved out of it);
      - any other partial overlap → conservatively drop d.
    """
    merged = list(base)
    for d in extra:
        skip = False
        subsumed: List[Detection] = []
        for b in merged:
            if d.end <= b.start or d.start >= b.end:
                continue  # disjoint
            if b.start <= d.start and b.end >= d.end and (b.end - b.start) >= (d.end - d.start):
                skip = True  # b contains d → keep b
                break
            if d.start <= b.start and d.end >= b.end:
                subsumed.append(b)  # d contains b → replace b with d
            else:
                skip = True  # partial overlap → conservative
                break
        if skip:
            continue
        for b in subsumed:
            merged.remove(b)
        merged.append(d)
    merged.sort(key=lambda x: x.start)
    return merged
