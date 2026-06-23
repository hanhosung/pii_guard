"""
Tests for positive proximity (context-gated) detection — Phase 2.

Promote ambiguous account / biz-no / Korean-password ONLY when a trigger keyword
is nearby; do NOT promote the same shapes without context (FP suppression).
"""
from __future__ import annotations

from pii_guard.proximity import scan
from pii_guard.engine import Engine


def _cats(text):
    return {(d.category, d.original) for d in scan(text)}


# ── Promote WITH context ──────────────────────────────────────────────────────
def test_account_336_promoted_with_bank_context():
    assert ("KR_ACCOUNT", "123-456-789012") in _cats("환불은 국민은행 123-456-789012 계좌로")


def test_account_427_kakao_promoted():
    assert ("KR_ACCOUNT", "3333-01-1234567") in _cats("입금 계좌는 카카오뱅크 3333-01-1234567 입니다")


def test_bare_biz_no_promoted_with_keyword_and_valid_checksum():
    # 1806341205 is a valid biz checksum (used in efficacy corpus)
    assert ("BIZ_NO", "1806341205") in _cats("사업자번호 1806341205 입니다")


def test_korean_password_label_promoted():
    assert ("PASSWORD", "SecurePw99") in _cats("인터넷뱅킹 비밀번호: SecurePw99 라고")


# ── Do NOT promote WITHOUT context (FP suppression) ───────────────────────────
def test_account_336_not_promoted_without_context():
    # bare ambiguous number with no bank/account word nearby → not an account
    assert not any(c == "KR_ACCOUNT" for c, _ in _cats("주문번호 123-456-789012 확인"))


def test_bare_10_digit_not_promoted_without_keyword():
    assert not any(c == "BIZ_NO" for c, _ in _cats("운송장 1806341205 조회"))


def test_invalid_biz_checksum_not_promoted():
    assert not any(c == "BIZ_NO" for c, _ in _cats("사업자번호 1234567890 입니다"))  # invalid checksum


# ── End-to-end via Engine (Stage-1.5 merge) ───────────────────────────────────
def test_engine_picks_up_proximity_account():
    eng = Engine()  # Stage-1 only — proximity still runs
    r = eng.scan("신한은행 123-456-789012 로 입금")
    assert any(d.category == "KR_ACCOUNT" for d in r.detections)


def test_engine_proximity_can_be_disabled():
    eng = Engine(proximity_enabled=False)
    r = eng.scan("신한은행 123-456-789012 로 입금")
    assert not any(d.category == "KR_ACCOUNT" for d in r.detections)


def test_account_subsumes_spurious_phone_submatch():
    # Stage-1 phone regex carves '02-7654321' out of the account; the proximity
    # account must subsume it (replace), not be skipped for overlapping.
    eng = Engine()
    r = eng.scan("입금은 카카오뱅크 3333-02-7654321 로 할게요.")
    cats = {(d.category, d.original) for d in r.detections}
    assert ("KR_ACCOUNT", "3333-02-7654321") in cats
    assert not any(c == "PHONE" for c, _ in cats)  # spurious phone removed
