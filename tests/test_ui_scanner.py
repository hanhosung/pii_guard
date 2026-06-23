"""
Tests for the UI scan/format helpers (ui/scanner.py).

These cover the verdict logic (block vs mask vs clean) and the console-report
rendering used by the Streamlit app, using a Stage-1 Engine (no NER subprocess)
so they stay fast and dependency-light.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))
from scanner import render_console_block, scan_text, verdict  # noqa: E402

from pii_guard.engine import Engine


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine()  # Stage-1 only — deterministic, no subprocess


def test_block_category_marks_blocked(engine):
    res = scan_text(engine, "key AKIAIOSFODNN7EXAMPLE")
    assert res["has_blocks"] is True
    assert verdict(res)[0].startswith("🔴")
    assert any(r["action"] == "BLOCK" for r in res["rows"])
    assert "BLOCKED" in res["masked"]  # placeholder is [AWS_SECRET_1_BLOCKED]


def test_mask_category_marks_masked(engine):
    res = scan_text(engine, "메일 minsu@corp.co.kr 전화 010-1234-5678")
    assert res["has_blocks"] is False
    assert res["has_masks"] is True
    assert verdict(res)[1] == "orange"
    assert "[EMAIL_1]" in res["masked"] and "[PHONE_1]" in res["masked"]
    assert "minsu@corp.co.kr" not in res["masked"]


def test_clean_text_marks_clean(engine):
    res = scan_text(engine, "오늘 날씨가 참 좋네요.")
    assert res["has_blocks"] is False
    assert res["has_masks"] is False
    assert verdict(res)[0].startswith("🟢")
    assert res["masked"] == res["original"]


def test_console_block_contains_key_sections(engine):
    res = scan_text(engine, "전화 010-1234-5678")
    report = render_console_block("chat message", res)
    assert "INPUT: chat message" in report
    assert "VERDICT:" in report
    assert "ORIGINAL:" in report
    assert "MASKED" in report
    assert "PHONE" in report


def test_console_block_no_detections(engine):
    res = scan_text(engine, "그냥 평범한 문장입니다.")
    report = render_console_block("clean", res)
    assert "DETECTIONS: none" in report
