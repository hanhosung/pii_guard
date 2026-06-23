"""
Tests for policy-YAML exposure of proximity settings (Phase 3).

The policy ``proximity:`` block drives trigger keywords / window / enable flags
and the NER false-positive filter knobs.
"""
from __future__ import annotations

import os

import pytest

from pii_guard.engine import Engine
from pii_guard.policy import _parse_and_validate, SECURE_DEFAULTS
from pii_guard.proximity import ProximityConfig


def _cfg(yaml_str):
    config, _ = _parse_and_validate(yaml_str, "<test>")
    return config.proximity


# ── Parsing ───────────────────────────────────────────────────────────────────
def test_default_proximity_is_secure_default():
    assert isinstance(SECURE_DEFAULTS.proximity, ProximityConfig)
    assert SECURE_DEFAULTS.proximity.enabled is True


def test_parse_custom_triggers_and_window():
    p = _cfg("proximity:\n  account_triggers: [사내은행, 입금]\n  window_chars: 30\n")
    assert p.account_triggers == ("사내은행", "입금")
    assert p.window_chars == 30


def test_parse_enabled_false():
    assert _cfg("proximity: { enabled: false }").enabled is False


def test_parse_password_keywords_and_ner_knobs():
    p = _cfg(
        "proximity:\n"
        "  password_keywords: [패스워드]\n"
        "  ner_filter_enabled: false\n"
        "  ner_extra_stopwords: [사내용어]\n"
    )
    assert p.password_keywords == ("패스워드",)
    assert p.ner_filter_enabled is False
    assert p.ner_extra_stopwords == ("사내용어",)


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        _cfg("proximity: { window_chars: 999 }")


def test_invalid_triggers_type_rejected():
    with pytest.raises(ValueError):
        _cfg("proximity: { account_triggers: 'not-a-list' }")


# ── End-to-end via Engine ─────────────────────────────────────────────────────
def test_custom_trigger_drives_detection():
    p = _cfg("proximity:\n  account_triggers: [사내은행]\n")
    eng = Engine(proximity_config=p)
    r = eng.scan("사내은행 123-456-789012 로 입금")
    assert any(d.category == "KR_ACCOUNT" for d in r.detections)


def test_policy_disable_turns_proximity_off():
    p = _cfg("proximity: { enabled: false }")
    eng = Engine(proximity_config=p)
    r = eng.scan("신한은행 123-456-789012 로 입금")
    assert not any(d.category == "KR_ACCOUNT" for d in r.detections)


def test_engine_sets_ner_filter_env_from_policy():
    p = _cfg("proximity:\n  ner_filter_enabled: false\n  ner_extra_stopwords: [홍길동]\n")
    Engine(proximity_config=p)  # constructor pushes env
    assert os.environ.get("PIIGUARD_NER_FILTER_OFF") == "1"
    assert "홍길동" in os.environ.get("PIIGUARD_NER_EXTRA_STOPWORDS", "")
    # restore default for other tests
    Engine(proximity_config=ProximityConfig())
    assert os.environ.get("PIIGUARD_NER_FILTER_OFF") == ""
