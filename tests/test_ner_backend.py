"""
Tests for the selectable Stage-2 NER backend (R18 / ADR-11).

Covers backend resolution precedence, engine-class loading, GLiNER engine
label→category mapping (with a fake model so neither torch nor a downloaded
model is required), policy parsing of ``stage2.ner_backend``, and Engine
env-propagation.
"""
from __future__ import annotations

import pytest

from pii_guard.stage2.backend import (
    ENV_NER_BACKEND,
    NERBackend,
    load_engine_class,
    resolve_ner_backend,
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_ner_backend — precedence: env > policy > default
# ─────────────────────────────────────────────────────────────────────────────

def test_default_backend_is_gliner(monkeypatch):
    monkeypatch.delenv(ENV_NER_BACKEND, raising=False)
    assert resolve_ner_backend() is NERBackend.GLINER


def test_policy_backend_used_when_no_env(monkeypatch):
    monkeypatch.delenv(ENV_NER_BACKEND, raising=False)
    assert resolve_ner_backend("spacy") is NERBackend.SPACY
    assert resolve_ner_backend("gliner") is NERBackend.GLINER


def test_env_overrides_policy(monkeypatch):
    monkeypatch.setenv(ENV_NER_BACKEND, "spacy")
    # policy says gliner, but env (spacy) must win
    assert resolve_ner_backend("gliner") is NERBackend.SPACY


def test_env_value_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(ENV_NER_BACKEND, "  GLiNER ")
    assert resolve_ner_backend("spacy") is NERBackend.GLINER


def test_unknown_env_backend_raises(monkeypatch):
    monkeypatch.setenv(ENV_NER_BACKEND, "bogus")
    with pytest.raises(ValueError):
        resolve_ner_backend()


def test_unknown_policy_backend_raises(monkeypatch):
    monkeypatch.delenv(ENV_NER_BACKEND, raising=False)
    with pytest.raises(ValueError):
        resolve_ner_backend("transformer")


# ─────────────────────────────────────────────────────────────────────────────
# load_engine_class — lazy import returns the right class (no model load)
# ─────────────────────────────────────────────────────────────────────────────

def test_load_engine_class_spacy():
    from pii_guard.stage2.korean_ner import KoreanNEREngine

    assert load_engine_class(NERBackend.SPACY) is KoreanNEREngine


def test_load_engine_class_gliner():
    # Importing the class must NOT require torch (imports are deferred to detect()).
    from pii_guard.stage2.gliner_ner import GLiNERNEREngine

    assert load_engine_class(NERBackend.GLINER) is GLiNERNEREngine


def test_load_engine_class_nunerzero():
    # NuNER Zero candidate (R21/ADR-14) — also lazy; loads via the gliner library.
    from pii_guard.stage2.nunerzero_ner import NuNERZeroNEREngine

    assert load_engine_class(NERBackend.NUNERZERO) is NuNERZeroNEREngine


def test_nunerzero_backend_resolves_from_env(monkeypatch):
    monkeypatch.setenv(ENV_NER_BACKEND, "nunerzero")
    assert resolve_ner_backend("gliner") is NERBackend.NUNERZERO


# ─────────────────────────────────────────────────────────────────────────────
# GLiNERNEREngine.detect — mapping logic with a fake model (no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeGLiNERModel:
    """Stand-in for a loaded GLiNER model: returns canned predict_entities output."""

    def __init__(self, entities):
        self._entities = entities
        self.calls = []

    def predict_entities(self, text, labels, threshold=0.0):
        self.calls.append((text, tuple(labels), threshold))
        return self._entities


def _engine_with_fake(entities):
    from pii_guard.stage2.gliner_ner import GLiNERNEREngine

    eng = GLiNERNEREngine()
    eng._model = _FakeGLiNERModel(entities)  # inject fake → _get_model returns it
    return eng


def test_gliner_detect_maps_labels_to_categories():
    text = "김철수 씨가 서울특별시 강남구 삼성전자에 다닌다."
    # start/end here are illustrative; mapping logic is what's under test.
    entities = [
        {"start": 0, "end": 3, "text": "김철수", "label": "사람", "score": 0.95},
        {"start": 7, "end": 15, "text": "서울특별시 강남구", "label": "주소", "score": 0.9},
        {"start": 16, "end": 20, "text": "삼성전자", "label": "회사", "score": 0.88},
    ]
    eng = _engine_with_fake(entities)
    dets = eng.detect(text)

    cats = {d.category for d in dets}
    assert cats == {"PERSON", "ADDRESS", "ORGANIZATION"}
    for d in dets:
        assert d.detection_stage.value == "stage2_ner"
        assert d.rule_id.startswith("ner_gliner_")
        assert d.category_class.value == "korean_pii"


def test_gliner_detect_drops_below_confidence():
    text = "홍길동 010-0000-0000"
    entities = [
        {"start": 0, "end": 3, "text": "홍길동", "label": "사람", "score": 0.30},  # below 0.50
    ]
    eng = _engine_with_fake(entities)
    assert eng.detect(text) == []


def test_gliner_detect_strips_korean_particle():
    text = "담당자는 홍길동은 아닙니다"
    # GLiNER returned the name with a trailing particle attached ("홍길동은")
    entities = [
        {"start": 5, "end": 9, "text": "홍길동은", "label": "사람", "score": 0.9},
    ]
    eng = _engine_with_fake(entities)
    dets = eng.detect(text)
    assert len(dets) == 1
    assert dets[0].original == "홍길동"          # particle "은" stripped
    assert dets[0].end == 5 + len("홍길동")       # end adjusted after strip


def test_gliner_detect_skips_unknown_label():
    text = "2026년 6월 22일"
    entities = [
        {"start": 0, "end": 11, "text": "2026년 6월 22일", "label": "날짜", "score": 0.99},
    ]
    eng = _engine_with_fake(entities)
    assert eng.detect(text) == []  # "날짜" is not in the PII label map


def test_gliner_detect_empty_text_returns_empty():
    eng = _engine_with_fake([{"start": 0, "end": 3, "text": "김철수",
                              "label": "사람", "score": 0.9}])
    assert eng.detect("   ") == []


# ─────────────────────────────────────────────────────────────────────────────
# NuNERZeroNEREngine — candidate backend (R21/ADR-14)
# Reuses the GLiNER detect path (same gliner library API), so we verify the
# overridden identity points: model name, env var, and rule_id prefix.
# ─────────────────────────────────────────────────────────────────────────────

def test_nunerzero_default_model_and_env(monkeypatch):
    from pii_guard.stage2.nunerzero_ner import (
        NuNERZeroNEREngine,
        resolve_nunerzero_model,
    )

    monkeypatch.delenv("PIIGUARD_NUNERZERO_MODEL", raising=False)
    eng = NuNERZeroNEREngine()  # lazy — no model load
    assert eng._resolve_model_name() == "numind/NuNER_Zero"  # MIT default
    assert eng._MODEL_ENV_VAR == "PIIGUARD_NUNERZERO_MODEL"
    assert resolve_nunerzero_model() == "numind/NuNER_Zero"


def test_nunerzero_model_env_override(monkeypatch):
    from pii_guard.stage2.nunerzero_ner import NuNERZeroNEREngine

    monkeypatch.setenv("PIIGUARD_NUNERZERO_MODEL", "local/finetuned-nunerzero")
    assert NuNERZeroNEREngine()._resolve_model_name() == "local/finetuned-nunerzero"


def test_nunerzero_detect_uses_nunerzero_rule_prefix():
    # Inject a fake model (no torch) and confirm the inherited detect path emits
    # the candidate's rule_id prefix and the same normalized categories as GLiNER.
    from pii_guard.stage2.nunerzero_ner import NuNERZeroNEREngine

    text = "김철수 씨가 서울특별시 강남구 삼성전자에 다닌다."
    entities = [
        {"start": 0, "end": 3, "text": "김철수", "label": "사람", "score": 0.95},
        {"start": 7, "end": 15, "text": "서울특별시 강남구", "label": "주소", "score": 0.9},
        {"start": 16, "end": 20, "text": "삼성전자", "label": "회사", "score": 0.88},
    ]
    eng = NuNERZeroNEREngine()
    eng._model = _FakeGLiNERModel(entities)  # identical predict_entities API
    dets = eng.detect(text)

    assert {d.category for d in dets} == {"PERSON", "ADDRESS", "ORGANIZATION"}
    for d in dets:
        assert d.detection_stage.value == "stage2_ner"
        assert d.rule_id.startswith("ner_nunerzero_")   # candidate prefix, not ner_gliner_
        assert d.category_class.value == "korean_pii"


def test_nunerzero_merge_adjacent_same_label():
    # Token-classifier output fragments one address into two adjacent spans;
    # _merge_adjacent_entities must stitch them back into one.
    from pii_guard.stage2.nunerzero_ner import _merge_adjacent_entities

    text = "서울특별시 강남구"  # one address, but model split it
    frags = [
        {"start": 0, "end": 5, "text": "서울특별시", "label": "주소", "score": 0.8},
        {"start": 6, "end": 9, "text": "강남구", "label": "주소", "score": 0.9},
    ]
    merged = _merge_adjacent_entities(frags, text)
    assert len(merged) == 1
    assert merged[0]["start"] == 0 and merged[0]["end"] == 9
    assert merged[0]["text"] == "서울특별시 강남구"
    assert merged[0]["score"] == 0.9   # conservative max


def test_nunerzero_merge_keeps_different_labels_separate():
    from pii_guard.stage2.nunerzero_ner import _merge_adjacent_entities

    text = "김철수 삼성전자"
    frags = [
        {"start": 0, "end": 3, "text": "김철수", "label": "사람", "score": 0.9},
        {"start": 4, "end": 8, "text": "삼성전자", "label": "회사", "score": 0.9},
    ]
    merged = _merge_adjacent_entities(frags, text)
    assert len(merged) == 2  # different labels → not merged


def test_nunerzero_merge_empty_returns_empty():
    from pii_guard.stage2.nunerzero_ner import _merge_adjacent_entities

    assert _merge_adjacent_entities([], "anything") == []


def test_nunerzero_detect_merges_fragments_end_to_end():
    # Through the full detect() path: fragmented address spans collapse to one
    # ADDRESS Detection with the restored boundary.
    from pii_guard.stage2.nunerzero_ner import NuNERZeroNEREngine

    text = "주소는 서울특별시 강남구 입니다"
    frags = [
        {"start": 4, "end": 9, "text": "서울특별시", "label": "주소", "score": 0.8},
        {"start": 10, "end": 13, "text": "강남구", "label": "주소", "score": 0.85},
    ]
    eng = NuNERZeroNEREngine()
    eng._model = _FakeGLiNERModel(frags)
    dets = eng.detect(text)
    assert len(dets) == 1
    assert dets[0].category == "ADDRESS"
    assert dets[0].original == "서울특별시 강남구"


# ─────────────────────────────────────────────────────────────────────────────
# Policy parsing — stage2.ner_backend
# ─────────────────────────────────────────────────────────────────────────────

def test_policy_default_ner_backend():
    from pii_guard.policy import PolicyConfig

    assert PolicyConfig().ner_backend == "gliner"


def test_policy_parses_stage2_backend(tmp_path):
    from pii_guard.policy import load_policy

    p = tmp_path / "policy.yaml"
    p.write_text("stage2:\n  ner_backend: spacy\n", encoding="utf-8")
    cfg = load_policy(str(p))
    assert cfg.ner_backend == "spacy"


def test_policy_parses_nunerzero_backend(tmp_path):
    from pii_guard.policy import load_policy

    p = tmp_path / "policy.yaml"
    p.write_text("stage2:\n  ner_backend: nunerzero\n", encoding="utf-8")
    cfg = load_policy(str(p))
    assert cfg.ner_backend == "nunerzero"


def test_policy_invalid_stage2_backend_rejected():
    # _parse_and_validate raises on schema errors (load_policy then retains
    # last-valid config — that fail-safe path is covered in test_policy.py).
    from pii_guard.policy import _parse_and_validate

    with pytest.raises(ValueError):
        _parse_and_validate("stage2:\n  ner_backend: nonsense\n", source="<test>")


def test_policy_invalid_stage2_backend_retains_last_valid(tmp_path):
    # End-to-end: an invalid value must NOT silently apply; loader keeps the
    # secure default (gliner) rather than the bogus value.
    from pii_guard.policy import load_policy

    p = tmp_path / "policy.yaml"
    p.write_text("stage2:\n  ner_backend: nonsense\n", encoding="utf-8")
    cfg = load_policy(str(p))
    assert cfg.ner_backend == "gliner"


# ─────────────────────────────────────────────────────────────────────────────
# Engine env-propagation — the worker (separate process) reads PIIGUARD_NER_BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_sets_backend_env_from_policy(monkeypatch):
    import os
    from pii_guard.engine import Engine

    monkeypatch.delenv(ENV_NER_BACKEND, raising=False)
    Engine(ner_backend="spacy")
    assert os.environ[ENV_NER_BACKEND] == "spacy"


def test_engine_env_beats_policy(monkeypatch):
    import os
    from pii_guard.engine import Engine

    monkeypatch.setenv(ENV_NER_BACKEND, "gliner")
    Engine(ner_backend="spacy")  # policy spacy, but env gliner wins
    assert os.environ[ENV_NER_BACKEND] == "gliner"


def test_engine_defaults_backend_env_to_gliner(monkeypatch):
    import os
    from pii_guard.engine import Engine

    monkeypatch.delenv(ENV_NER_BACKEND, raising=False)
    Engine()
    assert os.environ[ENV_NER_BACKEND] == "gliner"


# ─────────────────────────────────────────────────────────────────────────────
# Runner warmup — loads the model outside the per-block timeout (no real model)
# ─────────────────────────────────────────────────────────────────────────────

def test_runner_warmup_returns_true_on_ok(monkeypatch):
    # Use the noop test worker (responds immediately, no model) to verify the
    # warmup handshake without loading any backend.
    from pii_guard.stage2 import _workers
    from pii_guard.stage2.runner import Stage2NERRunner

    r = Stage2NERRunner(_worker_target=_workers._test_noop_worker)
    try:
        assert r.warmup() is True
    finally:
        r.close()


def test_runner_warmup_false_when_worker_times_out():
    # The slow worker never responds; warmup must time out gracefully → False
    # (non-fatal), and a short budget keeps the test quick.
    from pii_guard.stage2 import _workers
    from pii_guard.stage2.runner import Stage2NERRunner

    r = Stage2NERRunner(_worker_target=_workers._test_slow_worker)
    try:
        assert r.warmup(timeout_seconds=1.0) is False
    finally:
        r.close()
