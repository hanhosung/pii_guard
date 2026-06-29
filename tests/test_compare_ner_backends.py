"""
tests/test_compare_ner_backends.py

Unit tests for the ADR-14 NER backend comparison + adoption-gate logic
(``benchmarks/compare_ner_backends.py``). These cover the PURE logic (F1,
macro-F1, gate evaluation, Markdown rendering) with synthetic metric cells, so
no NER model / torch is loaded.
"""
from __future__ import annotations

from benchmarks.compare_ner_backends import (
    _evaluate_gate,
    _f1,
    _macro_f1,
    _render_markdown,
)


def _cell(backend, thr, *, person=(0.93, 0.96), address=(0.92, 0.90),
          org=(0.774, 0.96), available=True):
    metrics = {
        "PERSON":       {"precision": person[0], "recall": person[1]},
        "ADDRESS":      {"precision": address[0], "recall": address[1]},
        "ORGANIZATION": {"precision": org[0], "recall": org[1]},
    }
    return {
        "backend": backend,
        "min_confidence": thr,
        "ner_available": available,
        "load_error": None,
        "metrics": metrics,
        "macro_f1": round(_macro_f1(metrics), 4),
    }


# ── F1 helpers ────────────────────────────────────────────────────────────────

def test_f1_basic():
    assert _f1(1.0, 1.0) == 1.0
    assert _f1(0.0, 0.0) == 0.0
    assert abs(_f1(0.5, 1.0) - 0.6667) < 1e-3


def test_macro_f1_averages_categories():
    metrics = {
        "PERSON":       {"precision": 1.0, "recall": 1.0},   # F1 1.0
        "ADDRESS":      {"precision": 0.0, "recall": 0.0},   # F1 0.0
        "ORGANIZATION": {"precision": 1.0, "recall": 1.0},   # F1 1.0
    }
    assert abs(_macro_f1(metrics) - (2.0 / 3.0)) < 1e-6


# ── Adoption gate ─────────────────────────────────────────────────────────────

def test_gate_passes_on_org_precision_gain_no_recall_regression():
    cells = [
        _cell("gliner", 0.50, org=(0.774, 0.96)),
        _cell("nunerzero", 0.50, org=(0.880, 0.96)),  # ORG precision up, recall equal
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    assert gate["passed"] is True
    t = gate["per_threshold"][0]
    assert t["org_precision_gain"] is True
    assert t["no_recall_regression"] is True


def test_gate_fails_on_recall_regression():
    cells = [
        _cell("gliner", 0.50, person=(0.93, 0.96)),
        _cell("nunerzero", 0.50, person=(0.99, 0.80)),  # PERSON recall drops 0.96→0.80
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    assert gate["passed"] is False
    assert any("PERSON recall" in r
               for r in gate["per_threshold"][0]["recall_regressions"])


def test_gate_passes_on_macro_f1_win_without_org_gain():
    # No ORG precision gain, but candidate macro-F1 >= baseline and no recall loss.
    cells = [
        _cell("gliner", 0.50, address=(0.80, 0.80), org=(0.80, 0.80)),
        _cell("nunerzero", 0.50, address=(0.95, 0.95), org=(0.80, 0.80)),
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    t = gate["per_threshold"][0]
    assert t["org_precision_gain"] is False
    assert t["macro_f1_win"] is True
    assert gate["passed"] is True


def test_gate_passes_if_any_threshold_passes():
    cells = [
        _cell("gliner", 0.50, org=(0.80, 0.96)),
        _cell("nunerzero", 0.50, org=(0.70, 0.96)),   # fails at 0.50 (ORG prec down, f1 down)
        _cell("gliner", 0.35, org=(0.80, 0.96)),
        _cell("nunerzero", 0.35, org=(0.90, 0.96)),   # passes at 0.35
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    assert gate["passed"] is True


def test_gate_marks_unavailable_threshold_not_evaluable():
    cells = [
        _cell("gliner", 0.50, available=True),
        _cell("nunerzero", 0.50, available=False),   # deps missing
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    assert gate["passed"] is False
    assert gate["per_threshold"][0]["evaluable"] is False


def test_recall_tolerance_allows_small_drop():
    cells = [
        _cell("gliner", 0.50, person=(0.93, 0.96), org=(0.80, 0.96)),
        _cell("nunerzero", 0.50, person=(0.99, 0.94), org=(0.90, 0.96)),  # 0.02 recall drop
    ]
    strict = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                            recall_tolerance=0.0)
    assert strict["passed"] is False  # 0.02 drop counts as regression
    lenient = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                             recall_tolerance=0.03)
    assert lenient["passed"] is True  # within tolerance + ORG precision gain


# ── Markdown rendering ────────────────────────────────────────────────────────

def test_render_markdown_includes_table_and_gate():
    cells = [
        _cell("gliner", 0.50),
        _cell("nunerzero", 0.50, org=(0.88, 0.96)),
    ]
    gate = _evaluate_gate(cells, candidate="nunerzero", baseline="gliner",
                          recall_tolerance=0.0)
    md = _render_markdown(cells, [0.50], ["gliner", "nunerzero"], gate, 42, 5)
    assert "min_confidence = 0.5" in md
    assert "ORGANIZATION" in md
    assert "macro-F1" in md
    assert "Adoption gate" in md
    assert "PASS" in md


def test_render_markdown_flags_unavailable_cells():
    cells = [
        _cell("gliner", 0.50, available=True),
        {**_cell("nunerzero", 0.50, available=False), "load_error": "gliner not installed"},
    ]
    md = _render_markdown(cells, [0.50], ["gliner", "nunerzero"], None, 42, 5)
    assert "Unavailable cells" in md
    assert "gliner not installed" in md
