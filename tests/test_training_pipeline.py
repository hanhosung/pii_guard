"""
Tests for the GLiNER fine-tuning pipeline (ADR-13) — deterministic parts only.

Covers char-span→token-span conversion, ingest validation/conversion, split
leakage-guard + determinism, and augment output shape. Does NOT load torch/
gliner or train a model (that is off-box/GPU).
"""
from __future__ import annotations

import json

from training.common import (
    CANONICAL_LABELS,
    build_gliner_example,
    char_span_to_token_span,
    tokenize_with_offsets,
)
from training import augment, ingest, split


# ── common: tokenize + char→token span ──────────────────────────────────────

def test_tokenize_offsets_roundtrip():
    text = "담당자 김서연 님"
    toks = tokenize_with_offsets(text)
    assert [t for t, _, _ in toks] == ["담당자", "김서연", "님"]
    # offsets point back at the original substring
    for tok, s, e in toks:
        assert text[s:e] == tok


def test_char_span_to_token_span_inclusive():
    text = "담당자 김서연 님께 전달"          # tokens: 담당자(0) 김서연(1) 님께(2) 전달(3)
    toks = tokenize_with_offsets(text)
    cs = text.index("김서연")
    assert char_span_to_token_span(toks, cs, cs + 3) == (1, 1)   # inclusive token idx


def test_build_gliner_example_maps_spans():
    text = "협력사는 삼성전자 입니다."
    cs = text.index("삼성전자")
    ex = build_gliner_example(text, [[cs, cs + 4, "ORGANIZATION"]])
    assert ex["tokenized_text"][0] == "협력사는"
    assert len(ex["ner"]) == 1
    ts, te, label = ex["ner"][0]
    assert label == "ORGANIZATION"
    assert ex["tokenized_text"][ts:te + 1] == ["삼성전자"]


def test_build_gliner_example_negative_has_empty_ner():
    ex = build_gliner_example("주문번호 ORD-2026 확인", [])
    assert ex["ner"] == []
    assert ex["tokenized_text"]  # tokens still present


# ── ingest: validation + conversion ─────────────────────────────────────────

def test_ingest_converts_and_validates(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps({"text": "담당자 김서연 님", "spans": [[4, 7, "PERSON"]]}, ensure_ascii=False) + "\n"
        + json.dumps({"text": "버전 v3.2.0", "spans": []}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "g.jsonl"
    st = ingest.ingest(str(raw), str(out))
    assert st["in_docs"] == 2 and st["out_examples"] == 2 and st["rejected"] == 0
    assert st["negative_examples"] == 1
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["ner"][0][2] == "PERSON"


def test_ingest_rejects_bad_label_and_range(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps({"text": "abc", "spans": [[0, 1, "BOGUS"]]}, ensure_ascii=False) + "\n"
        + json.dumps({"text": "abc", "spans": [[0, 99, "PERSON"]]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "g.jsonl"
    st = ingest.ingest(str(raw), str(out))
    assert st["rejected"] == 2          # both docs rejected (bad label / out-of-range)
    assert st["errors"]


# ── split: leakage guard + determinism ──────────────────────────────────────

def _ex(text):
    return build_gliner_example(text, [])


def test_split_leakage_guard_drops_eval_overlap():
    exs = [_ex(f"문장 {i}") for i in range(10)]
    leak_text = "문장 3"
    fp = {split.fingerprint(leak_text)}
    res = split.split(exs, eval_fingerprints=fp, seed=1)
    assert res["dropped_leak"] == 1
    all_kept = res["train"] + res["val"] + res["test"]
    from training.common import example_text
    assert all(split.fingerprint(example_text(e)) not in fp for e in all_kept)


def test_split_deterministic_and_partitions():
    exs = [_ex(f"문장 {i}") for i in range(20)]
    a = split.split(exs, seed=42)
    b = split.split(exs, seed=42)
    assert [e["tokenized_text"] for e in a["train"]] == [e["tokenized_text"] for e in b["train"]]
    total = len(a["train"]) + len(a["val"]) + len(a["test"])
    assert total == 20  # no loss, no overlap (partition)


# ── augment: shape ──────────────────────────────────────────────────────────

def test_augment_positive_spans_valid_and_negatives_empty():
    docs = augment.generate(n_pos=10, n_neg=10, seed=7)
    assert len(docs) == 20
    pos = [d for d in docs if d["spans"]]
    neg = [d for d in docs if not d["spans"]]
    assert pos and neg
    for d in pos:
        cs, ce, label = d["spans"][0]
        assert label in CANONICAL_LABELS
        assert d["text"][cs:ce]            # span points at non-empty substring
