"""
training/ingest.py — 보유 라벨 데이터를 GLiNER 학습 포맷으로 적재·검증 (ADR-13, 1급 입력).

입력(JSONL, 한 줄당 한 문서):
    {"text": "...", "spans": [[char_start, char_end, "PERSON|ADDRESS|ORGANIZATION"], ...]}
  - spans=[] 인 줄은 hard-negative(엔티티 없음) 예시로 그대로 사용된다.

출력(JSONL, GLiNER 학습 포맷):
    {"tokenized_text": [...], "ner": [[tok_start, tok_end_inclusive, label], ...]}

검증: 라벨 유효성, span 범위, 문자→토큰 매핑 실패(span/token 경계 불일치) 카운트·경고.

실행:
    .venv/bin/python -m training.ingest <raw.jsonl> <out.jsonl>
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

from .common import CANONICAL_LABELS, build_gliner_example, tokenize_with_offsets, char_span_to_token_span


def _validate_doc(doc: Dict, idx: int) -> List[str]:
    """문서 하나의 형식 오류 메시지 리스트(빈 리스트 = 정상)."""
    errs: List[str] = []
    if "text" not in doc or not isinstance(doc["text"], str):
        errs.append(f"[{idx}] 'text'(문자열) 필요")
        return errs
    spans = doc.get("spans", [])
    if not isinstance(spans, list):
        errs.append(f"[{idx}] 'spans'는 리스트여야 함")
        return errs
    n = len(doc["text"])
    for s in spans:
        if not (isinstance(s, list) and len(s) == 3):
            errs.append(f"[{idx}] span 형식 [start,end,label] 아님: {s!r}"); continue
        cs, ce, label = s
        if not (isinstance(cs, int) and isinstance(ce, int) and 0 <= cs < ce <= n):
            errs.append(f"[{idx}] span 범위 오류(0≤start<end≤{n}): {s!r}")
        if label not in CANONICAL_LABELS:
            errs.append(f"[{idx}] 라벨 {label!r} ∉ {CANONICAL_LABELS}")
    return errs


def ingest(in_path: str, out_path: str) -> Dict:
    """raw JSONL → GLiNER JSONL. 통계 dict 반환."""
    docs: List[Dict] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    all_errs: List[str] = []
    examples: List[Dict] = []
    n_spans = n_mapfail = n_neg = 0

    for i, doc in enumerate(docs):
        errs = _validate_doc(doc, i)
        if errs:
            all_errs.extend(errs)
            continue
        spans = doc.get("spans", [])
        # 문자→토큰 매핑 실패 카운트(경계 불일치 진단)
        toks = tokenize_with_offsets(doc["text"])
        for cs, ce, _ in spans:
            n_spans += 1
            if char_span_to_token_span(toks, cs, ce) is None:
                n_mapfail += 1
        if not spans:
            n_neg += 1
        examples.append(build_gliner_example(doc["text"], spans))

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "in_docs": len(docs),
        "out_examples": len(examples),
        "rejected": len(docs) - len(examples),
        "spans": n_spans,
        "span_map_failures": n_mapfail,
        "negative_examples": n_neg,
        "errors": all_errs[:20],  # 처음 20개만
    }
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m training.ingest <raw.jsonl> <out.jsonl>")
    st = ingest(sys.argv[1], sys.argv[2])
    print(json.dumps(st, ensure_ascii=False, indent=2))
    if st["rejected"]:
        print(f"⚠️ {st['rejected']}개 문서 거부(형식 오류) — errors 참고", file=sys.stderr)
    if st["span_map_failures"]:
        print(f"⚠️ span 매핑 실패 {st['span_map_failures']}/{st['spans']} "
              f"(라벨 경계가 토큰 경계와 안 맞음 — 라벨 점검 권장)", file=sys.stderr)
