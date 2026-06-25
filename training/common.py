"""
training/common.py — GLiNER 파인튜닝 공용 유틸 (ADR-13 파일럿).

[이 파일이 하는 일]
보유 라벨 데이터(문자 오프셋 span)를 GLiNER 학습 포맷으로 바꾸는 데 필요한 공통 도구.
- 결정적 토크나이저(단어/구두점 단위, 각 토큰의 문자 오프셋 포함)
- 문자-span([char_start, char_end, label]) → 토큰-span([tok_start, tok_end(포함), label]) 변환
- GLiNER 학습 예시 빌더: {"tokenized_text":[...], "ner":[[ts,te,label],...]}

주의: 이 서브시스템은 **런타임(코어) 패키지가 아니다**(오프박스 학습용). 코어는 import하지 않는다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# GLiNER 학습 라벨 = 런타임 Detection 카테고리와 동일(PERSON/ADDRESS/ORGANIZATION).
# ⚠️ 통합 주의: 파인튜닝 모델을 배포해 쓰려면 런타임의 GLiNER 질의 라벨도 이 집합과 일치해야 한다
#   (현재 런타임 `gliner_ner._GLINER_LABELS`는 한국어 동의어를 쓰므로, 파인튜닝 모델 채택 시
#    질의 라벨을 이 캐노니컬 집합으로 정렬할 것 — README 통합 절 참고).
CANONICAL_LABELS: Tuple[str, ...] = ("PERSON", "ADDRESS", "ORGANIZATION")

# 토큰 = 연속한 단어문자(한글 포함) 덩어리 또는 단일 비공백 기호.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """텍스트를 (토큰, 문자시작, 문자끝(배타)) 리스트로 분해(결정적)."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def char_span_to_token_span(
    tokens: List[Tuple[str, int, int]], c_start: int, c_end: int
) -> Tuple[int, int] | None:
    """
    문자 span [c_start, c_end)를 덮는 토큰 인덱스 범위 (ts, te)를 반환(te는 **포함** 인덱스, GLiNER 규약).
    겹치는 토큰이 없으면 None.
    """
    covered = [
        i for i, (_, ts, te) in enumerate(tokens)
        if ts < c_end and te > c_start  # 토큰과 span이 겹침
    ]
    if not covered:
        return None
    return covered[0], covered[-1]


def build_gliner_example(text: str, spans: List[List]) -> Dict:
    """
    한 문서를 GLiNER 학습 예시로 변환.

    Parameters
    ----------
    text : 원문.
    spans : [[char_start, char_end, label], ...]  (label ∈ CANONICAL_LABELS).
            빈 리스트면 **hard negative**(엔티티 없음) 예시가 된다.

    Returns
    -------
    {"tokenized_text": [tok, ...], "ner": [[tok_start, tok_end_inclusive, label], ...]}
    """
    toks = tokenize_with_offsets(text)
    tokenized = [t for (t, _, _) in toks]
    ner: List[List] = []
    for cs, ce, label in spans:
        span = char_span_to_token_span(toks, cs, ce)
        if span is None:
            continue  # 매핑 실패 span은 건너뜀(검증 단계에서 경고)
        ner.append([span[0], span[1], label])
    return {"tokenized_text": tokenized, "ner": ner}


def example_text(example: Dict) -> str:
    """GLiNER 예시의 토큰을 공백으로 이어 붙인 근사 원문(누설 검사·디버그용)."""
    return " ".join(example.get("tokenized_text", []))
