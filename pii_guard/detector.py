"""
Stage 1 detection engine — pure regex + checksum scanning.

[이 파일이 하는 일 — 한 줄 요약]
탐지 1단계(Stage1)의 실제 엔진. 텍스트를 등록된 모든 카테고리 패턴(정규식)과 대조하고,
선택적으로 체크섬(예: 카드번호 Luhn) 검증까지 통과한 것만 Detection으로 만든다.
같은 자리에서 여러 패턴이 겹치면 '우선순위 높은 카테고리 → 더 긴 매칭'을 살리고 나머지는 버린다.

Scans a text string against all registered category patterns and returns
a list of Detection objects sorted by position.  Overlapping matches are
resolved by preferring the category that appears first in ALL_CATEGORIES
(i.e. higher-priority categories win) and then by longest match.
"""
from __future__ import annotations

import re                                          # 정규식 타입/매칭
from typing import Dict, List, Optional, Tuple     # 타입 힌트

from .categories import ALL_CATEGORIES, CATEGORY_MAP, CategorySpec, PatternRule  # 카테고리·규칙 정의
from .models import Action, Detection, DetectionStage  # 탐지 결과 타입/단계 표시


# 기본 HMAC 키 — 호출자(Engine)가 실제 비밀키를 넘겨주는 게 정상. 이 값은 절대 운영에서 쓰면 안 됨.
_DEFAULT_HMAC_KEY = b"pii-guard-default-do-not-use-in-prod"


def _resolve_capture_group(pattern: re.Pattern, m: re.Match) -> Tuple[int, int, str]:
    """
    가장 적절한 스팬의 (시작, 끝, 텍스트)를 반환.
    - 패턴에 캡처 그룹이 정확히 1개이고 그게 매칭됐으면 group(1)을 쓴다(앞 라벨 등 제외, 값만).
    - 그 외엔 전체 매칭(group(0))을 쓴다.
    """
    if m.lastindex and m.lastindex >= 1 and m.group(1) is not None:  # 캡처 그룹이 있으면
        return m.start(1), m.end(1), m.group(1)                      # 그 그룹만(값 부분)
    return m.start(0), m.end(0), m.group(0)                          # 없으면 전체 매칭


def scan_text(
    text: str,
    categories: Optional[List[CategorySpec]] = None,        # (선택) 적용할 카테고리 목록
    allowlist_patterns: Optional[List[re.Pattern]] = None,  # (선택) 허용목록(이건 PII 아님으로 건너뜀)
    min_confidence_override: Optional[float] = None,        # (선택) 신뢰도 하한(이 미만 규칙 무시)
) -> List[Detection]:
    """
    *text*에 Stage1 패턴 스캔을 실행.
    반환: 시작 위치순으로 정렬된 Detection 리스트. 겹치는 구간은 중복 제거(우선순위 높은 게 승).
    """
    if categories is None:                         # 카테고리 안 주면
        categories = ALL_CATEGORIES                #   기본 전체 카테고리
    if allowlist_patterns is None:                 # 허용목록 안 주면
        allowlist_patterns = []                    #   빈 목록(아무것도 안 건너뜀)

    raw_hits: List[Tuple[int, Detection]] = []     # (우선순위 인덱스, 탐지) — 겹침 정리 전 후보들

    for cat_idx, cat_spec in enumerate(categories):   # 카테고리마다(앞쪽일수록 우선순위 높음)
        for rule in cat_spec.rules:                    # 그 카테고리의 각 규칙(정규식)에 대해
            # 적용할 신뢰도 하한: override가 있으면 그걸, 없으면 카테고리 기본값.
            effective_min = min_confidence_override if min_confidence_override is not None \
                else cat_spec.min_confidence
            if rule.confidence < effective_min:        # 규칙 신뢰도가 하한 미만이면
                continue                               #   이 규칙은 건너뜀

            for m in rule.pattern.finditer(text):      # 텍스트에서 이 패턴의 모든 매칭을 순회
                start, end, matched = _resolve_capture_group(rule.pattern, m)  # 실제 값 스팬 추출

                # 빈 캡처는 건너뜀
                if not matched:
                    continue

                # 프로젝트 허용목록에 걸리면(=PII 아님) 건너뜀
                if any(ap.search(matched) for ap in allowlist_patterns):
                    continue

                # (선택) 체크섬/Luhn 검증기가 있으면 통과 못 한 값은 버림(오탐 방지)
                if rule.validator is not None and not rule.validator(matched):
                    continue

                det = Detection(                       # 표준 탐지 객체 생성
                    category=cat_spec.category,        #   카테고리 이름
                    category_class=cat_spec.category_class,  #   분류
                    action=cat_spec.action,           #   조치(마스킹/차단 등)
                    mask_style=cat_spec.mask_style,   #   마스킹 방식
                    start=start,                      #   시작 위치
                    end=end,                          #   끝 위치
                    original=matched,                 #   매칭된 원본 값
                    detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,  # 단계 표시(Stage1)
                    rule_id=rule.rule_id,             #   규칙 식별자
                    confidence=rule.confidence,       #   신뢰도
                )
                raw_hits.append((cat_idx, det))       # 후보로 추가(겹침은 아래에서 정리)

    # 겹침 해소를 위한 정렬: (시작 위치, 카테고리 우선순위, -스팬 길이).
    # 우선순위: 앞 위치 먼저 → 우선순위 높은 카테고리(작은 cat_idx) → 같은 카테고리면 더 긴 스팬.
    # 예: 같은 자리에서 시작하면 CARD가 ADDRESS를 이긴다.
    raw_hits.sort(key=lambda x: (x[1].start, x[0], -(x[1].end - x[1].start)))

    kept: List[Detection] = []                     # 최종 채택된 탐지들
    occupied: List[Tuple[int, int]] = []           # 이미 채택된 (시작, 끝) 구간들

    for _, det in raw_hits:                         # 우선순위 순으로 후보를 보며
        # 이미 채택된 어떤 구간과도 겹치는지 검사 (겹침 = 서로 떨어져 있지 않음)
        overlap = any(
            not (det.end <= s or det.start >= e)   # det이 [s,e)보다 완전히 앞/뒤가 아니면 겹침
            for s, e in occupied
        )
        if not overlap:                            # 안 겹치면
            kept.append(det)                       #   채택하고
            occupied.append((det.start, det.end))  #   점유 구간에 등록

    # 위치순으로 정렬해 반환
    kept.sort(key=lambda d: d.start)
    return kept
