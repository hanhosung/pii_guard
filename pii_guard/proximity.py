"""
pii_guard/proximity.py

[이 파일이 하는 일 — 한 줄 요약]
"근접 문맥(proximity)"을 이용한 양성 탐지(positive detection) 모듈이다.
모양이 애매한 PII(예: 계좌번호처럼 보이는 숫자)는 정규식(Stage1)이 일부러 안 잡는다.
무조건 잡으면 주문번호·송장번호까지 오탐하기 때문이다.
그래서 이 모듈은 "주변에 단서 단어(은행명·'계좌'·'사업자'·'비밀번호' 등)가 있을 때만"
그 애매한 값을 PII로 승격(promote)시킨다. → 오탐은 억제하면서 놓침(미검출)은 줄인다.

Positive proximity (context-gated) detection — PROXIMITY_DESIGN.md Phase 2.

Some real PII has an **ambiguous shape** that the Stage-1 regex rules deliberately
do NOT match, to avoid a false-positive explosion (e.g. a bare ``123-456-789012``
could be an order number, a courier id, …). The efficacy validation
(validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md) showed these slip through:

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

import re                                    # 정규식(패턴 매칭)용 표준 라이브러리
from dataclasses import dataclass            # 설정 객체(ProximityConfig)를 간단히 만들기 위한 데코레이터
from typing import Callable, List, NamedTuple, Optional, Tuple  # 타입 힌트(가독성/검증용)

from .categories import _kr_biz_checksum     # 사업자등록번호 체크섬 검증 함수(유효한 번호인지 확인)
from .models import Action, CategoryClass, Detection, DetectionStage, MaskStyle  # 탐지 결과 데이터 타입들

# ── 트리거(단서) 단어 사전 ────────────────────────────────────────────────────
# 계좌번호 근처에 이 은행명들이 있으면 "계좌 맥락"으로 본다.
_BANKS = (
    "국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "카카오",
    "토스뱅크", "토스", "케이뱅크", "SC", "씨티", "산업", "수협", "새마을",
    "신협", "우체국", "대구", "부산", "경남", "광주", "전북", "제주",
)
# 은행명 외에 "계좌/입금/이체…" 같은 동사·명사도 계좌 맥락 단서로 본다.
_ACCOUNT_VERBS = (
    "계좌", "입금", "이체", "송금", "환불", "예금주", "수령", "받을", "보낼", "보내",
)
_ACCOUNT_TRIGGERS = _BANKS + _ACCOUNT_VERBS   # 계좌 승격용 트리거 = 은행명 + 동사 (두 튜플을 합침)
_BIZ_TRIGGERS = ("사업자",)                    # 사업자번호 승격용 트리거(이 단어가 근처에 있어야 함)
_PASSWORD_KEYWORDS = ("비밀번호", "비번", "암호")  # 한글 비밀번호 라벨 키워드


@dataclass(frozen=True)                       # frozen=True → 생성 후 값 변경 불가(불변 설정 객체)
class ProximityConfig:
    """
    [정책으로 노출되는 proximity 설정 묶음]
    기본값은 위에서 정의한 내장 동작과 동일하다.
    정책 YAML의 ``proximity:`` 블록으로 어떤 필드든 덮어쓸 수 있고(핫리로드 지원).

    Policy-exposable proximity settings (PROXIMITY_DESIGN.md §7 / requirements R17).
    """
    enabled: bool = True                          # 양성 proximity 전체 on/off 스위치
    window_chars: int = 25                        # 값 기준 앞뒤로 몇 글자까지 트리거를 찾을지(검색 창 크기)
    account_triggers: Tuple[str, ...] = _ACCOUNT_TRIGGERS   # 계좌 트리거 목록(정책으로 교체 가능)
    biz_triggers: Tuple[str, ...] = _BIZ_TRIGGERS           # 사업자 트리거 목록
    password_keywords: Tuple[str, ...] = _PASSWORD_KEYWORDS # 비밀번호 라벨 키워드 목록
    # 아래 둘은 "음성 proximity"(NER 오탐 억제, stage2/ner_filters.py) 설정.
    # NER은 별도 서브프로세스에서 돌기 때문에 이 값들은 환경변수로 전달된다.
    ner_filter_enabled: bool = True               # NER 오탐 후필터 on/off
    ner_extra_stopwords: Tuple[str, ...] = ()     # NER 오탐으로 처리할 추가 단어(deny-list 보강)


DEFAULT_PROXIMITY_CONFIG = ProximityConfig()      # 아무 인자 없이 만든 기본 설정(= 내장 동작)


class ContextRule(NamedTuple):
    """[규칙 한 개] '어떤 모양의 값을, 어떤 트리거가 근처에 있을 때, 어떤 카테고리로' 승격할지 정의."""
    category: str                                 # 승격될 카테고리 이름(예: "KR_ACCOUNT")
    category_class: CategoryClass                 # 카테고리 분류(PII / KOREAN_PII / SECRET)
    action: Action                                # 처리 방식(마스킹 / 차단 등)
    mask_style: MaskStyle                         # 마스킹 스타일(토큰화 등)
    value_pattern: re.Pattern                     # 애매한 값 패턴. group(1)이 있으면 그게 실제 값, 없으면 전체 매치
    triggers: Tuple[str, ...]                     # 값 기준 ±window 안에 이 중 하나는 있어야 승격
    window: int                                   # 트리거 탐색 창(글자 수)
    confidence: float                             # 이 규칙으로 잡았을 때 부여할 신뢰도
    rule_id: str                                  # 감사용 규칙 식별자(어떤 규칙/트리거로 잡혔는지 기록)
    validator: Optional[Callable[[str], bool]] = None  # (선택) 정규화된 값에 추가 체크섬 검증


def build_rules(config: ProximityConfig) -> Tuple[ContextRule, ...]:
    """[설정 → 규칙 목록] 정책 설정(ProximityConfig)을 받아 실제 ContextRule 목록을 만든다."""
    acct = tuple(config.account_triggers)         # 계좌 트리거를 튜플로 고정(설정에서 가져옴)
    w = config.window_chars                       # 검색 창 크기(설정값)
    # 비밀번호 키워드들을 정규식 OR(|)로 합친다. 비어 있으면 안전하게 "비밀번호" 기본 사용.
    pw_alt = "|".join(re.escape(k) for k in config.password_keywords) or "비밀번호"
    # 비밀번호 패턴: (비밀번호|비번|암호) 뒤에 선택적 :/= 와 공백, 그 다음 값(4~40자, 구분문자 전까지)을 캡처.
    pw_pattern = re.compile(rf"(?:{pw_alt})\s*[:=]?\s*([^\s,，.。!?'\"]{{4,40}})")
    return (
        # 규칙1) KR_ACCOUNT — 비표준 3-3-6 포맷(예: 123-456-789012). 앞뒤가 숫자가 아닐 때만 매치.
        ContextRule(
            "KR_ACCOUNT", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
            re.compile(r"(?<!\d)(\d{3}-\d{3}-\d{6})(?!\d)"),   # (?<!\d)…(?!\d): 더 긴 숫자열 안쪽 매치 방지
            acct, w, 0.70, "prox_kr_acct_336",                 # 계좌 트리거, 창 w, 신뢰도 0.70, 규칙ID
        ),
        # 규칙2) KR_ACCOUNT — 카카오/토스뱅크식 4-2-7 포맷(예: 3333-01-1234567).
        ContextRule(
            "KR_ACCOUNT", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
            re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{7})(?!\d)"),
            acct, w, 0.70, "prox_kr_acct_427",
        ),
        # 규칙3) BIZ_NO — 하이픈 없는 10자리 숫자. "사업자"가 근처에 있고 + 체크섬까지 통과해야 승격(오탐 억제).
        ContextRule(
            "BIZ_NO", CategoryClass.KOREAN_PII, Action.TOKENIZE_ROUNDTRIP, MaskStyle.TOKENIZE,
            re.compile(r"(?<!\d)(\d{10})(?!\d)"),
            tuple(config.biz_triggers), min(w, 20), 0.85, "prox_biz_bare10",  # 창은 최대 20으로 제한
            validator=_kr_biz_checksum,                                       # 사업자번호 체크섬 검증 추가
        ),
        # 규칙4) PASSWORD — 한글 라벨(비밀번호/비번/암호) : 값. 트리거가 패턴 안에 이미 포함되므로 빈 트리거.
        ContextRule(
            "PASSWORD", CategoryClass.SECRET, Action.BLOCK, MaskStyle.TOKENIZE,  # 비번은 마스킹이 아니라 차단(block)
            pw_pattern, (), 0, 0.85, "prox_password_kr",                          # triggers=() → 별도 근접 검사 없음
        ),
    )


# 기본 설정용 규칙 목록을 미리 한 번 만들어 둔다(매 호출마다 다시 만들지 않으려고 캐시).
CONTEXT_RULES: Tuple[ContextRule, ...] = build_rules(DEFAULT_PROXIMITY_CONFIG)


def _norm(v: str) -> str:
    """값에서 하이픈/공백을 제거해 정규화(체크섬 검증 등에 쓰려고 숫자만 남김)."""
    return v.replace("-", "").replace(" ", "")


def scan(text: str, config: Optional[ProximityConfig] = None) -> List[Detection]:
    """[메인 함수] *text*에서 근접 문맥으로 승격되는 탐지 목록을 반환(없으면 빈 리스트)."""
    if not text:                                  # 빈 문자열이면 검사할 것이 없으니 즉시 종료
        return []
    if config is not None and not config.enabled: # 설정이 주어졌고 proximity가 꺼져 있으면 아무것도 안 함
        return []
    # 기본 설정이면 미리 만든 CONTEXT_RULES 재사용, 커스텀 설정이면 그때 규칙을 새로 만든다.
    rules = CONTEXT_RULES if (config is None or config is DEFAULT_PROXIMITY_CONFIG) \
        else build_rules(config)
    out: List[Detection] = []                     # 결과(탐지)들을 모을 리스트
    for rule in rules:                            # 규칙(계좌/사업자/비번)을 하나씩
        for m in rule.value_pattern.finditer(text):  # 텍스트에서 그 규칙 패턴에 맞는 모든 위치를 찾음
            if m.groups():                        # 캡처 그룹이 있으면(예: 비밀번호 값 부분)
                start, end, value = m.start(1), m.end(1), m.group(1)  # 그 그룹의 시작/끝/값을 사용
            else:                                 # 캡처 그룹이 없으면
                start, end, value = m.start(), m.end(), m.group()     # 전체 매치를 값으로 사용

            # ── 근접 게이트(트리거 검사) ──
            trig_hit = None                       # 어떤 트리거에 걸렸는지 기록(없으면 None)
            if rule.triggers:                     # 이 규칙이 트리거를 요구하면(계좌/사업자)
                # 값의 앞뒤 ±window 글자만큼 잘라낸 '창' 안에서 트리거를 찾는다.
                window = text[max(0, start - rule.window): end + rule.window]
                trig_hit = next((t for t in rule.triggers if t in window), None)  # 창 안에 있는 첫 트리거
                if trig_hit is None:              # 트리거가 하나도 없으면
                    continue                      # 이 매치는 승격하지 않고 건너뜀(오탐 방지)

            # ── (선택) 체크섬 검증 ── 사업자번호 등은 값이 산술적으로 유효해야만 통과
            if rule.validator is not None and not rule.validator(_norm(value)):
                continue                          # 체크섬 실패 → 가짜 번호이므로 건너뜀

            # 규칙ID에 어떤 트리거로 잡혔는지 덧붙여 감사 추적성을 높임(예: "prox_kr_acct_336+국민").
            rid = rule.rule_id + (f"+{trig_hit}" if trig_hit else "")
            out.append(Detection(                 # 최종 탐지 객체를 만들어 결과에 추가
                category=rule.category,           # 카테고리(KR_ACCOUNT 등)
                category_class=rule.category_class,  # 분류(KOREAN_PII/SECRET 등)
                action=rule.action,               # 처리(마스킹/차단)
                mask_style=rule.mask_style,       # 마스킹 스타일
                start=start,                      # 원문에서의 시작 위치
                end=end,                          # 원문에서의 끝 위치
                original=value,                   # 실제 탐지된 값
                detection_stage=DetectionStage.STAGE1_PROXIMITY,  # 탐지 단계 표시(= Stage1.5 proximity)
                rule_id=rid,                      # 감사용 규칙ID(+트리거)
                confidence=rule.confidence,       # 신뢰도
            ))
    return out                                    # 모은 탐지들을 반환


def merge(base: List[Detection], extra: List[Detection]) -> List[Detection]:
    """
    [병합] 기존 탐지(base)에 proximity 탐지(extra)를 합친다. 겹치는 스팬 처리 규칙:

    Overlap policy (per detection d in extra):
      - 기존 탐지가 d를 '포함'하면 → 기존 것을 두고 d는 버림;
      - d가 기존 탐지를 '완전히 포함'하면 → d가 더 완전한 해석이므로 그 하위 탐지들을 빼고 d를 넣음
        (예: 계좌 ``3333-02-7654321``가, Stage1이 그 안에서 잘못 잡은 전화 ``02-7654321``를 흡수);
      - 그 외 부분 겹침 → 안전하게 d를 버림.
    """
    merged = list(base)                           # 기존 탐지를 복사해 시작(원본 리스트 보호)
    for d in extra:                               # 추가하려는 proximity 탐지를 하나씩
        skip = False                              # d를 버릴지 여부
        subsumed: List[Detection] = []            # d가 흡수(대체)할 기존 탐지들
        for b in merged:                          # 이미 있는 탐지들과 하나씩 겹침 비교
            if d.end <= b.start or d.start >= b.end:
                continue                          # 서로 안 겹치면(완전 분리) 통과
            if b.start <= d.start and b.end >= d.end and (b.end - b.start) >= (d.end - d.start):
                skip = True                       # 기존 b가 d를 포함(더 큼) → b를 두고 d는 버림
                break
            if d.start <= b.start and d.end >= b.end:
                subsumed.append(b)                # d가 b를 완전히 포함 → 나중에 b를 빼고 d로 대체
            else:
                skip = True                       # 어정쩡하게 부분만 겹침 → 보수적으로 d 버림
                break
        if skip:                                  # 버리기로 했으면
            continue                              # 다음 d로
        for b in subsumed:                        # d가 흡수한 하위 탐지들을
            merged.remove(b)                      # 결과에서 제거
        merged.append(d)                          # d를 결과에 추가
    merged.sort(key=lambda x: x.start)            # 위치 순으로 정렬해 반환
    return merged
