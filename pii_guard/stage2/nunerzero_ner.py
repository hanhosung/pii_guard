"""
pii_guard/stage2/nunerzero_ner.py

NuNER Zero 기반 한국어 NER 엔진 — Stage2 **평가 후보 백엔드** (요구사항 R21 / DESIGN ADR-14).

[이 파일이 하는 일 — 한 줄 요약]
GLiNER와 **동급 성능을 내는 동계열 제로샷 NER**인 NuNER Zero(`numind/NuNER_Zero`)로
정규식(Stage1)이 못 잡는 비정형 한국어 PII — 사람 이름(PERSON), 주소/지역(ADDRESS),
기관/조직(ORGANIZATION) — 을 탐지해 PII-Guard Detection으로 변환한다. 출력 형식·카테고리·
후처리는 GLiNER 백엔드(`gliner_ner.py`)와 **완전히 동일**하므로, 후단(정책·마스킹·proximity
후필터·열화)은 어느 백엔드를 쓰든 똑같이 동작한다.

왜 후보인가 (R21 / ADR-14)
--------------------------
GLiNER 대안 조사(보고서 §2.3 + 외부 벤치) 결과, GLiNER와 유사 성능을 내면서 PII-Guard 제약
(완전 로컬·결정적·인젝션 불가·한국어·**상업 라이선스**·임의 라벨)을 동시에 만족하는 후보는
사실상 NuNER Zero가 유일했다. GLiNER와 동계열 제로샷이나 **토큰 분류(token classification)**
구조라, 임계값(ADR-12)으로 풀리지 않는 **ORG 정밀도(0.774)**·긴 복합 엔티티(주소·조직 경계)에서
개선 여지가 있다는 가설이다(외부 보고: GLiNER Large v2.1 대비 약 +3%).

⚠️ **무조건 채택이 아니다.** 이 어댑터는 ADR-14의 **배선(이 파일) → 벤치 비교 → 채택 게이트**
절차 중 1단계다. 코퍼스+외부 6리포트에서 GLiNER·spaCy와 동일 하니스로 비교해 (a) recall 무회귀
(b) ORG 정밀도/종합 F1 개선 (c) 런타임 예산 충족을 **모두** 만족할 때만 정식 옵션으로 승격한다.
미통과 시 GLiNER 기본을 유지한다.

설치/모델
---------
- 의존: NuNER Zero는 **GLiNER 라이브러리로 로드**되므로 `[ner-gliner]` 의존을 그대로 공유한다.
  (`pip install 'pii-guard[ner-gliner]'` — gliner + torch. 별도 패키지 불필요.)
- 모델: 기본 `numind/NuNER_Zero`(**MIT → 상업 사용 가능**). `PIIGUARD_NUNERZERO_MODEL`로 교체 가능.
- 이 모듈은 Stage2 서브프로세스 워커 안에서만 import되도록 설계됐다(무거운 모델을 부모에
  올리지 않기 위함). gliner/torch import는 첫 detect 호출 때까지 지연된다(부모 클래스가 처리).

구현 메모 (GLiNER와의 차이)
---------------------------
- 모델 로딩 API(`GLiNER.from_pretrained`)·추론 API(`predict_entities`)·결과 dict 구조
  (`{"start","end","text","label","score"}`)·후처리(조사 제거·신뢰도 컷·음성 proximity 필터)는
  GLiNER와 동일하다. 따라서 추론/변환 로직은 부모 `GLiNERNEREngine`을 **그대로 재사용**하고,
  여기서는 **모델명·env 변수·rule_id 접두**만 갈아끼운다(중복 제거).
- NuNER Zero 모델 카드는 라벨 소문자 사용·토큰 인접 엔티티 병합(`merge_entities`)을 권장한다.
  한국어 라벨(`사람`·`주소`·`조직`)은 대소문자 영향이 없고, 병합은 벤치에서 경계 품질을 보고
  필요 시 후속 튜닝한다(현 골격은 GLiNER와 동일 경로로 1차 비교가 목적).

Usage::

    engine = NuNERZeroNEREngine()         # lazy — 아직 모델 로드 안 함
    detections = engine.detect("김철수 씨가 서울특별시 강남구에 산다.")
    for det in detections:
        print(det.category, det.confidence, det.original)
"""
from __future__ import annotations

import os                                        # 환경변수(모델 오버라이드)
from typing import Optional                      # 타입 힌트

from .gliner_ner import GLiNERNEREngine          # 추론·후처리 로직 재사용(동계열, gliner 라이브러리)
from .korean_ner import MIN_CONFIDENCE           # 신뢰도 하한 기본값(전 백엔드 공통)

#: 모델 오버라이드용 환경변수 이름
_NUNERZERO_MODEL_ENV_VAR = "PIIGUARD_NUNERZERO_MODEL"

#: 기본 NuNER Zero 모델 — MIT(상업 사용 가능). GLiNER 라이브러리로 로드된다.
_DEFAULT_NUNERZERO_MODEL = "numind/NuNER_Zero"

#: 인접 조각 병합 시 허용하는 두 스팬 사이 최대 간격(문자). 0=맞붙음, 1=공백 한 칸.
_MERGE_MAX_GAP = 1


def _merge_adjacent_entities(entities: list, text: str) -> list:
    """
    NuNER Zero(토큰 분류) 출력의 **인접 동일 라벨 조각을 하나로 병합**한다.

    NuNER Zero는 span 분류인 GLiNER와 달리 토큰 단위로 라벨링하므로, 한 엔티티가
    여러 인접 조각으로 쪼개져 나올 수 있다(예: "서울특별시"+"강남구"). 모델 카드 권장대로
    **라벨이 같고 위치가 맞붙은(또는 공백 한 칸 떨어진)** 조각을 이어붙여 경계를 복원한다.

    Parameters
    ----------
    entities:
        ``model.predict_entities`` 원시 결과(dict 리스트, 각 {start,end,text,label,score}).
    text:
        원본 텍스트(병합 스팬의 text를 원문에서 다시 잘라 정확도 보장).

    Returns
    -------
    병합된 dict 리스트(시작 위치순). 입력이 비면 빈 리스트.
    """
    if not entities:
        return []

    # 위치순 정렬(predict_entities가 정렬을 보장하지 않을 수 있으므로 방어적으로).
    ordered = sorted(entities, key=lambda e: (int(e["start"]), int(e["end"])))

    merged: list = []
    current = dict(ordered[0])                        # 복사(원본 변형 방지)
    for nxt in ordered[1:]:
        same_label = nxt.get("label") == current.get("label")
        contiguous = int(nxt["start"]) <= int(current["end"]) + _MERGE_MAX_GAP
        if same_label and contiguous:
            # 병합: 끝 위치 확장 + 원문 재슬라이스 + 점수는 보수적으로 최댓값 유지.
            new_end = max(int(current["end"]), int(nxt["end"]))
            current["end"] = new_end
            current["text"] = text[int(current["start"]):new_end].strip()
            current["score"] = max(
                float(current.get("score", 0.0)),
                float(nxt.get("score", 0.0)),
            )
        else:
            merged.append(current)
            current = dict(nxt)
    merged.append(current)
    return merged


def resolve_nunerzero_model() -> str:
    """
    어떤 NuNER Zero 모델을 로드할지 결정.
      1. `PIIGUARD_NUNERZERO_MODEL` 환경변수가 있으면 그대로(폴백 없음 → 오타면 명확한 로드 오류).
      2. 없으면 MIT 기본 모델(`numind/NuNER_Zero`).
    """
    override = os.environ.get(_NUNERZERO_MODEL_ENV_VAR)   # 환경변수 우선
    if override:
        return override.strip()
    return _DEFAULT_NUNERZERO_MODEL                        # 기본 모델


class NuNERZeroNEREngine(GLiNERNEREngine):
    """
    NuNER Zero 기반 한국어 NER 엔진(Stage2 평가 후보 백엔드, R21·ADR-14).

    GLiNER 라이브러리로 로드되는 동계열 제로샷 모델이라 추론·후처리는 부모
    :class:`GLiNERNEREngine`을 그대로 상속한다. 모델명/식별자만 다음 세 지점으로 분리한다:
      * ``_RULE_PREFIX``        → ``ner_nunerzero`` (Detection.rule_id 접두로 백엔드 식별)
      * ``_MODEL_ENV_VAR``      → ``PIIGUARD_NUNERZERO_MODEL`` (오류 메시지에 사용)
      * ``_resolve_model_name`` → ``numind/NuNER_Zero`` 기본(env 오버라이드)

    Parameters
    ----------
    min_confidence:
        이 점수 미만 탐지는 버림. 기본 MIN_CONFIDENCE(0.50). (벤치 임계값 스윕은
        GLiNER와 동일 그리드로 공정 비교 — ADR-14.)
    strip_particles:
        True(기본)면 엔티티 끝의 한국어 조사를 떼어내 "홍길동은"→"홍길동".
    model_name:
        (선택) 모델 고정. None이면 로드 시 resolve_nunerzero_model()로 결정.
    """

    #: Detection.rule_id 접두 — 백엔드 구분("ner_nunerzero_person" 등)
    _RULE_PREFIX = "ner_nunerzero"
    #: 모델 오버라이드 환경변수 이름(오류 메시지에 사용)
    _MODEL_ENV_VAR = _NUNERZERO_MODEL_ENV_VAR

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,   # 신뢰도 하한(전 백엔드 동일 기본값)
        strip_particles: bool = True,             # 조사 제거 on(기본)
        model_name: Optional[str] = None,         # (선택) 모델 고정
    ) -> None:
        super().__init__(
            min_confidence=min_confidence,
            strip_particles=strip_particles,
            model_name=model_name,
        )

    def _resolve_model_name(self) -> str:
        """로드할 모델명 결정: 고정 model_name > resolve_nunerzero_model()(env > MIT 기본)."""
        return self._model_name or resolve_nunerzero_model()

    def _predict_entities(self, model, text: str) -> list:
        """
        추출 후 **인접 조각 병합**(토큰 분류 특성 보정)을 적용한다.
        부모(GLiNER)는 span 분류라 병합이 불필요하지만, NuNER Zero는 토큰 분류라
        쪼개진 인접 스팬을 이어붙여야 경계(주소·조직명)가 정확해진다(모델 카드 권장).
        """
        raw = super()._predict_entities(model, text)   # gliner 라이브러리 predict_entities
        return _merge_adjacent_entities(raw, text)
