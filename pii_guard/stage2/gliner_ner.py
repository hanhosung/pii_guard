"""
pii_guard/stage2/gliner_ner.py

GLiNER 기반 한국어 NER 엔진 — Stage2 **기본 백엔드** (요구사항 R18 / DESIGN ADR-10·ADR-11).

[이 파일이 하는 일 — 한 줄 요약]
GLiNER(제로샷 NER) 한국어 특화 모델로, 정규식(Stage1)이 못 잡는 비정형 한국어 PII —
사람 이름(PERSON), 주소/지역(ADDRESS), 기관/조직(ORGANIZATION) — 을 탐지해
PII-Guard의 Detection 객체로 변환한다. 출력 형식·카테고리는 spaCy 백엔드
(`korean_ner.py`)와 완전히 동일하므로, 후단(정책·마스킹·proximity 후필터·열화)은
어느 백엔드를 쓰든 똑같이 동작한다.

GLiNER란?
---------
라벨(찾고 싶은 엔티티 종류)을 '프롬프트로' 넘기면, 학습 때 못 본 종류라도 추출해 주는
제로샷 NER 모델이다. 여기서는 한국어 라벨(`사람`·`주소`·`조직`)을 넘겨 추출 결과를
PII-Guard 카테고리로 되매핑한다. 트랜스포머 기반이라 spaCy보다 무겁지만(재현율↑),
별도 워커 프로세스 + 옵션 설치(`[ner-gliner]`)로 격리한다.

설치/모델
---------
- 의존: `pip install pii-guard[ner-gliner]` (gliner + torch).
- 모델: 한국어 특화 GLiNER(기본 `taeminlee/gliner_ko`). `PIIGUARD_GLINER_MODEL` 환경변수로 교체 가능.
- 이 모듈은 Stage2 서브프로세스 워커 안에서만 import되도록 설계됐다(무거운 모델을 부모에
  올리지 않기 위함). gliner/torch import는 첫 detect 호출 때까지 지연된다.

Usage::

    engine = GLiNERNEREngine()            # lazy — 아직 모델 로드 안 함
    detections = engine.detect("김철수 씨가 서울특별시 강남구에 산다.")
    for det in detections:
        print(det.category, det.confidence, det.original)
"""
from __future__ import annotations

import logging                               # 진단 로그(모델 로딩/추론 실패)
import os                                      # 환경변수(모델 오버라이드)
from typing import List, Optional             # 타입 힌트

from ..models import (                        # PII-Guard 공용 데이터 타입(spaCy 백엔드와 동일하게 사용)
    Action,
    CategoryClass,
    Detection,
    DetectionStage,
    MaskStyle,
)
from .korean_ner import _strip_ko_particle, MIN_CONFIDENCE  # 조사 제거·신뢰도 하한 재사용(중복 방지)
from .ner_filters import is_ner_false_positive             # NER 오탐 억제(음성 proximity) 재사용

logger = logging.getLogger(__name__)          # 이 모듈 전용 로거

#: 모델 오버라이드용 환경변수 이름
_GLINER_MODEL_ENV_VAR = "PIIGUARD_GLINER_MODEL"

#: 기본 한국어 특화 GLiNER 모델
_DEFAULT_GLINER_MODEL = "taeminlee/gliner_ko"

# GLiNER에 넘길 '한국어 라벨' → PII-Guard 카테고리 매핑.
# GLiNER는 여기 키들을 라벨로 받아 추출하고, 반환 결과의 label은 넘긴 키와 같다.
# 같은 카테고리에 여러 표현(동의어)을 두어 재현율을 높인다.
_LABEL_TO_CATEGORY: dict = {
    "사람":   "PERSON",        # 인물 이름
    "이름":   "PERSON",        #   (동의어)
    "인물":   "PERSON",        #   (동의어)
    "주소":   "ADDRESS",       # 주소/소재지
    "장소":   "ADDRESS",       #   (지역/위치 동의어)
    "지역":   "ADDRESS",       #   (동의어)
    "조직":   "ORGANIZATION",  # 기관/단체
    "기관":   "ORGANIZATION",  #   (동의어)
    "회사":   "ORGANIZATION",  #   (동의어)
}

#: GLiNER 추론에 넘길 라벨 목록(위 매핑의 키들)
_GLINER_LABELS: List[str] = list(_LABEL_TO_CATEGORY.keys())

# 카테고리 → CategoryClass / Action (spaCy 백엔드와 동일하게 KOREAN_PII · TOKENIZE_ROUNDTRIP)
_CATEGORY_CLASS = CategoryClass.KOREAN_PII
_CATEGORY_ACTION = Action.TOKENIZE_ROUNDTRIP


def resolve_gliner_model() -> str:
    """
    어떤 GLiNER 모델을 로드할지 결정.
      1. `PIIGUARD_GLINER_MODEL` 환경변수가 있으면 그대로(폴백 없음 → 오타면 명확한 로드 오류).
      2. 없으면 한국어 특화 기본 모델(`taeminlee/gliner_ko`).
    """
    override = os.environ.get(_GLINER_MODEL_ENV_VAR)   # 환경변수 우선
    if override:
        return override.strip()
    return _DEFAULT_GLINER_MODEL                        # 기본 모델


class GLiNERNEREngine:
    """
    GLiNER 기반 한국어 NER 엔진(Stage2 기본 백엔드).

    지연 초기화(lazy): GLiNER 모델은 detect() '첫 호출'에 로드되어, 같은 프로세스 내
    이후 호출에서 재사용된다(서브프로세스 수명 동안 싱글턴).

    Parameters
    ----------
    min_confidence:
        이 점수 미만 탐지는 버림. 기본 MIN_CONFIDENCE(0.50).
    strip_particles:
        True(기본)면 엔티티 끝의 한국어 조사를 떼어내 "홍길동은"→"홍길동".
    model_name:
        (선택) 모델 고정. None이면 로드 시 resolve_gliner_model()로 결정.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,   # 신뢰도 하한(spaCy 백엔드와 동일 기본값)
        strip_particles: bool = True,             # 조사 제거 on(기본)
        model_name: Optional[str] = None,         # (선택) 모델 고정
    ) -> None:
        self._min_confidence = min_confidence
        self._strip_particles = strip_particles
        self._model_name = model_name
        self._model: Optional[object] = None      # 지연 초기화 — 첫 사용 때 채워짐

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> List[Detection]:
        """
        *text* 안의 한국어 PII 엔티티를 GLiNER로 탐지.
        반환: 시작 위치순으로 정렬된 Detection 리스트(빈 리스트일 수 있음).
        출력 형식은 spaCy 백엔드(KoreanNEREngine.detect)와 동일하다.
        """
        if not text or not text.strip():          # 빈/공백 입력이면 바로 빈 결과
            return []

        model = self._get_model()                 # (필요 시 로드해) 모델 확보
        try:
            # GLiNER 추론: 라벨 프롬프트로 엔티티 추출.
            # 반환은 dict 리스트: {"start","end","text","label","score"}.
            raw = model.predict_entities(
                text,
                _GLINER_LABELS,
                threshold=self._min_confidence,   # 모델 단계에서 1차로 낮은 점수 컷
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GLiNER prediction failed: %s", exc)  # 추론 실패는 경고만
            return []                                            # 빈 결과로 안전 폴백

        detections: List[Detection] = []          # 변환 결과
        for ent in raw:                           # 추출된 각 엔티티에 대해
            score = float(ent.get("score", 0.0))
            if score < self._min_confidence:      # 신뢰도 미달이면(이중 안전망) 건너뜀
                continue
            label = ent.get("label", "")
            category = _LABEL_TO_CATEGORY.get(label)  # 라벨 → PII-Guard 카테고리
            if category is None:                  # 모르는 라벨이면 건너뜀
                continue

            start = int(ent["start"])             # 스팬 시작
            end = int(ent["end"])                 # 스팬 끝
            original = ent.get("text") or text[start:end]  # 원본 값

            # (옵션) 끝에 붙은 한국어 조사 제거 + 끝 위치 보정(spaCy 백엔드와 동일 로직)
            clean_text = original
            adjusted_end = end
            if self._strip_particles:
                clean_text = _strip_ko_particle(original)
                if len(clean_text) < len(original):
                    adjusted_end = start + len(clean_text)

            det = Detection(                      # spaCy 백엔드와 동일한 표준 탐지 객체
                category=category,
                category_class=_CATEGORY_CLASS,
                action=_CATEGORY_ACTION,
                mask_style=MaskStyle.TOKENIZE,
                start=start,
                end=adjusted_end,
                original=clean_text,
                detection_stage=DetectionStage.STAGE2_NER,
                rule_id=f"ner_gliner_{category.lower()}",  # 규칙 식별자(백엔드 구분용 접두)
                confidence=score,
            )
            detections.append(det)

        detections.sort(key=lambda d: d.start)    # 시작 위치순 정렬
        # 음성 proximity 후필터: 코드토큰·약어·blob·일반명사 오탐 제거(spaCy와 동일하게 적용).
        return [d for d in detections
                if not is_ner_false_positive(d.category, d.original)]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_model(self):
        """
        GLiNER 모델을 반환(첫 호출에 지연 로드).
        gliner/torch import를 여기까지 미뤄, 메인 프로세스에 무거운 의존이 올라가지 않게 한다.
        의존 미설치 시 설치 안내와 함께 RuntimeError를 던진다.
        """
        if self._model is not None:               # 이미 로드돼 있으면 재사용
            return self._model

        model_name = self._model_name or resolve_gliner_model()  # 모델명 결정
        try:
            from gliner import GLiNER             # 지연 임포트(여기서 torch까지 로드)
        except ImportError as exc:
            raise RuntimeError(
                "gliner is not installed. "
                "Run: pip install 'pii-guard[ner-gliner]'  (installs gliner + torch)"
            ) from exc

        logger.info("Loading GLiNER model for NER: %s", model_name)
        try:
            self._model = GLiNER.from_pretrained(model_name)  # 모델 가중치 로드(무거움)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load GLiNER model '{model_name}'. "
                f"Check the model name or set {_GLINER_MODEL_ENV_VAR}."
            ) from exc
        return self._model
