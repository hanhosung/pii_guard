"""
pii_guard/stage2/korean_ner.py

[이 파일이 하는 일 — 한 줄 요약]
Stage2(NER) 단계의 핵심. Microsoft Presidio가 한국어 spaCy 모델을 지휘하도록 엮어서,
정규식(Stage1)으로는 못 잡는 '비정형 한국어 PII' — 사람 이름(PERSON), 주소/지역(ADDRESS),
기관/조직(ORGANIZATION) — 을 탐지하고, 그 결과를 PII-Guard의 Detection 객체로 변환한다.

Korean NER engine using Microsoft Presidio orchestrating a spaCy Korean model
to detect unstructured Korean PII — person names, addresses (locations), and
organizations — that Stage-1 regex cannot catch.

Model selection (see :func:`resolve_ko_spacy_model`): ``ko_core_news_lg`` is
preferred when installed (materially better PERSON/ADDRESS/ORGANIZATION recall),
with ``ko_core_news_sm`` as a lightweight fallback. Override with the
``PIIGUARD_KO_SPACY_MODEL`` environment variable.

Architecture
------------
``KoreanNEREngine`` wraps Presidio's ``AnalyzerEngine`` configured with the
resolved Korean spaCy model.  The spaCy Korean model uses its own entity
label scheme (PS=Person, LC=Location, OG=Organization); we register a
``NerModelConfiguration`` that maps these to Presidio's canonical entity types
(PERSON, LOCATION, ORGANIZATION).  A custom ``SpacyRecognizer`` restricted to
``supported_language='ko'`` is the only recognizer in the registry, so no
English-default recognizers interfere.

Detected entities are converted to PII-Guard ``Detection`` objects:
(탐지된 엔티티는 아래처럼 PII-Guard 카테고리로 변환된다.)

  Presidio entity    →  PII-Guard category  class           action
  -----------------     ------------------  ----------      -----------------
  PERSON             →  PERSON              KOREAN_PII      TOKENIZE_ROUNDTRIP
  LOCATION           →  ADDRESS             KOREAN_PII      TOKENIZE_ROUNDTRIP
  ORGANIZATION       →  ORGANIZATION        KOREAN_PII      TOKENIZE_ROUNDTRIP

Confidence scores come from the spaCy NER scorer (forwarded by Presidio).
The default NER score is 0.70 when the model does not emit per-span probabilities;
entities above MIN_CONFIDENCE are included.

This module is designed to be imported inside the Stage-2 subprocess worker
(``default_ner_worker_loop``), so models are loaded lazily once per subprocess
and amortised across requests.  It must never be imported in the main process
paths to avoid polluting the parent process memory.
(이 모듈은 Stage2 서브프로세스 워커 안에서만 import되도록 설계됐다. 무거운 spaCy 모델을
부모 프로세스 메모리에 올리지 않기 위해, 메인 경로에서는 절대 import하면 안 된다.)

Usage::

    engine = KoreanNEREngine()           # lazy — no model loaded yet
    detections = engine.detect("김철수 씨께서 서울특별시에 방문했습니다.")
    for det in detections:
        print(det.category, det.confidence, det.original)

Notes on model limitations
---------------------------
- ``ko_core_news_sm`` is a small model (~14 MB).  Precision and recall are
  moderate; it works best on well-formed Korean text.
- Particles can be attached to entity spans (e.g. "홍길동은" instead of
  "홍길동").  We post-process to strip common Korean postpositional particles
  from entity text so placeholders match cleanly.
  (spaCy가 "홍길동은"처럼 조사를 붙여 잡을 수 있어, 후처리로 조사를 떼어낸다.)
- When Presidio does not receive a per-token score from spaCy, it uses the
  ``default_score`` from NerModelConfiguration (0.70).  This is the minimum
  non-zero confidence value returned by this engine.
"""
from __future__ import annotations

import logging                                   # 진단 로그(모델 로딩/분석 실패 등) 출력용
import os                                          # 환경변수(모델 오버라이드) 읽기용
from typing import List, Optional                  # 타입 힌트

from ..models import (                             # PII-Guard 공용 데이터 타입들
    Action,                                        #   탐지 후 어떤 조치(토큰화 등)를 할지
    CategoryClass,                                 #   카테고리 분류(KOREAN_PII 등)
    Detection,                                     #   탐지 결과 1건을 표현하는 타입
    DetectionStage,                                #   어느 단계에서 잡았는지(STAGE2_NER)
    MaskStyle,                                     #   마스킹 방식(TOKENIZE 등)
)
from .ner_filters import is_ner_false_positive    # NER 오탐 억제(음성 proximity) 판정 함수

logger = logging.getLogger(__name__)              # 이 모듈 전용 로거

# 최소 신뢰도 임계값 — 이보다 낮은 탐지는 버린다.
MIN_CONFIDENCE: float = 0.50

# 한국어 spaCy 모델 — 선호 순서. lg가 sm보다 PERSON 재현율이 확연히 좋다(대신 크기/로딩 ~10배).
# 그래서 설치돼 있으면 lg 우선, sm은 경량 폴백. PIIGUARD_KO_SPACY_MODEL 환경변수로 명시 오버라이드 가능.
_PREFERRED_KO_MODELS = ("ko_core_news_lg", "ko_core_news_sm")
_KO_MODEL_ENV_VAR = "PIIGUARD_KO_SPACY_MODEL"      # 모델 오버라이드용 환경변수 이름


def resolve_ko_spacy_model() -> str:
    """
    어떤 한국어 spaCy 모델을 로드할지 결정.
    결정 순서:
      1. PIIGUARD_KO_SPACY_MODEL 환경변수가 있으면 그대로 사용(폴백 없음 → 오타면 명확한 오류로 드러남).
      2. _PREFERRED_KO_MODELS 중 '설치된 첫 번째' 모델(lg → sm 순).
      3. 최후의 보루로 ko_core_news_sm(없으면 로드 시점에 설치 안내와 함께 명확한 오류).
    """
    override = os.environ.get(_KO_MODEL_ENV_VAR)   # 1) 환경변수 우선
    if override:
        return override.strip()                    # 공백 제거 후 그대로 사용

    try:
        import spacy.util                          # 2) 설치 여부 확인용 유틸

        for name in _PREFERRED_KO_MODELS:          # lg → sm 순으로
            if spacy.util.is_package(name):        # 설치돼 있으면
                return name                        # 그 모델 사용
    except Exception:  # noqa: BLE001 — spaCy 자체가 없으면 아래 기본값으로 폴스루
        pass

    return "ko_core_news_sm"                       # 3) 최후 기본값

# Presidio 엔티티 타입 → PII-Guard 카테고리 이름 매핑.
_PRESIDIO_TO_CATEGORY: dict = {
    "PERSON":       "PERSON",                      # 사람 이름 → PERSON
    "LOCATION":     "ADDRESS",                     # 장소/지역 → ADDRESS(주소로 취급)
    "ORGANIZATION": "ORGANIZATION",               # 기관/조직 → ORGANIZATION
}

# Presidio 엔티티 타입 → PII-Guard CategoryClass 매핑(전부 KOREAN_PII로 분류).
_ENTITY_CLASS: dict = {
    "PERSON":       CategoryClass.KOREAN_PII,
    "LOCATION":     CategoryClass.KOREAN_PII,
    "ORGANIZATION": CategoryClass.KOREAN_PII,
}

# 각 엔티티에 적용할 조치 — 모두 TOKENIZE_ROUNDTRIP
# (플레이스홀더로 마스킹하되, 응답이 돌아오면 원래 값으로 복원해 에이전트는 실제 값을 보게 함).
_ENTITY_ACTION: dict = {
    "PERSON":       Action.TOKENIZE_ROUNDTRIP,
    "LOCATION":     Action.TOKENIZE_ROUNDTRIP,
    "ORGANIZATION": Action.TOKENIZE_ROUNDTRIP,
}

# spaCy 토크나이저가 엔티티 끝에 자주 붙여 잡는 한국어 조사(助詞)들.
# 탐지 텍스트에서 이 접미사를 떼어내야 플레이스홀더가 깔끔하게 맞는다.
_KO_PARTICLES = (
    "이", "가", "을", "를", "은", "는", "의", "에", "에서", "으로", "로",
    "와", "과", "이나", "나", "도", "만", "까지", "부터", "이라", "라",
    "에게", "한테", "께", "에서는", "에게서", "씨", "님",
)


def _strip_ko_particle(text: str) -> str:
    """
    *text* 끝에 붙은 한국어 조사 하나를 떼어낸다.
    - 부분 매칭을 피하려고 긴 조사부터 짧은 조사 순으로 시도.
    - 떼어내도 남는 글자가 2자 이상일 때만 제거(짧은 이름의 끝글자를 실수로 떼지 않기 위함).
    - 조사가 없으면 원본 그대로 반환.
    """
    for particle in sorted(_KO_PARTICLES, key=len, reverse=True):  # 긴 조사부터
        if text.endswith(particle) and len(text) - len(particle) >= 2:  # 끝이 조사 + 남는 글자 ≥2
            return text[: len(text) - len(particle)]               # 조사만큼 잘라 반환
    return text                                                    # 해당 없으면 원본


class KoreanNEREngine:
    """
    Presidio 기반 한국어 NER 엔진(ko_core_news_* spaCy 모델을 감싼다).

    지연 초기화(lazy): spaCy 모델/Presidio 파이프라인은 detect() '첫 호출'에 로드되고,
    같은 프로세스 내 이후 호출에서 재사용된다(서브프로세스 수명 동안 싱글턴).

    Parameters
    ----------
    min_confidence:
        이 점수 미만 탐지는 버림. 기본 MIN_CONFIDENCE(0.50).
    strip_particles:
        True(기본)면 엔티티 끝의 한국어 조사를 떼어내 "홍길동은"→"홍길동".
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,    # 신뢰도 하한
        strip_particles: bool = True,              # 조사 제거 on(기본)
        model_name: Optional[str] = None,          # (선택) 모델 고정. None이면 로드 시 자동 결정
    ) -> None:
        self._min_confidence = min_confidence
        self._strip_particles = strip_particles
        # None → 로드 시점에 결정(lg 우선, 환경변수 오버라이드 가능).
        # 명시적 이름을 주면 그 모델로 고정(테스트/벤치마크용).
        self._model_name = model_name
        self._analyzer: Optional[object] = None    # 지연 초기화 — 첫 사용 때 채워짐

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> List[Detection]:
        """
        *text* 안의 한국어 PII 엔티티를 탐지.
        Presidio+한국어 spaCy NER를 돌려 결과를 PII-Guard Detection 객체로 변환.
        반환: 시작 위치순으로 정렬된 Detection 리스트(빈 리스트일 수 있음).
        """
        if not text or not text.strip():           # 빈/공백 문자열이면
            return []                              # 바로 빈 결과

        analyzer = self._get_analyzer()            # (필요 시 로드해) 분석기 확보
        try:
            results = analyzer.analyze(text=text, language="ko")  # 한국어로 NER 분석
        except Exception as exc:  # noqa: BLE001
            logger.warning("Presidio analysis failed: %s", exc)   # 분석 실패는 경고만
            return []                                              # 빈 결과로 안전 폴백

        detections: List[Detection] = []           # 변환 결과를 담을 리스트
        for result in results:                     # Presidio가 찾은 각 엔티티에 대해
            if result.score < self._min_confidence:  # 신뢰도 미달이면
                continue                           #   건너뜀
            presidio_entity = result.entity_type   # Presidio 엔티티 타입(PERSON/LOCATION/…)
            category = _PRESIDIO_TO_CATEGORY.get(presidio_entity)  # PII-Guard 카테고리로 변환
            if category is None:                   # 우리가 안 쓰는 타입(DATE_TIME 등)이면
                continue                           #   건너뜀

            # 원본 스팬 텍스트 추출
            original = text[result.start: result.end]

            # (옵션) 끝에 붙은 한국어 조사 제거
            clean_text = original
            adjusted_end = result.end              # 조사를 떼면 끝 위치도 줄여야 함
            if self._strip_particles:
                clean_text = _strip_ko_particle(original)
                if len(clean_text) < len(original):     # 실제로 줄었으면
                    adjusted_end = result.start + len(clean_text)  # 끝 위치 보정

            det = Detection(                       # PII-Guard 표준 탐지 객체 생성
                category=category,                 #   카테고리(PERSON/ADDRESS/ORGANIZATION)
                category_class=_ENTITY_CLASS[presidio_entity],  #   분류(KOREAN_PII)
                action=_ENTITY_ACTION[presidio_entity],         #   조치(TOKENIZE_ROUNDTRIP)
                mask_style=MaskStyle.TOKENIZE,     #   마스킹 방식(토큰)
                start=result.start,                #   시작 위치
                end=adjusted_end,                  #   (보정된) 끝 위치
                original=clean_text,               #   원본 값(조사 제거 후)
                detection_stage=DetectionStage.STAGE2_NER,  #   단계 표시(Stage2)
                rule_id=f"ner_ko_{presidio_entity.lower()}",  #   규칙 식별자
                confidence=result.score,           #   신뢰도 점수
            )
            detections.append(det)

        # 시작 위치순 정렬
        detections.sort(key=lambda d: d.start)
        # 음성 proximity 억제: NER 오탐(코드 토큰·약어·blob·일반명사)을 걸러낸다.
        # 재현율은 지키면서 정밀도를 올리는 필터.
        return [d for d in detections
                if not is_ner_false_positive(d.category, d.original)]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_analyzer(self):
        """
        Presidio AnalyzerEngine을 반환(첫 호출에 지연 초기화).
        Presidio/spaCy import를 여기까지 미루는 이유: 이 모듈을 메인 프로세스에서
        (예: 타입 체크로) import하더라도 ~400MB spaCy 모델이 부모 메모리에 올라가지 않게 하려고.
        """
        if self._analyzer is not None:             # 이미 만들어 뒀으면
            return self._analyzer                  #   재사용

        model_name = self._model_name or resolve_ko_spacy_model()  # 고정값 없으면 자동 결정
        logger.info("Loading Korean spaCy model for NER: %s", model_name)  # 로딩 로그
        self._analyzer = _build_presidio_analyzer(model_name)      # 실제 구성
        return self._analyzer


def _build_presidio_analyzer(model_name: Optional[str] = None):
    """
    한국어용 Presidio AnalyzerEngine을 만들어 반환.
    구성 내용:
      * NerModelConfiguration: 한국어 spaCy 라벨 → Presidio 엔티티 매핑(PS→PERSON 등).
      * supported_language='ko'로 제한한 SpacyRecognizer(PII 관련 3종만).
      * 영어 기본 recognizer가 끼어들지 않도록, 레지스트리엔 이 한국어 recognizer만 등록.
    실패 시 RuntimeError(모델 미설치 / Presidio import 실패)를 명확히 던진다.
    """
    if model_name is None:                         # 안 받았으면
        model_name = resolve_ko_spacy_model()      #   자동 결정

    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import SpacyRecognizer
    except ImportError as exc:                     # presidio 미설치면
        raise RuntimeError(
            "presidio-analyzer is not installed. "
            "Run: pip install presidio-analyzer presidio-anonymizer"
        ) from exc

    try:
        provider = NlpEngineProvider(              # spaCy 기반 NLP 엔진 설정
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "ko", "model_name": model_name}  # 한국어=ko, 어떤 모델인지
                ],
                "ner_model_configuration": {
                    # 한국어 spaCy NER 라벨 → Presidio 표준 엔티티 이름으로 매핑
                    "model_to_presidio_entity_mapping": {
                        "PS": "PERSON",       # 사람/인물 (Person)
                        "LC": "LOCATION",     # 장소/지역 (Location)
                        "OG": "ORGANIZATION", # 기관/조직 (Organization)
                        "DT": "DATE_TIME",    # 날짜 (Date) — PII 매핑은 아니지만 포함
                    },
                    # spaCy가 스팬별 점수를 안 줄 때 쓰는 기본 신뢰도
                    "default_score": 0.70,
                },
            }
        )
        nlp_engine = provider.create_engine()      # 실제 NLP 엔진 생성(모델 로딩 발생)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(                        # 모델 로딩 실패 → 설치 안내와 함께 오류
            f"Failed to load Korean spaCy model '{model_name}'. "
            f"Run: python -m spacy download {model_name}"
        ) from exc

    # 한국어 recognizer — spaCy NER 엔티티를 PERSON/LOCATION/ORGANIZATION으로 인식
    korean_recognizer = SpacyRecognizer(
        supported_language="ko",                   # 한국어 전용
        supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],  # 이 3종만
        ner_strength=0.70,  # 모델이 점수를 안 줄 때의 기본 신뢰도
    )

    registry = RecognizerRegistry()                # 빈 레지스트리(영어 기본 recognizer 배제)
    registry.add_recognizer(korean_recognizer)     # 한국어 recognizer만 등록

    analyzer = AnalyzerEngine(                      # 최종 분석기 조립
        nlp_engine=nlp_engine,
        registry=registry,
    )
    return analyzer
