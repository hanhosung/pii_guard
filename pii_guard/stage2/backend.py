"""
pii_guard/stage2/backend.py

Stage-2 NER 백엔드 선택 추상화 (요구사항 R18 / DESIGN ADR-11).

[이 파일이 하는 일 — 한 줄 요약]
Stage2 NER 엔진을 **GLiNER(기본)·spaCy(경량 폴백)·NuNER Zero(평가 후보, R21·ADR-14)** 중
무엇으로 돌릴지 결정한다. 선택 우선순위는 **환경변수 `PIIGUARD_NER_BACKEND` > 정책
`stage2.ner_backend` > 기본 `gliner`**. 세 엔진 모두 동일한 `.detect(text) -> List[Detection]`
인터페이스를 가지므로, 워커 루프는 여기서 고른 엔진 클래스를 지연 임포트해 쓰기만 하면
백엔드와 무관하게 동작한다.

정책(YAML)은 별도 프로세스인 워커로 직접 전달되지 않으므로, 부모 프로세스(Engine)가
선택 결과를 `PIIGUARD_NER_BACKEND` 환경변수로 내보내고, 워커는 그 env만 읽는다(아래 resolve).
"""
from __future__ import annotations

import os                                    # 환경변수 읽기
from enum import Enum                        # 백엔드 종류 열거
from typing import Optional                  # 타입 힌트

#: 백엔드 선택을 전달하는 환경변수 이름(부모→워커 서브프로세스 전파 경로)
ENV_NER_BACKEND = "PIIGUARD_NER_BACKEND"

#: 기본 백엔드(재현율 우선 — 한국어 특화 GLiNER)
DEFAULT_BACKEND = "gliner"


class NERBackend(str, Enum):
    """Stage2 NER 백엔드 종류. (str 상속 → 값 비교/직렬화가 문자열처럼 동작)"""
    GLINER = "gliner"         # 다국어 PII GLiNER(기본·Apache-2.0)
    SPACY = "spacy"           # Presidio + spaCy(경량 폴백)
    NUNERZERO = "nunerzero"   # NuNER Zero(평가 후보·MIT, R21·ADR-14 — 벤치 게이트 후 승격)


#: 허용되는 백엔드 값 집합(검증용)
_VALID = {b.value for b in NERBackend}


def resolve_ner_backend(policy_backend: Optional[str] = None) -> NERBackend:
    """
    어떤 NER 백엔드를 쓸지 결정해서 반환.

    결정 순서(앞이 우선):
      1. 환경변수 `PIIGUARD_NER_BACKEND` (있으면 그대로)
      2. 인자 *policy_backend* (정책 YAML `stage2.ner_backend`에서 온 값)
      3. 기본값 `gliner`

    알 수 없는 값이면 ValueError를 던진다(침묵 폴백 금지 — P3/P5 거짓 안심 방지).
    워커 서브프로세스는 policy_backend 없이 호출하므로, 환경변수만으로 결정된다.
    """
    raw = os.environ.get(ENV_NER_BACKEND)    # 1) env 우선
    source = "env"
    if not raw:
        raw = policy_backend                 # 2) 정책값
        source = "policy"
    if not raw:
        raw = DEFAULT_BACKEND                # 3) 기본값
        source = "default"

    normalized = str(raw).strip().lower()    # 대소문자/공백 정규화
    if normalized not in _VALID:             # 모르는 값이면 명확한 오류
        raise ValueError(
            f"Unknown NER backend {raw!r} (from {source}). "
            f"Valid values: {sorted(_VALID)}. "
            f"Set the {ENV_NER_BACKEND} env var or policy 'stage2.ner_backend'."
        )
    return NERBackend(normalized)


def load_engine_class(backend: NERBackend):
    """
    선택된 백엔드에 해당하는 NER 엔진 '클래스'를 지연 임포트해서 반환.

    무거운 의존(GLiNER/NuNER Zero+PyTorch, spaCy+Presidio)이 부모(코어) 프로세스에 올라가지
    않도록, 임포트를 이 함수 호출 시점(=워커 서브프로세스 안)으로 미룬다.
    반환된 클래스는 모두 `__init__()` 후 `.detect(text) -> List[Detection]`를 제공한다.
    """
    if backend is NERBackend.GLINER:
        from .gliner_ner import GLiNERNEREngine  # noqa: PLC0415 — 지연 임포트 의도
        return GLiNERNEREngine
    if backend is NERBackend.NUNERZERO:
        # NuNER Zero(R21·ADR-14) — gliner 라이브러리로 로드되는 동계열 평가 후보.
        from .nunerzero_ner import NuNERZeroNEREngine  # noqa: PLC0415 — 지연 임포트 의도
        return NuNERZeroNEREngine
    from .korean_ner import KoreanNEREngine      # noqa: PLC0415 — 지연 임포트 의도
    return KoreanNEREngine
