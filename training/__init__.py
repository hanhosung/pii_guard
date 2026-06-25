"""
GLiNER 파인튜닝 서브시스템 (ADR-13). 오프박스 학습용 — 런타임 코어(`pii_guard/`)와 분리.
무거운 의존(torch/gliner)은 각 모듈의 함수 내부에서 지연 import한다.
"""
