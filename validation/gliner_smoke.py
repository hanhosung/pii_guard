"""
GLiNER 백엔드 스모크 테스트 (R18 실측).

실제 GLiNER 한국어 모델을 로드해 PII-Guard GLiNERNEREngine.detect()가
PERSON/ADDRESS/ORGANIZATION을 잡는지 확인한다. 모델은 첫 실행에 다운로드된다.

실행:  .venv/bin/python validation/gliner_smoke.py
모델 교체:  PIIGUARD_GLINER_MODEL=<hf-model> .venv/bin/python validation/gliner_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

# 패키지 import 경로 보장
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pii_guard.stage2.gliner_ner import GLiNERNEREngine, resolve_gliner_model

SAMPLES = [
    "안녕하세요, 저는 김철수입니다. 서울특별시 강남구 테헤란로 123에 있는 삼성전자에서 일합니다.",
    "담당자 박영희 과장(010-1234-5678)에게 전달했고, 네이버 본사 분당 사옥에서 회의했습니다.",
    "이순신 장군은 전라좌수영에 주둔했다.",
]


def main() -> int:
    model = resolve_gliner_model()
    print(f"[gliner-smoke] model = {model}")
    print(f"[gliner-smoke] backend env PIIGUARD_NER_BACKEND={os.environ.get('PIIGUARD_NER_BACKEND', '(unset)')}")

    t0 = time.time()
    eng = GLiNERNEREngine()
    # 첫 detect에서 모델 로드(다운로드 포함)
    first = eng.detect(SAMPLES[0])
    t_load = time.time() - t0
    print(f"[gliner-smoke] first detect (incl. model load) = {t_load:.1f}s")

    total = 0
    for i, text in enumerate(SAMPLES):
        t1 = time.time()
        dets = eng.detect(text)
        dt = (time.time() - t1) * 1000
        total += len(dets)
        print(f"\n--- sample {i} ({dt:.0f}ms) ---")
        print(f"    {text}")
        if not dets:
            print("    (no detections)")
        for d in dets:
            print(f"    [{d.category:<12}] {d.original!r}  conf={d.confidence:.2f}  span=({d.start},{d.end})")

    print(f"\n[gliner-smoke] total detections across {len(SAMPLES)} samples = {total}")
    cats = {d.category for text in SAMPLES for d in eng.detect(text)}
    ok = bool(cats & {"PERSON", "ADDRESS", "ORGANIZATION"})
    print(f"[gliner-smoke] categories seen = {sorted(cats)}")
    print(f"[gliner-smoke] RESULT = {'PASS' if ok else 'FAIL (no PII categories detected)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
