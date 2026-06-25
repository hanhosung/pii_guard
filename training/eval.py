"""
training/eval.py — 파인튜닝 모델 사전/사후 평가 + 채택 게이트 (ADR-13).

기존 코퍼스 벤치마크(`benchmarks/korean_ner_benchmark.py`)를 **베이스 모델**과 **파인튜닝 모델**
양쪽으로 돌려 카테고리별 recall/precision을 비교한다(누설 방지: 학습셋과 코퍼스는 분리돼 있어야 함).

채택 게이트(권장): 파인튜닝 모델이
  ① 전 임계값 통과(thresholds_met) AND
  ② 어떤 카테고리도 recall이 유의하게 하락하지 않음(기본 허용오차 0.02) AND
  ③ 목표 지표(예: ORG precision) 개선
일 때만 기본 모델로 승격.

실행:
    .venv/bin/python -m training.eval --finetuned runs/ko_pii_ft/final
    .venv/bin/python -m training.eval --finetuned <path> --base urchade/gliner_multi_pii-v1
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

CATS = ("PERSON", "ADDRESS", "ORGANIZATION")
BASE_MODEL = "urchade/gliner_multi_pii-v1"


def _bench(model: str, seed: int, samples: int, min_conf: float) -> Dict:
    """주어진 GLiNER 모델로 코퍼스 벤치마크 1회 실행 → metrics."""
    os.environ["PIIGUARD_NER_BACKEND"] = "gliner"
    os.environ["PIIGUARD_GLINER_MODEL"] = model
    # 런타임 측정 하니스 재사용(eval은 runtime을 쓰는 다리)
    from benchmarks.korean_ner_benchmark import run_benchmark
    rep = run_benchmark(corpus_seed=seed, samples_per_format=samples,
                        min_confidence=min_conf, apply_thresholds=True,
                        quiet=True, ner_backend="gliner")
    return rep


def compare(base_rep: Dict, ft_rep: Dict, recall_tol: float = 0.02) -> Dict:
    """베이스 vs 파인튜닝 카테고리별 delta + 게이트 판정."""
    b = base_rep["full_pipeline_metrics"]
    f = ft_rep["full_pipeline_metrics"]
    rows, regress = [], []
    for c in CATS:
        dr = round(f[c]["recall"] - b[c]["recall"], 4)
        dp = round(f[c]["precision"] - b[c]["precision"], 4)
        rows.append((c, b[c]["recall"], f[c]["recall"], dr, b[c]["precision"], f[c]["precision"], dp))
        if dr < -recall_tol:
            regress.append(f"{c} recall {b[c]['recall']:.3f}→{f[c]['recall']:.3f} ({dr:+.3f})")
    passed = bool(ft_rep.get("thresholds_met")) and not regress
    return {"rows": rows, "regress": regress,
            "thresholds_met": bool(ft_rep.get("thresholds_met")), "gate_pass": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True, help="파인튜닝 모델 경로/repo")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    a = ap.parse_args()

    print(f"[eval] base    = {a.base}")
    base_rep = _bench(a.base, a.seed, a.samples, a.min_confidence)
    print(f"[eval] finetuned = {a.finetuned}")
    ft_rep = _bench(a.finetuned, a.seed, a.samples, a.min_confidence)

    res = compare(base_rep, ft_rep)
    print(f"\n{'cat':<14}{'base R':>8}{'ft R':>8}{'ΔR':>8}{'base P':>9}{'ft P':>8}{'ΔP':>8}")
    for c, br, fr, dr, bp, fp, dp in res["rows"]:
        print(f"{c:<14}{br:>8.3f}{fr:>8.3f}{dr:>+8.3f}{bp:>9.3f}{fp:>8.3f}{dp:>+8.3f}")
    print(f"\nthresholds_met={res['thresholds_met']}  recall 회귀={res['regress'] or '없음'}")
    print(f"채택 게이트: {'✅ PASS' if res['gate_pass'] else '❌ FAIL'}")
    return 0 if res["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
