"""
training/split.py — train/val/test 분할 + 평가 누설(leakage) 방지 (ADR-13).

원칙: 학습셋과 **평가셋(benchmarks 코퍼스·외부 6리포트)은 절대 겹치면 안 된다.**
겹치면 점수가 부풀려진다(누설). 평가셋 지문(fingerprint) 파일이 주어지면, 그 텍스트와
일치하는 예시를 학습/검증/테스트에서 제거한다.

실행:
    .venv/bin/python -m training.split <gliner.jsonl> <out_dir> [--eval-fp eval_fingerprints.txt]
출력: <out_dir>/{train,val,test}.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Set

from .common import example_text


def fingerprint(text: str) -> str:
    """정규화(소문자·공백 1개·strip) 후 해시 — 누설 비교용 지문."""
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def load_eval_fingerprints(path: Optional[str]) -> Set[str]:
    """평가셋 지문 파일(한 줄당 원문 또는 지문) → 지문 집합. 없으면 빈 집합."""
    if not path:
        return set()
    out: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # 40자 hex면 이미 지문, 아니면 원문으로 보고 해시
            out.add(line if (len(line) == 40 and all(c in "0123456789abcdef" for c in line)) else fingerprint(line))
    return out


def split(
    examples: List[Dict],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
    eval_fingerprints: Optional[Set[str]] = None,
) -> Dict[str, List[Dict]]:
    """누설 제거 후 seeded 분할. 반환 {train,val,test, dropped_leak}."""
    eval_fingerprints = eval_fingerprints or set()
    kept, dropped = [], 0
    for ex in examples:
        if fingerprint(example_text(ex)) in eval_fingerprints:
            dropped += 1            # 평가셋과 겹침 → 학습에서 제외
        else:
            kept.append(ex)
    rng = random.Random(seed)
    rng.shuffle(kept)
    n = len(kept)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {
        "train": kept[:n_tr],
        "val": kept[n_tr:n_tr + n_va],
        "test": kept[n_tr + n_va:],
        "dropped_leak": dropped,
    }


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gliner_jsonl")
    ap.add_argument("out_dir")
    ap.add_argument("--eval-fp", default=None, help="평가셋 지문/원문 파일(누설 방지)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    exs = _read_jsonl(a.gliner_jsonl)
    res = split(exs, seed=a.seed, eval_fingerprints=load_eval_fingerprints(a.eval_fp))
    os.makedirs(a.out_dir, exist_ok=True)
    for name in ("train", "val", "test"):
        _write_jsonl(os.path.join(a.out_dir, f"{name}.jsonl"), res[name])
    print(json.dumps(
        {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()},
        ensure_ascii=False, indent=2,
    ))
    if res["dropped_leak"]:
        import sys as _sys
        print(f"⚠️ 누설 제거 {res['dropped_leak']}건(평가셋과 겹침)", file=_sys.stderr)
