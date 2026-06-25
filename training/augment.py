"""
training/augment.py — (보조) 합성 데이터 생성 (ADR-13).

보유 실데이터가 1급 입력이고, 합성은 **부족 카테고리·hard-negative 보강용 보조**다.
출력은 ingest와 동일한 raw 포맷({text, spans})이라, 실데이터 raw와 합쳐 ingest에 넣으면 된다.

두 종류를 만든다:
  ① positive — gazetteer(이름/주소/조직)를 템플릿에 채워 라벨된 문장 생성(재현율↑).
  ② hard-negative — GLiNER가 PERSON/ORG로 오인하기 쉬운 ID/코드 토큰·일반 조직어구를
     **엔티티 없음(spans=[])**으로 둬 "이건 PII 아님"을 학습(정밀도↑, 특히 ORG 과추출 교정).

실행:
    .venv/bin/python -m training.augment <out_raw.jsonl> [--n-pos 200] [--n-neg 200] [--seed 42]

⚠️ 합성은 템플릿 과적합 위험이 있다. 실데이터를 우선하고, 합성은 보조 비율로만 섞을 것.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

# 작은 내장 gazetteer (공개·합성 표면형 — 실 PII 아님). 실제로는 외부 공개 사전으로 확장.
_NAMES = ["김서연", "이준호", "박민지", "최우진", "정하늘", "강도윤", "윤서아", "임건우", "한지오", "오세훈"]
_ADDRS = ["서울특별시 강남구 테헤란로 152", "부산광역시 해운대구 센텀중앙로 97",
          "경기도 성남시 분당구 판교역로 235", "대구광역시 수성구 달구벌대로 2450",
          "인천광역시 연수구 컨벤시아대로 165"]
_ORGS = ["삼성전자", "네이버", "카카오", "현대자동차", "엘지화학", "쿠팡", "토스", "당근마켓"]

# hard-negative: 엔티티 아님인데 PERSON/ORG로 오인되기 쉬운 토큰·어구
_NEG_IDS = ["ORD-2026-1102", "TX-9988231", "TRACK-002931-KR", "v3.2.0", "SEQ-2026-44",
            "INV-20260612", "build-1.8.0", "REQ-77123"]
_NEG_ORGISH = ["저희 회사", "해당 대행업체", "관련 부서", "담당 팀", "고객센터"]

_POS_TEMPLATES = [
    ("담당자 {PERSON} 님께 전달했습니다.", "PERSON"),
    ("{PERSON} 고객님 본인 확인 완료.", "PERSON"),
    ("배송지는 {ADDRESS} 입니다.", "ADDRESS"),
    ("{ADDRESS} 로 방문 예정.", "ADDRESS"),
    ("{ORGANIZATION} 에 재직 중입니다.", "ORGANIZATION"),
    ("협력사는 {ORGANIZATION} 입니다.", "ORGANIZATION"),
]
_NEG_TEMPLATES = [
    "주문번호 {ID} 확인 부탁드립니다.",
    "현재 앱 버전은 {ID} 입니다.",
    "{ORGISH}에서 처리 예정입니다.",
    "{ORGISH} 담당자가 회신드립니다.",
]
_POOLS = {"PERSON": _NAMES, "ADDRESS": _ADDRS, "ORGANIZATION": _ORGS}


def _make_positive(rng: random.Random) -> Dict:
    tmpl, label = rng.choice(_POS_TEMPLATES)
    val = rng.choice(_POOLS[label])
    placeholder = "{" + label + "}"
    start = tmpl.index(placeholder)
    text = tmpl.replace(placeholder, val)
    return {"text": text, "spans": [[start, start + len(val), label]]}


def _make_negative(rng: random.Random) -> Dict:
    tmpl = rng.choice(_NEG_TEMPLATES)
    if "{ID}" in tmpl:
        text = tmpl.replace("{ID}", rng.choice(_NEG_IDS))
    else:
        text = tmpl.replace("{ORGISH}", rng.choice(_NEG_ORGISH))
    return {"text": text, "spans": []}   # 엔티티 없음 = hard negative


def generate(n_pos: int, n_neg: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    docs = [_make_positive(rng) for _ in range(n_pos)] + \
           [_make_negative(rng) for _ in range(n_neg)]
    rng.shuffle(docs)
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_raw_jsonl")
    ap.add_argument("--n-pos", type=int, default=200)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    docs = generate(a.n_pos, a.n_neg, a.seed)
    with open(a.out_raw_jsonl, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} synthetic raw docs "
          f"({a.n_pos} positive + {a.n_neg} hard-negative) → {a.out_raw_jsonl}")
