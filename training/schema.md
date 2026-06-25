# 학습 데이터 스키마 (ADR-13 파인튜닝)

## 1. 입력(raw) — 보유 라벨 데이터 (1급 입력)

JSONL, 한 줄당 한 문서:

```json
{"text": "담당자 김서연 님께 전달했습니다.", "spans": [[4, 7, "PERSON"]]}
{"text": "주문번호 ORD-2026-1102 확인 부탁드립니다.", "spans": []}
```

- `text` (str): 원문.
- `spans` (list): `[char_start, char_end, label]` — **문자 오프셋**, `end`는 **배타(exclusive)**.
  - `label` ∈ `PERSON` · `ADDRESS` · `ORGANIZATION` (= 런타임 Detection 카테고리).
  - `spans: []` 인 줄은 **hard-negative**(엔티티 없음) — ORG 과추출 등 오탐 교정용.

> 라벨 경계는 토큰 경계와 맞는 게 좋다(예: `김서연`만, 조사 `님` 제외). `ingest`가 토큰 경계 불일치를
> 경고로 카운트한다.

## 2. 학습(gliner) — `ingest`/`split` 산출

JSONL, GLiNER `train_model` 입력 포맷:

```json
{"tokenized_text": ["담당자","김서연","님께","전달했습니다","."], "ner": [[1, 1, "PERSON"]]}
```

- `tokenized_text` (list[str]): 단어/구두점 토큰.
- `ner` (list): `[tok_start, tok_end, label]` — **토큰 인덱스**, `end`는 **포함(inclusive)**(GLiNER 규약).

## 3. 라벨 통합 주의 (런타임 일치)

GLiNER는 라벨-텍스트 매칭이라 **학습 라벨 = 런타임 질의 라벨**이어야 효과가 난다.
- 학습 라벨: `PERSON`/`ADDRESS`/`ORGANIZATION` (캐노니컬).
- 현재 런타임(`pii_guard/stage2/gliner_ner.py::_GLINER_LABELS`)은 한국어 동의어(`사람`·`주소`·`조직`…)로 질의.
- **파인튜닝 모델 배포 시**: 런타임 질의 라벨을 캐노니컬 집합으로 정렬할 것(README 통합 절).
