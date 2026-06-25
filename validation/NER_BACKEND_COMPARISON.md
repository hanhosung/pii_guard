# Stage2 NER 백엔드 측정 — spaCy vs GLiNER (2026-06-25)

> 두 백엔드의 한국어 비정형 PII(PERSON·ADDRESS·ORGANIZATION) 검출 성능을 **두 종류의 데이터**로 측정한다.
> ① 라벨 코퍼스(합성, ground-truth) — 순수 모델 품질. ② 외부 LLM 생성 VOC/로그 케이스(현실형 한·영 혼합) — 실전 검출력.
> 모든 측정은 모델 로드 후 동일 입력으로 수행했다. 재현: `benchmarks/korean_ner_benchmark.py --ner-backend {spacy,gliner}`.
>
> - **백엔드**: `spacy` = Presidio + `ko_core_news_lg` · `gliner` = `taeminlee/gliner_ko`
> - **지표**: 정탐(TP)=정답을 맞게 검출 · 오탐(FP)=PII 아닌 것을 PII로 검출 · 미탐(FN)=정답을 놓침
> - precision = TP/(TP+FP) · recall = TP/(TP+FN)

---

## 1. 라벨 코퍼스 (합성·ground-truth) — 순수 모델 품질

NER 소유 3개 카테고리, full-pipeline 기준.

| 카테고리 | 백엔드 | precision | recall | 정탐(TP) | 오탐(FP) | 미탐(FN) |
| :-- | :-- | --: | --: | --: | --: | --: |
| **PERSON** | spaCy | 0.974 | 0.844 | 38 | 1 | 7 |
| | **GLiNER** | 0.800 | **0.978** | **44** | 11 | **1** |
| **ADDRESS** | spaCy | 1.000 | 1.000 | 25 | 0 | 0 |
| | **GLiNER** | 1.000 | 1.000 | 25 | 0 | 0 |
| **ORGANIZATION** | spaCy | 1.000 | 0.920 | 23 | 0 | 2 |
| | **GLiNER** | 0.862 | **1.000** | **25** | 4 | **0** |

- **GLiNER = 재현율 우위**: PERSON 미탐 7→1, ORG 미탐 2→0 (놓치는 이름·조직이 거의 없음).
- **spaCy = 정밀도 우위**: 오탐 PERSON 1·ORG 0 (군더더기 검출이 적음).
- ADDRESS는 두 백엔드 모두 완전 일치(25/25).

## 2. 외부 LLM 생성 VOC/로그 (현실형 한·영 혼합) — 실전 검출력

10개 케이스(VOC + 서버 로그), 전체 파이프라인(Stage1 + NER) 합산. Stage1 카테고리는 두 백엔드 공통이므로 차이는 NER 카테고리에서만 발생한다.

| 지표 | spaCy | GLiNER |
| :-- | --: | --: |
| 정탐(TP) | 63 | 62 |
| 미탐(FN) | 9 | 10 |
| **오탐(FP)** | **37** | **14** |
| recall | 0.875 | 0.861 |
| **precision** | **0.630** | **0.816** |

- 재현율은 거의 동일(0.875 vs 0.861)하나 **정밀도는 GLiNER가 크게 우위(0.816 vs 0.630)** — 오탐이 37→14로 절반 이하.

### 2-1. 카테고리별 정탐/미탐/오탐

NER 관련 카테고리만 발췌(Stage1 카테고리는 두 백엔드 동일).

| 카테고리 | spaCy TP/FN/FP | GLiNER TP/FN/FP |
| :-- | :-- | :-- |
| PERSON | 10 / 0 / **12** | 10 / 0 / **6** |
| ADDRESS | 3 / 0 / 3 | 2 / 1 / 1 |
| ORGANIZATION | 0 / 0 / **19** | 0 / 0 / **4** |

> PERSON·ADDRESS 정탐은 두 백엔드 모두 사실상 동일(이름 10/10). 차이는 **오탐 규모**: spaCy ORG 19·PERSON 12 vs GLiNER ORG 4·PERSON 6.

---

## 3. 오탐(FP) 분류 — 성격별

오탐 37/14건을 성격으로 나누면 "진짜 과검"과 "라벨 누락·채점 아티팩트"가 섞여 있다.

### 3-1. spaCy 오탐 (총 37)

| 성격 | 예시 | 비고 |
| :-- | :-- | :-- |
| **진짜 과검 — 영문 로그 토큰** | `PERSON`=auth(다수)·active / `ORGANIZATION`=detected·security·third·found·webhook·Received | 영문·코드성 로그 단어를 인물/조직으로 오인 → **실제 over-masking** |
| 라벨 누락 정탐(실제 PII) | `ORGANIZATION`=우리은행·국민카드·하나은행 | 진짜 조직명이나 ground truth에 라벨 없음 → FP로 집계 |
| 채점 아티팩트 | `RRN`=120923-1591783·700523-4376198 / `HOSTNAME`=api.internal | RRN 포맷을 RRN으로 정탐(정답 라벨은 FOREIGN_REG) · 미라벨 실제 호스트 |

### 3-2. GLiNER 오탐 (총 14)

| 성격 | 예시 | 비고 |
| :-- | :-- | :-- |
| **진짜 과검 — ID/코드 토큰** | `PERSON`=ORD-2026-1102·TX-9988231·v3.2.0·TRACK-002931-KR | 주문/트랜잭션 ID·버전 문자열을 인물로 오인 |
| 라벨 누락 정탐(실제 PII) | `ORGANIZATION`=우리은행·국민카드·하나은행·기업은행 | 진짜 은행명이나 라벨 없음 → FP로 집계 |
| 채점 아티팩트 | `RRN`=120923-1591783·700523-4376198 / `HOSTNAME`=api.internal | spaCy와 동일(Stage1/라벨 기인, 백엔드 무관) |

> **요지**: 두 백엔드의 "라벨 누락·아티팩트" 오탐(은행명·RRN·호스트)은 거의 동일하다. 백엔드 차이는 **진짜 과검**에서 나온다 — spaCy는 **영문 로그 토큰**을, GLiNER는 **ID/코드 토큰**을 오인하지만 GLiNER 쪽 규모가 훨씬 작다(영문 로그 노이즈에 강함).

---

## 4. 종합

| 관점 | 우위 |
| :-- | :-- |
| 라벨 코퍼스 재현율(이름·조직 놓침) | **GLiNER** (PERSON 0.978, ORG 1.00) |
| 라벨 코퍼스 정밀도 | spaCy (오탐 거의 0) |
| 현실형 VOC 재현율 | 동등 (0.86 vs 0.88) |
| 현실형 VOC 정밀도(오탐) | **GLiNER** (0.82 vs 0.63, 오탐 14 vs 37) |
| 영문 로그 노이즈 내성 | **GLiNER** |

- **깨끗한 한국어**에서는 GLiNER가 재현율(놓치지 않음), spaCy가 정밀도(군더더기 없음) 우위.
- **현실형 혼합 텍스트**에서는 GLiNER가 재현율 동등 + 정밀도 우위 → 종합 우위.

> 데이터: 합성 NER 코퍼스(`pii_guard/corpus/ner_benchmark_corpus.py`, seed=42), 외부 LLM 생성 VOC/로그 10건. 수치는 본 문서 표에 인라인 기록(원천은 `benchmarks/korean_ner_benchmark.py`로 재현 가능).
