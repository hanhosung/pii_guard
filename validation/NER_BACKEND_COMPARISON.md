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

## 2. 외부 LLM 생성 VOC/로그 (Gemini 데이터셋, 현실형 한·영 혼합) — 실전 검출력

Gemini 생성 10개 케이스(VOC + 서버 로그), 전체 파이프라인(Stage1 + NER) 합산. Stage1 카테고리는 두 백엔드 공통이므로 차이는 NER 카테고리에서만 발생한다. (3개 데이터셋 전체 비교는 §2-2.)

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

### 2-2. 6개 리포트 종합 대조 (3 데이터셋 × 2 백엔드)

`validation/`의 외부 LLM 테스트 6개 리포트(claude·codex·gemini × spaCy·GLiNER) raw 채점치를 한 표로 모았다. 전체 파이프라인(Stage1+NER) 기준이라 시크릿·정형 PII 등 Stage1 카테고리가 분모에 포함되며, 두 백엔드 차이는 NER 카테고리(PERSON/ADDRESS/ORGANIZATION)에서만 발생한다. 각 데이터셋은 spaCy/GLiNER가 **동일 입력**으로 채점됐다.

| 데이터셋(케이스수) | 백엔드 | 정탐 TP | 미탐 FN | 오탐 FP | 재현율 | 정밀도 |
| :-- | :-- | --: | --: | --: | --: | --: |
| **claude** (30) | spaCy | 192 | 12 | 27 | 0.941 | 0.877 |
| | **GLiNER** | 197 | 7 | 41 | **0.966** | 0.828 |
| **codex** (10) | spaCy | 71 | 18 | 17 | 0.798 | **0.807** |
| | **GLiNER** | 75 | 14 | 36 | **0.843** | 0.676 |
| **gemini** (10) | spaCy | 63 | 9 | 37 | 0.875 | 0.630 |
| | **GLiNER** | 62 | 10 | 14 | 0.861 | **0.816** |

**재현율 차이(GLiNER − spaCy)**: claude +0.025, codex +0.045, gemini −0.014 — **세 데이터셋 모두 GLiNER가 더 적게 놓침**(미탐 FN: claude 12→7, codex 18→14).

**정밀도 차이(GLiNER − spaCy)**: claude −0.049, codex **−0.131**, gemini **+0.186** — 데이터 성격에 따라 갈림.
- **gemini(서버 로그 다수)**: GLiNER 대폭 우위 — spaCy가 영문 로그 토큰(`auth`·`webhook`)을 인물/조직으로 과잉 추출(FP 37), GLiNER는 강함(FP 14).
- **codex(자연어 VOC)**: GLiNER 열세 — 주문/트랜잭션 ID·코드 토큰(`ORD-2026`·`TX-9988231`·`v3.2.0`)을 PERSON으로 오인해 FP 증가(17→36).
- **claude(자연어 VOC)**: 소폭 열세 — codex와 유사하게 GLiNER의 ID/코드 토큰 오탐.

> **요지**: **재현율(유출 방지)은 GLiNER가 전반 우위**. 정밀도는 **로그·코드 혼합 텍스트에서는 GLiNER가 크게 유리**(영문 토큰 노이즈 내성), **자연어 VOC에서는 spaCy가 유리**(GLiNER가 ID/코드를 이름으로 오인). 두 백엔드의 오탐 상당수는 라벨 누락 정탐(은행명=ORG)·채점 아티팩트(RRN 포맷)로, 실제 과검은 §3 분류 참고.
>
> 출처 리포트: `EXTERNAL_LLM_TEST_2026-06-23_{claude,codex,gemini}_{spaCy,GLiNER}.md`

---

## 3. 오탐(FP) 분류 — 성격별 (Gemini 데이터셋 기준)

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
| 외부 3개 데이터셋 재현율(유출 방지) | **GLiNER** (claude·codex·gemini 모두 미탐↓, §2-2) |
| 정밀도 — 로그·코드 혼합(gemini) | **GLiNER** (0.82 vs 0.63 · 영문 로그 토큰 내성) |
| 정밀도 — 자연어 VOC(codex·claude) | **spaCy** (GLiNER가 ID/코드 토큰을 이름으로 오인) |

- **재현율(놓치지 않음=유출 방지)은 GLiNER가 전반 우위** — 코퍼스·외부 3종 모두에서 미탐이 더 적다.
- **정밀도(과검)는 데이터 성격에 따라 갈린다**: 영문 로그·코드가 섞인 텍스트는 GLiNER가 깨끗하고, 순수 자연어 VOC에서는 GLiNER가 주문/트랜잭션 ID를 이름으로 오인해 spaCy가 더 정밀하다.
- 보안(유출 방지) 우선이면 GLiNER, 자연어 VOC 정밀도 우선이면 spaCy 선택이 합리적.

> 데이터: 합성 NER 코퍼스(`pii_guard/corpus/ner_benchmark_corpus.py`, seed=42) + 외부 LLM 생성 6개 리포트(claude 30 · codex 10 · gemini 10 케이스 × spaCy/GLiNER). 수치는 본 문서 표에 인라인 기록(코퍼스는 `benchmarks/korean_ner_benchmark.py`로 재현 가능).
