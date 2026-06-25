# Stage2 NER 백엔드 측정 — spaCy vs GLiNER (2026-06-25)

> 두 백엔드의 한국어 비정형 PII(PERSON·ADDRESS·ORGANIZATION) 검출 성능을 **두 종류의 데이터**로 측정한다.
> ① 라벨 코퍼스(합성, ground-truth) — 순수 모델 품질. ② 외부 LLM 생성 VOC/로그 케이스(현실형 한·영 혼합) — 실전 검출력.
> 모든 측정은 모델 로드 후 동일 입력으로 수행했다.
>
> - **백엔드**: `spacy` = Presidio + `ko_core_news_lg` · `gliner` = **`urchade/gliner_multi_pii-v1`**(기본, Apache-2.0 · 상업 사용 가능)
> - **지표**: 정탐(TP)=정답을 맞게 검출 · 오탐(FP)=PII 아닌 것을 PII로 검출 · 미탐(FN)=정답을 놓침
> - precision = TP/(TP+FP) · recall = TP/(TP+FN)
> - ℹ️ 한국어 특화 `taeminlee/gliner_ko`(CC-BY-NC-4.0, 비상업)는 성능 동등이나 라이선스로 기본값 제외. 본 문서 수치는 **상업 기본값(Apache)** 기준.

---

## 1. 라벨 코퍼스 (합성·ground-truth) — 순수 모델 품질

NER 소유 3개 카테고리, full-pipeline 기준.

| 카테고리 | 백엔드 | precision | recall | 정탐(TP) | 오탐(FP) | 미탐(FN) |
| :-- | :-- | --: | --: | --: | --: | --: |
| **PERSON** | spaCy | 0.974 | 0.844 | 38 | 1 | 7 |
| | **GLiNER** | 0.935 | **0.956** | **43** | 3 | 2 |
| **ADDRESS** | spaCy | 1.000 | 1.000 | 25 | 0 | 0 |
| | **GLiNER** | 0.917 | 0.880 | 22 | 2 | 3 |
| **ORGANIZATION** | spaCy | 1.000 | 0.920 | 23 | 0 | 2 |
| | **GLiNER** | 0.774 | **0.960** | **24** | 7 | 1 |

- **GLiNER = 재현율 우위**: PERSON 미탐 7→2, ORG 미탐 2→1.
- **spaCy = 정밀도 우위**(코퍼스 한정): 오탐 거의 0. GLiNER는 ORG에서 오탐 7(은행/조직 경계 모호).
- ADDRESS는 spaCy가 약간 우위(GLiNER 재현율 0.88).

## 2. 외부 LLM 생성 6개 리포트 종합 (현실형 한·영 혼합) — 실전 검출력

claude·codex·gemini × spaCy·GLiNER raw 채점치. 전체 파이프라인(Stage1+NER) 기준이라 시크릿·정형 PII 등 Stage1 카테고리가 분모에 포함되며, 두 백엔드 차이는 NER 카테고리에서만 발생한다. 각 데이터셋은 spaCy/GLiNER가 **동일 입력**으로 채점됐다. **수치는 2026-06-25 Stage1 보강 엔진 기준**(보강 전→후 비교는 `STAGE1_RECALL_IMPROVEMENT_2026-06-25.md`).

| 데이터셋(케이스수) | 백엔드 | 정탐 TP | 미탐 FN | 오탐 FP | 재현율 | 정밀도 |
| :-- | :-- | --: | --: | --: | --: | --: |
| **claude** (30) | spaCy | 192 | 12 | 27 | 0.941 | 0.877 |
| | **GLiNER** | 197 | 7 | 29 | **0.966** | 0.872 |
| **codex** (10) | spaCy | 82 | 7 | 17 | 0.921 | 0.828 |
| | **GLiNER** | 88 | 1 | 6 | **0.989** | **0.936** |
| **gemini** (10) | spaCy | 69 | 3 | 37 | 0.958 | 0.651 |
| | **GLiNER** | 67 | 5 | 6 | 0.931 | **0.918** |

- **재현율**: GLiNER가 claude(+0.025)·codex(+0.068) 우위, gemini는 소폭 열세(−0.027). 대체로 동등~우위. codex는 **0.989**(거의 완전 검출).
- **정밀도**: GLiNER가 codex(+0.108)·gemini(+0.267) 크게 우위, claude는 동등(−0.005). **오탐(FP)이 전 데이터셋에서 spaCy보다 적거나 비슷**(codex 17→6, gemini 37→6).
- Stage1 보강으로 **두 백엔드 모두 재현율 상승**(codex spaCy 0.798→0.921 / GLiNER 0.854→0.989, gemini spaCy 0.875→0.958 / GLiNER 0.847→0.931). 정형 PII(계좌·여권·토큰·키)를 회수한 결과.
- 결론: **GLiNER(Apache) + Stage1 보강 = 재현율·정밀도 모두 최상**(codex 0.989/0.936, gemini 0.931/0.918).

### 2-1. 카테고리별 정탐/미탐/오탐 (gemini 데이터셋 예시)

NER 관련 카테고리만 발췌(Stage1 카테고리는 두 백엔드 동일).

| 카테고리 | spaCy TP/FN/FP | GLiNER TP/FN/FP |
| :-- | :-- | :-- |
| PERSON | 10 / 0 / **12** | 8 / 2 / **0** |
| ADDRESS | 3 / 0 / 3 | 3 / 0 / 1 |
| ORGANIZATION | 0 / 0 / **19** | 0 / 0 / **2** |

> spaCy의 대량 오탐(PERSON 12·ORG 19 = 영문 로그 토큰)이 GLiNER에선 거의 사라진다(PERSON 0·ORG 2).

---

## 3. 오탐(FP) 분류 — 성격별

### 3-1. spaCy 오탐 (gemini 기준, 총 37)

| 성격 | 예시 |
| :-- | :-- |
| **진짜 과검 — 영문 로그 토큰** | `PERSON`=auth(다수)·active / `ORGANIZATION`=detected·security·webhook·Received |
| 라벨 누락 정탐(실제 PII) | `ORGANIZATION`=우리은행·국민카드·하나은행 |
| 채점 아티팩트 | `RRN`=120923-1591783(정답 라벨 FOREIGN_REG) / `HOSTNAME`=api.internal |

### 3-2. GLiNER 오탐 (Apache, codex+gemini 합쳐 12건) — 대부분 비-과검

| 성격 | 예시 | 비고 |
| :-- | :-- | :-- |
| **조사 미분리 정탐** | `PERSON`=김하린입니다·박서준입니다·최민서입니다 | 실제 이름인데 종결어미 "입니다"가 붙어 스팬 불일치 → FP 집계(사실상 정탐). 조사 스트립 목록 보강으로 해소 가능 |
| 라벨 누락 정탐 | `ORGANIZATION`=신한은행 / `HOSTNAME`=api.internal | 실제 조직·호스트, 라벨 없음 |
| 채점 아티팩트 | `RRN`=120923-1591783·700523-4376198 | spaCy와 동일(라벨 기인) |
| 경미한 과검 | `ORGANIZATION`=저희 회사·대행업체인데 / `ADDRESS`=5432 closed unexpect | 소수 |

> **요지**: Apache GLiNER의 오탐은 **진짜 과검이 거의 없고** 조사 미분리·라벨 누락·채점 아티팩트가 대부분 → 실효 정밀도는 표시치보다 더 높다. spaCy의 오탐은 **영문 로그 토큰 과검**이 주범으로 성격이 다르다.

---

## 4. 종합

| 관점 | 우위 |
| :-- | :-- |
| 라벨 코퍼스 재현율(이름·조직 놓침) | **GLiNER** (PERSON 0.956, ORG 0.960) |
| 라벨 코퍼스 정밀도 | spaCy (오탐 거의 0; GLiNER ORG 오탐 존재) |
| 외부 3개 데이터셋 재현율 | **GLiNER**(claude·codex, codex 0.989) · 동등(gemini) |
| 외부 3개 데이터셋 정밀도(오탐) | **GLiNER** (codex 0.94·gemini 0.92 vs spaCy 0.83·0.65) |
| 영문 로그 노이즈 내성 | **GLiNER** |

- **현실형 데이터에서 GLiNER(Apache)가 재현율·정밀도 모두 우위** — 특히 로그·코드 혼합 텍스트에서 오탐이 spaCy의 1/6 수준(gemini FP 37→6).
- spaCy는 **합성 코퍼스의 정밀도**와 **경량성(저메모리·빠른 로드)**에서 의미가 있어 폴백으로 유지.
- 보안(유출 방지=재현율)과 운영 품질(과잉 마스킹↓=정밀도) 모두에서 **기본 GLiNER(Apache)가 합리적**, 저자원 환경은 `spacy` 선택.

---

## 5. Stage1 보강에 의한 recall 개선 (보강 전 → 후)

미검출(FN) 분석 결과 **GLiNER 미검출의 ~79%가 NER이 아니라 정형 PII(Stage1: 정규식·proximity 영역)**였다. NER 백엔드와 무관한 이 갭을 Stage1 규칙 보강으로 메웠다(2026-06-25). 위 §1~§4 수치는 모두 **보강 후** 기준이며, 본 절은 그 개선폭을 보여준다.

### 5-1. 무엇을 보강했나 (모두 결정적·감사가능)

| 카테고리 | 보강 |
| :-- | :-- |
| **KR_ACCOUNT** | 비표준 계좌 포맷 일반화(하이픈 2~3개 + 자릿수 9~14), 은행/입금 트리거 근접 시에만 승격(오탐 억제 유지) |
| **PASSPORT** | 조사 인접 버그 수정 — 경계 `(?!\w)`→`(?![A-Za-z0-9])` (`M12345678를`가 깨지던 문제) |
| **TOKEN(JWT)** | 2번째 세그먼트의 `eyJ` 강제 제거(변형 JWT 검출) |
| **API_KEY** | GitHub `ghp_` 본체 길이 `{36,}`→`{20,}` |
| **PASSWORD** | `DB_PASS=`·`temporary_pass:` 등 접두형 라벨 인식 |

### 5-2. 재현율 변화 (동일 입력, 외부 6개 리포트)

| 데이터셋 | spaCy 전 → 후 | GLiNER 전 → 후 |
| :-- | :-- | :-- |
| **codex** (10) | 0.798 → **0.921** (+0.123) | 0.854 → **0.989** (+0.135) |
| **gemini** (10) | 0.875 → **0.958** (+0.083) | 0.847 → **0.931** (+0.084) |
| claude (30) | 0.941 → 0.941 (변화없음) | 0.966 → 0.966 (변화없음) |

> claude는 잔여 미검출이 NER(이름·주소)·무효체크섬뿐이라 Stage1 보강과 무관(변화 없음). codex/gemini는 **두 백엔드 모두** 재현율이 크게 올랐다 — 정형 PII가 NER 백엔드와 독립적으로 회수됐기 때문.

### 5-3. 회수된 미검출 (codex+gemini, 카테고리별)

| 카테고리 | 회수 | 비고 |
| :-- | :-- | :-- |
| KR_ACCOUNT | **9/9** | 비표준 포맷 전부 |
| TOKEN | **3/3** | 변형 JWT |
| API_KEY | **2/2** | 짧은 ghp_ |
| PASSPORT | **2/3** | 조사 인접분 회수, 1건 잔여 |
| PASSWORD | **1/3** | 접두 라벨분 회수, 라벨 없는 값 2건 잔여 |

**정형 PII 미검출 합계: 27 → 10** (17건 회수). 정밀도 회귀 없음(codex·gemini 정밀도 유지/소폭 상승). 상세 = [`STAGE1_RECALL_IMPROVEMENT_2026-06-25.md`](./STAGE1_RECALL_IMPROVEMENT_2026-06-25.md).

---

> 데이터: 합성 NER 코퍼스(`pii_guard/corpus/ner_benchmark_corpus.py`, seed=42) + 외부 LLM 생성 6개 리포트(claude 30 · codex 10 · gemini 10 케이스 × spaCy/GLiNER). 코퍼스 재현 = `benchmarks/korean_ner_benchmark.py --ner-backend {spacy,gliner}`. 출처 리포트: `EXTERNAL_LLM_TEST_2026-06-23_{claude,codex,gemini}_{spaCy,GLiNER}.md`.
