# NuNER Zero 백엔드 후보 평가 — 동급 인코더 NER 벤치 (2026-06-29)

> **요구사항 R21 · DESIGN ADR-14**의 평가 단계 결과물이다.
> GLiNER와 유사 성능을 내는 동계열 제로샷 인코더 NER **NuNER Zero**(`numind/NuNER_Zero`, **MIT·상업 가능**)를
> 세 번째 선택형 백엔드로 배선하고, **동일 코퍼스로 GLiNER와 정량 비교**한 뒤 ADR-14 채택 게이트를 판정한다.
>
> - **엔진**: `Engine` + `Stage2NERRunner` · 백엔드 `nunerzero`(`numind/NuNER_Zero`) vs `gliner`(`urchade/gliner_multi_pii-v1`, Apache-2.0)
> - **데이터**: 합성 NER 코퍼스(`pii_guard/corpus/ner_benchmark_corpus.py`, seed=42, samples_per_format=10) · full-pipeline(Stage1+Stage2 NER)
> - **코퍼스 규모**: positive 440 · ner_clean_negatives 18 · ORG 표본 50
> - **지표**: 정탐(TP)=정답을 맞게 검출 · 오탐(FP)=PII 아닌 것을 검출 · 미탐(FN)=정답을 놓침 · precision=TP/(TP+FP) · recall=TP/(TP+FN)
> - **재현**: `python benchmarks/compare_ner_backends.py --backends gliner,nunerzero --min-confidence 0.50,0.35 --samples-per-format 10 --gate`
> - **증거물**: 대조표 [`NER_BACKEND_COMPARISON_nunerzero.md`](./NER_BACKEND_COMPARISON_nunerzero.md) · 원시 셀/게이트 [`nunerzero_compare.json`](./nunerzero_compare.json) · 카테고리 TP/FP/FN [`ner_corpus_{nunerzero,gliner}_{0.50,0.35}.json`](./)

---

## 1. 핵심 결과 (한 줄)

**게이트 FAIL ❌ → GLiNER 기본 유지.** NuNER Zero는 **정밀도(FP↓)·ADDRESS·ORG 정밀도에서 우위**지만,
보안=재현율 1순위 도구에서 결격인 **PERSON·ORG 재현율 회귀**가 임계값 0.50/0.35 양쪽에서 관찰됐다.

| 임계값 | macro-F1 (GLiNER) | macro-F1 (NuNER Zero) | 게이트 |
| :-- | --: | --: | :-- |
| 0.50 | **0.949** | 0.919 | ❌ FAIL |
| 0.35 | **0.961** | 0.943 | ❌ FAIL |

**외부 LLM 6리포트 재생성 테스트(§5)도 동일 결론** — NuNER Zero가 **현실형 데이터에서 PERSON 재현율이 더 크게
붕괴**(codex 1.00→0.20·gemini 0.80→0.20). 한국어 라벨 사용이 주요 원인으로 추정(§6).

---

## 2. 라벨 코퍼스 — 카테고리별 정탐/오탐/미탐

### 2-1. min_confidence = 0.50 (현재 기본값)

| 카테고리 | 백엔드 | precision | recall | F1 | 정탐(TP) | 오탐(FP) | 미탐(FN) |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| **PERSON** | GLiNER | 0.967 | **0.978** | **0.972** | **88** | 3 | **2** |
| | NuNER Zero | **0.987** | 0.856 | 0.917 | 77 | **1** | 13 |
| **ADDRESS** | GLiNER | 0.959 | 0.940 | 0.950 | 47 | 2 | 3 |
| | **NuNER Zero** | **0.980** | **1.000** | **0.990** | **50** | **1** | **0** |
| **ORGANIZATION** | GLiNER | 0.875 | **0.980** | **0.925** | **49** | 7 | **1** |
| | NuNER Zero | **0.909** | 0.800 | 0.851 | 40 | **4** | 10 |

### 2-2. min_confidence = 0.35 (재현율 레버 적용)

| 카테고리 | 백엔드 | precision | recall | F1 | 정탐(TP) | 오탐(FP) | 미탐(FN) |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| **PERSON** | GLiNER | 0.957 | **1.000** | **0.978** | **90** | 4 | **0** |
| | NuNER Zero | **0.977** | 0.922 | 0.949 | 83 | **2** | 7 |
| **ADDRESS** | GLiNER | 0.962 | 1.000 | 0.980 | 50 | 2 | 0 |
| | NuNER Zero | 0.962 | 1.000 | 0.980 | 50 | 2 | 0 |
| **ORGANIZATION** | GLiNER | 0.875 | **0.980** | **0.925** | **49** | 7 | **1** |
| | NuNER Zero | **0.900** | 0.900 | 0.900 | 45 | **5** | 5 |

---

## 3. 해석

### 3-1. NuNER Zero의 강점 (가설 부분 입증)
- **정밀도 전반 우위 = 오탐↓**: 0.50 기준 PERSON FP 3→**1**, ORG FP 7→**4**. 영문/일반명사 과검이 GLiNER보다 적다.
- **ADDRESS 완승**(0.50): 50/1/0 vs GLiNER 47/2/3. **토큰 분류 + `_merge_adjacent_entities` 병합 후처리**로
  "서울특별시 강남구"처럼 쪼개지는 주소 경계를 정확히 복원 → ADR-14의 "토큰분류 경계 강점" 가설 입증.
- **ORG 정밀도 개선**(0.875→0.909 @0.50, →0.900 @0.35): ADR-14 핵심 가설(임계값으로 못 고치는 ORG 과추출을
  다른 모델로 교정)이 방향상 맞음.

### 3-2. NuNER Zero의 결격 (채택 차단)
- **PERSON 재현율 회귀**: 0.50에서 미탐 2→**13**(recall 0.978→0.856). honorific 없는 맨이름을 다수 놓침.
- **ORG 재현율 회귀**: 0.50에서 미탐 1→**10**(recall 0.980→0.800). 정밀도를 얻는 대신 실제 조직을 놓침.
- 임계값을 0.35로 낮추면 회수되나(PERSON 미탐 13→7, ORG 10→5) **여전히 GLiNER(미탐 0·1)에 못 미침**.
- **유출 방지 도구는 "놓침=유출"이라 recall이 1순위**(요구사항 P2·§2.4.0). 정밀도(과잉 마스킹) 우위로는
  recall 회귀를 상쇄할 수 없다.

---

## 4. 채택 게이트 판정 (ADR-14)

게이트 기준(임계값별): **(a)** 어떤 카테고리도 GLiNER 대비 recall 회귀 없음(tolerance 0.0) **AND**
**(b)** ORG 정밀도 개선 OR macro-F1 우위. 임계값 중 하나라도 통과 시 전체 PASS.

| min_conf | recall 무회귀 | ORG 정밀도 개선 | macro-F1 우위 | 판정 |
| :-- | :-- | :-- | :-- | :-- |
| 0.50 | ✗ (PERSON −0.122 · ORG −0.180) | ✓ | ✗ (0.919 < 0.949) | ❌ |
| 0.35 | ✗ (PERSON −0.078 · ORG −0.080) | ✓ | ✗ (0.943 < 0.961) | ❌ |

**결과: 두 임계값 모두 FAIL → GLiNER(`urchade/gliner_multi_pii-v1`, Apache-2.0) 기본 유지.**

조건 (b)는 충족했으나(ORG 정밀도 개선), 조건 (a) recall 무회귀를 **두 임계값 모두에서 위반**해 탈락.

---

## 5. 외부 LLM 6리포트 재생성 테스트 (현실형 한·영 혼합)

원본 입력 JSON·하니스는 레포에 없으나, **세 GLiNER 리포트가 케이스별 텍스트·정답(ground truth)을 본문에
임베드**하고 있어 이를 파싱해 동일 50케이스(claude 30·codex 10·gemini 10)를 복원하고, **GLiNER와 NuNER Zero를
같은 하니스(`validation/external_replay.py`)로 재채점**했다. 정답은 GLiNER 리포트 기준으로 복원하므로 공정성을
위해 GLiNER도 함께 재채점한다 — **GLiNER 재채점치가 원본(`NER_BACKEND_COMPARISON.md`)과 정확히 일치**해
하니스가 검증됐다(claude 0.966/0.872 · codex 0.989/0.936 · gemini 0.931/0.918, 원본과 동일).

### 5-1. 종합 (full-pipeline · raw FP, 트리아지 없음)

| 데이터셋(케이스) | 백엔드 | 재현율 | 정밀도 | 정탐 TP | 미탐 FN | 오탐 FP |
| :-- | :-- | --: | --: | --: | --: | --: |
| **claude** (30) | GLiNER | **0.966** | 0.872 | 197 | 7 | 29 |
| | NuNER Zero | 0.863 | **0.884** | 176 | 28 | 23 |
| **codex** (10) | GLiNER | **0.989** | 0.936 | 88 | 1 | 6 |
| | NuNER Zero | 0.899 | **0.941** | 80 | 9 | 5 |
| **gemini** (10) | GLiNER | **0.931** | **0.918** | 67 | 5 | 6 |
| | NuNER Zero | 0.819 | 0.908 | 59 | 13 | 6 |

### 5-2. NER 카테고리별 재현율 — PERSON 붕괴가 결정타

| 데이터셋 | 카테고리 | GLiNER (TP/FN, R) | NuNER Zero (TP/FN, R) |
| :-- | :-- | :-- | :-- |
| claude | PERSON | 39/3 (0.93) | **23/19 (0.55)** |
| claude | ORGANIZATION | 11/0 (1.00) | 7/4 (0.64) |
| claude | ADDRESS | 25/1 (0.96) | 25/1 (0.96) |
| codex | PERSON | 10/0 (1.00) | **2/8 (0.20)** |
| codex | ADDRESS | 11/0 (1.00) | 11/0 (1.00) |
| gemini | PERSON | 8/2 (0.80) | **2/8 (0.20)** |
| gemini | ADDRESS | 3/0 (1.00) | 1/2 (0.33) |

- **현실형 데이터에서 PERSON 재현율이 코퍼스보다 더 크게 붕괴**(codex·gemini 0.20 — 이름 10개 중 8개 놓침).
  코퍼스(§2)의 PERSON recall 0.856보다 외부셋에서 훨씬 나쁘다 → **유출 방지 관점에서 치명적**.
- **ADDRESS는 병합 후처리로 claude·codex 동률**(0.96·1.00), gemini만 소표본(1/3)으로 열세.
- 정밀도는 NuNER Zero가 claude·codex에서 소폭 우위(FP↓)지만, **recall 격차를 상쇄 못 함**.
- 산출물: `EXTERNAL_LLM_TEST_2026-06-23_{claude,codex,gemini}_NuNERZero.md`(NuNER Zero 리포트) ·
  `external_replay_{ds}_{gliner,nunerzero}.json`(원시 — GLiNER 재채점치 포함) ·
  `external_cases_reconstructed.json`(복원 데이터셋). 검증용 GLiNER 재채점 .md는 원본과 중복이라 보관하지 않음
  (`validation/external_replay.py --ner-backend gliner`로 재생성 가능).

---

## 6. 알려진 한계 · 재평가 여지 (정직 선언 — P3)

- **정답은 재구성본**: 외부셋 정답을 GLiNER 리포트의 (검출∪미검출) 또는 `심은` 라인에서 역추출했다. GLiNER
  재채점치가 원본과 정확히 일치해 신뢰할 만하나, 원본 수기 트리아지(비라벨 정탐 보정)는 적용 안 한 **raw FP**다.
  세 백엔드에 동일 적용하므로 상대 비교는 공정하다.
- **라벨 = 한국어**: GLiNER와의 공정 비교를 위해 동일 한국어 라벨(`사람`·`주소`·`조직`)을 썼다.
- **런타임 예산 미측정**: 메모리·콜드로드·p95는 이 벤치 범위 밖. 채택 시 별도 검증 필요(ADR-14 조건 c).

---

## 7. 종합

| 관점 | 우위 |
| :-- | :-- |
| 코퍼스 PERSON 재현율(이름 놓침) | **GLiNER** (0.978/1.000 vs 0.856/0.922) |
| 코퍼스 ORG 재현율(조직 놓침) | **GLiNER** (0.980 vs 0.800/0.900) |
| 코퍼스 ADDRESS | **NuNER Zero** (0.990 vs 0.950 @0.50, 병합 효과) |
| 정밀도(오탐↓)·ORG 정밀도 | **NuNER Zero** (FP 전반↓, ORG P 0.909 vs 0.875) |
| macro-F1(종합) | **GLiNER** (0.961 vs 0.943 @0.35) |
| **외부 6리포트 재현율**(현실형) | **GLiNER** (claude 0.966·codex 0.989·gemini 0.931 vs 0.863·0.899·0.819) |
| **외부 PERSON 재현율** | **GLiNER** (codex/gemini 1.00/0.80 vs 0.20/0.20 — 이름 대량 누락) |
| 보안 적합성(=recall 1순위) | **GLiNER** |

- **코퍼스·외부셋 양쪽에서 GLiNER 유지가 올바른 결정.** NuNER Zero는 "과잉 마스킹은 줄이지만 더 많이 놓치는"
  트레이드오프라, 유출 방지 목적과 방향이 어긋난다. 특히 현실형 데이터의 **PERSON 재현율 붕괴**(0.20)가 결정타.
- 단 **ADDRESS·정밀도·ORG 정밀도 강점은 실재**하므로 ORG hard-negative 관점에서 향후 재평가 가치는 남는다.
  어댑터·벤치·하니스가 모두 상주하므로 `--ner-backend nunerzero`(코퍼스) / `validation/external_replay.py`(외부셋)로
  즉시 재실행 가능.

---

> 데이터: ① 합성 NER 코퍼스(seed=42, spf=10) ② 외부 LLM 6리포트 복원 50케이스. 재현 =
> `benchmarks/compare_ner_backends.py`(코퍼스 대조표+게이트) · `benchmarks/korean_ner_benchmark.py --ner-backend nunerzero`
> (코퍼스 단일) · `validation/external_replay.py --ner-backend {gliner,nunerzero}`(외부셋 재생성·검증됨).
> 관련 문서: [`NER_BACKEND_COMPARISON.md`](./NER_BACKEND_COMPARISON.md)(spaCy·GLiNER 기준선) · 요구사항 §23.2 R21 · DESIGN §20.6 ADR-14.
