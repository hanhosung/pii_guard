# Stage2 NER 백엔드 비교 — spaCy vs GLiNER (2026-06-25)

> 두 백엔드(`spacy`=Presidio+ko_core_news_lg, `gliner`=taeminlee/gliner_ko)를 **두 종류의 데이터**로 비교했다.
> ① 라벨 코퍼스(합성, ground-truth) — 순수 모델 품질. ② 외부 LLM VOC 케이스(현실형 한·영 혼합) — 실전 검출력.
> 하니스: `benchmarks/korean_ner_benchmark.py --ner-backend`, `validation/load_external_test.py`.

---

## 0. 한눈에 (결론)

- **모델 품질만 보면 GLiNER가 우위**: 깨끗한 한국어에서 PERSON 재현율 0.84→**0.978**, 현실형 혼합 텍스트에서 정밀도 0.63→**0.82**(오탐 37→14).
- **실제 프록시(서브프로세스) 경로에 치명적 통합 결함이 있었음**: GLiNER 콜드 모델 로드(~14.4s)가 Stage2 **블록당 기본 타임아웃(10s)을 초과** → 매 요청이 타임아웃 degrade → GLiNER NER이 한 번도 기여하지 못하고 이름/주소가 Stage1로 새어나감(§3).
- **→ ✅ 해결**: serve 시작 시 워커를 미리 로드하는 **워밍업(`Stage2NERRunner.warmup()`)** 도입으로, 콜드로드를 블록 타임아웃 밖에서 끝낸 뒤 트래픽을 받게 함(§4). 폴백(spaCy 자동 전환)은 미도입으로 결정.

---

## 1. 라벨 코퍼스 (합성·ground-truth, in-process 워밍업) — 순수 모델 품질

| 카테고리 | spaCy P | spaCy R | GLiNER P | GLiNER R |
| :-- | --: | --: | --: | --: |
| PERSON | 0.974 | 0.844 | 0.800 | **0.978** |
| ADDRESS | 1.000 | 1.000 | 1.000 | 1.000 |
| ORGANIZATION | 1.000 | 0.920 | 0.862 | **1.000** |

- GLiNER: **재현율↑**(PERSON +0.13, ORG +0.08), **정밀도↓**(PERSON −0.17, ORG −0.14).
- 정밀도 하락 원인: 주소 분절(`전라`/`좌수영`), 전화번호를 PERSON으로 오인 등.
- 리포트: `validation/gliner_benchmark.json`, `validation/spacy_benchmark.json` (둘 다 전 임계값 통과).

## 2. 외부 LLM VOC 케이스 (현실형 한·영 혼합, 전체 파이프라인) — 실전 검출력

> ⚠️ **공정성 주의**: GLiNER를 기본 설정(10s 타임아웃)으로 돌리면 콜드로드 타임아웃으로 **NER 전부 0**이 된다(§3).
> 아래 GLiNER 수치는 **워밍업 + 60s 타임아웃**으로 그 결함을 우회해 모델 본연의 검출력만 측정한 값이다.

| 지표 | spaCy | GLiNER (공정) | GLiNER (기본 10s, 결함) |
| :-- | --: | --: | --: |
| 재현율 | 0.875 (63/72) | 0.861 (62/72) | 0.694 (50/72) |
| 정밀도 | 0.630 (63/100) | **0.816 (62/76)** | 0.926 |
| 오탐(FP) | **37** | **14** | 4 |
| PERSON | 10/10 | 10/10 | **0/10** ← degrade |
| ADDRESS | 3/3 | 2/3 | **0/3** ← degrade |

- **현실형 데이터에서는 GLiNER가 명확히 우수**: 재현율 거의 동일(0.861 vs 0.875)인데 **정밀도 0.82 vs 0.63**, 오탐 14 vs 37.
- spaCy의 오탐 37건은 알려진 약점 — 영문 로그 토큰(`auth`·`webhook`·`active`)을 인물/조직으로 과잉 추출. GLiNER는 이 노이즈에 강함.
- Stage1 카테고리(EMAIL·CARD·RRN·시크릿 등)는 두 백엔드 동일(차이는 NER 카테고리에서만 발생).

---

## 3. 🔴 치명적 발견 — GLiNER 콜드로드 vs Stage2 타임아웃 (실제 경로 결함)

`Stage2NERRunner`는 워커 응답에 **블록당 하드 타임아웃(기본 `DEFAULT_TIMEOUT=10.0s`)**을 건다. GLiNER **첫 호출 모델 로드는 ~14.4s**.

검증(가드 있는 실제 .py, `taeminlee/gliner_ko`):

| 타임아웃 | 콜드 결과 | 검출 |
| :-- | :-- | :-- |
| 10s (기본) | `Stage2NERTimeout: worker did not respond within 10.0s` → **degrade** | `[]` |
| 40s | 콜드 14.4s **성공** | PERSON 김철수, ADDRESS 서울특별시·강남구 |
| 40s (웜) | 0.09s | PERSON 홍길동, ADDRESS 부산광역시 |

- 타임아웃 시 워커가 **kill**되므로, 다음 요청은 다시 콜드로드→다시 타임아웃 → **영원히 웜업되지 못함**.
- 결과: **기본 백엔드가 GLiNER인데도 실제 serve 경로에서는 매 요청이 Stage1로 degrade** → 이름·주소가 마스킹되지 않고 통과(coverage_gap만 기록). **보안상 중대**(기본값이 조용히 무력화).
- spaCy는 로드가 더 빨라(≈수 초 < 10s) 이 경로에서 정상 동작 → 현재 **실제로 NER이 작동하는 건 spaCy뿐**.

---

## 4. 조치 / 결정

1. **✅ (완료) GLiNER 워밍업 분리** — `Stage2NERRunner.warmup()` 추가 + `serve`가 시작 시 호출. 모델 로드를 **블록 타임아웃 밖**(관대한 1회 예산 `WARMUP_TIMEOUT=90s`)에서 끝낸 뒤 트래픽을 받는다.
   - 검증: warmup 13.6s 후, 기본 10s 타임아웃에서 scan **0.15s 성공**(PERSON 김철수·ADDRESS 서울특별시/강남구, degrade 없음). §3의 결함 해소.
2. **폴백 정책 — 현행 유지로 결정(spaCy 자동 폴백 미구현)** — Stage2 실패 시 기존대로 **Stage1로 degrade + coverage_gap 가시화**를 유지한다. (워밍업으로 콜드로드 타임아웃이 사라져 상시 degrade 문제는 해소됨. `on_ner_backend_unavailable` spaCy 폴백은 도입하지 않음 — 의사결정 기록.)
3. **기본 백엔드 = GLiNER 유지** — 모델 품질(코퍼스 재현율)·현실형 정밀도 모두 GLiNER 우위이고, 워밍업(1)으로 실제 경로에서 정상 동작함이 확인됨. 정밀도보다 재현율이 덜 중요한(또는 저자원) 환경은 `spacy` 백엔드 선택.

> 데이터 출처: Codex/Gemini 생성 외부 케이스 10건, 합성 NER 코퍼스(`pii_guard/corpus/ner_benchmark_corpus.py`, seed=42). 수치는 본 문서 표에 인라인 기록(원천 산출물은 `benchmarks/korean_ner_benchmark.py --ner-backend {spacy,gliner}`로 재현 가능).
