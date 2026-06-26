# PII-Guard 기술 보고서 (v0.1.0)

> 버전 0.1.0 · commit `6552eec` · 2026-06-26 · 대상: 개발/아키텍처 독자(기술 상세)
> 근거 문서: [`docs/DESIGN.md`](./DESIGN.md) · [`docs/pii-guard-requirements.md`](./pii-guard-requirements.md) · [`validation/NER_BACKEND_COMPARISON.md`](../validation/NER_BACKEND_COMPARISON.md)

---

## 1. 프로젝트 개요

### 1.1 목적
PII-Guard는 **로컬 우선(local-first) LLM 게이트웨이 프록시**다. 에이전트/CLI가 외부 LLM(Claude·OpenAI·Gemini)으로 보내는 HTTP 요청을 가로채, 페이로드 속 **개인정보(PII)와 시크릿**을 **외부로 한 글자도 내보내지 않고 로컬에서** 탐지하고, 카테고리별 정책에 따라 **마스킹**(`[PERSON_1]` 치환) 또는 **차단**한 뒤 업스트림으로 중계한다. 응답이 돌아오면 로컬 세션 맵으로 플레이스홀더를 원본으로 **복원(rehydrate)**해 에이전트에 돌려준다.

핵심 불변 원칙:
- **P1 로컬 우선** — 탐지에 외부 LLM을 쓰지 않는다(정규식 + 로컬 NER). 결정적·감사가능·프롬프트 인젝션에 안전.
- **P2 secure-by-default** — 설정 0에서도 시크릿·주민번호 차단, 이름·이메일 마스킹.
- **fail-closed** — 검사 불가 시 차단, 프록시 크래시 시 무방비 직통 없음(TCP RST).
- **P4 no-raw** — Ledger(감사 원장)는 원본 없이 HMAC keyed-hash로만 기록.

탐지 카테고리는 **20종**: 시크릿(API_KEY·AWS_SECRET·GCP_KEY·TOKEN·PRIVATE_KEY·PASSWORD), 고위험 신원(RRN·FOREIGN_REG·PASSPORT·DRIVER_LICENSE·CARD), 연락·식별(EMAIL·PHONE·KR_ACCOUNT·BIZ_NO), 문맥(PERSON·ADDRESS·ORGANIZATION), 서버정보(IP_ADDRESS·HOSTNAME).

### 1.2 아키텍처

```
 ┌────────────┐      ┌──────────────────────────────────────────────┐      ┌──────────────┐
 │ LLM CLI /  │─────▶│  PII-Guard 프록시 코어 (http.server, 데몬)      │─────▶│ api.anthropic │
 │ ouroboros  │      │   _detect_provider → _scrub → _run_tripwire     │      │ /openai/gemini│
 │  에이전트   │◀─────│   block? 400 : forward(마스킹) → 응답 복원       │◀─────│   (응답)      │
 └────────────┘      │                                                │      └──────────────┘
                     │   ── Stage2 NER 워커 (별도 프로세스, spawn) ──   │
                     │      Stage2NERRunner ⇄ 워커(GLiNER 기본/spaCy)  │
                     │      OOM/timeout 시 코어 생존 → Stage1 degrade    │
                     │   ── 컨트롤 플레인(격리) ── 정책/HMAC키/Ledger/pf │
                     └──────────────────────────────────────────────┘
```

- **코어/워커 프로세스 분리(가용성)**: 경량 포워딩+Stage1은 항상 떠 있는 코어(인프로세스), 무거운 NER은 **별도 워커 프로세스**(spawn). 워커가 OOM/크래시해도 코어는 생존하고 Stage1 결과로 graceful degrade한다.
- **컨트롤/데이터 플레인 분리**: 정책·HMAC키·Ledger·egress(pf) 규칙은 통제 구역(파일 권한 600/700, pin-list out-of-band 승인). 에이전트(데이터 플레인)는 못 건드린다.
- **2-티어 강제**: 티어1(base_url 주입, 협조적 도구 대상 best-effort) + 티어2(옵트인 egress 락다운, pf 방화벽으로 우회 차단).
- 주요 모듈: `proxy.py`(인터셉트 프록시), `engine.py`(탐지 오케스트레이터), `detector.py`/`categories.py`(Stage1), `proximity.py`(Stage1.5), `stage2/`(NER 워커·백엔드), `policy.py`/`decision.py`(정책), `masker.py`/`session_map.py`(마스킹·복원), `ledger`(감사).

### 1.3 동작 Call Flow

```
1) 에이전트가 POST(/v1/messages 등) → 프록시 코어 do_POST
2) 본문 읽기 + JSON 파싱 (실패 시 400)
3) _detect_provider(path) → claude|openai|gemini (미지 경로는 스크럽 없이 통과)
4) _scrub(payload, provider)  ── 엔진 락 직렬화 ──
      └▶ Engine.scan(text):
           ① Stage1   detector.scan_text  (정규식 + 체크섬)
           ② Stage1.5 proximity.scan + merge  (문맥 게이팅 승격)
           ③ Stage2   Stage2NERRunner.scan  (GLiNER/spaCy 워커, 타임아웃/OOM 시 degrade)
           → 탐지 스팬을 masker가 [CAT_N]로 치환(같은 값=같은 토큰, SessionMap)
5) _run_tripwire(sanitized)  ── 마스킹된 전체 바디 재스윕(구조 파서 사각지대 포착)
6) should_block(스크러버 OR 트립와이어) ?  400 차단(업스트림 미전송)
                                       :  마스킹된 페이로드를 업스트림으로 forward
7) 업스트림 응답 → rehydrate([CAT_N]→원본) → 에이전트로 반환
      (스트리밍 SSE는 청크별 룩어헤드 복원으로 TTFT 보존)
```

- 차단 시 원문은 업스트림으로 **전송되지 않는다**. 마스킹 시 외부로 나가는 건 `[CAT_N]` 플레이스홀더뿐.
- `serve --log-masked`로 업스트림 전송 직전 **마스킹된 페이로드**만 콘솔 확인(원문 미출력, no-raw).

---

## 2. 핵심 엔진

### 2.1 Stage 1 · Stage 2 — 역할과 구조

| | **Stage 1 (정규식·체크섬)** | **Stage 2 (NER)** |
| :-- | :-- | :-- |
| 비유 | 바코드 스캐너 | 글을 이해하는 검사관 |
| 잡는 것 | **모양이 정해진** PII(키·카드·주민번호·이메일·전화·IP…) | **모양이 없는 문맥형** PII(이름·주소·조직) |
| 방식 | `categories.py`의 20개 패턴 + 체크섬(Luhn·RRN·사업자) | 트랜스포머/통계 NER 모델이 의미로 추출 |
| 성격 | 결정적·재현가능·인젝션 불가·경량(코어 상주) | 모델 추론(별도 워커·타임아웃·degrade) |
| 구현 | `detector.py`·`categories.py` | `stage2/runner.py`·`backend.py`·`gliner_ner.py`·`korean_ner.py` |

- **둘 다 필요한 이유**: "Stage1이 깨끗하면 NER 스킵"은 틀린 fast-path다. NER의 존재 이유가 정규식이 못 잡는 것을 잡는 것이므로, Stage1이 아무것도 못 찾은 순간이 오히려 NER이 가장 필요한 순간일 수 있다.
- **체크섬으로 FP 억제**: 무효 카드/주민번호는 일부러 미탐지(정밀도 우선).
- **Stage2 격리·degrade**: 워커 타임아웃/OOM/예외 시 Stage1 결과 반환 + `coverage_gap=True`(침묵 통과 금지, 가시화).
- **워밍업(필수)**: GLiNER 콜드 로드(~14.4s)가 블록당 타임아웃(10s)을 초과하면 매 요청이 degrade돼 이름·주소가 누출된다. `serve`는 시작 시 `Stage2NERRunner.warmup()`으로 모델을 블록 타임아웃 밖에서 1회 로드한 뒤 트래픽을 받는다.

### 2.2 Stage 1.5 — Proximity(근접 문맥) 개념·역할

**문제**: 일부 PII는 **모양이 애매**하다. 비표준 계좌번호(`302-04-918274`), 하이픈 없는 사업자번호, 라벨만 있는 비밀번호 등은 정규식으로 무조건 잡으면 주문번호·송장번호까지 오탐(FP)한다. 그래서 Stage1은 일부러 안 잡는다.

**해결(Stage 1.5)**: `proximity.py`가 Stage1 직후 실행되어, 이런 애매한 값을 **"단서 단어(트리거)가 근처에 있을 때만"** PII로 **승격(promote)**한다.
- KR_ACCOUNT — 은행명/`입금`/`계좌`가 근접 + 자릿수 9~14일 때만 (R19로 다양한 포맷 일반화).
- BIZ_NO — `사업자` 근접 + 체크섬 통과.
- PASSWORD — `비밀번호`/`비번`/`암호` 한글 라벨, `DB_PASS=` 등 접두 라벨.

성격: **규칙 기반(결정적·감사가능·인젝션 불가)**, 트리거가 `rule_id`에 기록됨. `merge()`는 containment 정책으로 겹침 정리(계좌가 내부 전화 하위오탐을 흡수). → **오탐은 억제하면서 놓침(미검출)을 줄인다.**

### 2.3 Stage 2 모델 비교 — spaCy vs GLiNER

Stage2 NER은 **선택형 백엔드**다: 기본 **`gliner`**(모델 `urchade/gliner_multi_pii-v1`, Apache-2.0·상업 가능), 경량 폴백 **`spacy`**(Presidio + `ko_core_news_lg`). 선택 = `PIIGUARD_NER_BACKEND` env 또는 정책 `stage2.ner_backend`(env 우선). 두 백엔드는 동일 카테고리(PERSON/ADDRESS/ORGANIZATION)로 정규화되어 후단(정책·마스킹·proximity 후필터·degrade)은 백엔드와 무관.

**① 라벨 코퍼스(합성·ground-truth, 순수 모델 품질)**

| 카테고리 | spaCy P / R | GLiNER P / R |
| :-- | :-- | :-- |
| PERSON | 0.974 / 0.844 | 0.935 / **0.956** |
| ADDRESS | 1.000 / 1.000 | 0.917 / 0.880 |
| ORGANIZATION | 1.000 / 0.920 | 0.774 / **0.960** |

**② 외부 LLM 생성 6개 리포트(현실형 한·영 혼합, full-pipeline, Stage1 보강 후)**

| 데이터셋 | spaCy R / P | GLiNER R / P |
| :-- | :-- | :-- |
| claude (30) | 0.941 / 0.877 | 0.966 / 0.872 |
| codex (10) | 0.921 / 0.828 | **0.989 / 0.936** |
| gemini (10) | 0.958 / 0.651 | 0.931 / **0.918** |

**해석**
- **재현율**: GLiNER가 코퍼스(PERSON 0.956·ORG 0.960)와 외부(claude·codex)에서 우위, gemini만 소폭 열세. 대체로 동등~우위.
- **정밀도**: 코퍼스에선 spaCy가 높지만(특히 ORG 1.0 vs 0.774), **현실형 데이터에선 GLiNER가 크게 우위**(codex 0.94·gemini 0.92 vs spaCy 0.83·0.65). spaCy는 영문 로그 토큰(`auth`·`webhook`)을 인물/조직으로 과잉 추출 → gemini 오탐 37 vs GLiNER 6.
- **결론**: 보안(유출 방지=recall)과 운영 품질(과잉 마스킹↓=precision) 모두에서 **기본 GLiNER(Apache)가 합리적**, 저메모리 환경은 `spacy` 폴백. 약점은 **GLiNER의 ORG 정밀도(0.774)** 한 곳에 집중.
- 라이선스: 기본 GLiNER `urchade/...`=Apache-2.0(상업 가능). 한국어 특화 `taeminlee/gliner_ko`는 성능 동등이나 **CC-BY-NC(비상업)**라 기본값 제외. spaCy `ko_core_news_lg`=CC BY-SA(런타임 다운로드만 하면 상업 OK).

### 2.4 Stage 2 recall 향상 방안 — 임계값 + 파인튜닝

미검출(FN) 분석 결과 **GLiNER 미검출의 ~79%가 NER이 아니라 정형 PII(Stage1 영역)**였다. 이미 **R19 Stage1 보강**(계좌 일반화·여권 조사경계·JWT·ghp·password)으로 외부 recall을 회수했다(codex 0.798→0.921, gemini 0.875→0.958, 정밀도 회귀 0). 남은 NER 갭은 두 레버로 개선한다.

**(a) 신뢰도 임계값 노브(무료, R20·ADR-12)** — GLiNER 스팬 점수 컷(`min_confidence`)을 낮추면:

| 임계값 | PERSON R/P | ADDRESS R/P | ORG R/P |
| :-- | :-- | :-- | :-- |
| 0.30 | 0.978 / 0.917 | **1.000** / 0.926 | 0.960 / 0.774 |
| 0.50(현재) | 0.956 / 0.935 | 0.880 / 0.917 | 0.960 / 0.774 |

→ 0.5→0.3에서 ADDRESS·PERSON recall↑·FP 거의 불변(거의 공짜). **단 ORG는 임계값 무반응** → 임계값으론 못 고침. 정책 `stage2.ner_min_confidence`로 노출 예정.

**(b) GLiNER 파인튜닝(곡선 자체 상승, R20·ADR-13)** — 임계값에 무반응인 **ORG 과추출(정밀도 0.774)**과 잔여 recall은 파인튜닝으로 **recall·정밀도 동시 개선**한다. 파일럿 구현 완료(`training/` 서브시스템, 배선 검증):
- **런타임 무변경**: 산출 모델은 기존 슬롯 `PIIGUARD_GLINER_MODEL`로 소비(코어 변경 0).
- **오프박스 학습**: `ingest`(보유 라벨 데이터 1급 입력→GLiNER 포맷) → `augment`(보조 합성: positive + ORG hard-negative) → `split`(누설 방지) → `train`(Apache 베이스 `train_model`) → `eval`(베이스↔파인튜닝 벤치마크 + 채택 게이트).
- **데이터 규모(2단계)**: 좁은 목표(ORG hard-negative) 수백~2,000 / 광역 도메인 적응 수천~수만 → 동시 대폭 향상.
- **거버넌스**: 학습셋↔평가셋 분리(누설 방지), 실 PII 오프호스트·레포 미커밋·암호화, 모델 암기 점검. Apache 베이스→산출물 상업 가능.

---

## 3. NLP 엔진 vs LLM 엔진 — 정성 비교 + 추가 검토 작업

### 3.1 왜 탐지에 NLP 인코더 NER을 쓰고 생성형 LLM을 쓰지 않는가 (DR-1)

PII-Guard는 설계상 **탐지에 생성형 LLM을 거부**한다. spaCy/GLiNER 같은 **인코더 NER**(텍스트를 *분류/추출*만, 생성 안 함)과 Claude·Codex·EXAONE 같은 **생성형 LLM**을 탐지기로 봤을 때:

| 관점 | NLP 인코더 NER (spaCy·GLiNER) | 생성형 LLM (Claude·Codex·EXAONE) |
| :-- | :-- | :-- |
| **로컬성(P1)** | ✅ 완전 로컬 | 외부 API LLM은 **PII를 외부로 전송 = 막으려는 유출을 자행(자기모순)**. 로컬 LLM만 가능 |
| **프롬프트 인젝션** | ✅ 생성 안 함 → *"이건 PII 아님"* 주입 **불가** | 🔴 비신뢰 콘텐츠가 탐지기를 무력화(위협모델 R13에서 공격면 확대) |
| **결정성·재현성** | ✅ 같은 입력=같은 결과(감사·테스트·Ledger 재현 용이) | ⚠️ 비결정적 → "거짓 안심"(P3) 위험 |
| **체크섬 검증** | ✅ 카드 Luhn·주민번호·사업자 산술 검증과 결합 | 산술 검증 불가/불안정 |
| **메모리·지연** | 수백MB급, ms~수백ms | 로컬 7B 양자화 ~4~5GB(8GB 예산 초과), 지연 큼 |
| **감사 설명가능성** | ✅ rule_id·신뢰도·단계 기록 | 약함 |

> **결론**: 보안 탐지기에 생성형 LLM은 퇴보다. "더 똑똑한 탐지"의 정답은 **로컬 인코더 NER 모델 교체/파인튜닝**(이미 설계된 업그레이드 경로)이며, 규칙은 고심각·체크섬 카테고리에서 1급으로 유지한다. (LLM을 *탐지기*로 실측하려면 실 PII를 외부로 보내야 해 P1 위반이므로, 본 비교는 정성 평가로 한정한다.)

참고로 외부 LLM(Codex·Gemini 등)은 본 프로젝트에서 **탐지기가 아니라 "테스트 데이터 생성기"**로 활용했다(다양한 현실형 VOC/로그 생성 → 6개 검증 리포트). 이는 P1과 충돌하지 않는다(합성 데이터).

### 3.2 LG EXAONE 등 — 향후 평가 후보

EXAONE(LG AI연구원)·KoELECTRA·KLUE-RoBERTa 등 한국어 특화 인코더/트랜스포머는 **로컬 NER 백엔드 후보**로서 의미가 있다(생성형이 아닌 *인코더* 용도로 한정 시 P1·인젝션 요건 충족). 현 아키텍처는 백엔드를 추상화(`resolve_ner_backend`·`PIIGUARD_GLINER_MODEL` 슬롯)했으므로, 이런 모델을:
- (a) GLiNER 백엔드의 **베이스 모델 교체**(한국어 특화 파인튜닝 베이스)로, 또는
- (b) **신규 백엔드**(토큰 분류 NER)로
추가 평가할 수 있다. 단 라이선스(상업 가능 여부)·메모리·정밀도 실측이 선행돼야 하며, 현재는 **향후 평가 후보**로만 위치시킨다.

### 3.3 precision·recall 향상 — 추가 검토 작업

| 우선 | 작업 | 대상 지표 | 상태 |
| :-- | :-- | :-- | :-- |
| 1 | **신뢰도 임계값 노브**(`stage2.ner_min_confidence`, 0.5→0.35) | recall↑(ADDRESS·PERSON), FP 불변 | 설계(R20·ADR-12) — 구현 대상 |
| 2 | **GLiNER 파인튜닝**(ORG hard-negative + 잔여 recall) | **ORG 정밀도↑** + recall 유지 | 파일럿 구현 완료(`training/`), 실학습은 사용자 환경 |
| 3 | **음성 proximity 강화**(`ner_filters` deny-list) | 코드/로그 토큰 과검↓(특히 spaCy 폴백) | 구현됨(R17), 확장 여지 |
| 4 | **양성 proximity 확장**(잔여 PASSWORD 라벨 등) | recall↑(정형 PII) | R17/R19, 잔여 갭 |
| 5 | **transformer/EXAONE 베이스 평가** | 한국어 정확도 천장↑ | 향후 후보(§3.2) |
| 6 | **per-category 임계값**(ADDRESS 공격적, ORG 보수적) | 카테고리별 미세조정 | 후속(전역 노브 이후) |

**권고 순서**: 임계값(즉시·무료) → 잔차 측정 → 파인튜닝(ORG) → 필요 시 베이스 모델 평가. 모든 변경은 `benchmarks/korean_ner_benchmark.py` + 외부 6리포트로 **사전/사후 회귀 검증** 후 채택한다.

---

## 부록 — 검증 재현
- 코퍼스 벤치마크: `benchmarks/korean_ner_benchmark.py --ner-backend {spacy,gliner}`
- 외부 리포트: `validation/EXTERNAL_LLM_TEST_2026-06-23_{claude,codex,gemini}_{spaCy,GLiNER}.md`
- 종합 비교/임계값 스윕: `validation/NER_BACKEND_COMPARISON.md`
- Stage1 보강: `validation/STAGE1_RECALL_IMPROVEMENT_2026-06-25.md`
- 파인튜닝 파일럿: `training/` (README·schema 참조)
- 전체 테스트: 2736 passed / 12 skipped / 0 failed
