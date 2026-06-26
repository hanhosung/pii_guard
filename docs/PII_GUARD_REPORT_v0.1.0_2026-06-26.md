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

#### 2.4.0 먼저 — recall을 올린다는 게 무슨 뜻인가
recall(재현율) = **놓치지 않은 비율** = `정탐(TP) ÷ (정탐 + 미탐(FN))`. 즉 **recall을 올린다 = 미탐(FN, 놓친 PII)을 줄인다**. PII-Guard는 유출 방지 도구라 "놓침"이 곧 유출이므로 recall이 1순위 지표다.

그러면 "무엇을 놓치는가"부터 봐야 올바른 레버를 고른다.

#### 2.4.1 핵심 진단 — 놓치는 것의 79%는 NER이 아니라 "정형 PII"
외부 6개 리포트의 미검출(FN)을 카테고리별로 분류한 결과:

| 영역 | 비중 | 예시 | 고치는 도구 |
| :-- | :-- | :-- | :-- |
| **정형 PII (Stage1 영역)** | **~79%** | 비표준 계좌·여권·JWT·API키·비밀번호 | 정규식 / proximity (NER 무관) |
| NER 영역 | ~21% | 이름(PERSON)·주소(ADDRESS) | NER 임계값 / 파인튜닝 |

→ **GLiNER를 아무리 손봐도 79%(정형 PII)는 안 잡힌다.** 그래서 recall 향상은 **두 갈래**로 나눠 접근한다.

#### 2.4.2 (이미 적용됨) Stage1 정형 PII 보강 — R19
가장 큰 갭(정형 PII)은 **결정적 규칙 보강**으로 이미 회수했다. 모델 학습이 아니라 정규식/proximity 수정이라 즉시·안전하다.

| 보강 | 내용 | 효과 |
| :-- | :-- | :-- |
| KR_ACCOUNT | 비표준 계좌 포맷 일반화(하이픈 2~3개+자릿수 9~14), 은행/입금 트리거 근접 시에만 승격 | 미검출 9/9 회수 |
| PASSPORT | 조사 인접 버그(`M12345678를`가 `\w` 경계로 깨짐) → `(?![A-Za-z0-9])` | 2/3 회수 |
| TOKEN(JWT) | 2번째 세그먼트 `eyJ` 강제 제거 → 변형 JWT도 검출 | 3/3 |
| API_KEY | GitHub `ghp_` 길이 `{36}`→`{20}` | 2/2 |
| PASSWORD | `DB_PASS=`·`temporary_pass:` 접두 라벨 | 1/3 |

결과: 외부 recall **codex 0.798→0.921 · gemini 0.875→0.958**, **정밀도 회귀 0**(정형 PII 미검출 27→10).

#### 2.4.3 (레버 a) 신뢰도 임계값 — "거의 공짜" recall (NER 영역)
GLiNER는 각 후보에 **점수(0~1)**를 매기고 임계값(`min_confidence`, 현재 0.50) 미만은 버린다. 이 커트라인을 **낮추면** 애매하게 잡힌 것까지 채택돼 놓침이 준다.

| 임계값 | PERSON R / P | ADDRESS R / P | ORG R / P |
| :-- | :-- | :-- | :-- |
| **0.30** | 0.978 / 0.917 | **1.000** / 0.926 | 0.960 / 0.774 |
| 0.50(현재) | 0.956 / 0.935 | 0.880 / 0.917 | 0.960 / 0.774 |

- **이득**: 0.50→0.30에서 ADDRESS 0.88→**1.00**, PERSON 0.956→0.978. **오탐(FP)은 거의 안 늘어** → "거의 공짜로 놓침을 줄이는" 구간.
- **한계**: **ORG는 임계값에 전혀 반응 안 함**(R 0.960·P 0.774 고정). 조직 과추출은 점수 컷 문제가 아니라 **모델이 경계를 잘못 잡는 특성**이라 → 임계값으론 못 고친다(→ 레버 b의 몫).
- **비유**: 검사 기준을 느슨하게 잡는 것. 깐깐하면(높은 임계값) 애매한 진짜를 놓치고, 느슨하면(낮은 임계값) 더 잡지만 헛잡음이 는다. 다만 GLiNER는 0.3~0.5 구간에서 헛잡음이 거의 안 늘어 **낮추는 게 유리**.
- **적용 방법**: 코드 하드코딩(0.50)을 정책 노브 `stage2.ner_min_confidence` / env `PIIGUARD_NER_MIN_CONFIDENCE`로 노출(전역, env>정책>0.50). 권장 운영점 ~0.35. (설계 R20·ADR-12 — 구현 대상)

> ⚠️ "카테고리 규칙 신뢰도(`CategoryPolicy.min_confidence`)"와는 **다른 값**이다. 전자는 Stage1 규칙 필터, 여기서 말하는 건 NER 모델의 스팬 점수 컷이다.

#### 2.4.4 (레버 b) GLiNER 파인튜닝 — "곡선 자체를 올리는" 유일한 방법
임계값은 *고정된 recall↔precision 곡선* 위에서 한쪽을 올리면 다른 쪽이 내려가는 **트레이드오프**다. 반면 **파인튜닝(추가 학습)**은 모델이 아는 것을 바꿔 **곡선 자체를 위로 올린다** → recall과 precision을 **동시에** 개선할 수 있다(특히 임계값에 무반응인 **ORG 정밀도 0.774**).

**무엇을 가르치나 (데이터 구성)**
- **positive(재현율↑)**: 다양한 한국어 이름·주소·조직을 문맥 속에 라벨링 → 놓치던 표면형을 학습.
- **hard-negative(정밀도↑, ORG 핵심)**: GLiNER가 조직/인물로 오인하는 것(`저희 회사`, `ORD-2026` 같은 ID·코드)을 **"엔티티 아님"으로** 라벨 → "이건 PII 아님"을 학습.

**파이프라인 (파일럿 구현 완료 — `training/` 서브시스템, 배선 검증됨)**
```
보유 라벨 데이터({text,spans})           ← 1급 입력
   └(+선택) augment 합성 보강(positive·hard-negative)
ingest  → GLiNER 포맷(문자-span→토큰-span 변환·검증)
split   → train/val/test (평가셋 누설 방지)
train   → Apache 베이스 urchade/gliner_multi_pii-v1 학습 (train_model)
eval    → 베이스↔파인튜닝 벤치마크 비교 + 채택 게이트
배포    → PIIGUARD_GLINER_MODEL=<파인튜닝 경로>  (런타임 코드 변경 0)
```

**핵심 설계 포인트**
- **런타임 무변경**: 파인튜닝 모델은 기존 모델 슬롯으로 갈아끼우기만 하면 됨(코어/워커 변경 없음).
- **데이터 규모 2단계**: 좁은 목표(ORG hard-negative) **수백~2,000** / 광역 도메인 적응(보유 **수천~수만**) → PERSON·ADDRESS·ORG recall·정밀도 동시 대폭 향상.
- **채택 게이트**: ① 전 임계값 통과 ② 어떤 카테고리도 recall 회귀 없음 ③ 목표(ORG 정밀도) 개선일 때만 기본 모델로 승격.
- **거버넌스(필수)**: 학습셋↔평가셋 분리(누설 방지), 실 PII는 오프박스 호스트에서만·레포 미커밋·암호화(P4), 배포 전 모델 암기 점검. 베이스 Apache-2.0 → 산출물도 상업 사용 가능.
- **상태**: 도구·배선·게이트는 검증 완료(1스텝 스모크 OK). **본 학습(실데이터·GPU)은 사용자 환경에서 수행**.

#### 2.4.5 권고 순서 (ROI)
1. **(완료) Stage1 보강(R19)** — 정형 PII 79%, 결정적·즉시·정밀도 회귀 0.
2. **임계값 노브(레버 a)** — NER 영역 무료 recall(ADDRESS·PERSON), FP 불변.
3. **(잔차 측정)** — 1·2 적용 후 남는 미검출·ORG 오탐 재측정.
4. **파인튜닝(레버 b)** — 임계값으로 안 되는 ORG 정밀도 + 잔여 recall.

> 한 줄: **놓침의 79%(정형)는 규칙으로(완료), NER의 쉬운 부분은 임계값으로(무료), 임계값으로 안 되는 ORG는 파인튜닝으로** — 순서대로 가는 게 비용 대비 최적이다.

---

## 3. NLP 엔진 vs LLM 엔진 — 탐지 엔진 역할 분담 (DR-1)

핵심 원칙: **탐지(런타임)는 NLP 인코더 NER이 맡고, LLM(로컬 EXAONE 포함)은 탐지기가 아니라 그 인코더를 더 잘 학습시키는 "오프라인 데이터·평가 엔진"으로 쓴다.** 둘은 경쟁이 아니라 **역할 분담**이다.

### (1) 탐지기 = NLP 인코더 NER — 왜 LLM을 *인라인 탐지기*로 안 쓰나
spaCy/GLiNER 같은 **인코더 NER**(텍스트를 *분류/추출*만, 생성 안 함)과 생성형 LLM(Claude·Codex·EXAONE)을 **탐지기**로 비교하면:

| 관점 | NLP 인코더 NER (spaCy·GLiNER) | 생성형 LLM (Claude·Codex·EXAONE) |
| :-- | :-- | :-- |
| **로컬성(P1)** | ✅ 완전 로컬 | 외부 API는 PII를 외부로 전송 = 자기모순. **로컬(EXAONE)이면 이 항목은 해결** |
| **프롬프트 인젝션** | ✅ 생성 안 함 → *"이건 PII 아님"* 주입 **불가** | 🔴 **로컬이어도 동일** — 비신뢰 콘텐츠가 탐지기를 무력화(targeted recall→0). **결정타** |
| **지연(latency)** | 수백MB급, ms~수백ms | 요청당 수백ms~수초 → 프록시 핫패스·p95<800ms 예산 위반 |
| **결정성·재현성** | ✅ 같은 입력=같은 결과(감사·Ledger 재현) | ⚠️ 비결정(temp=0로 완화되나 양자화/버전 드리프트) → "거짓 안심"(P3) |
| **정확 추출·체크섬** | ✅ 정확 스팬 + 카드 Luhn·주민번호 산술 검증 | 🔴 긴 토큰(카드·키) 환각/오타, 산술검증 불가, 정확 문자 스팬 불가(마스킹 어긋남) |
| **메모리(8GB)** | 수백MB | EXAONE-2.4B ~2~3GB(빠듯), 7.8B ~5GB(초과 위험) |

→ **로컬 EXAONE는 P1을 풀어 "검토 가치 있는 후보"가 되지만, 인젝션·지연·정확추출 때문에 라이브 단독 1차 탐지기로는 여전히 부적합**하다. 인젝션은 로컬로도 안 풀리는 본질적 결격이다.

### (2) 로컬 EXAONE의 최선 활용 = "오프라인 데이터 엔진" (precision·recall을 실제로 올림)
EXAONE를 *탐지기*가 아니라 **GLiNER 파인튜닝(§2.4·ADR-13)을 먹여 살리는 오프라인 도구**로 쓰면, 결정타(인젝션·지연)를 피하면서 지표를 올린다. 산출물은 결정적·인젝션-불가한 **인코더(GLiNER)**라 보안 성질도 유지된다.

| 역할 | 무엇 | 효과 |
| :-- | :-- | :-- |
| **학습 데이터 라벨링·생성** | EXAONE로 한국어 문장 생성 + PERSON/ADDRESS/ORG 라벨 부여 | "합성-only 제약상 데이터 어디서?"의 답 — **로컬이라 외부 유출 없이** 확보 (`training/augment.py` 확장) |
| **hard-negative 발굴** | "조직처럼 보이지만 아닌" 표현 다양화 | GLiNER **ORG 정밀도(0.774)** 교정용 → 정밀도↑ |
| **커버리지 갭 마이닝** | 트립와이어·coverage_gap 샘플을 **오프라인**에서 EXAONE로 재검토 → NER이 놓친 유형 발굴 | recall↑(라이브 아님 → 인젝션·지연 무관) |
| **테스트 데이터 생성** | 현실형 VOC/로그 생성(외부 Codex/Gemini가 하던 역할의 로컬 대체) | 검증 데이터 다변화, P1 무충돌 |

> **요지**: **"EXAONE가 직접 탐지" ❌ → "EXAONE가 만든 데이터로 GLiNER를 키운다" ✅**. 정 인라인에 넣어야 하면 단독 가드가 아니라 *오프라인/배치·인젝션 하드닝·NER 1차+EXAONE 2차(합집합)+결정적 후필터*로 한정한다(그래도 인젝션 완전 차단 불가, 심층방어).

### (3) 결론
보안 탐지기에 생성형 LLM(외부든 로컬 EXAONE든)을 인라인 1차로 두는 것은 퇴보다. "더 똑똑한 탐지"의 정답은 **인코더 NER 모델 파인튜닝/교체**(§2.4)이고, **로컬 EXAONE는 그 파인튜닝의 데이터·평가 엔진**으로 기여한다. 규칙(정규식·체크섬)은 고심각·정형 카테고리에서 계속 1급으로 유지한다.

---

## 부록 — 검증 재현
- 코퍼스 벤치마크: `benchmarks/korean_ner_benchmark.py --ner-backend {spacy,gliner}`
- 외부 리포트: `validation/EXTERNAL_LLM_TEST_2026-06-23_{claude,codex,gemini}_{spaCy,GLiNER}.md`
- 종합 비교/임계값 스윕: `validation/NER_BACKEND_COMPARISON.md`
- Stage1 보강: `validation/STAGE1_RECALL_IMPROVEMENT_2026-06-25.md`
- 파인튜닝 파일럿: `training/` (README·schema 참조)
- 전체 테스트: 2736 passed / 12 skipped / 0 failed
