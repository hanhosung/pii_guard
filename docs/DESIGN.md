# PII-Guard — 설계 문서 (Design Document, as-built)

> 버전: 1.0 · 작성: 2026-06-23 · 대상 커밋: `a0e925c` 기준
> 짝 문서: [`pii-guard-requirements.md`](../../pii-guard-requirements.md) (요구사항 v2 + v3 보완 §23)
> 본 문서는 **실제 구현된 시스템(as-built)**을 기술한다. 요구사항의 "무엇을/왜"에 대해, 본 문서는 "어떻게"를 다룬다.

---

## 0. 목차

1. 시스템 개요
2. 위협 모델 → 설계 매핑
3. 런타임 아키텍처 (프로세스·플레인)
4. 모듈 맵 (패키지 구조)
5. 요청 생명주기 (end-to-end 시퀀스)
6. 탐지 파이프라인 (Stage1 / Stage2)
7. 프로바이더 파싱 · 트립와이어 · 커버리지
8. 정책 엔진
9. 마스킹 · 세션 일관성 · 복원(rehydration)
10. 스트리밍 응답 처리
11. Ledger (감사)
12. 실패 모드 · 열화 · 가용성
13. 컨트롤 플레인 (정책·키·pin-list·egress)
14. CLI · 관찰가능성(`--log-masked`)
15. UI (Streamlit)
16. 데이터 모델 (핵심 타입)
17. 테스트 전략
18. 카테고리 카탈로그 (구현)
19. 알려진 한계 · 향후 작업
20. 설계 결정 기록 (ADR)

---

## 1. 시스템 개요

PII-Guard는 **로컬 인터셉트 프록시**다. ouroboros 워크플로·LLM CLI(Claude/OpenAI/Gemini)가 외부 LLM API로
보내는 HTTP 요청을 가로채, 페이로드 안의 **PII·시크릿을 로컬에서 탐지**하고 **카테고리별 정책(block/mask/allow)**을
적용한 뒤 업스트림으로 중계한다. 응답은 로컬 세션 맵으로 복원(rehydrate)해 에이전트에게 돌려준다.

**불변 원칙(요구사항 P1~P6)의 설계 반영:**

| 원칙 | 설계 구현 |
| :-- | :-- |
| P1 로컬 우선 | 탐지 = 정규식 + 로컬 spaCy NER. **외부 LLM 호출은 중계 외엔 없음.** (DR-1: LLM 탐지 거부) |
| P2 secure-by-default | `SECURE_DEFAULTS` 바이너리 baked-in. 정책 파일 없어도 시크릿·주민번호 block, 이름·이메일 mask. |
| P3 거짓 안심 금지 | `boundary.py`가 방어/미방어를 정직 선언. 한계 §19 명문화. coverage gap 가시 기록. |
| P4 세 금고 금지 | Ledger=메타+HMAC만. 세션 맵=메모리 한정·디스크 미영속. credential passthrough. |
| P5 침묵 통과 금지 | unscannable→block, Stage2 실패→coverage gap 기록, 미복원 block 토큰→응답 보류. |
| P6 통제면 격리 | pin-list 변경=out-of-band 승인. 정책/키/Ledger 600/700. egress pf=root. |

---

## 2. 위협 모델 → 설계 매핑

위협 행위자 = **trusted-but-compromisable**(선의이나 프롬프트 인젝션으로 탈취 가능). 상세는 요구사항 §2.

| 위협 | 설계 방어 | 모듈 |
| :-- | :-- | :-- |
| 에이전트가 tool_result에 실린 시크릿을 무의식 유출 | 전 의미 필드 스캔 + block | `providers/*`, `engine` |
| 탈취된 에이전트가 자기 유출을 화이트리스트 | pin-list/allowlist 변경 = 사용자 out-of-band 승인 | `pinlist_guard`, `pinlist_approval` |
| 탈취된 에이전트가 정책/키 변조 | 컨트롤 플레인 파일 600/700, 쓰기 권한 밖 | `policy`, `ledger`, `pf_manager` |
| **탐지기 자체를 프롬프트 인젝션으로 무력화** | **탐지기가 생성형 LLM이 아님** — 정규식·인코더 NER은 지시를 "실행"하지 않음 (DR-1) | `detector`, `stage2/korean_ner` |
| 미지 프로토콜 필드로 스캔 사각지대 생성 | unknown_field → 커버리지 알람(strict=block) | `providers/coverage_alarm`, `schema_coverage` |
| 구조 파서가 안 본 필드에 PII | 전체 바디 트립와이어 스윕 | `tripwire` |
| 프록시 크래시로 무방비 직통 | fail-closed (os._exit → TCP RST), 워치독 | `cli.cmd_serve`, `proxy` |
| root/커널 권한 에이전트 | **범위 밖** — 정직 선언(boundary), VM/샌드박스는 별 제품 | `boundary` |

---

## 3. 런타임 아키텍처

```
┌────────────┐   HTTP    ┌──────────────────── PII-Guard 프록시 (코어) ─────────────────────┐   TLS   ┌────────────┐
│ ouroboros  │  (localhost)│  PIIGuardProxy (proxy.py) — 항상 상주, 경량                       │         │ api.anthropic│
│ Claude CLI │──────────▶│   1. _detect_provider(path)                                       │────────▶│ api.openai   │
│ Codex CLI  │           │   2. _scrub() → 프로바이더 파서 + Engine(Stage1[+Stage2])          │         │ generativelang│
│ Gemini CLI │◀──────────│   3. _run_tripwire() 안전망                                        │◀────────│   (응답)      │
│            │           │   4. block? 400 : forward (마스킹된 페이로드)                       │         └────────────┘
└────────────┘           │   5. 응답 rehydrate → 클라이언트                                    │
                         │                                                                   │
                         │   ── Stage2 NER 워커 (별도 프로세스, spawn) ──                     │
                         │      Stage2NERRunner ⇄ _workers (Presidio+spaCy lg)               │
                         │      OOM/timeout 시 코어 생존, Stage1 degrade                       │
                         │                                                                   │
                         │   ── 컨트롤 플레인 (격리) ── 정책 / HMAC키 / Ledger / pf 규칙        │
                         └───────────────────────────────────────────────────────────────────┘
```

**프로세스 분할(가용성·R9·R13):**
- **코어**: `PIIGuardProxy`(`http.server` 기반, 데몬 스레드). Stage1은 인프로세스(경량·결정적).
- **Stage2 워커**: `multiprocessing`(spawn) 자식. Presidio+spaCy lg 로딩(무거움). **블록당 하드 타임아웃**.
  워커 OOM/SIGKILL이 코어를 못 죽임 → `Stage2NERRunner.scan()`이 Stage1 결과로 graceful degrade.
- **fail-closed**: SIGTERM/SIGINT → `os._exit(0)` → OS가 모든 TCP를 RST. 인플라이트 응답이 새지 않음.

---

## 4. 모듈 맵 (`pii_guard/`)

> 요구사항 §3.2의 계획(`core/ detectors/ …` 중첩)은 구현에서 **평면 패키지**로 수렴했다.

### 코어 탐지/마스킹
| 모듈 | 책임 |
| :-- | :-- |
| `engine.py` | `Engine.scan(text)` — Stage1 실행 후 (옵션)Stage2 위임, `RedactionResult` 생성 |
| `detector.py` | Stage1 정규식·체크섬·사전 탐지 실행기 |
| `categories.py` | **18개 `CategorySpec`** 정의 (패턴·체크섬·액션·신뢰도) |
| `models.py` | `Detection`, `RedactionResult`, `Action`, `CategoryClass`, `DetectionStage`, `MaskStyle` |
| `masker.py` | `maskPayload` — 순수 마스킹 함수 |
| `vault.py` | `RequestVault` — 요청 스코프 마스킹 + 마스크 스타일 적용 |
| `session_map.py` | `SessionMap` — 원본↔플레이스홀더 세션 일관 매핑(메모리) |

### Stage2 (NER 서브프로세스)
| 모듈 | 책임 |
| :-- | :-- |
| `stage2/runner.py` | `Stage2NERRunner` — 워커 수명·타임아웃·OOM 격리·degrade |
| `stage2/_workers.py` | 워커 루프 — 서브프로세스에서 NER 호출(지연 임포트로 모델 로딩 격리) |
| `stage2/korean_ner.py` | `KoreanNEREngine` — Presidio+spaCy. `resolve_ko_spacy_model()`(lg 우선) |
| `stage2/policy_layer.py` | `Stage2PolicyLayer` — Stage2 결과에 정책 적용 |

### 프록시 / 프로바이더
| 모듈 | 책임 |
| :-- | :-- |
| `proxy.py` | `PIIGuardProxy` — HTTP 인터셉트·스크럽·포워드·복원·`--log-masked` |
| `providers/{claude,openai,gemini}.py` | 프로바이더별 와이어 스크러버(`scrub_*_request`) |
| `providers/{claude,openai,gemini}_parser.py` | 구조 파서 (PII 필드 순회) |
| `providers/schema_coverage.py` | 스키마 커버리지(필드 방문 추적) |
| `providers/coverage_alarm.py` | unknown_field → 커버리지 알람(block/warn) |
| `tripwire.py` | 전체 바디 raw 스윕(안전망) |

### 복원 / 스트리밍
| 모듈 | 책임 |
| :-- | :-- |
| `response_rehydrator.py` | `ResponsePostProcessor` — 비스트리밍 응답 복원 |
| `streaming_buffer.py` | 경계 룩어헤드 버퍼(청크 경계 토큰 재조립) |
| `streaming_rehydrator.py` | 스트리밍 SSE 복원(TTFT 보존) |

### 정책 / 감사 / 통제면
| 모듈 | 책임 |
| :-- | :-- |
| `policy.py` | `PolicyConfig`, `PolicyLoader`, `SECURE_DEFAULTS`, 핫리로드 |
| `decision.py` | `PolicyDecisionEngine` — 카테고리×액션 결정, 실패 정책 |
| `ledger.py` | `Ledger` — append-only, HMAC, 600/700, 회전/보존/purge |
| `pinlist_guard.py` | 에이전트發 pin-list 변경 차단 |
| `pinlist_approval.py` | 사용자 out-of-band 승인 게이트 |
| `pf_manager.py` | egress 락다운 pf(4) 앵커 관리(root) |
| `boundary.py` | 보호 경계 정직 선언 리포트 |
| `updater.py` | 룰/모델 서명 검증 업데이트 |

### 진입점 / 부가
| 모듈 | 책임 |
| :-- | :-- |
| `cli.py` | `piiguard` CLI (`serve`/`egress`/`ledger`/`boundary`/`pin-list`) |
| `launcher.py` | `ProcessLauncher` — 자식 프로세스에 base_url env 자동 주입(티어1) |
| `corpus/korean_pii.py`, `corpus/ner_benchmark_corpus.py` | 합성 레드팀·벤치마크 코퍼스 |
| `ui/app.py`, `ui/scanner.py` | Streamlit UI + 순수 스캔 로직 |
| `benchmarks/korean_ner_benchmark.py` | 한국어 NER precision/recall 벤치마크 |

---

## 5. 요청 생명주기 (end-to-end)

`proxy.py::PIIGuardProxy._handle_post` 기준.

```
1. 클라이언트 POST  /v1/messages (Claude) | /v1/chat/completions (OpenAI) | /v1beta/... (Gemini)
2. Content-Length 만큼 raw body 읽기 → JSON 파싱 (실패 시 400, 미전송)
3. _detect_provider(path) → "claude"|"openai"|"gemini"|None
4. provider ≠ None:
   a. _scrub(payload, provider):
        - 프로바이더 파서가 PII 필드 순회
        - 각 필드 텍스트 → engine.scan() (Stage1 [+ Stage2 NER])
        - 탐지마다: action=block → should_block=True / action=mask → 플레이스홀더 치환
        - SessionMap에 원본↔토큰 등록(세션 일관)
   b. _run_tripwire(sanitized_payload): 마스킹된 JSON 전체 raw 스윕
        - 구조 파서가 못 본 필드의 block-급 PII 발견 시 should_block
   c. should_block(스크럽 OR 트립와이어) → 400 _BLOCKED_RESPONSE, 업스트림 미전달
   d. (--log-masked) 마스킹 페이로드 + 탐지 요약 stdout
   e. forwarded_payload = sanitized_payload
   provider == None: 패스스루(스크럽 없음)
5. _forward(handler, path, forwarded_payload):
   - 마스킹된 JSON을 업스트림(self.upstream_url + path)으로 전송 (정상 TLS)
   - 스트리밍 응답? → _forward_streaming (경계 버퍼 + 복원)
   - 비스트리밍? → ResponsePostProcessor로 [CAT_N] → 원본 복원(rehydrate ON 기본)
   - 미복원 block-카테고리 토큰 잔존 시 응답 보류/경고
6. 복원된 응답을 클라이언트로 반환
```

**보안 불변식:**
- 업스트림에 도달하는 것은 **항상 마스킹된 페이로드**(또는 차단되어 미도달).
- 복원은 **인바운드(응답) 경로에서만**, **로컬에서만** — 외부로 원본이 안 나감.
- 원문은 **stdout 로그·Ledger 어디에도** 안 남음.

---

## 6. 탐지 파이프라인

### 6.1 Stage 1 — 결정적 (인프로세스, 동기)
- `detector.py`가 `categories.py`의 18개 `CategorySpec`을 적용.
- 각 카테고리 = 정규식 패턴 + (옵션)**체크섬 검증자**:
  - RRN: 13자리 가중합 체크섬 (`_rrn_checksum`)
  - CARD: Luhn
  - BIZ_NO: 사업자번호 체크섬
- **체크섬으로 FP 억제** — 무효 번호는 미탐지(정밀도 우선).
- 결정적·재현 가능·**프롬프트 인젝션 불가**.

### 6.2 Stage 2 — 문맥 NER (서브프로세스, 타임아웃)
- `KoreanNEREngine`: Presidio `AnalyzerEngine` + spaCy 한국어 모델.
- **모델 해석 순서**(`resolve_ko_spacy_model`): `PIIGUARD_KO_SPACY_MODEL` env > `ko_core_news_lg` > `ko_core_news_sm`.
- spaCy 라벨 → Presidio 엔티티 매핑: `PS→PERSON, LC→LOCATION(=ADDRESS), OG→ORGANIZATION`.
- 한국어 조사 스트리핑("홍길동은"→"홍길동").
- `min_confidence`(기본 0.50) 미만 폐기.
- **격리·degrade**: `Stage2NERRunner.scan(text, stage1_dets)` — 워커 타임아웃/OOM/예외 시
  Stage1 결과 반환 + `coverage_gap=True` + `fail_reason`.

### 6.3 올바른 fast-path (요구사항 §6.3)
- ❌ "Stage1 clean이면 NER 스킵"은 틀림(NER은 정규식이 놓친 걸 담당).
- ✅ 콘텐츠 클래스 게이팅(자연어 스팬만 NER, 코드·base64·hex blob 스킵) + 블록 해시 캐시.

### 6.4 NER 품질 (ko_core_news_lg, full-pipeline)
| 엔티티 | precision | recall |
| :-- | :-- | :-- |
| PERSON | 0.97 | 0.84 |
| ADDRESS | 1.00 | 1.00 |
| ORGANIZATION | 1.00 | 0.92 |

(`benchmarks/korean_ner_benchmark.py`, `thresholds_met=true`)

---

## 7. 프로바이더 파싱 · 트립와이어 · 커버리지

3중 방어(요구사항 §5.2):

1. **구조 파서(주)** `providers/{provider}_parser.py` — 각 와이어 포맷의 PII 보유 필드를 구조적으로 순회.
   - Claude: `system`, `messages[].content[]`(text/tool_use/tool_result/document), …
   - OpenAI: `messages[].content`, `tool_calls[].function.arguments`, …
   - Gemini: `contents[].parts[]`, …
2. **커버리지 알람** `coverage_alarm.py` + `schema_coverage.py` — 미지 필드/미지 API 버전 →
   `unknown_field_action`(strict 기본 **block**). 스캔 사각지대를 침묵 통과시키지 않음.
3. **트립와이어(안전망)** `tripwire.py` — 마스킹된 페이로드 JSON 전체를 raw 스윕. 구조 파서가
   방문하지 않은 비표준/중첩 필드에서 PII-급 히트 발견 시 **커버리지 갭 확정** → block-급이면 fail-closed.

---

## 8. 정책 엔진

- **단일 스키마**(`policy.py::PolicyConfig`) — 핫리로드(`PolicyLoader`), 로드 실패 시 직전 유효 정책 유지.
- **레이어 우선순위**: `SECURE_DEFAULTS(baked-in) < 사용자 파일 < 채널 override < allowlist`.
- 정책 파일을 지워도 `SECURE_DEFAULTS`로 폴백(P2). 주요 필드:
  - `fail_mode=closed`, `on_content_failure=block`, `on_infra_failure=degrade_to_stage1`
  - `stage2_fail_action=mask_known_only`, `unscannable_action=block`, `unknown_field_action=block`
  - `rehydrate=True`, `memory_budget_mb=1024`, `egress_lockdown=False`
  - `categories{}`, `allowlist[]`, `pin_list[]`(+`pin_list_approved`), `channel_overrides{}`
- `decision.py::PolicyDecisionEngine` — 카테고리→액션 결정 + 실패 정책 해석.
- **액션 의미**: `allow < mask < block`. mask는 구조 보존(`[CAT_N]`)이라 요약·코드생성 등 정상 동작.

---

## 9. 마스킹 · 세션 일관성 · 복원

### 9.1 토큰 형식
- **인덱스 플레이스홀더** `[CATEGORY_N]` (예 `[PERSON_1]`, `[EMAIL_2]`).
- block 카테고리는 `[CAT_N_BLOCKED]` 형태로 표기되며, 프록시 경로에선 **요청 전체 차단**.

### 9.2 세션 일관성 (`SessionMap`)
- 같은 정규화 원본 → 같은 토큰(LLM이 "같은 사람" 문맥 유지).
- 키 = 정규화 원본 해시. **메모리 한정·디스크 미영속**(P4).

### 9.3 복원(rehydration) — 경로별 스코프 (요구사항 §9.2)
- **에이전트 왕복(tool_result·파일 블록) = 복원 ON 기본**: 인바운드 응답에서 `[CAT_N]`→실제값.
  - 외부 LLM은 플레이스홀더만 봄(유출 차단) + 에이전트는 실제값을 되씀(데이터 파괴 없음) + 복원 로컬 전용.
- **사람 종단 출력 = 복원 OFF**(엄격).
- **미복원 토큰 탐지**: 응답에 복원 못 한 block-카테고리 잔여 플레이스홀더 → 응답 보류/경고(조용한 손상 방지).

---

## 10. 스트리밍 응답 처리 (요구사항 §12)

- **아웃바운드(요청)는 비스트리밍** — 전체 바디 확보 후 전송 → egress 탐지에 청크 문제 없음.
- **인바운드(응답) 복원만 스트리밍 이슈** (`streaming_buffer.py`, `streaming_rehydrator.py`):
  - **경계 룩어헤드 버퍼**: 플레이스홀더 토큰 최대 길이만큼의 작은 슬라이딩 윈도우만 보류.
    청크 경계에 걸친 토큰(`[EMA`|`IL_1]`)만 재조립, 확정 prefix는 즉시 방출.
  - **TTFT 보존**: 전체 응답 버퍼링 안 함. (E2E·벤치마크로 검증 — 첫 청크 지연 최소.)
  - **미복원 block 토큰 미방출** 보장.

---

## 11. Ledger (감사) — 요구사항 §13

- `ledger.py::Ledger` — append-only, **메타데이터 전용**.
- 필드: `timestamp, channel, provider, category, action, rule_id, confidence, severity, span 길이, charclass 시그니처, fail/gap/degrade 사유, keyed-hash`.
- **원본 미영속**(P4). 저엔트로피 PII(전화·주민번호)는 **설치 로컬 HMAC 키**로 keyed-hash(전수 역산 방지) 또는 per-value 해시 생략(카테고리+개수만).
- 파일 600 / 디렉토리 700. 보존 30일 기본 + 회전 + 명시적 `purge`. 네트워크 export 없음.

---

## 12. 실패 모드 · 열화 · 가용성 (요구사항 §14)

| 실패 유형 | 동작 | 구현 |
| :-- | :-- | :-- |
| 콘텐츠 실패(파싱 불가 등) | 그 요청/블록 **block**(fail-closed) | `unscannable_action=block` |
| Stage2 인프라 실패(OOM/timeout/로드실패) | **Stage1-only degrade** + coverage gap | `Stage2NERRunner.scan` |
| 프록시 전체 실패(크래시/행/OOM) | **fail-closed**(직통 없음) + 워치독 | `os._exit` RST |

- **보안 바닥 유지**: Stage2 degrade 시에도 시크릿·주민번호 등 Stage1 고정 패턴은 잡음.
- **미보호 통과 가시화**: PERSON/ADDRESS는 degrade 시 통과시키되 coverage gap + 경고(P5).
- egress 락다운 ON에서 프록시 다운 = 하드 가용성 절벽 → 워치독 + **사용자만 가능한 break-glass**(root, 에이전트 불가).

---

## 13. 컨트롤 플레인

| 자산 | 보호 | 모듈 |
| :-- | :-- | :-- |
| 정책 파일 | 600, 핫리로드, baked-in 폴백 | `policy` |
| HMAC 키 | 600, 컨트롤플레인 분리 | `ledger` |
| Ledger | 600/700, append-only | `ledger` |
| pin-list 변경 | **out-of-band 사용자 승인**(에이전트 차단) | `pinlist_guard`, `pinlist_approval` |
| egress pf 규칙 | root 소유 앵커 | `pf_manager` |
| 룰/모델 업데이트 | 서명 검증 | `updater` |

- **티어1(협조적 게이트웨이)**: `launcher.py`가 자식 프로세스에 `ANTHROPIC_BASE_URL` 등 자동 주입(default-on).
- **티어2(egress 락다운, 옵트인)**: pf 아웃바운드 deny-by-default + 프록시만 화이트리스트. **로컬 루트 CA 미설치**(MITM 배제).
- **정직한 명명**: `boundary.py`가 티어1=best-effort 필터(우회 가능), 티어2=실제 강제임을 선언.

---

## 14. CLI · 관찰가능성

`piiguard` (= `python -m pii_guard.cli`):

| 서브커맨드 | 기능 |
| :-- | :-- |
| `serve --upstream-url URL [--port] [--no-ner] [--log-masked]` | 프록시 실행(포그라운드) |
| `egress enable/disable/status` | 티어2 pf 락다운(root) |
| `ledger ...` | 감사 로그 관리 |
| `boundary [--json]` | 보호 경계 리포트 |
| `pin-list ...` | out-of-band 승인 흐름 |

**`serve` 기본값(R14·R15):**
- NER **default-on**: `Engine(stage2_runner=Stage2NERRunner())`. `--no-ner`로만 비활성.
- `--log-masked`: 업스트림 전송 직전 **마스킹된 페이로드 + 탐지 요약(category→placeholder)**을 stdout 출력.
  - **원본 미출력** — `sanitized_payload`만 직렬화(no-raw-in-logs, Ledger와 동일 규율).
  - 차단 요청은 `✗ BLOCKED — NOT forwarded`로 표기.

---

## 15. UI (Streamlit) — R16

- `ui/app.py` — 채팅 메시지 입력 탭 + 다중 파일 업로드 탭 + NER on/off 토글.
- `ui/scanner.py` — **순수 로직**(Streamlit 비의존): `scan_text`, `verdict`, `render_console_block`. 단위테스트됨.
- 출력: 원문/마스킹 비교 + 탐지 테이블(카테고리·액션·신뢰도) + 판정(🔴BLOCK/🟡MASK/🟢CLEAN) + **콘솔 블록(화면 + 터미널 stdout)**.
- 바이너리/비-UTF-8 파일 → unscannable → **fail-closed BLOCK**(정책 일치).
- 실행: `.venv/bin/python -m streamlit run ui/app.py` → http://localhost:8501.

---

## 16. 데이터 모델 (핵심 타입, `models.py`)

```
Detection
  category: str               # "PERSON", "AWS_SECRET", ...
  category_class: CategoryClass  # PII | KOREAN_PII | SECRET
  action: Action              # BLOCK | TOKENIZE_ROUNDTRIP | ...
  mask_style: MaskStyle       # TOKENIZE | ...
  start, end: int             # 스팬
  original: str               # 원본 스팬(메모리 한정, 미영속)
  placeholder_token: str      # "[PERSON_1]"
  detection_stage: DetectionStage  # STAGE1_REGEX_CHECKSUM | STAGE2_NER
  rule_id, confidence, keyed_hash, char_class_signature, span_length

RedactionResult
  original_text, redacted_text: str
  detections: list[Detection]
  has_blocks, has_masks: bool
  coverage_gap: bool
  stage2_gap_reason: str | None
  rehydrate(...), summary(), add_detection(...)
```

---

## 17. 테스트 전략

- **39개 테스트 파일 / 2640 passed / 12 skipped / 0 failed** (`.venv` + ko_core_news_lg).
- 계층:
  - 단위: 카테고리·마스커·세션맵·정책·Ledger·복원·파서·트립와이어·스트리밍·NER 엔진.
  - 와이어: `test_{claude,openai,gemini}_wire.py` — 프로바이더 포맷 스크럽.
  - 통합: `test_pipeline_integration.py`, `test_serve_ner_wiring.py`(R14 회귀), `test_log_masked.py`(R15).
  - **E2E**: `scripts/e2e_smoke.py` — 실제 `serve` 서브프로세스 + mock 업스트림으로 MASK/BLOCK/REHYDRATE 검증.
  - 효능: `test_korean_pii_corpus/regression`, `test_ner_benchmark`, `benchmarks/korean_ner_benchmark.py`(precision/recall 게이트).
  - 실패: `test_fail_closed`, `test_crash_fail_closed`, `test_stage2_degradation`.
  - 12 skip = root+pf(4)+실네트워크 필요한 egress 통합(`-m integration`) — **미실행(§19 한계)**.
- **합성 데이터만**(실 PII 금지). 체크섬 유효한 가짜 한국 포맷 픽스처.

---

## 18. 카테고리 카탈로그 (구현, kr-strict)

| 클래스 | 카테고리 | 기본 액션 |
| :-- | :-- | :-- |
| SECRET | API_KEY, AWS_SECRET, GCP_KEY, TOKEN, PRIVATE_KEY, PASSWORD | **block** |
| KOREAN_PII (고위험) | RRN, FOREIGN_REG, PASSPORT | **block** |
| PII (고위험) | DRIVER_LICENSE, CARD | **block** |
| PII / KOREAN_PII (연락·식별) | EMAIL, PHONE, BIZ_NO, KR_ACCOUNT | **mask**(tokenize_roundtrip) |
| PII / KOREAN_PII (문맥 NER) | PERSON, ADDRESS, ORGANIZATION | **mask**(tokenize_roundtrip) |

(18개. `CategoryClass` = PII | KOREAN_PII | SECRET. 사용자 커스텀 카테고리 확장 가능.)

---

## 19. 알려진 한계 · 향후 작업

> 요구사항 §23.3과 동기화. 정직 선언(P3).

1. **KR_ACCOUNT 비표준 포맷 누락** — 라벨 없는 3-3-6 등은 오탐 억제 위해 미탐지. → 문맥 키워드 규칙 또는 인코더 NER 보강.
2. **PERSON recall 0.84** — 작은 모델 한계. → KoELECTRA/KLUE 교체(설계된 슬롯).
3. **egress 락다운 실검증 미수행** — 12 skip. 배포 전 `sudo pytest -m integration`.
4. **hwp/OCR 미구현** — 2차. 현재 unscannable→block.
5. **단일 프로세스 권한 모델** — 별도 UID 미적용(단일 사용자 가정).
6. **ouroboros 오케스트레이터 종결** — 3차 run은 벤치 타임아웃으로 FAILED 기록(코드는 건전, 로컬 2640 통과). 4차 closure run 준비됨.

**2차 로드맵(요구사항 §19)**: egress 락다운+break-glass, hwp/OCR, proxy_held/tokenize_roundtrip, transformer NER, 서명 자동 룰 갱신, VM/샌드박스(티어3).

---

## 20. 설계 결정 기록 (ADR)

| ID | 결정 | 핵심 근거 |
| :-- | :-- | :-- |
| ADR-1 | 가로채기 = base_url 재설정 (MITM/로컬 CA 배제) | 고가치 공격 표면(루트 CA) 회피. 막기 > 가로채기. |
| ADR-2 | 코어/Stage2 워커 프로세스 분할 | NER OOM이 코어를 못 죽이게(가용성·R9). |
| ADR-3 | 복원 ON(에이전트 왕복) | 복원 OFF는 사용자 시크릿 영구 파괴 = 자기모순(R10). |
| ADR-4 | Ledger 메타+HMAC, 원본 미영속 | 세 금고 역설 봉인(P4). 저엔트로피 역산 방지. |
| ADR-5 | **ko_core_news_lg 채택**(sm 폴백, env override) | recall +0.12~0.18, 정밀도 유지. |
| ADR-6 | **serve NER default-on**(R14) | E2E 갭: 미연결 시 한국어 이름 평문 유출. |
| ADR-7 | **`--log-masked` 관찰가능성**(R15), 원본 미출력 | 마스킹 검증 + no-raw-in-logs. |
| **ADR-8 (DR-1)** | **LLM 기반 탐지 거부, 규칙+로컬 인코더 NER 유지** | 외부 LLM=P1위반·자기모순. 생성형 LLM=메모리·비결정성·**프롬프트 인젝션으로 탐지기 무력화**. 인코더 NER은 생성 안 해 인젝션 불가. |

---

*문서 끝. 변경 시 짝 문서(`pii-guard-requirements.md` §23)와 동기화할 것.*
