# PII-Guard — 설계 문서 (Design Document, as-built)

> 버전: 1.0 · 작성: 2026-06-23 · 대상 커밋: `a0e925c` 기준
> 짝 문서: [`pii-guard-requirements.md`](./pii-guard-requirements.md) (요구사항 v2 + v3 보완 §23)
> 본 문서는 **실제 구현된 시스템(as-built)**을 기술한다. 요구사항의 "무엇을/왜"에 대해, 본 문서는 "어떻게"를 다룬다.

---

## 0. 목차

- **0.5 쉽게 이해하기 — 5분 개요** (비전문가용)
- **0.6 용어 사전 (쉬운 정의)**
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

## 0.5 쉽게 이해하기 — 5분 개요

> 이 절은 **비전문가도 전체 그림을 잡도록** 비유로 설명한다. 정확한 기술 내용은 §1부터다.

### 한 문장 비전
**PII-Guard는 내 PC와 외부 AI(Claude·OpenAI·Gemini) 사이에 세워둔 "보안 검색대"**다. AI에게 보내는 모든 메시지를
나가기 직전에 검사해서, 그 안의 **개인정보·비밀번호·키**를 **가리거나(마스킹) 막는다(차단).**

### 메시지 한 건의 여정 (이야기로)
```
① 에이전트가 외부 AI에게 메시지를 보내려 함
        │
② 그 메시지는 곧장 인터넷으로 안 나가고  ──▶  PII-Guard(검색대)를 먼저 통과
        │
③ PII-Guard가 메시지 속을 읽어 개인정보를 찾음 (바코드 스캐너 + 글 읽는 검사관)
        │
④ 찾은 것을 정책대로 처리:
     · 마스킹: 이름·이메일·전화  →  [PERSON_1] 같은 "가짜 이름표"로 바꿔서 보냄
     · 차단:   주민번호·API키    →  아예 안 보냄 (요청 거부)
        │
⑤ 가짜 이름표로 바뀐 메시지만 외부 AI에 도착  →  진짜 정보는 PC를 안 떠남
        │
⑥ AI 응답이 돌아오면, 가짜 이름표를 다시 진짜 값으로 "복원"해서 에이전트에게 전달 (복원은 내 PC 안에서만)
```

### 핵심 아이디어 5가지 (각 한 줄 비유)
| 개념 | 한 줄로 | 자세히 |
| :-- | :-- | :-- |
| **프록시(검색대)** | 모든 트래픽을 **한 문**으로 지나게 한다 | §3, §5 |
| **하이브리드 탐지(두 검사관)** | **바코드 스캐너**(정규식)로 키·카드를, **글 읽는 검사관**(NER)으로 이름·주소를 | §6 |
| **마스킹(가짜 이름표)** | **구조는 살리고 값만 지운다** → AI는 일을 하되 진짜 값은 못 봄 | §9 |
| **복원(되돌리기)** | 응답의 가짜 이름표를 진짜로 되돌린다 — **단, 내 PC 안에서만** | §9 |
| **컨트롤 플레인(심판 보호)** | 에이전트가 **규칙·열쇠·기록을 못 건드리게** 격리 | §13 |

### 왜 마스킹과 차단을 나누나?
- **마스킹**: AI가 "요약해줘" "코드 고쳐줘" 같은 일을 하려면 *문장 구조*만 있으면 되고 *진짜 이름값*은 필요 없다. 그래서 `[PERSON_1]`로 바꿔도 일이 된다 → **유출은 막고 작업은 살린다.**
- **차단**: API 키·주민번호는 *유출되면 끝*이라 가릴 여유 없이 **요청 자체를 막는다.**

### 정직한 한계 (숨기지 않음)
PII-Guard는 에이전트와 **같은 PC에서 나란히 도는 협조적 도구**다. 그래서 작정한 공격자(특히 root 권한)는 우회할 수
있다. **"완전히 가둔다"는 VM/샌드박스의 몫**이고, PII-Guard는 그걸 할 수 있는 척하지 않는다(§2, P3).

---

## 0.6 용어 사전 (쉬운 정의)

| 용어 | 쉬운 뜻 |
| :-- | :-- |
| **PII** | 개인을 식별하는 정보 (이름·전화·이메일·주민번호·주소 등). Personally Identifiable Information. |
| **프록시(proxy)** | 두 통신 사이에 끼어 트래픽을 중계하는 중간 서버. 여기선 PC와 외부 AI 사이의 "검색대". |
| **업스트림(upstream)** | 프록시가 메시지를 최종적으로 보내는 **진짜 목적지** (api.anthropic.com 등). |
| **Stage1** | **정규식 + 체크섬**으로 모양이 정해진 PII(키·카드·주민번호)를 잡는 1단계. 빠르고 결정적. |
| **Stage2 / NER** | **이름·주소처럼 모양 없는 PII**를 의미로 잡는 2단계. NER=개체명 인식(Named Entity Recognition). |
| **마스킹(masking)** | 값을 `[PERSON_1]` 같은 **플레이스홀더(가짜 이름표)**로 치환. 구조는 보존. |
| **차단(block)** | 위험이 커 마스킹 대신 **요청 자체를 거부**(업스트림에 안 보냄). |
| **복원(rehydration)** | 응답에 돌아온 플레이스홀더를 **진짜 값으로 되돌리기**. 내 PC 안에서만. |
| **세션 맵(session map)** | "진짜값 ↔ 가짜 이름표" 대응표. 메모리에만 있고 디스크에 안 남김. |
| **fail-closed** | 문제가 생기면 **안전한 쪽(차단)으로 실패**. (반대 fail-open=일단 통과 — 위험해서 기본 배제.) |
| **degrade(열화)** | 일부가 고장 나도 멈추지 않고 **할 수 있는 만큼(Stage1)이라도** 계속. |
| **트립와이어(tripwire)** | 구조 파서가 못 본 필드까지 **전체를 한 번 더 훑는 안전망 스윕**. |
| **커버리지 갭/알람** | "여기 검사 못 한 사각지대가 있다"는 표시·경보. 조용히 통과시키지 않기 위함. |
| **egress 락다운** | 방화벽으로 **프록시 외 모든 직접 인터넷 연결을 차단**하는 강제 모드(옵트인). |
| **컨트롤/데이터 플레인** | 통제 구역(규칙·키·기록=에이전트 못 건드림) vs 작업 구역(프로젝트 파일=자유). |
| **Ledger(원장)** | PII를 **원본 없이** 메타데이터로만 남기는 감사 기록. |
| **HMAC / keyed-hash** | 비밀키를 섞은 해시. 키 없이는 역산 불가 → 저엔트로피 PII도 안전하게 기록. |
| **Presidio** | MS의 오픈소스 PII 탐지 **프레임워크**(NER+정규식+신뢰도 통합). §20.1. |
| **spaCy** | NLP 라이브러리. `ko_core_news_lg` = 한국어 NER 모델. §20.2. |
| **rehydrate/scrub** | scrub=요청에서 PII 제거(마스킹), rehydrate=응답에서 되돌림. |

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
> 각 표: **모듈 · 주요 공개 API · 책임과 핵심 동작(상세)**.

### 4.A 코어 탐지/마스킹

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `engine.py` | `Engine`, `Engine.scan(text)→RedactionResult` | **탐지 오케스트레이터이자 공용 진입점.** ① `detector.scan_text()`로 Stage1 실행 → ② 생성자에 `stage2_runner`가 있으면 `runner.scan(text, stage1_dets)`로 Stage2 위임 → ③ 두 결과를 병합해 `RedactionResult` 생성. Stage2 실패 시 `coverage_gap=True`·`stage2_gap_reason` 설정. 프록시·CLI·UI·테스트가 **같은 `Engine`을 재사용**(요구사항 §3.2). |
| `detector.py` | `scan_text(text)` | **Stage1 실행기.** `categories.py`의 모든 `CategorySpec` 패턴을 텍스트에 적용해 매치 수집, 체크섬 검증 통과분만 `Detection`으로 반환. `_resolve_capture_group`로 라벨형 패턴(예: "계좌번호: …")의 캡처 그룹만 정확히 스팬 지정. |
| `categories.py` | `CategorySpec`, `PatternRule`, `_luhn_valid`·`_rrn_checksum`·`_kr_biz_checksum` | **18개 카테고리 정의의 단일 출처.** 각 `CategorySpec` = (카테고리명·`CategoryClass`·`Action`·`MaskStyle`·`min_confidence`·룰목록). `PatternRule` = (정규식 + 신뢰도 + 선택적 검증자). 검증자가 카드(Luhn)·주민번호·사업자번호의 **산술 유효성**을 확인해 오탐 억제. |
| `models.py` | `Detection`, `RedactionResult`, `Action`·`CategoryClass`·`MaskStyle`·`DetectionStage` | **시스템 공용 데이터 타입.** `Detection`(스팬·카테고리·액션·placeholder_token·confidence·keyed_hash 등), `RedactionResult`(`redacted_text`, `detections`, `has_blocks`/`has_masks`, `coverage_gap`, `rehydrate()`). 모든 모듈이 이 타입으로 소통. |
| `proximity.py` | `scan(text)`, `merge(base, extra)`, `ContextRule` | **Stage-1.5 양성 proximity**(context-gated). 모호한 정형 PII(비표준 계좌·맨 사업자번호·한글 비번)를 **트리거 키워드 근접 시에만 승격**. `merge`는 containment 정책(계좌가 전화 하위오탐 흡수). `STAGE1_PROXIMITY` 단계. |
| `masker.py` | `maskPayload`, `apply_redactions`, `rehydrate_text` | **순수 마스킹/복원 함수**(상태 없음). 탐지 스팬을 받아 텍스트를 `[CAT_N]`로 치환하거나 되돌림. 부수효과·세션 상태가 없어 테스트·재사용 용이. |
| `vault.py` | `RequestVault`, `apply_mask_style` | **요청 스코프 마스킹 금고 + 마스크 스타일.** 한 요청 안에서 원본↔토큰 매핑을 보관하며, `tokenize`(기본)·`partial`(부분 가림)·`format_preserving`(형식 보존 더미) 스타일 적용(`_partial_mask`, `_format_preserving_mask`). |
| `session_map.py` | `SessionMap` | **세션 일관성 매핑.** 같은 정규화 원본 → 항상 같은 토큰(LLM 문맥 유지). 양방향 조회(원본→토큰, 토큰→원본). **메모리에만** 존재, 디스크 미영속(P4). |

### 4.B Stage2 — NER 서브프로세스

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `stage2/runner.py` | `Stage2NERRunner`, `Stage2ScanResult`, `.scan(text, stage1_dets)` | **워커 수명·타임아웃·OOM 격리·열화 담당.** `multiprocessing`(spawn)으로 NER 워커를 띄우고 요청/응답 큐로 통신. 블록당 **하드 타임아웃**, 워커 OOM/크래시/예외 시 **Stage1 결과로 graceful degrade** + `fail_reason`·coverage_gap. `_merge_detections`로 Stage1+Stage2 병합. 코어는 절대 안 죽음. |
| `stage2/_workers.py` | `default_ner_worker_loop`, (테스트용 `_test_noop/slow/oom_worker`) | **서브프로세스에서 도는 워커 루프.** `KoreanNEREngine`을 **지연 임포트**해 무거운 spaCy 모델 로딩을 부모(코어)와 격리. 테스트 워커들은 열화 경로(타임아웃·OOM)를 결정적으로 재현. |
| `stage2/korean_ner.py` | `KoreanNEREngine`, `.detect(text)`, `resolve_ko_spacy_model()` | **실제 NER 엔진(Presidio+spaCy).** `resolve_ko_spacy_model`이 `PIIGUARD_KO_SPACY_MODEL`>`lg`>`sm` 순으로 모델 선택. `_build_presidio_analyzer`가 한국어 전용 `AnalyzerEngine` 구성, `_strip_ko_particle`로 조사 제거("홍길동은"→"홍길동"). spaCy 라벨(PS/LC/OG)→PERSON/ADDRESS/ORGANIZATION 매핑. |
| `stage2/policy_layer.py` | `Stage2PolicyLayer`, `Stage2PolicyResult` | **Stage2 탐지에 정책 적용.** NER가 찾은 엔티티에 카테고리별 액션(mask)·신뢰도 임계값을 입혀 최종 처리 결정 산출. |
| `stage2/ner_filters.py` | `is_ner_false_positive(category, text)` | **음성 proximity / NER FP 후필터.** NER 탐지 중 코드토큰(`API_KEY`,`send_email(...)`)·약어(`AWS`,`LGTM`)·base64 blob·일반명사 deny-list(`주석`,`수익자`)를 제거. **제거만**(recall-safe). 정밀도 0.79→0.87. |

### 4.C 프록시 / 프로바이더 파싱

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `proxy.py` | `PIIGuardProxy`, `_detect_provider(path)`, `_handle_post`, `_forward`, `_log_traffic` | **인터셉트 프록시 코어**(`http.server` 기반, 데몬 스레드). 요청 흐름: 바디 읽기→JSON 파싱→프로바이더 판별→`_scrub()`→`_run_tripwire()`→block이면 400/아니면 `_forward()`. 응답은 복원기로 rehydrate. `--log-masked` 시 `_log_traffic`이 **마스킹된 페이로드만** 출력. pin-list 제어 경로 가로채기, 마지막 scrub/tripwire/rehydration 결과를 테스트용으로 보관. |
| `providers/{claude,openai,gemini}.py` | `scrub_{claude,openai,gemini}_request(payload, engine, …)` → `*RequestScrubResult` | **프로바이더별 와이어 스크러버**(마스킹 결정 주체). 각 스키마의 PII 필드를 순회(`ScanField`)하며 `engine.scan()` 호출, 결과를 `sanitized_payload`로 치환. `field_events`(감사용)·`should_block` 반환. block-급 카테고리 발견 시 should_block. |
| `providers/{claude,openai,gemini}_parser.py` | `parse_{provider}_request()` → `{Provider}FieldMap` (`ParsedField`) | **순수 구조 파서**(스캔·마스킹 안 함). 와이어 바디에서 "PII가 들어갈 수 있는 필드 위치"만 구조적으로 추출해 스크러버에 공급. 구조 키 이름 오탐 회피. |
| `providers/schema_coverage.py` | `diff_{provider}_fields`, `FieldDelta`, `VersionDelta` | **프로토콜 staleness 추적.** 핀 고정된 스키마 대비 **미지 필드/미지 API 버전**을 델타로 산출 → 사각지대 감지. |
| `providers/coverage_alarm.py` | `emit_*`, `CoverageAlarmEvent`, `CoverageAlarmResult` | **커버리지 알람 발행.** 미지 필드/버전을 `unknown_field_action`(strict 기본 block)에 따라 차단/경고로 변환, 침묵 통과 방지(P5). |
| `tripwire.py` | `sweep_raw_body(json)` → `TripwireResult`(`TripwireHit`) | **전체 바디 raw 스윕(안전망).** 마스킹된 페이로드 JSON 전체를 다시 훑어 구조 파서가 방문 안 한 필드의 PII-급 히트 포착 → block-급이면 `should_block`. |

### 4.D 복원 / 스트리밍

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `response_rehydrator.py` | `ResponsePostProcessor`, `RehydrationResult` | **비스트리밍 응답 복원.** 업스트림 응답 JSON에서 `[CAT_N]`을 세션 맵으로 실제값 치환(프로바이더별 `_rehydrate_claude/openai/gemini`). 미복원 block-카테고리 토큰 잔존 시 응답 보류/경고. |
| `streaming_buffer.py` | `StreamingLookAheadBuffer`, `_could_be_placeholder_prefix` | **경계 룩어헤드 버퍼.** 청크 경계에 걸친 *플레이스홀더 후보 prefix*만 작은 윈도우로 보류하고 확정 텍스트는 즉시 방출 → 토큰이 청크에 쪼개져도 재조립. |
| `streaming_rehydrator.py` | `_extract/_inject_{provider}_stream_text` 등 | **스트리밍 SSE 복원 배선.** 버퍼를 프로바이더별 SSE 이벤트 스트림에 연결, 청크 텍스트를 꺼내 복원 후 재주입. **TTFT 보존**(전체 버퍼링 안 함), 미복원 block 토큰 미방출. |

### 4.E 정책 / 감사 / 통제면

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `policy.py` | `PolicyConfig`, `PolicyLoader`, `SECURE_DEFAULTS`, `CategoryPolicy`·`AllowlistEntry`·`PinListEntry`·`ChannelOverride` | **단일 스키마 정책 로딩·핫리로드.** YAML(`yaml.safe_load`) 파싱, 로드 실패 시 직전 유효 정책 유지. `SECURE_DEFAULTS`는 바이너리 baked-in → 파일 삭제해도 secure default 폴백(P2). 레이어: 기본<파일<채널 override<allowlist. |
| `decision.py` | `PolicyDecisionEngine`, `PolicyDecision`, `FailureDecision` | **카테고리→액션 결정 + 실패 정책 해석.** 탐지에 allow/mask/block을 매기고, 콘텐츠 실패=block / 인프라 실패=degrade / unscannable=block 등 실패 모드를 결정으로 환원. |
| `ledger.py` | `Ledger`, `LedgerEventType` | **append-only 감사 원장.** block/mask/fail/coverage-gap 이벤트를 **메타데이터 + HMAC keyed-hash로만** 기록(원본 미영속, P4). 파일 600/디렉토리 700, 회전·보존(기본 30일)·명시적 `purge`. |
| `pinlist_guard.py` | `PinListMutationGuard`, `classify_source`, `MutationSource` | **에이전트發 pin-list 변경 차단.** 제어 엔드포인트 요청을 출처 분류해 AGENT면 `AGENT_MUTATION_BLOCKED`로 거부 → 탈취된 에이전트의 자기 화이트리스트 방지(P6). |
| `pinlist_approval.py` | `PinListApprovalGate`, `run_interactive_approval`, 상태 `IDLE/STAGED/COMMITTED/REJECTED` | **사용자 out-of-band 승인 게이트.** pin-list 변경을 staged→사용자 대화형 승인→committed 흐름으로만 반영. 에이전트 루프 밖(사용자)에서만 승인 가능. |
| `pf_manager.py` | `PfManager`, `collect_all_cidrs`, `build_anchor_rules`, `_pfctl_*` | **egress 락다운(티어2) pf(4) 앵커 관리.** 프로바이더 IP 대역(CIDR) 수집→deny-by-default 앵커/테이블/규칙 빌드→`pfctl`로 로드/해제. root 소유 규칙. |
| `boundary.py` | `get_protection_boundary()→BoundaryReport`, `EnforcementTier`, `print_boundary_report` | **보호 경계 정직 선언(P3).** 방어/미방어 항목·우회 경로(bypass_paths)·위협 행위자 모델(root 범위 밖)을 구조화해 보고. "막는 척" 금지를 코드로 강제. |
| `updater.py` | `UpdateSigner`, `UpdateVerifier`, `UpdateManifest`, `UpdateRejectedError` | **룰/모델 서명 검증 업데이트.** 매니페스트 서명 생성·검증으로 비서명/변조 업데이트 거부 → 탐지 룰 공급망 위험 차단(R11). |

### 4.F 진입점 / 부가

| 모듈 | 주요 공개 API | 책임 · 핵심 동작 (상세) |
| :-- | :-- | :-- |
| `cli.py` | `cmd_serve`, `cmd_egress_*`, `cmd_ledger_*`, `cmd_boundary`, `cmd_pin_list`, `build_parser` | **`piiguard` CLI 진입점**(argparse). `serve`가 **NER default-on 배선**(`Engine(stage2_runner=…)`)·`--no-ner`·`--log-masked`·fail-closed(SIGTERM→`os._exit`) 담당. egress/ledger/boundary/pin-list 서브커맨드. |
| `launcher.py` | `ProcessLauncher`, `build_proxy_env`, `ALL_PROXY_ENV_VARS` | **티어1 자동 주입.** 자식 프로세스 env에 `ANTHROPIC_BASE_URL`·`OPENAI_BASE_URL`·`GEMINI_BASE_URL`를 주입해 협조적 도구를 프록시 경유시킴(default-on). |
| `corpus/korean_pii.py` · `corpus/ner_benchmark_corpus.py` | `KoreanPIICorpus`·`CorpusSample`·`PIISpan`, `NERBenchmarkCorpus` | **합성 레드팀·벤치마크 코퍼스.** 실데이터 없이 **유효 체크섬을 가진 가짜 한국 포맷**(`_make_rrn`, `_make_biz`)으로 정밀도/재현율 측정용 픽스처 생성. |
| `ui/app.py` · `ui/scanner.py` | (Streamlit 앱) · `scan_text`·`verdict`·`render_console_block` | **대화형 검증 UI + 순수 로직.** `scanner.py`는 Streamlit 비의존(단위테스트됨), `app.py`는 채팅/다중 파일 탭·NER 토글·콘솔 출력. |
| `benchmarks/korean_ner_benchmark.py` | `run_benchmark()`, `MIN_THRESHOLDS` | **NER precision/recall 벤치마크.** 코퍼스로 full-pipeline·NER-only 지표 측정, 임계값 게이트(`thresholds_met`). |

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
| **ADR-9** | **Stage2 PII 프레임워크 = Microsoft Presidio** | 완전 로컬 + PII 전용(정규식+NER+문맥+신뢰도 통합) + NLP엔진 교체 추상화 + 감사 설명가능성. 상세 §20.1. |
| **ADR-10** | **Stage2 NLP 엔진 = spaCy 한국어 모델** | 8GB 메모리·속도·Presidio 네이티브 통합·비생성형(인젝션 불가)·시스템 의존 없음. 상세 §20.2. (변형 sm/lg 선택은 ADR-5) |

---

## 20.1 ADR-9 (상세) — Stage2 PII 탐지 프레임워크로 Microsoft Presidio 채택

**맥락(Context).** Stage1(정규식·체크섬)이 못 잡는 **문맥 의존 PII**(이름·주소·조직)를 탐지할 엔진이 필요했다.
단순 NER 모델만으로는 부족하다 — PII 탐지는 NER 결과를 **카테고리·신뢰도·문맥**으로 정리하고 정규식 인식기와
통합하는 *프레임워크 레이어*가 필요하기 때문이다.

**검토한 대안(Alternatives considered).**

| 대안 | 종류 | 기각/비선택 사유 |
| :-- | :-- | :-- |
| AWS Comprehend PII · Google Cloud DLP · Azure | 클라우드 API | 🔴 **P1(로컬 우선) 정면 위반** — 탐지하려 PII를 클라우드로 전송 = 막으려는 유출을 자행(자기모순). 즉시 탈락. |
| Nightfall · Private AI · Skyflow · BigID | 상용 | 유료, 외부 의존/온프렘 제약, 블랙박스. |
| scrubadub | 경량 OSS | 정규식+얕은 NER만. 문맥 강화·신뢰도·확장성·언어 무관 NLP 엔진 부재. |
| GLiNER / HuggingFace PII 모델 직접 사용 | 신경망 OSS | *프레임워크가 아님* — 정규식·체크섬·문맥·신뢰도·결정 프로세스를 직접 다 구현해야 함. 무거움. |
| 자작(정규식만) | in-house | Stage1이 이미 담당. 문맥 PII 불가. |

**결정(Decision).** **Microsoft Presidio**(`presidio-analyzer`/`presidio-anonymizer`)를 Stage2 프레임워크로 채택.

**근거(Rationale).**
1. **완전 로컬** — 가장 강력한 클라우드 PII 서비스는 P1 위반으로 자동 탈락. Presidio는 온프렘/로컬.
2. **PII 전용 프레임워크** — 범용 NER이 아니라 *정규식 + deny-list + NER + 문맥 강화 + 신뢰도*를 이미 통합.
3. **확장·추상화** — NLP 엔진 교체(spaCy→Stanza→transformer), 커스텀 recognizer/엔티티. 요구사항 §6.2의 모델 교체 슬롯과 합치.
4. **언어 무관 설계** — `NlpEngineProvider` 설정만으로 한국어 모델 주입.
5. **설명가능성** — 탐지별 신뢰도·decision process → Ledger 감사에 유리.
6. 무료·성숙·MS 백업·활발한 커뮤니티.

**적용(Implementation).** `stage2/korean_ner.py` — 기본 영어 recognizer를 모두 비활성화하고, `NlpEngineProvider`에
한국어 spaCy 모델 주입 + `SpacyRecognizer(supported_language="ko", entities=[PERSON,LOCATION,ORGANIZATION])`로
재구성. spaCy 라벨(PS/LC/OG)→Presidio 엔티티 매핑.

**결과·트레이드오프(Consequences).** 의존성이 무겁고 기본 설정이 영어 중심이라 한국어 재구성이 필요 → 별도
프로세스 + `[ner]` 옵션 설치로 **격리**(코어는 표준 라이브러리만, ADR-2와 정합).

---

## 20.2 ADR-10 (상세) — Stage2 NLP 엔진으로 spaCy 한국어 모델 채택

**맥락(Context).** Presidio(ADR-9)에 꽂을 **NER 두뇌**가 필요했다. 핵심 제약: **타깃이 MacBook Air M2 8GB,
메모리 예산 ~1~1.5GB**, 요청당 지연 예산(p50<200ms), 그리고 **비생성형**이어야 함(DR-1: 생성형은 프롬프트
인젝션으로 탐지기가 무력화됨).

**검토한 대안(Alternatives considered).**

| 대안 | 정확도 | 메모리/속도 | 비선택 사유 |
| :-- | :-- | :-- | :-- |
| **HuggingFace transformer** (KoELECTRA·KLUE·KoBERT) | **최상(한국어 SOTA급)** | 🔴 수백MB~GB + PyTorch, 느림 | **8GB 메모리 예산 초과**. → 요구사항 §6.2가 명시한 *향후 업그레이드 경로*로 보류. |
| **Stanza** (Stanford NLP) | 높음 | 무겁고 느림 | spaCy 대비 메모리·지연 불리. (Presidio 지원은 됨) |
| **KoNLPy** (Mecab·Komoran·Okt·Kkma) | 형태소는 강하나 **NER 약함** | 중간 | 주 용도가 형태소·품사 분석. **Java/Mecab 시스템 설치** 필요 → 배포 복잡. |
| **Mecab-ko** | NER 아님 | 가벼움 | 형태소 분석기 — 목적 불일치. |
| 직접 학습 모델 | 도메인 최적 | 가변 | 학습 데이터·비용 大 (2차). |

**결정(Decision).** **spaCy + 공식 한국어 파이프라인 `ko_core_news_lg`**(sm 폴백). 변형 sm/lg 선택은 ADR-5.

**근거(Rationale).**
1. **메모리(결정적)** — spaCy lg는 수백MB로 8GB 예산에 들어감. transformer는 GB급 → 초과.
2. **속도** — Cython 기반으로 빠름, GPU 불필요 → 지연 예산 충족.
3. **Presidio 네이티브 통합** — Presidio가 spaCy NLP 엔진을 1급 지원 → 깔끔한 결합.
4. **시스템 의존 없음** — `ko_core_news_*`를 pip로 설치. KoNLPy류의 Java/Mecab 설치 불필요 → 배포 단순.
5. **비생성형(인코더)** — 텍스트를 *분류*만 하고 *생성*하지 않음 → **프롬프트 인젝션 불가**(DR-1 핵심 근거).
6. **공식 한국어 모델 존재** — 별도 학습 없이 PERSON/LOCATION/ORGANIZATION 인식.

**적용(Implementation).** `resolve_ko_spacy_model()` — `PIIGUARD_KO_SPACY_MODEL` env > `lg` > `sm`. 조사
스트리핑("홍길동은"→"홍길동")으로 플레이스홀더 매칭 정합.

**결과·트레이드오프(Consequences).** 정확도 천장은 transformer보다 낮음(PERSON recall 0.84) → `KoreanNEREngine`
추상화로 **모델 교체 슬롯**을 남겨, 더 높은 정확도가 필요하면 KoELECTRA/KLUE로 승급 가능(요구사항 §6.2·§23.4).

---

*문서 끝. 변경 시 짝 문서(`pii-guard-requirements.md` §23)와 동기화할 것.*
