# PII-Guard — 세션 인계 문서 (Session Handoff)

> 작성: 2026-06-22 / 다음 세션이 이어서 작업할 수 있도록 현재 상태를 정리한 문서.
> **다음 세션은 이 문서를 먼저 읽고 "다음 세션 할 일" 섹션부터 시작하세요.**

---

## 0. 한 줄 요약

ouroboros 워크플로로 개발 중인 **로컬 우선(local-first) LLM 게이트웨이 PII/시크릿 차단 에이전트**.
1차 워크플로(AC 1–7) 완료 → 검토 후 Seed에 AC 8·9·10 보강 → **2차 델타 워크플로 진행 중**.
현재 **AC 8·9 산출물이 working tree에 미커밋 상태로 존재**하고, **AC 10(실제 NER)은 아직 스텁**.

---

## 0.5 ✅ 2차 델타 — 완료됨 (2026-06-22 10:15:47 UTC 종료)

> 세션 `orch_0beb4d187a02`가 **성공 종료**(`orchestrator.session.completed`, status=completed).
> Duration 4556.5s(~76분), 2885 messages. lock 제거됨 = 프로세스 정상 종료.

- **결과: 9/9 satisfied (success 2 = AC 8·9, externally-satisfied 7 = AC 1–7), failure 0.**
- **로컬 전체 테스트 독립 검증: 2508 passed, 12 skipped, 0 failed** (baseline 1827 → +681).
  - 실행 중 잠깐 보였던 스트리밍 TTFT 실패 2건은 **최종 버전에서 해결됨**(전부 통과).
- **사후 QA verdict: 0.71 / 1.00 [REVISE]** (threshold 0.80, "Loop Action: continue").
  → ⚠️ **이 점수는 실제 테스트 실패가 아니라 "증거 가시성/감사성" 문제**임:
  7개 AC가 externally-satisfied라 in-run 증거 없음 / 리포트 truncation / `decomposition_depth_warning`
  (AC8 leaf 8.1.1–8.1.3, 8.3.1–8.3.2) / **"not a git repository"**(아래 주석 참고).
  실제 구현은 9/9 + 2508 테스트 통과로 건전함.

> **"not a git repository" QA 지적 해설**: 우리 프로젝트 디렉토리는 git 초기화돼 있고(워크플로가
> 실제로 `56250c6 feat(tripwire)` 커밋까지 남김), QA 검증 단계가 **다른 cwd**(`~/.ouroboros/seeds`
> 또는 temp)에서 git을 조회해 나온 메시지임. 우리 repo 자체는 정상.

### 🚨 AC 10(NER)은 이 실행에 **없었음** — 여전히 미구현
- 이 세션은 AC 10 추가 전 **9-AC 스냅샷**으로 시작됨. → NER은 구현 안 됨, `_run_ner()` 스텁 그대로.
- NER은 **3차 실행을 새로 시작**해야 함 (resume 아님 — 새 `run workflow`로 10-AC 시드 재로드).
  명령은 §6.A-2 참고.

---

## 1. 프로젝트 위치 & 환경

| 항목 | 값 |
|---|---|
| 프로젝트 루트 | `/Users/ho/workspace/Monoly_genAI/pii_guard/` |
| 요구사항 문서 | `/Users/ho/workspace/Monoly_genAI/pii-guard-requirements.md` |
| Seed 파일 | `/Users/ho/.ouroboros/seeds/seed_5cfc8e8ae623.yaml` (현재 **10 AC**) |
| 델타 마커 | `pii_guard/docs/delta/completed_round1.yaml` (AC 1–7 완료 표시) |
| ouroboros 패키지 | `uvx --from "ouroboros-ai[mcp,claude]" ouroboros ...` |
| Python | 3.9.6 (시스템) / pytest 8.4.2 / pyyaml 설치됨 |
| 1차 워크플로 세션 | `orch_9b02af28c3eb` (1827 passed 당시) |

> ⚠️ 이 프로젝트는 원래 `/Users/ho/pii_guard`에 생성됐으나, 이번 세션에서 `Monoly_genAI/pii_guard/`로 **이동**했음.

---

## 2. 현재 git 상태 (스냅샷)

**커밋 3개 (clean 부분):**
```
342d41b  Delta scope: include AC 10 (real Stage2 NER) alongside AC 8-9
29799e7  Add delta round-1 skip-completed marker (ACs 1-7 done, run AC 8-9)
89dc5c5  Initial commit: PII-Guard local-first LLM gateway (ouroboros workflow output)
```

**워크플로/정리 커밋 추가됨 (2차 델타 종료 후):**
- `56250c6 feat(tripwire): Sub-AC 8.2 — full-body tripwire sweep` ← **워크플로가 직접 커밋**
  (git init 해둔 덕에 ouroboros가 우리 repo에 커밋 가능했음).
- 그 외 AC 8·9 산출물은 워크플로가 커밋하지 않아, **이번 세션이 검증 후 커밋함** (아래 파일들).

**2차 델타 최종 AC 8·9 산출물 (검증 완료, 커밋함):**
```
AC 8: pii_guard/providers/claude_parser.py, openai_parser.py, gemini_parser.py
      pii_guard/providers/schema_coverage.py   (프로토콜 스키마 커버리지)
      pii_guard/providers/coverage_alarm.py    (unknown_field 커버리지 알람)
      pii_guard/tripwire.py                     (전체바디 트립와이어, 56250c6에 포함)
AC 9: pii_guard/streaming_buffer.py            (경계 룩어헤드 버퍼)
      pii_guard/streaming_rehydrator.py         (스트리밍 복원)
+ 각 test_*.py, providers/__init__.py 수정
```
> ⚠️ 중간 스냅샷의 단일 `tripwire.py` 구상이 최종엔 `schema_coverage.py` + `coverage_alarm.py` +
> `tripwire.py` 조합으로 재구성됨.

---

## 3. 테스트 현황 (2차 델타 최종, 독립 검증)

| 범위 | 결과 | 비고 |
|---|---|---|
| **전체 (AC 8·9 포함 최종)** | **2508 passed, 12 skipped, 0 failed** | baseline 1827 → +681 신규 |
| 1차 baseline | 1827 passed, 12 skipped | 참고 |

- skip 12개 = root+pf(4)+실네트워크 통합테스트(정상).
- ✅ **이전에 실패했던 스트리밍 TTFT 테스트 2건은 최종 버전에서 통과** (워크플로가 버퍼 로직 수정함).
- 캐노니컬 테스트 명령: `python3 -m pytest -q -p no:cacheprovider` (프로젝트 루트에서).

---

## 4. Seed 구조 — 10개 AC 상태

`seed_5cfc8e8ae623.yaml`, ambiguity 0.11. AC 번호는 1-based, `acceptance_criteria` 순서와 일치.

| AC | 이름(exit_condition) | 내용 | 상태 |
|---|---|---|---|
| 1 | — | 시크릿/고위험 신원 차단 + 연락처 PII 마스킹, 멀티프로바이더 | ✅ 완료(커밋) |
| 2 | RoundTripLossless | 인덱스 플레이스홀더 + 로컬 rehydration 무손실 왕복 | ✅ 완료 |
| 3 | FailureHandlingCorrect | fail-closed + Stage2 degrade + 프록시 크래시 fail-closed | ✅ 완료 |
| 4 | LedgerNoRaw | append-only Ledger, HMAC만, 600/700 권한, 회전/보존/퍼지 | ✅ 완료 |
| 5 | PolicyUnifiedHotReload | 단일 hot-reload YAML 정책 + 보안 기본값 + pin-list 승인 | ✅ 완료 |
| 6 | EnforcementHonest | 기본 on 인터셉션 + opt-in egress 락다운 + 정직한 경계 선언 | ✅ 완료 |
| 7 | DetectionValidated | 서명/수동 업데이트 + 한국어 레드팀 코퍼스 + 골든 회귀 | ✅ 완료 |
| **8** | **ProtocolSurfaceCovered** | 구조 파서 + 전체바디 트립와이어 + unknown_field_action | ✅ **2차 델타 완료, 커밋됨, 테스트 통과** (단 §0.5 depth-warning 잔존) |
| **9** | **StreamingRehydrationSafe** | SSE 청크 경계 룩어헤드 버퍼링 복원 | ✅ **2차 델타 완료, 커밋됨, TTFT 테스트 통과** |
| **10** | **Stage2EngineImplemented** | 실제 Presidio + spaCy 한국어 NER (스텁 대체) | ❌ **미구현. 2차 델타(9-AC 스냅샷)에 미포함 → 3차 새 실행 필요** |

> AC 8·9는 처음엔 9-AC Seed의 보강②①였고, AC 10(NER)은 이번 세션에서 추가됨.
> **2차 델타 세션(`orch_0beb4d187a02`)은 AC 10 추가 전 9-AC 시드로 시작됐으므로 AC 8·9만 다룸.**
> AC 10은 §0.5 참고 — 별도 3차 실행 필요.

---

## 5. 이번 세션에서 한 일 (시간순)

1. **프로젝트 이동**: `/Users/ho/pii_guard` → `Monoly_genAI/pii_guard/`.
2. **이동 부작용 수정**: `test_boundary.py`·`test_egress_lockdown_integration.py`에 하드코딩된
   `cwd="/Users/ho/pii_guard"` → `__file__` 기반 `_REPO_ROOT`로 동적화. (14 failed → 0)
3. **git 초기화 + baseline 커밋** (`89dc5c5`), `.gitignore` 작성(ledger/key 등 런타임 산출물 차단).
4. **1차 결과 검토**: 5개 체크리스트 대조 → AC 8·9 미구현, AC 10(NER) 스텁 확인.
5. **Seed 보강**: AC 8·9는 직전(다른 작업)에 추가돼 있었고, 이번에 **AC 10(Stage2EngineImplemented) 추가** → 10/10.
6. **델타 마커 작성·커밋** (`29799e7`, `342d41b`): AC 1–7 완료 표시 → 델타가 AC 8·9·10만 실행.
7. (이 시점에) 2차 델타가 AC 8·9 산출물을 working tree에 생성한 것으로 보임.

---

## 6. 다음 세션 할 일 (체크리스트)

### A. 먼저 2차 델타 상태 확인 (AC 8·9)
세션 `orch_0beb4d187a02`가 **아직 실행 중인지 / 끝났는지 / 끊겼는지** 확인 (§0.5의 쿼리 사용,
`ls ~/.ouroboros/locks/`).
- **끊겼다면 resume** (AC 8·9 이어감, **NER 아님**):
  ```bash
  cd /Users/ho/workspace/Monoly_genAI/pii_guard
  ouroboros run workflow /Users/ho/.ouroboros/seeds/seed_5cfc8e8ae623.yaml \
    --orchestrator --resume orch_0beb4d187a02 \
    --skip-completed docs/delta/completed_round1.yaml
  ```
- **끝났다면** → §B 검증으로.

### A-2. 3차 실행 — AC 10(NER), **새 run으로 시작 (resume 아님)**
> ⚠️ 2차 세션은 9-AC 스냅샷이라 AC 10이 없음. resume으로는 NER이 안 됨. **반드시 새 `run workflow`**로
> 10-AC 시드를 다시 로드해야 함. AC 8·9가 working tree에 이미 있으면 마커에 추가해 skip 권장.
```bash
cd /Users/ho/workspace/Monoly_genAI/pii_guard
# (선택) AC 8·9 검증·커밋 후 completed_round1.yaml에 8,9 추가 → AC 10만 실행됨
ouroboros run workflow /Users/ho/.ouroboros/seeds/seed_5cfc8e8ae623.yaml \
  --runtime claude \
  --skip-completed docs/delta/completed_round1.yaml
```
- NER은 무거움: spaCy 한국어 모델(`ko_core_news_*`, 수백 MB) + `presidio-analyzer` 설치 필요.
  인터넷 + 디스크 여유 확인. `pyproject.toml`에 의존성 추가될 것.

### B. 검증 (델타 완료 후)
- [ ] **AC 9 TTFT 실패 2건 원인 규명** — flaky인지 실제 결함인지. 실제면 버퍼 방출 로직 수정.
- [ ] **AC 10 확인** — `pii_guard/stage2/_workers.py`의 `_run_ner()`가 스텁이 아니라 실제
      spaCy/Presidio 호출인지. `grep -n "spacy\|presidio" pii_guard/stage2/`.
- [ ] **AC 8 확인** — `tripwire.py`가 미파싱 필드 PII를 실제로 잡는지, `*_parser.py`가
      `unknown_field_action`으로 미지 필드를 차단/경고하는지.
- [ ] **전체 테스트** `python3 -m pytest -q -p no:cacheprovider` → 0 failed 목표.
- [ ] **degradation 보존 확인** — NER 추가 후에도 OOM/타임아웃 시 Stage1 폴백 유지되는지
      (`test_stage2_degradation.py` 통과).

### C. 정리
- [ ] 검증 통과 시 working tree 변경 **커밋** (AC 8·9·10 묶거나 단계별로).
- [ ] 통과 시 `docs/delta/completed_round1.yaml`에 AC 8·9·10 추가하거나 round2 마커 작성.
- [ ] 최종 QA verdict 재확인 (1차 QA는 0.66 REVISE — 주로 git 부재·증거 가시성 문제였음, 현재 git 초기화로 일부 해소).

---

## 7. 주요 파일 맵 (`pii_guard/pii_guard/`)

| 파일 | 역할 |
|---|---|
| `engine.py` | 탐지 엔진 진입점 (Stage1+Stage2 통합) |
| `detector.py` `categories.py` | **Stage1** 정규식/패턴 탐지 (키·주민번호·이메일 등) |
| `stage2/runner.py` `_workers.py` | **Stage2** NER 서브프로세스 (현재 `_run_ner` 스텁) |
| `proxy.py` | 인터셉트 프록시 (base_url 주입) |
| `providers/{claude,openai,gemini}.py` | 프로바이더 와이어 포맷 파싱/마스킹 |
| `providers/{claude,openai,gemini}_parser.py` | 🆕 AC 8 구조 파서 (미커밋) |
| `tripwire.py` | 🆕 AC 8 전체바디 PII 스윕 (미커밋) |
| `streaming_buffer.py` `streaming_rehydrator.py` | 🆕 AC 9 스트리밍 경계 복원 (미커밋) |
| `response_rehydrator.py` | 응답 복원 (비스트리밍) |
| `masker.py` `vault.py` `session_map.py` | 마스킹/플레이스홀더/세션 매핑 |
| `ledger.py` | 감사 Ledger (HMAC, 600/700) |
| `policy.py` `decision.py` | 정책 로드/결정 엔진 |
| `pinlist_guard.py` `pinlist_approval.py` | pin-list 변경 out-of-band 승인 |
| `updater.py` | 서명 업데이트 메커니즘 |
| `boundary.py` `launcher.py` | 보호 경계 선언 / 프로세스 런처 |
| `cli.py` | CLI (`piiguard` 명령) |
| `corpus/korean_pii.py` | 한국어 레드팀 코퍼스 |

---

## 8. 핵심 설계 원칙 (요구사항 P1–P6, 절대 위반 금지)

- **P1** local-first — 외부 전송 없이 로컬에서 처리
- **P2** secure-by-default — 무설정으로 안전 (zero-config)
- **P3** 거짓 안심 금지 — 보호 못 하는 건 정직하게 선언
- **P4** 세 번째 금고 금지 — 새 비밀 저장소 만들지 않음
- **P5** 침묵 통과 금지 — 검사 못 하면 차단(fail-closed)하고 기록(coverage gap)
- **P6** 통제면 격리 — pin-list 등 통제는 에이전트가 아닌 사용자만

> 상세는 `pii-guard-requirements.md` 참조. NER 엔진/포맷 등 "어떻게"는 §5.2/§6.2/§12/§19에 있음.

---

## 9. 알려진 이슈 / 주의사항

1. **AC 9 TTFT 테스트 2건 실패** — §3 참고. 최우선 조사 대상.
2. **AC 10 NER 미구현** — `_run_ner()` 스텁. degradation 덕에 시스템은 돌지만 비정형 한국어
   PII(이름·주소·조직)는 못 잡음. **2차 델타(9-AC 스냅샷)에는 미포함** → §0.5/§6.A-2대로
   **3차 새 실행** 필요 (resume 아님).
3. **working tree 미커밋** — 델타 산출물이 커밋 안 됨. 워크플로를 또 돌리기 전에 커밋하거나
   stash 고려(충돌 방지).
4. **NER 정확도** — spaCy 한국어 기본 모델은 도메인 특수 이름/주소에서 한계 가능. 델타 후
   precision/recall 측정값 보고 임계값 조정/커스텀 학습 검토.
5. **presidio/spacy 미설치** — 현재 환경에 없음. AC 10 실행/테스트 전 설치 필요.

---

## 10. 빠른 재개 명령 모음

```bash
# 위치 이동
cd /Users/ho/workspace/Monoly_genAI/pii_guard

# 현재 상태 확인
git log --oneline && git status -s

# 전체 테스트 (커밋된 baseline)
python3 -m pytest -q -p no:cacheprovider

# 새 델타 테스트만
python3 -m pytest tests/test_tripwire.py tests/test_streaming_buffer.py \
  tests/test_streaming_rehydration_integration.py tests/test_*_parser.py -q

# Seed 상태 확인
python3 -c "import yaml; d=yaml.safe_load(open('/Users/ho/.ouroboros/seeds/seed_5cfc8e8ae623.yaml')); print('ACs:', len(d['acceptance_criteria']))"

# AC 10 (NER) 스텁 여부 확인
grep -n "_run_ner\|spacy\|presidio" pii_guard/stage2/_workers.py

# 2차 델타 실행
ouroboros run workflow /Users/ho/.ouroboros/seeds/seed_5cfc8e8ae623.yaml \
  --runtime claude --skip-completed docs/delta/completed_round1.yaml
```
