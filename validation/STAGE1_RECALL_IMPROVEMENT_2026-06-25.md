# Stage1 탐지 보강 — recall 개선 측정 (2026-06-25)

> FN(미검출) 분석에서 "GLiNER 미검출의 ~79%가 정형 PII(Stage1 영역)"임이 드러나, **Stage1 정규식/proximity를 보강**했다.
> 본 문서는 보강 전→후의 재현율을 **동일 입력·동일 백엔드(spaCy)**로 측정한 결과다(정형 PII는 NER 백엔드와 무관하므로 spaCy로 고정해 apples-to-apples 비교).

---

## 1. 보강 내용 (코드)

| # | 카테고리 | 변경 | 파일 |
| :-- | :-- | :-- | :-- |
| 1 | **KR_ACCOUNT** | 비표준 계좌 포맷을 일반화(하이픈 2~3개 + 자릿수 9~14 검증). **은행명/입금/계좌 트리거 근접 시에만 승격**(오탐 억제 유지). 3-2-6·3-4-5·4-4-4·3-4-4-2·3-6-2-3 등 포괄 | `proximity.py` |
| 2 | **PASSWORD** | 라벨 키워드에 `<prefix>_pass[word\|wd]` 추가 → `DB_PASS=`·`temporary_pass:` 등 접두형 라벨 인식. `pass` 단독은 제외(passport 오탐 방지) | `categories.py` |
| 3 | **PASSPORT** | 경계 `(?!\w)`→`(?![A-Za-z0-9])` — 한국어 조사(`M12345678를`)가 `\w`로 잡혀 매치가 깨지던 **조사 인접 버그** 수정 | `categories.py` |
| 4 | **TOKEN(JWT)** | 2번째 세그먼트의 `eyJ` 강제 제거(헤더 eyJ + 두 세그먼트). 변형·난독 JWT도 검출 | `categories.py` |
| 5 | **API_KEY(GitHub)** | `ghp_` 본체 길이 `{36,}`→`{20,}` — 짧은 변형 토큰도 검출(`ghp_` 접두가 특이해 오탐 위험 낮음) | `categories.py` |

> 모두 **결정적·감사가능**하며, KR_ACCOUNT는 문맥 게이팅(R17 proximity)으로 오탐을 억제한다.

## 2. 재현율 — 보강 전→후 (spaCy 백엔드, 동일 입력)

| 데이터셋(케이스) | BEFORE recall | AFTER recall | 정탐 TP | 비고 |
| :-- | --: | --: | :-- | :-- |
| **codex** (10) | 0.798 (71/89) | **0.921 (82/89)** | +11 | KR_ACCOUNT·TOKEN·API_KEY·PASSPORT 회수 |
| **gemini** (10) | 0.875 (63/72) | **0.958 (69/72)** | +6 | KR_ACCOUNT·TOKEN·API_KEY·PASSWORD 회수 |
| **claude** (30) | 0.941 (192/204) | 0.941 (192/204) | +0 | 잔여 미검출이 NER(이름·주소)·무효체크섬뿐 → Stage1 변경 무관 |

**정밀도는 유지·소폭 상승**(회귀 없음): codex 0.807→0.828, gemini 0.630→0.651, claude 0.877 동일.

## 3. 카테고리별 회수 내역 (codex+gemini, 미검출 FN)

| 카테고리 | codex BEFORE→AFTER | gemini BEFORE→AFTER | 결과 |
| :-- | :-- | :-- | :-- |
| **KR_ACCOUNT** | 6 → **0** | 3 → **0** | ✅ 전부 회수(9/9) |
| **TOKEN** | 2 → **0** | 1 → **0** | ✅ 전부 회수(3/3) |
| **API_KEY** | 1 → **0** | 1 → **0** | ✅ 전부 회수(2/2) |
| **PASSPORT** | 3 → 1 | — | ✅ 2/3 회수 |
| **PASSWORD** | 1 → 1 | 2 → **1** | ✅ 1/3 회수 |
| FOREIGN_REG | — | 2 → 2 | ➖ 채점 아티팩트(RRN 포맷을 FOREIGN_REG로 라벨) — 엔진은 RRN으로 정탐 |
| PERSON | 5 → 5 | — | ➖ NER 영역(Stage1 무관, GLiNER 백엔드의 몫) |

**정형 PII(Stage1) 미검출 합계: codex+gemini 27 → 10** (17건 회수).

## 4. 남은 갭

- **PASSWORD 2건**: 라벨 없이 본문에 흩어진 비밀번호 값(예: `Hjw!0623Reset`) — 라벨/키워드 단서가 없어 결정적 검출 곤란. (무리한 패턴은 오탐 폭증 → 보류.)
- **PASSPORT 1건**: 잔여 포맷/문맥 — 추가 점검 대상.
- **FOREIGN_REG 2건**: 실제로는 RRN 포맷 → 정답 라벨 오류(엔진은 RRN으로 정탐). 채점 아티팩트.
- **PERSON 5건(codex)**: NER 영역 → GLiNER 백엔드/임계값·라벨 보강의 몫(본 작업 범위 밖).

## 5. 검증

- **전체 테스트 2727 passed / 12 skipped / 0 failed** — 정밀도·기존 동작 회귀 없음.
- 기존 proximity 테스트(3-3-6·4-2-7 승격, 무문맥 미승격 FP 억제)도 모두 통과.

> 측정 재현: `validation/load_external_test.py`(codex/gemini), `efficacy_test.py`(claude) — 입력은 6개 외부 리포트 부록에서 재구성한 동일 데이터. 백엔드 `PIIGUARD_NER_BACKEND=spacy` 고정.
