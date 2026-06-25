# 🛡️ PII-Guard

**로컬 우선(local-first) LLM 게이트웨이 프록시 — 외부 LLM으로 나가는 요청에서 개인정보·시크릿을 로컬에서 탐지해 마스킹/차단합니다.**

내 PC에서 Claude / OpenAI / Gemini 같은 외부 LLM으로 데이터가 나갈 때, 그 안의 **PII**(이름·주민번호·전화·이메일·주소·계좌 등)와 **시크릿**(API 키·AWS 키·토큰·비밀번호 등)을 **외부로 한 글자도 보내지 않고 로컬에서** 탐지합니다. 카테고리별 정책에 따라 **마스킹**(`[PERSON_1]`로 치환)하거나 **차단**(요청 거부)합니다.

> 탐지는 **완전 로컬** — 외부 LLM을 탐지에 쓰지 않습니다(정규식 + 로컬 한국어 NER). 결정적·감사가능·프롬프트 인젝션에 안전합니다.

---

## ✨ 핵심 기능

- **인터셉트 프록시** — base_url만 바꾸면 모든 LLM 트래픽이 가드를 거칩니다. (Claude `/v1/messages`, OpenAI `/v1/chat/completions`, Gemini `/v1beta/...`)
- **하이브리드 탐지** — Stage1(정규식 + 체크섬: 키·카드·주민번호) + Stage2(한국어 NER: 이름·주소·조직).
- **마스킹 + 무손실 복원** — 외부엔 `[CAT_N]` 플레이스홀더만, 응답은 로컬에서 원본으로 복원(rehydrate).
- **secure-by-default** — 설정 0에서도 시크릿·주민번호 차단, 이메일·이름 마스킹.
- **fail-closed** — 검사 못 하면 차단, 프록시 크래시 시 무방비 직통 없음.
- **정직한 감사** — Ledger는 원본 없이 HMAC keyed-hash로만 기록.
- **관찰가능성** — `--log-masked`로 업스트림에 나가는 마스킹된 페이로드를 콘솔에서 확인.
- **대화형 UI** — Streamlit으로 채팅·파일 PII 즉시 확인.

## 📊 실효성 (검증됨)

합성 30케이스(한국어 ~1000자 + 영문 시크릿) 검증 결과:

| 지표 | 수치 (보정) |
| :-- | :-- |
| **재현율(Recall)** | **0.95** |
| **정밀도(Precision)** | **0.94** |

정형 PII(API키·카드·여권·이메일·전화)는 사실상 1.00. 전체 **2685 테스트 통과**. 상세 — [`validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md`](validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md).

---

## 🚀 빠른 시작

### 설치
```bash
# 가상환경 (Python 3.11 권장)
python3.11 -m venv .venv && source .venv/bin/activate

# Stage2 NER 백엔드 (택1 또는 둘 다)
pip install -e ".[ner-gliner,dev,ui]"   # 기본 백엔드 GLiNER (Apache-2.0, 상업 가능) — 모델 자동 다운로드
pip install -e ".[ner,dev,ui]"          # 경량 폴백 spaCy (Presidio)
python -m spacy download ko_core_news_lg  # spaCy 폴백용 한국어 모델 (lg 권장 / sm 경량)
```
> 백엔드 선택: `PIIGUARD_NER_BACKEND=gliner|spacy` (기본 `gliner`) 또는 정책 `stage2.ner_backend`. 모델 변형: `PIIGUARD_GLINER_MODEL` / `PIIGUARD_KO_SPACY_MODEL`. 상세·성능 비교 = [`validation/NER_BACKEND_COMPARISON.md`](validation/NER_BACKEND_COMPARISON.md).

### 1) 프록시 실행 — 실제 Anthropic 앞에 두기
```bash
# 터미널 A — 가드 띄우기 (--log-masked: 마스킹된 페이로드 콘솔 출력)
piiguard serve --upstream-url https://api.anthropic.com --port 4444 --log-masked

# 터미널 B — 클라이언트가 가드를 거치게
export ANTHROPIC_BASE_URL=http://127.0.0.1:4444
export ANTHROPIC_API_KEY=sk-ant-...
# 이제 claude CLI / SDK / curl 요청이 마스킹되어 전달됨
```

### 2) 대화형 UI
```bash
python -m streamlit run ui/app.py   # → http://localhost:8501
```

### 3) 라이브러리로 직접 사용
```python
from pii_guard import Engine

engine = Engine()
result = engine.scan("제 이름은 김민수, 전화 010-1234-5678, AWS 키 AKIAIOSFODNN7EXAMPLE")
print(result.redacted_text)
# 제 이름은 [PERSON_1], 전화 [PHONE_1], AWS 키 [AWS_SECRET_1_BLOCKED]
print(result.has_blocks)   # True (시크릿 포함 → 프록시 경로에선 요청 차단)
```

### 4) API 키 없이 빠른 검증 (E2E 스모크)
```bash
python scripts/e2e_smoke.py   # 프록시+mock 자동 구동: 마스킹/차단/복원 확인
```

---

## 🔍 탐지 카테고리 (20종)

| 클래스 | 카테고리 | 기본 액션 |
| :-- | :-- | :-- |
| 시크릿 | API_KEY, AWS_SECRET, GCP_KEY, TOKEN, PRIVATE_KEY, PASSWORD | **차단** |
| 고위험 신원 | RRN(주민번호), FOREIGN_REG, PASSPORT, DRIVER_LICENSE, CARD | **차단** |
| 연락·식별 | EMAIL, PHONE, KR_ACCOUNT(계좌), BIZ_NO(사업자) | 마스킹 |
| 문맥(NER) | PERSON(이름), ADDRESS(주소), ORGANIZATION(조직) | 마스킹 |
| 서버정보 | IP_ADDRESS(IPv4), HOSTNAME(내부 호스트명) | 마스킹 |

체크섬 검증(주민번호·카드 Luhn·사업자번호)으로 오탐을 억제하고, **proximity(근접 문맥)** 규칙으로 비표준 계좌·한글 비번 라벨까지 잡습니다.

---

## 🧱 아키텍처 (요약)

```
ouroboros / LLM CLI ──▶  PII-Guard 프록시  ──▶  api.anthropic / openai / gemini
                         1. 프로바이더 구조 파서 + 트립와이어
                         2. Stage1(정규식·체크섬) → Stage2(NER, 별도 워커)
                         3. 정책: block / mask / allow
                         4. 마스킹 → 전송, 응답은 로컬 복원
                         ── 컨트롤플레인(정책·HMAC키·Ledger) 격리 ──
```

- **코어/워커 분리**: 무거운 NER은 별도 프로세스 — OOM이 코어를 못 죽임.
- **2-티어 강제**: 티어1(base_url 주입, 협조적) + 티어2(옵트인 egress 락다운, pf 방화벽).

상세: [`docs/DESIGN.md`](docs/DESIGN.md) · 요구사항: [`docs/pii-guard-requirements.md`](docs/pii-guard-requirements.md)

---

## 📁 프로젝트 구조

```
pii_guard/        탐지 엔진·프록시·정책·복원·Ledger (코어 패키지)
  providers/      프로바이더별 파서·스크러버·커버리지
  stage2/         NER 엔진(Presidio+spaCy) + 워커 + FP 필터
ui/               Streamlit UI
scripts/          E2E 스모크 하니스
validation/       30케이스 실효성 검증 + 리포트
benchmarks/       한국어 NER precision/recall 벤치마크
docs/             요구사항 · 설계(DESIGN) · 설계제안(proximity)
tests/            2685 테스트
```

---

## 🧪 테스트

```bash
.venv/bin/python -m pytest -q          # 2685 passed (NER 통합테스트 포함)
```
egress 락다운 통합테스트는 root + macOS pf(4) 필요: `sudo .venv/bin/python -m pytest -m integration`

---

## ⚠️ 보호 범위 · 한계 (정직 선언)

PII-Guard는 에이전트와 **같은 PC에서 도는 협조적 게이트웨이**입니다.

- ✅ **방어함**: 선의 에이전트의 무의식·자동 유출, 인젝션으로 탈취된 에이전트가 데이터플레인으로 시도하는 유출(컨트롤플레인 격리 전제).
- ❌ **범위 밖**: 에이전트가 **root/커널 권한**을 가진 경우 — 어떤 호스트 내 가드도 봉쇄 불가. 진짜 격리는 **VM/샌드박스**의 몫이며, PII-Guard는 그걸 할 수 있는 척하지 않습니다.
- 티어1(base_url 주입)은 *협조하는 도구에 한한 best-effort 필터*이고, 우회 불가 경계는 티어2(egress 락다운, 옵트인)입니다.

자세한 위협 모델: [요구사항 §2](docs/pii-guard-requirements.md).

---

## 📄 문서

- [요구사항 명세](docs/pii-guard-requirements.md) — 원칙·위협모델·정책·결정기록
- [설계 문서(DESIGN)](docs/DESIGN.md) — as-built 아키텍처·모듈맵·ADR
- [Proximity 설계](docs/design/PROXIMITY_DESIGN.md) — 탐지 보완 설계
- [실효성 검증](validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md) · [NER 백엔드 비교](validation/NER_BACKEND_COMPARISON.md)

---

*로컬 우선 · secure-by-default · 거짓 안심 금지.*
