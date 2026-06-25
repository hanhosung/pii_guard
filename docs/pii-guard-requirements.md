# PII-Guard — 로컬 LLM 유출 차단 보안 에이전트 요구사항 (v2)

> 작성일: 2026-06-22 · 개정: 13라운드 Socratic 인터뷰 반영 전면 개정
> 대상 런타임: ouroboros (Agent OS, 저장소 외부 도구) + 로컬 LLM 게이트웨이 프록시
> 타깃 머신: MacBook Air 2022 (Apple M2, 통합 메모리 8GB)
> 상태: Seed 생성 준비 완료 (남은 미정은 §22)

---

## 1. 개요 / 목표

**PII-Guard**는 내 PC에서 외부 LLM(Claude / OpenAI / Gemini 등)으로 데이터가 나갈 때,
그 안의 **개인정보(PII)**·**비밀값(시크릿/키/토큰)**을 **로컬에서** 탐지해
**카테고리별 정책에 따라 마스킹 또는 차단**하는 보안 에이전트다.

보호 대상 입력면은 둘:
1. **문서/파일 본문** (에이전트가 읽어 외부로 보내는 첨부·tool_result)
2. **요청 컨텍스트(프롬프트)** — 사용자가 무의식적으로 입력한 PII 포함

### 1.1 설계 원칙 (인터뷰에서 도출된 불변 원칙)

| # | 원칙 | 근거 |
| :-- | :-- | :-- |
| P1 | **로컬 우선** — 탐지·정책·복원이 외부로 안 나간다. 탐지에 외부 LLM을 쓰지 않는다. | R0 |
| P2 | **secure-by-default** — 설정 0에서도 시크릿·주민번호는 차단, 이메일·이름은 마스킹된다. | R5 |
| P3 | **거짓 안심 금지** — 보호 범위와 한계를 정확히 선언한다. 막지 못하는 것을 "막는다"고 하지 않는다. | R7·R12 |
| P4 | **세 금고를 만들지 않는다** — 프록시·복원맵·Ledger 어디에도 원본 시크릿을 영속 수집하지 않는다. | R8·R10·R11 |
| P5 | **침묵 통과 금지** — 검사 불가·미보호 통과·열화는 항상 가시적으로 기록한다. | R6·R9 |
| P6 | **에이전트를 섬기되 통제면은 격리** — 선의 에이전트의 기능은 제공하되, 가드의 설정·키·정책은 에이전트 권한 밖에 둔다. | R13 |

---

## 2. 위협 모델 · 신뢰 경계 (keystone)

> 이 절이 나머지 모든 결정의 의미를 확정한다. 위협 행위자를 못박지 않으면
> 같은 기능(pin-list·tokenize_roundtrip·base_url 주입)이 "보호 장치"인지 "우회 통로"인지가 미정으로 남는다.

### 2.1 위협 행위자 = "신뢰하나 탈취 가능 (trusted-but-compromisable)" — R13

피보호 에이전트는 **기본적으로 선의**이나, **프롬프트 인젝션 등으로 탈취될 수 있다**고 가정한다.

- ❌ **"에이전트는 착하니 실수만 막으면 된다"는 가정("신뢰-부주의" 모델)은 버린다.** 코딩 에이전트는 외부의 믿을 수 없는 자료(웹페이지·남이 준 파일·깃허브 리포 등)를 일상적으로 읽고, 그 안에 숨겨진 명령에 **속아서 조종당할 수 있다(프롬프트 인젝션).** 이렇게 **고의로 가드를 피해 데이터를 빼내려는 시도(적대적 우회)까지 방어 범위에 넣어야** 요즘 가장 흔하고 위험한 공격에 대응할 수 있다. 단순 실수만 가정하면 이 위협 앞에서 무력해진다.
- ❌ **반대로 "에이전트를 아예 못 믿으니 완전히 가둬버리자"는 가정("비신뢰-봉쇄 컨테이너" 모델)도 버린다.** PII-Guard는 에이전트와 **같은 컴퓨터·같은 파일시스템**에서 돌아간다. 그래서 에이전트가 마음만 먹으면 가드를 우회하는 설정으로 되돌리거나(프록시 대신 진짜 API로 직접 연결, `base_url` 되돌리기) **가드 프로세스를 그냥 꺼버릴 수 있다.** 이런 상황에서 "에이전트를 가둔다"고 말하는 것 자체가 **지키지도 못할 약속(거짓 안심)**이다. **진짜로 가두려면 VM이나 샌드박스**가 필요한데, 그건 PII-Guard가 아니라 **별개의 제품**이다.
- ✅ 채택: **데이터플레인은 섬기고, 컨트롤플레인은 격리**한다.

### 2.2 신뢰 경계 — 에이전트가 만질 수 있는 것 vs 절대 못 만지는 것 (양보 불가 원칙)

> 원래 용어: "control plane / data plane 분리 — 비협상 요구사항".
> 쉽게 말하면, 시스템을 **두 구역**으로 나누고 **에이전트의 손이 닿는 범위를 분명히 긋는다**는 뜻이다.

**비유 — 축구 경기의 심판:** 선수(에이전트)는 경기장(작업 공간)에서 자유롭게 뛴다. 하지만 선수가 **규칙집을 고치거나, 심판을 매수하거나, 비디오 판독 카메라를 꺼버릴 수는 없어야** 한다. 만약 선수가 경기 중에 규칙을 마음대로 바꿀 수 있다면 심판이 있으나 마나다. PII-Guard도 똑같다 — **에이전트가 가드의 통제 장치(규칙·키·기록)를 건드릴 수 있으면 가드는 무의미**해진다. 그래서 이 구역 분리는 **타협 불가(양보 불가)**다.

| 구역 | 무엇이 들어가나 | 에이전트의 권한 |
| :-- | :-- | :-- |
| **통제 구역 (control plane)** = "규칙·심판" | 정책 파일(무엇을 막을지), **HMAC 키**(기록 보호용 비밀키), Ledger(감사 기록), egress 차단 규칙, 프록시 프로세스 자체 | **쓰기 금지** — 읽지도 고치지도 끄지도 못함 (별도 계정 또는 root 소유) |
| **작업 구역 (data plane)** = "경기장" | 에이전트의 프로젝트 파일·작업 디렉토리 | **읽기/쓰기 자유** (에이전트 본연의 일) |

이 분리를 강제하는 3가지 구체 규칙:

- **"이건 PII 아님" 등록(pin-list/allowlist)은 사용자만 할 수 있다.** 에이전트가 직접 못 바꾸고, **사용자의 별도 승인(out-of-band)**이 필요하다. → 탈취된 에이전트가 **자기 유출을 "안전한 것"으로 둔갑**시키는 걸 막는다.
- **외부 차단(egress 락다운) 규칙은 root 소유**라, root 권한이 없는 에이전트는 그 규칙을 **지울 수 없다.**
- **프록시(가드 본체)는 에이전트가 함부로 못 죽이는** 별도 프로세스/계정으로 돈다. → 에이전트가 "가드를 꺼서 우회"하는 걸 막는다.

### 2.3 명시적 범위선

- ✅ **방어함**: 선의 에이전트의 무의식·자동 유출(tool_result에 실린 시크릿 등) + **인젝션으로 탈취된 에이전트가 데이터플레인을 통해 시도하는 유출**(통제면 격리 전제).
- ❌ **범위 밖**: 에이전트가 **root/sudo/커널 권한**을 가진 경우 — 같은 호스트의 어떤 가드도 봉쇄 불가. **VM/샌드박스(잠재 티어3)** 필요. 숨기지 않고 선언.

---

## 3. 아키텍처

### 3.1 차단 지점 — 로컬 게이트웨이 프록시

모든 외부 LLM 트래픽이 통과하는 **로컬 리버스 프록시**(예 `127.0.0.1:8788`). 멀티프로바이더 중계.

```
┌─────────────┐   ┌────────────────────── PII-Guard ──────────────────────┐   ┌────────────┐
│ ouroboros   │   │  [데이터플레인 처리]                                      │   │ api.anthropic │
│ 워크플로     │──▶│  1. 프로바이더별 구조 파서 + 트립와이어 스윕              │──▶│ api.openai    │
│ Claude CLI  │   │  2. 탐지: Stage1(정규식·체크섬) → Stage2(Presidio+NER)    │   │ generativelang │
│ Codex CLI   │◀──│  3. 정책: 카테고리별 block/mask/allow                     │◀──│  (응답)        │
│ Gemini CLI  │   │  4. 마스킹/복원(인바운드) + 5. Ledger(메타전용)            │   └────────────┘
└─────────────┘   │  ── [컨트롤플레인: 정책·HMAC키·Ledger·락다운규칙] 격리 ──  │
                  └────────────────────────────────────────────────────────┘
```

- **코어/워커 분할(가용성·R9·R13)**: 항상 떠 있는 **경량 포워딩+Stage1 코어**(저메모리) + **별도 Stage2 워커 프로세스**(무거운 NER, 죽어도 코어 생존). Stage2 OOM이 코어를 못 죽임.

### 3.2 공용 엔진 (재사용 라이브러리)

> 여기서 "재사용"은 **두 가지 의미**를 동시에 담는다.
> (가) **내부 재사용** — PII-Guard의 탐지 로직을 *하나의 라이브러리*로 만들어, 프록시·CLI·UI·테스트가 **같은 엔진을 공유**한다.
> (나) **외부 재사용** — 바닥부터 새로 만들지 않고, 검증된 **오픈소스 라이브러리 위에** 얹어 만든다.

#### (가) 내부 재사용 — "엔진은 프록시 전용 스크립트가 아니라 공용 라이브러리"

핵심 탐지 로직(`pii_guard.engine.Engine`)을 **독립 라이브러리**로 두고, 여러 진입점이 그것을 **import 해서 똑같이 재사용**한다. 한 곳에 박아둔 일회용 코드가 아니다.

```python
from pii_guard import Engine
result = Engine().scan("연락처 010-1234-5678")   # 어디서든 동일하게 호출
```

| 같은 `Engine`을 재사용하는 곳 | 용도 |
| :-- | :-- |
| `proxy.py` (게이트웨이) | 실제 트래픽 스캔 |
| `cli.py` (`serve`) | 프록시 기동·NER 연결 |
| `ui/app.py`·`ui/scanner.py` (Streamlit) | 채팅/파일 PII 확인 |
| `benchmarks/`·`tests/` | 정확도 측정·회귀 검증 |

→ 덕분에 "UI에서 잡힌 것 = 프록시에서 잡히는 것"이 **자동으로 일치**한다(같은 엔진이므로). 패키지 구조는 아래처럼 기능별로 나뉜다(원래 계획):

```
pii_guard/
  core/        # 탐지·정책·마스킹·복원맵 (순수 로직)
  detectors/   # stage1(regex/checksum/dict), stage2(presidio+spaCy-ko)
  policy/      # 단일 스키마 로딩/평가/핫리로드
  gateway/     # 멀티프로바이더 프록시(코어) + stage2 워커(별도 프로세스)
  parsers/     # 프로바이더별 구조 파서 + 트립와이어 스윕
  extractors/  # 문서 포맷별 텍스트 추출
  ledger/      # ouroboros Ledger 연동 (메타데이터 전용)
  ruleset/     # 룰·모델 버전·서명 검증
  config/      # 정책·신뢰경계 스키마
```

> **as-built 주의**: 실제 구현은 위 중첩 디렉토리 대신 **평면 `pii_guard/` 패키지**로 수렴했다(모듈맵은 [`DESIGN.md`](./DESIGN.md) §4). 위 표는 *기능 그룹*으로 읽으면 된다.

#### (나) 외부 재사용 — 어떤 오픈소스를 어떻게 응용했나

PII-Guard는 NER·YAML 파싱·HTTP 서버 같은 바퀴를 **다시 발명하지 않는다.** 아래 라이브러리를 가져다 *조립·응용*한다.

| 외부 라이브러리 | 무슨 역할 | PII-Guard에서 어떻게 응용했나 | 사용 모듈 |
| :-- | :-- | :-- | :-- |
| **GLiNER** (`gliner`, 기본 `urchade/gliner_multi_pii-v1` · Apache-2.0) | 제로샷 NER 엔진(**Stage2 기본 백엔드**) | 정규식이 못 잡는 **문맥상 이름·주소·조직**을 라벨 프롬프트(`사람`·`주소`·`조직`)로 직접 추출 → PERSON/ADDRESS/ORGANIZATION 매핑. 트랜스포머 기반(높은 재현율+정밀도)이라 별도 워커·옵션 설치(`[ner-gliner]`)로 격리. **상업 사용 가능** | `stage2/gliner_ner.py` |
| **spaCy** + `ko_core_news_lg` 모델 (+ **Microsoft Presidio**) | 한국어 NLP·NER 엔진(**경량 폴백 백엔드**) | 메모리가 빠듯한 환경용 경량 대안. Presidio `AnalyzerEngine`이 spaCy NER 결과를 **PII 카테고리·신뢰도로 매핑**(`NlpEngineProvider`·`SpacyRecognizer` 한국어 재구성). spaCy 라벨(PS/LC/OG)→엔티티 매핑, 조사("홍길동은"→"홍길동") 스트리핑 | `stage2/korean_ner.py` |
| **PyYAML** (`yaml.safe_load`) | YAML 파싱 | **단일 스키마 정책 파일**·pin-list 승인 파일을 로드(핫리로드). `safe_load`로 임의 객체 역직렬화 차단 | `policy.py`, `pinlist_approval.py` |
| **Streamlit** | 웹 UI 프레임워크 | 채팅 입력·다중 파일 업로드·콘솔 출력 **로컬 UI**(R16) | `ui/app.py` |
| **pytest** | 테스트 러너 | 2685개 단위·통합·효능 테스트 + precision/recall 게이트 | `tests/` |
| **Python 표준 라이브러리** | — | 외부 의존 없이 핵심 보안 기능 구현(아래) | 코어 전반 |

**표준 라이브러리를 핵심 보안에 응용한 부분(외부 의존 최소화 = 공격면 축소, P1):**

| stdlib 모듈 | 응용 |
| :-- | :-- |
| `http.server` (`BaseHTTPRequestHandler`/`HTTPServer`) | **인터셉트 프록시 서버** 본체 | 
| `urllib.request` | 마스킹된 페이로드를 **업스트림으로 포워딩** |
| `multiprocessing` (spawn 컨텍스트) | **Stage2 NER 워커 격리**(OOM이 코어를 못 죽이게) |
| `hmac` / `hashlib` | **keyed-hash**(Ledger 저엔트로피 PII 역산 방지) |
| `re` | Stage1 **정규식 패턴** 탐지 |
| `signal` / `os._exit` | **fail-closed**(크래시 시 TCP RST) |

> 핵심 원칙: **무거운 신경망(GLiNER/spaCy/Presidio)은 외부 의존으로 두되 별도 프로세스·옵션 설치(`[ner-gliner]`/`[ner]`)로 격리**하고, **보안 핵심(프록시·해시·차단)은 표준 라이브러리만으로** 구현해 의존 공격면을 줄였다. Stage2 NER 백엔드는 **GLiNER(기본)와 spaCy(경량 폴백) 중 선택 가능**(§6.2).

### 3.3 Python 환경 분리

ouroboros 본체(Python 3.14)와 별도로 **Stage2(GLiNER+PyTorch 또는 Presidio+spaCy)는 Python 3.11~3.12 별도 venv/별도 프로세스**로 구동(의존성 충돌·OOM 격리). 두 백엔드 모두 PyTorch 계열 의존을 별도 venv에 둔다.

---

## 4. 가로채기 · 강제 (2-tier) — R7·R8·R12

### 4.1 정직한 2-티어 강제 모델

> **"강제(enforcement)"란?** — 외부로 나가는 트래픽이 **반드시 가드(프록시)를 거치게 만드는 강제력**을 말한다.
> 가드가 아무리 잘 탐지해도, 트래픽이 가드를 **안 거치고 빠져나가면** 무용지물이다. 그 강제력의 **세기**에 따라
> 두 단계(티어)로 나눈다. 그리고 **각 티어가 무엇을 막고 무엇을 못 막는지 솔직히 선언**(P3 거짓 안심 금지)하는 것이
> 이 모델의 핵심이라 "정직한 2-티어"라 부른다.

**비유 — 공항 보안검색:**
- **티어1** = "보안검색대로 가세요" 안내판 + 여행사가 보안 레인으로 예약해 줌. **협조하는 승객**은 다 검색대를 통과하지만, 작정하고 **담을 넘는 사람**은 못 막는다.
- **티어2** = 공항 **전체에 펜스**를 치고 **검색대를 유일한 출구**로 만든다. 이제 **비협조자도 무조건** 검색대를 통과해야 한다.
- **범위 밖** = 공항을 소유한 사람(=root 권한)은 아무 문이나 열 수 있다 — 어떤 가드도 못 막는다(§2.3).

| 티어 | 어떻게 강제하나 | 막는 것 | **못 막는 것**(정직 선언) |
| :-- | :-- | :-- | :-- |
| **티어1 (기본) — 협조적 게이트웨이** | ouroboros가 자식 프로세스에 `ANTHROPIC_BASE_URL` 등 **base_url 환경변수를 자동 주입**(default-on). 그 env를 따르는 도구는 자동으로 프록시 경유 | 관리되는 도구(ouroboros·자식 CLI)의 **무의식·자동** 유출 | 별도 셸에서 직접 실행, `curl` 직접 호출, **하드코딩된 base_url**, env를 무시·삭제하는 도구, 미등록 프로바이더, **탈취된 에이전트가 base_url을 원래대로 되돌리는** 경우 |
| **티어2 (옵트인) — egress 락다운 = 진짜 강제** | **방화벽(pf)이 아웃바운드를 기본 차단(deny-by-default)** 하고 **프록시 IP만 화이트리스트**. 규칙은 **root 소유** | 위의 모든 것 **+ 모든 직결 우회**(curl·하드코딩 포함). 프록시 안 거치면 네트워크 자체가 안 나감 | 에이전트가 **root 권한**을 가진 경우(§2.3, 범위 밖 — VM/샌드박스의 몫) |

**이 모델의 3가지 설계 원칙:**

- **① 정직한 명명(P3).** 티어1의 PII-Guard는 **"협조하는 도구에 한해 동작하는 best-effort 필터"**일 뿐, *우회 불가능한 벽이 아니다*. 이 한계를 제품 화면·문서에 그대로 선언한다. ("다 막는다"고 거짓말하지 않는다.)
- **② 로컬 루트 CA를 설치하지 않는다(P4·R12).** 티어1 구조는 *클라이언트→프록시*는 localhost 평문, *프록시→프로바이더*는 **정상 TLS**다. HTTPS 트래픽을 강제로 들여다보려면 보통 **로컬 루트 CA**(인증서)를 깔아야 하는데, **이 CA가 탈취되면 그 PC의 모든 사이트 인증서를 위조**할 수 있는 초고가치 공격 표면이 된다. PII-Guard는 **그 CA를 아예 안 만든다.**
- **③ "base_url 무시 도구"는 가로채기가 아니라 차단으로 처리한다.** env를 무시하는 도구를 *몰래 가로채려면* ②의 위험한 CA가 필요하다. 대신 티어2(방화벽)로 그 **직결을 그냥 막아버린다.** 막힌 연결은 **fail-safe**(최악의 경우 = 연결 실패 = 유출 없음)지만, 가로채기는 CA라는 새 위험을 만든다. → **막기(block) > 가로채기(intercept).**

---

## 5. 입력 표면 · 스캔 대상 — R6·R12

### 5.1 스캔 범위 — 모든 의미 콘텐츠 필드 (tool_result 최우선)

외부로 나가는 페이로드(구조화 JSON)에서 **PII가 들어갈 수 있는 모든 필드**를 검사:

| 대상 | 검사 | 비고 |
| :-- | :--: | :-- |
| 사용자 message 텍스트 | ✅ | 무의식적 입력 |
| system 프롬프트 | ✅ | |
| tool_use 인자 | ✅ | 경로·인자 민감값 |
| **tool_result (도구 실행 결과)** | ✅✅ **최우선** | `cat ~/.aws/credentials` 등 — **가장 흔한 유출 경로** |
| 첨부 파일/문서 블록 | ✅ | §11 추출 후 동일 엔진 |
| tool 정의(JSON schema description) | ✅ | |
| multimodal parts (이미지) | ⚠️ | §11 (unscannable 정책) |
| 제어 필드(model, temperature, role) | ❌ | PII 없음 |

### 5.2 멀티프로바이더 파싱 — 구조 파서(주) + 트립와이어 스윕(안전망)

> **왜 필요한가?** Claude(Anthropic)·OpenAI·Gemini는 **요청 JSON의 구조(스키마)가 서로 다르다.** 같은 "사용자 메시지"라도
> Claude는 `messages[].content[]`, OpenAI는 `messages[].content`, Gemini는 `contents[].parts[]`에 담는다. 그래서 PII가
> 어디 들어있는지 찾으려면 **프로바이더마다 다른 지도(스키마)를 따라가야** 한다. 이걸 **2중(주 + 안전망)**으로 처리한다.

**비유 — 공항 수하물 검사:**
- **(주) 구조 파서** = 그 항공사의 **표준 가방 구조를 아는 검사관**. 어느 주머니·칸에 물건이 들어가는지 정확히 알고 **그 칸만 콕 집어 검사**한다(빠르고 오탐 적음).
- **(안전망) 트립와이어** = 그래도 혹시 몰라 **가방 전체를 금속탐지기로 한 번 더 훑는다.** 검사관이 모르던 **숨은 주머니**에서 뭔가 잡히면 → "여기 검사 못 한 칸이 있다"고 **경보**를 울린다.
- **(staleness 가드)** = 항공사가 **가방 디자인을 바꿔 새 칸이 생기면**, 검사관이 그냥 통과시키지 않고 **"못 보던 칸이다"라고 신고**한다.

#### (주) 프로바이더별 구조 파서 — 마스킹 결정의 주체

- **하는 일**: 각 프로바이더의 스키마를 알고 **PII가 들어갈 수 있는 필드만 구조적으로 순회**해 탐지·마스킹한다.
  - Claude: `system`, `messages[].content[]`(text / tool_use 인자 / **tool_result** / document 블록) …
  - OpenAI: `messages[].content`, `tool_calls[].function.arguments` …
  - Gemini: `contents[].parts[]` …
- **왜 "구조적"인가(정밀도)**: 바디 전체를 무작정 정규식으로 긁으면 **구조 키 이름**(예: 필드명 `model`, `role`)이나 제어값까지 PII로 **오탐**할 수 있다. 구조 파서는 *값이 들어가는 콘텐츠 필드*만 보므로 정밀하다.
- **구현**: `providers/{claude,openai,gemini}_parser.py`. 어떤 필드를 방문했는지는 `providers/schema_coverage.py`가 추적.

#### (안전망) 전체 바디 트립와이어 스윕 — 사각지대 포착

- **하는 일**: 구조 파서가 끝난 뒤, **마스킹된 페이로드 JSON 전체를 raw로 한 번 더 스윕**한다. 구조 파서가 **방문하지 않은 비표준·중첩 필드**에 PII-급 값이 남아 있으면 잡아낸다.
- **왜 "마스킹"이 아니라 "경보"인가**: 예상 못 한 필드에서 PII가 나왔다는 건 **구조 파서에 사각지대가 있다는 신호**다. 조용히 가려버리면 그 사각지대가 영영 안 보인다. 그래서 **무작정 마스킹이 아니라 커버리지 알람**(Ledger 기록 + strict 모드에선 **fail-closed로 차단**)으로 **드러낸다**(P5 침묵 통과 금지).
- **구현**: `tripwire.py`. block-급 PII가 사각지대에서 발견되면 요청 차단.

#### 프로토콜 staleness 가드 — API가 바뀌어도 새지 않게

- **문제**: 프로바이더가 **API 버전을 올리거나 새 필드를 추가**하면, 기존 구조 파서가 모르는 **새 PII 통로**가 생길 수 있다.
- **하는 일**: 프로바이더×API버전별로 **스키마를 핀 고정**한다. **미지 필드 / 미지 API 버전**을 만나면 그냥 통과시키지 않고 **커버리지 알람**을 울린다.
- **정책 노브**: `unknown_field_action: block | warn` (kr-strict 기본 **block** — 모르는 건 일단 막는다).
- **구현**: `providers/coverage_alarm.py`.

> **세 겹의 관계**: **구조 파서**(정밀·주력) → 놓친 게 있으면 **트립와이어**(안전망)가 경보 → 스키마 자체가 바뀌면 **staleness 가드**가 경보. 핵심은 셋 다 **"못 본 것을 조용히 통과시키지 않는다"**(P5)는 점이다.

### 5.3 멀티턴 — 블록 해시 캐시

매 턴 전체 히스토리 재전송 → **콘텐츠 블록 단위 해시 캐시**로 기검사 블록은 탐지 재실행 없이 이전 판정 재사용. **신규 블록만** Stage1+2. (latency 선형 증가 차단 + 세션 일관성 동시 확보.)

---

## 6. 탐지 엔진 (하이브리드) — R0·R9

> **"하이브리드"란?** 성격이 다른 **두 탐지 방식을 한 팀으로** 쓴다는 뜻이다. 하나는 **모양이 정해진 PII**(키·주민번호·카드)를
> 정확히 잡는 **정규식(Stage1)**, 다른 하나는 **모양이 없는 문맥형 PII**(이름·주소)를 잡는 **NER(Stage2)**다. 둘 다
> **완전히 로컬**(P1 — 탐지에 외부 LLM 안 씀)이고, MVP 1차부터 **Stage2까지 포함**한다.

**비유 — 바코드 스캐너 + 똑똑한 검사관:**
- **Stage1** = **바코드 스캐너**. 정해진 패턴(키·카드번호)을 **빠르고 정확하게** 읽는다. 단, 바코드 없는 물건은 못 읽는다.
- **Stage2** = **글을 이해하는 검사관**. "김민수"가 문맥상 **사람 이름**임을 안다 — 이름엔 정해진 모양이 없어 바코드 스캐너로는 불가능한 일.
- 둘은 **경쟁이 아니라 보완** 관계다. (§6.3의 "틀린 fast-path"가 바로 이 점을 오해한 것.)

| | **Stage1 (정규식·체크섬)** | **Stage2 (NER)** |
| :-- | :-- | :-- |
| 잡는 것 | 모양 있는 PII (키·주민번호·카드·이메일·전화) | 문맥형 PII (이름·주소·조직) |
| 방식 | 패턴 매칭 + 체크섬 | 한국어 모델의 의미 이해 |
| 속도 | 매우 빠름(목표 p50<20ms) | 상대적으로 느림(모델 추론) |
| 성격 | **결정적**(같은 입력=같은 결과) | 확률적(신뢰도 점수) |
| 위치 | 항상 상주(인프로세스) | **별도 워커 프로세스** |

### 6.1 Stage 1 — 결정적 탐지 (항상 동기, 목표 p50 < 20ms)

- **방식**: **정규식 패턴** + **체크섬 검증** + **사전(사내 키워드)**.
  - 체크섬 = 단순 패턴 매칭을 넘어 **숫자가 실제로 유효한지 산술 검증**. 주민번호 가중합, 카드 Luhn, 사업자번호 검증식.
  - → 무효 번호(랜덤 숫자열)를 걸러 **오탐(FP)을 크게 줄인다.** (그래서 §23.3의 "라벨 없는 비표준 계좌"처럼 일부는 일부러 안 잡힘.)
- **성격**: **결정적·재현 가능**(같은 입력은 항상 같은 결과) → 감사·테스트에 유리, **프롬프트 인젝션 불가**.
- **경량·상주**: 코어 프로세스 안에서 항상 돈다(모델 로딩 불필요).
- **구현**: `detector.py` + `categories.py`(20개 카테고리 패턴·체크섬).

### 6.2 Stage 2 — 문맥 탐지 (선택형 NER 백엔드: GLiNER 기본 / spaCy 폴백, 타임아웃 인라인)

- **하는 일**: 정규식이 못 잡는 **문맥상 이름·주소·조직**(PERSON/ADDRESS/ORGANIZATION)을 의미로 탐지.
- **선택형 백엔드 (R18)**: Stage2 NER 엔진은 **두 가지 중 하나를 선택**해 동작한다. 두 백엔드 모두 같은 입력→같은 PII-Guard 카테고리(PERSON/ADDRESS/ORGANIZATION)로 정규화되며, 후단(정책·마스킹·proximity 후필터)은 백엔드와 무관하게 동일하다.

  | 백엔드 | 엔진 | 위치 | 성격 |
  | :-- | :-- | :-- | :-- |
  | **`gliner` (기본)** | GLiNER 모델 — 기본 **`urchade/gliner_multi_pii-v1`**(Apache-2.0, 상업 가능). 라벨 프롬프트(`사람`·`주소`·`조직`)로 제로샷 추출 후 카테고리 매핑 | `stage2/gliner_ner.py` | 트랜스포머 기반 → **재현율+정밀도 우위**. PyTorch 의존(더 무거움). 옵션 설치 `[ner-gliner]`. 비상업 시 `taeminlee/gliner_ko`(CC-BY-NC) 선택 가능 |
  | **`spacy` (경량 폴백)** | Microsoft Presidio + spaCy `ko_core_news_lg`(폴백 `sm`). spaCy 라벨(PS/LC/OG)→엔티티 매핑 | `stage2/korean_ner.py` | 가벼움·빠름 → **메모리 제약(8GB 빠듯) 환경**용. 옵션 설치 `[ner]` |

- **백엔드 선택 방법 (둘 다 지원)**:
  1. **환경변수** `PIIGUARD_NER_BACKEND=gliner|spacy` (미설정 시 기본 `gliner`).
  2. **정책 YAML** `stage2.ner_backend: gliner|spacy` (핫리로드). env가 있으면 env 우선.
  3. 각 백엔드의 **모델 변형**도 오버라이드 가능 — GLiNER는 `PIIGUARD_GLINER_MODEL`, spaCy는 기존 `PIIGUARD_KO_SPACY_MODEL`.
  - 추상화: `resolve_ner_backend()`가 백엔드를 결정하고, 백엔드별 리졸버가 모델을 고른다. 자세한 근거는 [`DESIGN.md`](./DESIGN.md) ADR-9·ADR-10·**ADR-11**.
- **블록당 하드 타임아웃**(기본 ~500ms~1s): 모델이 멈추거나 너무 느리면 무한 대기하지 않고 **끊고 `stage2_fail_action`으로 처리**(§14 열화). → 가용성 보장. (백엔드 무관 동일)
- **격리**: 무거운 모델은 **별도 워커 프로세스**(spawn)에서 돈다 → Stage2가 죽어도(OOM 등) 코어는 생존(§3.1). GLiNER가 더 무겁기 때문에 이 격리는 GLiNER 기본 채택의 전제다.
- **워밍업(필수)**: GLiNER 콜드 로드(~14.4s)가 블록당 타임아웃(기본 10s)을 초과하면 매 요청이 degrade돼 이름·주소가 누출된다. `serve`는 시작 시 `Stage2NERRunner.warmup()`으로 모델을 **블록 타임아웃 밖**에서 1회 로드한 뒤 트래픽을 받는다. (검증 = `tests/test_ner_backend.py`)
- **백엔드 성능 비교**(정탐/오탐/미탐 분류 포함) = [`validation/NER_BACKEND_COMPARISON.md`](../validation/NER_BACKEND_COMPARISON.md)
- **구현**: `stage2/gliner_ner.py`·`stage2/korean_ner.py`(백엔드 엔진) + `stage2/runner.py`(워커·타임아웃·열화, 백엔드 공통).

> **구현 상태(정직 선언, P3)**: dual-backend **설계·배선·실측 모두 완료**. `stage2/backend.py`(`resolve_ner_backend`)·`stage2/gliner_ner.py`(`GLiNERNEREngine`)·`_workers.py` 분기·`policy.py` `stage2.ner_backend` 파싱·`Engine(ner_backend=…)` env 전파·`serve` 연결·워밍업이 모두 배선되고 **선택 로직은 단위 테스트로 검증**됨(`tests/test_ner_backend.py`). **GLiNER 런타임·품질 실측 완료**(기본 `urchade/gliner_multi_pii-v1`, Apache-2.0): 코퍼스 PERSON recall 0.84→**0.956**, 외부 6리포트에서 spaCy 대비 재현율 동등~우위 + 정밀도 우위(codex 0.81→0.93, gemini 0.63→0.91). 6개 리포트·종합 대조 = [`validation/NER_BACKEND_COMPARISON.md`](../validation/NER_BACKEND_COMPARISON.md).

### 6.3 올바른 Fast-path & 자원 예산 (요구사항이지 단순 튜닝 아님) — R9·R13

- **🔴 흔한 착각 — "Stage1이 깨끗하면 NER 건너뛰자"는 틀린 fast-path.**
  - 왜 틀렸나: **NER(Stage2)의 존재 이유가 바로 "정규식(Stage1)이 못 잡은 것"을 잡는 것**이다. Stage1이 아무것도 못 찾았다는 건 오히려 NER이 **가장 필요한 순간**일 수 있다.
  - 비유: 바코드 스캐너가 아무것도 못 읽었다고 "가방은 깨끗해"라며 검사관을 돌려보내면, **바코드 없는 위험물**이 그대로 통과한다.
- **✅ 올바른 fast-path (성능을 위해 정당하게 건너뛰는 경우):**
  1. **콘텐츠 해시 캐시 적중**(§5.3) — 이미 검사한 동일 블록이면 재탐지 생략(결과 재사용).
  2. **콘텐츠 클래스 게이팅** — **자연어성 텍스트만** NER에 보내고, **순수 코드·base64·hex blob**은 스킵. (이런 데이터엔 사람 이름이 없어 NER이 무의미하고 오탐만 늘림.)
     > **as-built(R17)**: 스팬 사전 게이팅 대신 **NER 후필터**(`stage2/ner_filters.py`)로 더 정밀하게 실현됨 — 코드토큰·약어·blob·일반명사 deny-list를 탐지 후 제거. 정밀도 0.79→0.93. (§23.2 R17, DESIGN §6.6)
- **메모리 상주 전략**(8GB 환경): **lazy-warm**(첫 요청에 모델 지연 로드) + **idle 퇴거**(`model_idle_evict_seconds` — 안 쓰면 메모리에서 내림) + **메모리 예산 상한**(`memory_budget_mb`, 목표 ~≤1~1.5GB).
- **지연 예산(목표)**: 신규 블록당 추가 지연 **p50 < 200ms / p95 < 800ms**. (※ "예산을 둔다"는 것은 요구사항이고, 구체 수치는 환경에 맞춰 튜닝.)

---

## 7. PII / 시크릿 카탈로그 (kr-strict 기본 프로파일) — R5

| 카테고리 | 예시 | 기본 액션 |
| :-- | :-- | :-- |
| **시크릿/크리덴셜** `API_KEY, AWS_SECRET, GCP_KEY, TOKEN(JWT), PRIVATE_KEY, PASSWORD` | `sk-…`, `ghp_…`, `-----BEGIN … KEY-----` | **block** |
| **고위험 신원** `RRN(주민번호), FOREIGN_REG(외국인등록), PASSPORT, DRIVER_LICENSE, CARD` | 900101-1234567 | **block** |
| **연락/식별** `EMAIL, PHONE, KR_ACCOUNT(계좌), BIZ_NO(사업자)` | 010-1234-5678 | **mask** |
| **문맥(NER)** `PERSON, ADDRESS, ORGANIZATION` | 홍길동, 도로명주소, 삼성전자 | **mask** |
| **서버 토폴로지** `IP_ADDRESS(IPv4), HOSTNAME(내부)` | 10.0.12.45, prod-api-01.internal | **mask** |
| **저위험** `DOB(생년월일)` | 1990-01-01 | **allow** |

- PII와 시크릿은 같은 메커니즘이되 카테고리 분리. **한국 특화 항목은 1급 카테고리.**
- **확장형**: 사용자가 커스텀 카테고리(Presidio recognizer + 정책 엔트리) 추가 가능.

---

## 8. 정책 엔진 (단일 스키마) — R5·R2·R10

모든 노브를 **한 파일·한 스키마**에 통합. 핫리로드(프록시 재시작 불필요, 로드 실패 시 직전 유효 정책 유지).

### 8.1 해석 우선순위 (레이어)
`내장 secure default(kr-strict) < 사용자 정책 파일 < 채널 override < allowlist 예외`
- 내장 기본은 **바이너리 baked-in** → 정책 파일을 지워도 보호가 꺼지지 않고 secure default로 폴백(P2).

### 8.2 통합 스키마 골격
```yaml
version: 1
defaults_profile: kr-strict
fail_mode: closed              # 콘텐츠 실패 시 차단
on_infra_failure: degrade      # 인프라(엔진/프록시) 실패 시: degrade | block  (§14)
on_content_failure: block
credential_mode: passthrough   # passthrough | proxy_held  (§10)
egress_lockdown: false         # 티어2 강제 (옵트인, root)  (§4)
unknown_field_action: block    # 미지 프로토콜 필드 (§5.2)

stage2:
  ner_backend: gliner          # gliner(기본) | spacy(경량 폴백)  (§6.2, R18 / env PIIGUARD_NER_BACKEND 우선)

categories:
  RRN:        { action: block }
  API_KEY:    { action: block, fallback: tokenize_roundtrip }  # 시크릿 든 파일 편집 허용 옵션
  EMAIL:      { action: mask, mask_style: tokenize, rehydrate: true, min_confidence: 0.6 }
  PERSON:     { action: mask, min_confidence: 0.7, stage2_fail_action: degrade }
  ADDRESS:    { action: mask, stage2_fail_action: degrade }
  DOB:        { action: allow }

channels:
  cli/codex: { PERSON: { action: allow } }

# pin-list/allowlist 변경은 컨트롤플레인 — 사용자 승인 필요 (§2.2)
allowlist:
  - pattern: "test@example.com"
custom_categories:
  INTERNAL_CODE: { recognizer: "regex:PRJ-\\d{4}", action: mask }

diagnostics: { enabled: false, retention_days: 30 }
budgets: { memory_budget_mb: 1500, stage2_timeout_ms: 800, model_idle_evict_seconds: 600 }
```

### 8.3 액션 의미 (allow < mask < block)
- **allow**: 원본 통과.
- **mask**: 구조 보존·값 제거(작업 진행 가능). `mask_style: tokenize | partial | format_preserving`.
- **block**: 메시지 자체 미전송.
- mask ≠ block — 플레이스홀더가 구조(누가-누구·개수·관계)를 보존하므로 요약·분류·코드생성 등 **원본값이 결과에 불필요한 작업은 정상 동작**.

---

## 9. 마스킹 · 토큰화 · 복원 — R3·R10

### 9.1 토큰 형식 & 세션 일관성
- **인덱스 플레이스홀더 `[CATEGORY_N]`**(예 `[EMAIL_1]`, `[PERSON_2]`). 단순 라벨 배제(구분 불가), 형식보존 더미값은 옵트인(`format_preserving`).
- **세션 일관성**: 같은 원본 → 같은 토큰(LLM이 "같은 사람" 문맥 유지). 키 = 정규화 원본 해시. 세션 메모리 한정, 디스크 영속 없음.

### 9.2 응답 복원(rehydration) — 경로별 스코프 (R3를 R10이 정제)
- **에이전트 왕복 콘텐츠(tool_result·파일 블록) = 복원 ON 기본**:
  - **인바운드 응답 경로**에서, 에이전트에게 넘기기 직전 프록시가 세션 맵으로 `[CAT_N]` → 실제값 복원.
  - → 외부 LLM은 플레이스홀더만 봄(유출 차단) + 에이전트가 되써넣는 건 실제값(**데이터 파괴 없음**) + 복원은 로컬 전용(외부 미유출).
  - **이유**: 복원 OFF는 에이전트 루프에서 **사용자의 진짜 시크릿을 영구 파괴**하는 자기모순을 낳음. 복원의 보안 우려(맵=부채)는 외부 유출과 무관.
- **사람에게 보여주는 종단 출력 = 복원 OFF** 유지(엄격).
- **미복원 토큰 탐지**: 응답에 복원 못 한 잔여 플레이스홀더가 block-카테고리면 그 응답을 **경고/보류**(조용한 손상 방지).
- `tokenize_roundtrip`(시크릿 옵션): LLM에 노출 없이 **시크릿 든 파일을 에이전트가 편집**하게 허용. 기본 block 유지, 왕복 필요 시 선택(맵 실패 시 위험은 block이 안전).

---

## 10. 자격증명 처리 — R8·R12

- **기본 `credential_mode: passthrough`**: 에이전트 요청 헤더의 실제 키를 **보되 저장 안 함** → 프록시가 새 시크릿 금고 안 됨(P4).
- 옵트인 `proxy_held`: 프록시가 크레덴셜 보관·주입(클라이언트에서 키 분리). Ledger raw와 동일 규율(암호화·600·경고). **금고가 됨을 알고** 선택.
- **인증 헤더 마스킹 역설 해소**: 탐지·마스킹은 **메시지 바디에만**. 포워딩 전송 헤더(고정 allowlist: `Authorization, x-api-key, anthropic-*, openai-*`)는 **마스킹 제외**.
- **제외가 밀반출 통로 안 되게**: 제외 헤더는 "안 가린다"지 "안 본다"가 아님 — **정상 프로바이더 키 shape 검증**. 키 아닌 PII-스러운 데이터가 인증 헤더에 실리면 **이상 탐지→차단**. 바디엔 제외 절대 미적용.

---

## 11. 문서 / 파일 · 비텍스트 처리 — R6·R9·R11

| 포맷 | MVP | 처리 |
| :-- | :--: | :-- |
| txt/md/csv/json/소스코드 | ✅ | 직접 텍스트 |
| docx/xlsx/pptx | ✅ | 텍스트 추출 |
| pdf(텍스트) | ✅ | 추출 후 스캔 |
| pdf(스캔 이미지)·이미지 | ✕ | unscannable → 아래 |
| **hwp/hwpx (한글)** | ✕ **2차** | MVP 제외 |

- 파일 단위 결과가 block이면 **파일 전체 전송 차단** + 사유. 대용량은 청크 스캔.
- **비텍스트/파싱 불가(unscannable)**: 침묵 통과 금지(P5). `unscannable_action`:
  - 파싱 시도 실패 문서/바이너리 → 기본 **block**(fail-closed).
  - 이미지 → `block | warn_allow | ocr`. **kr-strict 기본 block**(스크린샷 속 주민번호). `warn_allow`는 통과시키되 **"미보호 통과"로 Ledger 기록**. **OCR은 2차.**

---

## 12. 스트리밍 처리 — R11

- **위험 방향(아웃바운드 요청)은 스트리밍 아님** — 전체 바디 확보 후 전송 → **egress 탐지·마스킹에 청크 문제 없음.**
- 스트리밍 이슈는 **인바운드 응답 복원**에만. 응답은 외부에서 *들어오는* 것이라 유출 아님, 일은 토큰 복원.
  - **경계 룩어헤드 버퍼링**: 플레이스홀더 토큰 길이 상한을 이용해 **청크 경계에 걸친 토큰만 재조립할 작은 슬라이딩 윈도우**만 잡고 확정 prefix는 즉시 흘려보냄(TTFT 소폭 지연, 스트리밍 유지). 꼬리 부분 토큰만 보류. (전체 버퍼링 아님.)

---

## 13. Ledger / 감사 — R4·R11 (확정)

> 세 번째 금고 역설을 설계로 봉인: **Ledger는 절대 원본을 영속화하지 않는다.**

### 13.1 컴포넌트 분리
- **Ledger** = append-only 감사 스토어, **메타데이터 전용** → ouroboros SQLite EventStore 탑재. R6 갭·R9 열화·R10 무음 mask·R11 커버리지 알람 **전부 여기에 원본 없이** 기록.
  - 필드: `timestamp, channel, provider, category, action, rule_id, confidence, severity, span 길이, charclass 시그니처, fail/gap/degrade 사유, keyed-hash`.
- **진단(옵트인)** = **마스킹된 문맥 스니펫**(리터럴 없음), 별도 파일·짧은 TTL — 튜닝용.
- **raw-capture** = Ledger에서 **분리·기본 제거**. 정말 필요 시에만 `proxy_held`급 규율의 격리 quarantine(암호화·자동만료·경고), Ledger 무관.

### 13.2 "해시만 남기면 안전"의 함정 — 키드 해시
- 저엔트로피 PII(전화 11자리·주민번호·생년월일)의 **단순 salted-hash는 전수 enumeration으로 즉시 역산**됨.
- 해소: 상관분석 해시는 **설치 로컬 비밀키 HMAC**(키 없이 brute-force 불가, 키는 600 분리·컨트롤플레인). 초저엔트로피 카테고리는 per-value 해시 생략, 카테고리+개수만.

### 13.3 수명·접근통제
- 로컬 단일 사용자가 유일 독자, **네트워크 export 없음**. 파일 600/디렉토리 700. **보존기간 기본 30일 + 로테이션 + 명시적 `purge`**. 진단/quarantine는 더 짧은 TTL·암호화.

---

## 14. 실패 모드 · 열화 · 가용성 — R2·R9·R13

### 14.1 실패 유형 구분 (R6의 fail-closed를 정제)
| 실패 유형 | 의미 | 동작 |
| :-- | :-- | :-- |
| **콘텐츠 실패** | 검사를 *돌렸으나* 콘텐츠 못 미더움(파싱 불가 등) | **그 요청/블록 차단**(fail-closed) |
| **인프라 실패 — Stage2** | NER OOM/타임아웃/로드 실패 | **Stage1-only로 열화** (기본 `on_infra_failure: degrade`) |
| **인프라 실패 — 프록시 전체** | 코어 크래시·행·OOM | **fail-closed**(직통 없음) + 탄력성 |

### 14.2 Stage2 열화 (degrade)
- 여전히 보호: 시크릿·주민번호 등 고정 패턴·고심각도는 Stage1이 잡음(**보안 바닥 유지**).
- 미보호 통과: `PERSON/ADDRESS`는 통과시키되 **Ledger 커버리지 갭 + 가시 경고**("Stage2 비활성 — 이름·주소 미보호"). 침묵 금지(P5).
- 고컴플라이언스는 카테고리별 `stage2_fail_action: block`으로 강제 가능(브릭 위험 인지).

### 14.3 가드(프록시) 실패 = fail-closed + 탄력성
- 보안 도구의 정직한 기본 = **fail-closed**(프록시 다운 시 LLM 호출 실패, 무방비 직통 없음). fail-open은 *거짓 안심*이라 기본 배제.
- 가용성 절벽은 fail-open이 아니라 **탄력성으로 완화**: 코어/워커 분할(§3.1) + **워치독 자동 재기동** + 메모리 예산.
- **egress 락다운 상호작용**: 락다운 ON에서 프록시 다운 = 하드 가용성 절벽(실제 강제의 비용, 명시). 완화 = 워치독 + **사용자(에이전트 아님)만 가능한 break-glass 수동 해제**(root 액션, 에이전트 접근 불가).
- fail-open은 가용성 우선 사용자용 **명시적 옵트인**(도구가 best-effort로 강등됨을 선언).

### 14.4 개입 UX
- **기본 = 자동 처리 + Ledger 기록 + 가시 알림**(논블로킹). 에이전트 루프 중 **동기 블로킹 프롬프트는 데드락**이라 배제.
- **인터랙티브 승인은 옵트인 + 고심각 block 한정 + 비동기**: ouroboros 결정 표면(ask_user) 사용, **타임아웃→안전 기본값(block)** 폴백으로 영구 멈춤 방지.

---

## 15. 룰 · 모델 라이프사이클 · 검증 — R11

- 갱신: **(기본) 번들 + 명시적 업데이트** + 사용자 커스텀 룰/pin. **자동 갱신은 옵트인·서명 검증 필수** — 탐지 룰의 사일런트 자동 갱신은 공급망 위험(탐지 무력화·우회).
- **검증 자산을 요구사항으로 명문화**:
  - **합성 PII 레드팀 코퍼스**(실데이터 금지, 유효 체크섬 가진 가짜 한국 포맷 픽스처) + 카테고리별 **정밀도/재현율** 측정.
  - **골든 회귀 스위트 + CI FP 게이트** → "실제로 잡나" / "업데이트가 기존 탐지를 깨지 않나" 증명. acceptance criteria에 포함.

---

## 16. 오탐 비용 · 튜닝 루프 — R10

- 비대칭: 코딩 에이전트에서 **과잉 마스킹은 코드를 깨 작업 실패**(함수명·픽스처·식별자 오인). 과소는 유출.
- 목표: NER 의존 카테고리 **정밀도 우선**(예 ≥0.9). 고심각 정규식은 체크섬으로 FP 억제.
- **튜닝 메커니즘 = 정책 1급 기능**:
  - **프로젝트별 allowlist/pin-list**("이건 PII 아님" 등록 — 값 해시·정규식·경로). **단 변경은 컨트롤플레인 = 사용자 승인**(§2.2, 에이전트 자기 화이트리스트 금지).
  - 코드 인지 화이트리스트 프로파일(식별자·키워드·픽스처) 기본 제공.
  - **R11 진단 모드**(마스킹된 문맥 + 발동 규칙)가 튜닝 루프의 입력.

---

## 17. 보안 고려 (게이트웨이 자신)

- 프록시 `127.0.0.1` 바인딩, 외부 인터페이스 노출 금지.
- **로컬 루트 CA 미설치**(§4) — 고가치 공격 표면 회피.
- 컨트롤플레인 파일(정책·HMAC키·Ledger·락다운규칙) 권한 600/700, **에이전트 쓰기 권한 밖**(§2.2).
- `passthrough` 기본으로 키 비영속(§10).
- 자체 테스트에 실제 PII 금지 — 합성 데이터(§15).

---

## 18. 비기능 요구사항

- 완전 로컬 동작(탐지·정책·복원맵·Ledger). 외부 의존은 실제 LLM API 호출뿐.
- 멀티프로바이더(anthropic/openai/gemini) 요청·응답 정확 파싱.
- 정책 핫리로드.
- §6.3 지연·메모리 예산 충족. §15 검증 스위트 통과.

---

## 19. 구현 단계 / MVP 범위

### MVP (1차)
- 게이트웨이 코어(멀티프로바이더 패스스루) + 티어1 자동 주입
- **Stage1(정규식·체크섬·사전) + Stage2(Presidio+spaCy-ko, 별도 프로세스)**
- 단일 스키마 정책 엔진(kr-strict 기본) + 핫리로드
- 입력 표면 전체 스캔(tool_result 포함) + 구조 파서 + 블록 해시 캐시
- 마스킹(인덱스 토큰) + **에이전트 왕복 인바운드 복원** + 미복원 토큰 탐지
- 문서 추출(txt/md/csv/json/code/docx/xlsx/pptx/pdf-텍스트)
- Ledger(메타 전용·HMAC) + 컨트롤/데이터 플레인 권한 분리
- 실패 모드(콘텐츠=block / Stage2=degrade / 프록시=fail-closed+워치독)
- 합성 레드팀 코퍼스 + 골든 회귀 CI

### 2차
- 티어2 egress 락다운(pf, root) + break-glass
- hwp/hwpx·OCR(이미지 PII)
- `proxy_held` 자격증명 모드, `tokenize_roundtrip` 시크릿 편집
- GLiNER 백엔드(R18, ✅ 완료) → **NER 정확도 추가 향상(R20)**: 신뢰도 임계값 노브 정책 노출 + GLiNER 파인튜닝(ORG 정밀도·잔여 recall) → 추가로 transformer 한국어 NER(KoELECTRA/KLUE) 옵션, 서명된 자동 룰 갱신 채널
- 브라우저/스크립트 채널 확장, (잠재 티어3) VM/샌드박스 봉쇄

---

## 20. ouroboros Seed (MVP)

```yaml
goal: >
  로컬 LLM 게이트웨이 프록시 'PII-Guard'를 구현한다. ouroboros 워크플로와 LLM CLI가
  외부 LLM(anthropic/openai/gemini)으로 보내는 요청을 base_url 자동 주입으로 프록시
  경유시키고, 페이로드의 모든 의미 필드(특히 tool_result)를 구조 파서로 순회해
  Stage1(정규식+체크섬+사전)과 Stage2(Presidio+한국어 spaCy, 별도 프로세스)로 로컬
  탐지한다. 카테고리별 정책(kr-strict 기본, block/mask/allow)을 적용하고, 마스킹은
  인덱스 토큰으로 치환하되 에이전트 왕복 콘텐츠는 인바운드 응답에서 복원해 데이터
  파괴를 막는다. 모든 판정을 ouroboros Ledger에 원본 없이(keyed-hash) 기록하며,
  정책·키·Ledger는 피보호 에이전트의 쓰기 권한 밖(컨트롤플레인)에 둔다.
task_type: code
constraints:
  - 탐지·정책·복원맵·Ledger는 완전 로컬, 외부 전송 없음 (탐지에 외부 LLM 금지)
  - secure-by-default: 정책 파일 없이도 시크릿·주민번호 block, 이메일·이름 mask
  - 프록시 127.0.0.1 바인딩, 로컬 루트 CA 미설치(MITM 배제), passthrough 기본
  - 컨트롤플레인(정책·HMAC키·Ledger·락다운규칙)은 에이전트 쓰기 권한 밖
  - 콘텐츠 실패=block / Stage2 인프라 실패=Stage1 degrade / 프록시 실패=fail-closed+워치독
  - 침묵 통과 금지: 미보호 통과·열화·커버리지 갭은 Ledger에 가시 기록
  - Ledger는 원본 미영속, 저엔트로피 PII는 keyed-hash 또는 해시 생략
  - Stage2는 블록 타임아웃·메모리 예산·콘텐츠 클래스 게이팅·해시 캐시 적용
  - 멀티프로바이더 요청·응답 파싱, 미지 필드는 커버리지 알람
  - MVP 파일 포맷: txt/md/csv/json/code/docx/xlsx/pptx/pdf-텍스트 (hwp 제외)
acceptance_criteria:
  - anthropic/openai/gemini 요청을 프록시가 정확히 파싱·중계한다
  - tool_result 안의 시크릿/주민번호가 block 정책 시 외부로 나가지 않는다
  - 정규식이 못 잡는 문맥상 이름/주소를 Stage2가 탐지해 정책을 적용한다
  - 이메일/전화가 mask 시 인덱스 토큰으로 치환되어 전송된다
  - 에이전트 왕복 콘텐츠의 토큰이 인바운드 응답에서 원본으로 복원된다
  - 복원 못 한 block-카테고리 잔여 토큰이 있으면 응답을 보류/경고한다
  - 모든 판정이 Ledger에 keyed-hash·카테고리·액션으로만 기록되고 원본은 없다
  - 저엔트로피 PII가 단순 해시로 역산 가능하게 저장되지 않는다
  - Stage2 OOM/타임아웃 시 Stage1으로 열화하고 갭을 Ledger에 기록한다
  - 프록시 다운 시 fail-closed로 동작하고 워치독이 재기동한다
  - 정책 파일을 지워도 secure default로 보호가 유지된다
  - 에이전트가 pin-list/정책 파일을 쓰기로 수정할 수 없다 (권한 경계)
  - 인증 헤더는 마스킹 제외하되 키 shape 검증으로 밀반출을 차단한다
  - 합성 레드팀 코퍼스에서 카테고리별 정밀도/재현율 목표를 만족하고 회귀가 없다
```

---

## 21. 결정 로그 (인터뷰 R0–R13)

| R | 주제 | 결정 |
| :-- | :-- | :-- |
| R0 | 기반 | 프록시 게이트웨이 / 카테고리별 정책 / 하이브리드 탐지 / ouroboros+CLI 채널 |
| — | 스택 | Stage2 NER 선택형 백엔드(GLiNER 기본 / Presidio+spaCy-ko 폴백, R18), Stage2 MVP 포함, hwp 2차, M2 8GB, 별도 3.11 venv |
| R1 | 가로채기 | base_url 재설정(MITM 배제) |
| R2 | 실패 모드 | fail-closed 기본, 단계별, `stage2_fail_action` |
| R3 | 마스킹 계약 | 인덱스 토큰 + 단방향 세션 일관성, mask≠block (복원은 R10에서 정제) |
| R4 | Ledger | 메타 기본 + 진단 옵트인 + raw 격리 (R11 확정) |
| R5 | 정책 모델 | secure-by-default(kr-strict), 단일 스키마, 핫리로드, 한국 1급 |
| R6 | 입력 표면 | 모든 의미 필드 스캔, **tool_result 최우선**, 블록 해시 캐시, unscannable 정책 |
| R7 | 강제(1) | 티어1 자동 주입 default-on + 위협모델 표 + egress 락다운 옵트인 |
| R8 | 자격증명/우회 | passthrough 기본(금고 회피), false-assurance 정직 선언 |
| R9 | 자원 열화 | 콘텐츠 vs 인프라 실패 구분, Stage1 degrade, 예산 1급 요구사항화 |
| R10 | 왕복 무결성 | **복원 ON(에이전트 왕복)** — R3 정제, 미복원 토큰 탐지, `tokenize_roundtrip`, 프로젝트 pin-list, 비동기 승인 |
| R11 | Ledger/스트리밍/룰 | 메타 전용·HMAC, 경계 버퍼링, 룰 서명·골든 회귀 |
| R12 | 강제(2) | 정직한 2-티어, CA 미설치, 구조 파서+트립와이어+staleness, 인증 헤더 shape 검증 |
| R13 | 위협 행위자 | **trusted-but-compromisable**, control/data plane 분리, 가드 실패 fail-closed+탄력성+break-glass, root=범위 밖 |

---

## 22. 남은 미정 (구현 단계 확정)

- 한국어 NER 모델 구체 선정(`ko_core_news_sm` vs `lg`) — 정밀도/메모리 실측 후. → **§23.1에서 확정**
- ouroboros TUI/CLI에서 차단·열화 알림 표시 방식. → **§23.1 (CLI `--log-masked` + Streamlit UI)**
- 컨트롤플레인 권한 분리 구체 구현(별도 UID vs root-owned vs 파일 ACL) — macOS 환경 제약 확인. → **§23.1**
- break-glass·egress 락다운(티어2)의 macOS pf 통합 상세. → **2차 범위 유지 (§23.3 미실행 명시)**

---

## 23. 구현 반영 보완 (v3 — 2026-06-23)

> v2(요구사항) → 4라운드 구현(ouroboros 워크플로 1~3차 + 수동 보강) 후 **as-built 사실**과
> 구현 중 발견된 신규 요구사항·한계·결정을 반영한다. 상세 아키텍처는 **[`DESIGN.md`](./DESIGN.md)** 참조.

### 23.1 §22 미정 항목 확정

| 미정 항목 | 확정 내용 |
| :-- | :-- |
| **NER 백엔드/모델** | **선택형 백엔드 채택·배선 완료(R18)** — 기본 **`gliner`**(모델 `urchade/gliner_multi_pii-v1`, **Apache-2.0·상업 가능**), 경량 폴백 **`spacy`**(`ko_core_news_lg`, sm 재폴백). 백엔드 선택 = `PIIGUARD_NER_BACKEND` env 또는 정책 `stage2.ner_backend`(env 우선). 모델 변형 = GLiNER `PIIGUARD_GLINER_MODEL` / spaCy `PIIGUARD_KO_SPACY_MODEL`. **코퍼스 실측**: spaCy recall PERSON 0.84/ADDR 1.00/ORG 0.92 · GLiNER recall PERSON 0.956/ADDR 0.88/ORG 0.96. **외부 6리포트**: GLiNER가 재현율 동등~우위 + 정밀도 우위(codex P 0.81→0.93, gemini 0.63→0.91). 비상업 시 `taeminlee/gliner_ko`(CC-BY-NC, 성능 동등) 선택 가능. 상세 = `validation/NER_BACKEND_COMPARISON.md`·DESIGN §6.4·ADR-10/11. |
| **알림 표시 방식** | ① CLI `piiguard serve --log-masked` — 업스트림 전송 직전 **마스킹된 페이로드 + 탐지 요약을 stdout 출력**(원문 미출력). ② **Streamlit UI**(`ui/app.py`) — 채팅 입력·다중 파일 업로드 시 마스킹/차단 결과를 화면+콘솔에 출력. |
| **컨트롤플레인 권한 분리** | 현 구현 = **파일 권한 600/700 + pin-list out-of-band 승인 게이트**(`pinlist_approval.py`) + egress pf 규칙 root 소유(`pf_manager.py`). 별도 UID 분리는 **배포 가이드 수준**으로 남김(단일 macOS 사용자 환경 가정). |

### 23.2 신규 요구사항 (구현 중 발견 — v2에 없던 항목)

| # | 요구사항 | 근거 |
| :-- | :-- | :-- |
| **R14** | **Stage2 NER은 프록시 데이터 경로에 default-on으로 연결되어야 한다.** `serve`는 `Engine(stage2_runner=Stage2NERRunner())`를 기본 구성하고 `--no-ner`로만 비활성. | E2E 스모크에서 발견: 단위테스트는 NER 엔진을 직접 호출해 통과하지만, 초기 `serve`는 기본 `Engine()`(Stage2 미연결)을 써 **실제 프록시 경로에서 한국어 이름이 평문 유출**됐다. "every outbound request"(AC1) 충족을 위해 wiring을 요구사항으로 승격. (`tests/test_serve_ner_wiring.py`, `scripts/e2e_smoke.py`) |
| **R15** | **마스킹 관찰가능성**: 운영자가 실제 업스트림(Anthropic 등) 호출 시 PII가 마스킹/차단된 채 나가는지 콘솔로 확인할 수 있어야 한다. 단 **원본은 절대 로그에 쓰지 않는다**(Ledger no-raw 원칙 일치). | `--log-masked`. 마스킹된 페이로드 + 카테고리→플레이스홀더 요약만 출력. |
| **R16** | **대화형 검증 UI**: 비개발 사용자도 임의 텍스트·파일의 PII 탐지 결과를 즉시 확인할 수 있는 로컬 UI 제공. | Streamlit. 순수 로직(`ui/scanner.py`)은 UI 비의존·단위테스트됨. |
| **R17** | **Proximity(근접 문맥) 기반 탐지 보완** (✅ Phase 1·2·3 구현 완료 — `proximity.py`·`stage2/ner_filters.py`·정책 `proximity:`): ① **양성 proximity** — 모호한 정형 PII(비표준 계좌·하이픈 없는 사업자번호·한글 라벨 비밀번호)를 **트리거 키워드가 근접할 때만 승격**해 재현율↑. ② **음성 proximity / 콘텐츠 클래스 게이팅**(§6.3 부채) — **코드·비자연어 구간에서 Stage2 NER 억제**해 과잉 마스킹 정밀도↑. 모두 **규칙 기반**(결정적·감사가능·인젝션 불가)이며 신뢰도 게이팅·정책 노출. | 30케이스 실효성 검증(§23.6)에서 발견된 재현율 갭(계좌 0.78·비번 0.80·사업자 0.86)과 코드 텍스트 과잉 마스킹(정밀도 0.79)을 직접 타격. 상세 설계 = [`docs/design/PROXIMITY_DESIGN.md`](./design/PROXIMITY_DESIGN.md). |
| **R18** | **선택형 Stage2 NER 백엔드** (✅ 구현·실측 완료 — `stage2/backend.py`·`stage2/gliner_ner.py`·`_workers.py`·`policy.py`·`Engine`·`serve`·워밍업; 단위테스트 `tests/test_ner_backend.py`; GLiNER 실측 = `validation/NER_BACKEND_COMPARISON.md`): Stage2 NER 엔진을 **GLiNER(기본 `urchade/gliner_multi_pii-v1`, Apache-2.0)와 spaCy(경량 폴백) 중 선택**할 수 있어야 한다. 선택은 **환경변수 `PIIGUARD_NER_BACKEND`**와 **정책 YAML `stage2.ner_backend`**(env 우선)로 노출하고, 모델 변형은 `PIIGUARD_GLINER_MODEL`/`PIIGUARD_KO_SPACY_MODEL`로 오버라이드. 두 백엔드는 **동일 카테고리(PERSON/ADDRESS/ORGANIZATION)로 정규화**되어 정책·마스킹·proximity 후필터·열화 경로가 백엔드와 무관하게 동작해야 한다. GLiNER는 트랜스포머라 **별도 워커·옵션 설치(`[ner-gliner]`)로 격리**(§3.1·§6.2). | spaCy(lg) PERSON recall 0.84(작은 모델 한계, §23.3)를 더 높은 재현율의 GLiNER로 끌어올리는 업그레이드 경로를, 메모리 제약 환경을 위한 경량 폴백을 유지한 채 **선택지로 제도화**. DR-1(§23.4)이 명시한 "로컬 인코더 NER 모델 교체" 경로의 1차 실현. 상세 = DESIGN ADR-11. |
| **R19** | **Stage1 정형 PII recall 보강** (✅ 구현·실측 완료 — `categories.py`·`proximity.py`): 정규식/proximity로 잡는 정형 PII의 비표준 변형을 결정적으로 보강한다. ① KR_ACCOUNT 비표준 계좌 포맷 일반화(하이픈 2~3개+자릿수 9~14, 트리거 게이팅 유지) ② PASSPORT 조사 인접 경계 버그 수정(`(?!\w)`→`(?![A-Za-z0-9])`) ③ JWT 2번째 세그먼트 `eyJ` 강제 제거 ④ GitHub `ghp_` 길이 `{36,}`→`{20,}` ⑤ PASSWORD 접두 라벨(`DB_PASS=`·`temporary_pass:`) 인식. **정밀도 회귀 없이** 재현율 상승(외부 codex 0.798→0.921·gemini 0.875→0.958, 정형 PII 미검출 27→10). | FN 분류 결과 GLiNER 미검출의 ~79%가 NER이 아니라 **정형 PII(Stage1 영역)** — NER 백엔드 무관. 상세 = `validation/STAGE1_RECALL_IMPROVEMENT_2026-06-25.md`·`NER_BACKEND_COMPARISON.md §5`. |
| **R20** | **NER 정확도 추가 향상 — 임계값 노브 + GLiNER 파인튜닝** (🔬 조사 완료, 구현 검토 대상): R19로 정형 PII를 회수한 뒤 남는 NER 갭(PERSON/ADDRESS recall, **ORG 정밀도**)을 두 레버로 개선. ① **신뢰도 임계값 노브**(`min_confidence`, 정책 노출) — GLiNER 0.50→0.35로 ADDRESS recall 0.88→1.00·PERSON 0.956→0.978, **FP 거의 불변**(무료에 가까운 Pareto 개선). ② **GLiNER 파인튜닝** — 임계값에 무반응인 **ORG 과추출(정밀도 0.774·FP 7)**을 hard-negative 학습으로 교정 → 임계값과 달리 **recall·정밀도 동시 개선(곡선 자체 상승)**. Apache 베이스라 파인튜닝 산출물도 상업 사용 가능. | 임계값 스윕 실측(코퍼스): 0.5→0.3에서 PERSON/ADDRESS recall↑·FP 불변, **ORG는 임계값 무반응**(P 0.774 고정) → ORG는 파인튜닝 영역. 합성-only 제약상 ORG 경계 교정(구문적)은 효과적, 광역 한국어 recall은 한계. 데이터: positive(재현율)+hard negative(ORG 정밀도) 수백~2,000문장. 권고 순서 = 임계값(즉시·무료) → 잔차 측정 → 파인튜닝. 상세 = `NER_BACKEND_COMPARISON.md §5`. |

### 23.3 알려진 한계 (정직 선언 — P3 거짓 안심 금지)

> ※ 아래 ★ 표시 한계는 30케이스 실효성 검증(§23.6)에서 정량 확인됐고, **R17 proximity 설계**(§23.6·DR-2)로 보완 예정.

| 한계 | 내용 | 영향 / 완화 |
| :-- | :-- | :-- |
| **KR_ACCOUNT 비표준 포맷** (✅ R17+R19로 해소) | 과거 한계: 인식된 은행 구획/라벨이 있을 때만 동작. | ✅ **R17 양성 proximity + R19 일반화**(하이픈 2~3개+자릿수 9~14, 트리거 게이팅)로 비표준 포맷 회수 — 검증 recall 0.78→1.00(codex/gemini KR_ACCOUNT 9/9). |
| **PASSWORD 라벨 변형** (✅ R17+R19로 대부분 해소) | 과거 한계: 영문 `password=`만 인식. | ✅ **R17(한글 비밀번호/비번/암호) + R19(`DB_PASS=`·`temporary_pass:` 접두 라벨)**. 잔여 = 라벨 없는 본문 비번 값(결정적 검출 곤란, 보류). |
| ★ **BIZ_NO 하이픈 없는 10자리 미검출** | `1806341205`처럼 구분자 없는 사업자번호 미탐지(검증 recall 0.86). | **완화 = R17 양성 proximity**("사업자등록번호" 근접 시 맨 10자리 승격). |
| ★ **코드/기술 텍스트 NER 과잉 마스킹** | 코드 리뷰·로그에서 식별자·일반명사(`주석`·`리턴값`·`LGTM`)를 인물/조직으로 NER 오분류(검증 정밀도 0.79, 진짜 오탐 36건). | **완화 = R17 음성 proximity / 콘텐츠 게이팅**(코드 구간 NER 억제, §6.3 부채 해소). |
| **PERSON recall < 1.0** | spaCy lg 0.84 / GLiNER(Apache) 0.956. honorific 없는 맨이름 일부 누락. | 고심각 아님(이름=mask). ✅ R18 GLiNER 기본 채택으로 완화. 추가 = **R20 임계값↓(0.5→0.35 시 0.978)** / 파인튜닝. |
| **ORGANIZATION 정밀도 < 1.0** | GLiNER(Apache) ORG 정밀도 0.774(FP 7, 조직 과추출·경계 오류). **임계값에 무반응** — 모델 특성. | 마스킹 방향이라 보안 영향 작음(과잉 마스킹). 완화 = **R20 GLiNER 파인튜닝**(hard-negative로 ORG 경계 교정, recall 유지·정밀도↑). |
| **egress 락다운(티어2) 실검증 미수행** | pf(4)·root·실네트워크 필요한 통합테스트 12개가 CI에서 **항상 skip**. 실제 macOS pf 차단을 한 번도 실행 검증 안 함. | 티어2는 2차 범위. 배포 전 `sudo pytest -m integration` 실검증 필요. |
| **hwp/hwpx·OCR 미구현** | 2차 범위. 현재 unscannable → fail-closed(block). | §11·§19대로 2차. |
| **단일 프로세스 권한 모델** | 컨트롤/데이터 플레인 분리는 파일 권한·승인 게이트로 구현. 별도 UID·root 소유는 미적용(단일 사용자 가정). | root 권한 에이전트는 §2.3대로 범위 밖. |

### 23.4 결정 기록 DR-1 — "규칙 기반 → LLM 기반 탐지" 검토 결과 (채택: 하이브리드 유지)

| 옵션 | 판정 | 사유 |
| :-- | :-- | :-- |
| **외부 LLM(Claude/GPT API) 탐지** | ❌ **불가** | **P1 명시 위반**("탐지에 외부 LLM 금지") + **자기모순**(막으려는 PII를 탐지하려 외부 LLM에 전송) + P4 세 금고. |
| **로컬 생성형 LLM(Ollama/llama.cpp) 탐지** | ⚠️ **비권장** | ① 메모리: 쓸만한 7B 양자화 ~4~5GB > 예산 1~1.5GB(M2 8GB 초과). ② 비결정성 → "거짓 안심"(P3) 위험. ③ 체크섬(주민번호·카드) 검증 불가. ④ 감사(Ledger) 재현성 약화. ⑤ **🔴 프롬프트 인젝션**: 위협모델 R13(탈취 가능 에이전트·비신뢰 콘텐츠)에서 *"이건 PII 아님"* 주입으로 **탐지기 자체가 무력화** — 공격면 확대. |
| **규칙 + 로컬 인코더 NER 하이브리드(현행)** | ✅ **채택·유지** | 규칙=구조적·체크섬 PII에 **결정적·검증가능·인젝션 불가**(생성형보다 우월). 인코더 NER(spaCy↔**GLiNER** 선택, →KoELECTRA/KLUE 추가 교체 가능)=문맥 PII. 생성 안 하므로 **인젝션 불가**. §6.2 선택형 백엔드(R18)·추상화 슬롯에 모델 교체로 재현율 향상. |

> **결론**: 보안 탐지기에 생성형 LLM은 퇴보. "더 똑똑한 탐지"의 정답은 **로컬 인코더 NER 모델 교체**(이미 설계된 업그레이드 경로 — R18 선택형 백엔드로 1차 실현: GLiNER 기본 / spaCy 폴백)이며, 규칙은 고심각·체크섬 카테고리에서 계속 1급으로 유지한다.

### 23.4b 결정 기록 DR-2 — 검증 갭 보완: "단순 규칙 확장 vs proximity vs LLM" (채택: proximity)

| 옵션 | 판정 | 사유 |
| :-- | :-- | :-- |
| **단순 정규식 확장**(모호 포맷도 무조건 탐지) | ❌ **기각** | 3-3-6 계좌 등은 송장·주문번호와 구분 불가 → **오탐 폭발**(정밀도 급락). 애초에 막아둔 이유. |
| **LLM/생성형으로 문맥 판단** | ❌ **기각** | DR-1과 동일(인젝션·메모리·비결정성). |
| **Proximity(근접 문맥) 규칙** | ✅ **채택** | 모호 값을 **트리거 키워드 근접 시에만** 승격/억제 → 오탐 억제하며 recall·precision 양방향 개선. **결정적·감사가능·인젝션 불가**(Stage1 강점 유지). 신뢰도 게이팅·정책 노출로 튜닝. |

> **결론**: 검증 갭의 정답은 "더 공격적인 탐지"가 아니라 **"문맥이 확인될 때만 더 탐지"**. 양성(승격)·음성(억제)
> 두 방향으로 적용하되, **회귀 스위트 + 30케이스 검증을 머지 게이트**로 두어 오탐 재유입을 막는다. 설계 = [`docs/design/PROXIMITY_DESIGN.md`](./design/PROXIMITY_DESIGN.md).

### 23.5 as-built 요약

- **20개 카테고리**(시크릿 6·고위험신원 5·연락식별 4·문맥NER 3·서버정보 2) — §7 카탈로그 구현 완료.
- **43개 모듈** / **42개 테스트 파일** / **2685 passed, 12 skipped, 0 failed**(`.venv` + lg 기준).
- 구현 모듈 구조는 §3.2의 계획(`core/ detectors/ …` 중첩)과 달리 **평면 `pii_guard/` 패키지**로 수렴 — DESIGN.md §4 모듈맵 참조.
- MVP(§19 1차) **전 항목 구현**. 2차(egress 락다운·hwp/OCR·proxy_held·transformer NER·서명 자동갱신)는 미착수.

### 23.6 실효성 검증 결과 + proximity 보완 계획 (R17)

**검증(완료)**: 합성 30케이스(한국어 ~1000자 + 영문 시크릿, 실전형)로 실제 엔진을 돌려 재현율·정밀도 측정.
산출물 = [`validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md`](../../validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md)(리포트) + `efficacy_test_log.txt`(증거 로그) + `efficacy_test.py`(재현 하니스).

| 지표 | 초기(보정) | **R17 적용 후(보정)** | 비고 |
| :-- | :-- | :-- | :-- |
| 재현율 | 0.92 | **0.95** | 정형 PII(키·카드·여권·이메일·전화)는 사실상 1.00 |
| 정밀도 | 0.85 | **0.94** | 진짜 over-masking 36→13건 |

**갭 → R17 매핑 (✅ 구현 완료)**:
- KR_ACCOUNT(비표준 3-3-6·토스/카카오 4-2-7)·BIZ_NO(맨숫자)·PASSWORD(한글라벨) → **양성 proximity**(`proximity.py`, 트리거 근접 시 승격).
- 코드 텍스트 NER 과잉 마스킹 → **음성 proximity**(`stage2/ner_filters.py`, NER FP 후필터: 코드토큰·약어·blob·일반명사 deny-list).

**구현 현황** — 상세 [`docs/design/PROXIMITY_DESIGN.md`](./design/PROXIMITY_DESIGN.md):
- ✅ **Phase 1(음성, NER FP 억제)** — `stage2/ner_filters.py`. 정밀도 0.85→0.93.
- ✅ **Phase 2(양성 proximity)** — `proximity.py`(Stage-1.5, `STAGE1_PROXIMITY`). 재현율 0.92→0.95. merge containment로 계좌가 전화 오탐 흡수.
- ✅ **Phase 3(정책 노출)** — 정책 YAML `proximity:` 블록(트리거 키워드·window·enable·NER필터 노브)을 `PolicyConfig.proximity`로 파싱·핫리로드. `serve --policy PATH`로 로드해 Engine에 연결. NER 필터 노브는 env로 Stage-2 서브프로세스에 전파.
- **수용 기준 충족**: 전체 **2685 테스트 무회귀**(0 failed) + 30케이스 재검증 **재현율 0.95 / 정밀도 0.94**(목표 0.94/0.87 초과).
