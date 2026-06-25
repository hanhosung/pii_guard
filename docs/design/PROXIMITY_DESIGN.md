# 설계 제안 — Proximity(근접 문맥) 기반 탐지 보완

> **구현 현황(2026-06-23)**: ✅ **Phase 1·2·3 구현·검증 완료** — `stage2/ner_filters.py`(음성),
> `proximity.py`(양성), 정책 YAML `proximity:` 노출(`policy.py`, serve `--policy`). 30케이스 검증
> 재현율 0.92→**0.95**, 정밀도 0.85→**0.94**, 전체 **2685 테스트 무회귀**.
>
> 상태: **구현 완료(Implemented)** · 작성: 2026-06-23 · 대상: `pii_guard` 탐지 파이프라인
> 짝 문서: [`../DESIGN.md`](../DESIGN.md)(as-built §6.5/6.6), [`../../validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md`](../../validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md)(검증 근거)
> 본 문서는 원래 제안서이며 Phase 1·2·3로 구현 완료됨(상단 배너). **단, 음성 메커니즘은 §3.2/§5.2의 *스팬 사전 게이팅* 대신 `stage2/ner_filters.py`의 *NER 후필터*로 실현**됐다(더 정밀·오프셋 재매핑 불필요). 본문 §3~§10은 제안 당시 기록으로 보존한다.

---

## 1. 배경 · 동기 (검증 근거)

30케이스 실효성 검증(`validation/EXTERNAL_LLM_TEST_2026-06-23_claude_spaCy.md`)에서 드러난 갭:

| 유형 | 갭 | 검증 수치 |
| :-- | :-- | :-- |
| **FN(놓침)** | KR_ACCOUNT 비표준 포맷(3-3-6, 토스/카카오 3-2-7) | recall 0.78 |
| **FN** | BIZ_NO 하이픈 없는 10자리 | recall 0.86 |
| **FN** | PASSWORD 한글 라벨("비밀번호:") | recall 0.80 |
| **FP(과잉)** | 코드/기술 텍스트의 식별자·일반명사를 인물/조직으로 NER 오분류 | 정밀도 0.79, 진짜 오탐 36건(케18 단독 7건) |

**근본 원인**: 모호한 값(3-3-6 숫자 등)을 *무조건* 잡으면 오탐이 폭발하므로 현재는 **아예 억제**한다.
그래서 "은연중 흘린" 비표준 PII가 빠져나간다. 반대로 NER은 *문맥을 안 가리고* 모든 자연어 스팬에 돌아
코드 토큰까지 인물로 오인한다.

**핵심 아이디어**: **"주변 문맥(proximity)"을 신호로 써서**, 모호한 값은 *문맥이 확인될 때만 승격*(recall↑),
NER은 *비자연어(코드) 구간에서 억제*(precision↑)한다. 규칙 기반이라 **결정적·감사가능·프롬프트 인젝션 불가**
(LLM 도입 거부한 ADR-8과 정합).

## 2. 목표 / 비목표

**목표**
- 모호한 정형 PII(계좌·사업자번호·비밀번호)를 **문맥 키워드 근접 시에만** 탐지 → recall↑, FP 최소.
- 코드/비자연어 구간에서 **Stage2 NER 억제** → 과잉 마스킹 precision↑.
- 결정성·재현성·감사성 유지(Stage1 특성 보존). 메모리/지연 영향 무시 가능.
- 정책으로 on/off·키워드·윈도우 튜닝 가능(secure-by-default).

**비목표**
- 트리거 단어가 전혀 없는 *완전 문맥-free* PII까지 잡는 것(잔여 갭, 정직 선언 P3).
- 생성형 LLM·무거운 모델 도입(ADR-8로 배제).
- 기존 정형 카테고리(키·카드·주민번호)의 동작 변경(이미 recall 1.0).

## 3. 두 가지 메커니즘 (반드시 분리)

### 3.1 양성 proximity — "문맥 있을 때만 승격" (recall↑)
모호한 값 패턴을 **약한 후보**로 잡되, **트리거 키워드가 윈도우 내에 있을 때만** 탐지로 확정.

### 3.2 음성 proximity / 콘텐츠 게이팅 — "코드 구간 NER 억제" (precision↑)
텍스트를 **자연어(NL) vs 코드/비자연어(CODE)** 스팬으로 분류하고, **NER은 NL 스팬에만** 적용.
※ **Stage1(정규식·키·시크릿)은 코드에도 그대로 적용**한다 — 키·토큰은 코드 안에 살기 때문(이것만 빼면 안 됨).
이것은 요구사항 §6.3("콘텐츠 클래스 게이팅")의 **미구현 부채를 해소**하는 것이기도 하다.

---

## 4. 데이터 모델

신규 `proximity.py` 모듈 제안:

```python
class ContextRule(NamedTuple):
    category: str                 # "KR_ACCOUNT", "BIZ_NO", "PASSWORD"
    value_pattern: re.Pattern     # 모호한 값(약한 후보)
    triggers: tuple[str, ...]     # 근접 트리거 키워드(또는 정규식)
    window_chars: int             # 값 기준 ±N 글자 탐색
    confidence: float             # 트리거 매칭 시 부여 신뢰도(≥ min_confidence)
    validator: Optional[Callable] = None   # 선택적 체크섬

class SpanClass(Enum):            # 콘텐츠 게이팅
    NATURAL_LANGUAGE = "nl"
    CODE = "code"                 # def/=/(){}; , camelCase, json, base64/hex
```

- `Detection`에 선택적 `context_trigger: str | None` 추가(감사용 — 어떤 키워드로 승격됐는지 Ledger 기록).
  또는 `rule_id`에 `ctx:<keyword>`로 인코딩(모델 변경 최소화).

## 5. 알고리즘

### 5.1 양성 proximity 스코어러
```
입력: text, ContextRule 목록
for rule in rules:
    for m in rule.value_pattern.finditer(text):
        window = text[max(0, m.start-W) : m.end+W]
        if any(trigger in window for trigger in rule.triggers):
            if rule.validator is None or rule.validator(m.group()):
                emit Detection(category, span=m, confidence=rule.confidence,
                               context_trigger=matched_keyword)
        # 트리거 없으면 미방출(억제) → 모호 값의 오탐 폭발 방지
```
- 기존 Stage1 탐지와 **병합** 시 동일 스팬 중복은 높은 신뢰도 우선.
- 카테고리별 트리거(초안):
  - **KR_ACCOUNT**: 은행명(국민·신한·우리·하나·농협·기업·카카오뱅크·토스뱅크·SC·씨티…) + 동사(입금·이체·송금·환불·예금주·계좌).
  - **BIZ_NO**: "사업자(등록)?\s*번호".
  - **PASSWORD**: 기존 영문 + 한글("비밀번호·비번·패스워드·암호").
  - **윈도우**: 초안 ±15~25자(코퍼스로 보정).

### 5.2 음성 proximity / 콘텐츠 게이팅
```
spans = split_into_spans(text)                 # 줄/블록 단위
for span in spans:
    span.cls = classify(span)                  # NL vs CODE (규칙 기반)
ner_input = concat(spans where cls == NL)      # 위치 오프셋 보존
run NER on ner_input only → 결과를 원위치로 재매핑
# Stage1(regex)은 전체 text에 그대로 적용
```
- `classify` 신호(CODE): `def `/`function`/`=>`/`;`/`{}`/`()` 빈도, `snake_case`·`camelCase` 식별자 밀도,
  JSON/`key: value` 구조, base64/hex blob, 들여쓰기 패턴.
- **보수적 폴백**: 애매하면 **NL로 간주**(NER 실행) → recall 쪽으로 안전(보안 도구 원칙).

## 6. 아키텍처 배치 (실제 코드 기준)

현재 흐름(`engine.py`):
```
scan(text):
  stage1 = detector.scan_text(text)            # line 103
  if stage2_runner:                            # line 114
      s2 = stage2_runner.scan(text, stage1)    # line 115  ← 전체 text에 NER
  merge → RedactionResult
```

제안 배치:
- **양성 proximity** = Stage1 직후 **보강 패스**. `detector.scan_text` 내부 또는 `engine.scan`에서
  `proximity.scan(text)` 호출 후 stage1 결과와 병합. (Stage1 영역의 정형 PII이므로 여기가 자연스러움.)
- **음성 게이팅** = `stage2_runner.scan(text, ...)` **호출 전**, text를 NL 스팬으로 필터링해 전달.
  `engine.scan`에서 `nl_text = content_gate(text)` 후 `stage2_runner.scan(nl_text, ...)`.
  Stage2 결과 스팬을 원본 오프셋으로 재매핑.

→ **프록시·정책·복원 등 다른 모듈은 무변경.** 탐지 코어에만 국소 추가.

## 7. 정책 / 설정 표면 (secure-by-default)

`policy.py` / `SECURE_DEFAULTS`에 추가(예시):
```yaml
proximity:
  positive:
    enabled: true
    window_chars: 20
    KR_ACCOUNT: { triggers: [국민, 신한, 우리, 하나, 농협, 기업, 카카오뱅크, 토스뱅크, 입금, 이체, 송금, 환불, 계좌, 예금주] }
    BIZ_NO:     { triggers: ["사업자등록번호", "사업자번호"] }
    PASSWORD:   { triggers: [비밀번호, 비번, 패스워드, 암호] }
  ner_content_gating:
    enabled: true       # 코드 구간 NER 억제
    conservative: true  # 애매하면 NL로(=NER 실행)
```
- 기본 on(secure-by-default), 운영자가 끄거나 키워드 추가 가능. 핫리로드 적용.

## 8. 리스크 · 완화

| 리스크 | 완화 |
| :-- | :-- |
| 양성 proximity가 비PII 숫자(송장·주문번호)를 승격 → FP 재유입 | 신뢰도 게이팅 + 길이/포맷 제약 + **골든 회귀 스위트 + 30케이스 검증을 acceptance 게이트로** |
| 콘텐츠 분류 오판으로 NER 스팬 누락 | **보수적 폴백(애매=NL)** → recall 쪽 안전 |
| 적대적: 트리거 단어 제거로 회피 | 잔여 갭(정직 선언). 오늘보다 나빠지지 않음 |
| 적대적: 트리거 단어 삽입으로 과잉마스킹 유발 | FP일 뿐 유출 아님 → 저위험 |
| 키워드/윈도우 튜닝 비용 | 코퍼스 기반 보정 + 정책 노출로 현장 조정 |

- **결정성·인젝션 불가·감사성 유지**: 모두 규칙/정규식 기반. 승격 근거(트리거)를 Ledger에 기록.

## 9. 테스트 · 수용 기준

- **단위**: 각 ContextRule(트리거 있을 때 검출 / 없을 때 억제), span classifier(NL vs CODE).
- **회귀**: 기존 골든·레드팀 스위트 **무회귀**(정밀도 보호) — 전체 2685 테스트 통과 유지.
- **수용(acceptance)**: `validation/efficacy_test.py` 재실행하여
  - 목표: **재현율 ~0.94+ / 정밀도 ~0.87+** (현재 0.91 / 0.85 보정), 진짜 오탐 36→~20 이하.
  - 케18(코드) 오탐 7→0 근접, KR_ACCOUNT FN 4건·BIZ/PASSWORD FN 2건 회수.

## 10. 구현 계획 (단계별) · 방안 검토

### Phase 1 — 음성 게이팅 (정밀도, 추천 선행)
- **이유**: 효과 가장 크고(§1 케18 등), 요구사항 §6.3 미구현 부채 해소, 양성보다 리스크 낮음(억제는 유출 위험 없음).
- 작업: `content_gate.py`(span 분류) + `engine.scan`에서 Stage2 입력 필터 + 오프셋 재매핑 + 단위테스트.

### Phase 2 — 양성 proximity (재현율)
- 작업: `proximity.py`(ContextRule + 스코어러) + KR_ACCOUNT/BIZ_NO/PASSWORD 룰 + 병합 로직 + 정책 노출.
- 신뢰도 게이팅·윈도우 보정 후 30케이스 검증으로 FP 무회귀 확인.

### Phase 3 — 정책·감사·문서
- 정책 스키마/핫리로드, Ledger context_trigger 기록, DESIGN.md as-built 반영, 검증 재실행 수치 갱신.

### 구현 방안 비교 (양성 proximity 배치)
| 방안 | 설명 | 장점 | 단점 |
| :-- | :-- | :-- | :-- |
| **A. categories.py 정규식 확장** | 윈도우를 정규식에 직접(`(은행)…(\d…)`) | 변경 최소, 기존 구조 | 정규식 복잡·취약, 윈도우 표현 난해 |
| **B. proximity.py 별도 스코어러**(추천) | 약한 후보 + 키워드 윈도우 분리 | 명확·튜닝·테스트 용이, 감사 기록 쉬움 | 신규 모듈·병합 로직 |
| **C. Presidio context enhancement** | Stage2에 내장 문맥 부스트 켜기 | 구현 적음 | NER 엔티티만 해당, 정형 PII 미해결 |

→ **권고: 양성=B(별도 스코어러), 음성=신규 content_gate, (선택) C는 NER 보조로 병행.**

## 11. 결정(권고)

- proximity 보완은 **도입 가치 명확**(검증 갭 직접 타격 + Stage1 강점 유지).
- **Phase 1(음성 게이팅) 선행 → Phase 2(양성) → Phase 3(정책/문서)** 순서 권고.
- 각 Phase 종료 시 **30케이스 검증 재실행으로 수치 입증**, 회귀 스위트 무회귀를 머지 게이트로.

---

*제안 문서 끝. 구현 착수 시 본 문서를 ADR로 승격하고 DESIGN.md와 동기화한다.*
