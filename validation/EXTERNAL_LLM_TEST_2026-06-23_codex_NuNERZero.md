# 외부 LLM(codex) 재생성 테스트 — PII-Guard 검출 · nunerzero 엔진 (2026-06-29)

> ⚠️ **재구성 리플레이**: 원본 입력 JSON이 없어, `EXTERNAL_LLM_TEST_2026-06-23_codex_GLiNER.md`에
> 임베드된 텍스트·정답(ground truth)을 파싱해 동일 케이스를 **nunerzero** 백엔드로 재채점한 결과다.
> 생성 하니스: `validation/external_replay.py` · 엔진: `Engine` + `Stage2NERRunner`(numind/NuNER_Zero).
> 정답·매칭은 GLiNER 리포트 기준으로 복원했으므로, 공정 비교를 위해 GLiNER도 동일 하니스로 재채점한다.
> 채점: 카테고리 호환(SECRET 클래스 내 호환·LOCATION≈ADDRESS) + 값 포함 매칭 · **raw FP(트리아지 없음)**.

## 1. 핵심 결과

- 케이스 10개 · 정답 PII 총 89개
- **검출(TP) 80 · 미검출(FN) 9 · 오탐후보(FP) 5**
- **재현율(recall) = 0.899** · **정밀도(precision) = 0.941**

## 2. NER 카테고리별 재현율 (PERSON/ADDRESS/ORGANIZATION)

| 카테고리 | TP | FN | recall |
| :-- | --: | --: | --: |
| PERSON | 2 | 8 | 0.200 |
| ADDRESS | 11 | 0 | 1.000 |

## 3. 부록 — 케이스별 미검출/오탐후보

### [1] VOC 01 · 결제오류  (505자) · 🔴 block
- TP 8 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [2] VOC 02 · 회원정보 정정  (501자) · 🔴 block
- TP 7 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=박서준 · ⚠️ 오탐후보: `ORGANIZATION`=신한은행

### [3] VOC 03 · 배송지 변경  (488자) · 🔴 block
- TP 9 · FN 1 · FP 2
- ❌ 미검출: `PERSON`=이도윤 · ⚠️ 오탐후보: `ADDRESS`=A동, `ORGANIZATION`=우리은행

### [4] VOC 04 · 환불 지연  (512자) · 🔴 block
- TP 8 · FN 0 · FP 1
- ⚠️ 오탐후보: `ORGANIZATION`=하나은행

### [5] VOC 05 · 사업자 세금계산서  (468자) · 🔴 block
- TP 8 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=정유나

### [6] VOC 06 · 로그인 잠김  (457자) · 🔴 block
- TP 7 · FN 2 · FP 0
- ❌ 미검출: `PERSON`=한지우, `PASSWORD`=Hjw!0623Reset

### [7] VOC 07 · 예약 취소  (434자) · 🔴 block
- TP 7 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=강태오

### [8] VOC 08 · 카드 재등록  (515자) · 🔴 block
- TP 8 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=문채원

### [9] VOC 09 · 정기구독 해지  (455자) · 🔴 block
- TP 9 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=오세훈

### [10] VOC 10 · 반품 수거  (487자) · 🔴 block
- TP 9 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=배수빈 · ⚠️ 오탐후보: `ORGANIZATION`=부산은행

