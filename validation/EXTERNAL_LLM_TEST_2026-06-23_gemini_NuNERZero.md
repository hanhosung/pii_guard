# 외부 LLM(gemini) 재생성 테스트 — PII-Guard 검출 · nunerzero 엔진 (2026-06-29)

> ⚠️ **재구성 리플레이**: 원본 입력 JSON이 없어, `EXTERNAL_LLM_TEST_2026-06-23_gemini_GLiNER.md`에
> 임베드된 텍스트·정답(ground truth)을 파싱해 동일 케이스를 **nunerzero** 백엔드로 재채점한 결과다.
> 생성 하니스: `validation/external_replay.py` · 엔진: `Engine` + `Stage2NERRunner`(numind/NuNER_Zero).
> 정답·매칭은 GLiNER 리포트 기준으로 복원했으므로, 공정 비교를 위해 GLiNER도 동일 하니스로 재채점한다.
> 채점: 카테고리 호환(SECRET 클래스 내 호환·LOCATION≈ADDRESS) + 값 포함 매칭 · **raw FP(트리아지 없음)**.

## 1. 핵심 결과

- 케이스 10개 · 정답 PII 총 72개
- **검출(TP) 59 · 미검출(FN) 13 · 오탐후보(FP) 6**
- **재현율(recall) = 0.819** · **정밀도(precision) = 0.908**

## 2. NER 카테고리별 재현율 (PERSON/ADDRESS/ORGANIZATION)

| 카테고리 | TP | FN | recall |
| :-- | --: | --: | --: |
| PERSON | 2 | 8 | 0.200 |
| ADDRESS | 1 | 2 | 0.333 |

## 3. 부록 — 케이스별 미검출/오탐후보

### [1] VOC 01 · 결제 수단 등록 실패 및 계정 잠김 문의  (434자) · 🔴 block
- TP 5 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=박지민 · ⚠️ 오탐후보: `ORGANIZATION`=쇼핑몰

### [2] LOG 01 · 인증 서버 API Key 노출 및 세션 만료 로그  (1007자) · 🔴 block
- TP 6 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=최승우 · ⚠️ 오탐후보: `HOSTNAME`=api.internal

### [3] VOC 02 · 오프라인 매장 영수증 인증 및 포인트 적립 누락  (367자) · 🔴 block
- TP 5 · FN 1 · FP 0
- ❌ 미검출: `ADDRESS`=서초구 반포대로 234

### [4] LOG 02 · 결제 게이트웨이 웹훅 수신 및 계정 검증 로그  (871자) · 🔴 block
- TP 8 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=정민우

### [5] VOC 03 · 법인 회원 정보 변경 및 정산 증빙 서류 제출 안내 요청  (409자) · 🔴 block
- TP 5 · FN 1 · FP 1
- ❌ 미검출: `FOREIGN_REG`=120923-1591783 · ⚠️ 오탐후보: `RRN`=120923-1591783

### [6] LOG 03 · 데이터베이스 마이그레이션 중 자격 증명 유출 예외 로그  (667자) · 🔴 block
- TP 7 · FN 2 · FP 1
- ❌ 미검출: `PERSON`=강동우, `ADDRESS`=강남구 테헤란로 501 · ⚠️ 오탐후보: `ADDRESS`=5432 closed unexpect

### [7] VOC 04 · 글로벌 배송 주소 수정 및 여권번호 예외 처리 요청  (401자) · 🔴 block
- TP 5 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=김현우

### [8] LOG 04 · 클라우드 스토리지 동기화 에러 및 자격증명 노출  (836자) · 🔴 block
- TP 6 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=윤서준

### [9] VOC 05 · 가상자산 대행 거래 환불 및 신원 검증 요청  (445자) · 🔴 block
- TP 5 · FN 2 · FP 2
- ❌ 미검출: `PERSON`=최예은, `FOREIGN_REG`=700523-4376198 · ⚠️ 오탐후보: `ORGANIZATION`=거래소, `RRN`=700523-4376198

### [10] LOG 05 · 인프라 통합 모니터링 에이전트 자격 증명 수집 로그  (883자) · 🔴 block
- TP 7 · FN 2 · FP 0
- ❌ 미검출: `PERSON`=정다은, `PASSWORD`=Mypassword99!

