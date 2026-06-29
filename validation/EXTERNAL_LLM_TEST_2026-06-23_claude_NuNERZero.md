# 외부 LLM(claude) 재생성 테스트 — PII-Guard 검출 · nunerzero 엔진 (2026-06-29)

> ⚠️ **재구성 리플레이**: 원본 입력 JSON이 없어, `EXTERNAL_LLM_TEST_2026-06-23_claude_GLiNER.md`에
> 임베드된 텍스트·정답(ground truth)을 파싱해 동일 케이스를 **nunerzero** 백엔드로 재채점한 결과다.
> 생성 하니스: `validation/external_replay.py` · 엔진: `Engine` + `Stage2NERRunner`(numind/NuNER_Zero).
> 정답·매칭은 GLiNER 리포트 기준으로 복원했으므로, 공정 비교를 위해 GLiNER도 동일 하니스로 재채점한다.
> 채점: 카테고리 호환(SECRET 클래스 내 호환·LOCATION≈ADDRESS) + 값 포함 매칭 · **raw FP(트리아지 없음)**.

## 1. 핵심 결과

- 케이스 30개 · 정답 PII 총 204개
- **검출(TP) 176 · 미검출(FN) 28 · 오탐후보(FP) 23**
- **재현율(recall) = 0.863** · **정밀도(precision) = 0.884**

## 2. NER 카테고리별 재현율 (PERSON/ADDRESS/ORGANIZATION)

| 카테고리 | TP | FN | recall |
| :-- | --: | --: | --: |
| PERSON | 23 | 19 | 0.548 |
| ADDRESS | 25 | 1 | 0.962 |
| ORGANIZATION | 7 | 4 | 0.636 |

## 3. 부록 — 케이스별 미검출/오탐후보

### [01] 고객 상담 메모
- TP 6 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=김서연

### [02] 개발자 슬랙 메시지 (시크릿 유출)
- TP 4 · FN 1 · FP 3
- ❌ 미검출: `PERSON`=박지훈 · ⚠️ 오탐후보: `ORGANIZATION`=OpenAI, `ORGANIZATION`=인프라팀, `HOSTNAME`=wiki.internal

### [03] 병원 접수 안내 이메일
- TP 7 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [04] 채용 지원 서류 요약
- TP 7 · FN 1 · FP 0
- ❌ 미검출: `ORGANIZATION`=네이버

### [05] 택배 분실 문의 (비표준 계좌 — 갭)
- TP 7 · FN 0 · FP 2
- ⚠️ 오탐후보: `ORGANIZATION`=신한은행, `DRIVER_LICENSE`=123456789012

### [06] 부동산 계약 메모
- TP 8 · FN 1 · FP 2
- ❌ 미검출: `PERSON`=한상우 · ⚠️ 오탐후보: `ORGANIZATION`=우리은행, `ORGANIZATION`=대박공인중개사

### [07] 외국인 직원 인사 등록
- TP 6 · FN 1 · FP 2
- ❌ 미검출: `PERSON`=존스미스 · ⚠️ 오탐후보: `PERSON`=John Smith, `ORGANIZATION`=하나은행

### [08] 회의록 (tool_result 시뮬레이션)
- TP 4 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=신우진

### [09] 은행 콜센터 스크립트 (한글 비번 라벨 — 갭)
- TP 6 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=배수지 · ⚠️ 오탐후보: `PERSON`=고객

### [10] 개인 일기 (문맥형 PII)
- TP 3 · FN 3 · FP 1
- ❌ 미검출: `ORGANIZATION`=카카오, `PERSON`=김태형, `PERSON`=박건우 · ⚠️ 오탐후보: `ADDRESS`=강남역

### [11] 온라인 쇼핑 주문 확인
- TP 5 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [12] 법률 상담 요청서
- TP 8 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [13] DevOps 인시던트 리포트
- TP 4 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [14] 학원 등록 상담
- TP 8 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [15] 환자 차트 인계 노트
- TP 6 · FN 1 · FP 0
- ❌ 미검출: `RRN`=760312-2345671

### [16] 프리랜서 계약/정산
- TP 6 · FN 0 · FP 1
- ⚠️ 오탐후보: `ORGANIZATION`=카카오뱅크

### [17] 보험 가입 설계
- TP 7 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=황지훈

### [18] 코드 리뷰 코멘트 (코드+PII 혼합)
- TP 4 · FN 1 · FP 0
- ❌ 미검출: `PERSON`=류현진

### [19] 동호회 회원 명부
- TP 6 · FN 4 · FP 2
- ❌ 미검출: `KR_ACCOUNT`=110-123456-78901, `PERSON`=문가영, `PHONE`=010-5566-1122, `ADDRESS`=경기 고양시 일산동구 중앙로 1275 · ⚠️ 오탐후보: `ADDRESS`=신한 110-123456-78901, `ADDRESS`=양재

### [20] 수출 통관 서류
- TP 6 · FN 0 · FP 2
- ⚠️ 오탐후보: `PERSON`=Mike Brown, `ADDRESS`=부산항

### [21] 민원 접수 (관공서)
- TP 5 · FN 0 · FP 1
- ⚠️ 오탐후보: `ADDRESS`=아파트 정문 앞

### [22] 스타트업 투자 메모 (혼합 영문)
- TP 7 · FN 1 · FP 1
- ❌ 미검출: `PERSON`=박준혁 · ⚠️ 오탐후보: `PERSON`=Kevin Park

### [23] 졸업생 추천서
- TP 3 · FN 3 · FP 0
- ❌ 미검출: `PERSON`=윤하린, `ORGANIZATION`=LG전자, `PERSON`=한석규

### [24] 중고거래 채팅
- TP 5 · FN 0 · FP 2
- ⚠️ 오탐후보: `ADDRESS`=신도림역, `ORGANIZATION`=카카오뱅크

### [25] 급여 명세 발송
- TP 6 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [26] 여행 예약 확정
- TP 6 · FN 0 · FP 0
- ✅ 미검출/오탐 없음

### [27] 전세 대출 상담 (민감 집약)
- TP 7 · FN 3 · FP 1
- ❌ 미검출: `PERSON`=이상화, `PERSON`=박명수, `RRN`=760312-2345671 · ⚠️ 오탐후보: `ORGANIZATION`=박명수

### [28] AI 챗봇 대화 로그 (은연중 유출)
- TP 5 · FN 2 · FP 0
- ❌ 미검출: `PERSON`=한예슬, `ORGANIZATION`=쿠팡

### [29] 비표준 포맷 집중 (약점 노출)
- TP 6 · FN 0 · FP 1
- ⚠️ 오탐후보: `PHONE`=02-555-1234

### [30] 복합 업무 메일 (총정리)
- TP 8 · FN 2 · FP 1
- ❌ 미검출: `PERSON`=정해인, `PERSON`=손나은 · ⚠️ 오탐후보: `ORGANIZATION`=팀

