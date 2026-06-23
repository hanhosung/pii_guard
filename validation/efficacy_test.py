# -*- coding: utf-8 -*-
"""
PII-Guard 실효성 검증 하니스 (30 케이스).

방법론
------
- 각 케이스 = 한국어 중심 ~1000자 텍스트 + **ground truth**(심어둔 실제 PII 목록) + **traps**(PII처럼
  보이지만 PII가 아닌 값 — 오탐 측정용).
- 실제 `Engine`(Stage1 + Stage2 NER, ko_core_news_lg)으로 scan.
- 채점:
    TP(검출)  = ground truth 항목 중 호환 카테고리 탐지가 스팬을 덮은 것
    FN(미검출) = ground truth 항목 중 탐지 못 한 것
    FP(오탐)  = 어떤 ground truth와도 매칭 안 되는 탐지 (trap 매칭 = 확정 오탐)
- 카테고리 동등성: 시크릿류는 SECRET 클래스 내 호환 매칭(github→API_KEY 등 엔진 분류 차이 흡수).
- 산출물: 콘솔 요약 + `efficacy_test_log.txt`(증거 로그) + `EFFICACY_REPORT.md`(리포트).

실행:  PYTHONPATH=. .venv/bin/python validation/efficacy_test.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from pii_guard.engine import Engine
from pii_guard.stage2.runner import Stage2NERRunner

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "efficacy_test_log.txt")
MD_PATH = os.path.join(HERE, "EFFICACY_REPORT.md")

SECRET_CATS = {"API_KEY", "AWS_SECRET", "GCP_KEY", "TOKEN", "PRIVATE_KEY", "PASSWORD"}


def cats_compatible(detected: str, expected: str) -> bool:
    if detected == expected:
        return True
    if detected in SECRET_CATS and expected in SECRET_CATS:
        return True
    return False


def norm(s: str) -> str:
    return s.replace(" ", "").replace("-", "").replace("\n", "").lower()


def span_match(det_val: str, exp_val: str) -> bool:
    a, b = norm(det_val), norm(exp_val)
    if not a or not b:
        return False
    return a in b or b in a


# ──────────────────────────────────────────────────────────────────────────────
# 30 테스트 케이스
#   expected: (category, value)  — 텍스트에 심은 실제 PII
#   traps:    value              — PII처럼 보이나 PII 아님 (탐지되면 오탐)
# ──────────────────────────────────────────────────────────────────────────────
TESTS = [
    {
        "id": 1, "title": "고객 상담 메모",
        "text": (
            "오늘 오전에 김서연 고객님이 직접 전화 주셨습니다. 연락처는 010-3344-5566이고, 추가 안내는 "
            "이메일 seoyeon.kim@gmail.com 으로 보내달라고 하셨어요. 본인 확인 과정에서 주민등록번호 "
            "920723-1641620 불러주셨는데 시스템에 저장은 안 했습니다. 배송지는 서울 마포구 월드컵북로 120 "
            "3층이고, 환불 계좌는 국민은행 123456-78-901234 라고 하셨습니다. 결제했던 카드 4111-1111-1111-1111 "
            "끝자리가 안 맞아 재확인 요청 드렸고요. 참고로 상담 티켓 번호는 ORD-20240613-0091 입니다. "
            "다음 상담은 6월 20일 오후 2시로 예약했습니다."
        ),
        "expected": [("PERSON", "김서연"), ("PHONE", "010-3344-5566"),
                     ("EMAIL", "seoyeon.kim@gmail.com"), ("RRN", "920723-1641620"),
                     ("ADDRESS", "서울 마포구 월드컵북로 120"), ("KR_ACCOUNT", "123456-78-901234"),
                     ("CARD", "4111-1111-1111-1111")],
        "traps": ["ORD-20240613-0091"],
    },
    {
        "id": 2, "title": "개발자 슬랙 메시지 (시크릿 유출)",
        "text": (
            "@here 긴급합니다. 방금 스테이징 배포하다가 실수로 환경변수를 로그에 찍어버렸어요. "
            "AWS 키 AKIAIOSFODNN7EXAMPLE 랑 시크릿이 그대로 노출됐고, OpenAI 키도 "
            "sk-abcdefghijklmnopqrstuvwxyz1234567890 이게 커밋에 들어갔습니다. 깃헙 토큰 "
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz12 도 같이요. 일단 전부 무효화(rotate) 하겠습니다. "
            "DB 접속 비번도 password=Sup3rSecret!2024 로 평문이라 바꿔야 해요. "
            "관련 PR은 #1423, 빌드 버전은 v2.3.1 입니다. 인프라팀 박지훈 님께 공유 부탁드려요. "
            "사내 위키 https://wiki.internal/runbook 참고하세요."
        ),
        "expected": [("AWS_SECRET", "AKIAIOSFODNN7EXAMPLE"),
                     ("API_KEY", "sk-abcdefghijklmnopqrstuvwxyz1234567890"),
                     ("API_KEY", "ghp_1234567890abcdefghijklmnopqrstuvwxyz12"),
                     ("PASSWORD", "Sup3rSecret!2024"), ("PERSON", "박지훈")],
        "traps": ["#1423", "v2.3.1"],
    },
    {
        "id": 3, "title": "병원 접수 안내 이메일",
        "text": (
            "안녕하세요, 환자분. 내일 진료 예약 확인차 연락드립니다. 성함 이도현, 생년월일 1988-03-12, "
            "연락처 010-7788-9900 으로 등록되어 있습니다. 접수 시 주민번호 770123-4279267 필요합니다. "
            "초진 문진표는 dohyun.lee@daum.net 으로 발송했으니 작성 부탁드립니다. 병원 위치는 "
            "서울 송파구 올림픽로 300 메디컬타워 5층이며, 주차 등록 차량은 12가 3456 입니다. "
            "수납은 카드 5500-0055-5555-5559 또는 현금 가능합니다. 진료과는 정형외과, 담당의는 최민호 과장입니다."
        ),
        "expected": [("PERSON", "이도현"), ("PHONE", "010-7788-9900"),
                     ("RRN", "770123-4279267"), ("EMAIL", "dohyun.lee@daum.net"),
                     ("ADDRESS", "서울 송파구 올림픽로 300"), ("CARD", "5500-0055-5555-5559"),
                     ("PERSON", "최민호")],
        "traps": ["1988-03-12", "12가 3456"],
    },
    {
        "id": 4, "title": "채용 지원 서류 요약",
        "text": (
            "지원자 정보 정리합니다. 이름 강하늘, 휴대폰 010-2233-4455, 이메일 haneul.kang@kakao.com. "
            "현 주소는 경기도 성남시 분당구 판교역로 235 이고, 전 직장은 네이버였습니다. "
            "포트폴리오는 깃헙 https://github.com/haneulk 에 있고, 비상연락망으로 부친 010-9001-2002 도 받았습니다. "
            "여권번호 M12345678 (해외 출장 이력 확인용), 운전면허 11-19-123456-01 소지. "
            "희망 연봉은 6,500만원, 입사 가능일 7월 1일. 학력은 2015년 졸업입니다."
        ),
        "expected": [("PERSON", "강하늘"), ("PHONE", "010-2233-4455"),
                     ("EMAIL", "haneul.kang@kakao.com"),
                     ("ADDRESS", "경기도 성남시 분당구 판교역로 235"), ("ORGANIZATION", "네이버"),
                     ("PHONE", "010-9001-2002"), ("PASSPORT", "M12345678"),
                     ("DRIVER_LICENSE", "11-19-123456-01")],
        "traps": ["6,500만원", "2015년"],
    },
    {
        "id": 5, "title": "택배 분실 문의 (비표준 계좌 — 갭)",
        "text": (
            "배송 못 받아서 문의드려요. 받는 사람 윤지우, 전화 010-5566-7788 입니다. "
            "주소는 인천 연수구 송도과학로 32 아파트 105동 1203호예요. 보상은 신한은행 계좌로 받을게요, "
            "계좌번호는 110-123456-78901 입니다. 혹시 안 되면 다른 계좌 123-456-789012 로 부탁해요. "
            "구매할 때 적은 이메일은 jiwoo.yoon@naver.com 이고요. 운송장 번호는 123456789012 입니다. "
            "빠른 처리 부탁드립니다. 사업자등록번호 180-63-41205 로 세금계산서도 필요해요."
        ),
        "expected": [("PERSON", "윤지우"), ("PHONE", "010-5566-7788"),
                     ("ADDRESS", "인천 연수구 송도과학로 32"),
                     ("KR_ACCOUNT", "110-123456-78901"),
                     ("KR_ACCOUNT", "123-456-789012"),  # 비표준 3-3-6 → 미검출 예상(갭)
                     ("EMAIL", "jiwoo.yoon@naver.com"),
                     ("BIZ_NO", "180-63-41205")],
        "traps": ["123456789012"],  # 운송장(맨 12자리) — 계좌로 오탐 가능성 점검
    },
    {
        "id": 6, "title": "부동산 계약 메모",
        "text": (
            "매매 계약 건 정리. 매도인 한상우, 연락처 010-1212-3434. 매수인 오세훈, 010-5656-7878. "
            "물건지는 서울 강남구 도곡로 401 타워팰리스 입니다. 매도인 주민번호 340128-1381696 확인했고, "
            "계약금은 우리은행 1002-345-678901 로 입금 예정입니다. 중개사무소는 대박공인중개사, "
            "사업자번호 469-80-91148. 등기 관련 서류는 sangwoo.han@gmail.com 으로 보냅니다. "
            "계약일은 6월 25일, 잔금일은 8월 30일입니다. 매매가 18억 5천."
        ),
        "expected": [("PERSON", "한상우"), ("PHONE", "010-1212-3434"),
                     ("PERSON", "오세훈"), ("PHONE", "010-5656-7878"),
                     ("ADDRESS", "서울 강남구 도곡로 401"), ("RRN", "340128-1381696"),
                     ("KR_ACCOUNT", "1002-345-678901"), ("BIZ_NO", "469-80-91148"),
                     ("EMAIL", "sangwoo.han@gmail.com")],
        "traps": ["18억 5천"],
    },
    {
        "id": 7, "title": "외국인 직원 인사 등록",
        "text": (
            "신규 입사 외국인 직원 등록 건입니다. 영문명 John Smith, 한글명 존스미스, "
            "외국인등록번호 900101-5234567 입니다. 연락처 010-4040-5050, 이메일 john.smith@company.io. "
            "거주지는 서울 용산구 이태원로 200 입니다. 여권번호는 S98765432 이고, "
            "급여 계좌는 하나은행 123456-78-901234 로 등록했습니다. 비자 만료일 2026-12-31. "
            "소속은 글로벌사업팀이며 사번은 EMP-2024-0312 입니다."
        ),
        "expected": [("PERSON", "존스미스"), ("FOREIGN_REG", "900101-5234567"),
                     ("PHONE", "010-4040-5050"), ("EMAIL", "john.smith@company.io"),
                     ("ADDRESS", "서울 용산구 이태원로 200"), ("PASSPORT", "S98765432"),
                     ("KR_ACCOUNT", "123456-78-901234")],
        "traps": ["EMP-2024-0312", "2026-12-31"],
    },
    {
        "id": 8, "title": "회의록 (tool_result 시뮬레이션)",
        "text": (
            "[시스템 로그 첨부] 운영 서버 점검 결과입니다. 관리자 계정 admin 의 임시 비밀번호는 "
            "password=Adm1n!Temp2024 로 설정되어 있었고, 즉시 변경 권고합니다. 서비스 계정 키 "
            "AIzaSyA1234567890abcdefghijklmnopqrstuvw 가 코드에 하드코딩되어 있습니다. "
            "JWT 세션 토큰 eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123ghiJKL "
            "도 로그에 남아 있었습니다. 점검 담당자 신우진 책임, 내선 010-6060-7070. "
            "다음 점검은 분기말. 영향받는 서비스는 결제 모듈 v4.1.0 입니다."
        ),
        "expected": [("PASSWORD", "Adm1n!Temp2024"),
                     ("GCP_KEY", "AIzaSyA1234567890abcdefghijklmnopqrstuvw"),
                     ("TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123ghiJKL"),
                     ("PERSON", "신우진"), ("PHONE", "010-6060-7070")],
        "traps": ["v4.1.0", "admin"],
    },
    {
        "id": 9, "title": "은행 콜센터 스크립트 (한글 비번 라벨 — 갭)",
        "text": (
            "고객님 본인확인 진행하겠습니다. 성함 배수지 고객님 맞으신가요? 등록된 전화 010-7070-8080 "
            "마지막 네 자리 확인됐습니다. 인터넷뱅킹 비밀번호: SecurePw99 라고 메모에 적혀 있는데 "
            "보안상 폐기하겠습니다. 주민등록번호 010809-2201058 확인 완료했고요. 자택 주소는 "
            "대전 유성구 대학로 99 입니다. 출금 계좌 신한 110-123456-78901, 이메일 suji.bae@hanmail.net. "
            "최근 거래 내역은 앱에서 확인 가능합니다. 상담사 번호는 1588-0000 입니다."
        ),
        "expected": [("PERSON", "배수지"), ("PHONE", "010-7070-8080"),
                     ("PASSWORD", "SecurePw99"),  # "비밀번호:" 한글 라벨 → 미검출 예상(갭)
                     ("RRN", "010809-2201058"), ("ADDRESS", "대전 유성구 대학로 99"),
                     ("KR_ACCOUNT", "110-123456-78901"), ("EMAIL", "suji.bae@hanmail.net")],
        "traps": ["1588-0000"],
    },
    {
        "id": 10, "title": "개인 일기 (문맥형 PII)",
        "text": (
            "오랜만에 대학 동기 모임에 다녀왔다. 민준이랑 서윤이는 여전했고, 특히 박건우는 "
            "삼성전자 들어갔다고 자랑하더라. 나는 다음 주에 카카오 면접이 있어서 좀 긴장된다. "
            "집에 오는 길에 강남역 근처에서 우연히 옛 직장 상사 김태형 부장님을 만났는데, "
            "연락처 받아뒀다. 010-3030-2020. 다음에 밥 한번 먹기로 했다. "
            "요즘 사는 동네 서초구 반포대로 58 은 조용하고 좋다. 내일은 운동 좀 해야지."
        ),
        "expected": [("PERSON", "박건우"), ("ORGANIZATION", "삼성전자"),
                     ("ORGANIZATION", "카카오"), ("PERSON", "김태형"),
                     ("PHONE", "010-3030-2020"), ("ADDRESS", "서초구 반포대로 58")],
        "traps": ["강남역"],
    },
    {
        "id": 11, "title": "온라인 쇼핑 주문 확인",
        "text": (
            "주문해주셔서 감사합니다. 주문자 임채원 님, 결제 금액 89,000원 확인되었습니다. "
            "배송지: 광주 서구 상무중앙로 7 오피스텔 808호. 연락처 010-8181-9292. "
            "결제 카드 4111-1111-1111-1111 승인 완료. 적립 포인트는 chaewon.lim@gmail.com 계정에 "
            "쌓였습니다. 교환/반품 문의는 고객센터로 부탁드립니다. 주문번호 2024-0613-77882, "
            "송장번호는 발송 후 안내드립니다. 멤버십 등급은 GOLD입니다."
        ),
        "expected": [("PERSON", "임채원"), ("ADDRESS", "광주 서구 상무중앙로 7"),
                     ("PHONE", "010-8181-9292"), ("CARD", "4111-1111-1111-1111"),
                     ("EMAIL", "chaewon.lim@gmail.com")],
        "traps": ["89,000원", "2024-0613-77882", "GOLD"],
    },
    {
        "id": 12, "title": "법률 상담 요청서",
        "text": (
            "교통사고 손해배상 관련 상담 요청합니다. 의뢰인 성명 조은별, 주민번호 300504-4502140. "
            "사고 일시 6월 1일, 장소는 부산 해운대구 우동 1408 사거리. 연락처 010-2727-3838, "
            "이메일 eunbyul.jo@outlook.com. 상대 차량 운전자는 면허번호 23-45-678901-23 소지자였고, "
            "보험사는 삼성화재입니다. 합의금 입금 계좌는 농협 1002-345-678901. "
            "관련 진단서와 사고 사진은 별도 첨부합니다. 사건번호는 추후 부여 예정."
        ),
        "expected": [("PERSON", "조은별"), ("RRN", "300504-4502140"),
                     ("ADDRESS", "부산 해운대구 우동 1408"), ("PHONE", "010-2727-3838"),
                     ("EMAIL", "eunbyul.jo@outlook.com"),
                     ("DRIVER_LICENSE", "23-45-678901-23"), ("ORGANIZATION", "삼성화재"),
                     ("KR_ACCOUNT", "1002-345-678901")],
        "traps": [],
    },
    {
        "id": 13, "title": "DevOps 인시던트 리포트",
        "text": (
            "장애 보고: 02시 14분 결제 API 오류 급증. 원인은 만료된 인증서와 유출된 자격증명입니다. "
            "유출된 프라이빗 키 일부:\n-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFh\n-----END RSA PRIVATE KEY-----\n"
            "그리고 슬랙 웹훅에 박힌 토큰 ghp_abcdEFGH1234567890ijklMNOP1234567890qZ 발견. "
            "대응 담당 한지민, 비상 연락 010-9999-8888. 영향받은 고객 약 1,200명. "
            "롤백 버전은 release-2024.06.13. 사후 분석 회의는 내일 10시입니다."
        ),
        "expected": [("PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----"),
                     ("API_KEY", "ghp_abcdEFGH1234567890ijklMNOP1234567890qZ"),
                     ("PERSON", "한지민"), ("PHONE", "010-9999-8888")],
        "traps": ["release-2024.06.13", "1,200명"],
    },
    {
        "id": 14, "title": "학원 등록 상담",
        "text": (
            "자녀 수학 단과 등록 문의 주셔서 감사합니다. 학생 이름 최서진, 학부모 성함 최영수. "
            "학부모 연락처 010-4545-6767, 비상연락 010-7654-3210. 집 주소는 "
            "대구 수성구 동대구로 23 입니다. 등록비는 카드 5500-0055-5555-5559 로 결제 가능하고, "
            "계좌이체는 국민 123456-78-901234 도 됩니다. 안내 메일은 youngsoo.choi@gmail.com 으로 "
            "보내드릴게요. 셔틀버스 노선은 A코스입니다. 개강은 7월 첫째 주."
        ),
        "expected": [("PERSON", "최서진"), ("PERSON", "최영수"),
                     ("PHONE", "010-4545-6767"), ("PHONE", "010-7654-3210"),
                     ("ADDRESS", "대구 수성구 동대구로 23"), ("CARD", "5500-0055-5555-5559"),
                     ("KR_ACCOUNT", "123456-78-901234"), ("EMAIL", "youngsoo.choi@gmail.com")],
        "traps": ["A코스"],
    },
    {
        "id": 15, "title": "환자 차트 인계 노트",
        "text": (
            "야간 인계합니다. 301호 환자 권나라, 760312-2345671 진통제 투여 후 안정. "
            "보호자 연락처 010-1313-2424. 302호 환자 정우성, 주소 서울 종로구 종로 1 거주, "
            "내일 오전 퇴원 예정입니다. 약 처방전은 nara.kwon@medi.kr 로 발송 요청 들어왔어요. "
            "303호는 금식 유지. 응급 연락은 당직의 010-5151-6262. 병동 비밀번호는 분기마다 변경됩니다. "
            "다음 라운딩은 6시입니다."
        ),
        "expected": [("PERSON", "권나라"), ("RRN", "760312-2345671"),
                     ("PHONE", "010-1313-2424"), ("PERSON", "정우성"),
                     ("ADDRESS", "서울 종로구 종로 1"), ("EMAIL", "nara.kwon@medi.kr"),
                     ("PHONE", "010-5151-6262")],
        "traps": [],
    },
    {
        "id": 16, "title": "프리랜서 계약/정산",
        "text": (
            "6월 외주 정산 안내드립니다. 작업자 성함 남도일, 사업자등록번호 010-24-69680 (간이과세). "
            "정산 금액 330만원, 입금 계좌는 카카오뱅크 3333-01-1234567 입니다. "
            "세금계산서 발행 위해 이메일 doil.nam@gmail.com 확인 부탁드려요. "
            "연락처 010-3636-4747. 계약서상 주소는 서울 영등포구 여의대로 24. "
            "다음 프로젝트는 8월 시작 예정이며, NDA 서명본은 별도 보관합니다. 인보이스 번호 INV-0613."
        ),
        "expected": [("PERSON", "남도일"), ("BIZ_NO", "010-24-69680"),
                     ("KR_ACCOUNT", "3333-01-1234567"), ("EMAIL", "doil.nam@gmail.com"),
                     ("PHONE", "010-3636-4747"), ("ADDRESS", "서울 영등포구 여의대로 24")],
        "traps": ["330만원", "INV-0613"],
    },
    {
        "id": 17, "title": "보험 가입 설계",
        "text": (
            "종신보험 설계 내역입니다. 피보험자 황민영, 주민번호 310124-2427977. "
            "월 납입액 12만원, 가입자 연락처 010-8282-9393, 이메일 minyoung.hwang@nate.com. "
            "자동이체 계좌 우리 1002-345-678901 등록 예정. 주소는 울산 남구 삼산로 100. "
            "수익자는 배우자 황지훈으로 지정. 청약 카드 4111-1111-1111-1111 로 첫 회 결제. "
            "보장 개시일은 익월 1일입니다. 증권번호는 발급 후 안내드립니다."
        ),
        "expected": [("PERSON", "황민영"), ("RRN", "310124-2427977"),
                     ("PHONE", "010-8282-9393"), ("EMAIL", "minyoung.hwang@nate.com"),
                     ("KR_ACCOUNT", "1002-345-678901"), ("ADDRESS", "울산 남구 삼산로 100"),
                     ("PERSON", "황지훈"), ("CARD", "4111-1111-1111-1111")],
        "traps": ["12만원"],
    },
    {
        "id": 18, "title": "코드 리뷰 코멘트 (코드+PII 혼합)",
        "text": (
            "리뷰 남깁니다. config.py 에 테스트용 자격증명이 그대로 있네요. "
            "API_KEY = 'sk-test1234567890ABCDEFGHIJKLMNOPQRSTUV' 이건 .env 로 빼주세요. "
            "그리고 send_email(to='customer.report@bizmail.com') 호출에서 실제 고객 메일이 하드코딩됐어요. "
            "주석에 '담당: 류현진 010-1010-2020' 도 지워주시고요. DB_PASSWORD=devpass2024! 도 위험합니다. "
            "함수명 get_user_rrn() 는 괜찮은데 리턴값 로깅은 빼주세요. 변수 order_id=ORD-2024-0613 는 유지. "
            "전반적으로 LGTM, 위 5개만 수정 부탁해요."
        ),
        "expected": [("API_KEY", "sk-test1234567890ABCDEFGHIJKLMNOPQRSTUV"),
                     ("EMAIL", "customer.report@bizmail.com"), ("PERSON", "류현진"),
                     ("PHONE", "010-1010-2020"), ("PASSWORD", "devpass2024!")],
        "traps": ["get_user_rrn()", "ORD-2024-0613", "config.py"],
    },
    {
        "id": 19, "title": "동호회 회원 명부",
        "text": (
            "테니스 동호회 신규 회원 명단 공유합니다. 1. 서지호 010-1122-3344 서울 노원구 동일로 1414. "
            "2. 문가영 010-5566-1122 경기 고양시 일산동구 중앙로 1275. 3. 백승호 010-9988-7766. "
            "회비 입금은 회장 계좌 신한 110-123456-78901 로 부탁드립니다. "
            "단체 카톡방 초대는 가입 신청서의 이메일 주소로 보냅니다. 서지호 님은 jiho.seo@gmail.com 이고요. "
            "다음 정기 모임은 6월 22일 오전 9시, 양재 시민의 숲 코트입니다."
        ),
        "expected": [("PERSON", "서지호"), ("PHONE", "010-1122-3344"),
                     ("ADDRESS", "서울 노원구 동일로 1414"), ("PERSON", "문가영"),
                     ("PHONE", "010-5566-1122"), ("ADDRESS", "경기 고양시 일산동구 중앙로 1275"),
                     ("PERSON", "백승호"), ("PHONE", "010-9988-7766"),
                     ("KR_ACCOUNT", "110-123456-78901"), ("EMAIL", "jiho.seo@gmail.com")],
        "traps": [],
    },
    {
        "id": 20, "title": "수출 통관 서류",
        "text": (
            "수출 신고 건 정리합니다. 수출자 상호 한빛무역, 사업자등록번호 887-13-78905. "
            "담당자 오지환, 연락처 010-3434-5656, 이메일 jihwan.oh@hanbit.co.kr. "
            "수입자는 미국 바이어 Mike Brown. 선적 항구는 부산항. 결제 조건 T/T, "
            "대금 수취 계좌 외환 123456-78-901234. 컨테이너 번호 ABCU-1234567. "
            "인보이스 금액 USD 52,000. HS 코드는 8517.12 입니다. 통관 예정일 6월 18일."
        ),
        "expected": [("ORGANIZATION", "한빛무역"), ("BIZ_NO", "887-13-78905"),
                     ("PERSON", "오지환"), ("PHONE", "010-3434-5656"),
                     ("EMAIL", "jihwan.oh@hanbit.co.kr"),
                     ("KR_ACCOUNT", "123456-78-901234")],
        "traps": ["ABCU-1234567", "8517.12", "USD 52,000"],
    },
    {
        "id": 21, "title": "민원 접수 (관공서)",
        "text": (
            "도로 보수 민원 접수합니다. 민원인 강도영, 연락처 010-6767-8989. "
            "주소는 세종특별자치시 한누리대로 2130 입니다. 주민번호 920723-1641620 으로 본인인증 완료. "
            "민원 회신은 문자 또는 이메일 doyoung.kang@korea.kr 로 받기 원하십니다. "
            "현장은 아파트 정문 앞 도로 파임. 사진 3장 첨부. 접수번호 2024-MIN-00913. "
            "처리 기한은 14일 이내입니다. 담당 부서는 도로과."
        ),
        "expected": [("PERSON", "강도영"), ("PHONE", "010-6767-8989"),
                     ("ADDRESS", "세종특별자치시 한누리대로 2130"), ("RRN", "920723-1641620"),
                     ("EMAIL", "doyoung.kang@korea.kr")],
        "traps": ["2024-MIN-00913"],
    },
    {
        "id": 22, "title": "스타트업 투자 메모 (혼합 영문)",
        "text": (
            "시드 라운드 검토 노트입니다. 대표 Kevin Park(박준혁), 연락처 010-2020-3030. "
            "회사는 클라우드비전, 사업자번호 469-80-91148. pitch deck는 kevin@cloudvision.io 로 받았습니다. "
            "은행 계좌(에스크로) 하나 1002-345-678901. 법인 주소 서울 강남구 테헤란로 521. "
            "기술 데모 서버 접속 토큰 AKIAIOSFODNN7EXAMPLE 이 메일에 그대로 와서 보안 지적했습니다. "
            "밸류에이션 80억, 지분 12% 협의 중. 다음 미팅 6월 19일."
        ),
        "expected": [("PERSON", "박준혁"), ("PHONE", "010-2020-3030"),
                     ("ORGANIZATION", "클라우드비전"), ("BIZ_NO", "469-80-91148"),
                     ("EMAIL", "kevin@cloudvision.io"), ("KR_ACCOUNT", "1002-345-678901"),
                     ("ADDRESS", "서울 강남구 테헤란로 521"),
                     ("AWS_SECRET", "AKIAIOSFODNN7EXAMPLE")],
        "traps": ["80억", "12%"],
    },
    {
        "id": 23, "title": "졸업생 추천서",
        "text": (
            "추천서를 작성합니다. 학생 윤하린은 2024년 우리 학과를 졸업한 우수한 인재입니다. "
            "현재 연락처는 010-7878-9090, 이메일 harin.yoon@univ.ac.kr 입니다. "
            "재학 중 LG전자 인턴십을 성실히 수행했고, 거주지는 서울 관악구 관악로 1 입니다. "
            "추천인 본인은 컴퓨터공학과 정교수 한석규이며, 연구실 내선으로 문의 가능합니다. "
            "학점은 4.2/4.5, 졸업 작품은 수상 이력이 있습니다. 추가 문의 환영합니다."
        ),
        "expected": [("PERSON", "윤하린"), ("PHONE", "010-7878-9090"),
                     ("EMAIL", "harin.yoon@univ.ac.kr"), ("ORGANIZATION", "LG전자"),
                     ("ADDRESS", "서울 관악구 관악로 1"), ("PERSON", "한석규")],
        "traps": ["4.2/4.5", "2024년"],
    },
    {
        "id": 24, "title": "중고거래 채팅",
        "text": (
            "안녕하세요 아이폰 판매글 보고 연락드려요. 직거래 가능하실까요? 저는 신도림역 근처 살아요. "
            "혹시 안 되면 택배도 괜찮아요. 받는 주소 알려드릴게요. 서울 구로구 경인로 662, 받는사람 임수호, "
            "연락처 010-4321-8765 입니다. 입금은 카카오뱅크 3333-02-7654321 로 할게요. "
            "이메일은 sooho.lim@gmail.com. 혹시 영수증 필요하면 말씀해주세요. 네고는 살짝만 부탁드려요. "
            "물건 상태 좋아 보이네요. 내일 오후에 거래 가능합니다."
        ),
        "expected": [("ADDRESS", "서울 구로구 경인로 662"), ("PERSON", "임수호"),
                     ("PHONE", "010-4321-8765"), ("KR_ACCOUNT", "3333-02-7654321"),
                     ("EMAIL", "sooho.lim@gmail.com")],
        "traps": ["신도림역"],
    },
    {
        "id": 25, "title": "급여 명세 발송",
        "text": (
            "6월 급여명세서 발송합니다. 직원 성명 조민서, 사번 EMP-0312. 실수령액 320만원. "
            "급여 이체 계좌는 국민 123456-78-901234 입니다. 4대보험 정산 위해 주민번호 "
            "770123-4279267 확인했습니다. 명세서 PDF는 minseo.cho@corp.com 으로 발송. "
            "문의는 인사팀 010-1357-2468. 주소지 변경 있으시면 알려주세요, 현재 등록은 "
            "경기 수원시 영통구 광교중앙로 145. 연말정산 자료는 12월에 별도 안내드립니다."
        ),
        "expected": [("PERSON", "조민서"), ("KR_ACCOUNT", "123456-78-901234"),
                     ("RRN", "770123-4279267"), ("EMAIL", "minseo.cho@corp.com"),
                     ("PHONE", "010-1357-2468"), ("ADDRESS", "경기 수원시 영통구 광교중앙로 145")],
        "traps": ["EMP-0312", "320만원"],
    },
    {
        "id": 26, "title": "여행 예약 확정",
        "text": (
            "제주 여행 예약 확정 안내드립니다. 대표 예약자 손흥민, 동반 1인. 연락처 010-7000-8000. "
            "항공권 발권 위해 여권번호 M12345678 등록 완료. 숙소는 제주 서귀포시 중문관광로 72 호텔. "
            "체크인 6월 20일. 결제 카드 4111-1111-1111-1111 승인. 예약 확인 메일은 "
            "heungmin.son@gmail.com 으로 발송했습니다. 렌터카는 H사 SUV, 보험 포함. "
            "예약번호 JEJU-240613-22. 즐거운 여행 되세요!"
        ),
        "expected": [("PERSON", "손흥민"), ("PHONE", "010-7000-8000"),
                     ("PASSPORT", "M12345678"), ("ADDRESS", "제주 서귀포시 중문관광로 72"),
                     ("CARD", "4111-1111-1111-1111"), ("EMAIL", "heungmin.son@gmail.com")],
        "traps": ["JEJU-240613-22"],
    },
    {
        "id": 27, "title": "전세 대출 상담 (민감 집약)",
        "text": (
            "전세자금대출 상담 기록입니다. 신청인 김연아, 주민등록번호 760312-2345671. "
            "재직 회사 현대자동차, 연소득 6천만원. 연락처 010-2468-1357, 이메일 yuna.kim@hmail.com. "
            "임차 주소 서울 송파구 백제고분로 50. 기존 거래 계좌 신한 110-123456-78901. "
            "결혼 예정으로 배우자 이상화도 공동 명의 검토. 대출 한도는 심사 후 안내. "
            "운전면허 11-19-123456-01 사본 제출 완료. 상담사 박명수, 사번 0913."
        ),
        "expected": [("PERSON", "김연아"), ("RRN", "760312-2345671"),
                     ("ORGANIZATION", "현대자동차"), ("PHONE", "010-2468-1357"),
                     ("EMAIL", "yuna.kim@hmail.com"), ("ADDRESS", "서울 송파구 백제고분로 50"),
                     ("KR_ACCOUNT", "110-123456-78901"), ("PERSON", "이상화"),
                     ("DRIVER_LICENSE", "11-19-123456-01"), ("PERSON", "박명수")],
        "traps": ["6천만원", "0913"],
    },
    {
        "id": 28, "title": "AI 챗봇 대화 로그 (은연중 유출)",
        "text": (
            "사용자: 내 정보로 자기소개서 좀 다듬어줘. 이름은 한예슬이고 1995년생, "
            "이메일 yeseul.han@gmail.com 으로 연락 와. 사는 곳은 서울 동작구 사당로 300 이고 "
            "전화는 010-1199-2288 이야. 전 직장은 SK하이닉스였어. 아 그리고 회사 노트북 비번이 "
            "password=Hanyeseul2024! 인데 이것도 어디 적어두면 편할까? 다음 주 면접 회사는 "
            "쿠팡이야. 긴장된다. 경력 5년차로 정리해줘. 고마워!"
        ),
        "expected": [("PERSON", "한예슬"), ("EMAIL", "yeseul.han@gmail.com"),
                     ("ADDRESS", "서울 동작구 사당로 300"), ("PHONE", "010-1199-2288"),
                     ("ORGANIZATION", "SK하이닉스"), ("PASSWORD", "Hanyeseul2024!"),
                     ("ORGANIZATION", "쿠팡")],
        "traps": ["1995년생", "5년차"],
    },
    {
        "id": 29, "title": "비표준 포맷 집중 (약점 노출)",
        "text": (
            "정보 정리: 담당자 노홍철(연락 010.3535.7979 — 점으로 구분). "
            "백업 연락처는 02-555-1234 유선. 계좌는 우리 123-456-789012 (자주 쓰는 거). "
            "사업자번호 1806341205 (하이픈 없이). 이메일 두 개: nono [at] example.com 과 "
            "real.email@example.com. 비밀번호는 그냥 '가나다1234!' 로 해놨어. "
            "주소는 서울특별시 중구 세종대로 110. 카드 뒷자리만: 1111. 끝."
        ),
        "expected": [("PERSON", "노홍철"),
                     ("PHONE", "010.3535.7979"),       # 점 구분 → 미검출 가능(갭)
                     ("KR_ACCOUNT", "123-456-789012"),  # 3-3-6 → 미검출 예상(갭)
                     ("BIZ_NO", "1806341205"),          # 하이픈 없음 → 미검출 예상(갭)
                     ("EMAIL", "real.email@example.com"),
                     ("ADDRESS", "서울특별시 중구 세종대로 110")],
        "traps": ["02-555-1234", "nono [at] example.com", "1111"],
    },
    {
        "id": 30, "title": "복합 업무 메일 (총정리)",
        "text": (
            "팀 공유: 신규 고객 온보딩 정보입니다. 회사 토스페이먼츠, 사업자 469-80-91148. "
            "키맨 정해인 이사, 010-8421-7421, haein.jung@toss.im. 계약 보증금은 "
            "기업은행 123456-78-901234 로 수령. 기술 연동용 임시 API 키 "
            "sk-live9876543210ZYXWVUTSRQPONMLKJIH 를 발급했으니 운영 전 폐기하세요. "
            "담당 개발 손나은, 주민번호 010809-2201058 은 보안서약서용으로만 받았습니다. "
            "사무실 주소 서울 강남구 강남대로 382. 본 메일은 대외비입니다. 회신 바랍니다."
        ),
        "expected": [("ORGANIZATION", "토스페이먼츠"), ("BIZ_NO", "469-80-91148"),
                     ("PERSON", "정해인"), ("PHONE", "010-8421-7421"),
                     ("EMAIL", "haein.jung@toss.im"), ("KR_ACCOUNT", "123456-78-901234"),
                     ("API_KEY", "sk-live9876543210ZYXWVUTSRQPONMLKJIH"),
                     ("PERSON", "손나은"), ("RRN", "010809-2201058"),
                     ("ADDRESS", "서울 강남구 강남대로 382")],
        "traps": [],
    },
]


def run():
    eng = Engine(stage2_runner=Stage2NERRunner())
    log_lines = []
    per_cat = defaultdict(lambda: {"tp": 0, "fn": 0})
    fp_total = 0
    case_rows = []

    def L(s=""):
        log_lines.append(s)

    L("=" * 90)
    L("PII-Guard 실효성 검증 로그 (30 케이스)")
    L("엔진: Engine + Stage2NERRunner (ko_core_news_lg) · 모드: 실전형 · 측정: 재현율+정밀도")
    L("=" * 90)

    for t in TESTS:
        res = eng.scan(t["text"])
        dets = [(d.category, d.original, str(d.action).split(".")[-1]) for d in res.detections]

        matched_det = set()
        tp, fn = [], []
        for (ecat, eval_) in t["expected"]:
            hit = None
            for i, (dcat, dval, _) in enumerate(dets):
                if i in matched_det:
                    continue
                if cats_compatible(dcat, ecat) and span_match(dval, eval_):
                    hit = i
                    break
            if hit is not None:
                matched_det.add(hit)
                tp.append((ecat, eval_, dets[hit][0], dets[hit][1]))
                per_cat[ecat]["tp"] += 1
            else:
                fn.append((ecat, eval_))
                per_cat[ecat]["fn"] += 1

        fps = [dets[i] for i in range(len(dets)) if i not in matched_det]
        fp_total += len(fps)

        L("")
        L("─" * 90)
        L(f"[{t['id']:02d}] {t['title']}")
        L("─" * 90)
        L("TEXT:")
        L("  " + t["text"].replace("\n", "\n  "))
        L(f"\nGROUND TRUTH ({len(t['expected'])}): " +
          ", ".join(f"{c}={v}" for c, v in t["expected"]))
        if t["traps"]:
            L(f"TRAPS (탐지되면 오탐): {', '.join(t['traps'])}")
        L(f"\nRAW DETECTIONS ({len(dets)}):")
        for dcat, dval, dact in dets:
            L(f"  · {dcat:14} [{dact}] {dval!r}")
        L(f"\n✅ 검출 TP ({len(tp)}):")
        for ecat, eval_, dcat, dval in tp:
            note = "" if ecat == dcat else f"  (분류:{dcat})"
            L(f"  · {ecat:14} {eval_!r}{note}")
        L(f"❌ 미검출 FN ({len(fn)}):")
        for ecat, eval_ in fn:
            L(f"  · {ecat:14} {eval_!r}")
        L(f"⚠️  오탐 후보 FP ({len(fps)}):")
        for dcat, dval, dact in fps:
            trap = "  [TRAP-확정오탐]" if any(span_match(dval, tv) for tv in t["traps"]) else ""
            L(f"  · {dcat:14} {dval!r}{trap}")

        case_rows.append({
            "id": t["id"], "title": t["title"], "n_expected": len(t["expected"]),
            "tp": len(tp), "fn": len(fn), "fp": len(fps),
            "fn_items": fn, "fp_items": fps, "n_det": len(dets),
            "has_block": res.has_blocks,
            "text": t["text"], "expected": t["expected"], "traps": t["traps"],
            "detections": dets, "tp_items": [(a, b) for a, b, _, _ in tp],
        })

    # 집계
    total_exp = sum(c["n_expected"] for c in case_rows)
    total_tp = sum(c["tp"] for c in case_rows)
    total_fn = sum(c["fn"] for c in case_rows)
    recall = total_tp / total_exp if total_exp else 0
    precision = total_tp / (total_tp + fp_total) if (total_tp + fp_total) else 0

    L("")
    L("=" * 90)
    L("집계")
    L("=" * 90)
    L(f"케이스: {len(case_rows)} · ground truth 항목: {total_exp}")
    L(f"검출 TP: {total_tp} · 미검출 FN: {total_fn} · 오탐 FP: {fp_total}")
    L(f"재현율(Recall) = TP/(TP+FN) = {total_tp}/{total_exp} = {recall:.3f}")
    L(f"정밀도(Precision) = TP/(TP+FP) = {total_tp}/{total_tp+fp_total} = {precision:.3f}")
    L("\n카테고리별:")
    L(f"  {'category':16} {'TP':>4} {'FN':>4} {'recall':>8}")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        L(f"  {cat:16} {c['tp']:>4} {c['fn']:>4} {r:>8.2f}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # 결과를 MD 생성기로 전달하기 위해 JSON 사이드카도 저장
    summary = {
        "n_cases": len(case_rows), "total_exp": total_exp, "total_tp": total_tp,
        "total_fn": total_fn, "fp_total": fp_total, "recall": recall, "precision": precision,
        "per_cat": {k: dict(v) for k, v in per_cat.items()}, "cases": case_rows,
    }
    with open(os.path.join(HERE, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    generate_md(case_rows, per_cat, summary)

    print(f"recall={recall:.3f} precision={precision:.3f} TP={total_tp} FN={total_fn} FP={fp_total}")
    print(f"log → {LOG_PATH}")
    print(f"report → {MD_PATH}")


# ── 트리아지(triage) — 채점 방법론의 한계 보정 ────────────────────────────────
# 비라벨 정탐: 실제 PII인데 ground truth에 안 넣은 것(은행명=조직, 영문 이름 등) → 진짜 오탐 아님
LEGIT_FP = {
    (1, "국민은행"), (5, "신한은행"), (6, "우리은행"), (7, "하나은행"), (12, "농협"),
    (16, "카카오뱅크"), (24, "카카오뱅크"), (30, "기업은행"),
    (7, "John Smith"), (20, "Mike Brown"), (22, "Kevin Park"),
    (26, "제주 서귀포시 중문관광로 72"), (19, "서지호"),
}
# 출제 오류 FN: 무효 체크섬 값 → 정상적으로 미검출(엔진 갭 아님)
AUTHORING_FN = {(15, "760312-2345671"), (27, "760312-2345671")}


def generate_md(cases, per_cat, summary):
    tp, fn, fp = summary["total_tp"], summary["total_fn"], summary["fp_total"]
    legit = sum(1 for c in cases for f in c["fp_items"] if (c["id"], f[1]) in LEGIT_FP)
    auth = sum(1 for c in cases for (cat, v) in c["fn_items"] if (c["id"], v) in AUTHORING_FN)
    true_fp = fp - legit
    tp_adj = tp + legit
    real_fn = fn - auth
    rec_raw = tp / (tp + fn)
    prec_raw = tp / (tp + fp)
    rec_adj = tp / (tp + real_fn)
    prec_adj = tp_adj / (tp_adj + true_fp)

    o = []
    def W(s=""):
        o.append(s)

    W("# PII-Guard 실효성 검증 리포트")
    W()
    W("> 생성: 자동화 하니스 `validation/efficacy_test.py` · 증거 로그: "
      "[`efficacy_test_log.txt`](./efficacy_test_log.txt)")
    W("> 엔진: `Engine` + `Stage2NERRunner`(ko_core_news_lg) · 30 케이스 · "
      "한국어 중심+영문 시크릿 · **실전형(까다롭게)** · 재현율+정밀도 측정")
    W()
    W("---")
    W()
    W("## 1. 방법론")
    W()
    W("- 30개 케이스 각각 = **한국어 ~1000자 텍스트** + **ground truth**(심어둔 실제 PII) + "
      "**traps**(PII처럼 보이나 PII 아님 — 오탐 측정용).")
    W("- 실제 엔진으로 `scan()` 후 채점: **TP**(검출), **FN**(미검출), **FP**(오탐).")
    W("- 매칭: 카테고리 호환 + 스팬 포함관계. 시크릿류는 SECRET 클래스 내 호환(github→API_KEY 등 흡수).")
    W("- **트리아지**: 채점 자동화의 한계상, ① ground truth에 안 넣은 실제 PII(은행명·영문이름)는 "
      "*비라벨 정탐*으로, ② 무효 체크섬 출제 오류 FN은 *정상 미검출*로 보정한 수치를 함께 제시.")
    W()
    W("## 2. 핵심 결과")
    W()
    W("### 2.1 먼저 — 세 가지 숫자만 알면 됩니다")
    W()
    W("탐지 결과는 항상 세 칸으로 나뉩니다. (비유: 가방 검사에서 *위험물*을 찾는 검사관)")
    W()
    W("| 약자 | 이름 | 쉬운 뜻 | 비유 |")
    W("| :-- | :-- | :-- | :-- |")
    W("| **TP** | 검출 (True Positive) | 진짜 PII를 **제대로 잡음** | 위험물을 위험물로 |")
    W("| **FN** | 미검출 (False Negative) | 진짜 PII를 **놓침** (빠져나감) | 위험물을 그냥 통과시킴 |")
    W("| **FP** | 오탐 (False Positive) | PII 아닌 걸 **PII로 잘못 잡음** (과잉) | 멀쩡한 물건을 위험물로 압수 |")
    W()
    W(f"이번 검증: 텍스트에 **심어둔 진짜 PII = {summary['total_exp']}개**. 이 중 "
      f"**{tp}개 잡고(TP)**, **{fn}개 놓치고(FN)**, 추가로 **{fp}개를 잘못 잡았습니다(FP 후보)**.")
    W()
    W("### 2.2 재현율(Recall) — \"놓치지 않은 비율\"")
    W()
    W("> **공식:  재현율 = TP ÷ (TP + FN) = 잡은 진짜 PII ÷ 전체 진짜 PII**")
    W()
    W("- **의미**: 실제로 있는 PII 중 **몇 %를 빠뜨리지 않고 잡았나.** 유출 방지 도구에서 **가장 중요한** 지표 "
      "(놓치면 곧 유출).")
    W(f"- **이번 수치(raw)**: {tp} ÷ ({tp} + {fn}) = {tp} ÷ {tp+fn} = **{rec_raw:.3f}**  "
      f"→ 진짜 PII의 약 **{rec_raw*100:.0f}%**를 잡음.")
    W(f"- 한 줄 해석: 100개 PII가 있으면 약 **{rec_raw*100:.0f}개를 막고 {(1-rec_raw)*100:.0f}개를 놓치는** 수준.")
    W()
    W("### 2.3 정밀도(Precision) — \"잡은 것 중 진짜 비율\"")
    W()
    W("> **공식:  정밀도 = TP ÷ (TP + FP) = 진짜 PII ÷ 내가 잡은 전체**")
    W()
    W("- **의미**: 엔진이 \"PII\"라고 잡은 것 중 **몇 %가 실제로 PII였나.** 낮으면 **과잉 마스킹**(멀쩡한 "
      "단어·코드를 가림)이 많다는 뜻 → 작업 방해.")
    W(f"- **이번 수치(raw)**: {tp} ÷ ({tp} + {fp}) = {tp} ÷ {tp+fp} = **{prec_raw:.3f}**  "
      f"→ 잡은 것의 약 **{prec_raw*100:.0f}%**가 진짜 PII.")
    W(f"- 한 줄 해석: 100번 \"PII다\"라고 가리면 약 **{(1-prec_raw)*100:.0f}번은 헛가림**(과잉).")
    W()
    W("### 2.4 왜 'Raw'와 'Triaged(보정)' 두 가지인가")
    W()
    W("자동 채점에는 두 가지 **불공정**이 끼어 있어, 보정 전(raw)과 후(triaged)를 함께 제시합니다.")
    W()
    W(f"- **보정① — 비라벨 정탐 {legit}건** (정밀도에 영향): 은행명(국민·신한은행 등)=조직, 영문 이름"
      "(John Smith 등)은 **사실 진짜 PII인데 제가 정답표(ground truth)에 안 적어둔** 것입니다. "
      "엔진은 맞게 잡았는데 채점이 \"오탐\"으로 처리 → 이걸 정탐으로 되돌립니다.")
    W(f"- **보정② — 출제 오류 FN {auth}건** (재현율에 영향): 제가 만든 더미 주민번호 중 "
      "**체크섬이 무효**인 게 있었습니다. 무효 번호는 엔진이 안 잡는 게 **정상**이므로, 이 미검출은 "
      "엔진 잘못이 아니라 제 출제 실수 → 분모에서 제외합니다.")
    W()
    W(f"| 지표 | Raw (보정 전) | **Triaged (보정 후)** | 보정 내용 |")
    W("| :-- | :-- | :-- | :-- |")
    W(f"| **재현율** | {rec_raw:.3f}  ({tp}/{tp+fn}) | **{rec_adj:.3f}**  ({tp}/{tp+real_fn}) | "
      f"출제오류 FN {auth}건 제외(분모 {tp+fn}→{tp+real_fn}) |")
    W(f"| **정밀도** | {prec_raw:.3f}  ({tp}/{tp+fp}) | **{prec_adj:.3f}**  ({tp_adj}/{tp_adj+true_fp}) | "
      f"비라벨 정탐 {legit}건을 TP로(오탐 {fp}→{true_fp}) |")
    W()
    W("> 즉 **실제 성능은 보정값(재현율 ~{0:.2f} / 정밀도 ~{1:.2f})에 가깝고**, 진짜 over-masking은 "
      "{2}건입니다.".format(rec_adj, prec_adj, true_fp))
    W()
    W("### 2.5 보안 관점 — 두 지표 중 무엇이 더 중요한가")
    W()
    W("- PII-Guard는 **유출 방지** 도구이므로 **재현율(놓침 최소화)이 1순위**입니다. 놓친 PII = 곧 유출.")
    W("- 정밀도(과잉 마스킹)는 **작업 편의** 문제 — 보안상으론 *과소보다 안전한 방향*이나, 코드 작업을 "
      "방해할 수 있어 개선 대상입니다(§5·§7).")
    W(f"- 또한 시크릿·주민번호 등 **block 카테고리**가 포함된 케이스는 마스킹이 아니라 **요청 자체가 차단**되어 "
      "업스트림에 도달하지 않습니다(케이스 표의 🔴).")
    W()
    W("## 3. 카테고리별 재현율")
    W()
    W("| 카테고리 | TP | FN | recall |")
    W("| :-- | --: | --: | --: |")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        flag = " ⚠️" if r < 0.85 else ""
        W(f"| {cat} | {c['tp']} | {c['fn']} | {r:.2f}{flag} |")
    W()
    W("> ⚠️ = recall < 0.85. 시크릿·카드·여권·이메일·전화는 사실상 완전 검출, "
      "**약점은 KR_ACCOUNT(비표준 포맷)·ORGANIZATION(NER)·RRN·PASSWORD**에 집중.")
    W()
    W("## 4. 미검출(FN) 분석 — 진짜 갭 vs 출제 오류")
    W()
    W("**(A) 실제 엔진 갭 (개선 대상):**")
    W()
    W("| 케이스 | 카테고리 | 값 | 원인 |")
    W("| :-- | :-- | :-- | :-- |")
    gap_reason = {
        "123-456-789012": "비표준 3-3-6 계좌(라벨 없음) — 오탐억제 설계상 미검출(기지 갭)",
        "3333-01-1234567": "카카오/토스뱅크 3-2-7 포맷 미등록",
        "3333-02-7654321": "카카오/토스뱅크 3-2-7 포맷 미등록",
        "1806341205": "하이픈 없는 10자리 사업자번호 미검출",
        "SecurePw99": "한글 라벨('비밀번호:') 비밀번호 미검출(영문 키워드만)",
        "010-5566-1122": "NER 인접 매칭/스팬 경계 영향(점검 필요)",
        "customer.report@bizmail.com": "코드 `send_email(to=…)` 내부 — NER가 통째로 ORG 오분류해 이메일 누락",
    }
    for c in cases:
        for cat, v in c["fn_items"]:
            if (c["id"], v) in AUTHORING_FN:
                continue
            reason = gap_reason.get(v, "NER 미탐지(단일 언급·문맥 부족)")
            W(f"| {c['id']:02d} | {cat} | `{v}` | {reason} |")
    W()
    W("**(B) 출제 오류 (엔진 정상 — 무효 체크섬이라 미검출이 정답):**")
    W()
    for c in cases:
        for cat, v in c["fn_items"]:
            if (c["id"], v) in AUTHORING_FN:
                W(f"- 케이스 {c['id']:02d} {cat} `{v}` — 체크섬 무효 더미값. 엔진이 안 잡는 게 정상.")
    W()
    W("## 5. 오탐(FP) 분석")
    W()
    W(f"오탐 후보 {fp}건을 분류하면:")
    W()
    W(f"- **비라벨 정탐 {legit}건** (실제 PII인데 ground truth에 안 넣음 → 사실상 정탐): "
      "은행명(국민·신한·우리·하나·농협·카카오뱅크·기업은행)=ORGANIZATION, 영문 이름(John Smith·Mike Brown·Kevin Park)=PERSON 등.")
    W(f"- **진짜 over-masking {true_fp}건** (비PII를 PII로 오인 — 실제 정밀도 손해):")
    W()
    W("| 케이스 | 오분류 | 값 | 유형 |")
    W("| :-- | :-- | :-- | :-- |")
    for c in cases:
        for f in c["fp_items"]:
            if (c["id"], f[1]) in LEGIT_FP:
                continue
            val = f[1]
            if any(t in val for t in ["send_email", "DB_PASSWORD", "API_KEY", "LGTM", "주석", "리턴값", "로깅", "MIIB"]):
                typ = "코드/기술 토큰 NER 오분류"
            elif f[0] == "PERSON":
                typ = "일반명사→인물 오분류"
            elif f[0] == "ORGANIZATION":
                typ = "일반명사/약어→조직 오분류"
            elif f[0] == "ADDRESS":
                typ = "지명/역명→주소 오분류"
            elif f[0] == "PHONE":
                typ = "유선번호(02-) — 사실상 정탐에 가까움"
            else:
                typ = "기타"
            W(f"| {c['id']:02d} | {f[0]} | `{val[:40]}` | {typ} |")
    W()
    W("> **패턴**: 진짜 오탐은 ① **코드 리뷰/로그 텍스트**(케이스 18·13·08)의 식별자·키워드를 NER가 "
      "인물/조직으로 오인, ② 일반명사('수익자'·'여권번호'·'네고')를 인물로 오인하는 데 집중. "
      "→ 요구사항 §6.3의 **콘텐츠 클래스 게이팅(코드·base64는 NER 스킵)**을 적용하면 상당수 제거 가능.")
    W()
    W("## 6. 케이스별 요약 (30)")
    W()
    W("| # | 제목 | PII | 검출 | 미검출 | 진짜오탐 | block |")
    W("| --: | :-- | --: | --: | --: | --: | :--: |")
    for c in cases:
        tfp = sum(1 for f in c["fp_items"] if (c["id"], f[1]) not in LEGIT_FP)
        blk = "🔴" if c["has_block"] else "—"
        W(f"| {c['id']:02d} | {c['title']} | {c['n_expected']} | {c['tp']} | {c['fn']} | {tfp} | {blk} |")
    W()
    W("## 7. 핵심 발견 & 권고")
    W()
    W("1. **정형 PII는 사실상 완벽**(재현율 1.00): API키·AWS·GCP·JWT·카드·여권·운전면허·외국인등록. "
      "체크섬 검증 덕에 시크릿/고위험 신원은 신뢰성 높게 차단.")
    W("2. **최우선 개선 = KR_ACCOUNT 포맷 커버리지**: 카카오/토스뱅크(3-2-7)·비표준(3-3-6) 미검출. "
      "문맥 키워드(은행명+입금/계좌) 규칙 추가 권고.")
    W("3. **PASSWORD 한글 라벨**('비밀번호:') 미검출 — 한글 키워드 패턴 추가 필요.")
    W("4. **BIZ_NO 하이픈 없는 10자리** 미검출 — 포맷 확장 또는 문맥 검증.")
    W("5. **NER over-masking이 정밀도의 주 손실원**: 코드/기술 텍스트에서 식별자·일반명사를 인물/조직으로 "
      "오인. **콘텐츠 클래스 게이팅(§6.3)** 적용이 가장 효과적. (보안상으론 과소보다 안전한 방향이나 "
      "코드 작업 방해 가능.)")
    W("6. **마스킹 vs 차단 동작 정상**: 시크릿·주민번호 포함 케이스는 block, 연락처·이름은 mask로 처리됨.")
    W()
    W("## 8. 재현")
    W()
    W("```bash")
    W("cd /Users/ho/workspace/Monoly_genAI/pii_guard")
    W("PYTHONPATH=. .venv/bin/python validation/efficacy_test.py")
    W("# → efficacy_test_log.txt(전체 텍스트·원시 탐지 증거) + EFFICACY_REPORT.md + _summary.json")
    W("```")
    W()
    W("## 부록 A. 케이스 전문 · 검출/미검출 상세")
    W()
    W("> 원시 탐지 덤프(액션 포함)와 전체 로그는 [`efficacy_test_log.txt`](./efficacy_test_log.txt) 참조.")
    W()
    for c in cases:
        W(f"### [{c['id']:02d}] {c['title']}")
        W()
        W("**텍스트:**")
        W()
        W("> " + c["text"].replace("\n", " ").strip())
        W()
        det_tp = ", ".join(f"`{cat}`={v}" for cat, v in c["tp_items"]) or "—"
        det_fn = ", ".join(f"`{cat}`={v}" for cat, v in c["fn_items"]) or "—"
        det_fp = ", ".join(f"`{f[0]}`={f[1]}" for f in c["fp_items"]
                           if (c["id"], f[1]) not in LEGIT_FP) or "—"
        W(f"- ✅ **검출({len(c['tp_items'])})**: {det_tp}")
        W(f"- ❌ **미검출({len(c['fn_items'])})**: {det_fn}")
        W(f"- ⚠️ **진짜 오탐**: {det_fp}")
        W()

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    run()
