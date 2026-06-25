# 외부 LLM(Gemini) 생성 테스트 결과 — PII-Guard 검출 · GLiNER 엔진 (2026-06-23)

> 입력: `gemini_cases.json` (10항목) · 외부 LLM(Codex 등) 생성 텍스트를 PII-Guard에 입력해 채점. 입력 텍스트는 spaCy 리포트 부록에서 재구성(동일 데이터).
> 엔진: Stage1(정규식·체크섬) + Stage2 NER(GLiNER · urchade/gliner_multi_pii-v1) + proximity · 20 카테고리.

## 1. 핵심 결과

| 지표 | 수치 |
| :-- | :-- |
| **재현율(Recall)** | **0.847**  (61/72) |
| **정밀도(Precision)** | **0.910**  (61/67) |
| 검출 TP / 미검출 FN / 오탐후보 FP | 61 / 11 / 6 |

> ※ 오탐후보(FP)에는 ground truth에 라벨 안 된 실제 PII(은행명 등)나 NER 변동분이 섞일 수 있으니, 아래 §4 부록의 항목별 오탐 목록을 검토해 진짜 over-masking과 구분하세요.

## 2. 카테고리별 재현율

| 카테고리 | TP | FN | recall |
| :-- | --: | --: | --: |
| ADDRESS | 3 | 0 | 1.00 |
| API_KEY | 2 | 1 | 0.67 ⚠️ |
| AWS_SECRET | 3 | 0 | 1.00 |
| BIZ_NO | 5 | 0 | 1.00 |
| CARD | 4 | 0 | 1.00 |
| DRIVER_LICENSE | 1 | 0 | 1.00 |
| EMAIL | 6 | 0 | 1.00 |
| FOREIGN_REG | 0 | 2 | 0.00 ⚠️ |
| GCP_KEY | 1 | 0 | 1.00 |
| HOSTNAME | 7 | 0 | 1.00 |
| IP_ADDRESS | 5 | 0 | 1.00 |
| KR_ACCOUNT | 1 | 3 | 0.25 ⚠️ |
| PASSPORT | 2 | 0 | 1.00 |
| PASSWORD | 0 | 2 | 0.00 ⚠️ |
| PERSON | 8 | 2 | 0.80 ⚠️ |
| PHONE | 7 | 0 | 1.00 |
| PRIVATE_KEY | 1 | 0 | 1.00 |
| RRN | 4 | 0 | 1.00 |
| TOKEN | 1 | 1 | 0.50 ⚠️ |

## 3. 항목별 요약

| # | 제목 | 길이 | 심은 | 검출 | 미검출 | 오탐 | block |
| --: | :-- | --: | --: | --: | --: | --: | :--: |
| 1 | VOC 01 · 결제 수단 등록 실패 및 계정 잠김 문의 | 434 | 6 | 6 | 0 | 1 | 🔴 |
| 2 | LOG 01 · 인증 서버 API Key 노출 및 세션 만료 로그 | 1007 | 7 | 7 | 0 | 1 | 🔴 |
| 3 | VOC 02 · 오프라인 매장 영수증 인증 및 포인트 적립 누락 | 367 | 6 | 6 | 0 | 0 | 🔴 |
| 4 | LOG 02 · 결제 게이트웨이 웹훅 수신 및 계정 검증 로그 | 871 | 9 | 8 | 1 | 0 | 🔴 |
| 5 | VOC 03 · 법인 회원 정보 변경 및 정산 증빙 서류 제출 안내 요청 | 409 | 6 | 4 | 2 | 2 | 🔴 |
| 6 | LOG 03 · 데이터베이스 마이그레이션 중 자격 증명 유출 예외 로그 | 667 | 9 | 8 | 1 | 1 | 🔴 |
| 7 | VOC 04 · 글로벌 배송 주소 수정 및 여권번호 예외 처리 요청 | 401 | 6 | 6 | 0 | 0 | 🔴 |
| 8 | LOG 04 · 클라우드 스토리지 동기화 에러 및 자격증명 노출 | 836 | 7 | 4 | 3 | 0 | 🔴 |
| 9 | VOC 05 · 가상자산 대행 거래 환불 및 신원 검증 요청 | 445 | 7 | 5 | 2 | 1 | 🔴 |
| 10 | LOG 05 · 인프라 통합 모니터링 에이전트 자격 증명 수집 로그 | 883 | 9 | 7 | 2 | 0 | 🔴 |

## 4. 부록 — 전체 항목(텍스트·검출/미검출)

### [1] VOC 01 · 결제 수단 등록 실패 및 계정 잠김 문의  (434자) · 🔴 block

```
안녕하세요. 쇼핑몰 결제 오류 때문에 글 남깁니다. 제 이름은 박지민이고, 연락처는 010-4321-8765입니다. 가입된 이메일 주소는 jimin.park@webmail.com입니다. 다름이 아니라 이번에 사업자 계정으로 전환하면서 결제 카드를 등록하려고 하는데, 계속 유효하지 않은 카드라고 뜨네요. 등록하려던 카드는 4633-8750-0474-3953입니다. 혹시 몰라서 저희 회사 사업자등록번호인 434-67-77758로 조회를 해봐도 연동에 실패했다고만 나옵니다. 지금 세션 오류 메시지로 ERR_CONN_403이 계속 발생하고 있고, 결제하려던 주문번호는 ORD-2026-1102입니다. 현재 급하게 물품을 구매해야 하는 상황이라 정산용 우리은행 계좌인 1002-987-654321로 직접 송금할 테니 수동 처리해 주실 수 있나요? 확인하시면 꼭 연락 부탁드립니다.
```

- **심은(6)**: `PERSON`=박지민, `PHONE`=010-4321-8765, `EMAIL`=jimin.park@webmail.com, `CARD`=4633-8750-0474-3953, `BIZ_NO`=434-67-77758, `KR_ACCOUNT`=1002-987-654321
- ✅ **검출(6)**: `PERSON`=박지민, `PHONE`=010-4321-8765, `EMAIL`=jimin.park@webmail.com, `CARD`=4633-8750-0474-3953, `BIZ_NO`=434-67-77758, `KR_ACCOUNT`=1002-987-654321
- ❌ **미검출(0)**: —
- ⚠️ **오탐후보(1)**: `ORGANIZATION`=저희 회사

### [2] LOG 01 · 인증 서버 API Key 노출 및 세션 만료 로그  (1007자) · 🔴 block

```
2026-06-23T14:32:01.002Z [INFO] [auth-svc] Starting authentication token verification for user_id=99281
2026-06-23T14:32:01.045Z [DEBUG] [auth-svc] Request headers: { host: "api.internal", x-forwarded-for: "192.168.1.15", user-agent: "Mozilla/5.0 (v2.4.1)" }
2026-06-23T14:32:01.120Z [WARN] [auth-svc] Hardcoded credential detected in environment variables. HOSTNAME: auth-master.corp, API_KEY: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
2026-06-23T14:32:01.250Z [ERROR] [auth-svc] Failed to connect to third-party verification API. Failed object dump:
{
  "client_name": "최승우",
  "driver_license": "11-19-123456-01",
  "aws_token_draft": "AKIA1234567890123456",
  "callback_url": "https://api.openai.com/v1"
}
2026-06-23T14:32:01.300Z [FATAL] [auth-svc] Critical security leak: PRIVATE_KEY found in log dump.
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0X8O3vGQ3p...[TRUNCATED].../9kBc=
-----END RSA PRIVATE KEY-----
2026-06-23T14:32:01.350Z [INFO] [auth-svc] System health check port 8080 status OK.
```

- **심은(7)**: `IP_ADDRESS`=192.168.1.15, `HOSTNAME`=auth-master.corp, `API_KEY`=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890, `PERSON`=최승우, `DRIVER_LICENSE`=11-19-123456-01, `AWS_SECRET`=AKIA1234567890123456, `PRIVATE_KEY`=-----BEGIN RSA PRIVATE KEY-----
- ✅ **검출(7)**: `IP_ADDRESS`=192.168.1.15, `HOSTNAME`=auth-master.corp, `API_KEY`=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890, `PERSON`=최승우, `DRIVER_LICENSE`=11-19-123456-01, `AWS_SECRET`=AKIA1234567890123456, `PRIVATE_KEY`=-----BEGIN RSA PRIVATE KEY-----
- ❌ **미검출(0)**: —
- ⚠️ **오탐후보(1)**: `HOSTNAME`=api.internal

### [3] VOC 02 · 오프라인 매장 영수증 인증 및 포인트 적립 누락  (367자) · 🔴 block

```
안녕하세요. 지난 주말에 서초구 반포대로 234에 있는 매장에서 물품을 구매한 소비자 이서연입니다. 당시에 적립을 깜빡해서 사후 적립을 하려고 홈페이지에 들어왔는데 자꾸 시스템 장애가 발생하네요. 영수증에 적힌 가맹점 번호는 189-24-78001로 되어 있고요. 결제한 카드는 국민카드 4755-1313-7353-7994 입니다. 본인 확인용으로 회원 가입할 때 썼던 제 주민번호 910722-3023658 정보와 휴대폰 번호 010-2233-4455를 남기니 확인 후 적립 누락 건 처리 부탁드립니다. 에러 팝업창에는 코드번호 ERR_POINT_VAL (v1.0.9) 라고 뜹니다. 영수증 일련번호는 TX-9988231 입니다. 신속한 확인 부탁드려요.
```

- **심은(6)**: `ADDRESS`=서초구 반포대로 234, `PERSON`=이서연, `BIZ_NO`=189-24-78001, `CARD`=4755-1313-7353-7994, `RRN`=910722-3023658, `PHONE`=010-2233-4455
- ✅ **검출(6)**: `ADDRESS`=서초구 반포대로 234, `PERSON`=이서연, `BIZ_NO`=189-24-78001, `CARD`=4755-1313-7353-7994, `RRN`=910722-3023658, `PHONE`=010-2233-4455
- ❌ **미검출(0)**: —
- ⚠️ **오탐후보(0)**: —

### [4] LOG 02 · 결제 게이트웨이 웹훅 수신 및 계정 검증 로그  (871자) · 🔴 block

```
2026-06-23 17:10:12 [INFO] Received webhook from external PG. Endpoint: /api/v2/payments
2026-06-23 17:10:12 [DEBUG] Request source IP: 10.240.0.34, Target Host: payment-worker.internal
2026-06-23 17:10:13 [INFO] Parsing payload. Metadata verification payload dump:
{
  "order_id": "ORD-ABC-9921",
  "customer": {
    "name": "정민우",
    "email": "minwoo.jung@gmail.corp",
    "phone": "010-5566-7788",
    "passport_num": "M12345678"
  },
  "billing": {
    "card_no": "4075-1163-7265-1675",
    "masked_rrn_sample": "710310-4151262"
  }
}
2026-06-23 17:10:14 [WARN] JWT validation debug token active: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
2026-06-23 17:10:15 [ERROR] Webhook verification handshake failed with error status 502. Retrying on port 8443...
```

- **심은(9)**: `IP_ADDRESS`=10.240.0.34, `HOSTNAME`=payment-worker.internal, `PERSON`=정민우, `EMAIL`=minwoo.jung@gmail.corp, `PHONE`=010-5566-7788, `PASSPORT`=M12345678, `CARD`=4075-1163-7265-1675, `RRN`=710310-4151262, `TOKEN`=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
- ✅ **검출(8)**: `IP_ADDRESS`=10.240.0.34, `HOSTNAME`=payment-worker.internal, `EMAIL`=minwoo.jung@gmail.corp, `PHONE`=010-5566-7788, `PASSPORT`=M12345678, `CARD`=4075-1163-7265-1675, `RRN`=710310-4151262, `TOKEN`=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
- ❌ **미검출(1)**: `PERSON`=정민우
- ⚠️ **오탐후보(0)**: —

### [5] VOC 03 · 법인 회원 정보 변경 및 정산 증빙 서류 제출 안내 요청  (409자) · 🔴 block

```
수고하십니다. 저희는 대행업체인데 이번에 법인 정산 정보 변경 건으로 연락드렸습니다. 담당자 이름은 한지영 대리이고 연동 메일은 jyhan@partner-company.com 입니다. 유선 연락처는 010-8888-9999로 주시면 됩니다. 이번에 대표자가 바뀌면서 사업자등록번호 688-17-36719 정보 변경과 함께 환급 계좌를 하나은행 333-9102-33445 로 교체하려고 서류를 준비 중입니다. 새로 등록할 대표자 신원 정보 확인을 위해 외국인등록번호 120923-1591783 자료도 함께 첨부파일로 업로드하려니 계속 파일 확장자 에러(ERR_EXT_LIMIT)가 나면서 전송이 거부되네요. 시스템 버전이 v3.2.0으로 업데이트된 이후에 이런 현상이 잦은 것 같은데, 보안 설정 때문에 차단된 것인지 확인 바랍니다.
```

- **심은(6)**: `PERSON`=한지영, `EMAIL`=jyhan@partner-company.com, `PHONE`=010-8888-9999, `BIZ_NO`=688-17-36719, `KR_ACCOUNT`=333-9102-33445, `FOREIGN_REG`=120923-1591783
- ✅ **검출(4)**: `PERSON`=한지영, `EMAIL`=jyhan@partner-company.com, `PHONE`=010-8888-9999, `BIZ_NO`=688-17-36719
- ❌ **미검출(2)**: `KR_ACCOUNT`=333-9102-33445, `FOREIGN_REG`=120923-1591783
- ⚠️ **오탐후보(2)**: `ORGANIZATION`=대행업체인데, `RRN`=120923-1591783

### [6] LOG 03 · 데이터베이스 마이그레이션 중 자격 증명 유출 예외 로그  (667자) · 🔴 block

```
2026-06-23 11:22:45 [INFO] db-migrator execution started. Target: prod-db.local
2026-06-23 11:22:46 [DEBUG] Connected source cluster via IP 172.16.22.81
2026-06-23 11:22:47 [ERROR] Migration crashed at row #45521. Dumping environment state for replication context.
-- ENVIRONMENT OVERVIEW --
DB_PASS=P@ssw0rd12345!
GCP_KEYS=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q
LAST_MODIFIED_BY="강동우"
LAST_KNOWN_ADDR="강남구 테헤란로 501"
CONTACT_NUM="010-7711-2233"
-- RECORD DUMP --
{
  "idx": "SEQ-2026-00923",
  "biz_code": "751-47-92277",
  "card_fallback": "4612-2202-9729-9758"
}
2026-06-23 11:22:48 [FATAL] Process aborted with status 500. Container port 5432 closed unexpectedly.
```

- **심은(9)**: `HOSTNAME`=prod-db.local, `IP_ADDRESS`=172.16.22.81, `PASSWORD`=P@ssw0rd12345!, `GCP_KEY`=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q, `PERSON`=강동우, `ADDRESS`=강남구 테헤란로 501, `PHONE`=010-7711-2233, `BIZ_NO`=751-47-92277, `CARD`=4612-2202-9729-9758
- ✅ **검출(8)**: `HOSTNAME`=prod-db.local, `IP_ADDRESS`=172.16.22.81, `GCP_KEY`=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q, `PERSON`=강동우, `ADDRESS`=강남구 테헤란로 501, `PHONE`=010-7711-2233, `BIZ_NO`=751-47-92277, `CARD`=4612-2202-9729-9758
- ❌ **미검출(1)**: `PASSWORD`=P@ssw0rd12345!
- ⚠️ **오탐후보(1)**: `ADDRESS`=5432 closed unexpect

### [7] VOC 04 · 글로벌 배송 주소 수정 및 여권번호 예외 처리 요청  (401자) · 🔴 block

```
해외 직구 배송대행지 입력 오류로 수정 요청 드립니다. 주문자 김현우이고, 전화번호는 010-1111-2222입니다. 제가 원래 배송받을 주소를 영등포구 여의서로 43으로 적어야 하는데 오타가 났습니다. 그리고 통관용 여권번호 입력란에 착오로 제 여권번호 S98765432 대신 주민등록번호인 940413-4961351을 적어 넣은 것을 방금 확인했습니다. 마이페이지에서 직접 수정하려고 하니 '이미 출고 절차가 진행 중(CODE-409-SHIPPED)'이라면서 수정이 불가능하다고 나옵니다. 송장번호는 TRACK-002931-KR 인데 통관 거부될까 봐 너무 걱정되네요. 수동으로 여권번호와 주소 변경 처리 반영을 부탁드립니다. 가입 메일주소인 hwkim@global-post.com 으로 처리 결과 회신해 주세요.
```

- **심은(6)**: `PERSON`=김현우, `PHONE`=010-1111-2222, `ADDRESS`=영등포구 여의서로 43, `PASSPORT`=S98765432, `RRN`=940413-4961351, `EMAIL`=hwkim@global-post.com
- ✅ **검출(6)**: `PERSON`=김현우, `PHONE`=010-1111-2222, `ADDRESS`=영등포구 여의서로 43, `PASSPORT`=S98765432, `RRN`=940413-4961351, `EMAIL`=hwkim@global-post.com
- ❌ **미검출(0)**: —
- ⚠️ **오탐후보(0)**: —

### [8] LOG 04 · 클라우드 스토리지 동기화 에러 및 자격증명 노출  (836자) · 🔴 block

```
2026-06-23 09:05:11 [INFO] [storage-syncer] Job ID sync-task-883 initiated.
2026-06-23 09:05:11 [DEBUG] Destination node details - Host: backup-target.corp, Binding IP: 10.0.12.145
2026-06-23 09:05:12 [WARN] [storage-syncer] Client access validation triggered. Checking internal mapping data...
2026-06-23 09:05:12 [INFO] Mapping owner details found. Owner Name: 윤서준, Registered Business ID: 898-08-21922
2026-06-23 09:05:13 [ERROR] Synchronizer encountered 401 Unauthorized while accessing S3 bucket resources.
Failed parameters dump:
{
  "aws_access_key_id": "AKIA9988776655443322",
  "github_token_draft": "ghp_1234567890abcdefghijklmnopqrstVXYZ",
  "account_routing": "농협은행 302-1234-5678-90",
  "target_patch_version": "v4.1.2"
}
2026-06-23 09:05:14 [FATAL] Backup task terminated unexpectedly on local port 9000. Stack trace saved.
```

- **심은(7)**: `HOSTNAME`=backup-target.corp, `IP_ADDRESS`=10.0.12.145, `PERSON`=윤서준, `BIZ_NO`=898-08-21922, `AWS_SECRET`=AKIA9988776655443322, `API_KEY`=ghp_1234567890abcdefghijklmnopqrstVXYZ, `KR_ACCOUNT`=302-1234-5678-90
- ✅ **검출(4)**: `HOSTNAME`=backup-target.corp, `IP_ADDRESS`=10.0.12.145, `BIZ_NO`=898-08-21922, `AWS_SECRET`=AKIA9988776655443322
- ❌ **미검출(3)**: `PERSON`=윤서준, `API_KEY`=ghp_1234567890abcdefghijklmnopqrstVXYZ, `KR_ACCOUNT`=302-1234-5678-90
- ⚠️ **오탐후보(0)**: —

### [9] VOC 05 · 가상자산 대행 거래 환불 및 신원 검증 요청  (445자) · 🔴 block

```
안녕하세요. 거래소 서비스 연동 오류로 긴급 환불 요청 드립니다. 신원확인 대상자 이름은 최예은이며 연락처는 010-3344-5566입니다. 거래 중 자꾸 세션이 튕겨서 수동 인증을 신청합니다. 제 외국인등록번호는 700523-4376198 입니다. 환불을 진행하고자 하는 제 계좌는 기업은행 010-992341-12-011 입니다. 연동 중이던 외부 API Gateway 주소는 api.internal 이고 마스터 토큰 값으로 설정해 둔 ghp_ABCdefGHIjklMNOpqrSTUvwxyz1234567890 이 자꾸 만료되었다고 메시지가 발생하는데 해결이 안 됩니다. 에러 당시의 가상자산 거래 해시 트랜잭션 번호는 TX_HASH_2026_0623 이며 포트 443 환경에서 작동 중이었습니다. 확인해주시고 빠른 처리 및 메일(yeeun.choi@cryptomail.net)로 답변 주세요.
```

- **심은(7)**: `PERSON`=최예은, `PHONE`=010-3344-5566, `FOREIGN_REG`=700523-4376198, `KR_ACCOUNT`=010-992341-12-011, `HOSTNAME`=api.internal, `API_KEY`=ghp_ABCdefGHIjklMNOpqrSTUvwxyz1234567890, `EMAIL`=yeeun.choi@cryptomail.net
- ✅ **검출(5)**: `PERSON`=최예은, `PHONE`=010-3344-5566, `HOSTNAME`=api.internal, `API_KEY`=ghp_ABCdefGHIjklMNOpqrSTUvwxyz1234567890, `EMAIL`=yeeun.choi@cryptomail.net
- ❌ **미검출(2)**: `FOREIGN_REG`=700523-4376198, `KR_ACCOUNT`=010-992341-12-011
- ⚠️ **오탐후보(1)**: `RRN`=700523-4376198

### [10] LOG 05 · 인프라 통합 모니터링 에이전트 자격 증명 수집 로그  (883자) · 🔴 block

```
2026-06-23 15:40:55 [INFO] [monitor-agent] Heartbeat verification started for node-03.local
2026-06-23 15:40:55 [DEBUG] Network context: { local_ip: "192.168.100.221", outbound_proxy: "proxy.internal" }
2026-06-23 15:40:56 [WARN] [monitor-agent] Insufficient masking on core system process memory string dump:
========================================================================
USER_CTX: {
  "user_name": "정다은",
  "residential_id": "120923-1591783",
  "contact_email": "daeun.jung@enterprise.corp",
  "temporary_pass": "Mypassword99!"
}
CREDENTIALS_CTX: {
  "jwt_access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dXNlcl9pZCI6MTIzNDU.signature",
  "aws_access_key": "AKIA9999888877776666"
}
========================================================================
2026-06-23 15:40:57 [ERROR] Sync task aborted due to timeout code TIMEOUT_ERR_99. Software configuration level v5.0.1.
```

- **심은(9)**: `HOSTNAME`=node-03.local, `IP_ADDRESS`=192.168.100.221, `HOSTNAME`=proxy.internal, `PERSON`=정다은, `RRN`=120923-1591783, `EMAIL`=daeun.jung@enterprise.corp, `PASSWORD`=Mypassword99!, `TOKEN`=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dXNlcl9pZCI6MTIzNDU.signature, `AWS_SECRET`=AKIA9999888877776666
- ✅ **검출(7)**: `HOSTNAME`=node-03.local, `IP_ADDRESS`=192.168.100.221, `HOSTNAME`=proxy.internal, `PERSON`=정다은, `RRN`=120923-1591783, `EMAIL`=daeun.jung@enterprise.corp, `AWS_SECRET`=AKIA9999888877776666
- ❌ **미검출(2)**: `PASSWORD`=Mypassword99!, `TOKEN`=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dXNlcl9pZCI6MTIzNDU.signature
- ⚠️ **오탐후보(0)**: —
