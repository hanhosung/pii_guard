"""Calibration probe — confirm which value formats each category actually detects.
Run from a real file (NER subprocess uses spawn and cannot import from stdin)."""
import random
from pii_guard.engine import Engine
from pii_guard.stage2.runner import Stage2NERRunner
from pii_guard.categories import _rrn_checksum, _kr_biz_checksum


def gen_rrn(seed):
    random.seed(seed)
    for _ in range(300000):
        y = f"{random.randint(0,99):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}"
        g = random.choice("1234"); m = f"{random.randint(0,999999):06d}"
        r = f"{y}-{g}{m}"
        if _rrn_checksum(r.replace('-', '')):
            return r


def gen_biz(seed):
    random.seed(seed)
    for _ in range(400000):
        c = f"{random.randint(0,9999999999):010d}"
        if _kr_biz_checksum(c):
            return c


def main():
    eng = Engine(stage2_runner=Stage2NERRunner())
    rrns = [gen_rrn(i) for i in range(6)]
    bizs = [gen_biz(i) for i in range(4)]
    print("VALID_RRN =", rrns)
    print("VALID_BIZ =", bizs)
    print()

    cands = {
        "EMAIL": "hong.gildong@navercorp.com",
        "PHONE": "010-9876-5432",
        "CARD_visa": "4111-1111-1111-1111",
        "CARD_mc": "5500005555555559",
        "RRN": rrns[0],
        "BIZ_NO": bizs[0],
        "PASSPORT_M": "M12345678",
        "PASSPORT_S": "S98765432",
        "DRIVER_a": "11-19-123456-01",
        "DRIVER_b": "서울 12-34-567890-12",
        "FOREIGN_REG": "900101-5234567",
        "ACCT_kookmin_626": "123456-78-901234",
        "ACCT_365": "110-123456-78901",
        "ACCT_woori_436": "1002-345-678901",
        "ACCT_bare_336": "123-456-789012",
        "API_KEY_sk": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
        "AWS_SECRET": "AKIAIOSFODNN7EXAMPLE",
        "GCP_AIza": "AIzaSyA1234567890abcdefghijklmnopqrstuvw",
        "GITHUB_ghp": "ghp_1234567890abcdefghijklmnopqrstuvwxyz12",
        "JWT": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123ghiJKL",
        "PASSWORD_label": "비밀번호: Hunter2!xZ9",
        "PASSWORD_en": "password=Sup3rSecret!2024",
        "PERSON_honorific": "홍길동 대리가 보고했다",
        "PERSON_bare": "어제 김서연이 전화했어",
        "ADDRESS": "서울 강남구 테헤란로 123",
        "ADDRESS_road": "부산 해운대구 우동 1408",
        "ORG_company": "삼성전자에서 근무한다",
        "ORG_bank": "국민은행 계좌로 입금",
        "DOB": "1990-01-15 생",
    }
    print(f"{'label':22} | {'detected (category: value)'}")
    print("-" * 80)
    for label, v in cands.items():
        r = eng.scan(v)
        got = [(d.category, d.original) for d in r.detections]
        print(f"{label:22} | {got if got else 'MISS'}")


if __name__ == "__main__":
    main()
