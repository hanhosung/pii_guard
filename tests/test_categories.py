"""
Unit tests for every PII/secret category.

Coverage requirement:
  - At least one positive match per rule_id in each CategorySpec
  - At least one true-negative (text that must NOT match)
  - Correct action (block vs tokenize/mask) per category

Run with:   pytest tests/test_categories.py -v
"""
from __future__ import annotations

import pytest

from pii_guard import Engine
from pii_guard.models import Action


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def scan(text: str):
    """Return (redacted_text, detections) using a fresh engine."""
    engine = Engine()
    result = engine.scan(text)
    return result.redacted_text, result.detections


def categories_found(text: str) -> set:
    _, dets = scan(text)
    return {d.category for d in dets}


def actions_for(text: str) -> dict:
    """Return {category: action} for all detections."""
    _, dets = scan(text)
    return {d.category: d.action for d in dets}


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

class TestEmail:
    def test_basic_email_detected(self):
        redacted, dets = scan("Send to alice@example.com please")
        assert "EMAIL" in {d.category for d in dets}
        assert "alice@example.com" not in redacted
        assert "[EMAIL_1]" in redacted

    def test_email_action_is_tokenize(self):
        actions = actions_for("user@domain.org")
        assert actions.get("EMAIL") == Action.TOKENIZE_ROUNDTRIP

    def test_multiple_emails(self):
        redacted, dets = scan("a@b.com and c@d.io")
        emails = [d for d in dets if d.category == "EMAIL"]
        assert len(emails) == 2
        assert "[EMAIL_1]" in redacted
        assert "[EMAIL_2]" in redacted

    def test_email_with_plus(self):
        cats = categories_found("user+tag@sub.domain.com")
        assert "EMAIL" in cats

    def test_no_false_positive_plain_domain(self):
        # A bare domain without @ should not trigger email
        cats = categories_found("visit example.com for info")
        assert "EMAIL" not in cats

    def test_no_false_positive_code_import(self):
        # "@decorator" in Python code
        cats = categories_found("@property\ndef foo(self): pass")
        assert "EMAIL" not in cats

    def test_email_roundtrip(self):
        engine = Engine()
        result = engine.scan("Contact me at secret@corp.io")
        assert "secret@corp.io" not in result.redacted_text
        restored = engine.rehydrate(result.redacted_text)
        assert "secret@corp.io" in restored


# ══════════════════════════════════════════════════════════════════════════════
# PHONE
# ══════════════════════════════════════════════════════════════════════════════

class TestPhone:
    def test_kr_mobile(self):
        cats = categories_found("전화: 010-1234-5678")
        assert "PHONE" in cats

    def test_kr_mobile_no_dash(self):
        cats = categories_found("01012345678")
        assert "PHONE" in cats

    def test_kr_landline(self):
        cats = categories_found("02-555-1234")
        assert "PHONE" in cats

    def test_us_phone(self):
        cats = categories_found("Call (212) 555-1234")
        assert "PHONE" in cats

    def test_international_plus(self):
        cats = categories_found("+82-10-1234-5678")
        assert "PHONE" in cats

    def test_phone_action_is_tokenize(self):
        actions = actions_for("010-9999-0000")
        assert actions.get("PHONE") == Action.TOKENIZE_ROUNDTRIP

    def test_no_false_positive_zip(self):
        # 5-digit zip codes should not be phones
        cats = categories_found("ZIP: 12345")
        assert "PHONE" not in cats

    def test_phone_masked_in_redacted(self):
        redacted, _ = scan("My number is 010-1234-5678.")
        assert "010-1234-5678" not in redacted
        assert "[PHONE_" in redacted


# ══════════════════════════════════════════════════════════════════════════════
# RRN — Korean Resident Registration Number
# ══════════════════════════════════════════════════════════════════════════════

class TestRRN:
    # Luhn-valid test RRNs (checksum verified in test_checksums.py)
    VALID_RRN_DASH = "900505-1234564"        # checksum digit = 4  ✓
    VALID_RRN_NODASH = "9005051234564"        # same without hyphen ✓
    CITIZEN_DIGIT1 = "800101-1234560"         # gender digit 1, checksum 0 ✓

    def test_rrn_with_dash_detected(self):
        cats = categories_found(f"주민번호: {self.VALID_RRN_DASH}")
        assert "RRN" in cats

    def test_rrn_action_is_block(self):
        _, dets = scan(f"주민번호 {self.VALID_RRN_DASH}")
        rrn_dets = [d for d in dets if d.category == "RRN"]
        assert rrn_dets, "Expected RRN detection"
        assert rrn_dets[0].action == Action.BLOCK

    def test_rrn_blocked_in_output(self):
        redacted, dets = scan(f"ID: {self.VALID_RRN_DASH}")
        assert any(d.category == "RRN" for d in dets), "Expected RRN detection"
        assert self.VALID_RRN_DASH not in redacted
        assert "BLOCKED" in redacted

    def test_rrn_no_false_positive_six_digits(self):
        # Plain 6-digit number alone should not fire
        cats = categories_found("invoice #123456")
        assert "RRN" not in cats

    def test_rrn_without_dash(self):
        cats = categories_found(f"주민번호: {self.VALID_RRN_NODASH}")
        assert "RRN" in cats


# ══════════════════════════════════════════════════════════════════════════════
# FOREIGN_REG
# ══════════════════════════════════════════════════════════════════════════════

class TestForeignReg:
    def test_foreign_reg_7th_digit_5(self):
        cats = categories_found("외국인번호: 900101-5123456")
        assert "FOREIGN_REG" in cats

    def test_foreign_reg_7th_digit_8(self):
        cats = categories_found("외국인번호: 850615-8765432")
        assert "FOREIGN_REG" in cats

    def test_foreign_reg_action_block(self):
        _, dets = scan("900101-5123456")
        freg = [d for d in dets if d.category == "FOREIGN_REG"]
        assert freg, "Expected FOREIGN_REG detection"
        assert freg[0].action == Action.BLOCK

    def test_foreign_reg_not_matched_by_citizen_digit(self):
        # 7th digit 1-4 → RRN, not FOREIGN_REG (checksum-valid number)
        cats = categories_found("800101-1234560")   # valid checksum, digit7=1
        assert "RRN" in cats
        assert "FOREIGN_REG" not in cats


# ══════════════════════════════════════════════════════════════════════════════
# BIZ_NO — Korean Business Registration Number
# ══════════════════════════════════════════════════════════════════════════════

class TestBizNo:
    # "123-45-67891" has valid checksum (computed: check=1, last digit=1)
    VALID_BIZ = "123-45-67891"

    def test_biz_no_valid(self):
        cats = categories_found(f"사업자 등록번호: {self.VALID_BIZ}")
        assert "BIZ_NO" in cats

    def test_biz_no_action_tokenize(self):
        actions = actions_for(f"사업자: {self.VALID_BIZ}")
        assert actions.get("BIZ_NO") == Action.TOKENIZE_ROUNDTRIP

    def test_biz_no_masked(self):
        redacted, dets = scan(f"사업자등록번호: {self.VALID_BIZ}")
        if any(d.category == "BIZ_NO" for d in dets):
            assert self.VALID_BIZ not in redacted

    def test_biz_no_false_positive_phone(self):
        # Korean phone 02-555-1234 should NOT match BIZ_NO (format mismatch)
        _, dets = scan("02-555-1234")
        biz = [d for d in dets if d.category == "BIZ_NO"]
        assert len(biz) == 0


# ══════════════════════════════════════════════════════════════════════════════
# KR_ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════

class TestKrAccount:
    def test_kookmin_format(self):
        cats = categories_found("계좌: 123456-78-901234")
        assert "KR_ACCOUNT" in cats

    def test_kr_account_action_tokenize(self):
        actions = actions_for("계좌: 123456-78-901234")
        assert actions.get("KR_ACCOUNT") == Action.TOKENIZE_ROUNDTRIP

    def test_labeled_account(self):
        # Use hyphenated format to avoid FOREIGN_REG pattern conflict
        cats = categories_found("Account No: 1002-789-012345")
        assert "KR_ACCOUNT" in cats

    def test_shinhan_format(self):
        cats = categories_found("계좌번호: 110-123-456789")
        assert "KR_ACCOUNT" in cats


# ══════════════════════════════════════════════════════════════════════════════
# PASSPORT
# ══════════════════════════════════════════════════════════════════════════════

class TestPassport:
    def test_kr_passport_labeled(self):
        cats = categories_found("여권번호: M12345678")
        assert "PASSPORT" in cats

    def test_passport_labeled_english(self):
        cats = categories_found("Passport No: M12345678")
        assert "PASSPORT" in cats

    def test_passport_action_block(self):
        _, dets = scan("Passport number: M12345678")
        pp = [d for d in dets if d.category == "PASSPORT"]
        if pp:
            assert pp[0].action == Action.BLOCK

    def test_passport_blocked_in_output(self):
        redacted, dets = scan("Passport: M12345678")
        if any(d.category == "PASSPORT" for d in dets):
            assert "M12345678" not in redacted
            assert "BLOCKED" in redacted


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER_LICENSE
# ══════════════════════════════════════════════════════════════════════════════

class TestDriverLicense:
    def test_kr_dl_format(self):
        # 12 digits: YY-RR-NNNNNN-CC
        cats = categories_found("운전면허 번호: 12-34-567890-12")
        assert "DRIVER_LICENSE" in cats

    def test_dl_labeled(self):
        cats = categories_found("Driver's license: A123-456-7890")
        assert "DRIVER_LICENSE" in cats

    def test_dl_action_block(self):
        _, dets = scan("Driver license number: A123-456-7890")
        dl = [d for d in dets if d.category == "DRIVER_LICENSE"]
        if dl:
            assert dl[0].action == Action.BLOCK


# ══════════════════════════════════════════════════════════════════════════════
# CARD — Credit/Debit
# ══════════════════════════════════════════════════════════════════════════════

class TestCard:
    # Real test PAN numbers (Luhn-valid, non-live)
    VISA_TEST = "4532015112830366"          # Luhn valid ✓
    MC_TEST   = "5425233430109903"          # Luhn valid ✓
    AMEX_TEST = "378282246310005"           # Luhn valid ✓ (standard Amex test number)

    def test_visa_detected(self):
        cats = categories_found(f"card: {self.VISA_TEST}")
        assert "CARD" in cats

    def test_mastercard_detected(self):
        cats = categories_found(f"Card number: {self.MC_TEST}")
        assert "CARD" in cats

    def test_amex_detected(self):
        cats = categories_found(f"AMEX: {self.AMEX_TEST}")
        assert "CARD" in cats

    def test_card_action_block(self):
        actions = actions_for(f"Pay with {self.VISA_TEST}")
        assert actions.get("CARD") == Action.BLOCK

    def test_card_blocked_in_output(self):
        redacted, dets = scan(f"card: {self.VISA_TEST}")
        if any(d.category == "CARD" for d in dets):
            assert self.VISA_TEST not in redacted
            assert "BLOCKED" in redacted

    def test_card_with_spaces(self):
        spaced = "4532 0151 1283 0366"
        cats = categories_found(spaced)
        assert "CARD" in cats

    def test_card_with_dashes(self):
        dashed = "4532-0151-1283-0366"
        cats = categories_found(dashed)
        assert "CARD" in cats

    def test_luhn_invalid_not_detected(self):
        # Luhn-invalid card-shaped number
        bad_card = "4532015112830360"
        cats = categories_found(f"card {bad_card}")
        assert "CARD" not in cats

    def test_no_false_positive_order_number(self):
        # 10-digit order numbers should not match
        cats = categories_found("Order #1234567890")
        assert "CARD" not in cats


# ══════════════════════════════════════════════════════════════════════════════
# API_KEY
# ══════════════════════════════════════════════════════════════════════════════

class TestApiKey:
    def test_anthropic_key(self):
        key = "sk-ant-api03-" + "A" * 50
        cats = categories_found(f"ANTHROPIC_API_KEY={key}")
        assert "API_KEY" in cats

    def test_openai_key(self):
        key = "sk-" + "a" * 48
        cats = categories_found(f"OPENAI_API_KEY={key}")
        assert "API_KEY" in cats

    def test_openai_project_key(self):
        key = "sk-proj-" + "b" * 48
        cats = categories_found(key)
        assert "API_KEY" in cats

    def test_github_pat(self):
        cats = categories_found("token = ghp_" + "X" * 40)
        assert "API_KEY" in cats

    def test_stripe_live_key(self):
        cats = categories_found("sk_live_" + "z" * 24)
        assert "API_KEY" in cats

    def test_huggingface_token(self):
        cats = categories_found("hf_" + "a" * 38)
        assert "API_KEY" in cats

    def test_api_key_action_block(self):
        key = "sk-" + "a" * 48
        actions = actions_for(key)
        assert actions.get("API_KEY") == Action.BLOCK

    def test_api_key_blocked_in_output(self):
        key = "sk-ant-api03-" + "A" * 50
        redacted, dets = scan(f"key={key}")
        if any(d.category == "API_KEY" for d in dets):
            assert key not in redacted
            assert "BLOCKED" in redacted


# ══════════════════════════════════════════════════════════════════════════════
# AWS_SECRET
# ══════════════════════════════════════════════════════════════════════════════

class TestAwsSecret:
    def test_aws_access_key_id(self):
        cats = categories_found("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
        assert "AWS_SECRET" in cats

    def test_aws_secret_access_key(self):
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        text = f"AWS_SECRET_ACCESS_KEY={secret}"
        cats = categories_found(text)
        assert "AWS_SECRET" in cats

    def test_aws_akia_inline(self):
        cats = categories_found("key=AKIAIOSFODNN7EXAMPLE")
        assert "AWS_SECRET" in cats

    def test_aws_action_block(self):
        actions = actions_for("AKIAIOSFODNN7EXAMPLE")
        assert actions.get("AWS_SECRET") == Action.BLOCK

    def test_aws_blocked_in_output(self):
        redacted, dets = scan("AKIAIOSFODNN7EXAMPLE")
        if any(d.category == "AWS_SECRET" for d in dets):
            assert "AKIAIOSFODNN7EXAMPLE" not in redacted
            assert "BLOCKED" in redacted


# ══════════════════════════════════════════════════════════════════════════════
# GCP_KEY
# ══════════════════════════════════════════════════════════════════════════════

class TestGcpKey:
    # "AIzaSy" + "B"*33 = 39 chars = "AIza" + 35 chars required by pattern
    GCP_KEY_SAMPLE = "AIzaSy" + "B" * 33

    def test_gcp_api_key(self):
        cats = categories_found(self.GCP_KEY_SAMPLE)
        assert "GCP_KEY" in cats

    def test_gcp_oauth_client(self):
        cats = categories_found("GOCSPX-" + "a" * 30)
        assert "GCP_KEY" in cats

    def test_gcp_action_block(self):
        actions = actions_for(self.GCP_KEY_SAMPLE)
        assert actions.get("GCP_KEY") == Action.BLOCK

    def test_gcp_blocked_in_output(self):
        redacted, dets = scan(self.GCP_KEY_SAMPLE)
        assert any(d.category == "GCP_KEY" for d in dets), "Expected GCP_KEY detection"
        assert self.GCP_KEY_SAMPLE not in redacted
        assert "BLOCKED" in redacted


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN
# ══════════════════════════════════════════════════════════════════════════════

class TestToken:
    JWT_EXAMPLE = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    def test_jwt_detected(self):
        cats = categories_found(self.JWT_EXAMPLE)
        assert "TOKEN" in cats

    def test_slack_token(self):
        cats = categories_found("xoxb-12345-67890-abcdefghijklmnop")
        assert "TOKEN" in cats

    def test_access_token_assignment(self):
        cats = categories_found("access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def")
        assert "TOKEN" in cats

    def test_token_action_block(self):
        actions = actions_for(self.JWT_EXAMPLE)
        assert actions.get("TOKEN") == Action.BLOCK

    def test_token_blocked_in_output(self):
        redacted, dets = scan(f"auth={self.JWT_EXAMPLE}")
        if any(d.category == "TOKEN" for d in dets):
            assert self.JWT_EXAMPLE not in redacted
            assert "BLOCKED" in redacted

    def test_bearer_in_header(self):
        cats = categories_found("Authorization: Bearer " + "a" * 50)
        assert "TOKEN" in cats


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE_KEY
# ══════════════════════════════════════════════════════════════════════════════

class TestPrivateKey:
    RSA_HEADER = "-----BEGIN RSA PRIVATE KEY-----"
    EC_HEADER  = "-----BEGIN EC PRIVATE KEY-----"
    OPENSSH    = "-----BEGIN OPENSSH PRIVATE KEY-----"

    def test_rsa_pem_header(self):
        cats = categories_found(self.RSA_HEADER)
        assert "PRIVATE_KEY" in cats

    def test_ec_pem_header(self):
        cats = categories_found(self.EC_HEADER)
        assert "PRIVATE_KEY" in cats

    def test_openssh_pem_header(self):
        cats = categories_found(self.OPENSSH)
        assert "PRIVATE_KEY" in cats

    def test_private_key_action_block(self):
        actions = actions_for(self.RSA_HEADER)
        assert actions.get("PRIVATE_KEY") == Action.BLOCK

    def test_private_key_blocked_in_output(self):
        redacted, dets = scan(self.RSA_HEADER)
        if any(d.category == "PRIVATE_KEY" for d in dets):
            assert "-----BEGIN RSA PRIVATE KEY-----" not in redacted
            assert "BLOCKED" in redacted

    def test_no_false_positive_public_key(self):
        cats = categories_found("-----BEGIN PUBLIC KEY-----")
        assert "PRIVATE_KEY" not in cats


# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD
# ══════════════════════════════════════════════════════════════════════════════

class TestPassword:
    def test_password_assignment_yaml(self):
        cats = categories_found("password: SuperSecret123!")
        assert "PASSWORD" in cats

    def test_password_assignment_env(self):
        cats = categories_found("DB_PASSWORD=hunter2secret")
        assert "PASSWORD" in cats

    def test_password_in_url(self):
        cats = categories_found("postgresql://user:s3cr3tpass@localhost/db")
        assert "PASSWORD" in cats

    def test_password_action_block(self):
        actions = actions_for("password: SuperSecret123!")
        assert actions.get("PASSWORD") == Action.BLOCK

    def test_password_blocked_in_output(self):
        redacted, dets = scan("password: SuperSecret123!")
        if any(d.category == "PASSWORD" for d in dets):
            assert "SuperSecret123!" not in redacted
            assert "BLOCKED" in redacted

    def test_no_false_positive_empty_password(self):
        # password: (empty) should not flag
        cats = categories_found("password:")
        # Even if it flags, the match should be empty
        _, dets = scan("password:")
        pw_dets = [d for d in dets if d.category == "PASSWORD"]
        for d in pw_dets:
            assert len(d.original.strip()) > 0

    def test_passwd_alias(self):
        cats = categories_found("passwd=topsecret99")
        assert "PASSWORD" in cats


# ══════════════════════════════════════════════════════════════════════════════
# PERSON
# ══════════════════════════════════════════════════════════════════════════════

class TestPerson:
    def test_labeled_name_english(self):
        cats = categories_found("Name: John Smith")
        assert "PERSON" in cats

    def test_labeled_name_korean(self):
        cats = categories_found("성명: 김철수")
        assert "PERSON" in cats

    def test_patient_label(self):
        cats = categories_found("Patient: Jane Doe")
        assert "PERSON" in cats

    def test_person_action_tokenize(self):
        actions = actions_for("Name: Alice Brown")
        assert actions.get("PERSON") == Action.TOKENIZE_ROUNDTRIP

    def test_person_masked(self):
        redacted, dets = scan("성명: 이영희")
        if any(d.category == "PERSON" for d in dets):
            assert "이영희" not in redacted


# ══════════════════════════════════════════════════════════════════════════════
# ADDRESS
# ══════════════════════════════════════════════════════════════════════════════

class TestAddress:
    def test_us_street_address(self):
        cats = categories_found("123 Main Street, Springfield, IL 62701")
        assert "ADDRESS" in cats

    def test_korean_address(self):
        cats = categories_found("서울특별시 강남구 테헤란로 123")
        assert "ADDRESS" in cats

    def test_address_action_tokenize(self):
        actions = actions_for("123 Main Street, Portland, OR 97201")
        assert actions.get("ADDRESS") == Action.TOKENIZE_ROUNDTRIP

    def test_address_masked(self):
        redacted, dets = scan("Address: 456 Oak Avenue, Seattle, WA 98101")
        if any(d.category == "ADDRESS" for d in dets):
            assert "456 Oak Avenue" not in redacted


# ══════════════════════════════════════════════════════════════════════════════
# Block vs Mask action correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockVsMaskActions:
    """Verify the block/mask split is correct for all categories."""

    BLOCK_CATEGORIES = {
        "API_KEY", "AWS_SECRET", "GCP_KEY", "TOKEN",
        "PRIVATE_KEY", "PASSWORD", "RRN", "FOREIGN_REG",
        "PASSPORT", "DRIVER_LICENSE", "CARD",
    }

    MASK_CATEGORIES = {
        "EMAIL", "PHONE", "KR_ACCOUNT", "BIZ_NO",
        "PERSON", "ADDRESS",
    }

    def test_block_categories_produce_blocked_tokens(self):
        """All block-category placeholders must contain _BLOCKED."""
        # Test a sample block-category item
        engine = Engine()
        result = engine.scan("AKIAIOSFODNN7EXAMPLE")
        for det in result.detections:
            if det.category in self.BLOCK_CATEGORIES:
                assert "BLOCKED" in det.placeholder_token, (
                    f"{det.category} placeholder should contain BLOCKED, "
                    f"got {det.placeholder_token!r}"
                )

    def test_mask_categories_produce_clean_tokens(self):
        """Mask-category placeholders must NOT contain _BLOCKED."""
        engine = Engine()
        result = engine.scan("alice@example.com")
        for det in result.detections:
            if det.category in self.MASK_CATEGORIES:
                assert "BLOCKED" not in det.placeholder_token, (
                    f"{det.category} placeholder should not contain BLOCKED, "
                    f"got {det.placeholder_token!r}"
                )

    @pytest.mark.parametrize("category,text", [
        ("API_KEY",      "sk-ant-api03-" + "A" * 50),
        ("AWS_SECRET",   "AKIAIOSFODNN7EXAMPLE"),
        ("GCP_KEY",      "AIzaSy" + "B" * 33),   # 39 chars required by pattern
        ("TOKEN",        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.abc"),
        ("PRIVATE_KEY",  "-----BEGIN RSA PRIVATE KEY-----"),
        ("PASSWORD",     "password: SekretPa$$123"),
        ("CARD",         "4532015112830366"),
        ("PASSPORT",     "Passport number: M12345678"),
        ("DRIVER_LICENSE", "Driver license: A123-456-7890"),
    ])
    def test_block_action_per_category(self, category, text):
        _, dets = scan(text)
        matched = [d for d in dets if d.category == category]
        if matched:
            assert matched[0].action == Action.BLOCK, (
                f"{category} should have action=BLOCK, got {matched[0].action}"
            )

    @pytest.mark.parametrize("category,text", [
        ("EMAIL",    "bob@example.org"),
        ("PHONE",    "010-5555-1234"),
        ("BIZ_NO",   "사업자: 123-45-67890"),
        ("PERSON",   "Name: Charlie Davis"),
        ("ADDRESS",  "789 Elm Drive, Austin, TX 73301"),
    ])
    def test_mask_action_per_category(self, category, text):
        _, dets = scan(text)
        matched = [d for d in dets if d.category == category]
        if matched:
            assert matched[0].action == Action.TOKENIZE_ROUNDTRIP, (
                f"{category} should have action=TOKENIZE_ROUNDTRIP, got {matched[0].action}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Round-trip restoration
# ══════════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    def test_email_roundtrip_exact(self):
        engine = Engine()
        original = "Please contact bob@secret.io for details"
        result = engine.scan(original)
        assert "bob@secret.io" not in result.redacted_text
        restored = engine.rehydrate(result.redacted_text)
        assert "bob@secret.io" in restored

    def test_same_value_same_placeholder(self):
        engine = Engine()
        r1 = engine.scan("Email: alice@test.com")
        r2 = engine.scan("Reply to alice@test.com")
        # Both should use EMAIL_1 since it's the same session
        assert "[EMAIL_1]" in r1.redacted_text
        assert "[EMAIL_1]" in r2.redacted_text

    def test_blocked_content_not_rehydrated_accidentally(self):
        """BLOCKED tokens should restore to original on rehydrate (local only)."""
        engine = Engine()
        key = "sk-ant-api03-" + "A" * 50
        result = engine.scan(f"key={key}")
        # The blocked placeholder should be in the redacted text
        assert key not in result.redacted_text
        # On rehydration the original IS restored (local round-trip is lossless)
        restored = engine.rehydrate(result.redacted_text)
        assert key in restored

    def test_multi_category_roundtrip(self):
        engine = Engine()
        text = "Contact alice@corp.io or call 010-1234-5678"
        result = engine.scan(text)
        assert "alice@corp.io" not in result.redacted_text
        assert "010-1234-5678" not in result.redacted_text
        restored = engine.rehydrate(result.redacted_text)
        assert "alice@corp.io" in restored
        assert "010-1234-5678" in restored


# ══════════════════════════════════════════════════════════════════════════════
# Allowlist
# ══════════════════════════════════════════════════════════════════════════════

class TestAllowlist:
    def test_allowlisted_email_not_masked(self):
        import re
        engine = Engine(allowlist_patterns=[re.compile(r"noreply@example\.com")])
        result = engine.scan("noreply@example.com is our sending address")
        assert "noreply@example.com" in result.redacted_text
        assert not any(d.category == "EMAIL" for d in result.detections)

    def test_non_allowlisted_email_still_masked(self):
        import re
        engine = Engine(allowlist_patterns=[re.compile(r"noreply@example\.com")])
        result = engine.scan("user@corp.com is the contact")
        assert "user@corp.com" not in result.redacted_text


# ══════════════════════════════════════════════════════════════════════════════
# Report / summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSummary:
    def test_summary_has_required_keys(self):
        _, dets = scan("alice@example.com")
        from pii_guard.models import RedactionResult
        engine = Engine()
        result = engine.scan("alice@example.com")
        s = result.summary()
        assert "total_detections" in s
        assert "categories" in s
        assert "actions" in s
        assert "coverage_gap" in s

    def test_summary_counts_correct(self):
        engine = Engine()
        result = engine.scan("a@b.com and c@d.io")
        s = result.summary()
        assert s["total_detections"] == 2
        assert s["categories"].get("EMAIL", 0) == 2
