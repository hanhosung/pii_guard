"""
Tests for Sub-AC 4.1 — Ledger write engine.

Verification matrix
-------------------
(a) All four event types are written — block, mask, fail, coverage_gap —
    and each produces a parseable JSONL entry with the expected ``event_type``
    field and required metadata.

(b) HMAC output is deterministic and non-reversible:
    - Same key + same value → identical keyed_hash on every call.
    - Different key → different keyed_hash (key isolation).
    - keyed_hash is a 64-char hex string, NOT the original value.
    - keyed_hash is NOT the SHA-256 of the original (it's HMAC-keyed).

(c) Raw PII/secrets are absent from persisted entries:
    - The ``original`` field of a Detection is never written to any JSONL
      line, even for long or multi-byte inputs.
    - The placeholder_token (e.g. ``EMAIL_1``) is also not persisted.
    - Fail/coverage-gap entries contain only safe metadata.

(d) File and directory permissions are enforced:
    - Parent directory: stat.S_IMODE == 0o700.
    - Ledger file: stat.S_IMODE == 0o600.
    - Permissions are set on first write even when the directory already
      exists with looser permissions.
    - Permissions are preserved after rotation.

Additional invariants
---------------------
- Rotation: after the file exceeds max_bytes a new file is started and the
  old one is renamed .1; the new file has 0o600 permissions.
- Retention: old files beyond backup_count are deleted at rotation time.
- Purge: explicit purge deletes current file and all rotated backups.
- Thread safety: concurrent writes do not interleave or lose entries.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac_mod
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from pii_guard.ledger import (
    DEFAULT_MAX_AGE_SECONDS,
    DEFAULT_RETENTION_WINDOW_SECONDS,
    Ledger,
    LedgerEventType,
    _DIR_MODE,
    _FILE_MODE,
)
from pii_guard.models import (
    Action,
    CategoryClass,
    Detection,
    DetectionStage,
    MaskStyle,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TEST_KEY = b"test-hmac-key-32bytes-padding-xx"  # 32 bytes


def _make_key(seed: int = 0) -> bytes:
    """Return a deterministic 32-byte HMAC key for tests."""
    return bytes([seed % 256] * 32)


def _make_detection(
    category: str = "EMAIL",
    category_class: CategoryClass = CategoryClass.PII,
    action: Action = Action.BLOCK,
    original: str = "alice@example.com",
    rule_id: str = "email_rfc5322",
    confidence: float = 0.95,
    start: int = 0,
    end: Optional[int] = None,
    placeholder_token: str = "EMAIL_1",
) -> Detection:
    """Construct a Detection for testing."""
    if end is None:
        end = start + len(original)
    return Detection(
        category=category,
        category_class=category_class,
        action=action,
        mask_style=MaskStyle.TOKENIZE,
        start=start,
        end=end,
        original=original,
        detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
        rule_id=rule_id,
        confidence=confidence,
        placeholder_token=placeholder_token,
    )


def _read_entries(path: Path) -> List[dict]:
    """Read all JSONL entries from a ledger file."""
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _file_mode(path: Path) -> int:
    """Return the permission bits (octal) of a file or directory."""
    return stat.S_IMODE(path.stat().st_mode)


# ─────────────────────────────────────────────────────────────────────────────
# (a) All four event types are written
# ─────────────────────────────────────────────────────────────────────────────

class TestAllFourEventTypesWritten:
    """Verify that each record_* method emits a well-formed JSONL entry."""

    def test_record_block_writes_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(action=Action.BLOCK)

        ledger.record_block(det, channel="cli/codex", scan_field="message_text")

        entries = _read_entries(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["event_type"] == "block"

    def test_record_mask_writes_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(action=Action.MASK)

        ledger.record_mask(det, channel="ouroboros", scan_field="tool_result")

        entries = _read_entries(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["event_type"] == "mask"

    def test_record_fail_writes_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_fail(
            "Stage2 NER OOM after 5s",
            category="EMAIL",
            channel="cli/codex",
            scan_field="system_prompt",
        )

        entries = _read_entries(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["event_type"] == "fail"

    def test_record_coverage_gap_writes_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_coverage_gap(
            reason="image/png — unscannable",
            channel="ouroboros",
            scan_field="image",
        )

        entries = _read_entries(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["event_type"] == "coverage_gap"

    def test_all_four_event_types_in_same_ledger(self, tmp_path):
        """All four event types can coexist in the same file."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        det_block = _make_detection(action=Action.BLOCK)
        det_mask = _make_detection(
            category="PHONE",
            action=Action.TOKENIZE_ROUNDTRIP,
            original="010-1234-5678",
            placeholder_token="PHONE_1",
        )

        ledger.record_block(det_block)
        ledger.record_mask(det_mask)
        ledger.record_fail("Stage2 timeout")
        ledger.record_coverage_gap(reason="binary content")

        entries = _read_entries(path)
        assert len(entries) == 4
        types_seen = {e["event_type"] for e in entries}
        assert types_seen == {"block", "mask", "fail", "coverage_gap"}

    # ── Required fields per event type ──────────────────────────────────────

    def test_block_entry_has_required_metadata_fields(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(
            category="AWS_SECRET",
            category_class=CategoryClass.SECRET,
            action=Action.BLOCK,
            original="AKIAIOSFODNN7EXAMPLE",
            rule_id="aws_akid",
            confidence=0.98,
        )
        ledger.record_block(det, channel="cli/codex", scan_field="message_text")

        e = _read_entries(path)[0]
        assert e["event_type"] == "block"
        assert e["category"] == "AWS_SECRET"
        assert e["action"] == "block"
        assert e["detector_id"] == "aws_akid"
        assert e["rule"] == "aws_akid"
        assert e["confidence"] == pytest.approx(0.98)
        assert e["span_length"] == len("AKIAIOSFODNN7EXAMPLE")
        assert "char_class_signature" in e
        assert "keyed_hash" in e
        assert "timestamp" in e
        assert e["channel"] == "cli/codex"
        assert e["scan_field"] == "message_text"

    def test_mask_entry_has_required_metadata_fields(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(
            category="EMAIL",
            category_class=CategoryClass.PII,
            action=Action.TOKENIZE_ROUNDTRIP,
            original="bob@corp.io",
            rule_id="email_rfc5322",
            confidence=0.95,
        )
        ledger.record_mask(det, channel="ouroboros", scan_field="tool_use_input")

        e = _read_entries(path)[0]
        assert e["event_type"] == "mask"
        assert e["category"] == "EMAIL"
        assert e["action"] == "tokenize_roundtrip"
        assert e["confidence"] == pytest.approx(0.95)
        assert e["span_length"] == len("bob@corp.io")
        assert "keyed_hash" in e
        assert "timestamp" in e

    def test_fail_entry_has_required_metadata_fields(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_fail(
            "RuntimeError: scanner unavailable",
            category="PHONE",
            channel="cli/codex",
            scan_field="system_prompt",
        )

        e = _read_entries(path)[0]
        assert e["event_type"] == "fail"
        assert e["fail_reason"] == "RuntimeError: scanner unavailable"
        assert e["category"] == "PHONE"
        assert e["channel"] == "cli/codex"
        assert e["scan_field"] == "system_prompt"
        assert "timestamp" in e

    def test_coverage_gap_entry_has_required_metadata_fields(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_coverage_gap(
            reason="Stage2 NER timeout after 3s",
            channel="ouroboros",
            scan_field="document_block",
        )

        e = _read_entries(path)[0]
        assert e["event_type"] == "coverage_gap"
        assert "timestamp" in e
        assert e["channel"] == "ouroboros"
        assert e["scan_field"] == "document_block"

    def test_timestamp_is_iso8601_utc(self, tmp_path):
        """Timestamps must be ISO-8601 UTC strings ending in Z."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("test")
        e = _read_entries(path)[0]
        ts = e["timestamp"]
        assert isinstance(ts, str)
        assert ts.endswith("Z"), f"Timestamp must end with Z: {ts!r}"
        # Must be parseable as ISO 8601
        import datetime
        # Strip trailing Z and parse
        datetime.datetime.fromisoformat(ts.rstrip("Z"))

    def test_ledger_event_type_enum_values(self):
        """LedgerEventType enum covers all four required values."""
        assert LedgerEventType.BLOCK.value == "block"
        assert LedgerEventType.MASK.value == "mask"
        assert LedgerEventType.FAIL.value == "fail"
        assert LedgerEventType.COVERAGE_GAP.value == "coverage_gap"


# ─────────────────────────────────────────────────────────────────────────────
# (b) HMAC output is deterministic and non-reversible
# ─────────────────────────────────────────────────────────────────────────────

class TestHMACDeterministicAndNonReversible:
    """HMAC-keyed hashes must be deterministic (same key+value→same hash) and
    non-reversible (hash does not expose original, different key→different hash)."""

    def test_same_key_same_value_produces_same_hash(self, tmp_path):
        """Two ledgers sharing the same key produce identical keyed_hash for
        the same detection original."""
        key = _make_key(1)
        path1 = tmp_path / "l1.jsonl"
        path2 = tmp_path / "l2.jsonl"
        ledger1 = Ledger(path1, key)
        ledger2 = Ledger(path2, key)

        det = _make_detection(original="alice@example.com")
        ledger1.record_block(det)
        ledger2.record_block(det)

        h1 = _read_entries(path1)[0]["keyed_hash"]
        h2 = _read_entries(path2)[0]["keyed_hash"]
        assert h1 == h2, "Same key + same original → identical keyed_hash"

    def test_same_key_different_values_produce_different_hashes(self, tmp_path):
        key = _make_key(2)
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, key)

        det1 = _make_detection(original="alice@example.com")
        det2 = _make_detection(original="bob@corp.io")
        ledger.record_block(det1)
        ledger.record_block(det2)

        entries = _read_entries(path)
        h1 = entries[0]["keyed_hash"]
        h2 = entries[1]["keyed_hash"]
        assert h1 != h2, "Different originals → different keyed_hash"

    def test_different_key_same_value_produces_different_hash(self, tmp_path):
        """Different HMAC keys produce different hashes for the same value."""
        key1 = _make_key(10)
        key2 = _make_key(20)
        path1 = tmp_path / "l1.jsonl"
        path2 = tmp_path / "l2.jsonl"
        ledger1 = Ledger(path1, key1)
        ledger2 = Ledger(path2, key2)

        original = "alice@example.com"
        det = _make_detection(original=original)
        ledger1.record_block(det)
        ledger2.record_block(det)

        h1 = _read_entries(path1)[0]["keyed_hash"]
        h2 = _read_entries(path2)[0]["keyed_hash"]
        assert h1 != h2, "Different keys → different keyed_hash for same value"

    def test_keyed_hash_is_64_char_hex_string(self, tmp_path):
        """HMAC-SHA256 hex digest is always 64 hex characters."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(original="test@example.com")
        ledger.record_block(det)

        h = _read_entries(path)[0]["keyed_hash"]
        assert isinstance(h, str)
        assert len(h) == 64, f"Expected 64-char hex, got len={len(h)}"
        assert all(c in "0123456789abcdef" for c in h), "keyed_hash must be hex"

    def test_keyed_hash_is_not_original_value(self, tmp_path):
        """The keyed_hash must not equal the original value."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        original = "alice@example.com"
        det = _make_detection(original=original)
        ledger.record_block(det)

        h = _read_entries(path)[0]["keyed_hash"]
        assert h != original, "keyed_hash must not equal original value"

    def test_keyed_hash_is_not_plain_sha256(self, tmp_path):
        """Verify it's a KEYED hash (HMAC), not a plain SHA-256 of the value."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        original = "alice@example.com"
        det = _make_detection(original=original)
        ledger.record_block(det)

        h = _read_entries(path)[0]["keyed_hash"]
        plain_sha = hashlib.sha256(original.strip().lower().encode()).hexdigest()
        assert h != plain_sha, (
            "keyed_hash must be HMAC-SHA256 (keyed), not plain SHA-256"
        )

    def test_keyed_hash_matches_expected_hmac(self, tmp_path):
        """keyed_hash must match manually computed HMAC-SHA256 with same key."""
        key = _make_key(42)
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, key)
        original = "sk-ant-test-key-12345"
        det = _make_detection(original=original, action=Action.BLOCK)
        ledger.record_block(det)

        h = _read_entries(path)[0]["keyed_hash"]

        # Compute expected manually
        normalised = original.strip().lower()
        expected = _hmac_mod.new(key, normalised.encode("utf-8"), hashlib.sha256).hexdigest()
        assert h == expected

    def test_normalisation_lowercase_strip_applied(self, tmp_path):
        """HMAC normalises original: strip + lower, so case/whitespace variants
        produce the same hash."""
        key = _make_key(5)
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, key)

        det_upper = _make_detection(original="  Alice@Example.COM  ")
        det_lower = _make_detection(original="alice@example.com")
        ledger.record_block(det_upper)
        ledger.record_block(det_lower)

        entries = _read_entries(path)
        assert entries[0]["keyed_hash"] == entries[1]["keyed_hash"], (
            "Normalised originals (strip+lower) must produce identical keyed_hash"
        )

    def test_repeated_writes_produce_same_hash(self, tmp_path):
        """Writing the same detection multiple times always yields same hash."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(original="secret@corp.io")

        for _ in range(5):
            ledger.record_block(det)

        entries = _read_entries(path)
        hashes = [e["keyed_hash"] for e in entries]
        assert len(set(hashes)) == 1, "Hash must be deterministic across repeated writes"


# ─────────────────────────────────────────────────────────────────────────────
# (c) Raw PII/secrets absent from persisted entries
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRawPIIInLedger:
    """The ledger file must never contain recoverable original PII/secrets."""

    def _raw_file_content(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_block_event_does_not_persist_original_email(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(original="alice@sensitive.com", action=Action.BLOCK)
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert "alice@sensitive.com" not in content, (
            "Raw email must not appear in ledger file"
        )

    def test_mask_event_does_not_persist_original_phone(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(
            category="PHONE",
            original="010-9876-5432",
            action=Action.TOKENIZE_ROUNDTRIP,
        )
        ledger.record_mask(det)

        content = self._raw_file_content(path)
        assert "010-9876-5432" not in content, (
            "Raw phone number must not appear in ledger file"
        )

    def test_block_event_does_not_persist_api_key(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        api_key = "sk-ant-api03-AABBCCDDEE1122334455667788990011AABBCCDDEE1122334455"
        det = _make_detection(
            category="API_KEY",
            category_class=CategoryClass.SECRET,
            action=Action.BLOCK,
            original=api_key,
            rule_id="apikey_anthropic",
        )
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert api_key not in content, "Raw API key must not appear in ledger file"

    def test_block_event_does_not_persist_aws_secret(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        det = _make_detection(
            category="AWS_SECRET",
            category_class=CategoryClass.SECRET,
            action=Action.BLOCK,
            original=aws_secret,
            rule_id="aws_secret_key",
        )
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert aws_secret not in content, "Raw AWS secret must not appear in ledger"

    def test_block_event_does_not_persist_private_key_pem(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        pem_header = "-----BEGIN RSA PRIVATE KEY-----"
        det = _make_detection(
            category="PRIVATE_KEY",
            category_class=CategoryClass.SECRET,
            action=Action.BLOCK,
            original=pem_header,
            rule_id="privkey_pem",
        )
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert pem_header not in content, "Raw PEM header must not appear in ledger"

    def test_block_event_does_not_persist_rrn(self, tmp_path):
        """Korean RRN (high-risk PII) must not appear in ledger."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        rrn = "900101-1234567"
        det = _make_detection(
            category="RRN",
            category_class=CategoryClass.KOREAN_PII,
            action=Action.BLOCK,
            original=rrn,
            rule_id="rrn_kr",
        )
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert rrn not in content, "Raw RRN must not appear in ledger"

    def test_block_event_does_not_persist_card_number(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        card = "4111-1111-1111-1111"
        det = _make_detection(
            category="CARD",
            action=Action.BLOCK,
            original=card,
            rule_id="card_pan",
        )
        ledger.record_block(det)

        content = self._raw_file_content(path)
        assert card not in content
        # Also check without hyphens
        assert "4111111111111111" not in content

    def test_block_event_does_not_persist_placeholder_token(self, tmp_path):
        """placeholder_token (e.g. EMAIL_1) must not appear in the ledger."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(
            original="alice@example.com",
            placeholder_token="EMAIL_1",
            action=Action.BLOCK,
        )
        ledger.record_block(det)

        # Parse entries to check the JSON structure
        entries = _read_entries(path)
        e = entries[0]
        # placeholder_token must not appear as a value
        assert "placeholder_token" not in e, (
            "placeholder_token field must not be persisted"
        )
        assert "EMAIL_1" not in e.get("keyed_hash", ""), (
            "Placeholder token must not be in the hash field"
        )

    def test_fail_event_contains_only_safe_metadata(self, tmp_path):
        """Fail events must not contain raw PII — only error type/timing."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        # Fail reason may contain exception class but not PII
        ledger.record_fail(
            "TimeoutError: NER stage2 timed out after 5s",
            category="EMAIL",
        )

        entries = _read_entries(path)
        e = entries[0]
        assert "keyed_hash" not in e or e.get("keyed_hash") is None, (
            "fail events have no keyed_hash (no original to hash)"
        )
        assert "original" not in e
        assert "placeholder_token" not in e

    def test_coverage_gap_event_contains_only_safe_metadata(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_coverage_gap(reason="image/png — unscannable block")

        entries = _read_entries(path)
        e = entries[0]
        assert "keyed_hash" not in e or e.get("keyed_hash") is None
        assert "original" not in e

    def test_no_raw_pii_in_any_entry_after_mixed_events(self, tmp_path):
        """Full-file scan: no raw PII literal survives in any event type."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        sensitive_values = [
            "alice@corp.io",
            "010-1234-5678",
            "sk-ant-api03-AAABBBCCCDDDEEEFFFGGG",
            "4111-1111-1111-1111",
        ]

        dets = [_make_detection(original=v) for v in sensitive_values]
        for det in dets:
            ledger.record_block(det)
        ledger.record_fail("RuntimeError: crash", category="EMAIL")
        ledger.record_coverage_gap(reason="stage2 degraded")

        content = self._raw_file_content(path)
        for v in sensitive_values:
            assert v not in content, (
                f"Raw sensitive value {v!r} found in ledger file"
            )


# ─────────────────────────────────────────────────────────────────────────────
# (d) File and directory permissions are enforced
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionsEnforced:
    """Ledger file: 0o600; parent directory: 0o700.  Enforced on first write."""

    def test_directory_created_with_700_permissions(self, tmp_path):
        nested = tmp_path / "piiguard" / "audit"
        path = nested / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("test event")

        assert nested.exists()
        assert _file_mode(nested) == _DIR_MODE, (
            f"Directory must have mode 0o700, got 0o{_file_mode(nested):o}"
        )

    def test_ledger_file_created_with_600_permissions(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("test")

        assert path.exists()
        assert _file_mode(path) == _FILE_MODE, (
            f"File must have mode 0o600, got 0o{_file_mode(path):o}"
        )

    def test_dir_mode_constant_is_700(self):
        assert _DIR_MODE == 0o700

    def test_file_mode_constant_is_600(self):
        assert _FILE_MODE == 0o600

    def test_permissions_enforced_even_when_dir_exists_with_looser_perms(self, tmp_path):
        """If the parent directory already exists with mode 0o755, the Ledger
        must tighten it to 0o700 on first write."""
        parent = tmp_path / "piiguard"
        parent.mkdir(mode=0o755)
        # Confirm initial mode is 755
        os.chmod(parent, 0o755)
        assert _file_mode(parent) == 0o755

        path = parent / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("test")

        assert _file_mode(parent) == 0o700, (
            "Ledger must tighten directory permissions to 0o700"
        )

    def test_file_permissions_enforced_when_file_exists_with_looser_perms(self, tmp_path):
        """If the ledger file already exists with mode 0o644, the Ledger must
        tighten it to 0o600 on first write."""
        path = tmp_path / "ledger.jsonl"
        # Pre-create with wrong permissions
        path.touch()
        os.chmod(path, 0o644)
        assert _file_mode(path) == 0o644

        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("test")

        assert _file_mode(path) == 0o600, (
            "Ledger must tighten file permissions to 0o600"
        )

    def test_permissions_preserved_after_multiple_writes(self, tmp_path):
        """Multiple writes must not change the file permissions."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        for i in range(5):
            ledger.record_fail(f"error {i}")

        assert _file_mode(path) == 0o600

    def test_new_file_after_rotation_has_600_permissions(self, tmp_path):
        """After rotation, the new current ledger file must have 0o600 perms."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1)  # tiny limit → immediate rotation

        ledger.record_fail("first entry")  # triggers rotation on second write
        ledger.record_fail("second entry")

        # The current file (after rotation) must have 0o600
        assert _file_mode(path) == 0o600

    def test_intermediate_parent_dirs_exist(self, tmp_path):
        """Deeply nested parent directories are all created."""
        path = tmp_path / "a" / "b" / "c" / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("deep nesting test")

        assert path.exists()
        # Direct parent must have 0o700
        assert _file_mode(path.parent) == 0o700


# ─────────────────────────────────────────────────────────────────────────────
# Rotation and retention
# ─────────────────────────────────────────────────────────────────────────────

class TestRotationAndRetention:
    """Rotation, backup numbering, and retention (backup_count cap)."""

    def test_rotation_creates_numbered_backup(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        # max_bytes=1 → any write of 2+ chars triggers rotation
        ledger = Ledger(path, _TEST_KEY, max_bytes=1)

        ledger.record_fail("first")   # written to .jsonl
        ledger.record_fail("second")  # triggers rotation → first goes to .jsonl.1

        assert Path(f"{path}.1").exists(), "Rotated file .1 must exist"

    def test_rotated_backup_has_600_permissions(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1)
        ledger.record_fail("first")
        ledger.record_fail("second")  # rotation happens here

        backup = Path(f"{path}.1")
        assert backup.exists()
        # Backup retains the permissions it had when it was the current file
        assert _file_mode(backup) == 0o600

    def test_multiple_rotations_number_correctly(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=5)

        # Write enough to trigger multiple rotations
        for i in range(6):
            ledger.record_fail(f"entry {i}")

        # After 5 rotations we should have .1 through .5 plus the current
        assert Path(f"{path}.1").exists()
        assert Path(f"{path}.2").exists()

    def test_backup_count_limits_retained_files(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=2)

        # Write enough to rotate more than backup_count times
        for i in range(6):
            ledger.record_fail(f"entry {i}")

        # .3 and beyond must be deleted
        assert not Path(f"{path}.3").exists(), (
            "Files beyond backup_count must be deleted"
        )

    def test_manual_rotate_method(self, tmp_path):
        """ledger.rotate() forces an immediate rotation."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_fail("before rotation")
        ledger.rotate()

        assert Path(f"{path}.1").exists()
        assert path.exists()  # new current file created
        assert _file_mode(path) == 0o600

    def test_new_entries_go_to_new_file_after_rotation(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_fail("entry before rotation")
        ledger.rotate()
        ledger.record_fail("entry after rotation")

        # Current file has the post-rotation entry
        current_entries = _read_entries(path)
        assert len(current_entries) == 1
        assert current_entries[0]["fail_reason"] == "entry after rotation"

        # Backup has the pre-rotation entry
        backup_entries = _read_entries(Path(f"{path}.1"))
        assert len(backup_entries) == 1
        assert backup_entries[0]["fail_reason"] == "entry before rotation"


# ─────────────────────────────────────────────────────────────────────────────
# Purge
# ─────────────────────────────────────────────────────────────────────────────

class TestPurge:
    """Explicit purge deletes current file and all rotated backups."""

    def test_purge_deletes_current_file(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("some event")

        assert path.exists()
        ledger.purge()
        assert not path.exists()

    def test_purge_deletes_rotated_backups(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=3)

        for i in range(5):
            ledger.record_fail(f"entry {i}")

        ledger.purge()

        assert not path.exists()
        for i in range(1, 5):
            assert not Path(f"{path}.{i}").exists(), (
                f"Backup .{i} must be deleted by purge"
            )

    def test_after_purge_new_write_recreates_file(self, tmp_path):
        """After purge, writing a new event recreates the file with correct perms."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("before purge")
        ledger.purge()

        assert not path.exists()

        # New write should recreate
        ledger.record_fail("after purge")
        assert path.exists()
        assert _file_mode(path) == 0o600

    def test_purge_idempotent_on_missing_file(self, tmp_path):
        """Purge on a never-written ledger must not raise."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        # No writes yet
        ledger.purge()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_hmac_key_must_be_bytes(self):
        with pytest.raises(TypeError):
            Ledger("/tmp/l.jsonl", "string-key-is-wrong")  # type: ignore[arg-type]

    def test_hmac_key_minimum_length(self):
        with pytest.raises(ValueError):
            Ledger("/tmp/l.jsonl", b"short")  # < 16 bytes

    def test_hmac_key_exactly_16_bytes_accepted(self, tmp_path):
        path = tmp_path / "l.jsonl"
        ledger = Ledger(path, b"x" * 16)  # exactly 16 bytes — OK
        ledger.record_fail("test")  # should not raise

    def test_path_is_pathlib_path(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        assert isinstance(ledger.path, Path)

    def test_path_exposed_via_property(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        assert ledger.path == path


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:
    """Concurrent writes must not corrupt the ledger."""

    def test_concurrent_writes_all_persisted(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        N = 50

        errors = []

        def write_events():
            try:
                for _ in range(10):
                    ledger.record_fail("concurrent event")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_events) for _ in range(N // 10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"

        entries = _read_entries(path)
        assert len(entries) == N, (
            f"Expected {N} entries after concurrent writes, got {len(entries)}"
        )

    def test_all_jsonl_lines_are_valid_after_concurrent_writes(self, tmp_path):
        """No partial writes: every line must be valid JSON."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        N_THREADS = 20
        N_PER_THREAD = 5

        def writer():
            for _ in range(N_PER_THREAD):
                ledger.record_fail("thread write")

        threads = [threading.Thread(target=writer) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    json.loads(line)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: empty originals, optional fields, unicode."""

    def test_optional_channel_and_scan_field_default_to_none(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection()
        ledger.record_block(det)  # no channel/scan_field

        e = _read_entries(path)[0]
        assert e["channel"] is None
        assert e["scan_field"] is None

    def test_unicode_in_fail_reason_is_safe(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("에러: NER 타임아웃 5초")

        entries = _read_entries(path)
        assert len(entries) == 1
        assert "에러" in entries[0]["fail_reason"]

    def test_detection_span_length_recorded_correctly(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        original = "test@example.com"  # 16 chars
        det = _make_detection(original=original, start=5, end=5 + len(original))
        ledger.record_block(det)

        e = _read_entries(path)[0]
        assert e["span_length"] == len(original)

    def test_char_class_signature_recorded_and_safe(self, tmp_path):
        """char_class_signature collapses runs of U/l/d/s — not recoverable."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(original="AKIAIOSFODNN7EXAMPLE")
        ledger.record_block(det)

        e = _read_entries(path)[0]
        sig = e["char_class_signature"]
        assert isinstance(sig, str)
        assert len(sig) > 0
        # Must not contain the original value
        assert "AKIAIOSFODNN7EXAMPLE" not in sig

    def test_coverage_gap_without_reason(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_coverage_gap()  # no reason

        e = _read_entries(path)[0]
        assert e["event_type"] == "coverage_gap"
        assert e.get("reason") is None

    def test_fail_without_category(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("generic infra failure")

        e = _read_entries(path)[0]
        assert e["event_type"] == "fail"
        assert e.get("category") is None

    def test_entries_are_valid_json_objects(self, tmp_path):
        """Every line in the ledger is a valid JSON object (dict)."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        det = _make_detection(action=Action.BLOCK)
        ledger.record_block(det)
        ledger.record_mask(_make_detection(action=Action.MASK))
        ledger.record_fail("infra error")
        ledger.record_coverage_gap()

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    assert isinstance(obj, dict)

    def test_tokenize_roundtrip_action_recorded_as_mask_event(self, tmp_path):
        """TOKENIZE_ROUNDTRIP action → record_mask → event_type == 'mask'."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection(action=Action.TOKENIZE_ROUNDTRIP)
        ledger.record_mask(det)

        e = _read_entries(path)[0]
        assert e["event_type"] == "mask"
        assert e["action"] == "tokenize_roundtrip"

    def test_detection_stage_recorded(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        det = _make_detection()
        ledger.record_block(det)

        e = _read_entries(path)[0]
        assert e["detection_stage"] == "stage1_regex_checksum"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-AC 4.2.1 — Time-based rotation
# ─────────────────────────────────────────────────────────────────────────────

class TestTimedRotation:
    """
    Verify that the time-based rotation trigger works correctly.

    Sub-AC 4.2.1 verification matrix
    ---------------------------------
    (a) Rotation produces a correctly named archive file (``.1``) AND
        initialises a fresh active ledger (empty / only post-rotation entries).
    (d) The archive (``.1``) and the fresh active ledger are both created with
        0o600 permissions; the parent directory retains 0o700 permissions.
    """

    # ── (a) Correctly named archive + fresh active ledger ───────────────────

    def test_time_rotation_triggers_after_max_age(self, tmp_path):
        """
        Writing past the max_age_seconds threshold must trigger rotation and
        produce a ``.1`` archive file.
        """
        path = tmp_path / "ledger.jsonl"
        # Use a tiny age limit so the test doesn't have to wait long.
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("before rotation")   # creates the file; starts timer
        time.sleep(0.10)                         # exceed 50 ms threshold
        ledger.record_fail("triggers rotation")  # should rotate before writing

        archive = Path(f"{path}.1")
        assert archive.exists(), (
            "Archive ledger.jsonl.1 must exist after time-based rotation"
        )

    def test_time_rotation_archive_is_named_with_dot_one_suffix(self, tmp_path):
        """
        The archive produced by time-based rotation must be named
        ``<active-ledger>.1`` — the standard ``.N`` numbering scheme.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("entry 1")
        time.sleep(0.10)
        ledger.record_fail("entry 2")  # triggers rotation

        archive = Path(f"{path}.1")
        assert archive.exists(), "Archive must use .1 suffix"
        # Name must end with the base name plus ".1"
        assert archive.name == "ledger.jsonl.1"

    def test_time_rotation_initialises_fresh_active_ledger(self, tmp_path):
        """
        After time-based rotation the active ledger is empty initially;
        the entry written at rotation time goes into the NEW file, not the
        archive.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("pre-rotation entry")
        time.sleep(0.10)
        ledger.record_fail("post-rotation entry")  # written to fresh file

        # Fresh active file must contain only the post-rotation entry
        current_entries = _read_entries(path)
        assert len(current_entries) == 1, (
            "Fresh active ledger must contain only the entry written after rotation"
        )
        assert current_entries[0]["fail_reason"] == "post-rotation entry"

    def test_time_rotation_archive_contains_pre_rotation_entries(self, tmp_path):
        """
        The archive (``.1``) must contain all entries that were written before
        the rotation was triggered.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("pre-rotation entry A")
        ledger.record_fail("pre-rotation entry B")
        time.sleep(0.10)
        ledger.record_fail("post-rotation entry")

        archive_entries = _read_entries(Path(f"{path}.1"))
        reasons = [e["fail_reason"] for e in archive_entries]
        assert "pre-rotation entry A" in reasons
        assert "pre-rotation entry B" in reasons
        assert "post-rotation entry" not in reasons

    # ── (d) Permissions: archive and fresh ledger are 600; directory 700 ────

    def test_time_rotation_archive_has_600_permissions(self, tmp_path):
        """
        The archive file (``.1``) created by time-based rotation must have
        0o600 permissions (owner read/write only).
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("entry before rotation")
        time.sleep(0.10)
        ledger.record_fail("entry after rotation")  # triggers rotation

        archive = Path(f"{path}.1")
        assert archive.exists()
        assert _file_mode(archive) == _FILE_MODE, (
            f"Archive must have mode 0o600, got 0o{_file_mode(archive):o}"
        )

    def test_time_rotation_fresh_ledger_has_600_permissions(self, tmp_path):
        """
        The fresh active ledger created after time-based rotation must have
        0o600 permissions.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("first entry")
        time.sleep(0.10)
        ledger.record_fail("second entry")  # triggers rotation

        assert _file_mode(path) == _FILE_MODE, (
            f"Fresh active ledger must have mode 0o600, got 0o{_file_mode(path):o}"
        )

    def test_time_rotation_parent_dir_retains_700_permissions(self, tmp_path):
        """
        The parent directory must retain 0o700 permissions after a
        time-based rotation.
        """
        parent = tmp_path / "piiguard"
        path = parent / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("first entry")
        time.sleep(0.10)
        ledger.record_fail("triggers rotation")

        assert _file_mode(parent) == _DIR_MODE, (
            f"Parent directory must have mode 0o700 after rotation, "
            f"got 0o{_file_mode(parent):o}"
        )

    # ── Timer-reset: fresh file starts a new age budget ─────────────────────

    def test_fresh_ledger_timer_resets_after_time_rotation(self, tmp_path):
        """
        After time-based rotation a second write within the same age window
        must NOT trigger a second rotation.  The timer must reset when the
        new file is created.
        """
        path = tmp_path / "ledger.jsonl"
        # 50 ms threshold; we sleep 60 ms once then write two more entries
        # without sleeping again.
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05)

        ledger.record_fail("first entry")
        time.sleep(0.10)
        ledger.record_fail("triggers rotation")  # rotation here → timer reset
        ledger.record_fail("third entry, no rotation")  # must NOT rotate again

        # .2 must not exist (only one rotation happened)
        assert not Path(f"{path}.2").exists(), (
            "A second rotation must not happen immediately after the first"
        )
        # Current file must now have two entries (post-rotation ones)
        current_entries = _read_entries(path)
        assert len(current_entries) == 2

    # ── Default value: time rotation disabled by default ────────────────────

    def test_default_max_age_seconds_is_zero_disabled(self):
        """DEFAULT_MAX_AGE_SECONDS must be 0.0 so time rotation is opt-in."""
        assert DEFAULT_MAX_AGE_SECONDS == 0.0

    def test_time_rotation_disabled_by_default(self, tmp_path):
        """
        Without setting max_age_seconds the time-based trigger must never
        fire, even when the writes span an artificially long period.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)  # default: time rotation OFF

        ledger.record_fail("entry 1")
        time.sleep(0.10)
        ledger.record_fail("entry 2")

        # No rotation should have occurred
        assert not Path(f"{path}.1").exists(), (
            "No archive must be created when max_age_seconds is 0 (disabled)"
        )
        # Both entries in the same file
        assert len(_read_entries(path)) == 2

    # ── Both triggers active simultaneously ─────────────────────────────────

    def test_size_and_time_triggers_coexist(self, tmp_path):
        """
        When both max_bytes and max_age_seconds are set, whichever fires first
        produces a rotation.  Confirm no crash and correct archive exists.
        """
        path = tmp_path / "ledger.jsonl"
        # max_bytes=1 will trigger size rotation before time threshold
        ledger = Ledger(
            path, _TEST_KEY,
            max_bytes=1,
            max_age_seconds=9999.0,  # won't fire; size fires first
        )

        ledger.record_fail("first")
        ledger.record_fail("second")  # size rotation fires here

        archive = Path(f"{path}.1")
        assert archive.exists()
        assert _file_mode(archive) == _FILE_MODE
        assert _file_mode(path) == _FILE_MODE

    # ── Mtime-based seeding on re-open ───────────────────────────────────────

    def test_existing_old_file_rotated_on_first_write(self, tmp_path):
        """
        When an existing ledger file is older than max_age_seconds the very
        first write on the re-opened Ledger must trigger a rotation (the mtime
        is used to seed the age timer).
        """
        path = tmp_path / "ledger.jsonl"

        # Create a file whose mtime we back-date by 1 second
        path.write_text('{"event_type":"fail","timestamp":"2000-01-01T00:00:00Z"}\n')
        os.chmod(path, _FILE_MODE)

        # Back-date the mtime so the file appears 1 second old
        old_time = time.time() - 1.0
        os.utime(path, (old_time, old_time))

        # max_age_seconds=0.5 → the file is already 1 s old → rotate on write
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.5)
        ledger.record_fail("new entry after re-open")

        archive = Path(f"{path}.1")
        assert archive.exists(), (
            "Old pre-existing file must be archived on first write "
            "when mtime exceeds max_age_seconds"
        )
        assert _file_mode(archive) == _FILE_MODE
        assert _file_mode(path) == _FILE_MODE

    # ── Multiple time-based rotations number correctly ───────────────────────

    def test_multiple_time_rotations_number_correctly(self, tmp_path):
        """
        Each time-triggered rotation shifts the existing archives up by one.
        After N rotations the archive numbers must be .1, .2, …, .N.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05, backup_count=5)

        # Three rotation cycles
        for cycle in range(3):
            ledger.record_fail(f"cycle {cycle}")
            time.sleep(0.10)

        ledger.record_fail("final entry triggers third rotation")

        assert Path(f"{path}.1").exists(), "Archive .1 must exist"
        assert Path(f"{path}.2").exists(), "Archive .2 must exist"
        assert Path(f"{path}.3").exists(), "Archive .3 must exist"

    def test_multiple_time_rotation_archives_have_600_permissions(self, tmp_path):
        """
        All numbered archive files created by repeated time-based rotation
        must have 0o600 permissions.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_age_seconds=0.05, backup_count=5)

        for cycle in range(3):
            ledger.record_fail(f"cycle {cycle}")
            time.sleep(0.10)

        ledger.record_fail("third rotation trigger")

        for n in range(1, 4):
            archive = Path(f"{path}.{n}")
            assert archive.exists(), f"Archive .{n} must exist"
            assert _file_mode(archive) == _FILE_MODE, (
                f"Archive .{n} must have mode 0o600, got 0o{_file_mode(archive):o}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-AC 4.2.2 — Retention enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestRetentionEnforcement:
    """
    Verify automatic discovery and removal of archived ledger files whose age
    exceeds the configured retention window.

    Verification matrix (Sub-AC 4.2.2)
    ------------------------------------
    (b) Only files **strictly beyond** the window are deleted; archives whose
        mtime is within (or exactly at) the boundary are untouched.
    (d) Any replacement files created during cleanup (i.e. the fresh active
        ledger produced by the rotation that triggers the sweep) carry 0o600
        permissions; the parent directory retains 0o700.
    """

    # ── helper ──────────────────────────────────────────────────────────────

    def _backdate(self, path: Path, age_seconds: float) -> None:
        """Set the file's atime and mtime to `age_seconds` seconds in the past."""
        t = time.time() - age_seconds
        os.utime(path, (t, t))

    # ── Default disabled ────────────────────────────────────────────────────

    def test_default_retention_window_is_zero_disabled(self):
        """DEFAULT_RETENTION_WINDOW_SECONDS must be 0.0 (opt-in, not default-on)."""
        assert DEFAULT_RETENTION_WINDOW_SECONDS == 0.0

    def test_retention_disabled_by_default_does_not_delete_old_archives(
        self, tmp_path
    ):
        """When retention_window_seconds is 0.0 (default), no archives are deleted
        regardless of their age."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=5)
        # No retention_window_seconds → disabled

        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # rotation → .1 created

        archive = Path(f"{path}.1")
        assert archive.exists()
        self._backdate(archive, 99_999.0)  # very old

        deleted = ledger.apply_retention()

        assert deleted == 0, "Disabled retention must not delete any archive"
        assert archive.exists(), "Archive must survive when retention is disabled"

    # ── (b) Only strictly-beyond-window files deleted ────────────────────────

    def test_expired_archive_is_deleted(self, tmp_path):
        """An archive whose mtime is strictly older than the window is removed."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=1.0,
        )

        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # rotation → .1 created

        archive = Path(f"{path}.1")
        assert archive.exists()
        # Back-date to 2 s ago — strictly beyond the 1 s window
        self._backdate(archive, 2.0)

        deleted = ledger.apply_retention()

        assert deleted == 1, "One expired archive must be reported as deleted"
        assert not archive.exists(), "Expired archive must be removed"

    def test_fresh_archive_within_window_is_kept(self, tmp_path):
        """An archive whose mtime is within the retention window must not be deleted."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=3600.0,  # 1-hour window
        )

        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # rotation → .1 created (mtime now)

        archive = Path(f"{path}.1")
        assert archive.exists()
        # Do NOT back-date — archive is just-created, well within the window

        deleted = ledger.apply_retention()

        assert deleted == 0, "Fresh archive must not be deleted"
        assert archive.exists(), "Fresh archive must survive retention sweep"

    def test_archive_exactly_at_boundary_is_not_deleted(self, tmp_path):
        """
        An archive whose mtime equals exactly ``time.time() - window`` must NOT
        be deleted — the semantics are **strictly beyond**, not at-or-beyond.
        """
        path = tmp_path / "ledger.jsonl"
        window = 1.0
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=window,
        )

        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")

        archive = Path(f"{path}.1")
        assert archive.exists()

        # Set mtime to exactly now — boundary, not expired
        t = time.time()
        os.utime(archive, (t, t))

        deleted = ledger.apply_retention()

        assert deleted == 0, (
            "Archive exactly at boundary must not be deleted "
            "(strictly-beyond semantics)"
        )
        assert archive.exists()

    def test_only_expired_archives_deleted_leaves_fresh_ones(self, tmp_path):
        """
        (b) Mixed-age archives: only those strictly beyond the window are removed;
        in-window archives are untouched.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=10,
            retention_window_seconds=1.0,
        )

        # Produce 4 archives (.1 through .4) by writing 5 entries
        for i in range(5):
            ledger.record_fail(f"entry {i}")

        a1 = Path(f"{path}.1")
        a2 = Path(f"{path}.2")
        a3 = Path(f"{path}.3")
        a4 = Path(f"{path}.4")

        assert all(p.exists() for p in (a1, a2, a3, a4)), (
            "All four archives must exist before retention sweep"
        )

        # Keep .1 and .2 within window; expire .3 and .4
        # (a1 is the newest; a4 is the oldest)
        self._backdate(a3, 2.0)
        self._backdate(a4, 3.0)
        # a1 and a2 are fresh (within the 1 s window)

        deleted = ledger.apply_retention()

        assert deleted == 2, "Exactly two expired archives must be deleted"
        assert a1.exists(), "Fresh archive .1 must survive"
        assert a2.exists(), "Fresh archive .2 must survive"
        assert not a3.exists(), "Expired archive .3 must be deleted"
        assert not a4.exists(), "Expired archive .4 must be deleted"

    def test_apply_retention_returns_deleted_count(self, tmp_path):
        """apply_retention() must return the exact number of deleted files."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=10,
            retention_window_seconds=1.0,
        )

        for i in range(6):
            ledger.record_fail(f"entry {i}")
        # Archives .1–.5 created; current file is empty (6th write went to new file)

        # Expire 3 of the 5 archives
        for n in (3, 4, 5):
            self._backdate(Path(f"{path}.{n}"), 2.0)

        count = ledger.apply_retention()
        assert count == 3, f"Expected 3 deletions, got {count}"

    def test_retention_no_op_when_no_archives_exist(self, tmp_path):
        """apply_retention() on a ledger with no rotated files returns 0."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, backup_count=5,
            retention_window_seconds=1.0,
        )

        ledger.record_fail("only entry, no rotation triggered")

        count = ledger.apply_retention()
        assert count == 0, "No archives → nothing to delete"

    def test_apply_retention_idempotent_on_already_clean_state(self, tmp_path):
        """Calling apply_retention() twice does not raise and returns 0 the second time."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=1.0,
        )

        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # rotation → .1

        archive = Path(f"{path}.1")
        self._backdate(archive, 2.0)

        first_pass = ledger.apply_retention()
        assert first_pass == 1

        second_pass = ledger.apply_retention()
        assert second_pass == 0, "Second pass on an already-clean ledger returns 0"

    # ── Automatic trigger on rotation ────────────────────────────────────────

    def test_retention_triggered_automatically_on_rotation(self, tmp_path):
        """
        When rotation happens, expired archives are deleted automatically
        without requiring an explicit apply_retention() call.

        Timeline:
          write "entry 1"         → stored in ledger.jsonl
          write "entry 2"         → size rotation: ledger.jsonl→.1, new ledger.jsonl;
                                     "entry 2" written to new file
          _backdate(.1, 2.0)      → .1 mtime is 2 s in the past (beyond 1 s window)
          write "entry 3"         → size rotation triggered (ledger.jsonl has "entry 2"):
                                     .1 (expired) shifts to .2
                                     ledger.jsonl → new .1
                                     new ledger.jsonl created
                                     _apply_retention_locked() runs → .2 expired → DELETED
                                     "entry 3" written to new ledger.jsonl
          assert .2 missing       → was swept in the same rotation that created it
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=1.0,
        )

        # First rotation cycle — creates .1
        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # size rotation → .1 created; "entry 2" in new file

        a1 = Path(f"{path}.1")
        assert a1.exists()
        # Back-date .1 so it will be expired on the very next sweep
        self._backdate(a1, 2.0)

        # This write's size check finds ledger.jsonl ("entry 2") > 1 byte → rotation:
        #   .1 (expired, backdated) shifts to .2
        #   ledger.jsonl ("entry 2") → .1
        #   new ledger.jsonl created
        #   retention sweep: .2 mtime = 2 s > 1 s window → DELETE
        #   "entry 3" written to new ledger.jsonl
        ledger.record_fail("entry 3")

        a2 = Path(f"{path}.2")
        assert not a2.exists(), (
            "Auto-retention must delete the expired archive (.2) "
            "immediately after rotation, without requiring apply_retention()"
        )

    def test_explicit_rotate_also_triggers_retention(self, tmp_path):
        """ledger.rotate() must also invoke the retention sweep."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=1.0,
        )

        # Create an archive and expire it
        ledger.record_fail("entry 1")
        ledger.record_fail("entry 2")  # rotation → .1

        a1 = Path(f"{path}.1")
        self._backdate(a1, 2.0)

        # Explicit rotate — shifts .1 → .2 (still expired) then sweeps
        ledger.rotate()

        a2 = Path(f"{path}.2")
        assert not a2.exists(), (
            "Explicit rotate() must trigger retention and delete expired .2"
        )

    # ── (d) Permissions of replacement files ────────────────────────────────

    def test_new_active_ledger_after_rotation_plus_retention_has_600_perms(
        self, tmp_path
    ):
        """
        (d) The fresh active ledger file created by a rotation that also triggers
        a retention sweep must have 0o600 permissions.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=0.05,  # tiny window
        )

        ledger.record_fail("entry before rotation")
        time.sleep(0.10)                        # let archive age beyond window
        ledger.record_fail("triggers rotation and retention sweep")

        assert _file_mode(path) == _FILE_MODE, (
            "New active ledger after rotation+retention must have 0o600 permissions"
        )

    def test_parent_dir_retains_700_perms_after_retention(self, tmp_path):
        """
        (d) The parent directory must retain 0o700 permissions after a retention
        sweep removes expired archives.
        """
        parent = tmp_path / "piiguard"
        path = parent / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=0.05,
        )

        ledger.record_fail("entry 1")
        time.sleep(0.10)
        ledger.record_fail("triggers rotation and sweep")

        assert _file_mode(parent) == _DIR_MODE, (
            "Parent directory must have 0o700 after retention sweep"
        )

    def test_surviving_archive_retains_600_perms_after_sweep(self, tmp_path):
        """
        (d) Archives that survive a retention sweep (i.e. are within the window)
        must retain their original 0o600 permissions — the sweep must not
        alter permissions of kept files.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=10,
            retention_window_seconds=1.0,
        )

        for i in range(4):
            ledger.record_fail(f"entry {i}")

        a1 = Path(f"{path}.1")
        a2 = Path(f"{path}.2")
        a3 = Path(f"{path}.3")

        # Expire only .3; keep .1 and .2 fresh
        self._backdate(a3, 2.0)

        ledger.apply_retention()

        assert a1.exists() and a2.exists(), "Fresh archives must survive"
        assert _file_mode(a1) == _FILE_MODE, (
            "Surviving archive .1 must retain 0o600 permissions"
        )
        assert _file_mode(a2) == _FILE_MODE, (
            "Surviving archive .2 must retain 0o600 permissions"
        )
        assert not a3.exists(), "Expired archive .3 must be deleted"

    # ── All archives expired ────────────────────────────────────────────────

    def test_all_archives_expired_deletes_all(self, tmp_path):
        """When every archive is beyond the window, all are deleted."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1, backup_count=5,
            retention_window_seconds=1.0,
        )

        for i in range(4):
            ledger.record_fail(f"entry {i}")

        for n in range(1, 4):
            self._backdate(Path(f"{path}.{n}"), float(n + 1))

        deleted = ledger.apply_retention()

        assert deleted == 3
        for n in range(1, 4):
            assert not Path(f"{path}.{n}").exists(), (
                f"All expired archives must be deleted; .{n} still exists"
            )

        # Active ledger must still exist and be writable
        assert path.exists()
        ledger.record_fail("post-retention entry")
        entries_after = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries_after.append(json.loads(line))
        assert len(entries_after) >= 1

    # ── Interaction with backup_count ────────────────────────────────────────

    def test_retention_and_backup_count_both_enforced(self, tmp_path):
        """
        backup_count limits how many archives are created; retention_window_seconds
        then deletes any of those that are too old.  Both constraints are active
        simultaneously.
        """
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(
            path, _TEST_KEY, max_bytes=1,
            backup_count=3,           # count-cap at 3 archives
            retention_window_seconds=1.0,
        )

        for i in range(6):
            ledger.record_fail(f"entry {i}")

        # backup_count=3 means only .1, .2, .3 can exist; .4+ are deleted by rotation
        assert not Path(f"{path}.4").exists(), (
            "backup_count must have already discarded .4"
        )

        # Now expire .2 and .3
        self._backdate(Path(f"{path}.2"), 2.0)
        self._backdate(Path(f"{path}.3"), 3.0)

        deleted = ledger.apply_retention()

        assert deleted == 2
        assert Path(f"{path}.1").exists(), ".1 must survive (within window)"
        assert not Path(f"{path}.2").exists()
        assert not Path(f"{path}.3").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-AC 3 — Purge command: verification matrix (c) and (d)
# ─────────────────────────────────────────────────────────────────────────────

class TestPurgeSubAC3:
    """
    Sub-AC 3 targeted verification.

    (c) No ledger files remain after purge and state is fully reset.
        - Active file is deleted.
        - Every rotated archive (.1 … .N) is deleted.
        - ``_initialized`` is ``False`` after purge.
        - ``_file_created_at`` is ``None`` after purge.

    (d) Any newly created seed/empty ledger file produced by the reset
        carries 0o600 (file) / 0o700 (parent dir) permissions.
        - ``initialize()`` after purge creates an empty file with 0o600.
        - Parent directory has 0o700 after ``initialize()``.
        - A regular write after purge (without explicit ``initialize()``)
          also produces the correct 0o600 file and 0o700 directory.
    """

    # ── (c) No files remain ─────────────────────────────────────────────────

    def test_c_no_active_file_remains_after_purge(self, tmp_path):
        """(c) The active ledger file is absent immediately after purge."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("before purge")
        assert path.exists()

        ledger.purge()

        assert not path.exists(), (
            "(c) Active ledger file must not exist after purge"
        )

    def test_c_all_rotated_archives_deleted_after_purge(self, tmp_path):
        """(c) Every rotated archive (.1 … .N) is deleted by purge."""
        path = tmp_path / "ledger.jsonl"
        N_ROTATIONS = 5
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=N_ROTATIONS)

        # Trigger N_ROTATIONS rotations
        for i in range(N_ROTATIONS + 1):
            ledger.record_fail(f"entry {i}")

        # Confirm archives exist before purge
        for n in range(1, N_ROTATIONS + 1):
            assert Path(f"{path}.{n}").exists(), (
                f"Archive .{n} must exist before purge"
            )

        ledger.purge()

        # No file or archive may survive
        assert not path.exists(), "(c) Active file must be gone"
        for n in range(1, N_ROTATIONS + 2):   # +2 for any stray
            assert not Path(f"{path}.{n}").exists(), (
                f"(c) Archive .{n} must be deleted by purge"
            )

    def test_c_initialized_flag_is_false_after_purge(self, tmp_path):
        """(c) In-process state: _initialized is False after purge."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("before purge")

        # Confirm initialized before purge
        assert ledger._initialized is True, (
            "Ledger should be initialized after first write"
        )

        ledger.purge()

        assert ledger._initialized is False, (
            "(c) _initialized must be False after purge — state fully reset"
        )

    def test_c_file_created_at_is_none_after_purge(self, tmp_path):
        """(c) In-process state: _file_created_at is None after purge."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("before purge")

        # Confirm _file_created_at was set by the first write
        assert ledger._file_created_at is not None, (
            "_file_created_at must be set after first write"
        )

        ledger.purge()

        assert ledger._file_created_at is None, (
            "(c) _file_created_at must be None after purge — age timer reset"
        )

    def test_c_purge_on_never_written_ledger_leaves_no_files(self, tmp_path):
        """(c) Purge on a never-written (no files) ledger is safe and idempotent."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.purge()  # must not raise

        assert not path.exists(), "(c) File must not exist after purge of empty ledger"
        assert ledger._initialized is False
        assert ledger._file_created_at is None

    def test_c_purge_twice_leaves_no_files(self, tmp_path):
        """(c) Calling purge twice is idempotent — no exceptions, no files."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY, max_bytes=1, backup_count=3)

        for i in range(4):
            ledger.record_fail(f"entry {i}")

        ledger.purge()
        ledger.purge()  # second purge must also be safe

        assert not path.exists()
        for n in range(1, 5):
            assert not Path(f"{path}.{n}").exists()
        assert ledger._initialized is False
        assert ledger._file_created_at is None

    def test_c_full_state_reset_confirmed_by_re_initialization(self, tmp_path):
        """(c) After purge the Ledger can be re-used without residual state."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)

        ledger.record_fail("before purge")
        ledger.purge()

        # Re-use: a new write must start from scratch (no stale state)
        ledger.record_fail("after purge")

        entries = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    import json as _json
                    entries.append(_json.loads(line))

        assert len(entries) == 1, (
            "(c) Only the post-purge entry must exist — no pre-purge residue"
        )
        assert entries[0]["fail_reason"] == "after purge"

    # ── (d) Seed file has correct permissions ───────────────────────────────

    def test_d_initialize_after_purge_creates_file_with_600_perms(self, tmp_path):
        """(d) initialize() after purge creates an empty file with mode 0o600."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("some event")

        ledger.purge()
        assert not path.exists(), "File must not exist immediately after purge"

        ledger.initialize()

        assert path.exists(), "(d) initialize() must create the seed file"
        assert _file_mode(path) == _FILE_MODE, (
            f"(d) Seed file must have mode 0o600, got 0o{_file_mode(path):o}"
        )

    def test_d_initialize_after_purge_gives_parent_dir_700_perms(self, tmp_path):
        """(d) initialize() after purge creates (or enforces) parent dir with 0o700."""
        parent = tmp_path / "piiguard"
        path = parent / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("some event")

        ledger.purge()
        ledger.initialize()

        assert parent.exists(), "(d) Parent directory must exist after initialize()"
        assert _file_mode(parent) == _DIR_MODE, (
            f"(d) Parent dir must have mode 0o700, got 0o{_file_mode(parent):o}"
        )

    def test_d_initialize_seed_file_is_empty(self, tmp_path):
        """(d) The seed file created by initialize() contains no data (empty)."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("before purge")

        ledger.purge()
        ledger.initialize()

        content = path.read_text(encoding="utf-8")
        assert content == "", (
            "(d) Seed file created by initialize() must be empty (no event data)"
        )

    def test_d_write_after_purge_creates_file_with_600_perms(self, tmp_path):
        """(d) The first write after purge produces a 0o600 file (no explicit initialize())."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("pre-purge event")

        ledger.purge()
        ledger.record_fail("post-purge event")

        assert path.exists()
        assert _file_mode(path) == _FILE_MODE, (
            f"(d) File after first post-purge write must have 0o600, "
            f"got 0o{_file_mode(path):o}"
        )

    def test_d_write_after_purge_gives_parent_dir_700_perms(self, tmp_path):
        """(d) The first write after purge enforces 0o700 on the parent directory."""
        parent = tmp_path / "piiguard"
        path = parent / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("pre-purge")

        ledger.purge()
        ledger.record_fail("post-purge")

        assert _file_mode(parent) == _DIR_MODE, (
            f"(d) Parent dir must have mode 0o700 after first post-purge write, "
            f"got 0o{_file_mode(parent):o}"
        )

    def test_d_initialize_idempotent_when_file_already_exists(self, tmp_path):
        """(d) initialize() is safe to call when the file already exists."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("some event")

        # Do NOT purge first — file already exists
        ledger.initialize()  # must not raise, must not alter content

        entries = []
        with open(path, encoding="utf-8") as fh:
            import json as _json2
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(_json2.loads(line))

        assert len(entries) == 1, (
            "(d) initialize() on an existing ledger must not truncate content"
        )
        assert _file_mode(path) == _FILE_MODE

    def test_d_initialize_tightens_loose_permissions(self, tmp_path):
        """(d) initialize() corrects a seed file that was created with loose perms."""
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, _TEST_KEY)
        ledger.record_fail("event")
        ledger.purge()

        # Manually create the file with wrong permissions
        path.write_text("")
        os.chmod(path, 0o644)
        # Also loosen the directory
        os.chmod(tmp_path, 0o755)

        # Reinstate Ledger (simulate re-open after purge)
        ledger2 = Ledger(path, _TEST_KEY)
        ledger2.initialize()

        assert _file_mode(path) == _FILE_MODE, (
            "(d) initialize() must tighten file permissions to 0o600"
        )
        assert _file_mode(tmp_path) == _DIR_MODE, (
            "(d) initialize() must tighten parent dir permissions to 0o700"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-AC 3 — CLI ledger purge command
# ─────────────────────────────────────────────────────────────────────────────

class TestLedgerPurgeCLI:
    """
    Verify that the ``piiguard ledger purge`` CLI command:

    (c) deletes every ledger file and leaves no files on disk;
    (d) the seed file (created by default) carries 0o600/0o700 permissions.
    """

    def _run_purge(
        self,
        ledger_path: Path,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        """Invoke the CLI ledger-purge handler via ``main()``."""
        from pii_guard.cli import main
        argv = ["ledger", "purge", "--ledger-path", str(ledger_path)]
        if extra_args:
            argv.extend(extra_args)
        return main(argv)

    def test_cli_purge_exits_zero(self, tmp_path):
        """``piiguard ledger purge`` exits with code 0 on success."""
        path = tmp_path / "ledger.jsonl"
        # Create a ledger file to purge
        key = os.urandom(32)
        Ledger(path, key).record_fail("test")

        rc = self._run_purge(path)
        assert rc == 0

    def test_cli_purge_c_no_active_file_remains(self, tmp_path):
        """(c) Active ledger file is gone after CLI purge (without reseed)."""
        path = tmp_path / "ledger.jsonl"
        Ledger(path, os.urandom(32)).record_fail("test")

        self._run_purge(path, ["--no-reseed"])

        assert not path.exists(), (
            "(c) Active file must not remain after CLI purge --no-reseed"
        )

    def test_cli_purge_c_all_archives_deleted(self, tmp_path):
        """(c) All rotated archive files are deleted by CLI purge."""
        path = tmp_path / "ledger.jsonl"
        key = os.urandom(32)
        ledger = Ledger(path, key, max_bytes=1, backup_count=4)
        for i in range(5):
            ledger.record_fail(f"entry {i}")

        # Archives .1–.4 exist
        for n in range(1, 5):
            assert Path(f"{path}.{n}").exists()

        self._run_purge(path, ["--no-reseed"])

        assert not path.exists(), "(c) Active file must be gone"
        for n in range(1, 6):
            assert not Path(f"{path}.{n}").exists(), (
                f"(c) Archive .{n} must be deleted by CLI purge"
            )

    def test_cli_purge_d_seed_file_has_600_perms(self, tmp_path):
        """(d) Default CLI purge creates a seed file with 0o600 permissions."""
        path = tmp_path / "ledger.jsonl"
        Ledger(path, os.urandom(32)).record_fail("test")

        # Default (no --no-reseed) creates the seed file
        rc = self._run_purge(path)
        assert rc == 0

        assert path.exists(), "(d) Seed file must exist after default CLI purge"
        assert _file_mode(path) == _FILE_MODE, (
            f"(d) Seed file must have mode 0o600, got 0o{_file_mode(path):o}"
        )

    def test_cli_purge_d_parent_dir_has_700_perms(self, tmp_path):
        """(d) Default CLI purge creates seed file with parent dir 0o700."""
        parent = tmp_path / "piiguard"
        path = parent / "ledger.jsonl"
        parent.mkdir()
        Ledger(path, os.urandom(32)).record_fail("test")

        rc = self._run_purge(path)
        assert rc == 0

        assert _file_mode(parent) == _DIR_MODE, (
            f"(d) Parent dir must have mode 0o700 after CLI purge, "
            f"got 0o{_file_mode(parent):o}"
        )

    def test_cli_purge_d_seed_file_is_empty(self, tmp_path):
        """(d) The seed file created by default CLI purge is empty."""
        path = tmp_path / "ledger.jsonl"
        Ledger(path, os.urandom(32)).record_fail("test")

        self._run_purge(path)

        content = path.read_text(encoding="utf-8")
        assert content == "", (
            "(d) Seed file after CLI purge must be empty (no event entries)"
        )

    def test_cli_purge_no_reseed_leaves_no_files(self, tmp_path):
        """``--no-reseed`` leaves no files at all after purge."""
        path = tmp_path / "ledger.jsonl"
        key = os.urandom(32)
        ledger = Ledger(path, key, max_bytes=1, backup_count=3)
        for i in range(4):
            ledger.record_fail(f"entry {i}")

        rc = self._run_purge(path, ["--no-reseed"])
        assert rc == 0

        assert not path.exists()
        for n in range(1, 5):
            assert not Path(f"{path}.{n}").exists()

    def test_cli_purge_on_nonexistent_ledger_succeeds(self, tmp_path):
        """CLI purge on a path with no existing ledger must exit 0."""
        path = tmp_path / "nonexistent_ledger.jsonl"
        assert not path.exists()

        rc = self._run_purge(path, ["--no-reseed"])
        assert rc == 0

    def test_cli_purge_env_var_sets_default_path(self, tmp_path, monkeypatch):
        """PIIGUARD_LEDGER_PATH env var sets the default ledger path."""
        path = tmp_path / "env_ledger.jsonl"
        Ledger(path, os.urandom(32)).record_fail("test")

        monkeypatch.setenv("PIIGUARD_LEDGER_PATH", str(path))

        from pii_guard.cli import main
        # No --ledger-path flag; must pick up from env var
        rc = main(["ledger", "purge", "--no-reseed"])
        assert rc == 0
        assert not path.exists(), "Env-var path must have been purged"
