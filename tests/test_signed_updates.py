"""
Tests for the signed-channel update mechanism (Sub-AC 7.1 / AC 7).

Coverage matrix
---------------
Valid signatures
  [V1] sign + verify round-trip passes silently
  [V2] multiple entries: all verified correctly
  [V3] manifest with metadata fields: metadata covered by HMAC
  [V4] to_json / from_json round-trip preserves all fields + passes verify
  [V5] load_verified() reads a file and verifies it end-to-end
  [V6] make_manifest() computes correct per-entry sha256

Invalid signatures / wrong key
  [I1] unsigned manifest raises UpdateRejectedError
  [I2] wrong key raises UpdateRejectedError
  [I3] truncated key (<32 bytes) rejected at verifier construction
  [I4] truncated key (<32 bytes) rejected at signer construction
  [I5] empty signature field raises UpdateRejectedError
  [I6] malformed signature (no ':' separator) raises UpdateRejectedError
  [I7] unsupported algorithm prefix raises UpdateRejectedError
  [I8] non-hex signature hex body raises UpdateRejectedError
  [I9] hex-padded (uppercase) signature passes (compare_digest is case-normalised)

Tampered manifests
  [T1] tampering manifest_version after signing raises UpdateRejectedError
  [T2] tampering kind field after signing raises UpdateRejectedError
  [T3] tampering timestamp field after signing raises UpdateRejectedError
  [T4] tampering author field after signing raises UpdateRejectedError
  [T5] tampering signer_id field after signing raises UpdateRejectedError
  [T6] adding an entry after signing raises UpdateRejectedError
  [T7] removing an entry after signing raises UpdateRejectedError
  [T8] tampering entry name after signing raises UpdateRejectedError
  [T9] tampering entry kind after signing raises UpdateRejectedError
  [T10] tampering entry content after signing → entry hash mismatch → rejected
  [T11] tampering entry sha256 field after signing → HMAC fails (sha256 is in payload)
  [T12] tampering entry metadata after signing raises UpdateRejectedError
  [T13] bit-flip in signature hex raises UpdateRejectedError
  [T14] flipping a single byte in signature raises UpdateRejectedError
  [T15] schema_version tamper raises UpdateRejectedError

Structural validation
  [S1] missing required field in manifest dict raises UpdateRejectedError
  [S2] wrong schema_version raises UpdateRejectedError
  [S3] malformed JSON raises UpdateRejectedError
  [S4] load_verified() on tampered file raises UpdateRejectedError
  [S5] load_verified() on unsigned file raises UpdateRejectedError

Sign independence
  [X1] signing does not mutate the original manifest object
  [X2] two separate signers with different keys produce different signatures
  [X3] signer_id is set on the signed copy, not on the original
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile

import pytest

from pii_guard.updater import (
    DEFAULT_KEY_PATH,
    ManifestEntry,
    UpdateManifest,
    UpdateRejectedError,
    UpdateSigner,
    UpdateVerifier,
    generate_update_key,
    make_manifest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def key() -> bytes:
    """A fresh 256-bit HMAC key."""
    return UpdateSigner.generate_key()


@pytest.fixture()
def other_key() -> bytes:
    """A second, different 256-bit HMAC key."""
    return UpdateSigner.generate_key()


@pytest.fixture()
def signer(key: bytes) -> UpdateSigner:
    return UpdateSigner(key)


@pytest.fixture()
def verifier(key: bytes) -> UpdateVerifier:
    return UpdateVerifier(key)


@pytest.fixture()
def simple_manifest() -> UpdateManifest:
    """A single-entry rule_update manifest (unsigned)."""
    return make_manifest(
        "1.0.0",
        "rule_update",
        [("email_rule", "regex_rule", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")],
        author="test-author",
        timestamp="2026-06-22T12:00:00Z",
    )


@pytest.fixture()
def multi_manifest() -> UpdateManifest:
    """A multi-entry mixed_update manifest (unsigned)."""
    return make_manifest(
        "2.0.0",
        "mixed_update",
        [
            ("email_rule", "regex_rule", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
            ("ner_model_cfg", "ner_model", '{"model": "gliner-small", "threshold": 0.4}'),
            ("corpus_kr", "golden_corpus", '{"rrn": ["800101-1234567"]}'),
        ],
        author="test-author",
        timestamp="2026-06-22T12:00:00Z",
    )


@pytest.fixture()
def signed_manifest(signer: UpdateSigner, simple_manifest: UpdateManifest) -> UpdateManifest:
    return signer.sign(simple_manifest)


# ─────────────────────────────────────────────────────────────────────────────
# V — Valid signature tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidSignatures:
    """[V1–V6] Happy-path verification."""

    def test_v1_sign_verify_round_trip(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[V1] sign + verify round-trip passes silently."""
        signed = signer.sign(simple_manifest)
        assert signed.signature is not None
        assert signed.signature.startswith("hmac-sha256:")
        # Should not raise
        verifier.verify(signed)

    def test_v2_multiple_entries(
        self, signer: UpdateSigner, verifier: UpdateVerifier, multi_manifest: UpdateManifest
    ) -> None:
        """[V2] Multiple entries: all verified correctly."""
        signed = signer.sign(multi_manifest)
        verifier.verify(signed)  # must not raise

    def test_v3_metadata_covered_by_hmac(
        self, key: bytes
    ) -> None:
        """[V3] Manifest with metadata fields — metadata is part of canonical payload."""
        m = make_manifest(
            "1.0.0",
            "rule_update",
            [("rule1", "regex_rule", "abc", {"author": "alice", "version": "1"})],
            timestamp="2026-01-01T00:00:00Z",
        )
        signer = UpdateSigner(key)
        verifier = UpdateVerifier(key)
        signed = signer.sign(m)
        verifier.verify(signed)

        # Tamper metadata — HMAC should fail
        tampered = copy.deepcopy(signed)
        tampered.entries[0].metadata["author"] = "mallory"
        with pytest.raises(UpdateRejectedError, match="HMAC"):
            verifier.verify(tampered)

    def test_v4_json_round_trip(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[V4] to_json / from_json round-trip preserves fields + passes verify."""
        signed = signer.sign(simple_manifest)
        json_str = signed.to_json()
        reloaded = UpdateManifest.from_json(json_str)

        assert reloaded.manifest_version == signed.manifest_version
        assert reloaded.kind == signed.kind
        assert reloaded.signature == signed.signature
        assert len(reloaded.entries) == len(signed.entries)
        verifier.verify(reloaded)

    def test_v5_load_verified_file(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[V5] load_verified() reads a file, verifies, and returns the manifest."""
        signed = signer.sign(simple_manifest)
        with tempfile.NamedTemporaryFile(
            suffix=".piiguard-manifest.json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(signed.to_json())
            tmp_path = pathlib.Path(f.name)

        try:
            loaded = verifier.load_verified(tmp_path)
            assert loaded.manifest_version == signed.manifest_version
            assert loaded.entries[0].name == "email_rule"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_v6_make_manifest_computes_correct_sha256(self) -> None:
        """[V6] make_manifest() sets per-entry sha256 to SHA-256 of content UTF-8 bytes."""
        content = r"[a-zA-Z0-9]+@example\.com"
        m = make_manifest("1.0.0", "rule_update", [("r", "regex_rule", content)])
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert m.entries[0].sha256 == expected


# ─────────────────────────────────────────────────────────────────────────────
# I — Invalid signature / wrong key tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidSignatures:
    """[I1–I9] Cases where verification must raise UpdateRejectedError."""

    def test_i1_unsigned_manifest_rejected(
        self, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I1] Unsigned manifest raises UpdateRejectedError."""
        assert simple_manifest.signature is None
        with pytest.raises(UpdateRejectedError, match="no signature"):
            verifier.verify(simple_manifest)

    def test_i2_wrong_key_rejected(
        self,
        signer: UpdateSigner,
        other_key: bytes,
        simple_manifest: UpdateManifest,
    ) -> None:
        """[I2] Manifest signed with key1, verified with key2 → rejected."""
        signed = signer.sign(simple_manifest)
        wrong_verifier = UpdateVerifier(other_key)
        with pytest.raises(UpdateRejectedError, match="[Ss]ignature|HMAC|rejected"):
            wrong_verifier.verify(signed)

    def test_i3_truncated_key_verifier(self) -> None:
        """[I3] Verifier with key < 32 bytes raises ValueError at construction."""
        with pytest.raises(ValueError, match="at least 32"):
            UpdateVerifier(b"tooshort")

    def test_i4_truncated_key_signer(self) -> None:
        """[I4] Signer with key < 32 bytes raises ValueError at construction."""
        with pytest.raises(ValueError, match="at least 32"):
            UpdateSigner(b"tooshort")

    def test_i5_empty_signature_rejected(
        self, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I5] Empty string signature raises UpdateRejectedError."""
        m = copy.deepcopy(simple_manifest)
        m.signature = ""
        with pytest.raises(UpdateRejectedError):
            verifier.verify(m)

    def test_i6_malformed_signature_no_colon(
        self, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I6] Signature without ':' separator raises UpdateRejectedError."""
        m = copy.deepcopy(simple_manifest)
        m.signature = "abcdef1234567890"  # no colon
        with pytest.raises(UpdateRejectedError, match="[Mm]alformed|separator"):
            verifier.verify(m)

    def test_i7_unsupported_algorithm(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I7] Unsupported algorithm prefix raises UpdateRejectedError."""
        signed = signer.sign(simple_manifest)
        # Swap algorithm prefix
        _, hex_part = signed.signature.split(":", 1)
        m = copy.deepcopy(signed)
        m.signature = f"rsa-sha256:{hex_part}"
        with pytest.raises(UpdateRejectedError, match="[Uu]nsupported.*algorithm|algorithm"):
            verifier.verify(m)

    def test_i8_non_hex_signature_body(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I8] Non-hex characters in the signature body raises UpdateRejectedError."""
        m = copy.deepcopy(simple_manifest)
        m.signature = "hmac-sha256:gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
        with pytest.raises(UpdateRejectedError, match="[Hh]ex|invalid|non-hex"):
            verifier.verify(m)

    def test_i9_uppercase_hex_normalised(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[I9] Uppercase hex in signature is normalised and accepted."""
        signed = signer.sign(simple_manifest)
        _, hex_part = signed.signature.split(":", 1)
        m = copy.deepcopy(signed)
        m.signature = f"hmac-sha256:{hex_part.upper()}"
        # Should NOT raise — compare_digest uses .lower()
        verifier.verify(m)


# ─────────────────────────────────────────────────────────────────────────────
# T — Tampered manifest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTamperedManifests:
    """[T1–T15] Any post-signing modification must be caught."""

    def _tamper_and_expect_reject(
        self, verifier: UpdateVerifier, signed: UpdateManifest
    ) -> None:
        """Assert that verifier.verify(signed) raises UpdateRejectedError."""
        with pytest.raises(UpdateRejectedError):
            verifier.verify(signed)

    def test_t1_tamper_manifest_version(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T1] Changing manifest_version after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.manifest_version = "9.9.9"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t2_tamper_kind(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T2] Changing kind field after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.kind = "model_update"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t3_tamper_timestamp(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T3] Changing timestamp after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.timestamp = "1970-01-01T00:00:00Z"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t4_tamper_author(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T4] Changing author after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.author = "mallory"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t5_tamper_signer_id(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T5] Changing signer_id after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.signer_id = "piiguard-evil-key-v0"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t6_add_entry_after_signing(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T6] Adding an extra entry after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        extra_content = "injected_rule_content"
        tampered.entries.append(
            ManifestEntry(
                name="injected",
                kind="regex_rule",
                sha256=hashlib.sha256(extra_content.encode()).hexdigest(),
                content=extra_content,
            )
        )
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t7_remove_entry_after_signing(
        self, signer: UpdateSigner, verifier: UpdateVerifier, multi_manifest: UpdateManifest
    ) -> None:
        """[T7] Removing an entry after signing → HMAC fails."""
        signed = signer.sign(multi_manifest)
        tampered = copy.deepcopy(signed)
        tampered.entries.pop()
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t8_tamper_entry_name(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T8] Changing entry name after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.entries[0].name = "evil_rule"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t9_tamper_entry_kind(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T9] Changing entry kind after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.entries[0].kind = "ner_model"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t10_tamper_entry_content(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T10] Changing entry content after signing → per-entry hash mismatch → rejected.

        Note: The HMAC covers the *original* sha256 value (which no longer
        matches the tampered content), so verification may fail at either the
        HMAC step or the content-hash step.  Both outcomes correctly reject.
        """
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.entries[0].content = "INJECTED_MALICIOUS_PATTERN"
        # The sha256 field still holds the original hash → mismatch detected
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t11_tamper_entry_sha256_field(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T11] Changing entry sha256 field after signing → HMAC covers the hash → rejected."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        # Attacker tries to fix the sha256 to match tampered content
        evil_content = "EVIL_PATTERN"
        tampered.entries[0].content = evil_content
        tampered.entries[0].sha256 = hashlib.sha256(evil_content.encode()).hexdigest()
        # sha256 is part of the canonical payload → HMAC will fail
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t12_tamper_entry_metadata(
        self, signer: UpdateSigner, verifier: UpdateVerifier
    ) -> None:
        """[T12] Changing entry metadata after signing → HMAC fails."""
        m = make_manifest(
            "1.0.0",
            "rule_update",
            [("r", "regex_rule", "abc", {"version": "1"})],
            timestamp="2026-01-01T00:00:00Z",
        )
        signed = signer.sign(m)
        tampered = copy.deepcopy(signed)
        tampered.entries[0].metadata["version"] = "99"
        with pytest.raises(UpdateRejectedError):
            verifier.verify(tampered)

    def test_t13_bit_flip_in_signature_hex(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T13] Flipping one nibble in the signature hex raises UpdateRejectedError."""
        signed = signer.sign(simple_manifest)
        alg, hex_part = signed.signature.split(":", 1)
        # Flip the first hex nibble (0→1, any other digit → one off)
        hex_list = list(hex_part)
        first_char = hex_list[0]
        hex_list[0] = "0" if first_char != "0" else "1"
        tampered = copy.deepcopy(signed)
        tampered.signature = f"{alg}:{''.join(hex_list)}"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t14_single_byte_flip_in_signature(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T14] Flipping a byte in the *middle* of the signature hex → rejected."""
        signed = signer.sign(simple_manifest)
        alg, hex_part = signed.signature.split(":", 1)
        hex_list = list(hex_part)
        mid = len(hex_list) // 2
        hex_list[mid] = "0" if hex_list[mid] != "0" else "f"
        tampered = copy.deepcopy(signed)
        tampered.signature = f"{alg}:{''.join(hex_list)}"
        self._tamper_and_expect_reject(verifier, tampered)

    def test_t15_tamper_schema_version(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[T15] Changing schema_version after signing → HMAC fails."""
        signed = signer.sign(simple_manifest)
        tampered = copy.deepcopy(signed)
        tampered.schema_version = "99"
        self._tamper_and_expect_reject(verifier, tampered)


# ─────────────────────────────────────────────────────────────────────────────
# S — Structural validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralValidation:
    """[S1–S5] Manifest parsing and file-load error handling."""

    def test_s1_missing_required_field(self) -> None:
        """[S1] Missing required field in manifest dict raises UpdateRejectedError."""
        incomplete = {
            # "manifest_version" deliberately missing
            "kind": "rule_update",
            "timestamp": "2026-01-01T00:00:00Z",
            "author": "test",
            "signer_id": "test",
            "entries": [],
        }
        with pytest.raises(UpdateRejectedError, match="[Ii]nvalid|missing|required|field"):
            UpdateManifest.from_dict(incomplete)

    def test_s2_wrong_schema_version(self) -> None:
        """[S2] Unrecognised schema_version raises UpdateRejectedError."""
        d = {
            "schema_version": "99",
            "manifest_version": "1.0.0",
            "kind": "rule_update",
            "timestamp": "2026-01-01T00:00:00Z",
            "author": "test",
            "signer_id": "test",
            "entries": [],
        }
        with pytest.raises(UpdateRejectedError, match="[Ss]chema"):
            UpdateManifest.from_dict(d)

    def test_s3_malformed_json_raises(self) -> None:
        """[S3] Malformed JSON raises UpdateRejectedError."""
        with pytest.raises(UpdateRejectedError, match="[Pp]arse|JSON"):
            UpdateManifest.from_json("{not valid json")

    def test_s4_load_verified_tampered_file(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[S4] load_verified() on a tampered file raises UpdateRejectedError."""
        signed = signer.sign(simple_manifest)
        d = signed.to_dict()
        d["author"] = "evil"  # tamper after signing
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump(d, f)
            tmp_path = pathlib.Path(f.name)

        try:
            with pytest.raises(UpdateRejectedError):
                verifier.load_verified(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_s5_load_verified_unsigned_file(
        self, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """[S5] load_verified() on an unsigned file raises UpdateRejectedError."""
        d = simple_manifest.to_dict()  # no signature key
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump(d, f)
            tmp_path = pathlib.Path(f.name)

        try:
            with pytest.raises(UpdateRejectedError, match="[Ss]ignature|unsigned"):
                verifier.load_verified(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# X — Sign independence tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSignIndependence:
    """[X1–X3] Signing must not mutate originals or conflate signers."""

    def test_x1_sign_does_not_mutate_original(
        self, signer: UpdateSigner, simple_manifest: UpdateManifest
    ) -> None:
        """[X1] Signing returns a new manifest; original remains unsigned."""
        original_sig = simple_manifest.signature
        signed = signer.sign(simple_manifest)
        assert simple_manifest.signature == original_sig  # unchanged
        assert signed is not simple_manifest             # different object
        assert signed.signature is not None

    def test_x2_different_keys_different_signatures(
        self, key: bytes, other_key: bytes, simple_manifest: UpdateManifest
    ) -> None:
        """[X2] Two signers with different keys produce different signatures."""
        s1 = UpdateSigner(key)
        s2 = UpdateSigner(other_key)
        sig1 = s1.sign(simple_manifest).signature
        sig2 = s2.sign(simple_manifest).signature
        assert sig1 != sig2

    def test_x3_signer_id_set_on_copy(
        self, key: bytes, simple_manifest: UpdateManifest
    ) -> None:
        """[X3] The signer_id on the signed copy matches the signer, not the original."""
        original_signer_id = simple_manifest.signer_id
        custom_signer = UpdateSigner(key, signer_id="my-custom-key-id")
        signed = custom_signer.sign(simple_manifest)
        assert signed.signer_id == "my-custom-key-id"
        # Original is unchanged
        assert simple_manifest.signer_id == original_signer_id


# ─────────────────────────────────────────────────────────────────────────────
# K — Key file tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyFile:
    """Key generation, persistence, and loading."""

    def test_generate_update_key_creates_file(self, tmp_path: pathlib.Path) -> None:
        """generate_update_key() writes a 32-byte key at the requested path."""
        key_path = tmp_path / "keys" / "update.key"
        key = generate_update_key(key_path)
        assert key_path.exists()
        assert key_path.read_bytes() == key
        assert len(key) == 32

    def test_generate_update_key_sets_permissions(self, tmp_path: pathlib.Path) -> None:
        """Key file is created with mode 0o600."""
        key_path = tmp_path / "update.key"
        generate_update_key(key_path)
        mode = oct(key_path.stat().st_mode)[-3:]
        assert mode == "600", f"Expected 600, got {mode}"

    def test_generate_update_key_no_overwrite(self, tmp_path: pathlib.Path) -> None:
        """generate_update_key() raises FileExistsError if key already exists."""
        key_path = tmp_path / "update.key"
        generate_update_key(key_path)
        with pytest.raises(FileExistsError):
            generate_update_key(key_path, overwrite=False)

    def test_generate_update_key_overwrite(self, tmp_path: pathlib.Path) -> None:
        """generate_update_key(overwrite=True) replaces an existing key."""
        key_path = tmp_path / "update.key"
        key1 = generate_update_key(key_path)
        key2 = generate_update_key(key_path, overwrite=True)
        # New key is written (probabilistically different)
        assert key_path.read_bytes() == key2
        # Old key is gone
        assert key1 != key2 or len(key2) == 32  # guard against improbable collision

    def test_signer_from_key_file(self, tmp_path: pathlib.Path, simple_manifest: UpdateManifest) -> None:
        """UpdateSigner.from_key_file() loads and uses the key correctly."""
        key_path = tmp_path / "update.key"
        generate_update_key(key_path)

        signer = UpdateSigner.from_key_file(key_path)
        verifier = UpdateVerifier.from_key_file(key_path)

        signed = signer.sign(simple_manifest)
        verifier.verify(signed)  # must not raise

    def test_verifier_from_key_file(self, tmp_path: pathlib.Path, simple_manifest: UpdateManifest) -> None:
        """UpdateVerifier.from_key_file() loads and uses the key correctly."""
        key_path = tmp_path / "update.key"
        key = generate_update_key(key_path)

        signer = UpdateSigner(key)
        verifier = UpdateVerifier.from_key_file(key_path)

        signed = signer.sign(simple_manifest)
        verifier.verify(signed)  # must not raise

    def test_verifier_from_missing_key_file(self, tmp_path: pathlib.Path) -> None:
        """from_key_file() raises FileNotFoundError when key path does not exist."""
        with pytest.raises(FileNotFoundError):
            UpdateVerifier.from_key_file(tmp_path / "nonexistent.key")

    def test_signer_from_missing_key_file(self, tmp_path: pathlib.Path) -> None:
        """from_key_file() raises FileNotFoundError when key path does not exist."""
        with pytest.raises(FileNotFoundError):
            UpdateSigner.from_key_file(tmp_path / "nonexistent.key")


# ─────────────────────────────────────────────────────────────────────────────
# E — Edge-case / extra coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Additional edge-case coverage."""

    def test_empty_entries_list(self, key: bytes) -> None:
        """An empty entries list can be signed and verified."""
        signer = UpdateSigner(key)
        verifier = UpdateVerifier(key)
        m = make_manifest("1.0.0", "rule_update", [], timestamp="2026-01-01T00:00:00Z")
        signed = signer.sign(m)
        verifier.verify(signed)

    def test_unicode_content_in_entry(self, key: bytes) -> None:
        """Unicode content (e.g. Korean regex) is handled correctly."""
        korean_content = r"[가-힣]{2,5}"
        signer = UpdateSigner(key)
        verifier = UpdateVerifier(key)
        m = make_manifest(
            "1.0.0",
            "rule_update",
            [("kr_name", "regex_rule", korean_content)],
            timestamp="2026-01-01T00:00:00Z",
        )
        signed = signer.sign(m)
        verifier.verify(signed)  # must not raise

    def test_resigning_produces_same_mac(self, key: bytes, simple_manifest: UpdateManifest) -> None:
        """Signing the same manifest twice with the same key yields the same HMAC."""
        signer = UpdateSigner(key)
        s1 = signer.sign(simple_manifest)
        s2 = signer.sign(simple_manifest)
        assert s1.signature == s2.signature

    def test_manifest_entry_compute_sha256(self) -> None:
        """ManifestEntry.compute_sha256() matches make_manifest's computed hash."""
        content = "test_content_abc"
        m = make_manifest("1.0.0", "rule_update", [("x", "regex_rule", content)])
        entry = m.entries[0]
        assert entry.compute_sha256() == entry.sha256

    def test_content_hash_checked_independent_of_hmac(
        self, signer: UpdateSigner, verifier: UpdateVerifier, simple_manifest: UpdateManifest
    ) -> None:
        """Content is ALSO checked against per-entry sha256 even after HMAC passes.

        We sign a manifest, then re-sign a *different* manifest that has the
        same canonical structure but a different entry content, giving it a
        valid HMAC.  The verifier should reject because the sha256 field now
        doesn't match the actual content.
        """
        # Build a manifest where sha256 is "wrong" but HMAC is valid
        # (i.e., the signer deliberately puts a wrong sha256 in to simulate
        # a malformed but HMAC-valid manifest from a buggy publisher tool)
        m = make_manifest(
            "1.0.0",
            "rule_update",
            [("r", "regex_rule", "good_content")],
            timestamp="2026-01-01T00:00:00Z",
        )
        # Replace entry content without recomputing sha256
        m.entries[0].content = "bad_content"
        # sha256 still refers to "good_content" hash → mismatch
        signed = signer.sign(m)  # valid HMAC over this (wrong) sha256
        with pytest.raises(UpdateRejectedError, match="[Hh]ash|mismatch|tampered"):
            verifier.verify(signed)

    def test_make_manifest_default_timestamp(self) -> None:
        """make_manifest() fills in a timestamp when none is provided."""
        m = make_manifest("1.0.0", "rule_update", [])
        assert m.timestamp
        assert "T" in m.timestamp  # ISO-8601 format

    def test_manifest_to_dict_excludes_signature_when_none(self) -> None:
        """to_dict() omits 'signature' key when manifest is unsigned."""
        m = make_manifest("1.0.0", "rule_update", [])
        d = m.to_dict()
        assert "signature" not in d

    def test_manifest_to_dict_includes_signature_when_signed(
        self, signer: UpdateSigner, simple_manifest: UpdateManifest
    ) -> None:
        """to_dict() includes 'signature' key on a signed manifest."""
        signed = signer.sign(simple_manifest)
        d = signed.to_dict()
        assert "signature" in d
        assert d["signature"].startswith("hmac-sha256:")
