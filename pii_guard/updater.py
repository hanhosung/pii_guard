"""
Signed-channel update mechanism for PII-Guard detection rules and NER models.

Architecture
------------
Detection rules (regex patterns, checksums, category specs) and NER model
artefacts are distributed as *UpdateManifest* packages — JSON documents that
pair a list of named entries with a cryptographic signature.

Signing uses **HMAC-SHA256** with a locally-held 256-bit key stored in the
control-plane directory (outside the protected agent's write path, default:
``~/.config/piiguard/keys/update.key``, mode 0o600, parent dir 0o700).

Two-layer integrity
~~~~~~~~~~~~~~~~~~~
1. **Manifest HMAC** — computed over the canonical JSON payload (all fields
   except ``signature``, sorted keys, no whitespace).  Proves the *entire*
   manifest—including entry names, hashes, and metadata—came from a holder
   of the local update key and has not been altered in transit or at rest.

2. **Per-entry SHA-256** — each entry carries a ``sha256`` hex-digest of its
   content bytes.  Verified *after* the HMAC passes, proving entry content
   exactly matches what was committed into the signed manifest.

Threat model (in-scope)
~~~~~~~~~~~~~~~~~~~~~~~
- Malicious content injection (unsigned package rejected immediately).
- Manifest tampering (HMAC fails → rejected).
- Entry-level tampering (per-entry hash fails → rejected).
- Algorithm substitution (only ``hmac-sha256`` accepted).
- Replay with wrong key (HMAC fails → rejected).

Out of scope (per project threat model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Attacker with write access to the control-plane key file (root/kernel
  level — see boundary declaration and project docs).

Silent auto-update is explicitly blocked: any attempt to *load* a manifest
without a valid signature raises :exc:`UpdateRejectedError` immediately.

Public API summary
------------------
- :class:`ManifestEntry`     — one rule/model artefact in the package
- :class:`UpdateManifest`    — the full signed package
- :class:`UpdateSigner`      — signs manifests with the local key
- :class:`UpdateVerifier`    — verifies manifests; raises on any failure
- :class:`UpdateRejectedError` — raised for any integrity / auth failure
- :func:`make_manifest`      — helper to build a manifest with correct hashes
- :func:`generate_update_key` — generate and persist a fresh 256-bit key
"""
from __future__ import annotations

import copy
import hashlib
import hmac as _hmac
import json
import os
import pathlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ── Constants ─────────────────────────────────────────────────────────────────

#: The only accepted signature algorithm identifier.
_ALGORITHM = "hmac-sha256"

#: Minimum acceptable HMAC key size in bytes (256-bit).
_MIN_KEY_BYTES = 32

#: Current manifest schema version.
_SCHEMA_VERSION = "1"

#: Default path for the control-plane update key.
DEFAULT_KEY_PATH: pathlib.Path = (
    pathlib.Path.home() / ".config" / "piiguard" / "keys" / "update.key"
)

#: Accepted manifest ``kind`` values.
ALLOWED_KINDS = frozenset({"rule_update", "model_update", "mixed_update"})

#: Accepted entry ``kind`` values.
ALLOWED_ENTRY_KINDS = frozenset({"regex_rule", "ner_model", "category_spec", "golden_corpus"})


# ── Exceptions ────────────────────────────────────────────────────────────────

class UpdateRejectedError(Exception):
    """
    Raised when an update manifest fails any part of the verification pipeline.

    This covers:
    - Missing or malformed signature field
    - Unsupported signing algorithm
    - HMAC verification failure (wrong key, tampered payload)
    - Per-entry content hash mismatch (tampered entry content)
    - Structural validation failures (missing required fields, bad schema version)

    Callers must catch this to distinguish a rejected update from unexpected
    I/O or JSON errors.
    """


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ManifestEntry:
    """
    A single artefact in an :class:`UpdateManifest`.

    Attributes
    ----------
    name:
        Logical name for this entry (e.g. ``"email_rule"``, ``"ner_model_v2"``).
    kind:
        Entry type — one of ``regex_rule``, ``ner_model``, ``category_spec``,
        ``golden_corpus``.
    sha256:
        Lowercase hex SHA-256 digest of ``content`` encoded as UTF-8.  Computed
        by :func:`make_manifest` and verified by :meth:`UpdateVerifier.verify`.
    content:
        The raw content string (regex source, model config YAML, base64-encoded
        binary, etc.).  Never persisted to the Ledger.
    metadata:
        Optional free-form dict of metadata (description, target_version, etc.).
        Included in the canonical payload and therefore covered by the HMAC.
    """

    name: str
    kind: str  # one of ALLOWED_ENTRY_KINDS
    sha256: str  # hex digest of content.encode("utf-8")
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def _content_bytes(self) -> bytes:
        return self.content.encode("utf-8")

    def compute_sha256(self) -> str:
        """Return the hex SHA-256 of the content bytes (for verification/generation)."""
        return hashlib.sha256(self._content_bytes()).hexdigest()


@dataclass
class UpdateManifest:
    """
    A signed update package for PII-Guard detection rules or NER models.

    The manifest is serialised to JSON for storage and transport.  The
    ``signature`` field is excluded from the payload before signing so that
    the signing round-trip is deterministic.

    Attributes
    ----------
    manifest_version:
        Semantic version of this manifest (e.g. ``"1.0.0"``).
    kind:
        Package kind — one of ``rule_update``, ``model_update``, ``mixed_update``.
    timestamp:
        ISO-8601 UTC timestamp of when the manifest was created.
    author:
        Descriptive author / publisher identifier (not authenticated).
    signer_id:
        Opaque key identifier for the signing key (e.g. ``"piiguard-local-key-v1"``).
        Used for auditing; not used in cryptographic verification.
    entries:
        Ordered list of :class:`ManifestEntry` objects.
    signature:
        Populated by :class:`UpdateSigner`.  Format: ``"hmac-sha256:<hex>"``.
        ``None`` means the manifest has not been signed yet.
    schema_version:
        Schema version of this format (currently ``"1"``).  Checked during load.
    """

    manifest_version: str
    kind: str
    timestamp: str
    author: str
    signer_id: str
    entries: List[ManifestEntry]
    signature: Optional[str] = None
    schema_version: str = _SCHEMA_VERSION

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _canonical_payload(self) -> bytes:
        """
        Return the canonical JSON bytes used as the HMAC input.

        All fields *except* ``signature`` are serialised with sorted keys and
        no extra whitespace.  Entry metadata dicts are also sorted.  This
        ensures the HMAC is deterministic regardless of field insertion order.
        """
        d: dict = {
            "schema_version": self.schema_version,
            "manifest_version": self.manifest_version,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "author": self.author,
            "signer_id": self.signer_id,
            "entries": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "sha256": e.sha256,
                    "content": e.content,
                    "metadata": dict(sorted(e.metadata.items())),
                }
                for e in self.entries
            ],
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for ``json.dumps``)."""
        d: dict = {
            "schema_version": self.schema_version,
            "manifest_version": self.manifest_version,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "author": self.author,
            "signer_id": self.signer_id,
            "entries": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "sha256": e.sha256,
                    "content": e.content,
                    "metadata": dict(e.metadata),
                }
                for e in self.entries
            ],
        }
        if self.signature is not None:
            d["signature"] = self.signature
        return d

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "UpdateManifest":
        """
        Deserialise from a plain dict.

        Raises
        ------
        UpdateRejectedError
            If required fields are missing or ``schema_version`` is unrecognised.
        """
        try:
            schema_ver = d.get("schema_version", "1")
            if schema_ver != _SCHEMA_VERSION:
                raise UpdateRejectedError(
                    f"Unsupported manifest schema_version: {schema_ver!r} "
                    f"(expected {_SCHEMA_VERSION!r})"
                )
            entries = [
                ManifestEntry(
                    name=str(e["name"]),
                    kind=str(e["kind"]),
                    sha256=str(e["sha256"]),
                    content=str(e["content"]),
                    metadata=dict(e.get("metadata", {})),
                )
                for e in d["entries"]
            ]
            return cls(
                manifest_version=str(d["manifest_version"]),
                kind=str(d["kind"]),
                timestamp=str(d["timestamp"]),
                author=str(d["author"]),
                signer_id=str(d.get("signer_id", "")),
                entries=entries,
                signature=d.get("signature"),
                schema_version=schema_ver,
            )
        except (KeyError, TypeError) as exc:
            raise UpdateRejectedError(
                f"Invalid manifest structure — required field missing or wrong type: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, text: str) -> "UpdateManifest":
        """
        Parse and deserialise from a JSON string.

        Raises
        ------
        UpdateRejectedError
            If the JSON is malformed or has structural errors.
        """
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise UpdateRejectedError(f"Manifest JSON parse error: {exc}") from exc
        return cls.from_dict(d)


# ── Signer ────────────────────────────────────────────────────────────────────

class UpdateSigner:
    """
    Signs :class:`UpdateManifest` objects using HMAC-SHA256.

    The signer holds the **private** side of the update channel.  In a
    local-first deployment, ``signer`` and ``verifier`` share the same key
    file, so "private" here means "stored in the control-plane directory
    outside the protected agent's write path."

    Parameters
    ----------
    key:
        Raw HMAC key bytes.  Must be at least 32 bytes (256-bit).
    signer_id:
        Opaque key identifier embedded in signed manifests for auditing.

    Examples
    --------
    >>> key = UpdateSigner.generate_key()
    >>> signer = UpdateSigner(key)
    >>> manifest = make_manifest("1.0.0", "rule_update", [("my_rule", "regex_rule", "pattern")])
    >>> signed = signer.sign(manifest)
    >>> signed.signature.startswith("hmac-sha256:")
    True
    """

    def __init__(
        self,
        key: bytes,
        signer_id: str = "piiguard-local-key-v1",
    ) -> None:
        if len(key) < _MIN_KEY_BYTES:
            raise ValueError(
                f"HMAC key must be at least {_MIN_KEY_BYTES} bytes, got {len(key)}"
            )
        self._key = key
        self.signer_id = signer_id

    @staticmethod
    def generate_key() -> bytes:
        """Generate a cryptographically random 256-bit HMAC key."""
        return secrets.token_bytes(_MIN_KEY_BYTES)

    @classmethod
    def from_key_file(
        cls,
        path: pathlib.Path = DEFAULT_KEY_PATH,
        signer_id: str = "piiguard-local-key-v1",
    ) -> "UpdateSigner":
        """Load the signing key from *path*."""
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Update key not found: {p}")
        key = p.read_bytes()
        if len(key) < _MIN_KEY_BYTES:
            raise UpdateRejectedError(
                f"Key file {p} is too short ({len(key)} bytes, need ≥{_MIN_KEY_BYTES})"
            )
        return cls(key, signer_id=signer_id)

    def _compute_mac(self, manifest: UpdateManifest) -> str:
        """Return the HMAC-SHA256 hex digest of the manifest's canonical payload."""
        payload = manifest._canonical_payload()
        return _hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def sign(self, manifest: UpdateManifest) -> UpdateManifest:
        """
        Sign *manifest* and return a **new** manifest with ``signature`` set.

        The original manifest is not mutated.  The ``signer_id`` on the
        returned manifest is set to this signer's ``signer_id``.

        Parameters
        ----------
        manifest:
            The manifest to sign.  May already have a ``signature`` value
            (it will be overwritten).

        Returns
        -------
        UpdateManifest
            A shallow copy of *manifest* with ``signature`` populated.
        """
        # Work on a copy so we don't mutate the caller's object
        m = copy.copy(manifest)
        m.signer_id = self.signer_id
        # Clear any existing signature before computing (canonical payload excludes it)
        m.signature = None
        mac_hex = self._compute_mac(m)
        m.signature = f"{_ALGORITHM}:{mac_hex}"
        return m


# ── Verifier ──────────────────────────────────────────────────────────────────

class UpdateVerifier:
    """
    Verifies :class:`UpdateManifest` objects — raises on any integrity failure.

    Verification pipeline (in order)
    ---------------------------------
    1. **Signature present** — manifests with no ``signature`` field are
       immediately rejected (silent auto-update is blocked).
    2. **Algorithm check** — only ``hmac-sha256`` is accepted; any other value
       in the ``"<alg>:<hex>"`` prefix raises immediately.
    3. **HMAC** — recomputed over the canonical payload and compared in
       constant time (:func:`hmac.compare_digest`).
    4. **Per-entry content hashes** — each entry's ``sha256`` is recomputed
       over ``content.encode("utf-8")`` and compared in constant time.

    All failures raise :exc:`UpdateRejectedError` with a descriptive message
    that does not expose the actual HMAC key or content.

    Parameters
    ----------
    key:
        Raw HMAC key bytes.  Must be at least 32 bytes.

    Examples
    --------
    >>> verifier = UpdateVerifier(key)
    >>> verifier.verify(signed_manifest)   # passes silently
    >>> verifier.verify(unsigned_manifest)  # raises UpdateRejectedError
    """

    def __init__(self, key: bytes) -> None:
        if len(key) < _MIN_KEY_BYTES:
            raise ValueError(
                f"HMAC key must be at least {_MIN_KEY_BYTES} bytes, got {len(key)}"
            )
        self._key = key

    @classmethod
    def from_key_file(cls, path: pathlib.Path = DEFAULT_KEY_PATH) -> "UpdateVerifier":
        """Load the verification key from *path*."""
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Update key not found: {p}")
        key = p.read_bytes()
        if len(key) < _MIN_KEY_BYTES:
            raise UpdateRejectedError(
                f"Key file {p} is too short ({len(key)} bytes, need ≥{_MIN_KEY_BYTES})"
            )
        return cls(key)

    def verify(self, manifest: UpdateManifest) -> None:
        """
        Verify *manifest* signature and all entry content hashes.

        Raises
        ------
        UpdateRejectedError
            On any verification failure.  The exception message describes
            *what* failed without revealing the key or raw content.

        Notes
        -----
        - Both the HMAC and entry hash comparisons use :func:`hmac.compare_digest`
          (constant-time) to avoid timing side-channels.
        - The method is intentionally fail-fast: it raises as soon as the
          first check fails rather than accumulating errors.
        """
        # ── 1. Signature present ──────────────────────────────────────────────
        if not manifest.signature:
            raise UpdateRejectedError(
                "Manifest has no signature — unsigned updates are not allowed. "
                "Sign the manifest with UpdateSigner.sign() before distributing."
            )

        # ── 2. Algorithm check ────────────────────────────────────────────────
        parts = manifest.signature.split(":", 1)
        if len(parts) != 2:
            raise UpdateRejectedError(
                f"Malformed signature field (expected '<alg>:<hex>'): "
                f"{manifest.signature!r}"
            )
        alg, provided_hex = parts
        if alg != _ALGORITHM:
            raise UpdateRejectedError(
                f"Unsupported signature algorithm: {alg!r} — "
                f"only {_ALGORITHM!r} is accepted"
            )

        # Validate hex string to avoid subtle compare failures
        try:
            bytes.fromhex(provided_hex)
        except ValueError:
            raise UpdateRejectedError(
                "Signature hex string is invalid (non-hex characters detected)"
            )

        # ── 3. HMAC verification ──────────────────────────────────────────────
        # Build a temp manifest without the signature to compute canonical payload
        m_no_sig = copy.copy(manifest)
        m_no_sig.signature = None
        payload = m_no_sig._canonical_payload()
        expected_hex = _hmac.new(self._key, payload, hashlib.sha256).hexdigest()

        if not _hmac.compare_digest(expected_hex, provided_hex.lower()):
            raise UpdateRejectedError(
                "HMAC-SHA256 signature verification failed — manifest rejected. "
                "The manifest may have been tampered with, or the wrong key is in use."
            )

        # ── 4. Per-entry content hash verification ────────────────────────────
        for entry in manifest.entries:
            actual_hex = hashlib.sha256(entry._content_bytes()).hexdigest()
            if not _hmac.compare_digest(actual_hex, entry.sha256.lower()):
                raise UpdateRejectedError(
                    f"Content hash mismatch for entry {entry.name!r} "
                    f"(kind={entry.kind!r}) — entry content has been tampered with. "
                    f"Manifest rejected."
                )

    def load_verified(self, path: pathlib.Path) -> UpdateManifest:
        """
        Load a manifest JSON file from *path*, verify it, and return it.

        Parameters
        ----------
        path:
            Path to the ``*.piiguard-manifest.json`` file.

        Returns
        -------
        UpdateManifest
            The verified manifest (safe to apply).

        Raises
        ------
        UpdateRejectedError
            If the file cannot be parsed, has structural errors, or fails
            any verification step.
        OSError
            If the file cannot be read.
        """
        raw = pathlib.Path(path).read_text(encoding="utf-8")
        manifest = UpdateManifest.from_json(raw)
        self.verify(manifest)
        return manifest


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_manifest(
    manifest_version: str,
    kind: str,
    entries: Sequence[tuple],
    *,
    timestamp: Optional[str] = None,
    author: str = "piiguard-local",
    signer_id: str = "piiguard-local-key-v1",
) -> UpdateManifest:
    """
    Convenience factory: build an :class:`UpdateManifest` with per-entry
    SHA-256 hashes automatically computed.

    Parameters
    ----------
    manifest_version:
        Semantic version string for this manifest (e.g. ``"1.0.0"``).
    kind:
        Package kind (``rule_update``, ``model_update``, or ``mixed_update``).
    entries:
        An iterable of 2- or 3-tuples:
        ``(name, entry_kind, content)`` or
        ``(name, entry_kind, content, metadata_dict)``.
    timestamp:
        ISO-8601 UTC timestamp.  Defaults to the current UTC time.
    author:
        Publisher identifier.
    signer_id:
        Key identifier embedded in the manifest.

    Returns
    -------
    UpdateManifest
        A **unsigned** manifest (call :meth:`UpdateSigner.sign` next).

    Examples
    --------
    >>> m = make_manifest("1.0.0", "rule_update", [
    ...     ("email_rule", "regex_rule", r"[a-z]+@[a-z]+\\.com"),
    ... ])
    >>> signed = signer.sign(m)
    """
    if timestamp is None:
        import datetime
        timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    built_entries: List[ManifestEntry] = []
    for item in entries:
        if len(item) == 3:
            name, entry_kind, content = item
            metadata: dict = {}
        elif len(item) == 4:
            name, entry_kind, content, metadata = item
        else:
            raise ValueError(
                f"Each entry must be a 2–4-tuple (name, kind, content[, metadata]), "
                f"got {len(item)}-tuple"
            )
        content_str = str(content)
        sha256_hex = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
        built_entries.append(
            ManifestEntry(
                name=str(name),
                kind=str(entry_kind),
                sha256=sha256_hex,
                content=content_str,
                metadata=dict(metadata),
            )
        )

    return UpdateManifest(
        manifest_version=manifest_version,
        kind=kind,
        timestamp=timestamp,
        author=author,
        signer_id=signer_id,
        entries=built_entries,
    )


def generate_update_key(
    path: pathlib.Path = DEFAULT_KEY_PATH,
    *,
    overwrite: bool = False,
) -> bytes:
    """
    Generate a new 256-bit HMAC key and persist it to *path*.

    The key file is created with permissions 0o600; its parent directory is
    created with permissions 0o700 if it does not exist.

    Parameters
    ----------
    path:
        Destination path for the key file.
    overwrite:
        If ``False`` (default) and *path* already exists, raise
        :exc:`FileExistsError`.  Set to ``True`` to replace an existing key.

    Returns
    -------
    bytes
        The generated key bytes (also written to *path*).

    Raises
    ------
    FileExistsError
        If *path* already exists and *overwrite* is ``False``.
    """
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"Update key already exists at {p}. "
            "Pass overwrite=True to replace it (this will invalidate all existing signatures)."
        )
    key = secrets.token_bytes(_MIN_KEY_BYTES)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(key)
    p.chmod(0o600)
    # tighten parent dir permissions
    try:
        p.parent.chmod(0o700)
    except OSError:
        pass  # best-effort (may fail on read-only parent mounts)
    return key
