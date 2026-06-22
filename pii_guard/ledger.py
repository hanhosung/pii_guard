"""
PII-Guard append-only Ledger — Sub-AC 4.1 / 4.2.1.

Records block/mask/fail/coverage-gap events as metadata fields plus HMAC-keyed
hashes of any sensitive values.  No recoverable original PII/secret values are
ever persisted to disk.

Design goals
------------
* **NoNewVault**: raw originals are never written; only HMAC-keyed hashes
  that allow cross-request correlation without recovery.
* **Permissions**: ledger file created with mode 0o600 (rw-------); its parent
  directory with mode 0o700 (rwx------) on first write.
* **Append-only JSONL**: one JSON object per line for easy streaming/grep.
* **Rotation**: size-based (default 10 MB) *and* time-based (default off);
  archives are numbered ``.1`` → ``.N`` via the standard shifting scheme.
* **Retention**: keeps at most ``backup_count`` rotated files; older ones
  are deleted at rotation time.
* **Purge**: explicit :meth:`Ledger.purge` deletes current + all rotated files.
* **Thread-safe**: a ``threading.Lock`` serialises all writes and rotations.

Usage::

    from pii_guard.ledger import Ledger

    key = os.urandom(32)                        # one key per process lifetime
    ledger = Ledger("/var/piiguard/ledger.jsonl", key)

    # Block event (from a Detection)
    ledger.record_block(detection, channel="cli/codex", scan_field="tool_result")

    # Mask event
    ledger.record_mask(detection, channel="ouroboros", scan_field="message_text")

    # Scan / infrastructure failure
    ledger.record_fail("Stage2 NER timeout after 5s", scan_field="system_prompt")

    # Content passed unscanned
    ledger.record_coverage_gap(reason="image/png — unscannable", scan_field="image")

    # Explicit purge (wipes all ledger files)
    ledger.purge()
"""
from __future__ import annotations

import datetime
import hashlib
import hmac as _hmac_mod
import json
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .models import Detection


# ─────────────────────────────────────────────────────────────────────────────
# Event type enum
# ─────────────────────────────────────────────────────────────────────────────

class LedgerEventType(str, Enum):
    """Four audit event types recorded by the Ledger."""
    BLOCK = "block"
    MASK = "mask"
    FAIL = "fail"
    COVERAGE_GAP = "coverage_gap"


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Default maximum ledger file size before rotation (10 MB).
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024

#: Default number of rotated backup files to retain.
DEFAULT_BACKUP_COUNT: int = 7

#: Default maximum age (in seconds) of the active ledger before time-based
#: rotation.  0.0 means disabled.  Example: 86400.0 = rotate after 24 h.
DEFAULT_MAX_AGE_SECONDS: float = 0.0

#: Default retention window (in seconds) for *archived* ledger files.
#: When a rotated backup's mtime is strictly older than this value, it is
#: automatically removed.  0.0 means disabled — archives are never
#: age-pruned (only the count-based ``backup_count`` limit applies).
#: Example: 2_592_000.0 = 30 days.
DEFAULT_RETENTION_WINDOW_SECONDS: float = 0.0

#: octal permission for the ledger file (owner read/write only).
_FILE_MODE: int = 0o600

#: octal permission for the ledger directory (owner read/write/exec only).
_DIR_MODE: int = 0o700


# ─────────────────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────────────────

class Ledger:
    """
    Append-only audit ledger that records PII-Guard events without persisting
    recoverable PII or secret values.

    Parameters
    ----------
    path:
        Absolute path to the ledger file (e.g. ``/var/piiguard/ledger.jsonl``).
        The parent directory is created on first write with 0o700 permissions.
    hmac_key:
        32-byte secret key used for HMAC-SHA256 correlation hashes.  This key
        is NEVER written to disk.  Use a key generated at process startup
        (``os.urandom(32)``).  Different processes using the same key can
        correlate the same value across audit entries without recovering the
        original.
    max_bytes:
        Rotate the ledger file when it exceeds this size (bytes).
        Default: 10 MB.  Set to 0 to disable size-based rotation.
    backup_count:
        Maximum number of rotated files to retain.  Older files are deleted
        at rotation time.  Default: 7.
    max_age_seconds:
        Rotate the ledger file when the active file is older than this many
        seconds.  Default: 0.0 (disabled).  Example: ``86400.0`` rotates
        once per day.  On first write the file's mtime is used as its
        creation time, so a ledger that survived a restart will respect any
        remaining age budget rather than immediately rotating.

        Both ``max_bytes`` and ``max_age_seconds`` can be active at the same
        time; whichever threshold is reached first triggers the rotation.

        Archived files follow the standard ``.N`` numbering scheme (same as
        size-based rotation) and are created with 0o600 permissions.
    retention_window_seconds:
        Maximum age (in seconds) that a *rotated archive* may retain before
        being automatically deleted.  0.0 (default) disables age-based
        retention — archives accumulate until ``backup_count`` forces removal.
        Example: ``2_592_000.0`` (30 days) deletes archives older than 30 days.

        Retention is enforced after every rotation and can also be triggered
        explicitly via :meth:`apply_retention`.  Only archives with an mtime
        **strictly older** than ``time.time() - retention_window_seconds`` are
        removed; archives exactly at the boundary are kept.
    """

    def __init__(
        self,
        path: Union[str, Path],
        hmac_key: bytes,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        retention_window_seconds: float = DEFAULT_RETENTION_WINDOW_SECONDS,
    ) -> None:
        if not isinstance(hmac_key, (bytes, bytearray)):
            raise TypeError(f"hmac_key must be bytes, got {type(hmac_key).__name__}")
        if len(hmac_key) < 16:
            raise ValueError("hmac_key must be at least 16 bytes")

        self._path = Path(path)
        self._hmac_key: bytes = bytes(hmac_key)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._max_age_seconds = float(max_age_seconds)
        self._retention_window_seconds = float(retention_window_seconds)
        self._lock = threading.Lock()
        self._initialized = False
        # Monotonic wall-clock second at which the current active ledger file
        # was created (or the mtime of a pre-existing file).  Used by the
        # time-based rotation trigger.  None until _ensure_initialized() runs.
        self._file_created_at: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Absolute path to the current ledger file."""
        return self._path

    def record_block(
        self,
        detection: Detection,
        *,
        channel: Optional[str] = None,
        scan_field: Optional[str] = None,
    ) -> None:
        """
        Record a **block** event — a high-risk detection that caused the
        request to be blocked.

        Parameters
        ----------
        detection:
            The :class:`~pii_guard.models.Detection` that triggered the block.
        channel:
            Originating channel (e.g. ``"cli/codex"``, ``"ouroboros"``).
        scan_field:
            Payload field that was scanned (e.g. ``"tool_result"``).
        """
        entry = self._detection_entry(
            LedgerEventType.BLOCK, detection,
            channel=channel, scan_field=scan_field,
        )
        self._write_entry(entry)

    def record_mask(
        self,
        detection: Detection,
        *,
        channel: Optional[str] = None,
        scan_field: Optional[str] = None,
    ) -> None:
        """
        Record a **mask** event — contact/context PII that was tokenised and
        will be rehydrated on the inbound response.

        Parameters
        ----------
        detection:
            The :class:`~pii_guard.models.Detection` that was masked.
        channel:
            Originating channel.
        scan_field:
            Payload field that was scanned.
        """
        entry = self._detection_entry(
            LedgerEventType.MASK, detection,
            channel=channel, scan_field=scan_field,
        )
        self._write_entry(entry)

    def record_fail(
        self,
        fail_reason: str,
        *,
        category: Optional[str] = None,
        channel: Optional[str] = None,
        scan_field: Optional[str] = None,
    ) -> None:
        """
        Record a **fail** event — the scanner or infrastructure failed to
        process a content block.

        The ``fail_reason`` must contain only error metadata (exception type,
        timing information) and MUST NOT contain any raw PII or secret text.

        Parameters
        ----------
        fail_reason:
            Human-readable reason for the failure (e.g. ``"Stage2 NER OOM"``).
            Must not contain raw PII.
        category:
            Category that was being scanned when failure occurred (optional).
        channel:
            Originating channel.
        scan_field:
            Payload field being scanned when failure occurred.
        """
        entry: dict = {
            "event_type": LedgerEventType.FAIL.value,
            "timestamp": self._now(),
            "category": category,
            "fail_reason": fail_reason,
            "channel": channel,
            "scan_field": scan_field,
        }
        self._write_entry(entry)

    def record_coverage_gap(
        self,
        *,
        reason: Optional[str] = None,
        channel: Optional[str] = None,
        scan_field: Optional[str] = None,
    ) -> None:
        """
        Record a **coverage_gap** event — content passed without full scanning.

        This is emitted when Stage-2 NER degrades to Stage-1 only, when an
        image/binary block is encountered, or when any other condition causes
        content to bypass the full detection pipeline.

        Parameters
        ----------
        reason:
            Short human-readable description of why the gap occurred (e.g.
            ``"Stage2 NER timeout"`` or ``"image/png — unscannable"``).
            Must not contain raw PII.
        channel:
            Originating channel.
        scan_field:
            Payload field with the coverage gap.
        """
        entry: dict = {
            "event_type": LedgerEventType.COVERAGE_GAP.value,
            "timestamp": self._now(),
            "reason": reason,
            "channel": channel,
            "scan_field": scan_field,
        }
        self._write_entry(entry)

    def rotate(self) -> None:
        """
        Force an immediate rotation of the current ledger file.

        After rotation the old file is renamed to ``<path>.1`` and a new
        empty file is created at ``<path>`` with 0o600 permissions.
        Excess backup files beyond ``backup_count`` are deleted.
        """
        with self._lock:
            self._rotate_locked()

    def purge(self) -> None:
        """
        Delete all ledger files — the current file and all rotated backups —
        and fully reset in-process ledger state.

        This is a destructive operation.  After purge:

        * No ledger files remain on disk (current file and all rotated
          archives ``.1`` … ``.N`` are deleted).
        * In-process state is fully reset: ``_initialized = False`` and
          ``_file_created_at = None``.
        * The next :meth:`record_*` (or :meth:`initialize`) call will
          re-create the ledger file with correct 0o600 / 0o700 permissions.

        Idempotent: calling purge on a ledger that has never been written
        (or has already been purged) is safe and does not raise.
        """
        with self._lock:
            # Delete current file
            if self._path.exists():
                self._path.unlink()
            # Delete rotated backups — scan up to backup_count + a buffer
            # so that archives created when backup_count was larger are also
            # removed.  Stop at the first missing index (contiguous numbering
            # invariant means nothing can exist beyond the first gap).
            for i in range(1, self._backup_count + 100):
                backup = Path(f"{self._path}.{i}")
                if backup.exists():
                    backup.unlink()
                else:
                    break
            # Reset ALL in-process ledger state
            self._initialized = False
            self._file_created_at = None  # reset age timer

    def initialize(self) -> None:
        """
        Ensure the parent directory (0o700) and an empty ledger file (0o600)
        exist without writing any event entries.

        Idempotent: if the file already exists its permissions are enforced
        but no content is altered.

        This is typically called after :meth:`purge` to produce a known-clean
        seed file with the correct ownership and permissions before the first
        event is recorded — allowing the operator to confirm the ledger is
        ready without waiting for the first real event.

        Example::

            ledger.purge()      # wipe all files + reset state
            ledger.initialize() # create fresh 0o600 seed file at ledger.path
        """
        with self._lock:
            self._ensure_initialized()

    def apply_retention(self) -> int:
        """
        Scan all rotated archive files and delete any whose mtime is strictly
        older than ``retention_window_seconds``.

        This is called automatically after every rotation, but can also be
        invoked explicitly (e.g. from a maintenance cron) to enforce the
        policy on an already-running ledger without waiting for the next
        rotation.

        Returns
        -------
        int
            The number of archive files deleted.  Returns 0 when
            ``retention_window_seconds`` is 0.0 (disabled) or when no
            archives exist or all are within the window.

        Notes
        -----
        Archives are numbered ``.1`` (most recent) through ``.N`` (oldest).
        Because the rotation scheme always produces contiguous numbering,
        scanning stops at the first missing index.  Only files whose mtime
        is **strictly less than** ``time.time() - retention_window_seconds``
        are removed; files at or newer than the boundary are kept.
        """
        with self._lock:
            return self._apply_retention_locked()

    # ── Internal ────────────────────────────────────────────────────────────

    def _detection_entry(
        self,
        event_type: LedgerEventType,
        detection: Detection,
        *,
        channel: Optional[str],
        scan_field: Optional[str],
    ) -> dict:
        """Build a ledger entry dict from a Detection — no raw original stored."""
        return {
            "event_type": event_type.value,
            "timestamp": self._now(),
            "category": detection.category,
            "category_class": detection.category_class.value,
            "count": 1,
            "action": detection.action.value,
            "detector_id": detection.rule_id,
            "rule": detection.rule_id,
            "confidence": detection.confidence,
            "detection_stage": detection.detection_stage.value,
            "span_length": detection.span_length(),
            "char_class_signature": detection.char_class_signature(),
            # HMAC-keyed hash for correlation — non-reversible, no original stored
            "keyed_hash": self._keyed_hash(detection.original),
            "channel": channel,
            "scan_field": scan_field,
        }

    def _keyed_hash(self, value: str) -> str:
        """
        HMAC-SHA256 of the normalised value, keyed with this ledger's secret
        key.  Same input + same key → same output (deterministic correlation).
        Different key → different output (non-recoverable without the key).
        """
        normalised = value.strip().lower()
        return _hmac_mod.new(
            self._hmac_key,
            normalised.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _now(self) -> str:
        """Current UTC timestamp as ISO 8601 string (e.g. ``2026-01-01T12:00:00Z``)."""
        return datetime.datetime.utcnow().isoformat() + "Z"

    def _ensure_initialized(self) -> None:
        """
        On first call: create the parent directory with 0o700 and the ledger
        file with 0o600.  Idempotent — subsequent calls return immediately.

        Also records ``_file_created_at`` for the time-based rotation trigger:
        * New file  → current wall-clock time.
        * Existing file → the file's mtime (so surviving restarts do not
          immediately rotate a file that is still within its age budget).

        Must be called while holding ``self._lock``.
        """
        if self._initialized:
            return

        parent = self._path.parent

        # Create parent directory — enforce 0o700 regardless of umask
        parent.mkdir(parents=True, exist_ok=True)
        os.chmod(parent, _DIR_MODE)

        # Create or fix the ledger file permissions
        if not self._path.exists():
            # Create empty file — use os.open for atomic permission setting
            fd = os.open(str(self._path), os.O_CREAT | os.O_WRONLY, _FILE_MODE)
            os.close(fd)
            self._file_created_at = time.time()
        else:
            # Seed the age timer from the existing file's mtime so the
            # remaining age budget is respected across process restarts.
            try:
                self._file_created_at = self._path.stat().st_mtime
            except OSError:
                self._file_created_at = time.time()

        # Always enforce mode (existing files may have wrong perms)
        os.chmod(self._path, _FILE_MODE)

        self._initialized = True

    def _write_entry(self, entry: dict) -> None:
        """
        Serialise *entry* to a JSONL line and append it to the ledger file.

        Thread-safe; checks both the size and time thresholds and rotates
        before writing if either is exceeded.
        """
        with self._lock:
            self._ensure_initialized()

            # Size-based rotation check
            if self._max_bytes > 0 and self._path.exists():
                try:
                    if self._path.stat().st_size >= self._max_bytes:
                        self._rotate_locked()
                except OSError:
                    pass  # best-effort; rotation failure does not drop the entry

            # Time-based rotation check — only when not already rotated above
            if self._max_age_seconds > 0.0 and self._file_created_at is not None:
                try:
                    age = time.time() - self._file_created_at
                    if age >= self._max_age_seconds:
                        self._rotate_locked()
                except OSError:
                    pass  # best-effort

            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def _rotate_locked(self) -> None:
        """
        Rotate the current ledger file.  Must be called while holding
        ``self._lock``.

        Algorithm
        ---------
        1. Shift existing backups: ``.N-1`` → ``.N``, …, ``.1`` → ``.2``
           (files beyond ``backup_count`` are deleted).
        2. Move current file → ``.1``.
        3. Create a new empty current file with 0o600.
        """
        # Purge files beyond backup_count
        oldest = Path(f"{self._path}.{self._backup_count}")
        if oldest.exists():
            oldest.unlink()

        # Shift existing backups upward
        for i in range(self._backup_count - 1, 0, -1):
            src = Path(f"{self._path}.{i}")
            dst = Path(f"{self._path}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Move current → .1
        if self._path.exists():
            self._path.rename(Path(f"{self._path}.1"))

        # Create fresh current file
        fd = os.open(str(self._path), os.O_CREAT | os.O_WRONLY, _FILE_MODE)
        os.close(fd)
        os.chmod(self._path, _FILE_MODE)
        # Reset the time-based age timer for the new active file.
        self._file_created_at = time.time()

        # Apply age-based retention: delete archives older than the window.
        self._apply_retention_locked()

    def _apply_retention_locked(self) -> int:
        """
        Scan rotated archive files and delete those strictly older than
        ``retention_window_seconds``.  Must be called while holding
        ``self._lock``.

        Archives are numbered ``.1`` (most recent) to ``.N`` (oldest).  The
        rotation scheme guarantees contiguous numbering, so scanning stops at
        the first missing index — any higher-numbered archive that might exist
        beyond a gap would itself be a candidate for deletion on the next pass.

        Only archives whose mtime is **strictly less than**
        ``time.time() - retention_window_seconds`` are removed; files at or
        newer than the cutoff are kept ("strictly beyond the window").

        Returns
        -------
        int
            Number of archive files deleted.
        """
        if self._retention_window_seconds <= 0.0:
            return 0

        cutoff = time.time() - self._retention_window_seconds
        deleted = 0

        # Upper bound: scan at most backup_count + a generous buffer so that
        # archives not yet expired but beyond the nominal backup_count are also
        # covered (e.g. when backup_count was recently lowered).
        max_scan = max(self._backup_count, 1) + 200
        for i in range(1, max_scan + 1):
            archive = Path(f"{self._path}.{i}")
            if not archive.exists():
                break  # contiguous sequence ends here
            try:
                mtime = archive.stat().st_mtime
                if mtime < cutoff:  # strictly older → delete
                    archive.unlink()
                    deleted += 1
            except OSError:
                pass  # best-effort; continue to next index

        return deleted
