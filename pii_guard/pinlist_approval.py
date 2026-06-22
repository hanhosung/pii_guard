"""
PII-Guard pin-list out-of-band approval flow (Sub-AC 5d-ii).

The approval gate is the **sole write path** for pin-list mutations.
It presents proposed changes to the user, requires explicit confirmation,
and commits or discards atomically:

* ``approve()``  — writes the new pin-list + ``pin_list_approved: true`` to
  the policy YAML atomically (write-to-tmp + ``os.replace``).
* ``reject()``   — discards the staged change with **zero file I/O**.

Protection against direct mutations
-------------------------------------
The :class:`~pii_guard.policy.PolicyLoader` (Sub-AC 5d-i) blocks any
file-level pin-list change where ``pin_list_approved`` is not ``true``.
The approval gate is the **only** code path that legitimately sets
``pin_list_approved: true`` alongside a pin-list change.

Any attempt to bypass the gate by writing to the policy YAML directly
without ``pin_list_approved: true`` is caught and rejected by
``PolicyLoader`` on the next hot-reload cycle.

Usage (programmatic / test)::

    gate = PinListApprovalGate(policy_path)
    gate.propose(new_entries)
    result = gate.approve()           # or gate.reject()
    assert result.committed           # True → written to disk

Usage (interactive CLI)::

    result = run_interactive_approval(
        policy_path, new_entries,
        input_fn=input,   # replace with test hook for non-interactive tests
        output_fn=print,
    )

State machine::

    idle ──propose()──▶ staged ──approve()──▶ committed
                          │
                          └──reject()──▶ rejected
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .policy import PinListEntry

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApprovalResult:
    """
    Result of an interactive pin-list approval flow.

    Attributes
    ----------
    approved:
        ``True`` if the user (or automated test hook) confirmed the change.
    committed:
        ``True`` if the change was actually written to the policy file.
        Always ``False`` when ``approved`` is ``False``.
    entries_added:
        Hash strings of pin-list entries that were added by this change.
    entries_removed:
        Hash strings of pin-list entries that were removed by this change.
    discarded_reason:
        Human-readable reason the change was discarded (``None`` when
        ``approved`` is ``True``).
    """

    approved: bool
    committed: bool
    entries_added: List[str] = field(default_factory=list)
    entries_removed: List[str] = field(default_factory=list)
    discarded_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal YAML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml() -> Any:
    """Import and return the PyYAML module, raising ImportError with a hint."""
    try:
        import yaml  # type: ignore[import]
        return yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for policy management. "
            "Install with: pip install pyyaml"
        ) from exc


def _read_policy_raw(path: Path) -> dict:
    """
    Read and parse the policy YAML at *path*.

    Returns an empty dict if the file does not exist, is empty, or is not
    a mapping.  Never raises — all errors are swallowed and logged.
    """
    if not path.exists():
        return {}
    try:
        yaml = _load_yaml()
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
        return {}
    except Exception as exc:  # pragma: no cover
        log.warning("PinListApprovalGate: could not read %s: %s", path, exc)
        return {}


def _entries_to_yaml_list(entries: List[PinListEntry]) -> List[Dict[str, str]]:
    """Convert ``PinListEntry`` objects to plain dicts for YAML serialisation."""
    result = []
    for e in entries:
        d: Dict[str, str] = {
            "hash": e.hash,
            "category": e.category,
            "action": e.action,
        }
        if e.label:
            d["label"] = e.label
        result.append(d)
    return result


def _extract_pin_list(data: dict) -> List[PinListEntry]:
    """Extract and parse pin-list entries from a raw YAML dict."""
    raw_list = data.get("pin_list", [])
    if not isinstance(raw_list, list):
        return []
    entries: List[PinListEntry] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        h = item.get("hash", "")
        cat = item.get("category", "")
        action = item.get("action", "")
        label = str(item.get("label", ""))
        if h and cat and action:
            entries.append(PinListEntry(hash=str(h), category=str(cat),
                                        action=str(action), label=label))
    return entries


def _write_policy_atomic(path: Path, data: dict) -> None:
    """
    Write *data* as YAML to *path* atomically (write-to-tmp + ``os.replace``).

    Uses ``os.replace`` which is atomic on POSIX and Windows (Vista+), so
    readers never see a partial write.

    Raises ``OSError`` on failure; the temp file is cleaned up on error.
    """
    yaml = _load_yaml()
    content = yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    # Write to a sibling temp file so os.replace is on the same filesystem
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".piiguard_pinlist_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path_str, str(path))
    except Exception:
        # Best-effort cleanup; if the rename already happened the unlink
        # will fail silently — that's fine.
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PinListApprovalGate
# ─────────────────────────────────────────────────────────────────────────────

class PinListApprovalGate:
    """
    Sole write path for pin-list mutations (Sub-AC 5d-ii).

    The gate enforces a strict state machine:

    * ``idle`` — initial state; no change staged.
    * ``staged`` — a proposed new pin-list has been staged via
      :meth:`propose`; the file has not been touched yet.
    * ``committed`` — user approved; the file was written with
      ``pin_list_approved: true``.
    * ``rejected`` — user or caller rejected; no file I/O occurred.

    Each gate instance is single-use: once it reaches ``committed`` or
    ``rejected`` it can no longer be re-proposed (create a new instance).

    Bypassing the gate
    ------------------
    Because the underlying :class:`~pii_guard.policy.PolicyLoader` requires
    ``pin_list_approved: true`` for any pin-list change to take effect, the
    only way to commit a pin-list mutation is to go through this gate.
    Any direct write to the policy YAML that **omits** or **sets false** the
    ``pin_list_approved`` flag is caught and blocked on the next reload.

    Parameters
    ----------
    policy_path:
        Path to the policy YAML file managed by this gate.
    """

    #: Allowed state values
    IDLE = "idle"
    STAGED = "staged"
    COMMITTED = "committed"
    REJECTED = "rejected"

    def __init__(self, policy_path: str) -> None:
        self._policy_path = Path(policy_path)
        self._state: str = self.IDLE
        self._proposed: Optional[List[PinListEntry]] = None
        self._original: Optional[List[PinListEntry]] = None

    # ── Public read-only properties ────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state of the gate (``idle``, ``staged``, ``committed``, or ``rejected``)."""
        return self._state

    @property
    def proposed(self) -> Optional[List[PinListEntry]]:
        """A copy of the staged proposed pin-list, or ``None`` if not staged."""
        return list(self._proposed) if self._proposed is not None else None

    @property
    def original(self) -> Optional[List[PinListEntry]]:
        """A copy of the original pin-list at the time of staging, or ``None``."""
        return list(self._original) if self._original is not None else None

    # ── Core approval flow ─────────────────────────────────────────────────────

    def propose(self, new_entries: List[PinListEntry]) -> "PinListApprovalGate":
        """
        Stage a proposed new pin-list for review.

        Reads the current pin-list from the policy file to compute the diff.
        No file I/O occurs beyond the read.  The gate transitions from
        ``idle`` → ``staged``.

        Parameters
        ----------
        new_entries:
            The **complete** replacement pin-list (not a delta).  All current
            entries not present in this list will be removed; all new entries
            will be added.

        Returns
        -------
        PinListApprovalGate
            ``self``, for method chaining.

        Raises
        ------
        RuntimeError
            If the gate is not in the ``idle`` state (already used).
        """
        if self._state != self.IDLE:
            raise RuntimeError(
                f"PinListApprovalGate.propose() called in state {self._state!r}. "
                "Create a new gate instance to propose a new change."
            )
        raw = _read_policy_raw(self._policy_path)
        self._original = _extract_pin_list(raw)
        self._proposed = list(new_entries)
        self._state = self.STAGED
        log.debug(
            "PinListApprovalGate: staged proposal — "
            "%d entries proposed (was %d entries)",
            len(self._proposed), len(self._original),
        )
        return self

    def approve(self) -> ApprovalResult:
        """
        Commit the staged change to the policy file.

        Writes the new pin-list with ``pin_list_approved: true`` atomically
        using write-to-tmp + ``os.replace``.  The gate transitions to
        ``committed``.

        Returns
        -------
        ApprovalResult
            ``approved=True``, ``committed=True``.

        Raises
        ------
        RuntimeError
            If the gate is not in the ``staged`` state.
        OSError
            If the atomic write fails (permission denied, disk full, etc.).
        """
        if self._state != self.STAGED:
            raise RuntimeError(
                f"PinListApprovalGate.approve() called in state {self._state!r}. "
                "Call propose() first to stage a change before approving."
            )

        added, removed = self._compute_diff()

        # Read the full current policy to preserve non-pin-list fields
        raw_data = _read_policy_raw(self._policy_path)
        raw_data["pin_list"] = _entries_to_yaml_list(self._proposed or [])
        raw_data["pin_list_approved"] = True

        _write_policy_atomic(self._policy_path, raw_data)
        log.info(
            "PinListApprovalGate: committed — %d added, %d removed, "
            "pin_list_approved=true written to %s",
            len(added), len(removed), self._policy_path,
        )

        self._state = self.COMMITTED
        result = ApprovalResult(
            approved=True,
            committed=True,
            entries_added=[e.hash for e in added],
            entries_removed=[e.hash for e in removed],
        )
        self._proposed = None
        return result

    def reject(self) -> ApprovalResult:
        """
        Discard the staged change with zero file I/O.

        The policy file is **not touched**.  The gate transitions to
        ``rejected``.

        Returns
        -------
        ApprovalResult
            ``approved=False``, ``committed=False``.

        Raises
        ------
        RuntimeError
            If the gate is not in the ``staged`` state.
        """
        if self._state != self.STAGED:
            raise RuntimeError(
                f"PinListApprovalGate.reject() called in state {self._state!r}. "
                "Call propose() first to stage a change before rejecting."
            )

        added, removed = self._compute_diff()
        log.info(
            "PinListApprovalGate: rejected — no changes written to %s",
            self._policy_path,
        )

        self._state = self.REJECTED
        result = ApprovalResult(
            approved=False,
            committed=False,
            entries_added=[e.hash for e in added],
            entries_removed=[e.hash for e in removed],
            discarded_reason="User rejected the proposed pin-list change.",
        )
        self._proposed = None
        return result

    def diff(self) -> Tuple[List[PinListEntry], List[PinListEntry]]:
        """
        Return ``(added, removed)`` for the staged change.

        *added*   — entries in the proposed list that are not in the original.
        *removed* — entries in the original list that are not in the proposed.

        Raises
        ------
        RuntimeError
            If no change is currently staged.
        """
        if self._state != self.STAGED:
            raise RuntimeError(
                f"PinListApprovalGate.diff() called in state {self._state!r}. "
                "Call propose() first."
            )
        return self._compute_diff()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_diff(self) -> Tuple[List[PinListEntry], List[PinListEntry]]:
        """Compute (added, removed) without requiring STAGED state."""
        original_hashes = {e.hash for e in (self._original or [])}
        proposed_hashes = {e.hash for e in (self._proposed or [])}
        added = [e for e in (self._proposed or []) if e.hash not in original_hashes]
        removed = [e for e in (self._original or []) if e.hash not in proposed_hashes]
        return added, removed


# ─────────────────────────────────────────────────────────────────────────────
# Interactive approval flow
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_approval(
    policy_path: str,
    new_entries: List[PinListEntry],
    *,
    input_fn: Optional[Callable[[str], str]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> ApprovalResult:
    """
    Run the interactive pin-list approval flow.

    Reads the current pin-list from the policy file, displays a
    human-readable diff of the proposed changes, and prompts the user for
    confirmation.

    * ``y`` / ``yes``   → calls :meth:`PinListApprovalGate.approve`
    * anything else (including ``n``, empty, EOF, Ctrl-C)
                        → calls :meth:`PinListApprovalGate.reject`

    This function is the **sole CLI-facing write path** for pin-list changes.
    Automated tests can inject ``input_fn`` and ``output_fn`` hooks to
    exercise the full flow without interactive terminal I/O.

    Parameters
    ----------
    policy_path:
        Path to the policy YAML file to update.
    new_entries:
        The complete replacement pin-list (all entries, not a delta).
    input_fn:
        Callable that prompts the user and returns the response string.
        Defaults to the built-in ``input()``.  Raise ``EOFError`` or
        ``KeyboardInterrupt`` to simulate an aborted prompt.
    output_fn:
        Callable used to display output to the user.  Defaults to
        ``print()``.

    Returns
    -------
    ApprovalResult
        Describes what happened: approved/rejected, committed/not, diff.
    """
    _in = input_fn if input_fn is not None else input
    _out = output_fn if output_fn is not None else print

    gate = PinListApprovalGate(policy_path)
    gate.propose(new_entries)

    added, removed = gate.diff()

    # ── Display the proposal ─────────────────────────────────────────────────
    _out("")
    _out("─── PII-Guard Pin-List Change Proposal ─────────────────────────────────────")
    _out(f"Policy file: {policy_path}")
    _out("")

    if not added and not removed:
        _out("No changes detected — the proposed pin-list is identical to the current one.")
        _out("─────────────────────────────────────────────────────────────────────────────")
        return gate.reject()

    if added:
        _out("  ADDED entries:")
        for entry in added:
            label_part = f"  label={entry.label!r}" if entry.label else ""
            _out(f"    +  hash={entry.hash}  category={entry.category}  "
                 f"action={entry.action}{label_part}")

    if removed:
        _out("  REMOVED entries:")
        for entry in removed:
            label_part = f"  label={entry.label!r}" if entry.label else ""
            _out(f"    -  hash={entry.hash}  category={entry.category}  "
                 f"action={entry.action}{label_part}")

    _out("")
    _out(
        "WARNING: Pin-list changes affect which PII values receive special treatment.\n"
        "Review the diff carefully — each 'hash' is an opaque key for a specific\n"
        "real value; ensure you trust what each entry represents."
    )
    _out("─────────────────────────────────────────────────────────────────────────────")

    # ── Prompt for confirmation ───────────────────────────────────────────────
    try:
        response = _in("Approve this change? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _out("\nAborted. No changes were made.")
        return gate.reject()

    if response in ("y", "yes"):
        result = gate.approve()
        _out(f"✓  Pin-list change committed to {policy_path}")
        _out(f"   ({len(result.entries_added)} added, {len(result.entries_removed)} removed)")
        return result

    result = gate.reject()
    _out("✗  Pin-list change discarded. The policy file was not modified.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

#: The approval gate states, exported for test assertions.
GATE_IDLE = PinListApprovalGate.IDLE
GATE_STAGED = PinListApprovalGate.STAGED
GATE_COMMITTED = PinListApprovalGate.COMMITTED
GATE_REJECTED = PinListApprovalGate.REJECTED

__all__ = [
    "ApprovalResult",
    "PinListApprovalGate",
    "run_interactive_approval",
    "GATE_IDLE",
    "GATE_STAGED",
    "GATE_COMMITTED",
    "GATE_REJECTED",
]
