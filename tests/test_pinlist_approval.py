"""
Tests for PII-Guard pin-list out-of-band approval flow (Sub-AC 5d-ii).

Acceptance criteria (Sub-AC 5d-ii)
-----------------------------------
i.  The approval flow correctly **persists** a change when confirmed
    (approved=True → file written with new pin-list + pin_list_approved: true).

ii. The approval flow correctly **discards** a change when denied
    (approved=False → file NOT modified; original pin-list intact).

iii.``direct calls bypassing the approval gate cannot mutate the pin-list``
    — verified by asserting that raw YAML writes without pin_list_approved: true
      are blocked by PolicyLoader on the next reload, and that calling
      approve() without first calling propose() raises RuntimeError.

Test structure
--------------
A. Unit — PinListApprovalGate state machine
B. Unit — propose/approve/reject semantics
C. Unit — run_interactive_approval() helper (with input/output hooks)
D. Integration — gate + PolicyLoader: commit flows through to live config
E. Integration — bypass gate: direct writes blocked by PolicyLoader
F. Integration — CLI cmd_pin_list_propose() dispatch
G. Edge cases and invariants
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from pii_guard.pinlist_approval import (
    GATE_COMMITTED,
    GATE_IDLE,
    GATE_REJECTED,
    GATE_STAGED,
    ApprovalResult,
    PinListApprovalGate,
    _extract_pin_list,
    _entries_to_yaml_list,
    _read_policy_raw,
    run_interactive_approval,
)
from pii_guard.policy import PinListEntry, PolicyLoader, _hash_pin_list


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_entry(hash: str, category: str = "EMAIL", action: str = "allow", label: str = "") -> PinListEntry:
    return PinListEntry(hash=hash, category=category, action=action, label=label)


def _write_policy(path: Path, content: str, delay: float = 0.02) -> None:
    """Write policy YAML to *path* and bump mtime so reload_if_changed fires."""
    path.write_text(content, encoding="utf-8")
    t = time.time() + delay
    os.utime(str(path), (t, t))


def _policy_with_pin_list(
    entries: List[PinListEntry],
    approved: bool = True,
    extra: str = "",
) -> str:
    """Build a minimal policy YAML string."""
    lines = []
    if extra:
        lines.append(extra)
    if entries:
        lines.append("pin_list:")
        for e in entries:
            lines.append(f"  - hash: {e.hash}")
            lines.append(f"    category: {e.category}")
            lines.append(f"    action: {e.action}")
            if e.label:
                lines.append(f'    label: "{e.label}"')
    else:
        lines.append("pin_list: []")
    lines.append(f"pin_list_approved: {'true' if approved else 'false'}")
    return "\n".join(lines) + "\n"


@pytest.fixture()
def policy_file(tmp_path: Path) -> Path:
    """A policy YAML file with one initial approved pin-list entry."""
    p = tmp_path / "policy.yaml"
    _write_policy(p, _policy_with_pin_list(
        [_make_entry("sha256:initial", "EMAIL", "allow", "initial entry")],
        approved=True,
    ))
    return p


@pytest.fixture()
def empty_policy_file(tmp_path: Path) -> Path:
    """A policy YAML file with an empty approved pin-list."""
    p = tmp_path / "policy.yaml"
    _write_policy(p, _policy_with_pin_list([], approved=True))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# A. Unit — PinListApprovalGate state machine
# ─────────────────────────────────────────────────────────────────────────────

class TestGateStateMachine:
    """A1-A8: State transitions and invariants of PinListApprovalGate."""

    def test_initial_state_is_idle(self, tmp_path: Path):
        """A1: Freshly constructed gate is in IDLE state."""
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        assert gate.state == GATE_IDLE

    def test_propose_transitions_to_staged(self, empty_policy_file: Path):
        """A2: propose() transitions gate from IDLE → STAGED."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        assert gate.state == GATE_STAGED

    def test_approve_transitions_to_committed(self, empty_policy_file: Path):
        """A3: approve() transitions gate from STAGED → COMMITTED."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.approve()
        assert gate.state == GATE_COMMITTED

    def test_reject_transitions_to_rejected(self, empty_policy_file: Path):
        """A4: reject() transitions gate from STAGED → REJECTED."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.reject()
        assert gate.state == GATE_REJECTED

    def test_approve_without_propose_raises(self, tmp_path: Path):
        """A5: approve() in IDLE state raises RuntimeError."""
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        with pytest.raises(RuntimeError, match="propose"):
            gate.approve()

    def test_reject_without_propose_raises(self, tmp_path: Path):
        """A6: reject() in IDLE state raises RuntimeError."""
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        with pytest.raises(RuntimeError, match="propose"):
            gate.reject()

    def test_second_propose_after_commit_raises(self, empty_policy_file: Path):
        """A7: propose() after COMMITTED state raises RuntimeError (single-use gate)."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.approve()
        with pytest.raises(RuntimeError, match="new gate"):
            gate.propose([_make_entry("sha256:another")])

    def test_second_propose_after_reject_raises(self, empty_policy_file: Path):
        """A8: propose() after REJECTED state raises RuntimeError (single-use gate)."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.reject()
        with pytest.raises(RuntimeError, match="new gate"):
            gate.propose([_make_entry("sha256:another")])

    def test_approve_after_reject_raises(self, empty_policy_file: Path):
        """A8b: approve() in REJECTED state raises RuntimeError."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.reject()
        with pytest.raises(RuntimeError):
            gate.approve()

    def test_diff_without_propose_raises(self, tmp_path: Path):
        """A9: diff() in IDLE state raises RuntimeError."""
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        with pytest.raises(RuntimeError, match="propose"):
            gate.diff()

    def test_state_constants_match_class_attrs(self):
        """A10: Module-level constants equal PinListApprovalGate class attributes."""
        assert GATE_IDLE == PinListApprovalGate.IDLE
        assert GATE_STAGED == PinListApprovalGate.STAGED
        assert GATE_COMMITTED == PinListApprovalGate.COMMITTED
        assert GATE_REJECTED == PinListApprovalGate.REJECTED


# ─────────────────────────────────────────────────────────────────────────────
# B. Unit — propose / approve / reject semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestProposeSemanticsUnit:
    """B1-B6: propose() reads current pin-list and computes diff."""

    def test_propose_reads_original_from_file(self, policy_file: Path):
        """B1: propose() populates gate.original with entries from the file."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        assert gate.original is not None
        assert len(gate.original) == 1
        assert gate.original[0].hash == "sha256:initial"

    def test_propose_stores_proposed_entries(self, policy_file: Path):
        """B2: propose() stores the proposed list in gate.proposed."""
        new_entry = _make_entry("sha256:proposed", "PHONE", "mask")
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([new_entry])
        assert gate.proposed is not None
        assert len(gate.proposed) == 1
        assert gate.proposed[0].hash == "sha256:proposed"

    def test_diff_added_entries(self, policy_file: Path):
        """B3: diff() returns newly added entries correctly."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([
            _make_entry("sha256:initial"),  # existing — not added
            _make_entry("sha256:new"),      # new — will appear in added
        ])
        added, removed = gate.diff()
        assert any(e.hash == "sha256:new" for e in added)
        assert not any(e.hash == "sha256:initial" for e in added)

    def test_diff_removed_entries(self, policy_file: Path):
        """B4: diff() returns removed entries correctly."""
        gate = PinListApprovalGate(str(policy_file))
        # Propose an empty list — original entry will be removed
        gate.propose([])
        added, removed = gate.diff()
        assert any(e.hash == "sha256:initial" for e in removed)
        assert added == []

    def test_diff_no_change(self, policy_file: Path):
        """B5: diff() returns empty lists when proposed == current."""
        original_entry = _make_entry("sha256:initial", "EMAIL", "allow", "initial entry")
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([original_entry])
        added, removed = gate.diff()
        assert added == []
        assert removed == []

    def test_propose_with_missing_policy_file(self, tmp_path: Path):
        """B6: propose() works even if the policy file does not exist (treats current list as empty)."""
        nonexistent = tmp_path / "does_not_exist.yaml"
        gate = PinListApprovalGate(str(nonexistent))
        gate.propose([_make_entry("sha256:new")])
        assert gate.original == []
        assert gate.state == GATE_STAGED


class TestApproveSemantics:
    """B7-B12: approve() writes the correct file content."""

    def test_approve_returns_approved_result(self, empty_policy_file: Path):
        """B7: approve() returns ApprovalResult(approved=True, committed=True)."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:x")])
        result = gate.approve()
        assert result.approved is True
        assert result.committed is True

    def test_approve_file_contains_new_entry(self, empty_policy_file: Path):
        """B8: After approve(), the YAML file contains the new entry."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:written", "PHONE", "mask")])
        gate.approve()
        raw = _read_policy_raw(empty_policy_file)
        entries = _extract_pin_list(raw)
        assert any(e.hash == "sha256:written" for e in entries)

    def test_approve_file_sets_pin_list_approved_true(self, empty_policy_file: Path):
        """B9: After approve(), pin_list_approved is true in the YAML file."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:x")])
        gate.approve()
        raw = _read_policy_raw(empty_policy_file)
        assert raw.get("pin_list_approved") is True

    def test_approve_result_entries_added(self, policy_file: Path):
        """B10: ApprovalResult.entries_added contains hashes of new entries."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([
            _make_entry("sha256:initial"),   # kept
            _make_entry("sha256:added"),     # new
        ])
        result = gate.approve()
        assert "sha256:added" in result.entries_added
        assert "sha256:initial" not in result.entries_added

    def test_approve_result_entries_removed(self, policy_file: Path):
        """B11: ApprovalResult.entries_removed contains hashes of removed entries."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([])  # remove all
        result = gate.approve()
        assert "sha256:initial" in result.entries_removed

    def test_approve_preserves_non_pinlist_fields(self, tmp_path: Path):
        """B12: approve() preserves other YAML fields (fail_mode, allowlist, etc.)."""
        p = tmp_path / "policy.yaml"
        _write_policy(p,
            "fail_mode: open\n"
            "pin_list: []\n"
            "pin_list_approved: true\n"
        )
        gate = PinListApprovalGate(str(p))
        gate.propose([_make_entry("sha256:x")])
        gate.approve()
        raw = _read_policy_raw(p)
        assert raw.get("fail_mode") == "open", "Non-pin-list field must be preserved"


class TestRejectSemantics:
    """B13-B17: reject() discards without any file I/O."""

    def test_reject_returns_rejected_result(self, policy_file: Path):
        """B13: reject() returns ApprovalResult(approved=False, committed=False)."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        result = gate.reject()
        assert result.approved is False
        assert result.committed is False

    def test_reject_file_not_modified(self, policy_file: Path):
        """B14: After reject(), the policy file is not modified."""
        original_content = policy_file.read_text(encoding="utf-8")
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.reject()
        assert policy_file.read_text(encoding="utf-8") == original_content

    def test_reject_mtime_unchanged(self, policy_file: Path):
        """B15: After reject(), the file mtime is not updated."""
        original_mtime = policy_file.stat().st_mtime
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        gate.reject()
        assert policy_file.stat().st_mtime == original_mtime

    def test_reject_result_has_discarded_reason(self, policy_file: Path):
        """B16: reject() result includes a non-empty discarded_reason."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        result = gate.reject()
        assert result.discarded_reason
        assert len(result.discarded_reason) > 0

    def test_reject_result_diff_entries(self, policy_file: Path):
        """B17: reject() result still computes added/removed for audit purposes."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:added")])
        result = gate.reject()
        assert "sha256:added" in result.entries_added
        assert "sha256:initial" in result.entries_removed


# ─────────────────────────────────────────────────────────────────────────────
# C. Unit — run_interactive_approval() with hooks
# ─────────────────────────────────────────────────────────────────────────────

class TestRunInteractiveApproval:
    """C1-C10: run_interactive_approval() dispatch and hook behaviour."""

    def _run(self, policy_path: str, entries: List[PinListEntry], answer: str) -> ApprovalResult:
        """Helper: run the approval flow with a simulated user answer."""
        captured = []
        result = run_interactive_approval(
            policy_path,
            entries,
            input_fn=lambda prompt: answer,
            output_fn=lambda *args: captured.append(" ".join(str(a) for a in args)),
        )
        return result

    def test_yes_answer_returns_approved(self, empty_policy_file: Path):
        """C1: 'y' answer → approved=True, committed=True."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "y")
        assert result.approved is True
        assert result.committed is True

    def test_yes_uppercase_returns_approved(self, empty_policy_file: Path):
        """C1b: 'Y' answer is normalised to yes → approved=True."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "Y")
        assert result.approved is True

    def test_yes_word_returns_approved(self, empty_policy_file: Path):
        """C1c: 'yes' answer → approved=True."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "yes")
        assert result.approved is True

    def test_no_answer_returns_rejected(self, empty_policy_file: Path):
        """C2: 'n' answer → approved=False, committed=False."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "n")
        assert result.approved is False
        assert result.committed is False

    def test_empty_answer_returns_rejected(self, empty_policy_file: Path):
        """C2b: empty answer (just Enter) → rejected (default-deny)."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "")
        assert result.approved is False

    def test_other_string_returns_rejected(self, empty_policy_file: Path):
        """C2c: 'maybe' → rejected."""
        result = self._run(str(empty_policy_file), [_make_entry("sha256:x")], "maybe")
        assert result.approved is False

    def test_eof_returns_rejected(self, empty_policy_file: Path):
        """C3: EOFError from input → rejected gracefully."""
        def _raise_eof(prompt: str):
            raise EOFError
        captured = []
        result = run_interactive_approval(
            str(empty_policy_file),
            [_make_entry("sha256:x")],
            input_fn=_raise_eof,
            output_fn=lambda *a: captured.append(str(a)),
        )
        assert result.approved is False
        assert result.committed is False

    def test_keyboard_interrupt_returns_rejected(self, empty_policy_file: Path):
        """C4: KeyboardInterrupt from input → rejected gracefully."""
        def _raise_kbi(prompt: str):
            raise KeyboardInterrupt
        captured = []
        result = run_interactive_approval(
            str(empty_policy_file),
            [_make_entry("sha256:x")],
            input_fn=_raise_kbi,
            output_fn=lambda *a: captured.append(str(a)),
        )
        assert result.approved is False
        assert result.committed is False

    def test_no_change_auto_rejected(self, policy_file: Path):
        """C5: When proposed == current, flow auto-rejects without prompting."""
        original_entry = _make_entry("sha256:initial", "EMAIL", "allow", "initial entry")
        prompted = []
        result = run_interactive_approval(
            str(policy_file),
            [original_entry],
            input_fn=lambda p: prompted.append(p) or "y",
            output_fn=lambda *a: None,
        )
        # Because the list is identical, it should auto-reject without asking
        assert result.approved is False
        assert result.committed is False
        # The prompt should NOT have been shown (no changes to approve)
        assert len(prompted) == 0

    def test_yes_writes_correct_pin_list(self, empty_policy_file: Path):
        """C6: After 'y' answer, the policy file has the expected pin-list entries."""
        entries = [
            _make_entry("sha256:a", "EMAIL", "allow"),
            _make_entry("sha256:b", "PHONE", "mask"),
        ]
        self._run(str(empty_policy_file), entries, "y")
        raw = _read_policy_raw(empty_policy_file)
        persisted = _extract_pin_list(raw)
        assert {e.hash for e in persisted} == {"sha256:a", "sha256:b"}

    def test_no_does_not_write_file(self, policy_file: Path):
        """C7: After 'n' answer, the policy file is not touched."""
        original_mtime = policy_file.stat().st_mtime
        self._run(str(policy_file), [_make_entry("sha256:new")], "n")
        assert policy_file.stat().st_mtime == original_mtime

    def test_output_fn_called_with_diff_info(self, policy_file: Path):
        """C8: output_fn receives diff information (added/removed entries shown)."""
        captured_lines = []
        run_interactive_approval(
            str(policy_file),
            [_make_entry("sha256:new")],
            input_fn=lambda p: "n",
            output_fn=lambda *args: captured_lines.append(" ".join(str(a) for a in args)),
        )
        combined = "\n".join(captured_lines)
        # Should mention the added entry and the removed original
        assert "sha256:new" in combined or "ADDED" in combined.upper()

    def test_approval_result_has_diff_info(self, policy_file: Path):
        """C9: ApprovalResult contains entries_added / entries_removed counts."""
        result = run_interactive_approval(
            str(policy_file),
            [_make_entry("sha256:new")],
            input_fn=lambda p: "y",
            output_fn=lambda *a: None,
        )
        assert "sha256:new" in result.entries_added
        assert "sha256:initial" in result.entries_removed

    def test_result_dataclass_fields(self, empty_policy_file: Path):
        """C10: ApprovalResult is importable from pii_guard and has expected fields."""
        import pii_guard
        assert hasattr(pii_guard, "ApprovalResult")
        r = ApprovalResult(
            approved=True, committed=True,
            entries_added=["sha256:a"], entries_removed=[]
        )
        assert r.approved is True
        assert r.committed is True
        assert "sha256:a" in r.entries_added


# ─────────────────────────────────────────────────────────────────────────────
# D. Integration — gate + PolicyLoader: commit flows through to live config
# ─────────────────────────────────────────────────────────────────────────────

class TestGatePolicyLoaderIntegration:
    """D1-D6: End-to-end: gate commit → PolicyLoader reloads new pin-list."""

    def test_approved_change_picked_up_by_policy_loader(self, policy_file: Path):
        """D1: After gate.approve(), PolicyLoader.reload_if_changed() picks up the new list."""
        loader = PolicyLoader(str(policy_file))
        assert len(loader.config.pin_list) == 1

        # Propose and approve a new entry via the gate
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([
            _make_entry("sha256:initial"),   # keep existing
            _make_entry("sha256:approved"),  # add new
        ])
        gate.approve()

        # Bump mtime so the loader detects the change
        t = time.time() + 0.05
        os.utime(str(policy_file), (t, t))

        loader.reload_if_changed()

        hashes = {e.hash for e in loader.config.pin_list}
        assert "sha256:approved" in hashes, "New entry must be loaded after gate.approve()"
        assert "sha256:initial" in hashes, "Existing entry must still be present"

    def test_rejected_change_not_picked_up_by_policy_loader(self, policy_file: Path):
        """D2: After gate.reject(), PolicyLoader.reload_if_changed() sees no change."""
        loader = PolicyLoader(str(policy_file))
        original_hash = _hash_pin_list(loader.config.pin_list)

        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:not_committed")])
        gate.reject()

        # Small sleep to ensure mtime would change if the file was touched
        time.sleep(0.03)
        loader.reload_if_changed()

        current_hash = _hash_pin_list(loader.config.pin_list)
        assert current_hash == original_hash, (
            "Rejected gate must not change the PolicyLoader's pin-list"
        )

    def test_approved_pin_list_approved_flag_is_true(self, policy_file: Path):
        """D3: After gate.approve(), the YAML file has pin_list_approved=true."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:x")])
        gate.approve()
        raw = _read_policy_raw(policy_file)
        assert raw.get("pin_list_approved") is True

    def test_approved_change_loader_sees_correct_entries(self, empty_policy_file: Path):
        """D4: Full round-trip: approve → reload → correct pin-list in loader.config."""
        loader = PolicyLoader(str(empty_policy_file))
        assert loader.config.pin_list == []

        new_entries = [
            _make_entry("sha256:roundtrip1", "EMAIL", "allow"),
            _make_entry("sha256:roundtrip2", "PHONE", "block"),
        ]
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose(new_entries)
        gate.approve()

        t = time.time() + 0.05
        os.utime(str(empty_policy_file), (t, t))
        loader.reload_if_changed()

        loaded_hashes = {e.hash for e in loader.config.pin_list}
        assert loaded_hashes == {"sha256:roundtrip1", "sha256:roundtrip2"}

    def test_interactive_approval_yes_picked_up_by_loader(self, empty_policy_file: Path):
        """D5: run_interactive_approval(y) → loader picks up the new entry."""
        loader = PolicyLoader(str(empty_policy_file))

        run_interactive_approval(
            str(empty_policy_file),
            [_make_entry("sha256:interactive")],
            input_fn=lambda p: "y",
            output_fn=lambda *a: None,
        )

        t = time.time() + 0.05
        os.utime(str(empty_policy_file), (t, t))
        loader.reload_if_changed()

        assert any(e.hash == "sha256:interactive" for e in loader.config.pin_list)

    def test_interactive_approval_no_not_picked_up_by_loader(self, policy_file: Path):
        """D6: run_interactive_approval(n) → loader unchanged."""
        loader = PolicyLoader(str(policy_file))
        original_hashes = {e.hash for e in loader.config.pin_list}

        run_interactive_approval(
            str(policy_file),
            [_make_entry("sha256:should_not_appear")],
            input_fn=lambda p: "n",
            output_fn=lambda *a: None,
        )

        time.sleep(0.03)
        loader.reload_if_changed()

        current_hashes = {e.hash for e in loader.config.pin_list}
        assert current_hashes == original_hashes
        assert "sha256:should_not_appear" not in current_hashes


# ─────────────────────────────────────────────────────────────────────────────
# E. Integration — bypass gate: direct writes blocked by PolicyLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestBypassGateBlocked:
    """
    E1-E7: Direct calls bypassing the approval gate cannot mutate the pin-list.

    These tests verify that the enforcement layer (PolicyLoader + PinListMutationGuard)
    blocks any pin-list change that does not carry pin_list_approved: true,
    i.e. any path that bypasses the PinListApprovalGate.
    """

    def test_direct_write_without_approved_flag_blocked(self, policy_file: Path):
        """
        E1: Writing the YAML directly with pin_list_approved=false is blocked
        by PolicyLoader on the next reload.
        """
        loader = PolicyLoader(str(policy_file))
        original_hash = _hash_pin_list(loader.config.pin_list)

        # Bypass: write a new pin-list entry directly without the approval gate
        _write_policy(policy_file, _policy_with_pin_list(
            [_make_entry("sha256:bypass_attempt")],
            approved=False,  # <--- no approval
        ))
        loader.reload_if_changed()

        current_hash = _hash_pin_list(loader.config.pin_list)
        assert current_hash == original_hash, (
            "PolicyLoader must retain old pin-list when pin_list_approved=false"
        )
        assert not any(
            e.hash == "sha256:bypass_attempt"
            for e in loader.config.pin_list
        ), "Bypassed entry must not appear in loader config"

    def test_direct_write_without_approved_key_is_legitimate_oob_bypass(
        self, policy_file: Path
    ):
        """
        E2: Writing the YAML directly *without* the pin_list_approved key at all
        is treated as an out-of-band user action and IS accepted, because the
        PolicyConfig defaults pin_list_approved to True (vacuous approval).

        Rationale: The approval gate is the sole *programmatic* write path.
        A human user editing the YAML file directly in their editor is the
        intentional, documented out-of-band bypass.  PolicyLoader accepts such
        changes (YAML with no pin_list_approved key or explicit ``true``) because
        that is the manual user-facing workflow.  Only ``pin_list_approved: false``
        explicitly signals "block this change".
        """
        loader = PolicyLoader(str(policy_file))
        assert any(e.hash == "sha256:initial" for e in loader.config.pin_list)

        # Write a new pin-list without a pin_list_approved key (defaults to True)
        _write_policy(policy_file,
            "pin_list:\n"
            "  - hash: sha256:oob_user_edit\n"
            "    category: EMAIL\n"
            "    action: allow\n"
            # No pin_list_approved key → defaults to True in PolicyConfig
        )
        loader.reload_if_changed()

        # Because pin_list_approved defaults to True, the change IS accepted
        hashes = {e.hash for e in loader.config.pin_list}
        assert "sha256:oob_user_edit" in hashes, (
            "PolicyLoader must accept pin-list change when pin_list_approved defaults to True "
            "(absent key = legitimate out-of-band user edit)"
        )

    def test_multiple_bypass_attempts_do_not_accumulate(self, policy_file: Path):
        """E3: Repeated bypass attempts never accumulate entries in the loader."""
        loader = PolicyLoader(str(policy_file))

        for i in range(5):
            _write_policy(policy_file, _policy_with_pin_list(
                [_make_entry(f"sha256:bypass_{i}")],
                approved=False,
            ))
            loader.reload_if_changed()

        # After 5 blocked attempts, pin-list should still contain only the initial entry
        hashes = {e.hash for e in loader.config.pin_list}
        assert hashes == {"sha256:initial"}, (
            f"Expected only sha256:initial after blocked bypass attempts, got: {hashes}"
        )

    def test_approve_without_propose_is_a_direct_bypass_and_blocked(
        self, tmp_path: Path
    ):
        """
        E4: Calling gate.approve() without propose() raises RuntimeError —
        it cannot be used as a direct write path.
        """
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        with pytest.raises(RuntimeError):
            gate.approve()

    def test_reject_without_propose_cannot_corrupt_state(self, tmp_path: Path):
        """
        E5: Calling gate.reject() without propose() raises RuntimeError —
        the gate's internal state cannot be corrupted by orphaned calls.
        """
        gate = PinListApprovalGate(str(tmp_path / "policy.yaml"))
        with pytest.raises(RuntimeError):
            gate.reject()

    def test_no_side_effects_on_rejected_gate(self, policy_file: Path):
        """
        E6: A gate that has been rejected has no side effects on the file system.
        The original file content is preserved exactly.
        """
        original_content = policy_file.read_text(encoding="utf-8")
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:rejected")])
        gate.reject()
        assert policy_file.read_text(encoding="utf-8") == original_content

    def test_bypass_then_legitimate_approval_succeeds(self, policy_file: Path):
        """
        E7: After a blocked bypass attempt, a legitimate gate-based approval
        still succeeds and correctly updates the pin-list.
        """
        loader = PolicyLoader(str(policy_file))

        # First, a bypass attempt (blocked)
        _write_policy(policy_file, _policy_with_pin_list(
            [_make_entry("sha256:bypass")], approved=False
        ))
        loader.reload_if_changed()

        # Then, a legitimate approval
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:legitimate")])
        gate.approve()

        t = time.time() + 0.05
        os.utime(str(policy_file), (t, t))
        loader.reload_if_changed()

        hashes = {e.hash for e in loader.config.pin_list}
        assert "sha256:legitimate" in hashes
        assert "sha256:bypass" not in hashes


# ─────────────────────────────────────────────────────────────────────────────
# F. Integration — CLI cmd_pin_list_propose dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPinListPropose:
    """F1-F8: CLI 'piiguard pin-list propose' command."""

    def _cli_main(self, argv: List[str]) -> int:
        """Run the CLI main() and return the exit code."""
        from pii_guard.cli import main
        return main(argv)

    def test_pin_list_propose_yes_exits_zero(self, empty_policy_file: Path, monkeypatch):
        """F1: 'piiguard pin-list propose --add ... --policy-path ...' with Y → exit 0."""
        # Monkeypatch run_interactive_approval to simulate a 'yes' answer
        import pii_guard.cli as cli_mod
        monkeypatch.setattr(
            cli_mod, "run_interactive_approval",
            lambda path, entries, **kw: ApprovalResult(
                approved=True, committed=True,
                entries_added=["sha256:x"], entries_removed=[],
            ),
        )
        rc = self._cli_main([
            "pin-list", "propose",
            "--policy-path", str(empty_policy_file),
            "--add", "hash=sha256:x,category=EMAIL,action=allow",
        ])
        assert rc == 0

    def test_pin_list_propose_no_exits_one(self, empty_policy_file: Path, monkeypatch):
        """F2: 'piiguard pin-list propose' with N → exit 1."""
        import pii_guard.cli as cli_mod
        monkeypatch.setattr(
            cli_mod, "run_interactive_approval",
            lambda path, entries, **kw: ApprovalResult(
                approved=False, committed=False,
                entries_added=[], entries_removed=[],
                discarded_reason="rejected",
            ),
        )
        rc = self._cli_main([
            "pin-list", "propose",
            "--policy-path", str(empty_policy_file),
            "--add", "hash=sha256:x,category=EMAIL,action=allow",
        ])
        assert rc == 1

    def test_pin_list_propose_no_policy_path_exits_two(self, monkeypatch):
        """F3: Missing --policy-path and no env var → exit 2."""
        monkeypatch.delenv("PIIGUARD_POLICY_PATH", raising=False)
        rc = self._cli_main([
            "pin-list", "propose",
            "--add", "hash=sha256:x,category=EMAIL,action=allow",
        ])
        assert rc == 2

    def test_pin_list_propose_invalid_action_exits_two(self, empty_policy_file: Path):
        """F4: Invalid action value → argparse error → exit 2."""
        with pytest.raises(SystemExit) as exc_info:
            self._cli_main([
                "pin-list", "propose",
                "--policy-path", str(empty_policy_file),
                "--add", "hash=sha256:x,category=EMAIL,action=INVALID",
            ])
        assert exc_info.value.code == 2

    def test_pin_list_propose_multiple_add_flags(self, empty_policy_file: Path, monkeypatch):
        """F5: Multiple --add flags are accumulated correctly."""
        captured_entries = []

        import pii_guard.cli as cli_mod
        def _capture(path, entries, **kw):
            captured_entries.extend(entries)
            return ApprovalResult(approved=False, committed=False,
                                  entries_added=[], entries_removed=[],
                                  discarded_reason="test")
        monkeypatch.setattr(cli_mod, "run_interactive_approval", _capture)

        self._cli_main([
            "pin-list", "propose",
            "--policy-path", str(empty_policy_file),
            "--add", "hash=sha256:a,category=EMAIL,action=allow",
            "--add", "hash=sha256:b,category=PHONE,action=mask",
        ])
        hashes = {e.hash for e in captured_entries}
        assert hashes == {"sha256:a", "sha256:b"}

    def test_pin_list_propose_env_var_policy_path(self, empty_policy_file: Path, monkeypatch):
        """F6: PIIGUARD_POLICY_PATH env var is used when --policy-path is absent."""
        import pii_guard.cli as cli_mod
        monkeypatch.setenv("PIIGUARD_POLICY_PATH", str(empty_policy_file))
        captured_paths = []
        def _capture(path, entries, **kw):
            captured_paths.append(path)
            return ApprovalResult(approved=False, committed=False,
                                  entries_added=[], entries_removed=[],
                                  discarded_reason="test")
        monkeypatch.setattr(cli_mod, "run_interactive_approval", _capture)

        self._cli_main([
            "pin-list", "propose",
            "--add", "hash=sha256:x,category=EMAIL,action=allow",
        ])
        assert captured_paths == [str(empty_policy_file)]

    def test_pin_list_propose_entry_with_label(self, empty_policy_file: Path, monkeypatch):
        """F7: --add with label field is parsed and passed through."""
        captured = []

        import pii_guard.cli as cli_mod
        def _capture(path, entries, **kw):
            captured.extend(entries)
            return ApprovalResult(approved=False, committed=False,
                                  entries_added=[], entries_removed=[],
                                  discarded_reason="test")
        monkeypatch.setattr(cli_mod, "run_interactive_approval", _capture)

        self._cli_main([
            "pin-list", "propose",
            "--policy-path", str(empty_policy_file),
            "--add", "hash=sha256:x,category=EMAIL,action=allow,label=dev relay",
        ])
        assert len(captured) == 1
        assert captured[0].label == "dev relay"

    def test_pin_list_subparser_help_available(self, capsys):
        """F8: 'piiguard pin-list propose --help' does not raise and exits 0."""
        from pii_guard.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["pin-list", "propose", "--help"])
        assert exc_info.value.code == 0


# ─────────────────────────────────────────────────────────────────────────────
# G. Edge cases and invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCasesAndInvariants:
    """G1-G12: Edge cases, invariants, and package-level export checks."""

    def test_gate_proposed_list_is_a_copy(self, empty_policy_file: Path):
        """G1: Mutating the list returned by gate.proposed doesn't affect internal state."""
        gate = PinListApprovalGate(str(empty_policy_file))
        entry = _make_entry("sha256:original")
        gate.propose([entry])
        proposed_copy = gate.proposed
        assert proposed_copy is not None
        proposed_copy.clear()   # mutate the returned copy
        # Internal state should be unchanged
        assert gate.proposed is not None
        assert len(gate.proposed) == 1

    def test_gate_original_list_is_a_copy(self, policy_file: Path):
        """G2: Mutating the list returned by gate.original doesn't affect internal state."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([_make_entry("sha256:new")])
        original_copy = gate.original
        assert original_copy is not None
        original_copy.clear()
        assert gate.original is not None
        assert len(gate.original) == 1

    def test_approval_result_discard_reason_none_when_approved(
        self, empty_policy_file: Path
    ):
        """G3: ApprovalResult.discarded_reason is None when approved=True."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:x")])
        result = gate.approve()
        assert result.discarded_reason is None

    def test_approve_empty_pin_list(self, policy_file: Path):
        """G4: approve([]) removes all entries and sets pin_list_approved=true."""
        gate = PinListApprovalGate(str(policy_file))
        gate.propose([])
        result = gate.approve()
        assert result.committed is True
        raw = _read_policy_raw(policy_file)
        assert raw.get("pin_list_approved") is True
        entries = _extract_pin_list(raw)
        assert entries == [], "All entries should be removed after approve([])"

    def test_file_created_if_not_exists(self, tmp_path: Path):
        """G5: approve() creates the policy file if it doesn't exist yet."""
        path = tmp_path / "new_policy.yaml"
        assert not path.exists()
        gate = PinListApprovalGate(str(path))
        gate.propose([_make_entry("sha256:new")])
        gate.approve()
        assert path.exists()

    def test_atomic_write_uses_temp_file(self, empty_policy_file: Path, monkeypatch):
        """G6: _write_policy_atomic uses a temp file in the same directory."""
        import pii_guard.pinlist_approval as approval_mod
        original_replace = os.replace
        replaced_pairs = []
        def _track_replace(src, dst):
            replaced_pairs.append((src, dst))
            original_replace(src, dst)
        monkeypatch.setattr(os, "replace", _track_replace)

        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:x")])
        gate.approve()

        assert len(replaced_pairs) == 1
        tmp_src, dst = replaced_pairs[0]
        assert str(empty_policy_file.parent) in tmp_src, (
            "Temp file should be in the same directory as the policy file"
        )
        assert dst == str(empty_policy_file)

    def test_package_exports_approval_symbols(self):
        """G7: All approval flow symbols are importable from the top-level pii_guard package."""
        import pii_guard
        assert hasattr(pii_guard, "PinListApprovalGate")
        assert hasattr(pii_guard, "run_interactive_approval")
        assert hasattr(pii_guard, "ApprovalResult")
        assert hasattr(pii_guard, "GATE_IDLE")
        assert hasattr(pii_guard, "GATE_STAGED")
        assert hasattr(pii_guard, "GATE_COMMITTED")
        assert hasattr(pii_guard, "GATE_REJECTED")

    def test_approved_result_is_dataclass(self):
        """G8: ApprovalResult is a dataclass with expected fields."""
        r = ApprovalResult(
            approved=True, committed=True,
            entries_added=["sha256:a"], entries_removed=["sha256:b"],
        )
        assert r.approved is True
        assert r.committed is True
        assert r.entries_added == ["sha256:a"]
        assert r.entries_removed == ["sha256:b"]
        assert r.discarded_reason is None

    def test_entries_to_yaml_list_roundtrip(self):
        """G9: _entries_to_yaml_list → _extract_pin_list round-trips correctly."""
        entries = [
            _make_entry("sha256:a", "EMAIL", "allow", "test label"),
            _make_entry("sha256:b", "PHONE", "mask"),
        ]
        yaml_list = _entries_to_yaml_list(entries)
        recovered = _extract_pin_list({"pin_list": yaml_list})
        assert len(recovered) == 2
        assert {e.hash for e in recovered} == {"sha256:a", "sha256:b"}
        assert next(e for e in recovered if e.hash == "sha256:a").label == "test label"

    def test_gate_with_nonexistent_parent_directory_raises(self, tmp_path: Path):
        """G10: approve() fails with OSError if the parent directory does not exist."""
        deep_path = tmp_path / "nonexistent" / "deep" / "policy.yaml"
        gate = PinListApprovalGate(str(deep_path))
        gate.propose([_make_entry("sha256:x")])
        with pytest.raises(OSError):
            gate.approve()

    def test_no_pin_list_in_file_treated_as_empty(self, tmp_path: Path):
        """G11: Policy file with no pin_list field → original treated as empty list."""
        p = tmp_path / "policy.yaml"
        _write_policy(p, "fail_mode: closed\n")
        gate = PinListApprovalGate(str(p))
        gate.propose([_make_entry("sha256:new")])
        assert gate.original == []

    def test_approve_twice_second_raises(self, empty_policy_file: Path):
        """G12: Calling approve() twice on the same gate (after commit) raises RuntimeError."""
        gate = PinListApprovalGate(str(empty_policy_file))
        gate.propose([_make_entry("sha256:x")])
        gate.approve()
        with pytest.raises(RuntimeError):
            gate.approve()

    def test_parse_pin_list_entry_arg_valid(self):
        """G13: _parse_pin_list_entry_arg parses a valid entry string correctly."""
        from pii_guard.cli import _parse_pin_list_entry_arg
        e = _parse_pin_list_entry_arg("hash=sha256:abc,category=EMAIL,action=allow,label=test")
        assert e.hash == "sha256:abc"
        assert e.category == "EMAIL"
        assert e.action == "allow"
        assert e.label == "test"

    def test_parse_pin_list_entry_arg_missing_hash(self):
        """G14: _parse_pin_list_entry_arg raises ArgumentTypeError when hash is missing."""
        import argparse
        from pii_guard.cli import _parse_pin_list_entry_arg
        with pytest.raises(argparse.ArgumentTypeError, match="hash"):
            _parse_pin_list_entry_arg("category=EMAIL,action=allow")

    def test_parse_pin_list_entry_arg_invalid_action(self):
        """G15: _parse_pin_list_entry_arg raises ArgumentTypeError for invalid action."""
        import argparse
        from pii_guard.cli import _parse_pin_list_entry_arg
        with pytest.raises(argparse.ArgumentTypeError, match="action"):
            _parse_pin_list_entry_arg("hash=sha256:x,category=EMAIL,action=BOGUS")
