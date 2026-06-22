"""
pii_guard/session_map.py

Per-session mapping store for PII-Guard placeholder substitution.

Provides session-consistent indexed placeholder assignment:
  - Same value within a session always gets the same [CATEGORY_N] placeholder
    (idempotent re-encoding).
  - Counters are per-category and monotonically increasing (cross-category
    counter isolation: EMAIL counter and PHONE counter are independent).
  - Blocked items get [CATEGORY_N_BLOCKED] tokens.
  - ``reset()`` wipes all state so the next call starts from index 1 again.

Usage::

    from pii_guard.session_map import SessionMap

    sm = SessionMap()

    # Encode: returns the token string (no surrounding brackets)
    tok = sm.encode("alice@corp.io", "EMAIL")        # → "EMAIL_1"
    tok2 = sm.encode("alice@corp.io", "EMAIL")       # → "EMAIL_1" (idempotent)
    tok3 = sm.encode("bob@corp.io", "EMAIL")         # → "EMAIL_2"
    tok4 = sm.encode("010-1234-5678", "PHONE")       # → "PHONE_1" (isolated counter)
    tok5 = sm.encode("sk-xxx", "API_KEY", blocked=True)  # → "API_KEY_1_BLOCKED"

    # Decode: returns original or None
    original = sm.decode("EMAIL_1")                  # → "alice@corp.io"
    missing  = sm.decode("EMAIL_99")                 # → None

    # Brackets helper
    bracket = sm.bracket("EMAIL_1")                  # → "[EMAIL_1]"

    # Session reset — all counters and mappings cleared
    sm.reset()
    tok6 = sm.encode("alice@corp.io", "EMAIL")       # → "EMAIL_1" (fresh session)

Design notes
------------
- NOT thread-safe: designed for single-request / single-session use within the
  PII-Guard proxy core.  Callers needing concurrent access must provide external
  locking.
- Raw original values are held in memory only and never written to disk.
- The ``restoration_map`` property exposes the placeholder→original mapping
  for rehydration; it returns a snapshot (copy) so callers cannot mutate the
  live state.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional


class SessionMap:
    """
    Per-session mapping store for deterministic placeholder assignment.

    Parameters
    ----------
    None — all state is initialised empty on construction.

    Attributes (private, accessed via properties)
    -------------------
    _value_to_placeholder : dict[str, str]
        Maps original_value → placeholder_token.  Used for idempotent
        re-encoding within the same session.
    _placeholder_to_value : dict[str, str]
        Maps placeholder_token → original_value.  Used for inbound
        rehydration (decode / restoration).
    _counters : defaultdict[str, int]
        Per-category monotonically increasing counter.  Category keys are
        fully isolated so "EMAIL_1" does not advance the PHONE counter.
    """

    def __init__(self) -> None:
        self._value_to_placeholder: Dict[str, str] = {}
        self._placeholder_to_value: Dict[str, str] = {}
        self._counters: Dict[str, int] = defaultdict(int)

    # ── Primary encode / decode API ─────────────────────────────────────────────

    def encode(
        self,
        value: str,
        category: str,
        *,
        blocked: bool = False,
    ) -> str:
        """
        Return the placeholder token for *value* under *category*.

        If *value* has already been encoded in this session the same token is
        returned unchanged (idempotent — same value always same placeholder).
        Otherwise a new monotonically increasing index is assigned for
        *category* and both forward/reverse mappings are stored.

        Parameters
        ----------
        value:
            The raw PII/secret string to encode.  Must be non-empty.
        category:
            Detection category string, e.g. ``"EMAIL"`` or ``"API_KEY"``.
            Must be non-empty.  The counter namespace is the exact string
            value, so category comparisons are case-sensitive.
        blocked:
            When ``True`` the token is suffixed with ``_BLOCKED``, e.g.
            ``"API_KEY_1_BLOCKED"``.  This flag is stored with the mapping
            and is consistent for the same *value* across re-encodes.

        Returns
        -------
        str
            The placeholder token *without* surrounding brackets, e.g.
            ``"EMAIL_1"`` or ``"API_KEY_1_BLOCKED"``.

        Raises
        ------
        ValueError
            If *value* or *category* is an empty string.
        """
        if not value:
            raise ValueError("encode() called with empty value")
        if not category:
            raise ValueError("encode() called with empty category")

        # Idempotent: re-use existing placeholder for the same value
        existing = self._value_to_placeholder.get(value)
        if existing is not None:
            return existing

        # Assign new index for this category
        self._counters[category] += 1
        idx = self._counters[category]

        token = f"{category}_{idx}_BLOCKED" if blocked else f"{category}_{idx}"

        self._value_to_placeholder[value] = token
        self._placeholder_to_value[token] = value
        return token

    def decode(self, token: str) -> Optional[str]:
        """
        Return the original value for *token*, or ``None`` if unknown.

        Parameters
        ----------
        token:
            The placeholder token without brackets, e.g. ``"EMAIL_1"`` or
            ``"API_KEY_1_BLOCKED"``.

        Returns
        -------
        str | None
            The original raw value, or ``None`` if the token was never
            registered in this session.
        """
        return self._placeholder_to_value.get(token)

    def decode_bracketed(self, bracketed: str) -> Optional[str]:
        """
        Return the original value for a bracketed token like ``"[EMAIL_1]"``.

        Strips the surrounding ``[`` / ``]`` and delegates to :meth:`decode`.
        Returns ``None`` if the brackets are not present or the token is unknown.
        """
        if bracketed.startswith("[") and bracketed.endswith("]"):
            return self.decode(bracketed[1:-1])
        return None

    # ── Rehydration helper ──────────────────────────────────────────────────────

    def rehydrate(self, text: str) -> str:
        """
        Replace all ``[TOKEN]`` placeholders in *text* with their original values.

        Replacement is performed longest-token-first to avoid partial-token
        substitution bugs (e.g. ``EMAIL_10`` being partially replaced by
        ``EMAIL_1`` match).

        Parameters
        ----------
        text:
            String potentially containing ``[CATEGORY_N]`` or
            ``[CATEGORY_N_BLOCKED]`` tokens.

        Returns
        -------
        str
            Text with all known placeholders restored.  Unknown tokens are
            left unchanged.
        """
        result = text
        # Sort by token length descending to avoid short-token shadowing
        for token in sorted(self._placeholder_to_value, key=len, reverse=True):
            original = self._placeholder_to_value[token]
            result = result.replace(f"[{token}]", original)
        return result

    # ── Convenience helpers ─────────────────────────────────────────────────────

    @staticmethod
    def bracket(token: str) -> str:
        """Wrap *token* in square brackets: ``"EMAIL_1"`` → ``"[EMAIL_1]"``."""
        return f"[{token}]"

    # ── Session lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all per-session state (counters, forward map, reverse map).

        After reset, encoding the same value produces a fresh index starting
        from 1 again — as if this is a brand-new session.
        """
        self._value_to_placeholder.clear()
        self._placeholder_to_value.clear()
        self._counters.clear()

    # ── Read-only views ─────────────────────────────────────────────────────────

    @property
    def restoration_map(self) -> Dict[str, str]:
        """
        Snapshot of the current placeholder→original mapping.

        Used by the rehydration step to restore outbound placeholders in
        inbound LLM responses.  Returns a *copy* — callers cannot mutate
        live session state through this property.

        Never persisted to disk.
        """
        return dict(self._placeholder_to_value)

    @property
    def encode_map(self) -> Dict[str, str]:
        """
        Snapshot of the current original→placeholder mapping.

        Returns a *copy*.
        """
        return dict(self._value_to_placeholder)

    @property
    def counters(self) -> Dict[str, int]:
        """
        Snapshot of per-category counters (category → last assigned index).

        Returns a *copy* — mutations do not affect the live state.
        """
        return dict(self._counters)

    # ── Dunder helpers ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of distinct values encoded in this session."""
        return len(self._value_to_placeholder)

    def __contains__(self, value: str) -> bool:
        """``True`` if *value* has already been encoded in this session."""
        return value in self._value_to_placeholder

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SessionMap(entries={len(self)}, "
            f"categories={list(self._counters.keys())})"
        )
