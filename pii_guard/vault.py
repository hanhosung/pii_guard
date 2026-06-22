"""
pii_guard/vault.py

Request-scoped masking vault and rehydration  (Sub-AC 5c-ii).

Provides:

  RequestVault
      Per-request mapping store.  Created fresh for each inbound LLM
      request; unlike :class:`~pii_guard.session_map.SessionMap` it is
      **not** shared across requests.  Stores token→original pairs in
      memory only (never persisted to disk).

  apply_mask_style(original, category, mask_style, vault, ...)
      Apply a single mask style to *original*, persist the original in
      *vault*, and return the masked representation:

      ``tokenize``          – ``[CATEGORY_N]`` or ``[CATEGORY_N_BLOCKED]``
      ``partial``           – first/last chars revealed, middle obscured with ``***``
      ``format_preserving`` – same length, character class preserved (U→X, l→x, d→0,
                              specials unchanged)

  mask_payload_with_vault(text, detections, vault=None)
      Walk *text* span-by-span, apply per-detection mask styles, return
      ``(masked_text, vault)``.  Works with
      :class:`~pii_guard.models.Detection` objects or any mapping/object
      that exposes ``category``, ``action``, ``mask_style``, ``start``,
      ``end``, and ``original`` attributes.

Rehydration
-----------
For the ``tokenize`` mask style the masked text embeds the placeholder
token (``[CATEGORY_N]``), so ``vault.rehydrate(text)`` restores all
known tokens automatically.

For ``partial`` and ``format_preserving`` styles the masked text does
**not** embed a recoverable reference, so automatic text-scan rehydration
is not possible.  The vault still stores ``token→original`` — callers
that know the token can use ``vault.restore(token)`` for explicit
restoration.

Design notes
------------
* NOT thread-safe — designed for single-request, single-threaded use.
  The proxy serialises requests through :attr:`PIIGuardProxy._engine_lock`.
* Raw original values are held in-process memory only; they are never
  written to disk, logs, or ledger entries.
* Idempotent: the same *original* value in the same request always gets
  the same token (first-seen assignment; re-encodes return cached token).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import Action, Detection, MaskStyle


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _partial_mask(value: str, *, reveal_chars: int = 2) -> str:
    """
    Reveal the first and last *reveal_chars* characters; obscure the rest.

    Rules
    -----
    * If ``len(value) <= 4`` — return ``"***"`` (full obscure; too short to
      reveal any chars without leaking the value).
    * Otherwise reveal ``min(reveal_chars, len(value) // 4)`` chars at each
      end to ensure at most ~50 % of the original is shown, with a
      minimum of 1 revealed char per side.
    * The obscured middle is always ``"***"`` regardless of the number of
      hidden characters (no length leakage).

    Parameters
    ----------
    value:
        The raw string to partially mask.
    reveal_chars:
        Maximum number of characters to show at each end (default 2).

    Returns
    -------
    str
        Partially masked string, e.g. ``"al***io"`` for
        ``"alice@corp.io"`` with *reveal_chars*=2 (email domain still
        visible in the original, but this function operates on the raw
        string as a whole).

    Examples
    --------
    >>> _partial_mask("alice@corp.io")
    'al***io'
    >>> _partial_mask("ab")
    '***'
    >>> _partial_mask("abcde")
    'a***e'
    """
    n = len(value)
    if n <= 4:
        return "***"

    # Cap reveal to at most len//4 chars per side (minimum 1)
    reveal = min(reveal_chars, max(1, n // 4))
    return value[:reveal] + "***" + value[-reveal:]


def _format_preserving_mask(value: str) -> str:
    """
    Return a format-preserving mask of *value*.

    Character class mapping
    -----------------------
    * Uppercase letter (A–Z)   → ``'X'``
    * Lowercase letter (a–z)   → ``'x'``
    * Digit (0–9)              → ``'0'``
    * Any other character      → unchanged (preserves punctuation / format)

    The output has exactly the same length as *value* and preserves the
    structural pattern (e.g. email-like, key-like, phone-like) so that
    downstream LLM context understands the format without seeing the real
    data.

    Parameters
    ----------
    value:
        The raw string to format-preservingly mask.

    Returns
    -------
    str
        Format-preserving masked string of the same length.

    Examples
    --------
    >>> _format_preserving_mask("alice@corp.io")
    'xxxxx@xxxx.xx'
    >>> _format_preserving_mask("AKIAIOSFODNN7EXAMPLE")
    'XXXXXXXXXXXX0XXXXXXX'
    >>> _format_preserving_mask("sk-ant-api03-XYZ")
    'xx-xxx-xxx00-XXX'
    """
    parts: List[str] = []
    for ch in value:
        if ch.isupper():
            parts.append("X")
        elif ch.islower():
            parts.append("x")
        elif ch.isdigit():
            parts.append("0")
        else:
            parts.append(ch)
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# RequestVault
# ──────────────────────────────────────────────────────────────────────────────

class RequestVault:
    """
    Per-request scoped mapping store for masking and rehydration.

    Unlike :class:`~pii_guard.session_map.SessionMap` (which is session-scoped
    and shared across multiple requests), a ``RequestVault`` is created fresh
    for each inbound LLM request.  It stores the ``token → original`` mapping
    for all masked values encountered in that single request, enabling
    on-demand rehydration of the upstream LLM response.

    Usage
    -----
    ::

        vault = RequestVault()

        # Mask a value and get back the assigned token
        token = vault.assign_token("alice@corp.io", "EMAIL")
        # → "EMAIL_1"

        # Later, restore the original from the token
        original = vault.restore("EMAIL_1")
        # → "alice@corp.io"

        # Or rehydrate the whole LLM response text
        response = "Reply to [EMAIL_1] for confirmation."
        restored = vault.rehydrate(response)
        # → "Reply to alice@corp.io for confirmation."

    Design notes
    ------------
    * Idempotent: the same *original* value always gets the same token within
      a single request (re-assigns the cached token on duplicate calls).
    * Per-category counter isolation: ``EMAIL_1`` and ``PHONE_1`` use
      independent counters.
    * Thread safety: NOT thread-safe.  The proxy serialises concurrent
      requests via a per-proxy lock; each request gets its own vault.
    * Raw values are held in memory only and are never persisted to disk.
    """

    def __init__(self) -> None:
        self._token_to_original: Dict[str, str] = {}
        self._original_to_token: Dict[str, str] = {}
        self._counters: Dict[str, int] = defaultdict(int)

    # ── Token assignment ──────────────────────────────────────────────────────

    def assign_token(
        self,
        original: str,
        category: str,
        *,
        blocked: bool = False,
    ) -> str:
        """
        Assign a placeholder token to *original* and store it in the vault.

        If *original* has already been assigned a token in this request, the
        existing token is returned unchanged (idempotent).

        Parameters
        ----------
        original:
            The raw PII/secret string to encode.  Must be non-empty.
        category:
            Detection category string (e.g. ``"EMAIL"``).  Case-sensitive.
        blocked:
            When ``True`` the token is suffixed with ``_BLOCKED``, e.g.
            ``"API_KEY_1_BLOCKED"``.

        Returns
        -------
        str
            The assigned token, without surrounding brackets.

        Raises
        ------
        ValueError
            If *original* or *category* is empty.
        """
        if not original:
            raise ValueError("assign_token() called with empty original")
        if not category:
            raise ValueError("assign_token() called with empty category")

        existing = self._original_to_token.get(original)
        if existing is not None:
            return existing

        self._counters[category] += 1
        idx = self._counters[category]
        token = f"{category}_{idx}_BLOCKED" if blocked else f"{category}_{idx}"

        self._token_to_original[token] = original
        self._original_to_token[original] = token
        return token

    # ── Restoration / rehydration ─────────────────────────────────────────────

    def restore(self, token: str) -> Optional[str]:
        """
        Return the original value for *token*, or ``None`` if unknown.

        Parameters
        ----------
        token:
            Placeholder token without brackets, e.g. ``"EMAIL_1"``.

        Returns
        -------
        str | None
            The original raw value, or ``None`` for unknown tokens.
        """
        return self._token_to_original.get(token)

    def rehydrate(self, text: str) -> str:
        """
        Replace all ``[TOKEN]`` placeholders in *text* with their originals.

        Only ``tokenize``-style masked text embeds ``[TOKEN]`` references
        that this method can restore.  ``partial`` and ``format_preserving``
        masked text does not contain embedded token references, so this method
        leaves them unchanged.

        Replacement is performed longest-token-first to avoid partial-token
        shadowing (e.g. ``EMAIL_10`` being confused with ``EMAIL_1``).

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
        for token in sorted(self._token_to_original, key=len, reverse=True):
            original = self._token_to_original[token]
            result = result.replace(f"[{token}]", original)
        return result

    # ── Convenience helpers ───────────────────────────────────────────────────

    def has_token(self, token: str) -> bool:
        """Return ``True`` if *token* was assigned in this vault."""
        return token in self._token_to_original

    def has_original(self, original: str) -> bool:
        """Return ``True`` if *original* was stored in this vault."""
        return original in self._original_to_token

    def token_for(self, original: str) -> Optional[str]:
        """Return the token assigned to *original*, or ``None``."""
        return self._original_to_token.get(original)

    # ── Read-only views ───────────────────────────────────────────────────────

    @property
    def snapshot(self) -> Dict[str, str]:
        """
        Snapshot of the current ``token → original`` mapping.

        Returns a *copy* — mutations do not affect vault state.
        Never persisted to disk.
        """
        return dict(self._token_to_original)

    @property
    def size(self) -> int:
        """Number of distinct values stored in this vault."""
        return len(self._token_to_original)

    @property
    def counters(self) -> Dict[str, int]:
        """Snapshot of per-category counters (category → last assigned index)."""
        return dict(self._counters)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.size

    def __contains__(self, original: str) -> bool:
        """``True`` if *original* has been stored in this vault."""
        return self.has_original(original)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RequestVault(entries={self.size}, "
            f"categories={list(self._counters.keys())})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-style masking helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_mask_style(
    original: str,
    category: str,
    mask_style: MaskStyle,
    vault: RequestVault,
    *,
    blocked: bool = False,
    reveal_chars: int = 2,
) -> str:
    """
    Apply *mask_style* to *original*, persist in *vault*, return masked text.

    The original value is **always** stored in the vault under an assigned
    token, regardless of which mask style is used.  This enables on-demand
    rehydration via :meth:`RequestVault.restore` even for ``partial`` and
    ``format_preserving`` styles (where the masked output text does not
    embed a recoverable token reference).

    Parameters
    ----------
    original:
        The raw PII/secret string to mask.  Must be non-empty.
    category:
        Detection category string (e.g. ``"EMAIL"`` or ``"API_KEY"``).
    mask_style:
        Which masking representation to apply:

        ``MaskStyle.TOKENIZE``
            Returns ``"[CATEGORY_N]"`` (or ``"[CATEGORY_N_BLOCKED]"`` when
            *blocked=True*).  Token is embedded in the masked text so
            :meth:`RequestVault.rehydrate` can restore it automatically.

        ``MaskStyle.PARTIAL``
            Returns a partial representation exposing the first and last
            *reveal_chars* characters, separated by ``"***"``.  Short
            values (≤4 chars) are fully obscured as ``"***"``.

        ``MaskStyle.FORMAT_PRESERVING``
            Returns a format-preserving mask of the same length:
            uppercase letters → ``'X'``, lowercase letters → ``'x'``,
            digits → ``'0'``, punctuation/specials unchanged.

    vault:
        Per-request :class:`RequestVault` instance.  The original is
        stored here under the assigned token.
    blocked:
        When ``True``, the token receives the ``_BLOCKED`` suffix.
        Only meaningful for the ``tokenize`` mask style (the suffix is
        not embedded in ``partial`` or ``format_preserving`` output).
    reveal_chars:
        For ``partial`` style only — number of chars to reveal at each
        end.  Default 2.

    Returns
    -------
    str
        The masked representation.

    Side effects
    ------------
    Calls ``vault.assign_token(original, category, blocked=blocked)``
    unconditionally, so the mapping is always available for retrieval.

    Examples
    --------
    ::

        vault = RequestVault()

        # Tokenize style
        masked = apply_mask_style("alice@corp.io", "EMAIL",
                                  MaskStyle.TOKENIZE, vault)
        # → "[EMAIL_1]"
        vault.restore("EMAIL_1")  # → "alice@corp.io"

        # Partial style
        masked = apply_mask_style("alice@corp.io", "EMAIL",
                                  MaskStyle.PARTIAL, vault)
        # → "al***io"
        vault.restore(vault.token_for("alice@corp.io"))  # → "alice@corp.io"

        # Format-preserving style
        masked = apply_mask_style("AKIAIOSFODNN7EXAMPLE", "AWS_SECRET",
                                  MaskStyle.FORMAT_PRESERVING, vault)
        # → "XXXXXXXXXXXX0XXXXXXX"
        vault.restore(vault.token_for("AKIAIOSFODNN7EXAMPLE"))  # → original
    """
    # Always store in vault (enables explicit token-based restoration for all styles)
    token = vault.assign_token(original, category, blocked=blocked)

    if mask_style == MaskStyle.TOKENIZE:
        return f"[{token}]"
    elif mask_style == MaskStyle.PARTIAL:
        return _partial_mask(original, reveal_chars=reveal_chars)
    elif mask_style == MaskStyle.FORMAT_PRESERVING:
        return _format_preserving_mask(original)
    else:
        # Unknown style — fall back to tokenize (safe default)
        return f"[{token}]"


# ──────────────────────────────────────────────────────────────────────────────
# Full payload masking
# ──────────────────────────────────────────────────────────────────────────────

def _entity_attr(entity: Any, name: str) -> Any:
    """Return *name* from *entity* whether it is an object or a mapping."""
    try:
        return getattr(entity, name)
    except AttributeError:
        return entity[name]


def mask_payload_with_vault(
    text: str,
    detections: List[Any],
    vault: Optional[RequestVault] = None,
) -> Tuple[str, RequestVault]:
    """
    Apply per-detection mask decisions to *text* and return the vault.

    For each detection, the appropriate mask style is applied and the
    original value is persisted in *vault*.  Span-based replacement
    follows the same non-overlapping, start-sorted rules as
    :func:`~pii_guard.masker.maskPayload`.

    Parameters
    ----------
    text:
        The original string to mask.
    detections:
        Collection of detection descriptors.  Each element must expose
        (as attributes or mapping keys):

        ``category`` (str)
            Detection category name.
        ``action`` (str | Action)
            Policy action — used to set the ``blocked`` flag when action
            is ``"block"`` or :attr:`Action.BLOCK`.
        ``mask_style`` (str | MaskStyle)
            One of ``"tokenize"``, ``"partial"``, or
            ``"format_preserving"``.
        ``start`` (int)
            Inclusive start offset in *text*.
        ``end`` (int)
            Exclusive end offset in *text*.
        ``original`` (str)
            Raw matched string.

        Both :class:`~pii_guard.models.Detection` objects and plain
        ``dict`` mappings are accepted.

    vault:
        Existing :class:`RequestVault` to extend.  A fresh vault is
        created and returned when ``None`` (default).

    Returns
    -------
    tuple[str, RequestVault]
        ``(masked_text, vault)`` where *masked_text* has all
        non-overlapping spans replaced per their mask style, and
        *vault* holds the complete ``token → original`` mapping for
        all replaced spans.

    Notes
    -----
    * Zero-length or inverted spans are silently skipped.
    * Overlapping spans: the span with the earlier start position wins;
      the later span is skipped and absent from the vault.
    * Detections whose action is ``"allow"`` / :attr:`Action.ALLOW` are
      left in-place (text is unchanged for that span).
    """
    if vault is None:
        vault = RequestVault()

    if not detections:
        return text, vault

    # Sort by start position (stable)
    sorted_dets = sorted(detections, key=lambda d: _entity_attr(d, "start"))

    parts: List[str] = []
    cursor = 0
    accepted: List[Tuple[int, int]] = []  # accepted (start, end) pairs

    for det in sorted_dets:
        start: int = _entity_attr(det, "start")
        end: int = _entity_attr(det, "end")
        original: str = _entity_attr(det, "original")
        category: str = _entity_attr(det, "category")

        # Get action and mask_style — accept enum or string
        raw_action = _entity_attr(det, "action")
        action_val: str = raw_action.value if hasattr(raw_action, "value") else str(raw_action)

        raw_style = _entity_attr(det, "mask_style")
        # Normalise mask_style to MaskStyle enum
        if isinstance(raw_style, MaskStyle):
            mask_style = raw_style
        else:
            try:
                mask_style = MaskStyle(str(raw_style))
            except ValueError:
                mask_style = MaskStyle.TOKENIZE  # safe fallback

        # Skip zero-length or inverted spans
        if end <= start:
            continue

        # Skip overlapping spans
        overlaps = any(
            not (end <= s or start >= e)
            for s, e in accepted
        )
        if overlaps:
            continue

        # Append unchanged text before this span
        if start > cursor:
            parts.append(text[cursor:start])

        if action_val == Action.ALLOW.value:
            # Allowed — keep original text
            parts.append(original)
        else:
            # Determine blocked flag
            blocked = action_val == Action.BLOCK.value

            # Apply mask style via vault
            masked = apply_mask_style(
                original, category, mask_style, vault, blocked=blocked
            )
            parts.append(masked)

        cursor = end
        accepted.append((start, end))

    # Append any trailing text after the last span
    if cursor < len(text):
        parts.append(text[cursor:])

    return "".join(parts), vault
