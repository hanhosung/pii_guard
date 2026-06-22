"""
Masking / redaction engine.

Takes the output of detector.scan_text() and applies the per-category action:

  BLOCK              → replace with [CATEGORY_N_BLOCKED]
  TOKENIZE_ROUNDTRIP → replace with [CATEGORY_N] (restoration map kept in memory)
  MASK               → replace with [CATEGORY_N] (alias of tokenize for Stage1)
  ALLOW              → leave original text unchanged

Session-consistent placeholder assignment is delegated to SessionMap so that
the same value always gets the same placeholder within a session.

Pure masking API
----------------
``maskPayload(text, detectedEntities)`` is a standalone stateless function
(no SessionMap, no Engine) intended for callers that already have a
pre-built entity list and just need span substitution with an indexed
placeholder and a corresponding reverse-mapping store.  Each call starts
its own per-category counters from 1.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import Action, Detection, RedactionResult
from .session_map import SessionMap


# ──────────────────────────────────────────────────────────────────────────────
# Pure maskPayload  (Sub-AC 2b-i)
# ──────────────────────────────────────────────────────────────────────────────

def _entity_attr(entity: Any, name: str) -> Any:
    """Return *name* from *entity* whether it is an object or a mapping."""
    try:
        return getattr(entity, name)
    except AttributeError:
        return entity[name]


def maskPayload(
    text: str,
    detectedEntities: List[Any],
) -> Tuple[str, Dict[str, str]]:
    """
    Replace every detected entity span in *text* with an indexed
    ``[CATEGORY_N]`` placeholder and return both the masked string and a
    reverse-mapping store.

    This is a **pure, stateless function**: each call initialises its own
    per-category counters starting from 1.  It does *not* use or mutate any
    SessionMap instance — call :func:`apply_redactions` when you need
    cross-turn session consistency.

    Parameters
    ----------
    text:
        The original string whose spans should be replaced.
    detectedEntities:
        An ordered or unordered collection of entity descriptors.  Each
        element must expose the following four attributes (or mapping keys):

        ``category`` (str)
            Detection category name, e.g. ``"EMAIL"`` or ``"API_KEY"``.
        ``start`` (int)
            Inclusive start offset of the match in *text*.
        ``end`` (int)
            Exclusive end offset of the match in *text*.
        ``original`` (str)
            The raw matched string (must equal ``text[start:end]`` for
            correct reconstruction, but the function trusts the caller).

        Both :class:`~pii_guard.models.Detection` objects and plain
        ``dict``/``TypedDict`` mappings are accepted.

    Returns
    -------
    tuple[str, dict[str, str]]
        ``(masked_text, reverse_mapping)`` where:

        * ``masked_text`` — *text* with every non-overlapping entity span
          replaced by ``[CATEGORY_N]``.
        * ``reverse_mapping`` — ``{placeholder_token: original_value}``
          containing **exactly one entry per replaced entity** and no extras.
          Keys are bare token strings (no brackets), e.g. ``"EMAIL_1"``.

    Notes
    -----
    * Overlapping spans: the entity that starts earlier in the text wins;
      any later entity whose range overlaps an already-accepted one is
      silently skipped (and therefore absent from ``reverse_mapping``).
    * Per-category counters are independent and monotonically increasing
      within a single call: ``EMAIL_1, EMAIL_2, …`` and
      ``PHONE_1, PHONE_2, …`` do not share a counter.
    * Empty entity list: returns ``(text, {})``.
    * Raw original values are **never** persisted to disk by this function.
    """
    if not detectedEntities:
        return text, {}

    # Sort by start position (ascending); ties broken by entity order (stable sort)
    sorted_entities = sorted(detectedEntities, key=lambda e: _entity_attr(e, "start"))

    counters: Dict[str, int] = defaultdict(int)
    reverse_mapping: Dict[str, str] = {}

    parts: List[str] = []
    cursor = 0
    # Track accepted (start, end) intervals for overlap detection
    accepted: List[Tuple[int, int]] = []

    for entity in sorted_entities:
        start: int = _entity_attr(entity, "start")
        end: int = _entity_attr(entity, "end")
        original: str = _entity_attr(entity, "original")
        category: str = _entity_attr(entity, "category")

        # Skip zero-length or inverted spans
        if end <= start:
            continue

        # Skip if this span overlaps any already-accepted interval
        overlaps = any(
            not (end <= s or start >= e)
            for s, e in accepted
        )
        if overlaps:
            continue

        # Append unchanged text between cursor and this entity
        if start > cursor:
            parts.append(text[cursor:start])

        # Assign next index for this category (isolated per-category counter)
        counters[category] += 1
        token = f"{category}_{counters[category]}"

        parts.append(f"[{token}]")
        reverse_mapping[token] = original
        cursor = end
        accepted.append((start, end))

    # Append any trailing text after the last replaced span
    if cursor < len(text):
        parts.append(text[cursor:])

    return "".join(parts), reverse_mapping


def apply_redactions(
    text: str,
    detections: List[Detection],
    session_map: Optional[SessionMap] = None,
) -> RedactionResult:
    """
    Apply all detections to *text* and return a RedactionResult.

    Parameters
    ----------
    text:
        The original string to redact.
    detections:
        Sorted list of Detection objects from scan_text().
    session_map:
        :class:`SessionMap` instance for this session.  Pass the same object
        across calls to maintain cross-turn consistency (same value → same
        placeholder).  A fresh temporary map is created if not supplied.

    Returns
    -------
    :class:`RedactionResult` with redacted_text, detections, and restoration map.
    """
    if session_map is None:
        session_map = SessionMap()

    result = RedactionResult(
        original_text=text,
        redacted_text="",
    )

    # Sort detections by start position (should already be sorted)
    sorted_dets = sorted(detections, key=lambda d: d.start)

    parts: List[str] = []
    cursor = 0

    for det in sorted_dets:
        # Copy unchanged text before this detection
        if det.start > cursor:
            parts.append(text[cursor:det.start])

        if det.action == Action.ALLOW:
            parts.append(det.original)
            cursor = det.end
            result.add_detection(det)
            continue

        # Determine whether this is a blocked item
        is_blocked = det.action == Action.BLOCK

        # Delegate to SessionMap for consistent placeholder assignment
        token = session_map.encode(det.original, det.category, blocked=is_blocked)
        det.placeholder_token = token

        parts.append(f"[{token}]")
        cursor = det.end
        result.add_detection(det)

    # Append any trailing text
    if cursor < len(text):
        parts.append(text[cursor:])

    result.redacted_text = "".join(parts)
    # Expose a snapshot of the restoration map in the result
    result._restoration_map = session_map.restoration_map
    return result


def rehydrate_text(text: str, restoration_map: Dict[str, str]) -> str:
    """
    Replace [PLACEHOLDER] tokens in *text* with their original values.

    Used for inbound LLM responses so the agent receives real values.
    Terminal output restoration must remain OFF (the caller decides whether
    to call this).

    Parameters
    ----------
    text:
        String possibly containing ``[CATEGORY_N]`` tokens.
    restoration_map:
        Mapping of ``placeholder_token → original_value``.

    Returns
    -------
    str with all known placeholders restored.
    """
    result = text
    # Sort by placeholder length descending to avoid substring conflicts
    for placeholder in sorted(restoration_map, key=len, reverse=True):
        original = restoration_map[placeholder]
        result = result.replace(f"[{placeholder}]", original)
    return result
