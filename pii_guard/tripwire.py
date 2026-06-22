"""
Full-body tripwire sweep — Sub-AC 8.2.

Runs a PII-class regex/pattern sweep over the raw serialised request body
(or any fields not covered by the structured parser output), flags hits,
and surfaces them alongside the structured-parse results.

This is a *complementary* scanner — it does not replace the structured
per-provider parsers.  Its role is to catch PII that slips through in:

  - Non-standard fields added by proxies, middleware, or custom clients
    (e.g. ``metadata.user_email``, ``x_custom_context``, ``debug_info``)
  - Deeply nested objects absent from the provider's known schema
  - Fields added by future API versions not yet tracked by the provider
    parsers
  - Any text region in the serialised body not visited by the provider
    parser's structured walk

Algorithm
---------
The sweep runs all Stage-1 category patterns directly against the raw JSON
string.  Because the structured scrubber has already replaced PII in *known*
fields with ``[PLACEHOLDER_N]`` tokens before the tripwire runs, any PII
remaining in the serialised sanitised payload definitively represents a
coverage gap — content the structured parser did not visit.

Overlap resolution uses the same priority + longest-match strategy as
:func:`~pii_guard.detector.scan_text`: higher-priority (earlier in
``ALL_CATEGORIES``) categories win; ties go to the longer span.

Usage
-----
Run the tripwire on the *sanitised* payload (after the provider scrubber has
already masked known fields) to surface the structural gaps::

    import json
    from pii_guard.tripwire import sweep_raw_body

    scrub_result = scrub_claude_request(payload, engine)
    sanitised_json = json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)
    tripwire = sweep_raw_body(sanitised_json)

    if tripwire.should_block:
        # Residual BLOCK-category PII found in a non-standard field — block
        ...
    if tripwire.has_detections:
        # Residual PII found — log coverage gap
        ...

Or sweep the original raw body to see *all* PII in the entire payload::

    raw_json = body_bytes.decode("utf-8")
    tripwire = sweep_raw_body(raw_json)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .categories import ALL_CATEGORIES, CategorySpec
from .models import Action, CategoryClass, DetectionStage


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TripwireHit:
    """
    A single PII/secret hit found in the raw serialised body.

    Attributes
    ----------
    category:
        Detection category (e.g. ``"EMAIL"``, ``"API_KEY"``).
    category_class:
        High-level class — ``pii``, ``korean_pii``, or ``secret``.
    action:
        Policy action for this category — ``BLOCK``, ``MASK``, or
        ``TOKENIZE_ROUNDTRIP``.
    rule_id:
        Identifier of the specific pattern rule that fired.
    confidence:
        Rule confidence score (0.0–1.0).
    matched_text:
        Raw matched text found in the serialised body.  **Never persist
        this field to the Ledger** — log only the HMAC hash for audit.
    raw_offset:
        Character offset of the match start within the raw body string.
    raw_end:
        Character offset of the match end (exclusive).
    detection_stage:
        Always ``STAGE1_REGEX_CHECKSUM`` for tripwire hits — the sweep
        uses the same Stage-1 patterns as the structured parser.
    """

    category: str
    category_class: CategoryClass
    action: Action
    rule_id: str
    confidence: float
    matched_text: str          # raw match — never persist; log HMAC only
    raw_offset: int            # char offset in the raw body
    raw_end: int               # end offset (exclusive)
    detection_stage: DetectionStage = DetectionStage.STAGE1_REGEX_CHECKSUM

    @property
    def span_length(self) -> int:
        """Length of the matched span in characters."""
        return self.raw_end - self.raw_offset


@dataclass
class TripwireResult:
    """
    Result of running the full-body tripwire sweep on a serialised body.

    Attributes
    ----------
    hits:
        All PII/secret hits found, sorted by position.  This list covers
        the *entire* body; when the sweep is run on the sanitised payload
        (after structured scrubbing), every hit in this list represents a
        field the structured parser did not visit — a true coverage gap.
    """

    hits: List[TripwireHit] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        """``True`` if any hit carries a ``BLOCK`` action."""
        return any(h.action == Action.BLOCK for h in self.hits)

    @property
    def has_detections(self) -> bool:
        """``True`` if the sweep found at least one PII/secret hit."""
        return bool(self.hits)

    @property
    def block_hits(self) -> List[TripwireHit]:
        """Hits whose action is ``BLOCK``."""
        return [h for h in self.hits if h.action == Action.BLOCK]

    @property
    def mask_hits(self) -> List[TripwireHit]:
        """Hits whose action is ``MASK`` or ``TOKENIZE_ROUNDTRIP``."""
        return [
            h for h in self.hits
            if h.action in (Action.MASK, Action.TOKENIZE_ROUNDTRIP)
        ]

    def summary(self) -> dict:
        """
        Non-PII summary suitable for audit / Ledger logging.

        **Does not include raw matched text** — callers must log only
        HMAC-keyed hashes of matched values via
        :meth:`~pii_guard.models.Detection.keyed_hash`.
        """
        from collections import Counter
        category_counts = Counter(h.category for h in self.hits)
        action_counts = Counter(h.action.value for h in self.hits)
        return {
            "tripwire_hits": len(self.hits),
            "categories": dict(category_counts),
            "actions": dict(action_counts),
            "should_block": self.should_block,
            "has_detections": self.has_detections,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep function
# ─────────────────────────────────────────────────────────────────────────────

def sweep_raw_body(
    raw_body: str,
    *,
    categories: Optional[List[CategorySpec]] = None,
    allowlist_patterns: Optional[List[re.Pattern]] = None,
    min_confidence_override: Optional[float] = None,
) -> TripwireResult:
    """
    Run a PII-class regex/pattern sweep over a raw serialised body string.

    Unlike the structured per-provider parsers (which walk the JSON schema
    and scan only the *known* text-bearing fields they enumerate), this
    function applies all category patterns against the full body string in
    one pass.  PII embedded in any non-standard, deeply nested, or future
    API field is therefore caught regardless of its structural location.

    Overlap resolution
    ------------------
    When multiple patterns match at the same offset, the higher-priority
    category (earlier position in *categories*) wins.  Within the same
    category a longer span wins.  This is identical to the behaviour of
    :func:`~pii_guard.detector.scan_text`.

    Parameters
    ----------
    raw_body:
        Raw serialised request body — typically a JSON string, but any
        text is valid.  Must be ``str`` (not ``bytes``).  When sweeping
        the *sanitised* payload (recommended), the caller should
        ``json.dumps(scrub_result.sanitized_payload, ensure_ascii=False)``
        first so that placeholder tokens are preserved in the body and
        only genuinely unscanned PII remains.
    categories:
        Which :class:`~pii_guard.categories.CategorySpec` objects to
        apply.  Defaults to :data:`~pii_guard.categories.ALL_CATEGORIES`.
    allowlist_patterns:
        Compiled :class:`re.Pattern` objects; any match whose full
        matched text matches one of these is skipped (project-scoped
        allow-list, identical to the allow-list in
        :func:`~pii_guard.detector.scan_text`).
    min_confidence_override:
        Hard minimum confidence; rules below this threshold are ignored.
        When ``None`` (default) each category's own ``min_confidence``
        is used.

    Returns
    -------
    TripwireResult
        All PII/secret hits found in *raw_body*, sorted by position.
        Overlapping hits are deduplicated using the priority + longest-match
        strategy described above.

    Raises
    ------
    TypeError
        If *raw_body* is not a ``str``.
    """
    if not isinstance(raw_body, str):
        raise TypeError(
            f"sweep_raw_body() expects str, got {type(raw_body).__name__}"
        )

    if categories is None:
        categories = ALL_CATEGORIES
    if allowlist_patterns is None:
        allowlist_patterns = []

    # Collect raw hits as (category_priority_index, TripwireHit) tuples so
    # the overlap resolver can apply the same priority ordering as the
    # structured detector.
    raw_hits: List[Tuple[int, TripwireHit]] = []

    for cat_idx, cat_spec in enumerate(categories):
        for rule in cat_spec.rules:
            effective_min = (
                min_confidence_override
                if min_confidence_override is not None
                else cat_spec.min_confidence
            )
            if rule.confidence < effective_min:
                continue

            for m in rule.pattern.finditer(raw_body):
                # Resolve capture group — same logic as detector._resolve_capture_group
                if m.lastindex and m.lastindex >= 1 and m.group(1) is not None:
                    start, end, matched = m.start(1), m.end(1), m.group(1)
                else:
                    start, end, matched = m.start(0), m.end(0), m.group(0)

                if not matched:
                    continue

                # Apply project allow-list
                if any(ap.search(matched) for ap in allowlist_patterns):
                    continue

                # Run optional checksum/Luhn/RRN validator
                if rule.validator is not None and not rule.validator(matched):
                    continue

                hit = TripwireHit(
                    category=cat_spec.category,
                    category_class=cat_spec.category_class,
                    action=cat_spec.action,
                    rule_id=rule.rule_id,
                    confidence=rule.confidence,
                    matched_text=matched,
                    raw_offset=start,
                    raw_end=end,
                )
                raw_hits.append((cat_idx, hit))

    # ── Resolve overlaps ──────────────────────────────────────────────────────
    # Sort by (start_offset, category_priority, -span_length):
    #   - Earlier positions come first
    #   - At the same position, higher-priority (lower cat_idx) category wins
    #   - Ties within the same category resolved by longest span
    raw_hits.sort(
        key=lambda x: (x[1].raw_offset, x[0], -(x[1].raw_end - x[1].raw_offset))
    )

    kept: List[TripwireHit] = []
    occupied: List[Tuple[int, int]] = []  # (start, end) of accepted spans

    for _, hit in raw_hits:
        overlap = any(
            not (hit.raw_end <= s or hit.raw_offset >= e)
            for s, e in occupied
        )
        if not overlap:
            kept.append(hit)
            occupied.append((hit.raw_offset, hit.raw_end))

    # Return hits sorted by position for deterministic output
    kept.sort(key=lambda h: h.raw_offset)

    return TripwireResult(hits=kept)
