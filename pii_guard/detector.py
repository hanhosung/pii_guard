"""
Stage 1 detection engine — pure regex + checksum scanning.

Scans a text string against all registered category patterns and returns
a list of Detection objects sorted by position.  Overlapping matches are
resolved by preferring the category that appears first in ALL_CATEGORIES
(i.e. higher-priority categories win) and then by longest match.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .categories import ALL_CATEGORIES, CATEGORY_MAP, CategorySpec, PatternRule
from .models import Action, Detection, DetectionStage


# Default HMAC key — callers should supply a real secret via Engine
_DEFAULT_HMAC_KEY = b"pii-guard-default-do-not-use-in-prod"


def _resolve_capture_group(pattern: re.Pattern, m: re.Match) -> Tuple[int, int, str]:
    """
    Return (start, end, text) for the best span:
    - If the pattern has exactly one capture group and it matched, use group(1).
    - Otherwise use the full match (group(0)).
    """
    if m.lastindex and m.lastindex >= 1 and m.group(1) is not None:
        return m.start(1), m.end(1), m.group(1)
    return m.start(0), m.end(0), m.group(0)


def scan_text(
    text: str,
    categories: Optional[List[CategorySpec]] = None,
    allowlist_patterns: Optional[List[re.Pattern]] = None,
    min_confidence_override: Optional[float] = None,
) -> List[Detection]:
    """
    Run Stage-1 pattern scan on *text*.

    Parameters
    ----------
    text:
        Raw string to scan.
    categories:
        Which CategorySpec objects to apply (default: ALL_CATEGORIES).
    allowlist_patterns:
        Compiled regex patterns; any match whose full text matches one of these
        is skipped (project-scoped allow-list).
    min_confidence_override:
        If provided, ignore any rule with confidence below this threshold.

    Returns
    -------
    List of Detection objects sorted by start position.  Overlapping ranges
    are deduplicated: the first (higher-priority category) match wins.
    """
    if categories is None:
        categories = ALL_CATEGORIES
    if allowlist_patterns is None:
        allowlist_patterns = []

    raw_hits: List[Tuple[int, Detection]] = []  # (priority_index, detection)

    for cat_idx, cat_spec in enumerate(categories):
        for rule in cat_spec.rules:
            effective_min = min_confidence_override if min_confidence_override is not None \
                else cat_spec.min_confidence
            if rule.confidence < effective_min:
                continue

            for m in rule.pattern.finditer(text):
                start, end, matched = _resolve_capture_group(rule.pattern, m)

                # Skip empty captures
                if not matched:
                    continue

                # Apply project allowlist
                if any(ap.search(matched) for ap in allowlist_patterns):
                    continue

                # Run optional checksum / Luhn validator
                if rule.validator is not None and not rule.validator(matched):
                    continue

                det = Detection(
                    category=cat_spec.category,
                    category_class=cat_spec.category_class,
                    action=cat_spec.action,
                    mask_style=cat_spec.mask_style,
                    start=start,
                    end=end,
                    original=matched,
                    detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
                    rule_id=rule.rule_id,
                    confidence=rule.confidence,
                )
                raw_hits.append((cat_idx, det))

    # Resolve overlaps: sort by (start, cat_idx, -span_length).
    # Priority order: earlier position → higher-priority category (lower cat_idx)
    # → longer span within the same category.
    # This ensures e.g. CARD beats ADDRESS when both start at the same position.
    raw_hits.sort(key=lambda x: (x[1].start, x[0], -(x[1].end - x[1].start)))

    kept: List[Detection] = []
    occupied: List[Tuple[int, int]] = []  # (start, end) of accepted matches

    for _, det in raw_hits:
        # Check for overlap with any already-accepted detection
        overlap = any(
            not (det.end <= s or det.start >= e)
            for s, e in occupied
        )
        if not overlap:
            kept.append(det)
            occupied.append((det.start, det.end))

    # Return sorted by position
    kept.sort(key=lambda d: d.start)
    return kept
