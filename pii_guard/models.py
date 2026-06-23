"""
Data models for PII-Guard detection results.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CategoryClass(str, Enum):
    """High-level category class."""
    PII = "pii"
    KOREAN_PII = "korean_pii"
    SECRET = "secret"
    CUSTOM = "custom"


class Action(str, Enum):
    """Policy action for a detected item."""
    ALLOW = "allow"
    MASK = "mask"
    BLOCK = "block"
    TOKENIZE_ROUNDTRIP = "tokenize_roundtrip"


class MaskStyle(str, Enum):
    """How a masked item is represented."""
    TOKENIZE = "tokenize"          # [EMAIL_1] placeholder
    PARTIAL = "partial"            # first/last chars shown
    FORMAT_PRESERVING = "format_preserving"  # same length, char class preserved


class DetectionStage(str, Enum):
    """Which detection stage found this."""
    STAGE1_REGEX_CHECKSUM = "stage1_regex_checksum"
    STAGE1_PROXIMITY = "stage1_proximity"
    STAGE2_NER = "stage2_ner"


@dataclass
class Detection:
    """A single PII/secret detection hit."""

    # What was detected
    category: str               # e.g. "EMAIL", "AWS_SECRET"
    category_class: CategoryClass
    action: Action
    mask_style: MaskStyle

    # Where in the text
    start: int
    end: int
    original: str               # raw matched text (never persisted to ledger)

    # Detection metadata
    detection_stage: DetectionStage
    rule_id: str                # which pattern/rule fired
    confidence: float           # 0.0 – 1.0

    # Output
    placeholder_token: str = ""  # e.g. "EMAIL_1"

    def keyed_hash(self, hmac_key: bytes) -> str:
        """HMAC-SHA256 of normalised original — for ledger correlation only."""
        normalised = self.original.strip().lower()
        return hmac.new(hmac_key, normalised.encode(), hashlib.sha256).hexdigest()

    def char_class_signature(self) -> str:
        """Replace chars with class abbreviations for ledger (not recoverable)."""
        def cls(c: str) -> str:
            if c.isupper():
                return "U"
            if c.islower():
                return "l"
            if c.isdigit():
                return "d"
            return "s"
        sig = "".join(cls(c) for c in self.original)
        # Collapse runs
        out, prev, count = [], None, 0
        for ch in sig:
            if ch == prev:
                count += 1
            else:
                if prev is not None:
                    out.append(f"{prev}{count}" if count > 1 else prev)
                prev, count = ch, 1
        if prev:
            out.append(f"{prev}{count}" if count > 1 else prev)
        return "".join(out)

    def span_length(self) -> int:
        return self.end - self.start


@dataclass
class RedactionResult:
    """Result of scanning and redacting a text block."""
    original_text: str
    redacted_text: str
    detections: List[Detection] = field(default_factory=list)
    coverage_gap: bool = False  # True if content passed unscanned

    # Stage-2 NER metadata (populated when Stage-2 is enabled)
    # Human-readable reason Stage-2 failed (None when Stage-2 succeeded or was
    # not enabled).  Never contains raw PII — only error type / timing info.
    stage2_gap_reason: Optional[str] = None

    # Restoration map: placeholder → original (held in memory, never to disk)
    _restoration_map: dict = field(default_factory=dict, repr=False)

    def add_detection(self, det: Detection) -> None:
        self.detections.append(det)
        if det.placeholder_token:
            self._restoration_map[det.placeholder_token] = det.original

    def rehydrate(self, text: str) -> str:
        """Replace placeholders with original values (for inbound LLM responses)."""
        result = text
        for placeholder, original in self._restoration_map.items():
            result = result.replace(f"[{placeholder}]", original)
        return result

    @property
    def has_blocks(self) -> bool:
        return any(d.action == Action.BLOCK for d in self.detections)

    @property
    def has_masks(self) -> bool:
        return any(d.action in (Action.MASK, Action.TOKENIZE_ROUNDTRIP)
                   for d in self.detections)

    def summary(self) -> dict:
        """Non-PII summary for logging."""
        from collections import Counter
        counts = Counter(d.category for d in self.detections)
        actions = Counter(d.action.value for d in self.detections)
        return {
            "total_detections": len(self.detections),
            "categories": dict(counts),
            "actions": dict(actions),
            "coverage_gap": self.coverage_gap,
        }
