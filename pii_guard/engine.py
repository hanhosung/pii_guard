"""
PII-Guard Stage-1 text scanning engine — top-level API.

Usage::

    from pii_guard.engine import Engine

    engine = Engine()
    result = engine.scan("Send the report to alice@example.com")
    print(result.redacted_text)   # "Send the report to [EMAIL_1]"
    print(result.summary())       # {"total_detections": 1, "categories": {...}, ...}

    # Rehydrate an inbound LLM response (round-trip restoration):
    restored = engine.rehydrate(llm_response_text)

Session-level state (placeholder counters and the restoration map) is held
inside the :class:`SessionMap` owned by this :class:`Engine` instance so the
same real value always produces the same placeholder within a session.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional

from .categories import ALL_CATEGORIES, CategorySpec
from .detector import scan_text
from .masker import apply_redactions, rehydrate_text
from .models import RedactionResult
from .session_map import SessionMap

if TYPE_CHECKING:
    from .stage2.runner import Stage2NERRunner


class Engine:
    """
    Stateful scanning engine for a single session.

    Parameters
    ----------
    categories:
        Override the default category list.
    allowlist_patterns:
        Compiled regex patterns; matches are skipped (project allow-list).
    min_confidence_override:
        Hard minimum confidence; rules below this are ignored.
    hmac_key:
        Secret bytes for keyed-hash ledger correlation.  A random key is
        generated at startup and never persisted.
    stage2_runner:
        Optional :class:`~pii_guard.stage2.runner.Stage2NERRunner` instance.
        When provided, Stage-2 NER is attempted after Stage-1 for each block.
        On Stage-2 failure the engine degrades gracefully to Stage-1 results
        and sets ``coverage_gap=True`` with ``stage2_gap_reason`` on the result.
        Pass ``None`` (default) to run Stage-1 only.
    """

    def __init__(
        self,
        categories: Optional[List[CategorySpec]] = None,
        allowlist_patterns: Optional[List[re.Pattern]] = None,
        min_confidence_override: Optional[float] = None,
        hmac_key: Optional[bytes] = None,
        stage2_runner: Optional["Stage2NERRunner"] = None,
    ) -> None:
        import os
        self._categories = categories or ALL_CATEGORIES
        self._allowlist = allowlist_patterns or []
        self._min_confidence = min_confidence_override
        self._hmac_key: bytes = hmac_key or os.urandom(32)

        # Optional Stage-2 NER runner (subprocess-isolated)
        self._stage2_runner = stage2_runner

        # Per-session mutable state — never written to disk
        # All placeholder assignment is delegated to SessionMap
        self._session_map = SessionMap()

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, text: str) -> RedactionResult:
        """
        Scan *text* and return a RedactionResult with the redacted version
        and full detection metadata.

        Detection pipeline
        ------------------
        1. **Stage 1** (always): regex + checksum scanning via
           :func:`~pii_guard.detector.scan_text`.
        2. **Stage 2** (optional): NER via the subprocess runner set at
           construction.  On any Stage-2 failure the engine falls back to
           Stage-1 results, sets ``result.coverage_gap = True``, and records
           the failure in ``result.stage2_gap_reason``.

        Returns
        -------
        RedactionResult
        """
        if not isinstance(text, str):
            raise TypeError(f"scan() expects str, got {type(text).__name__}")

        # ── Stage 1: regex / checksum ─────────────────────────────────────────
        stage1_detections = scan_text(
            text,
            categories=self._categories,
            allowlist_patterns=self._allowlist,
            min_confidence_override=self._min_confidence,
        )

        # ── Stage 2: NER (subprocess-isolated, optional) ──────────────────────
        final_detections = stage1_detections
        stage2_gap_reason: Optional[str] = None

        if self._stage2_runner is not None:
            s2 = self._stage2_runner.scan(text, stage1_detections)
            if s2.coverage_gap:
                # Stage-2 failed → keep Stage-1 detections + annotate gap
                stage2_gap_reason = s2.fail_reason
                # final_detections stays as stage1_detections
            else:
                # Stage-2 succeeded → use merged detections
                final_detections = s2.detections

        # ── Apply redactions ──────────────────────────────────────────────────
        result = apply_redactions(
            text,
            final_detections,
            session_map=self._session_map,
        )

        # Annotate with Stage-2 gap information if applicable
        if stage2_gap_reason is not None:
            result.coverage_gap = True
            result.stage2_gap_reason = stage2_gap_reason

        return result

    def rehydrate(self, text: str) -> str:
        """
        Replace [PLACEHOLDER] tokens in an inbound LLM response with the
        original values from this session's restoration map.

        NOTE: terminal output restoration must remain OFF — the proxy calls
        this only for agent-visible content, never for user-visible output.
        """
        return self._session_map.rehydrate(text)

    @property
    def session_map(self) -> SessionMap:
        """Direct access to the underlying :class:`SessionMap` for this session."""
        return self._session_map

    @property
    def restoration_map(self) -> Dict[str, str]:
        """Read-only snapshot of the current session's placeholder→original map."""
        return self._session_map.restoration_map

    def reset_session(self) -> None:
        """Clear per-session state (counters and restoration map)."""
        self._session_map.reset()

    def add_allowlist(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern string to the project allow-list."""
        self._allowlist.append(re.compile(pattern, flags))
