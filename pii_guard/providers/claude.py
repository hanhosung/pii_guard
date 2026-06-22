"""
Claude wire format traversal and payload reconstruction (Sub-AC 2a).

Walks every text-bearing location in Anthropic's Messages API request schema
and pipes each extracted text span through the detection engine, then rebuilds
a fully sanitized Claude API request payload.

Text-bearing locations covered
------------------------------
  system                  - string or TextBlock array
  messages[*].content     - string or content block array:
      text                  .text
      tool_use              .input (recursively scans all string leaf values)
      tool_result           .content (string or TextBlock array)
      document              .source.data when source.type == "text"
      image                 unscannable → coverage gap / block per policy
  Unknown block types / unrecognized API fields → coverage alarm

Scan-field taxonomy (maps to ontology scan_field)
--------------------------------------------------
  SYSTEM_PROMPT     system prompt text
  MESSAGE_TEXT      user / assistant text block
  TOOL_USE_INPUT    tool_use .input string values
  TOOL_RESULT       tool_result .content
  DOCUMENT_BLOCK    document .source.data
  IMAGE             image block (unscannable Stage-1)
  UNKNOWN           unrecognized block type

Failure semantics
-----------------
  - Any BLOCK-category detection → should_block=True on the returned result.
  - Image / base64 document sources are unscannable; the default action is
    "block" (fail-closed), recording a coverage gap event.
  - Unknown block types raise a coverage alarm; unknown_field_action="block"
    (default/strict) → should_block=True.
  - All non-text leaf values (numbers, booleans, None) in tool_use.input are
    passed through untouched — they carry no text and need no scanning.

Usage
-----
    from pii_guard import Engine
    from pii_guard.providers.claude import scrub_claude_request

    engine = Engine()
    result = scrub_claude_request(payload, engine)
    if result.should_block:
        # Return 400 to the client; do not forward payload
        ...
    else:
        forward(result.sanitized_payload)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..engine import Engine
from ..models import Action


# Sentinel redacted text used when the scanner fails — callers must check
# should_block before using the sanitized payload.
_SCAN_ERROR_REDACTED = ""


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

class ScanField(str, Enum):
    """Ontology scan_field values for Claude-specific locations."""
    SYSTEM_PROMPT  = "system_prompt"
    MESSAGE_TEXT   = "message_text"
    TOOL_USE_INPUT = "tool_use_input"
    TOOL_RESULT    = "tool_result"
    DOCUMENT_BLOCK = "document_block"
    IMAGE          = "image"
    UNKNOWN        = "unknown"


@dataclass
class FieldScanEvent:
    """Audit record for a single text span that was scanned (or not)."""

    scan_field: ScanField
    # Dot-notation location in the payload, e.g. "messages[0].content[2].input.email"
    location: str
    # Detection hits from Stage-1 (empty if nothing found)
    detections: list = field(default_factory=list)
    # Text after redaction (may be unchanged if no detections)
    redacted_text: str = ""
    # True if content was present but could not be text-scanned
    coverage_gap: bool = False
    # True if block-action detection found — request must be blocked
    should_block: bool = False
    # Non-None when the scan could not complete (error/timeout/unavailable).
    # Maps to ledger_event.fail_reason in the ontology.
    fail_reason: Optional[str] = None


@dataclass
class ClaudeRequestScrubResult:
    """Full result of scrubbing one Claude Messages API request payload."""

    # Deep-copy of the original payload with PII/secrets replaced in-place
    sanitized_payload: Dict[str, Any]
    # Per-field audit events (ledger source)
    field_events: List[FieldScanEvent]
    # True → caller must NOT forward; return 400 to the client
    should_block: bool
    # Dot-notation paths where content passed without text scanning
    coverage_gaps: List[str]
    # Dot-notation paths + descriptions of unrecognized fields / block types
    unknown_fields: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def scrub_claude_request(
    payload: Dict[str, Any],
    engine: Engine,
    *,
    unknown_field_action: str = "block",
    unscannable_action: str = "block",
) -> ClaudeRequestScrubResult:
    """
    Scrub a Claude Messages API request payload.

    Walks every text-bearing location, pipes each text span through *engine*
    (Stage-1 detection), and rebuilds a sanitized payload with PII/secrets
    replaced by indexed placeholders or BLOCKED tokens.

    Parameters
    ----------
    payload:
        Original Claude Messages API request dict (will not be mutated).
    engine:
        A PII-Guard Engine instance.  Stateful: maintains session placeholder
        map so the same real value always maps to the same placeholder within
        a session.  Pass the same instance across turns.
    unknown_field_action:
        ``"block"`` (default/strict) — unknown block types cause
        ``should_block=True`` and a coverage alarm.
        ``"warn_allow"`` — log but forward.
    unscannable_action:
        ``"block"`` (default) — image blocks and base64 document sources set
        ``should_block=True`` and record a coverage gap.
        ``"warn_allow"`` — log gap and allow.

    Returns
    -------
    ClaudeRequestScrubResult
    """
    # Deep-copy so we never mutate the caller's dict
    out: Dict[str, Any] = copy.deepcopy(payload)
    events: List[FieldScanEvent] = []
    should_block = False
    coverage_gaps: List[str] = []
    unknown_fields: List[str] = []

    # ── 1. system prompt ─────────────────────────────────────────────────────
    if "system" in out:
        blk_flag, sys_out, sys_evts, sys_uk = _scrub_system(
            out["system"], engine, unknown_field_action
        )
        out["system"] = sys_out
        events.extend(sys_evts)
        unknown_fields.extend(sys_uk)
        if blk_flag:
            should_block = True

    # ── 2. messages ──────────────────────────────────────────────────────────
    messages = out.get("messages")
    if isinstance(messages, list):
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if content is None:
                continue
            base_loc = f"messages[{msg_idx}].content"
            blk_flag, sanitized_content, msg_evts, msg_uk = _scrub_content(
                content, engine, base_loc, unknown_field_action, unscannable_action
            )
            out["messages"][msg_idx]["content"] = sanitized_content
            events.extend(msg_evts)
            unknown_fields.extend(msg_uk)
            if blk_flag:
                should_block = True

    # Collect coverage gaps from events
    for evt in events:
        if evt.coverage_gap and evt.location not in coverage_gaps:
            coverage_gaps.append(evt.location)

    # If any event carries should_block, propagate upward
    for evt in events:
        if evt.should_block:
            should_block = True
            break

    return ClaudeRequestScrubResult(
        sanitized_payload=out,
        field_events=events,
        should_block=should_block,
        coverage_gaps=coverage_gaps,
        unknown_fields=unknown_fields,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal scrubbers
# ─────────────────────────────────────────────────────────────────────────────

def _scrub_system(
    system: Any,
    engine: Engine,
    unknown_field_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """
    Scrub the ``system`` field.

    Returns (should_block, sanitized_system, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if isinstance(system, str):
        evt = _scan_span(system, engine, ScanField.SYSTEM_PROMPT, "system")
        events.append(evt)
        if evt.should_block:
            should_block = True
        return should_block, evt.redacted_text, events, unknown_fields

    if isinstance(system, list):
        out_blocks = []
        for i, block in enumerate(system):
            loc = f"system[{i}]"
            if not isinstance(block, dict):
                out_blocks.append(block)
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "") or ""
                evt = _scan_span(text, engine, ScanField.SYSTEM_PROMPT, f"{loc}.text")
                events.append(evt)
                if evt.should_block:
                    should_block = True
                new_block = {**block, "text": evt.redacted_text}
                out_blocks.append(new_block)
            else:
                # Unrecognised system block type (e.g. future "image" system blocks)
                warn = f"{loc}: unrecognized system block type={btype!r}"
                unknown_fields.append(warn)
                if unknown_field_action == "block":
                    should_block = True
                out_blocks.append(block)  # pass through unchanged
        return should_block, out_blocks, events, unknown_fields

    # Unexpected type for system (future API extension)
    warn = f"system: unexpected type {type(system).__name__!r}"
    unknown_fields.append(warn)
    if unknown_field_action == "block":
        should_block = True
    return should_block, system, events, unknown_fields


def _scrub_content(
    content: Any,
    engine: Engine,
    base_loc: str,
    unknown_field_action: str,
    unscannable_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """
    Scrub a ``content`` field (string or content block array).

    Returns (should_block, sanitized_content, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    # Plain string shorthand (older API / single-turn)
    if isinstance(content, str):
        evt = _scan_span(content, engine, ScanField.MESSAGE_TEXT, base_loc)
        events.append(evt)
        return evt.should_block, evt.redacted_text, events, unknown_fields

    if not isinstance(content, list):
        warn = f"{base_loc}: unexpected content type {type(content).__name__!r}"
        unknown_fields.append(warn)
        if unknown_field_action == "block":
            should_block = True
        return should_block, content, events, unknown_fields

    out_blocks = []
    for blk_idx, block in enumerate(content):
        blk_loc = f"{base_loc}[{blk_idx}]"
        if not isinstance(block, dict):
            out_blocks.append(block)
            continue
        btype = block.get("type", "")

        # ── text ─────────────────────────────────────────────────────────────
        if btype == "text":
            text = block.get("text", "") or ""
            evt = _scan_span(text, engine, ScanField.MESSAGE_TEXT, f"{blk_loc}.text")
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_blocks.append({**block, "text": evt.redacted_text})

        # ── tool_use ─────────────────────────────────────────────────────────
        elif btype == "tool_use":
            tool_input = block.get("input") or {}
            sanitized_input, tool_evts, tool_block, tool_uk = _scrub_json_object(
                tool_input, engine, f"{blk_loc}.input"
            )
            events.extend(tool_evts)
            unknown_fields.extend(tool_uk)
            if tool_block:
                should_block = True
            out_blocks.append({**block, "input": sanitized_input})

        # ── tool_result ───────────────────────────────────────────────────────
        elif btype == "tool_result":
            result_content = block.get("content")
            if result_content is None:
                out_blocks.append(block)
            else:
                blk_flag, san_rc, rc_evts, rc_uk = _scrub_tool_result_content(
                    result_content, engine, f"{blk_loc}.content",
                    unknown_field_action, unscannable_action
                )
                events.extend(rc_evts)
                unknown_fields.extend(rc_uk)
                if blk_flag:
                    should_block = True
                out_blocks.append({**block, "content": san_rc})

        # ── document ──────────────────────────────────────────────────────────
        elif btype == "document":
            blk_flag, new_block, doc_evts, doc_uk = _scrub_document_block(
                block, engine, blk_loc, unknown_field_action, unscannable_action
            )
            events.extend(doc_evts)
            unknown_fields.extend(doc_uk)
            if blk_flag:
                should_block = True
            out_blocks.append(new_block)

        # ── image ─────────────────────────────────────────────────────────────
        elif btype == "image":
            evt = _make_unscannable_event(
                blk_loc, ScanField.IMAGE, unscannable_action,
                "image block is not text-scannable in Stage-1"
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_blocks.append(block)  # pass image through unchanged

        # ── unknown block type ────────────────────────────────────────────────
        else:
            warn = f"{blk_loc}: unrecognized content block type={btype!r}"
            unknown_fields.append(warn)
            if unknown_field_action == "block":
                should_block = True
                # Emit a coverage-gap event so the ledger records it
                evt = FieldScanEvent(
                    scan_field=ScanField.UNKNOWN,
                    location=blk_loc,
                    detections=[],
                    redacted_text="",
                    coverage_gap=True,
                    should_block=True,
                )
                events.append(evt)
            out_blocks.append(block)

    return should_block, out_blocks, events, unknown_fields


def _scrub_tool_result_content(
    content: Any,
    engine: Engine,
    loc: str,
    unknown_field_action: str,
    unscannable_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """Scrub tool_result .content (string or TextBlock array)."""
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if isinstance(content, str):
        evt = _scan_span(content, engine, ScanField.TOOL_RESULT, loc)
        events.append(evt)
        return evt.should_block, evt.redacted_text, events, unknown_fields

    if not isinstance(content, list):
        warn = f"{loc}: unexpected tool_result content type {type(content).__name__!r}"
        unknown_fields.append(warn)
        if unknown_field_action == "block":
            should_block = True
        return should_block, content, events, unknown_fields

    out_blocks = []
    for i, block in enumerate(content):
        item_loc = f"{loc}[{i}]"
        if not isinstance(block, dict):
            out_blocks.append(block)
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "") or ""
            evt = _scan_span(text, engine, ScanField.TOOL_RESULT, f"{item_loc}.text")
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_blocks.append({**block, "text": evt.redacted_text})
        elif btype == "image":
            evt = _make_unscannable_event(
                item_loc, ScanField.IMAGE, unscannable_action,
                "image inside tool_result is not text-scannable"
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_blocks.append(block)
        else:
            warn = f"{item_loc}: unrecognized tool_result block type={btype!r}"
            unknown_fields.append(warn)
            if unknown_field_action == "block":
                should_block = True
            out_blocks.append(block)

    return should_block, out_blocks, events, unknown_fields


def _scrub_document_block(
    block: dict,
    engine: Engine,
    blk_loc: str,
    unknown_field_action: str,
    unscannable_action: str,
) -> Tuple[bool, dict, List[FieldScanEvent], List[str]]:
    """Scrub a document content block."""
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    source = block.get("source")
    if not isinstance(source, dict):
        warn = f"{blk_loc}.source: missing or non-dict source"
        unknown_fields.append(warn)
        if unknown_field_action == "block":
            should_block = True
        return should_block, block, events, unknown_fields

    src_type = source.get("type", "")
    doc_loc = f"{blk_loc}.source"

    if src_type == "text":
        # Plain-text document body — fully scannable
        data = source.get("data", "") or ""
        evt = _scan_span(data, engine, ScanField.DOCUMENT_BLOCK, f"{doc_loc}.data")
        events.append(evt)
        if evt.should_block:
            should_block = True
        new_source = {**source, "data": evt.redacted_text}
        new_block = {**block, "source": new_source}
        return should_block, new_block, events, unknown_fields

    elif src_type in ("base64", "url"):
        # Non-text source — cannot scan without decoding/fetching
        evt = _make_unscannable_event(
            doc_loc, ScanField.DOCUMENT_BLOCK, unscannable_action,
            f"document source type={src_type!r} is not text-scannable in Stage-1"
        )
        events.append(evt)
        if evt.should_block:
            should_block = True
        return should_block, block, events, unknown_fields

    else:
        warn = f"{doc_loc}: unrecognized document source type={src_type!r}"
        unknown_fields.append(warn)
        if unknown_field_action == "block":
            should_block = True
        return should_block, block, events, unknown_fields


def _scrub_json_object(
    obj: Any,
    engine: Engine,
    location: str,
) -> Tuple[Any, List[FieldScanEvent], bool, List[str]]:
    """
    Recursively walk a JSON-serialisable object and scan all string leaf
    values through the detection engine.

    Returns (sanitized_obj, events, should_block, unknown_field_warnings).
    Non-string leaves (int, float, bool, None) are passed through unchanged.
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if isinstance(obj, str):
        evt = _scan_span(obj, engine, ScanField.TOOL_USE_INPUT, location)
        events.append(evt)
        return evt.redacted_text, events, evt.should_block, unknown_fields

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            child_loc = f"{location}.{k}"
            sanitized_v, child_evts, child_block, child_uk = _scrub_json_object(
                v, engine, child_loc
            )
            out[k] = sanitized_v
            events.extend(child_evts)
            unknown_fields.extend(child_uk)
            if child_block:
                should_block = True
        return out, events, should_block, unknown_fields

    if isinstance(obj, list):
        out_list = []
        for i, item in enumerate(obj):
            child_loc = f"{location}[{i}]"
            sanitized_item, child_evts, child_block, child_uk = _scrub_json_object(
                item, engine, child_loc
            )
            out_list.append(sanitized_item)
            events.extend(child_evts)
            unknown_fields.extend(child_uk)
            if child_block:
                should_block = True
        return out_list, events, should_block, unknown_fields

    # Scalar (int, float, bool, None) — not scannable, pass through
    return obj, events, False, unknown_fields


# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scan_span(
    text: str,
    engine: Engine,
    scan_field: ScanField,
    location: str,
) -> FieldScanEvent:
    """Scan a single text span and return a FieldScanEvent.

    Fail-closed semantics: if the engine raises *any* exception the event is
    marked as a coverage gap with should_block=True.  The sanitized text is
    cleared (empty string) so that unscanned content cannot leak even if a
    caller accidentally forwards a blocked payload.
    """
    if not text:
        return FieldScanEvent(
            scan_field=scan_field,
            location=location,
            detections=[],
            redacted_text=text,
        )
    try:
        result = engine.scan(text)
    except Exception as exc:  # noqa: BLE001 — intentionally broad; all errors → block
        # Scan failed (error / timeout / scanner unavailable) — fail-closed:
        # block the request and log a coverage gap.  Never pass unscanned text.
        reason = f"{type(exc).__name__}: {exc}"
        return FieldScanEvent(
            scan_field=scan_field,
            location=location,
            detections=[],
            redacted_text=_SCAN_ERROR_REDACTED,
            coverage_gap=True,
            should_block=True,
            fail_reason=reason,
        )
    has_block = any(d.action == Action.BLOCK for d in result.detections)
    return FieldScanEvent(
        scan_field=scan_field,
        location=location,
        detections=result.detections,
        redacted_text=result.redacted_text,
        coverage_gap=False,
        should_block=has_block,
    )


def _make_unscannable_event(
    location: str,
    scan_field: ScanField,
    action: str,
    reason: str,
) -> FieldScanEvent:
    """Create a coverage-gap event for content that cannot be text-scanned."""
    blocking = action == "block"
    return FieldScanEvent(
        scan_field=scan_field,
        location=location,
        detections=[],
        redacted_text="",
        coverage_gap=True,
        should_block=blocking,
    )
