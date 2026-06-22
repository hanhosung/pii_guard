"""
OpenAI wire format traversal and payload reconstruction (Sub-AC 2b).

Walks every text-bearing location in OpenAI's chat-completions request schema
and pipes each extracted text span through the detection engine, then rebuilds
a fully sanitized OpenAI API request payload.

Text-bearing locations covered
------------------------------
  messages[*] where role == "system"     → content (string or content-part array)
  messages[*] where role == "user"       → content (string or content-part array)
  messages[*] where role == "assistant"  → content (string or content-part array)
                                           tool_calls[*].function.arguments (JSON string,
                                           recursively walked for string leaf values)
  messages[*] where role == "tool"       → content (string or content-part array)
  messages[*] where role == "developer"  → content (string or content-part array)

Content part types covered
--------------------------
  type == "text"        → .text field scanned
  type == "refusal"     → .refusal field scanned
  type == "image_url"   → unscannable (coverage gap recorded)
  type == "input_audio" → unscannable (coverage gap recorded)
  type == "file"        → unscannable (coverage gap recorded)
  unknown types         → coverage alarm (unknown_field_action governs blocking)

Scan-field taxonomy (maps to ontology scan_field)
--------------------------------------------------
  SYSTEM_MESSAGE   system role message content
  MESSAGE_TEXT     user/assistant/developer text content parts
  TOOL_CALL_ARGS   tool_calls[*].function.arguments parsed string leaves
  TOOL_RESULT      tool role content
  IMAGE_URL        image_url parts (unscannable Stage-1)
  UNKNOWN          unrecognized part types / unscannable audio/file parts

Failure semantics
-----------------
  - Any BLOCK-category detection → should_block=True on the returned result.
  - image_url / input_audio / file parts are unscannable; default action is
    "block" (fail-closed), recording a coverage gap event.
  - tool_calls.function.arguments that cannot be JSON-parsed are treated as
    plain text and scanned directly; a coverage gap is still recorded because
    the structure could not be validated.
  - Unknown content part types raise a coverage alarm; unknown_field_action="block"
    (default/strict) → should_block=True.
  - All non-string leaf values (numbers, booleans, None) in parsed arguments
    are passed through untouched — they carry no text.

Usage
-----
    from pii_guard import Engine
    from pii_guard.providers.openai import scrub_openai_request

    engine = Engine()
    result = scrub_openai_request(payload, engine)
    if result.should_block:
        # Return 400 to the client; do not forward payload
        ...
    else:
        forward(result.sanitized_payload)
"""
from __future__ import annotations

import copy
import json
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
    """Ontology scan_field values for OpenAI-specific locations."""
    SYSTEM_MESSAGE = "system_message"
    MESSAGE_TEXT   = "message_text"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_RESULT    = "tool_result"
    IMAGE_URL      = "image_url"
    UNKNOWN        = "unknown"


@dataclass
class FieldScanEvent:
    """Audit record for a single text span that was scanned (or not)."""

    scan_field: ScanField
    # Dot-notation location in the payload, e.g. "messages[0].tool_calls[0].function.arguments.email"
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
class OpenAIRequestScrubResult:
    """Full result of scrubbing one OpenAI chat-completions request payload."""

    # Deep-copy of the original payload with PII/secrets replaced in-place
    sanitized_payload: Dict[str, Any]
    # Per-field audit events (ledger source)
    field_events: List[FieldScanEvent]
    # True → caller must NOT forward; return 400 to the client
    should_block: bool
    # Dot-notation paths where content passed without text scanning
    coverage_gaps: List[str]
    # Dot-notation paths + descriptions of unrecognized fields / part types
    unknown_fields: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def scrub_openai_request(
    payload: Dict[str, Any],
    engine: Engine,
    *,
    unknown_field_action: str = "block",
    unscannable_action: str = "block",
) -> OpenAIRequestScrubResult:
    """
    Scrub an OpenAI chat-completions request payload.

    Walks every text-bearing location, pipes each text span through *engine*
    (Stage-1 detection), and rebuilds a sanitized payload with PII/secrets
    replaced by indexed placeholders or BLOCKED tokens.

    Parameters
    ----------
    payload:
        Original OpenAI chat-completions request dict (will not be mutated).
    engine:
        A PII-Guard Engine instance.  Stateful: maintains session placeholder
        map so the same real value always maps to the same placeholder within
        a session.  Pass the same instance across turns.
    unknown_field_action:
        ``"block"`` (default/strict) — unknown content part types cause
        ``should_block=True`` and a coverage alarm.
        ``"warn_allow"`` — log but forward.
    unscannable_action:
        ``"block"`` (default) — image_url/audio/file parts set
        ``should_block=True`` and record a coverage gap.
        ``"warn_allow"`` — log gap and allow.

    Returns
    -------
    OpenAIRequestScrubResult
    """
    # Deep-copy so we never mutate the caller's dict
    out: Dict[str, Any] = copy.deepcopy(payload)
    events: List[FieldScanEvent] = []
    should_block = False
    coverage_gaps: List[str] = []
    unknown_fields: List[str] = []

    # ── Walk the messages array ───────────────────────────────────────────────
    messages = out.get("messages")
    if isinstance(messages, list):
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue

            role = message.get("role", "")
            base_loc = f"messages[{msg_idx}]"

            # ── Determine scan_field from role ────────────────────────────────
            if role == "system":
                content_scan_field = ScanField.SYSTEM_MESSAGE
            elif role == "tool":
                content_scan_field = ScanField.TOOL_RESULT
            else:
                # user, assistant, developer, and any future roles
                content_scan_field = ScanField.MESSAGE_TEXT

            # ── Scrub content field ───────────────────────────────────────────
            content = message.get("content")
            if content is not None:
                blk_flag, sanitized_content, msg_evts, msg_uk = _scrub_content(
                    content, engine,
                    f"{base_loc}.content",
                    content_scan_field,
                    unknown_field_action,
                    unscannable_action,
                )
                out["messages"][msg_idx]["content"] = sanitized_content
                events.extend(msg_evts)
                unknown_fields.extend(msg_uk)
                if blk_flag:
                    should_block = True

            # ── Scrub tool_calls (assistant role) ─────────────────────────────
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc_idx, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue
                        tc_loc = f"{base_loc}.tool_calls[{tc_idx}]"
                        func = tc.get("function")
                        if not isinstance(func, dict):
                            continue
                        func_loc = f"{tc_loc}.function"
                        args_str = func.get("arguments")
                        if args_str is None:
                            continue

                        blk_flag, san_args, tc_evts, tc_uk = _scrub_tool_call_arguments(
                            args_str, engine,
                            f"{func_loc}.arguments",
                            unscannable_action,
                        )
                        (out["messages"][msg_idx]
                            ["tool_calls"][tc_idx]
                            ["function"]["arguments"]) = san_args
                        events.extend(tc_evts)
                        unknown_fields.extend(tc_uk)
                        if blk_flag:
                            should_block = True

    # ── Collect coverage gaps from events ─────────────────────────────────────
    for evt in events:
        if evt.coverage_gap and evt.location not in coverage_gaps:
            coverage_gaps.append(evt.location)

    # ── Propagate should_block from events ────────────────────────────────────
    for evt in events:
        if evt.should_block:
            should_block = True
            break

    return OpenAIRequestScrubResult(
        sanitized_payload=out,
        field_events=events,
        should_block=should_block,
        coverage_gaps=coverage_gaps,
        unknown_fields=unknown_fields,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal scrubbers
# ─────────────────────────────────────────────────────────────────────────────

def _scrub_content(
    content: Any,
    engine: Engine,
    base_loc: str,
    scan_field: ScanField,
    unknown_field_action: str,
    unscannable_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """
    Scrub a message content field (string or content-part array).

    Returns (should_block, sanitized_content, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    # Plain string shorthand (all roles accept this)
    if isinstance(content, str):
        evt = _scan_span(content, engine, scan_field, base_loc)
        events.append(evt)
        return evt.should_block, evt.redacted_text, events, unknown_fields

    if not isinstance(content, list):
        warn = f"{base_loc}: unexpected content type {type(content).__name__!r}"
        unknown_fields.append(warn)
        if unknown_field_action == "block":
            should_block = True
        return should_block, content, events, unknown_fields

    out_parts: List[Any] = []
    for part_idx, part in enumerate(content):
        part_loc = f"{base_loc}[{part_idx}]"

        if not isinstance(part, dict):
            out_parts.append(part)
            continue

        ptype = part.get("type", "")

        # ── text part ────────────────────────────────────────────────────────
        if ptype == "text":
            text = part.get("text", "") or ""
            evt = _scan_span(text, engine, scan_field, f"{part_loc}.text")
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append({**part, "text": evt.redacted_text})

        # ── refusal part (assistant only) ─────────────────────────────────────
        elif ptype == "refusal":
            text = part.get("refusal", "") or ""
            evt = _scan_span(text, engine, scan_field, f"{part_loc}.refusal")
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append({**part, "refusal": evt.redacted_text})

        # ── image_url part — unscannable in Stage-1 ───────────────────────────
        elif ptype == "image_url":
            evt = _make_unscannable_event(
                part_loc, ScanField.IMAGE_URL, unscannable_action,
                "image_url content part is not text-scannable in Stage-1",
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append(part)  # pass image through unchanged

        # ── input_audio part — unscannable ────────────────────────────────────
        elif ptype == "input_audio":
            evt = _make_unscannable_event(
                part_loc, ScanField.UNKNOWN, unscannable_action,
                "input_audio content part is not text-scannable in Stage-1",
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append(part)

        # ── file part — unscannable without decoding ──────────────────────────
        elif ptype == "file":
            evt = _make_unscannable_event(
                part_loc, ScanField.UNKNOWN, unscannable_action,
                "file content part is not text-scannable in Stage-1",
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append(part)

        # ── unknown part type ────────────────────────────────────────────────
        else:
            warn = f"{part_loc}: unrecognized content part type={ptype!r}"
            unknown_fields.append(warn)
            if unknown_field_action == "block":
                should_block = True
                # Emit a coverage-gap event so the ledger records it
                evt = FieldScanEvent(
                    scan_field=ScanField.UNKNOWN,
                    location=part_loc,
                    detections=[],
                    redacted_text="",
                    coverage_gap=True,
                    should_block=True,
                )
                events.append(evt)
            out_parts.append(part)

    return should_block, out_parts, events, unknown_fields


def _scrub_tool_call_arguments(
    args_str: Any,
    engine: Engine,
    location: str,
    unscannable_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """
    Scrub tool_calls[*].function.arguments (a JSON string).

    Parses the JSON string, recursively scans all string leaf values through
    the detection engine, then serializes the sanitized object back to a
    JSON string.

    If the string cannot be parsed as JSON, it is scanned directly as plain
    text (best-effort protection) and a coverage gap is recorded because the
    structural scan could not be completed.

    Returns (should_block, sanitized_arguments_str, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if not isinstance(args_str, str):
        warn = f"{location}: unexpected arguments type {type(args_str).__name__!r}"
        unknown_fields.append(warn)
        return should_block, args_str, events, unknown_fields

    # Try to parse as JSON
    try:
        args_obj = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        # Fallback: scan the raw string directly and record a coverage gap
        evt = _scan_span(args_str, engine, ScanField.TOOL_CALL_ARGS, location)
        # Mark as coverage gap: we couldn't walk the structure
        evt.coverage_gap = True
        events.append(evt)
        return evt.should_block, evt.redacted_text, events, unknown_fields

    # Recursively scan the parsed object
    sanitized_obj, obj_evts, obj_block, obj_uk = _scrub_json_object(
        args_obj, engine, location
    )
    events.extend(obj_evts)
    unknown_fields.extend(obj_uk)
    if obj_block:
        should_block = True

    # Serialize back to JSON string
    try:
        sanitized_args_str = json.dumps(sanitized_obj, ensure_ascii=False)
    except (TypeError, ValueError):
        # Serialization failed (shouldn't happen with well-formed input)
        sanitized_args_str = args_str

    return should_block, sanitized_args_str, events, unknown_fields


def _scrub_json_object(
    obj: Any,
    engine: Engine,
    location: str,
) -> Tuple[Any, List[FieldScanEvent], bool, List[str]]:
    """
    Recursively walk a JSON-serialisable object and scan all string leaf values
    through the detection engine.

    Returns (sanitized_obj, events, should_block, unknown_field_warnings).
    Non-string leaves (int, float, bool, None) are passed through unchanged.
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if isinstance(obj, str):
        evt = _scan_span(obj, engine, ScanField.TOOL_CALL_ARGS, location)
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
        out_list: List[Any] = []
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

    # Scalar (int, float, bool, None) — not text-bearing, pass through
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
