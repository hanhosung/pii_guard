"""
Gemini wire format traversal and payload reconstruction (Sub-AC 2c).

Walks every text-bearing location in Google's Gemini API request schema and
pipes each extracted text span through the detection engine, then rebuilds a
fully sanitized Gemini API request payload.

Text-bearing locations covered
------------------------------
  systemInstruction (or system_instruction)
      .parts[*].text           — system instruction text parts

  contents[*]
      .parts[*].text           — regular text content parts
      .parts[*].functionCall.args
                               — function call arguments (JSON object;
                                 all string leaf values recursively scanned)
      .parts[*].functionResponse.response
                               — function response object (JSON object;
                                 all string leaf values recursively scanned)
      .parts[*].executableCode.code
                               — source code (text; scanned)
      .parts[*].codeExecutionResult.output
                               — code execution output (text; scanned)
      .parts[*].inlineData     — binary/base64 data (unscannable → coverage gap)
      .parts[*].fileData       — file URI reference (unscannable → coverage gap)

Field-name variants
-------------------
The REST API uses camelCase; the Python SDK may use snake_case.  Both forms
are recognised transparently:

  camelCase form          → snake_case form
  systemInstruction       → system_instruction
  functionCall            → function_call
  functionResponse        → function_response
  inlineData              → inline_data
  fileData                → file_data
  executableCode          → executable_code
  codeExecutionResult     → code_execution_result

Scan-field taxonomy (maps to ontology scan_field)
--------------------------------------------------
  SYSTEM_INSTRUCTION       systemInstruction parts text
  MESSAGE_TEXT             contents text parts
  FUNCTION_CALL_ARGS       functionCall.args string leaf values
  FUNCTION_RESPONSE        functionResponse.response string leaf values
  EXECUTABLE_CODE          executableCode.code
  CODE_EXECUTION_RESULT    codeExecutionResult.output
  INLINE_DATA              inlineData (unscannable)
  FILE_DATA                fileData (unscannable)
  UNKNOWN                  unrecognized part type

Failure semantics
-----------------
  - Any BLOCK-category detection → should_block=True on the returned result.
  - inlineData and fileData parts are unscannable; the default action is
    "block" (fail-closed), recording a coverage gap event.
  - Unknown part types raise a coverage alarm; unknown_field_action="block"
    (default/strict) → should_block=True.
  - All non-string leaf values (numbers, booleans, None) in functionCall.args
    and functionResponse.response are passed through untouched.
  - The original payload dict is never mutated (deep-copied before editing).

Usage
-----
    from pii_guard import Engine
    from pii_guard.providers.gemini import scrub_gemini_request

    engine = Engine()
    result = scrub_gemini_request(payload, engine)
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
    """Ontology scan_field values for Gemini-specific locations."""
    SYSTEM_INSTRUCTION    = "system_instruction"
    MESSAGE_TEXT          = "message_text"
    FUNCTION_CALL_ARGS    = "function_call_args"
    FUNCTION_RESPONSE     = "function_response"
    EXECUTABLE_CODE       = "executable_code"
    CODE_EXECUTION_RESULT = "code_execution_result"
    INLINE_DATA           = "inline_data"
    FILE_DATA             = "file_data"
    UNKNOWN               = "unknown"


@dataclass
class FieldScanEvent:
    """Audit record for a single text span that was scanned (or not)."""

    scan_field: ScanField
    # Dot-notation location in the payload, e.g.
    # "contents[0].parts[1].functionCall.args.email"
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
class GeminiRequestScrubResult:
    """Full result of scrubbing one Gemini API request payload."""

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
# Field-name normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Pairs of (camelCase, snake_case) for part-type keys.
# We look for the first key that exists in a dict to support both API styles.
_PART_KEY_ALIASES: Dict[str, str] = {
    # camelCase → snake_case
    "systemInstruction":    "system_instruction",
    "functionCall":         "function_call",
    "functionResponse":     "function_response",
    "inlineData":           "inline_data",
    "fileData":             "file_data",
    "executableCode":       "executable_code",
    "codeExecutionResult":  "code_execution_result",
}
# Also build reverse map (snake → camel) for lookup
_PART_KEY_ALIASES.update({v: k for k, v in _PART_KEY_ALIASES.items()})


def _get_part_key(part: dict, *candidates: str) -> Optional[str]:
    """
    Return the first candidate key (or its alias) present in *part*.

    Accepts both camelCase and snake_case names transparently.
    """
    for candidate in candidates:
        if candidate in part:
            return candidate
        # Try alias
        alias = _PART_KEY_ALIASES.get(candidate)
        if alias and alias in part:
            return alias
    return None


def _get_top_key(payload: dict, *candidates: str) -> Optional[str]:
    """
    Like _get_part_key but for top-level payload keys.
    Used to find systemInstruction / system_instruction.
    """
    return _get_part_key(payload, *candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def scrub_gemini_request(
    payload: Dict[str, Any],
    engine: Engine,
    *,
    unknown_field_action: str = "block",
    unscannable_action: str = "block",
) -> GeminiRequestScrubResult:
    """
    Scrub a Gemini API request payload.

    Walks every text-bearing location, pipes each text span through *engine*
    (Stage-1 detection), and rebuilds a sanitized payload with PII/secrets
    replaced by indexed placeholders or BLOCKED tokens.

    Parameters
    ----------
    payload:
        Original Gemini API request dict (will not be mutated).
    engine:
        A PII-Guard Engine instance.  Stateful: maintains session placeholder
        map so the same real value always maps to the same placeholder within
        a session.  Pass the same instance across turns.
    unknown_field_action:
        ``"block"`` (default/strict) — unknown part types cause
        ``should_block=True`` and a coverage alarm.
        ``"warn_allow"`` — log but forward.
    unscannable_action:
        ``"block"`` (default) — inlineData/fileData parts set
        ``should_block=True`` and record a coverage gap.
        ``"warn_allow"`` — log gap and allow.

    Returns
    -------
    GeminiRequestScrubResult
    """
    # Deep-copy so we never mutate the caller's dict
    out: Dict[str, Any] = copy.deepcopy(payload)
    events: List[FieldScanEvent] = []
    should_block = False
    coverage_gaps: List[str] = []
    unknown_fields: List[str] = []

    # ── 1. systemInstruction (or system_instruction) ─────────────────────────
    si_key = _get_top_key(out, "systemInstruction", "system_instruction")
    if si_key is not None:
        blk_flag, si_out, si_evts, si_uk = _scrub_system_instruction(
            out[si_key], engine, si_key, unknown_field_action, unscannable_action
        )
        out[si_key] = si_out
        events.extend(si_evts)
        unknown_fields.extend(si_uk)
        if blk_flag:
            should_block = True

    # ── 2. contents ──────────────────────────────────────────────────────────
    contents = out.get("contents")
    if isinstance(contents, list):
        for ci, content_item in enumerate(contents):
            if not isinstance(content_item, dict):
                continue
            parts = content_item.get("parts")
            if not isinstance(parts, list):
                continue
            base_loc = f"contents[{ci}].parts"
            blk_flag, san_parts, p_evts, p_uk = _scrub_parts(
                parts, engine, base_loc, unknown_field_action, unscannable_action
            )
            out["contents"][ci]["parts"] = san_parts
            events.extend(p_evts)
            unknown_fields.extend(p_uk)
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

    return GeminiRequestScrubResult(
        sanitized_payload=out,
        field_events=events,
        should_block=should_block,
        coverage_gaps=coverage_gaps,
        unknown_fields=unknown_fields,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal scrubbers
# ─────────────────────────────────────────────────────────────────────────────

def _scrub_system_instruction(
    si: Any,
    engine: Engine,
    key_name: str,
    unknown_field_action: str,
    unscannable_action: str,
) -> Tuple[bool, Any, List[FieldScanEvent], List[str]]:
    """
    Scrub the systemInstruction field.

    The Gemini API accepts:
      - A dict with a ``parts`` list (standard Content object, role ignored)
      - A plain string (shorthand; some client SDK versions)

    Returns (should_block, sanitized_si, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False

    if isinstance(si, str):
        # Shorthand string form
        evt = _scan_span(si, engine, ScanField.SYSTEM_INSTRUCTION, key_name)
        events.append(evt)
        if evt.should_block:
            should_block = True
        return should_block, evt.redacted_text, events, unknown_fields

    if isinstance(si, dict):
        parts = si.get("parts")
        if not isinstance(parts, list):
            # Unexpected structure — treat as coverage alarm
            warn = f"{key_name}: missing or non-list 'parts'"
            unknown_fields.append(warn)
            if unknown_field_action == "block":
                should_block = True
            return should_block, si, events, unknown_fields

        blk_flag, san_parts, p_evts, p_uk = _scrub_parts(
            parts, engine, f"{key_name}.parts",
            unknown_field_action, unscannable_action,
            default_scan_field=ScanField.SYSTEM_INSTRUCTION,
        )
        events.extend(p_evts)
        unknown_fields.extend(p_uk)
        if blk_flag:
            should_block = True
        new_si = {**si, "parts": san_parts}
        return should_block, new_si, events, unknown_fields

    # Unexpected type
    warn = f"{key_name}: unexpected type {type(si).__name__!r}"
    unknown_fields.append(warn)
    if unknown_field_action == "block":
        should_block = True
    return should_block, si, events, unknown_fields


def _scrub_parts(
    parts: list,
    engine: Engine,
    base_loc: str,
    unknown_field_action: str,
    unscannable_action: str,
    default_scan_field: ScanField = ScanField.MESSAGE_TEXT,
) -> Tuple[bool, list, List[FieldScanEvent], List[str]]:
    """
    Scrub a Gemini ``parts`` list (from either contents or systemInstruction).

    Each part is a dict containing exactly one of the following keys:
      text, functionCall (function_call), functionResponse (function_response),
      inlineData (inline_data), fileData (file_data),
      executableCode (executable_code), codeExecutionResult (code_execution_result).

    Returns (should_block, sanitized_parts, events, unknown_field_warnings).
    """
    events: List[FieldScanEvent] = []
    unknown_fields: List[str] = []
    should_block = False
    out_parts: list = []

    for pi, part in enumerate(parts):
        part_loc = f"{base_loc}[{pi}]"

        if not isinstance(part, dict):
            out_parts.append(part)
            continue

        # ── text ─────────────────────────────────────────────────────────────
        if "text" in part:
            text = part.get("text", "") or ""
            evt = _scan_span(text, engine, default_scan_field, f"{part_loc}.text")
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append({**part, "text": evt.redacted_text})

        # ── functionCall / function_call ──────────────────────────────────────
        elif (fc_key := _get_part_key(part, "functionCall", "function_call")) is not None:
            fc = part[fc_key]
            if isinstance(fc, dict):
                # args is a JSON object (dict) — recursively scan string leaves
                args_key = _get_part_key(fc, "args") or "args"
                args = fc.get(args_key) or {}
                loc = f"{part_loc}.{fc_key}.{args_key}"
                san_args, arg_evts, arg_block, arg_uk = _scrub_json_object(
                    args, engine, loc, ScanField.FUNCTION_CALL_ARGS
                )
                events.extend(arg_evts)
                unknown_fields.extend(arg_uk)
                if arg_block:
                    should_block = True
                new_fc = {**fc, args_key: san_args}
            else:
                # Unexpected functionCall type — alarm
                warn = f"{part_loc}.{fc_key}: unexpected type {type(fc).__name__!r}"
                unknown_fields.append(warn)
                if unknown_field_action == "block":
                    should_block = True
                new_fc = fc
            out_parts.append({**part, fc_key: new_fc})

        # ── functionResponse / function_response ──────────────────────────────
        elif (fr_key := _get_part_key(part, "functionResponse", "function_response")) is not None:
            fr = part[fr_key]
            if isinstance(fr, dict):
                # response is a JSON object (dict) — recursively scan string leaves
                resp_key = _get_part_key(fr, "response") or "response"
                response = fr.get(resp_key) or {}
                loc = f"{part_loc}.{fr_key}.{resp_key}"
                san_resp, resp_evts, resp_block, resp_uk = _scrub_json_object(
                    response, engine, loc, ScanField.FUNCTION_RESPONSE
                )
                events.extend(resp_evts)
                unknown_fields.extend(resp_uk)
                if resp_block:
                    should_block = True
                new_fr = {**fr, resp_key: san_resp}
            else:
                warn = f"{part_loc}.{fr_key}: unexpected type {type(fr).__name__!r}"
                unknown_fields.append(warn)
                if unknown_field_action == "block":
                    should_block = True
                new_fr = fr
            out_parts.append({**part, fr_key: new_fr})

        # ── executableCode / executable_code ──────────────────────────────────
        elif (ec_key := _get_part_key(part, "executableCode", "executable_code")) is not None:
            ec = part[ec_key]
            if isinstance(ec, dict):
                code_text = ec.get("code", "") or ""
                loc = f"{part_loc}.{ec_key}.code"
                evt = _scan_span(code_text, engine, ScanField.EXECUTABLE_CODE, loc)
                events.append(evt)
                if evt.should_block:
                    should_block = True
                new_ec = {**ec, "code": evt.redacted_text}
            else:
                warn = f"{part_loc}.{ec_key}: unexpected type {type(ec).__name__!r}"
                unknown_fields.append(warn)
                if unknown_field_action == "block":
                    should_block = True
                new_ec = ec
            out_parts.append({**part, ec_key: new_ec})

        # ── codeExecutionResult / code_execution_result ───────────────────────
        elif (cer_key := _get_part_key(part, "codeExecutionResult", "code_execution_result")) is not None:
            cer = part[cer_key]
            if isinstance(cer, dict):
                output_text = cer.get("output", "") or ""
                loc = f"{part_loc}.{cer_key}.output"
                evt = _scan_span(output_text, engine, ScanField.CODE_EXECUTION_RESULT, loc)
                events.append(evt)
                if evt.should_block:
                    should_block = True
                new_cer = {**cer, "output": evt.redacted_text}
            else:
                warn = f"{part_loc}.{cer_key}: unexpected type {type(cer).__name__!r}"
                unknown_fields.append(warn)
                if unknown_field_action == "block":
                    should_block = True
                new_cer = cer
            out_parts.append({**part, cer_key: new_cer})

        # ── inlineData / inline_data — unscannable ────────────────────────────
        elif (id_key := _get_part_key(part, "inlineData", "inline_data")) is not None:
            loc = f"{part_loc}.{id_key}"
            evt = _make_unscannable_event(
                loc, ScanField.INLINE_DATA, unscannable_action,
                f"{id_key} part is binary data — not text-scannable in Stage-1",
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append(part)  # pass through unchanged

        # ── fileData / file_data — unscannable ────────────────────────────────
        elif (fd_key := _get_part_key(part, "fileData", "file_data")) is not None:
            loc = f"{part_loc}.{fd_key}"
            evt = _make_unscannable_event(
                loc, ScanField.FILE_DATA, unscannable_action,
                f"{fd_key} part references an external file — not text-scannable in Stage-1",
            )
            events.append(evt)
            if evt.should_block:
                should_block = True
            out_parts.append(part)  # pass through unchanged

        # ── unknown part type ─────────────────────────────────────────────────
        else:
            # Try to describe what keys the unknown part has
            part_keys = ", ".join(repr(k) for k in part.keys()) if part else "(empty)"
            warn = f"{part_loc}: unrecognized part keys={part_keys}"
            unknown_fields.append(warn)
            if unknown_field_action == "block":
                should_block = True
                # Emit a coverage-gap event so the ledger records it
                gap_evt = FieldScanEvent(
                    scan_field=ScanField.UNKNOWN,
                    location=part_loc,
                    detections=[],
                    redacted_text="",
                    coverage_gap=True,
                    should_block=True,
                )
                events.append(gap_evt)
            out_parts.append(part)

    return should_block, out_parts, events, unknown_fields


def _scrub_json_object(
    obj: Any,
    engine: Engine,
    location: str,
    scan_field: ScanField,
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
        evt = _scan_span(obj, engine, scan_field, location)
        events.append(evt)
        return evt.redacted_text, events, evt.should_block, unknown_fields

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            child_loc = f"{location}.{k}"
            sanitized_v, child_evts, child_block, child_uk = _scrub_json_object(
                v, engine, child_loc, scan_field
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
                item, engine, child_loc, scan_field
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
