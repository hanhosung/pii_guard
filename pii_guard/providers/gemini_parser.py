"""
Gemini provider request parser — Sub-AC 3.

Extracts the canonical message/content fields from Google's Gemini API
request schema and returns a structured :class:`GeminiFieldMap`.

This module is **pure parse only** — it walks the request schema, classifies
every field location, and records whether each field is text-scannable.  It
does *not* run detection, apply masking, or modify the payload in any way.

The :class:`GeminiFieldMap` is the authoritative declaration of what the
scrubber is allowed to process: masking (in :mod:`pii_guard.providers.gemini`)
targets *only* the text-bearing fields enumerated here — never arbitrary keys,
structural fields (model, role, name, language, outcome, mimeType, fileUri,
generationConfig, safetySettings, …), or fields whose location is not listed
in the returned field map.

Gemini API coverage
-------------------
  systemInstruction (or system_instruction)
    string form                   →  one SYSTEM_INSTRUCTION text field
    dict form with parts[*].text  →  one SYSTEM_INSTRUCTION text field per
                                      text part
    dict form with non-text part  →  unknown field entry

  contents[*].parts[*] where part contains:
    text                          →  MESSAGE_TEXT text field
    functionCall (function_call)
      .args                       →  FUNCTION_CALL_ARGS text fields (all str
                                      leaf values of args, walked recursively)
    functionResponse (function_response)
      .response                   →  FUNCTION_RESPONSE text fields (all str
                                      leaf values of response, walked recursively)
    executableCode (executable_code)
      .code                       →  EXECUTABLE_CODE text field
    codeExecutionResult (code_execution_result)
      .output                     →  CODE_EXECUTION_RESULT text field
    inlineData (inline_data)      →  unscannable INLINE_DATA field
    fileData (file_data)          →  unscannable FILE_DATA field
    unknown part key              →  unknown field entry (coverage alarm)

Field-name variants
-------------------
Both camelCase (REST API) and snake_case (Python SDK) are supported
transparently:

  camelCase form          snake_case form
  systemInstruction   →   system_instruction
  functionCall        →   function_call
  functionResponse    →   function_response
  inlineData          →   inline_data
  fileData            →   file_data
  executableCode      →   executable_code
  codeExecutionResult →   code_execution_result

Field map entries not created for
-----------------------------------
  model, generationConfig, safetySettings, tools, toolConfig  (top-level params)
  contents[*].role                                            (structural)
  functionCall.name, functionResponse.name                    (structural)
  executableCode.language, codeExecutionResult.outcome        (structural)
  inlineData.mimeType, fileData.mimeType, fileData.fileUri    (structural)
  numeric / boolean / null leaf values in functionCall.args
    and functionResponse.response

Typical usage
-------------
    from pii_guard.providers.gemini_parser import parse_gemini_request

    field_map = parse_gemini_request(payload)
    for field in field_map.text_fields:
        redacted = engine.scan(field.text).redacted_text
        ...  # apply only to identified text locations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Public enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ScanField(str, Enum):
    """
    Ontology ``scan_field`` values — Gemini-specific text-bearing locations.

    These values mirror the Gemini scrubber's ScanField enum so the parser
    and scrubber share a common vocabulary without circular imports.
    """
    SYSTEM_INSTRUCTION    = "system_instruction"
    MESSAGE_TEXT          = "message_text"
    FUNCTION_CALL_ARGS    = "function_call_args"
    FUNCTION_RESPONSE     = "function_response"
    EXECUTABLE_CODE       = "executable_code"
    CODE_EXECUTION_RESULT = "code_execution_result"
    INLINE_DATA           = "inline_data"
    FILE_DATA             = "file_data"
    UNKNOWN               = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ParsedField — one extracted field entry in the field map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedField:
    """
    Metadata for a single field extracted from a Gemini API request.

    Attributes
    ----------
    location:
        Dot-notation path in the original payload, e.g.
        ``"contents[0].parts[1].functionCall.args.email"``.
    scan_field:
        The ontology category for this location.
    text:
        The raw text string at this location, or ``None`` for unscannable
        (inlineData, fileData) and unknown fields.  Empty string ``""``
        is a valid scannable value (empty text part).
    is_scannable:
        ``True`` when *text* holds the actual string content and Stage-1
        regex detection can be applied.
    is_unscannable:
        ``True`` for inlineData and fileData parts that the Stage-1 regex
        engine cannot process.  These fields still appear in the map so the
        caller can apply the ``unscannable_action`` policy (default: block).
    is_unknown:
        ``True`` for part types not defined in the Gemini API schema.
        These trigger a coverage alarm.
    coverage_gap_reason:
        Human-readable description of *why* the field is not text-scannable.
        Always ``None`` when ``is_scannable=True``.
    """

    location: str
    scan_field: ScanField
    text: Optional[str]              # None for unscannable / unknown
    is_scannable: bool               # True → text is a string ready to scan
    is_unscannable: bool = False     # True → inlineData / fileData
    is_unknown: bool = False         # True → unrecognized part type
    coverage_gap_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# GeminiFieldMap — the structured result of parsing one request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeminiFieldMap:
    """
    Structured map of all field locations extracted from a Gemini API request.

    Masking must target *only* the fields enumerated here.  Structural,
    numeric, and boolean payload fields are never listed and must not be
    modified by the scrubbing layer.

    Attributes
    ----------
    all_fields:
        Every parsed field in document order (text, unscannable, unknown).
    text_fields:
        Subset of ``all_fields`` where ``is_scannable=True``.
        These are the *only* locations the scrubber may modify.
    unscannable_fields:
        Subset of ``all_fields`` where ``is_unscannable=True``.
        Require a policy decision (block / warn_allow) from the caller.
    unknown_fields:
        Subset of ``all_fields`` where ``is_unknown=True``.
        Trigger a coverage alarm — caller decides block / warn_allow.
    model:
        The ``model`` string from the request if present, else ``None``.
        Informational only — not a scan target.
    api_version:
        The Gemini API version string if injected into the payload by the
        caller, else ``None``.  Informational only.
    """

    all_fields: List[ParsedField] = field(default_factory=list)
    model: Optional[str] = None
    api_version: Optional[str] = None

    @property
    def text_fields(self) -> List[ParsedField]:
        """Scannable text fields — the *only* masking targets."""
        return [f for f in self.all_fields if f.is_scannable]

    @property
    def unscannable_fields(self) -> List[ParsedField]:
        """inlineData / fileData parts — require policy decision."""
        return [f for f in self.all_fields if f.is_unscannable]

    @property
    def unknown_fields(self) -> List[ParsedField]:
        """Unrecognized part types — trigger coverage alarm."""
        return [f for f in self.all_fields if f.is_unknown]

    def get_field(self, location: str) -> Optional[ParsedField]:
        """Return the ParsedField for *location*, or ``None`` if not found."""
        for f in self.all_fields:
            if f.location == location:
                return f
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Field-name normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map of camelCase ↔ snake_case aliases for top-level and part-type keys.
_FIELD_ALIASES: Dict[str, str] = {
    "systemInstruction":    "system_instruction",
    "functionCall":         "function_call",
    "functionResponse":     "function_response",
    "inlineData":           "inline_data",
    "fileData":             "file_data",
    "executableCode":       "executable_code",
    "codeExecutionResult":  "code_execution_result",
}
# Build reverse (snake → camel) as well
_FIELD_ALIASES.update({v: k for k, v in list(_FIELD_ALIASES.items())})


def _resolve_key(d: dict, *candidates: str) -> Optional[str]:
    """
    Return the first candidate key (or its alias) that exists in *d*.

    Accepts both camelCase and snake_case forms transparently.
    """
    for candidate in candidates:
        if candidate in d:
            return candidate
        alias = _FIELD_ALIASES.get(candidate)
        if alias and alias in d:
            return alias
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_gemini_request(
    payload: Dict[str, Any],
    *,
    api_version: Optional[str] = None,
) -> GeminiFieldMap:
    """
    Parse a Gemini API request payload and return a structured
    :class:`GeminiFieldMap`.

    This function does **not** modify the payload and performs no scanning or
    masking.  It purely walks the schema and records every text-bearing location
    plus any unscannable or unknown locations.

    Parameters
    ----------
    payload:
        Gemini API request dict (not mutated).
    api_version:
        Optional Gemini API version string, used to populate
        ``GeminiFieldMap.api_version`` for version-mismatch alarms.

    Returns
    -------
    GeminiFieldMap
        Structured field map.  Use ``field_map.text_fields`` to enumerate
        the exact set of locations that the scrubber may mask.
    """
    if not isinstance(payload, dict):
        return GeminiFieldMap(api_version=api_version)

    model = payload.get("model")
    result = GeminiFieldMap(
        model=model if isinstance(model, str) else None,
        api_version=api_version,
    )

    # ── 1. systemInstruction / system_instruction ─────────────────────────────
    si_key = _resolve_key(payload, "systemInstruction", "system_instruction")
    if si_key is not None:
        _parse_system_instruction(payload[si_key], si_key, result)

    # ── 2. contents ───────────────────────────────────────────────────────────
    contents = payload.get("contents")
    if isinstance(contents, list):
        for ci, content_item in enumerate(contents):
            if not isinstance(content_item, dict):
                continue
            parts = content_item.get("parts")
            if not isinstance(parts, list):
                continue
            base_loc = f"contents[{ci}].parts"
            _parse_parts(parts, base_loc, result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsers — one per schema section
# ─────────────────────────────────────────────────────────────────────────────

def _parse_system_instruction(
    si: Any,
    key_name: str,
    result: GeminiFieldMap,
) -> None:
    """
    Parse the systemInstruction (or system_instruction) field.

    The Gemini API accepts:
      - A plain string (shorthand; some Python SDK versions)
      - A dict with a ``parts`` list (standard Content object)

    Mutates *result* in-place by appending parsed fields.
    """
    if isinstance(si, str):
        # Shorthand string form
        result.all_fields.append(ParsedField(
            location=key_name,
            scan_field=ScanField.SYSTEM_INSTRUCTION,
            text=si,
            is_scannable=True,
        ))
        return

    if isinstance(si, dict):
        parts = si.get("parts")
        if not isinstance(parts, list):
            # Unexpected structure — unknown field entry
            result.all_fields.append(ParsedField(
                location=f"{key_name}",
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=(
                    f"{key_name}: missing or non-list 'parts'"
                ),
            ))
            return

        # Walk the parts list using SYSTEM_INSTRUCTION as the default scan_field
        _parse_parts(
            parts,
            f"{key_name}.parts",
            result,
            default_scan_field=ScanField.SYSTEM_INSTRUCTION,
        )
        return

    # Unexpected type for systemInstruction
    result.all_fields.append(ParsedField(
        location=key_name,
        scan_field=ScanField.UNKNOWN,
        text=None,
        is_scannable=False,
        is_unknown=True,
        coverage_gap_reason=(
            f"{key_name}: unexpected type {type(si).__name__!r}"
        ),
    ))


def _parse_parts(
    parts: list,
    base_loc: str,
    result: GeminiFieldMap,
    *,
    default_scan_field: ScanField = ScanField.MESSAGE_TEXT,
) -> None:
    """
    Parse a Gemini ``parts`` list from either contents[*] or systemInstruction.

    Each part is a dict containing exactly one of the supported part-type keys:
      text, functionCall (function_call), functionResponse (function_response),
      inlineData (inline_data), fileData (file_data),
      executableCode (executable_code), codeExecutionResult (code_execution_result).

    Mutates *result* in-place.
    """
    for pi, part in enumerate(parts):
        part_loc = f"{base_loc}[{pi}]"

        if not isinstance(part, dict):
            # Non-dict part — skip (pass through without error)
            continue

        # ── text ──────────────────────────────────────────────────────────────
        if "text" in part:
            text = part.get("text", "") or ""
            result.all_fields.append(ParsedField(
                location=f"{part_loc}.text",
                scan_field=default_scan_field,
                text=text,
                is_scannable=True,
            ))

        # ── functionCall / function_call ───────────────────────────────────────
        elif (fc_key := _resolve_key(part, "functionCall", "function_call")) is not None:
            fc = part[fc_key]
            if isinstance(fc, dict):
                # args is a JSON object — recursively walk string leaves
                args = fc.get("args") or {}
                loc = f"{part_loc}.{fc_key}.args"
                _parse_json_object(args, loc, ScanField.FUNCTION_CALL_ARGS, result)
            else:
                # Unexpected functionCall type — unknown field entry
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{fc_key}",
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=(
                        f"{part_loc}.{fc_key}: unexpected type "
                        f"{type(fc).__name__!r}"
                    ),
                ))

        # ── functionResponse / function_response ───────────────────────────────
        elif (fr_key := _resolve_key(part, "functionResponse", "function_response")) is not None:
            fr = part[fr_key]
            if isinstance(fr, dict):
                # response is a JSON object — recursively walk string leaves
                response = fr.get("response") or {}
                loc = f"{part_loc}.{fr_key}.response"
                _parse_json_object(response, loc, ScanField.FUNCTION_RESPONSE, result)
            else:
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{fr_key}",
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=(
                        f"{part_loc}.{fr_key}: unexpected type "
                        f"{type(fr).__name__!r}"
                    ),
                ))

        # ── executableCode / executable_code ───────────────────────────────────
        elif (ec_key := _resolve_key(part, "executableCode", "executable_code")) is not None:
            ec = part[ec_key]
            if isinstance(ec, dict):
                code_text = ec.get("code", "") or ""
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{ec_key}.code",
                    scan_field=ScanField.EXECUTABLE_CODE,
                    text=code_text,
                    is_scannable=True,
                ))
            else:
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{ec_key}",
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=(
                        f"{part_loc}.{ec_key}: unexpected type "
                        f"{type(ec).__name__!r}"
                    ),
                ))

        # ── codeExecutionResult / code_execution_result ────────────────────────
        elif (cer_key := _resolve_key(part, "codeExecutionResult", "code_execution_result")) is not None:
            cer = part[cer_key]
            if isinstance(cer, dict):
                output_text = cer.get("output", "") or ""
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{cer_key}.output",
                    scan_field=ScanField.CODE_EXECUTION_RESULT,
                    text=output_text,
                    is_scannable=True,
                ))
            else:
                result.all_fields.append(ParsedField(
                    location=f"{part_loc}.{cer_key}",
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=(
                        f"{part_loc}.{cer_key}: unexpected type "
                        f"{type(cer).__name__!r}"
                    ),
                ))

        # ── inlineData / inline_data — unscannable binary data ─────────────────
        elif (id_key := _resolve_key(part, "inlineData", "inline_data")) is not None:
            result.all_fields.append(ParsedField(
                location=f"{part_loc}.{id_key}",
                scan_field=ScanField.INLINE_DATA,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason=(
                    f"{id_key} part is binary data — not text-scannable in Stage-1"
                ),
            ))

        # ── fileData / file_data — unscannable external file reference ──────────
        elif (fd_key := _resolve_key(part, "fileData", "file_data")) is not None:
            result.all_fields.append(ParsedField(
                location=f"{part_loc}.{fd_key}",
                scan_field=ScanField.FILE_DATA,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason=(
                    f"{fd_key} part references an external file — "
                    f"not text-scannable in Stage-1"
                ),
            ))

        # ── unknown part type — coverage alarm ─────────────────────────────────
        else:
            part_keys = ", ".join(repr(k) for k in part.keys()) if part else "(empty)"
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=(
                    f"unrecognized part keys={part_keys}"
                ),
            ))


def _parse_json_object(
    obj: Any,
    location: str,
    scan_field: ScanField,
    result: GeminiFieldMap,
) -> None:
    """
    Recursively walk a JSON-serialisable object and register all string leaf
    values as scan targets with the given *scan_field*.

    Non-string leaves (int, float, bool, None) are not registered — they carry
    no text and are not masking targets.

    Mutates *result* in-place.
    """
    if isinstance(obj, str):
        result.all_fields.append(ParsedField(
            location=location,
            scan_field=scan_field,
            text=obj,
            is_scannable=True,
        ))
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _parse_json_object(v, f"{location}.{k}", scan_field, result)
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _parse_json_object(item, f"{location}[{i}]", scan_field, result)
        return

    # Scalar (int, float, bool, None) — not a text masking target; skip
