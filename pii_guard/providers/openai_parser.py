"""
OpenAI provider request parser — Sub-AC 2.

Extracts the canonical message/content fields from OpenAI's chat-completions
request schema and returns a structured :class:`OpenAIFieldMap`.

This module is **pure parse only** — it walks the request schema, classifies
every field location, and records whether each field is text-scannable.  It
does *not* run detection, apply masking, or modify the payload in any way.

The :class:`OpenAIFieldMap` is the authoritative declaration of what the
scrubber is allowed to process: masking (in :mod:`pii_guard.providers.openai`)
targets *only* the text-bearing fields enumerated here — never arbitrary keys,
structural fields (model, id, name, role, type, tool_call_id, max_tokens,
temperature, …), or fields whose location is not listed in the returned
field map.

OpenAI chat-completions API coverage
--------------------------------------
  messages[*] where role == "system"
    .content  string                  →  one SYSTEM_MESSAGE text field
    .content  array                   →
      type == "text"                  →  SYSTEM_MESSAGE text field (.text)
      unknown types                   →  unknown field entry

  messages[*] where role in {"user", "assistant", "developer", <future>}
    .content  string                  →  one MESSAGE_TEXT text field
    .content  array                   →
      type == "text"                  →  MESSAGE_TEXT text field (.text)
      type == "refusal"               →  MESSAGE_TEXT text field (.refusal)
      type == "image_url"             →  unscannable IMAGE_URL field
      type == "input_audio"           →  unscannable UNKNOWN field
      type == "file"                  →  unscannable UNKNOWN field
      unknown types                   →  unknown field entry

  messages[*] where role == "assistant"
    .tool_calls[*].function.arguments
      → JSON string parsed recursively; every string leaf value is a
        TOOL_CALL_ARGS scan target.  If arguments cannot be parsed as JSON
        the raw string is registered as a single TOOL_CALL_ARGS field
        (coverage gap noted).

  messages[*] where role == "tool"
    .content  string                  →  one TOOL_RESULT text field
    .content  array                   →
      type == "text"                  →  TOOL_RESULT text field (.text)
      unknown types                   →  unknown field entry

Field map entries NOT created for
------------------------------------
  model, max_tokens, stream, temperature, top_p, stop, n, … (top-level API
    parameters — never PII-bearing)
  messages[*].role                    (structural)
  messages[*].name                    (structural)
  tool_calls[*].id, type             (structural)
  tool_calls[*].function.name        (structural)
  messages[*].tool_call_id           (structural)
  numeric / boolean / null leaf values in function.arguments JSON

Typical usage
-------------
    from pii_guard.providers.openai_parser import parse_openai_request

    field_map = parse_openai_request(payload)
    for field in field_map.text_fields:
        redacted = engine.scan(field.text).redacted_text
        ...  # apply only to identified text locations
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Public enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ScanField(str, Enum):
    """
    Ontology ``scan_field`` values — OpenAI-specific text-bearing locations.

    These values mirror the OpenAI scrubber's ScanField enum so the parser
    and scrubber share a common vocabulary without circular imports.
    """
    SYSTEM_MESSAGE = "system_message"
    MESSAGE_TEXT   = "message_text"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_RESULT    = "tool_result"
    IMAGE_URL      = "image_url"
    UNKNOWN        = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ParsedField — one extracted field entry in the field map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedField:
    """
    Metadata for a single field extracted from an OpenAI chat-completions
    request.

    Attributes
    ----------
    location:
        Dot-notation path in the original payload, e.g.
        ``"messages[0].tool_calls[0].function.arguments.email"``.
    scan_field:
        The ontology category for this location.
    text:
        The raw text string at this location, or ``None`` for unscannable
        (image_url, audio, file) and unknown fields.  Empty string ``""``
        is a valid scannable value (empty text block).
    is_scannable:
        ``True`` when *text* holds the actual string content and Stage-1
        regex detection can be applied.
    is_unscannable:
        ``True`` for image_url parts, input_audio parts, and file parts that
        the Stage-1 regex engine cannot process.  These fields still appear in
        the map so the caller can apply the ``unscannable_action`` policy
        (default: block).
    is_unknown:
        ``True`` for part types not defined in the OpenAI chat-completions
        schema.  These trigger a coverage alarm.
    coverage_gap_reason:
        Human-readable description of *why* the field is not text-scannable.
        Always ``None`` when ``is_scannable=True``.
    has_json_parse_gap:
        ``True`` when this is a tool_call arguments field whose JSON could not
        be parsed — the raw string was registered as a fallback scan target
        but structural validation could not be performed.
    """

    location: str
    scan_field: ScanField
    text: Optional[str]              # None for unscannable / unknown
    is_scannable: bool               # True → text is a string ready to scan
    is_unscannable: bool = False     # True → image_url / audio / file
    is_unknown: bool = False         # True → unrecognized part type
    coverage_gap_reason: Optional[str] = None
    has_json_parse_gap: bool = False  # True → arguments JSON parse failed


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIFieldMap — the structured result of parsing one request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenAIFieldMap:
    """
    Structured map of all field locations extracted from an OpenAI
    chat-completions request.

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
    """

    all_fields: List[ParsedField] = field(default_factory=list)
    model: Optional[str] = None

    @property
    def text_fields(self) -> List[ParsedField]:
        """Scannable text fields — the *only* masking targets."""
        return [f for f in self.all_fields if f.is_scannable]

    @property
    def unscannable_fields(self) -> List[ParsedField]:
        """Image_url / audio / file parts — require policy decision."""
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
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_openai_request(
    payload: Dict[str, Any],
) -> OpenAIFieldMap:
    """
    Parse an OpenAI chat-completions request payload and return a structured
    :class:`OpenAIFieldMap`.

    This function does **not** modify the payload and performs no scanning or
    masking.  It purely walks the schema and records every text-bearing location
    plus any unscannable or unknown locations.

    Parameters
    ----------
    payload:
        OpenAI chat-completions request dict (not mutated).

    Returns
    -------
    OpenAIFieldMap
        Structured field map.  Use ``field_map.text_fields`` to enumerate
        the exact set of locations that the scrubber may mask.
    """
    if not isinstance(payload, dict):
        return OpenAIFieldMap()

    result = OpenAIFieldMap(
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
    )

    # ── Walk the messages array ───────────────────────────────────────────────
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return result

    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        base_loc = f"messages[{msg_idx}]"

        # ── Determine scan_field for this message's content ───────────────────
        if role == "system":
            content_scan_field = ScanField.SYSTEM_MESSAGE
        elif role == "tool":
            content_scan_field = ScanField.TOOL_RESULT
        else:
            # user, assistant, developer, and any future roles
            content_scan_field = ScanField.MESSAGE_TEXT

        # ── Parse content field ───────────────────────────────────────────────
        content = message.get("content")
        if content is not None:
            _parse_content(
                content,
                f"{base_loc}.content",
                content_scan_field,
                result,
            )

        # ── Parse tool_calls (assistant role) ─────────────────────────────────
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
                    _parse_tool_call_arguments(
                        args_str,
                        f"{func_loc}.arguments",
                        result,
                    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsers — one per schema section
# ─────────────────────────────────────────────────────────────────────────────

def _parse_content(
    content: Any,
    base_loc: str,
    scan_field: ScanField,
    result: OpenAIFieldMap,
) -> None:
    """
    Parse a message content field (string or content-part array).

    Mutates *result* in-place by appending parsed fields.
    """
    # Plain string shorthand (all roles accept this)
    if isinstance(content, str):
        result.all_fields.append(ParsedField(
            location=base_loc,
            scan_field=scan_field,
            text=content,
            is_scannable=True,
        ))
        return

    if not isinstance(content, list):
        # Unexpected content type — coverage alarm
        result.all_fields.append(ParsedField(
            location=base_loc,
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason=(
                f"unexpected content type: {type(content).__name__}"
            ),
        ))
        return

    for part_idx, part in enumerate(content):
        part_loc = f"{base_loc}[{part_idx}]"

        if not isinstance(part, dict):
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=f"non-dict content part: {type(part).__name__}",
            ))
            continue

        ptype = part.get("type", "")

        # ── text part ────────────────────────────────────────────────────────
        if ptype == "text":
            text = part.get("text", "") or ""
            result.all_fields.append(ParsedField(
                location=f"{part_loc}.text",
                scan_field=scan_field,
                text=text,
                is_scannable=True,
            ))

        # ── refusal part (assistant only) ─────────────────────────────────────
        elif ptype == "refusal":
            text = part.get("refusal", "") or ""
            result.all_fields.append(ParsedField(
                location=f"{part_loc}.refusal",
                scan_field=scan_field,
                text=text,
                is_scannable=True,
            ))

        # ── image_url part — unscannable ──────────────────────────────────────
        elif ptype == "image_url":
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.IMAGE_URL,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason="image_url content part is not text-scannable in Stage-1",
            ))

        # ── input_audio part — unscannable ────────────────────────────────────
        elif ptype == "input_audio":
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason="input_audio content part is not text-scannable in Stage-1",
            ))

        # ── file part — unscannable without decoding ──────────────────────────
        elif ptype == "file":
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason="file content part is not text-scannable in Stage-1",
            ))

        # ── unknown part type ────────────────────────────────────────────────
        else:
            result.all_fields.append(ParsedField(
                location=part_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=f"unrecognized content part type={ptype!r}",
            ))


def _parse_tool_call_arguments(
    args_str: Any,
    location: str,
    result: OpenAIFieldMap,
) -> None:
    """
    Parse tool_calls[*].function.arguments (a JSON string).

    Parses the JSON string and recursively registers all string leaf values
    as TOOL_CALL_ARGS scan targets.

    If the string cannot be parsed as JSON, the raw string is registered as
    a single TOOL_CALL_ARGS field with ``has_json_parse_gap=True`` (best-effort
    protection, but structural validation could not be performed).

    Non-string arguments types are registered as unknown fields.
    """
    if not isinstance(args_str, str):
        result.all_fields.append(ParsedField(
            location=location,
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason=(
                f"unexpected arguments type: {type(args_str).__name__}"
            ),
        ))
        return

    # Try to parse as JSON
    try:
        args_obj = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        # Fallback: register the raw string as a scan target with a gap note
        result.all_fields.append(ParsedField(
            location=location,
            scan_field=ScanField.TOOL_CALL_ARGS,
            text=args_str,
            is_scannable=True,
            has_json_parse_gap=True,
            coverage_gap_reason=(
                "function.arguments could not be parsed as JSON; "
                "registered as raw text scan target"
            ),
        ))
        return

    # Recursively walk the parsed object
    _parse_json_object(args_obj, location, result)


def _parse_json_object(
    obj: Any,
    location: str,
    result: OpenAIFieldMap,
) -> None:
    """
    Recursively walk a JSON-serialisable tool_call arguments object and
    register all string leaf values as TOOL_CALL_ARGS scan targets.

    Non-string leaves (int, float, bool, None) are not registered — they carry
    no text and are not masking targets.
    """
    if isinstance(obj, str):
        result.all_fields.append(ParsedField(
            location=location,
            scan_field=ScanField.TOOL_CALL_ARGS,
            text=obj,
            is_scannable=True,
        ))
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _parse_json_object(v, f"{location}.{k}", result)
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _parse_json_object(item, f"{location}[{i}]", result)
        return

    # Scalar (int, float, bool, None) — not a text masking target; skip
