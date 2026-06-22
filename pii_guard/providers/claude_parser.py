"""
Claude provider request parser — Sub-AC 1.

Extracts the canonical message/content fields from Anthropic's Messages API
request schema and returns a structured :class:`ClaudeFieldMap`.

This module is **pure parse only** — it walks the request schema, classifies
every field location, and records whether each field is text-scannable.  It
does *not* run detection, apply masking, or modify the payload in any way.

The :class:`ClaudeFieldMap` is the authoritative declaration of what the
scrubber is allowed to process: masking (in :mod:`pii_guard.providers.claude`)
targets *only* the text-bearing fields enumerated here — never arbitrary keys,
structural fields (model, id, name, role, type, max_tokens, tool_use_id, …),
or fields whose location is not listed in the returned field map.

Claude Messages API coverage
----------------------------
  system               string  →  one SYSTEM_PROMPT text field
  system               list    →  one SYSTEM_PROMPT text field per text block
                                  non-text blocks → unknown field entry
  messages[*].content  string  →  one MESSAGE_TEXT text field
  messages[*].content  list    →
    type == "text"              →  MESSAGE_TEXT text field (.text)
    type == "tool_use"          →  TOOL_USE_INPUT text fields (all str leaves
                                   of .input, walked recursively)
    type == "tool_result"       →  TOOL_RESULT text fields (.content string
                                   or .content[*].text of TextBlock array)
    type == "document"
      source.type == "text"     →  DOCUMENT_BLOCK text field (.source.data)
      source.type == "base64"
               or "url"         →  unscannable DOCUMENT_BLOCK field
    type == "image"             →  unscannable IMAGE field
    other types                 →  unknown field entry (coverage alarm)

Field map entries not created for
-----------------------------------
  model, max_tokens, stream, temperature, … (top-level API parameters)
  messages[*].role              (structural)
  block id, name, type, tool_use_id, media_type  (structural metadata)
  numeric / boolean / null leaf values in tool_use.input

Typical usage
-------------
    from pii_guard.providers.claude_parser import parse_claude_request

    field_map = parse_claude_request(payload)
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
    Ontology ``scan_field`` values — Claude-specific text-bearing locations.

    These values mirror the Claude scrubber's ScanField enum so the parser
    and scrubber share a common vocabulary without circular imports.
    """
    SYSTEM_PROMPT  = "system_prompt"
    MESSAGE_TEXT   = "message_text"
    TOOL_USE_INPUT = "tool_use_input"
    TOOL_RESULT    = "tool_result"
    DOCUMENT_BLOCK = "document_block"
    IMAGE          = "image"
    UNKNOWN        = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ParsedField — one extracted field entry in the field map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedField:
    """
    Metadata for a single field extracted from a Claude API request.

    Attributes
    ----------
    location:
        Dot-notation path in the original payload, e.g.
        ``"messages[0].content[2].input.email"``.
    scan_field:
        The ontology category for this location.
    text:
        The raw text string at this location, or ``None`` for unscannable
        (image, base64 document) and unknown fields.  Empty string ``""``
        is a valid scannable value (empty text block).
    is_scannable:
        ``True`` when *text* holds the actual string content and Stage-1
        regex detection can be applied.
    is_unscannable:
        ``True`` for image blocks and non-text document sources that the
        Stage-1 regex engine cannot process.  These fields still appear in
        the map so the caller can apply the ``unscannable_action`` policy
        (default: block).
    is_unknown:
        ``True`` for block types not defined in the Claude Messages API
        schema.  These trigger a coverage alarm.
    coverage_gap_reason:
        Human-readable description of *why* the field is not text-scannable.
        Always ``None`` when ``is_scannable=True``.
    """

    location: str
    scan_field: ScanField
    text: Optional[str]            # None for unscannable / unknown
    is_scannable: bool             # True → text is a string ready to scan
    is_unscannable: bool = False   # True → image / base64 / url source
    is_unknown: bool = False       # True → unrecognized block type
    coverage_gap_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# ClaudeFieldMap — the structured result of parsing one request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClaudeFieldMap:
    """
    Structured map of all field locations extracted from a Claude request.

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
        The ``anthropic-version`` header value if injected into the payload
        by the caller, else ``None``.  Informational only.
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
        """Image / base64 / url sources — require policy decision."""
        return [f for f in self.all_fields if f.is_unscannable]

    @property
    def unknown_fields(self) -> List[ParsedField]:
        """Unrecognized block types — trigger coverage alarm."""
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

def parse_claude_request(
    payload: Dict[str, Any],
    *,
    api_version: Optional[str] = None,
) -> ClaudeFieldMap:
    """
    Parse a Claude Messages API request payload and return a structured
    :class:`ClaudeFieldMap`.

    This function does **not** modify the payload and performs no scanning or
    masking.  It purely walks the schema and records every text-bearing location
    plus any unscannable or unknown locations.

    Parameters
    ----------
    payload:
        Claude Messages API request dict (not mutated).
    api_version:
        Optional ``anthropic-version`` header value, used to populate
        ``ClaudeFieldMap.api_version`` for version-mismatch alarms.

    Returns
    -------
    ClaudeFieldMap
        Structured field map.  Use ``field_map.text_fields`` to enumerate
        the exact set of locations that the scrubber may mask.
    """
    result = ClaudeFieldMap(
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        api_version=api_version,
    )

    if not isinstance(payload, dict):
        return result

    # ── 1. system prompt ─────────────────────────────────────────────────────
    if "system" in payload:
        _parse_system(payload["system"], result)

    # ── 2. messages ──────────────────────────────────────────────────────────
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if content is None:
                continue
            base_loc = f"messages[{msg_idx}].content"
            _parse_content(content, base_loc, result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsers — one per schema section
# ─────────────────────────────────────────────────────────────────────────────

def _parse_system(system: Any, result: ClaudeFieldMap) -> None:
    """
    Parse the ``system`` field.

    Mutates *result* in-place by appending parsed fields.
    """
    if isinstance(system, str):
        result.all_fields.append(ParsedField(
            location="system",
            scan_field=ScanField.SYSTEM_PROMPT,
            text=system,
            is_scannable=True,
        ))
        return

    if isinstance(system, list):
        for i, block in enumerate(system):
            loc = f"system[{i}]"
            if not isinstance(block, dict):
                # Non-dict item in system array — unknown
                result.all_fields.append(ParsedField(
                    location=loc,
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=f"non-dict item in system array: {type(block).__name__}",
                ))
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "") or ""
                result.all_fields.append(ParsedField(
                    location=f"{loc}.text",
                    scan_field=ScanField.SYSTEM_PROMPT,
                    text=text,
                    is_scannable=True,
                ))
            else:
                # Unrecognised system block type
                result.all_fields.append(ParsedField(
                    location=loc,
                    scan_field=ScanField.UNKNOWN,
                    text=None,
                    is_scannable=False,
                    is_unknown=True,
                    coverage_gap_reason=(
                        f"unrecognized system block type={btype!r}"
                    ),
                ))
        return

    # Unexpected type for system (future API extension)
    result.all_fields.append(ParsedField(
        location="system",
        scan_field=ScanField.UNKNOWN,
        text=None,
        is_scannable=False,
        is_unknown=True,
        coverage_gap_reason=f"unexpected system type: {type(system).__name__}",
    ))


def _parse_content(
    content: Any,
    base_loc: str,
    result: ClaudeFieldMap,
) -> None:
    """
    Parse a ``content`` field (string or content block array).

    Mutates *result* in-place.
    """
    # Plain string shorthand (older API / single-turn)
    if isinstance(content, str):
        result.all_fields.append(ParsedField(
            location=base_loc,
            scan_field=ScanField.MESSAGE_TEXT,
            text=content,
            is_scannable=True,
        ))
        return

    if not isinstance(content, list):
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

    for blk_idx, block in enumerate(content):
        blk_loc = f"{base_loc}[{blk_idx}]"
        if not isinstance(block, dict):
            result.all_fields.append(ParsedField(
                location=blk_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=f"non-dict content block: {type(block).__name__}",
            ))
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "") or ""
            result.all_fields.append(ParsedField(
                location=f"{blk_loc}.text",
                scan_field=ScanField.MESSAGE_TEXT,
                text=text,
                is_scannable=True,
            ))

        elif btype == "tool_use":
            tool_input = block.get("input") or {}
            _parse_json_object(tool_input, f"{blk_loc}.input", result)

        elif btype == "tool_result":
            result_content = block.get("content")
            if result_content is not None:
                _parse_tool_result_content(
                    result_content, f"{blk_loc}.content", result
                )

        elif btype == "document":
            _parse_document_block(block, blk_loc, result)

        elif btype == "image":
            result.all_fields.append(ParsedField(
                location=blk_loc,
                scan_field=ScanField.IMAGE,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason="image block is not text-scannable in Stage-1",
            ))

        else:
            # Unknown content block type → coverage alarm
            result.all_fields.append(ParsedField(
                location=blk_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=f"unrecognized content block type={btype!r}",
            ))


def _parse_tool_result_content(
    content: Any,
    loc: str,
    result: ClaudeFieldMap,
) -> None:
    """Parse tool_result .content (string or TextBlock array)."""
    if isinstance(content, str):
        result.all_fields.append(ParsedField(
            location=loc,
            scan_field=ScanField.TOOL_RESULT,
            text=content,
            is_scannable=True,
        ))
        return

    if not isinstance(content, list):
        result.all_fields.append(ParsedField(
            location=loc,
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason=(
                f"unexpected tool_result content type: {type(content).__name__}"
            ),
        ))
        return

    for i, block in enumerate(content):
        item_loc = f"{loc}[{i}]"
        if not isinstance(block, dict):
            result.all_fields.append(ParsedField(
                location=item_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=f"non-dict item in tool_result content",
            ))
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "") or ""
            result.all_fields.append(ParsedField(
                location=f"{item_loc}.text",
                scan_field=ScanField.TOOL_RESULT,
                text=text,
                is_scannable=True,
            ))
        elif btype == "image":
            result.all_fields.append(ParsedField(
                location=item_loc,
                scan_field=ScanField.IMAGE,
                text=None,
                is_scannable=False,
                is_unscannable=True,
                coverage_gap_reason="image inside tool_result is not text-scannable",
            ))
        else:
            result.all_fields.append(ParsedField(
                location=item_loc,
                scan_field=ScanField.UNKNOWN,
                text=None,
                is_scannable=False,
                is_unknown=True,
                coverage_gap_reason=(
                    f"unrecognized tool_result block type={btype!r}"
                ),
            ))


def _parse_document_block(
    block: dict,
    blk_loc: str,
    result: ClaudeFieldMap,
) -> None:
    """Parse a document content block."""
    source = block.get("source")
    if not isinstance(source, dict):
        result.all_fields.append(ParsedField(
            location=f"{blk_loc}.source",
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason="missing or non-dict document source",
        ))
        return

    src_type = source.get("type", "")
    doc_loc = f"{blk_loc}.source"

    if src_type == "text":
        data = source.get("data", "") or ""
        result.all_fields.append(ParsedField(
            location=f"{doc_loc}.data",
            scan_field=ScanField.DOCUMENT_BLOCK,
            text=data,
            is_scannable=True,
        ))

    elif src_type in ("base64", "url"):
        result.all_fields.append(ParsedField(
            location=doc_loc,
            scan_field=ScanField.DOCUMENT_BLOCK,
            text=None,
            is_scannable=False,
            is_unscannable=True,
            coverage_gap_reason=(
                f"document source type={src_type!r} is not text-scannable in Stage-1"
            ),
        ))

    else:
        result.all_fields.append(ParsedField(
            location=doc_loc,
            scan_field=ScanField.UNKNOWN,
            text=None,
            is_scannable=False,
            is_unknown=True,
            coverage_gap_reason=f"unrecognized document source type={src_type!r}",
        ))


def _parse_json_object(
    obj: Any,
    location: str,
    result: ClaudeFieldMap,
) -> None:
    """
    Recursively walk a JSON-serialisable tool_use input object and register
    all string leaf values as TOOL_USE_INPUT scan targets.

    Non-string leaves (int, float, bool, None) are not registered — they carry
    no text and are not masking targets.
    """
    if isinstance(obj, str):
        result.all_fields.append(ParsedField(
            location=location,
            scan_field=ScanField.TOOL_USE_INPUT,
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
