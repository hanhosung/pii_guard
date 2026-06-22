"""
Schema Coverage Detection — Sub-AC 3a.

Pure detection functions that:

  (a) Diff an incoming request's field set against the provider's known field
      schema at every structural level, returning a :class:`FieldDelta` for
      each location that contains keys not present in the published API spec.

  (b) Compare a request's declared API version against the per-provider
      known-good version list, returning a :class:`VersionDelta` for any
      unrecognized or future-looking version string.

Both functions are **pure / side-effect-free** — they never modify the payload
and do not run any detection engine.  They are designed to be composed with
the scrubbing pipeline so that the caller can raise a coverage alarm
(``unknown_field_action``) before forwarding the request.

Design invariants
-----------------
  - False negatives are forbidden: every extra key in the request that is not
    in the known schema must appear in at least one returned ``FieldDelta``.
  - False positives are forbidden: all keys that belong to the stable published
    API for the provider (including optional ones) must be in the schema so
    they do not generate spurious deltas.
  - Tool-use *inputs* and function-call *arguments* carry user-defined
    schemas — they are intentionally excluded from field-set diffing so that
    novel keys inside those objects do not generate false alarms.  The content
    field itself (the value of ``input`` / ``arguments``) is still scanned for
    PII by the scrubbing layer.

Providers covered
-----------------
  claude   Anthropic Messages API  (anthropic-version header / body field)
  openai   OpenAI Chat Completions (header x-stainless-* / path /v1/)
  gemini   Google Generative AI    (path /v1/ or /v1beta/)

Usage
-----
    from pii_guard.providers.schema_coverage import (
        diff_claude_fields,
        diff_openai_fields,
        diff_gemini_fields,
        diff_api_version,
        FieldDelta,
        VersionDelta,
    )

    # (a) Field-set diff
    deltas = diff_claude_fields(payload)
    for delta in deltas:
        if delta.extra_keys:
            raise_coverage_alarm(delta)

    # (b) Version diff
    vd = diff_api_version("2099-01-01", "claude")
    if vd.is_unknown:
        raise_coverage_alarm(vd)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Public result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldDelta:
    """
    Structured delta for unknown/extra fields found at one structural level
    inside a provider request payload.

    Attributes
    ----------
    provider:
        ``"claude"``, ``"openai"``, or ``"gemini"``.
    path:
        Dot-/bracket-notation path to the parent dict that contains extra
        keys, e.g. ``""`` = root, ``"messages[0]"``, ``"messages[0].content[1]"``.
    extra_keys:
        Keys present in the request at *path* that are **not** in the
        published API schema.  Non-empty ↔ there is a coverage gap.
    known_keys:
        Complete set of keys the provider schema defines at *path*.
    actual_keys:
        All keys present in the request dict at *path*.
    """

    provider: str
    path: str
    extra_keys: FrozenSet[str]
    known_keys: FrozenSet[str]
    actual_keys: FrozenSet[str]


@dataclass
class VersionDelta:
    """
    Structured delta for an unrecognized or future-looking API version.

    Attributes
    ----------
    provider:
        ``"claude"``, ``"openai"``, or ``"gemini"``.
    declared_version:
        The version string found in the request or header.
    known_versions:
        Tuple of version strings the proxy knows how to parse completely.
    is_future:
        ``True`` when the declared version looks newer than all known-good
        versions (date-ordered for Claude, numeric-prefix-ordered for Gemini /
        OpenAI).  A future version likely introduces new fields we cannot
        fully scan.
    is_unknown:
        ``True`` whenever *declared_version* is not in *known_versions*
        (covers both future and unrecognized/malformed versions).
    location:
        Human-readable description of where the version was found,
        e.g. ``"header:anthropic-version"``, ``"path:/v2beta/"``.
    """

    provider: str
    declared_version: str
    known_versions: Tuple[str, ...]
    is_future: bool
    is_unknown: bool
    location: str


# ─────────────────────────────────────────────────────────────────────────────
# Per-provider API version registries
# ─────────────────────────────────────────────────────────────────────────────

# Anthropic Messages API: ISO date string sent as the ``anthropic-version``
# HTTP header (or sometimes embedded in the request body).
_CLAUDE_KNOWN_VERSIONS: Tuple[str, ...] = (
    "2023-06-01",  # current stable
)

# OpenAI Chat Completions: path prefix (``/v1/…``).  If the caller passes an
# explicit openai-version header, compare against these.
_OPENAI_KNOWN_VERSIONS: Tuple[str, ...] = (
    "v1",          # current stable
)

# Google Generative AI (Gemini): path prefix (``/v1/…`` or ``/v1beta/…``).
_GEMINI_KNOWN_VERSIONS: Tuple[str, ...] = (
    "v1",          # stable GA release
    "v1beta",      # public beta (explicitly supported by the proxy)
    "v1alpha",     # early preview (explicitly supported by the proxy)
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-provider field schemas
#
# Each schema is a dict mapping a *path pattern* (using ``[*]`` for array
# wildcards) to a frozenset of valid key names at that structural level.
#
# Rules:
#   - Include every key in the *published stable spec* for that provider.
#   - Do NOT include keys inside tool-use ``input`` / function ``arguments``
#     (user-defined; intentionally excluded to prevent false alarms).
#   - Both camelCase and snake_case variants are listed for Gemini.
#   - ``cache_control`` is included for Claude because prompt-caching is GA.
# ─────────────────────────────────────────────────────────────────────────────

# ── Claude ────────────────────────────────────────────────────────────────────

_CLAUDE_SCHEMA: Dict[str, FrozenSet[str]] = {
    # Top-level request object
    "": frozenset({
        "model", "messages", "max_tokens",
        "system", "stream",
        "temperature", "top_p", "top_k", "stop_sequences",
        "metadata", "tools", "tool_choice",
        "anthropic_version", "betas",
    }),

    # Each element of messages[]
    "messages[*]": frozenset({
        "role", "content",
    }),

    # Known block-type keys at the content-block level.
    # Content blocks are dispatched by "type"; here we list the union of all
    # known block-type-specific keys plus the base "type" key.
    "messages[*].content[*]": frozenset({
        # structural
        "type",
        # text block
        "text",
        # tool_use block
        "id", "name", "input",
        # tool_result block
        "tool_use_id", "content",
        # document block
        "source",
        # thinking block (extended thinking GA)
        "thinking", "signature",
        # cache control (prompt-caching GA) — appears on most block types
        "cache_control",
        # media_type appears on some block subtypes (document, image)
        "media_type",
    }),

    # system block array (when system is a TextBlock array)
    "system[*]": frozenset({
        "type", "text", "cache_control",
    }),

    # document / image source objects
    "**.source": frozenset({
        "type", "media_type", "data", "url",
    }),

    # Tool definitions in the tools[] array
    "tools[*]": frozenset({
        "name", "description", "input_schema", "cache_control",
    }),

    # tool_choice object
    "tool_choice": frozenset({
        "type", "name",
    }),
}

# ── OpenAI ────────────────────────────────────────────────────────────────────

_OPENAI_SCHEMA: Dict[str, FrozenSet[str]] = {
    # Top-level chat-completions request object
    "": frozenset({
        "model", "messages",
        # Generation parameters
        "temperature", "top_p", "n", "stream", "stream_options",
        "stop", "max_tokens", "max_completion_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias",
        "logprobs", "top_logprobs",
        # Tool use
        "tools", "tool_choice", "parallel_tool_calls",
        # Legacy function calling
        "function_call", "functions",
        # Output format
        "response_format", "modalities", "audio",
        # Reproducibility / tracking
        "seed", "service_tier", "metadata", "store", "user",
        # Reasoning models
        "reasoning_effort", "prediction",
        # Web search (Responses API overlap)
        "web_search_options",
    }),

    # Each element of messages[]
    "messages[*]": frozenset({
        "role", "content", "name",
        "tool_calls", "tool_call_id",
        "refusal", "audio",
    }),

    # Each tool call inside messages[*].tool_calls[]
    "messages[*].tool_calls[*]": frozenset({
        "id", "type", "function",
    }),

    # The function object inside a tool_call
    "messages[*].tool_calls[*].function": frozenset({
        "name", "arguments",
    }),

    # Content parts when messages[*].content is an array
    "messages[*].content[*]": frozenset({
        "type", "text",
        "image_url", "input_audio", "file", "refusal",
    }),

    # Tool definitions
    "tools[*]": frozenset({
        "type", "function",
    }),

    "tools[*].function": frozenset({
        "name", "description", "parameters", "strict",
    }),

    # Legacy function definitions (function_call API)
    "functions[*]": frozenset({
        "name", "description", "parameters",
    }),
}

# ── Gemini ────────────────────────────────────────────────────────────────────

_GEMINI_SCHEMA: Dict[str, FrozenSet[str]] = {
    # Top-level generateContent request object (REST format)
    "": frozenset({
        "contents",
        # System instruction — both naming styles
        "systemInstruction", "system_instruction",
        # Tools / tool config — both naming styles
        "tools", "toolConfig", "tool_config",
        # Safety — both naming styles
        "safetySettings", "safety_settings",
        # Generation config — both naming styles
        "generationConfig", "generation_config",
        # Caching — both naming styles
        "cachedContent", "cached_content",
        # Model is sometimes in the request body (Python SDK)
        "model",
    }),

    # Each element of contents[]
    "contents[*]": frozenset({
        "role", "parts",
    }),

    # Each part inside contents[*].parts[]
    # A part contains exactly one "payload" key identifying its type.
    "contents[*].parts[*]": frozenset({
        "text",
        "functionCall", "function_call",
        "functionResponse", "function_response",
        "inlineData", "inline_data",
        "fileData", "file_data",
        "executableCode", "executable_code",
        "codeExecutionResult", "code_execution_result",
        # thought / thinking parts (Gemini 2.0+ flash-thinking)
        "thought",
        # video metadata
        "videoMetadata", "video_metadata",
    }),

    # systemInstruction / system_instruction body
    "systemInstruction": frozenset({"role", "parts"}),
    "system_instruction": frozenset({"role", "parts"}),

    # Parts inside systemInstruction.parts[]
    "systemInstruction.parts[*]": frozenset({
        "text",
        "functionCall", "function_call",
        "functionResponse", "function_response",
        "inlineData", "inline_data",
        "fileData", "file_data",
        "executableCode", "executable_code",
        "codeExecutionResult", "code_execution_result",
        "thought",
        "videoMetadata", "video_metadata",
    }),

    # generationConfig / generation_config keys
    "generationConfig": frozenset({
        "stopSequences", "stop_sequences",
        "responseMimeType", "response_mime_type",
        "responseSchema", "response_schema",
        "candidateCount", "candidate_count",
        "maxOutputTokens", "max_output_tokens",
        "temperature", "topP", "top_p", "topK", "top_k",
        "presencePenalty", "presence_penalty",
        "frequencyPenalty", "frequency_penalty",
        "responseLogprobs", "response_logprobs",
        "logprobs",
        "enableEnhancedCivicAnswers", "enable_enhanced_civic_answers",
        "speechConfig", "speech_config",
        "audioTimestamp", "audio_timestamp",
        "thinkingConfig", "thinking_config",
        "mediaResolution", "media_resolution",
    }),
    "generation_config": frozenset({
        "stopSequences", "stop_sequences",
        "responseMimeType", "response_mime_type",
        "responseSchema", "response_schema",
        "candidateCount", "candidate_count",
        "maxOutputTokens", "max_output_tokens",
        "temperature", "topP", "top_p", "topK", "top_k",
        "presencePenalty", "presence_penalty",
        "frequencyPenalty", "frequency_penalty",
        "responseLogprobs", "response_logprobs",
        "logprobs",
        "enableEnhancedCivicAnswers", "enable_enhanced_civic_answers",
        "speechConfig", "speech_config",
        "audioTimestamp", "audio_timestamp",
        "thinkingConfig", "thinking_config",
        "mediaResolution", "media_resolution",
    }),
}


# ─────────────────────────────────────────────────────────────────────────────
# (a) Field-set diff functions
# ─────────────────────────────────────────────────────────────────────────────

def diff_claude_fields(payload: Dict[str, Any]) -> List[FieldDelta]:
    """
    Diff a Claude Messages API request against the known field schema.

    Walks the request at the root, messages, content-block, system-block,
    source, and tool-definition levels and returns one :class:`FieldDelta`
    per structural location that contains keys not present in the published
    Anthropic API spec.

    Tool-use ``input`` objects are **not** diffed (user-defined schema).

    Parameters
    ----------
    payload:
        Claude Messages API request dict (not mutated).

    Returns
    -------
    list[FieldDelta]
        Empty list ↔ no unknown fields found.  Non-empty ↔ coverage alarm.
    """
    if not isinstance(payload, dict):
        return []

    deltas: List[FieldDelta] = []

    # Root
    _check_keys(payload, "", "claude", _CLAUDE_SCHEMA[""], deltas)

    # system — when it's an array of TextBlocks
    system = payload.get("system")
    if isinstance(system, list):
        for i, block in enumerate(system):
            if isinstance(block, dict):
                path = f"system[{i}]"
                _check_keys(block, path, "claude", _CLAUDE_SCHEMA["system[*]"], deltas)

    # messages
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg_i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            msg_path = f"messages[{msg_i}]"
            _check_keys(msg, msg_path, "claude", _CLAUDE_SCHEMA["messages[*]"], deltas)

            # content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for blk_i, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    blk_path = f"messages[{msg_i}].content[{blk_i}]"
                    btype = block.get("type", "")
                    known = _claude_block_known_keys(btype)
                    _check_keys(block, blk_path, "claude", known, deltas)

                    # document / image source
                    source = block.get("source")
                    if isinstance(source, dict):
                        _check_keys(
                            source,
                            f"{blk_path}.source",
                            "claude",
                            _CLAUDE_SCHEMA["**.source"],
                            deltas,
                        )

                    # tool_result nested content blocks
                    if btype == "tool_result":
                        nested = block.get("content")
                        if isinstance(nested, list):
                            for nb_i, nblock in enumerate(nested):
                                if not isinstance(nblock, dict):
                                    continue
                                nb_path = f"{blk_path}.content[{nb_i}]"
                                nb_type = nblock.get("type", "")
                                nknown = _claude_block_known_keys(nb_type)
                                _check_keys(nblock, nb_path, "claude", nknown, deltas)

    # tools array
    tools = payload.get("tools")
    if isinstance(tools, list):
        for ti, tool in enumerate(tools):
            if isinstance(tool, dict):
                _check_keys(
                    tool, f"tools[{ti}]", "claude",
                    _CLAUDE_SCHEMA["tools[*]"], deltas,
                )

    return deltas


def diff_openai_fields(payload: Dict[str, Any]) -> List[FieldDelta]:
    """
    Diff an OpenAI Chat Completions request against the known field schema.

    Walks the root, messages, tool_calls, tool_call function, and content-part
    levels.  Function ``arguments`` JSON is not diffed (user-defined schema).

    Parameters
    ----------
    payload:
        OpenAI Chat Completions request dict (not mutated).

    Returns
    -------
    list[FieldDelta]
        Empty list ↔ no unknown fields found.  Non-empty ↔ coverage alarm.
    """
    if not isinstance(payload, dict):
        return []

    deltas: List[FieldDelta] = []

    # Root
    _check_keys(payload, "", "openai", _OPENAI_SCHEMA[""], deltas)

    # messages
    messages = payload.get("messages")
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            msg_path = f"messages[{mi}]"
            _check_keys(msg, msg_path, "openai", _OPENAI_SCHEMA["messages[*]"], deltas)

            # content parts when content is an array
            content = msg.get("content")
            if isinstance(content, list):
                for pi, part in enumerate(content):
                    if not isinstance(part, dict):
                        continue
                    part_path = f"messages[{mi}].content[{pi}]"
                    ptype = part.get("type", "")
                    known = _openai_content_part_known_keys(ptype)
                    _check_keys(part, part_path, "openai", known, deltas)

            # tool_calls
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for ti, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    tc_path = f"messages[{mi}].tool_calls[{ti}]"
                    _check_keys(
                        tc, tc_path, "openai",
                        _OPENAI_SCHEMA["messages[*].tool_calls[*]"],
                        deltas,
                    )
                    func = tc.get("function")
                    if isinstance(func, dict):
                        _check_keys(
                            func, f"{tc_path}.function", "openai",
                            _OPENAI_SCHEMA["messages[*].tool_calls[*].function"],
                            deltas,
                        )

    # tools array
    tools = payload.get("tools")
    if isinstance(tools, list):
        for ti, tool in enumerate(tools):
            if isinstance(tool, dict):
                _check_keys(
                    tool, f"tools[{ti}]", "openai",
                    _OPENAI_SCHEMA["tools[*]"], deltas,
                )
                func = tool.get("function")
                if isinstance(func, dict):
                    _check_keys(
                        func, f"tools[{ti}].function", "openai",
                        _OPENAI_SCHEMA["tools[*].function"], deltas,
                    )

    return deltas


def diff_gemini_fields(payload: Dict[str, Any]) -> List[FieldDelta]:
    """
    Diff a Gemini (Google Generative AI) request against the known field schema.

    Walks the root, contents, parts, systemInstruction, and generationConfig
    levels.  Both camelCase and snake_case field-name variants are recognised.

    functionCall.args and functionResponse.response are not diffed
    (user-defined schemas).

    Parameters
    ----------
    payload:
        Gemini API request dict (not mutated).

    Returns
    -------
    list[FieldDelta]
        Empty list ↔ no unknown fields found.  Non-empty ↔ coverage alarm.
    """
    if not isinstance(payload, dict):
        return []

    deltas: List[FieldDelta] = []

    # Root
    _check_keys(payload, "", "gemini", _GEMINI_SCHEMA[""], deltas)

    # contents
    contents = payload.get("contents")
    if isinstance(contents, list):
        for ci, content_item in enumerate(contents):
            if not isinstance(content_item, dict):
                continue
            ci_path = f"contents[{ci}]"
            _check_keys(
                content_item, ci_path, "gemini",
                _GEMINI_SCHEMA["contents[*]"], deltas,
            )
            parts = content_item.get("parts")
            if isinstance(parts, list):
                for pi, part in enumerate(parts):
                    if not isinstance(part, dict):
                        continue
                    part_path = f"contents[{ci}].parts[{pi}]"
                    _check_keys(
                        part, part_path, "gemini",
                        _GEMINI_SCHEMA["contents[*].parts[*]"], deltas,
                    )

    # systemInstruction / system_instruction
    for si_key in ("systemInstruction", "system_instruction"):
        si = payload.get(si_key)
        if si is None:
            continue
        if isinstance(si, dict):
            _check_keys(
                si, si_key, "gemini",
                _GEMINI_SCHEMA.get(si_key, frozenset({"role", "parts"})),
                deltas,
            )
            parts = si.get("parts")
            if isinstance(parts, list):
                pattern = f"{si_key}.parts[*]"
                known = _GEMINI_SCHEMA.get(
                    pattern, _GEMINI_SCHEMA["systemInstruction.parts[*]"]
                )
                for pi, part in enumerate(parts):
                    if not isinstance(part, dict):
                        continue
                    _check_keys(
                        part, f"{si_key}.parts[{pi}]", "gemini", known, deltas,
                    )

    # generationConfig / generation_config
    for gc_key in ("generationConfig", "generation_config"):
        gc = payload.get(gc_key)
        if isinstance(gc, dict):
            _check_keys(
                gc, gc_key, "gemini",
                _GEMINI_SCHEMA.get(gc_key, frozenset()),
                deltas,
            )

    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# (b) API version diff
# ─────────────────────────────────────────────────────────────────────────────

def diff_api_version(
    declared_version: str,
    provider: str,
    *,
    location: Optional[str] = None,
) -> VersionDelta:
    """
    Compare *declared_version* against the known-good version list for the
    given *provider*.

    Parameters
    ----------
    declared_version:
        The version string found in the request header or path.
        Examples: ``"2023-06-01"`` (Claude), ``"v1"`` (Gemini/OpenAI),
        ``"v2beta"`` (future Gemini), ``"2099-12-31"`` (future Claude).
    provider:
        ``"claude"``, ``"openai"``, or ``"gemini"``.
    location:
        Human-readable description of where the version was obtained,
        e.g. ``"header:anthropic-version"``.  Defaults to a sensible
        per-provider string.

    Returns
    -------
    VersionDelta
        ``is_unknown=True`` whenever *declared_version* is not in the
        known-good list.  ``is_future=True`` when the version appears
        lexicographically / numerically newer than the latest known version.
    """
    known, default_loc = _version_registry(provider)

    is_unknown = declared_version not in known
    is_future = _is_version_future(declared_version, known, provider) if is_unknown else False

    return VersionDelta(
        provider=provider,
        declared_version=declared_version,
        known_versions=known,
        is_future=is_future,
        is_unknown=is_unknown,
        location=location if location is not None else default_loc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: diff both fields and version in one call
# ─────────────────────────────────────────────────────────────────────────────

def diff_request(
    payload: Dict[str, Any],
    provider: str,
    *,
    api_version: Optional[str] = None,
    version_location: Optional[str] = None,
) -> Tuple[List[FieldDelta], Optional[VersionDelta]]:
    """
    Convenience wrapper: run both field-set diff and version diff.

    Parameters
    ----------
    payload:
        Provider request dict.
    provider:
        ``"claude"``, ``"openai"``, or ``"gemini"``.
    api_version:
        Optional declared version string (header / path segment).
        If ``None``, no version delta is produced.
    version_location:
        Forwarded to :func:`diff_api_version` as *location*.

    Returns
    -------
    (field_deltas, version_delta)
        *field_deltas* is always a list (may be empty).
        *version_delta* is ``None`` when *api_version* is not supplied.
    """
    _DIFF_FN = {
        "claude": diff_claude_fields,
        "openai": diff_openai_fields,
        "gemini": diff_gemini_fields,
    }
    diff_fn = _DIFF_FN.get(provider)
    if diff_fn is None:
        raise ValueError(f"Unknown provider: {provider!r}")

    field_deltas = diff_fn(payload)
    version_delta: Optional[VersionDelta] = None
    if api_version is not None:
        version_delta = diff_api_version(
            api_version, provider, location=version_location
        )

    return field_deltas, version_delta


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_keys(
    obj: dict,
    path: str,
    provider: str,
    known: FrozenSet[str],
    out: List[FieldDelta],
) -> None:
    """
    Compare the keys of *obj* against *known*.

    Appends a :class:`FieldDelta` to *out* if and only if there are extra
    keys (keys in *obj* not present in *known*).

    Does **not** append a delta when all keys are known (no false positives)
    or when *obj* is empty.
    """
    actual = frozenset(obj.keys())
    extra = actual - known
    if extra:
        out.append(FieldDelta(
            provider=provider,
            path=path,
            extra_keys=extra,
            known_keys=known,
            actual_keys=actual,
        ))


def _claude_block_known_keys(btype: str) -> FrozenSet[str]:
    """
    Return the known keys for a Claude content block of the given *btype*.

    Falls back to the generic base set so that any *extra* keys in blocks
    of known types are still flagged.
    """
    _TYPE_SPECIFIC: Dict[str, FrozenSet[str]] = {
        "text": frozenset({"type", "text", "cache_control"}),
        "tool_use": frozenset({"type", "id", "name", "input", "cache_control"}),
        "tool_result": frozenset({"type", "tool_use_id", "content", "cache_control",
                                   "is_error"}),
        "document": frozenset({"type", "source", "title", "cache_control",
                                "citations"}),
        "image": frozenset({"type", "source", "cache_control"}),
        "thinking": frozenset({"type", "thinking", "signature"}),
        "redacted_thinking": frozenset({"type", "data"}),
    }
    return _TYPE_SPECIFIC.get(btype, _CLAUDE_SCHEMA["messages[*].content[*]"])


def _openai_content_part_known_keys(ptype: str) -> FrozenSet[str]:
    """Return the known keys for an OpenAI content part of the given *ptype*."""
    _TYPE_SPECIFIC: Dict[str, FrozenSet[str]] = {
        "text": frozenset({"type", "text"}),
        "image_url": frozenset({"type", "image_url"}),
        "input_audio": frozenset({"type", "input_audio"}),
        "file": frozenset({"type", "file"}),
        "refusal": frozenset({"type", "refusal"}),
    }
    return _TYPE_SPECIFIC.get(ptype, _OPENAI_SCHEMA["messages[*].content[*]"])


def _version_registry(provider: str) -> Tuple[Tuple[str, ...], str]:
    """Return (known_versions_tuple, default_location_string) for *provider*."""
    _REGISTRY: Dict[str, Tuple[Tuple[str, ...], str]] = {
        "claude": (_CLAUDE_KNOWN_VERSIONS, "header:anthropic-version"),
        "openai": (_OPENAI_KNOWN_VERSIONS, "path:/v{N}/"),
        "gemini": (_GEMINI_KNOWN_VERSIONS, "path:/v{N}[beta|alpha]/"),
    }
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider for version registry: {provider!r}")
    return _REGISTRY[provider]


# Version-comparison helpers
# --------------------------------------------------------------------------
# Claude:  ISO date strings — lexicographic order is chronological.
# OpenAI:  path prefix like "v1" — compare leading integer after 'v'.
# Gemini:  path prefix like "v1", "v1beta", "v2" — compare leading integer.

def _is_version_future(
    declared: str,
    known: Tuple[str, ...],
    provider: str,
) -> bool:
    """
    Return ``True`` when *declared* looks newer than every version in *known*.

    This is a heuristic: it uses ISO-date ordering for Claude and
    leading-integer ordering for OpenAI / Gemini.  Malformed strings that
    cannot be parsed return ``False`` (unknown but not asserted future).
    """
    if not known:
        return False

    if provider == "claude":
        # ISO date format YYYY-MM-DD — lexicographic comparison is correct.
        # Validate that declared matches the format before comparing.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", declared):
            return False
        try:
            max_known = max(known)  # latest date lexicographically
            return declared > max_known
        except (ValueError, TypeError):
            return False

    if provider in ("openai", "gemini"):
        # Extract the leading integer from strings like "v1", "v1beta", "v2alpha"
        def _num(s: str) -> Optional[int]:
            m = re.match(r"[vV]?(\d+)", s)
            return int(m.group(1)) if m else None

        declared_num = _num(declared)
        if declared_num is None:
            return False
        known_nums = [n for s in known if (n := _num(s)) is not None]
        if not known_nums:
            return False
        return declared_num > max(known_nums)

    return False
