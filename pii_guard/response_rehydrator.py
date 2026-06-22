"""
PII-Guard response post-processor (Sub-AC 2c).

Rehydrates ``[CATEGORY_N]`` placeholder tokens that appear inside LLM responses
before returning them to the calling agent.  A separate ``terminal_restore`` flag
(default **OFF**) controls whether the same substitution is applied to
terminal-rendered output.

Motivation
----------
When the proxy masks outbound requests it replaces real values with indexed
placeholders (e.g. ``alice@corp.io`` → ``[EMAIL_1]``).  An LLM may echo those
placeholders back in its response (e.g. "I'll reply to [EMAIL_1] for you.").
Without rehydration the calling **agent** would receive a placeholder string
instead of the real value it sent, breaking its round-trip.

This module implements the inbound post-processing step:

1. Parse the upstream JSON response.
2. Walk the provider-specific response fields that carry text content.
3. Replace every ``[TOKEN]`` whose bare token exists in *restoration_map* with
   its original value.
4. Return the rehydrated bytes as the agent-facing response.

``terminal_restore`` flag
-------------------------
Terminal-rendered output (e.g. streaming text printed in the user's shell) is
separate from what the agent code receives over HTTP.  By default this flag is
``False`` so that terminal output retains ``[CATEGORY_N]`` tokens — the user
can see that PII was detected and masked.  Setting ``terminal_restore=True``
causes the terminal text to also be rehydrated, useful for interactive sessions
where the user needs to verify round-trip fidelity.

Provider response field coverage
---------------------------------
Claude  (``/v1/messages``):
    ``content[*].text``  (type == "text")
    ``content[*].input`` (type == "tool_use"  — recursive string walk)

OpenAI  (``/v1/chat/completions``):
    ``choices[*].message.content``
    ``choices[*].message.tool_calls[*].function.arguments``  (JSON string)

Gemini  (``/v1beta/models/*:generateContent``):
    ``candidates[*].content.parts[*].text``
    ``candidates[*].content.parts[*].functionCall.args``  (recursive string walk)

Unknown / non-JSON responses are returned verbatim (no modification).

Usage::

    from pii_guard.response_rehydrator import ResponsePostProcessor

    processor = ResponsePostProcessor(terminal_restore=False)
    result = processor.process(
        response_body=upstream_bytes,
        restoration_map=engine.restoration_map,
        provider="claude",
    )
    # Send result.agent_body back to the calling agent.
    # result.terminal_text retains placeholders (terminal_restore=False).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Pattern for bracket-wrapped placeholder tokens: [CATEGORY_N] or [CATEGORY_N_BLOCKED]
# ─────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+(?:_BLOCKED)?\]")


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RehydrationResult:
    """
    Result of running an LLM response through the rehydration post-processor.

    Attributes
    ----------
    agent_body:
        Serialised JSON bytes to return to the calling agent.  All known
        ``[TOKEN]`` placeholders have been replaced with their original values.
    terminal_text:
        Human-readable text extracted from the response for terminal display.
        This is rehydrated **only** when ``terminal_restore=True``; otherwise
        it retains the ``[CATEGORY_N]`` tokens so the user can see what was
        masked.
    substitution_count:
        Total number of placeholder tokens replaced in ``agent_body``.
    was_rehydrated:
        ``True`` if at least one substitution was made in ``agent_body``.
    provider:
        Provider string as detected by the caller (``"claude"``, ``"openai"``,
        ``"gemini"``, or ``None``).
    """
    agent_body: bytes
    terminal_text: str
    substitution_count: int = 0
    was_rehydrated: bool = False
    provider: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core string-level rehydration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rehydrate_str(text: str, restoration_map: Dict[str, str]) -> Tuple[str, int]:
    """
    Replace all ``[TOKEN]`` placeholders in *text* with their original values.

    Replacement is performed longest-token-first to avoid partial-token
    shadowing (``EMAIL_10`` before ``EMAIL_1``).

    Returns
    -------
    (rehydrated_text, count_substitutions)
    """
    if not text or not restoration_map:
        return text, 0

    count = 0
    result = text
    for token in sorted(restoration_map, key=len, reverse=True):
        bracketed = f"[{token}]"
        if bracketed in result:
            original = restoration_map[token]
            result = result.replace(bracketed, original)
            count += 1
    return result, count


def _rehydrate_obj(obj: Any, restoration_map: Dict[str, str]) -> Tuple[Any, int]:
    """
    Recursively walk *obj* (dict / list / str) and rehydrate every string leaf.

    Returns the modified structure and the total substitution count.
    """
    if isinstance(obj, str):
        new_str, cnt = _rehydrate_str(obj, restoration_map)
        return new_str, cnt
    if isinstance(obj, dict):
        total = 0
        new_dict = {}
        for k, v in obj.items():
            new_v, cnt = _rehydrate_obj(v, restoration_map)
            new_dict[k] = new_v
            total += cnt
        return new_dict, total
    if isinstance(obj, list):
        total = 0
        new_list = []
        for item in obj:
            new_item, cnt = _rehydrate_obj(item, restoration_map)
            new_list.append(new_item)
            total += cnt
        return new_list, total
    # Non-string scalar (int, float, bool, None) — unchanged
    return obj, 0


def _extract_text_from_obj(obj: Any) -> str:
    """
    Recursively gather all string leaf values from *obj* and join with spaces.
    Used to produce a simplified human-readable terminal representation.
    """
    parts = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            parts.append(_extract_text_from_obj(v))
    elif isinstance(obj, list):
        for item in obj:
            parts.append(_extract_text_from_obj(item))
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Provider-specific response rehydrators
# ─────────────────────────────────────────────────────────────────────────────

def _rehydrate_claude_response(
    response: Dict[str, Any],
    restoration_map: Dict[str, str],
) -> Tuple[Dict[str, Any], int]:
    """
    Walk a Claude Messages API response and rehydrate all text-bearing fields.

    Fields covered
    --------------
    * ``content[*].text``        (type == "text")
    * ``content[*].input``       (type == "tool_use"  — recursive object walk)
    * ``content[*].thinking``    (type == "thinking" — text field)
    """
    import copy
    result = copy.deepcopy(response)
    total = 0

    content_blocks = result.get("content")
    if not isinstance(content_blocks, list):
        return result, total

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            new_text, cnt = _rehydrate_str(block.get("text", ""), restoration_map)
            block["text"] = new_text
            total += cnt

        elif block_type == "thinking":
            new_text, cnt = _rehydrate_str(block.get("thinking", ""), restoration_map)
            block["thinking"] = new_text
            total += cnt

        elif block_type == "tool_use":
            new_input, cnt = _rehydrate_obj(block.get("input"), restoration_map)
            block["input"] = new_input
            total += cnt

    return result, total


def _rehydrate_openai_response(
    response: Dict[str, Any],
    restoration_map: Dict[str, str],
) -> Tuple[Dict[str, Any], int]:
    """
    Walk an OpenAI chat-completions response and rehydrate all text-bearing fields.

    Fields covered
    --------------
    * ``choices[*].message.content``
    * ``choices[*].message.tool_calls[*].function.arguments``  (JSON string)
    * ``choices[*].delta.content``  (streaming chunk)
    """
    import copy
    result = copy.deepcopy(response)
    total = 0

    choices = result.get("choices")
    if not isinstance(choices, list):
        return result, total

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        # Non-streaming: message
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                new_content, cnt = _rehydrate_str(content, restoration_map)
                message["content"] = new_content
                total += cnt
            elif isinstance(content, list):
                # Multi-modal content array (OpenAI vision format)
                new_content, cnt = _rehydrate_obj(content, restoration_map)
                message["content"] = new_content
                total += cnt

            # tool_calls
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    func = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(func, dict):
                        args = func.get("arguments", "")
                        if isinstance(args, str):
                            new_args, cnt = _rehydrate_str(args, restoration_map)
                            func["arguments"] = new_args
                            total += cnt

        # Streaming: delta
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                new_content, cnt = _rehydrate_str(content, restoration_map)
                delta["content"] = new_content
                total += cnt

    return result, total


def _rehydrate_gemini_response(
    response: Dict[str, Any],
    restoration_map: Dict[str, str],
) -> Tuple[Dict[str, Any], int]:
    """
    Walk a Gemini generateContent response and rehydrate all text-bearing fields.

    Fields covered
    --------------
    * ``candidates[*].content.parts[*].text``
    * ``candidates[*].content.parts[*].functionCall.args``  (recursive walk)
    """
    import copy
    result = copy.deepcopy(response)
    total = 0

    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return result, total

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        content = candidate.get("content")
        if not isinstance(content, dict):
            continue

        parts = content.get("parts")
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue

            # Text part
            text = part.get("text")
            if isinstance(text, str):
                new_text, cnt = _rehydrate_str(text, restoration_map)
                part["text"] = new_text
                total += cnt

            # FunctionCall args (dict with string values)
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                args = fc.get("args")
                if args is not None:
                    new_args, cnt = _rehydrate_obj(args, restoration_map)
                    fc["args"] = new_args
                    total += cnt

    return result, total


# ─────────────────────────────────────────────────────────────────────────────
# Terminal text extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_terminal_text_claude(response: Dict[str, Any]) -> str:
    """Extract human-readable text from a Claude response for terminal display."""
    parts = []
    for block in response.get("content", []):
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
    return "\n".join(p for p in parts if p)


def _extract_terminal_text_openai(response: Dict[str, Any]) -> str:
    """Extract human-readable text from an OpenAI response for terminal display."""
    parts = []
    for choice in response.get("choices", []):
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("delta") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.append(_extract_text_from_obj(content))
    return "\n".join(p for p in parts if p)


def _extract_terminal_text_gemini(response: Dict[str, Any]) -> str:
    """Extract human-readable text from a Gemini response for terminal display."""
    parts = []
    for candidate in response.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        for part in (content.get("parts") or []) if isinstance(content, dict) else []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# ResponsePostProcessor — public API
# ─────────────────────────────────────────────────────────────────────────────

class ResponsePostProcessor:
    """
    Post-processor for inbound LLM responses.

    Applies placeholder rehydration to LLM response bodies before they are
    returned to the calling agent, while controlling terminal-output restoration
    via the ``terminal_restore`` flag.

    Parameters
    ----------
    terminal_restore:
        When ``False`` (default) terminal-rendered output retains the original
        ``[CATEGORY_N]`` placeholder tokens so the user can see that PII was
        masked.  When ``True`` the terminal text is also rehydrated to show
        real values (useful for round-trip verification sessions).

    Usage::

        proc = ResponsePostProcessor(terminal_restore=False)
        result = proc.process(
            response_body=raw_upstream_bytes,
            restoration_map=engine.restoration_map,
            provider="openai",
        )
        send_to_client(result.agent_body)           # rehydrated
        display_in_terminal(result.terminal_text)   # placeholders kept (default)
    """

    def __init__(self, terminal_restore: bool = False) -> None:
        self.terminal_restore = terminal_restore

    # ── Main entry-point ─────────────────────────────────────────────────────

    def process(
        self,
        response_body: bytes,
        restoration_map: Dict[str, str],
        provider: Optional[str],
    ) -> RehydrationResult:
        """
        Parse *response_body*, rehydrate text fields, and return
        a :class:`RehydrationResult`.

        Parameters
        ----------
        response_body:
            Raw bytes from the upstream LLM (expected to be JSON).
        restoration_map:
            ``{placeholder_token: original_value}`` snapshot from the session
            :class:`~pii_guard.session_map.SessionMap`.  An empty map causes
            the response to be returned verbatim.
        provider:
            ``"claude"``, ``"openai"``, ``"gemini"``, or ``None``.  When
            ``None`` or unrecognised the response is returned verbatim.

        Returns
        -------
        :class:`RehydrationResult`
        """
        # Short-circuit: empty restoration map or empty body → nothing to do
        if not restoration_map or not response_body:
            terminal_text = self._extract_terminal_text(
                response_body, provider, restoration_map
            )
            return RehydrationResult(
                agent_body=response_body,
                terminal_text=terminal_text,
                substitution_count=0,
                was_rehydrated=False,
                provider=provider,
            )

        # Attempt JSON parse; fall back to verbatim if non-JSON
        try:
            response_obj = json.loads(response_body)
        except (json.JSONDecodeError, ValueError):
            return RehydrationResult(
                agent_body=response_body,
                terminal_text="",
                substitution_count=0,
                was_rehydrated=False,
                provider=provider,
            )

        if not isinstance(response_obj, dict):
            return RehydrationResult(
                agent_body=response_body,
                terminal_text=str(response_obj),
                substitution_count=0,
                was_rehydrated=False,
                provider=provider,
            )

        # Provider-specific rehydration of the response JSON
        rehydrated_obj, count = self._rehydrate_by_provider(
            response_obj, restoration_map, provider
        )

        # Serialise rehydrated response → agent_body
        agent_body = json.dumps(rehydrated_obj, ensure_ascii=False).encode("utf-8")

        # Build terminal text
        if self.terminal_restore:
            # terminal_restore=True: terminal also sees rehydrated content
            terminal_text = self._extract_terminal_text_from_obj(
                rehydrated_obj, provider
            )
        else:
            # terminal_restore=False (default): terminal keeps placeholders
            terminal_text = self._extract_terminal_text_from_obj(
                response_obj, provider
            )

        return RehydrationResult(
            agent_body=agent_body,
            terminal_text=terminal_text,
            substitution_count=count,
            was_rehydrated=count > 0,
            provider=provider,
        )

    # ── Provider dispatch ─────────────────────────────────────────────────────

    def _rehydrate_by_provider(
        self,
        response_obj: Dict[str, Any],
        restoration_map: Dict[str, str],
        provider: Optional[str],
    ) -> Tuple[Dict[str, Any], int]:
        if provider == "claude":
            return _rehydrate_claude_response(response_obj, restoration_map)
        elif provider == "openai":
            return _rehydrate_openai_response(response_obj, restoration_map)
        elif provider == "gemini":
            return _rehydrate_gemini_response(response_obj, restoration_map)
        else:
            # Unknown provider: fall back to a full recursive string walk
            rehydrated, cnt = _rehydrate_obj(response_obj, restoration_map)
            return rehydrated, cnt

    def _extract_terminal_text_from_obj(
        self,
        response_obj: Dict[str, Any],
        provider: Optional[str],
    ) -> str:
        if provider == "claude":
            return _extract_terminal_text_claude(response_obj)
        elif provider == "openai":
            return _extract_terminal_text_openai(response_obj)
        elif provider == "gemini":
            return _extract_terminal_text_gemini(response_obj)
        else:
            return _extract_text_from_obj(response_obj)

    def _extract_terminal_text(
        self,
        response_body: bytes,
        provider: Optional[str],
        restoration_map: Dict[str, str],
    ) -> str:
        """Extract terminal text from raw bytes (used for short-circuit case)."""
        try:
            obj = json.loads(response_body)
            if isinstance(obj, dict):
                raw_text = self._extract_terminal_text_from_obj(obj, provider)
                if self.terminal_restore and restoration_map:
                    raw_text, _ = _rehydrate_str(raw_text, restoration_map)
                return raw_text
        except (json.JSONDecodeError, ValueError):
            pass
        return ""
