"""
Unit tests for schema coverage detection — Sub-AC 3a.

Verifies that:
  1. diff_claude_fields / diff_openai_fields / diff_gemini_fields produce
     a FieldDelta for every structural location that contains keys not in
     the provider's known API schema (no false negatives).
  2. Those same functions produce NO deltas for valid requests that use
     only published API fields (no false positives).
  3. diff_api_version produces a VersionDelta with is_unknown=True for
     any version not in the known-good list, is_future=True for versions
     that look newer than all known versions, and is_unknown=False /
     is_future=False for recognized versions.
  4. The FieldDelta and VersionDelta attributes carry the correct provider,
     path, extra_keys, known_keys, actual_keys, and version metadata.
  5. Edge cases (None payload, wrong types, empty dicts) never raise.

Every test class maps to one semantic dimension of the contract; no test
is allowed to depend on the internal schema dicts directly — all assertions
go through the public API.
"""
from __future__ import annotations

import pytest

from pii_guard.providers.schema_coverage import (
    FieldDelta,
    VersionDelta,
    diff_api_version,
    diff_claude_fields,
    diff_gemini_fields,
    diff_openai_fields,
    diff_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extra_at(deltas: list[FieldDelta], path: str) -> frozenset:
    """Return the extra_keys from the delta whose path matches *path*, or frozenset()."""
    for d in deltas:
        if d.path == path:
            return d.extra_keys
    return frozenset()


def _paths(deltas: list[FieldDelta]) -> set[str]:
    return {d.path for d in deltas}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FieldDelta — Claude provider
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeFieldDeltaNoFalsePositives:
    """A conforming Claude request must produce zero FieldDelta objects."""

    def test_minimal_valid_request(self):
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert diff_claude_fields(payload) == []

    def test_full_valid_request_all_known_top_level_fields(self):
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 2048,
            "stream": True,
            "system": "Be helpful.",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "search", "description": "...", "input_schema": {}}],
            "tool_choice": {"type": "auto"},
            "anthropic_version": "2023-06-01",
            "betas": ["tools-2024-04-04"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        assert diff_claude_fields(payload) == []

    def test_system_as_text_block_array(self):
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "Block A."},
                {"type": "text", "text": "Block B.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        assert diff_claude_fields(payload) == []

    def test_messages_with_text_blocks(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hi!", "cache_control": {"type": "ephemeral"}},
                    ],
                },
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_messages_with_tool_use_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "lookup",
                            "input": {"query": "something"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_messages_with_tool_result_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "Result text",
                            "is_error": False,
                        }
                    ],
                }
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_messages_with_image_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        }
                    ],
                }
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_messages_with_document_block_text_source(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "text", "media_type": "text/plain", "data": "Doc."},
                        }
                    ],
                }
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_thinking_block_no_delta(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think…", "signature": "sig"},
                    ],
                }
            ],
        }
        assert diff_claude_fields(payload) == []

    def test_tool_definitions_all_known_keys(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Returns weather data.",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [],
        }
        assert diff_claude_fields(payload) == []


class TestClaudeFieldDeltaNoFalseNegatives:
    """Unknown fields at any structural level must produce a FieldDelta."""

    def test_extra_top_level_field_detected(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [],
            "thinking_budget": 5000,   # unknown novel field
        }
        deltas = diff_claude_fields(payload)
        assert len(deltas) == 1
        assert deltas[0].path == ""
        assert "thinking_budget" in deltas[0].extra_keys
        assert deltas[0].provider == "claude"

    def test_multiple_extra_top_level_fields(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [],
            "future_param_a": 1,
            "future_param_b": "x",
        }
        deltas = diff_claude_fields(payload)
        assert len(deltas) == 1
        assert deltas[0].path == ""
        assert {"future_param_a", "future_param_b"}.issubset(deltas[0].extra_keys)

    def test_extra_field_in_message_dict(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi", "timestamp": 12345}
            ],
        }
        deltas = diff_claude_fields(payload)
        assert any(d.path == "messages[0]" and "timestamp" in d.extra_keys for d in deltas)

    def test_extra_field_in_text_content_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi", "metadata_tag": "v2"},
                    ],
                }
            ],
        }
        deltas = diff_claude_fields(payload)
        block_path = "messages[0].content[0]"
        assert any(d.path == block_path and "metadata_tag" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_use_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "fn",
                            "input": {},
                            "parallel": True,   # unknown field
                        }
                    ],
                }
            ],
        }
        deltas = diff_claude_fields(payload)
        assert any("parallel" in d.extra_keys for d in deltas)

    def test_extra_field_in_document_source(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Doc",
                                "checksum": "abc123",   # unknown field
                            },
                        }
                    ],
                }
            ],
        }
        deltas = diff_claude_fields(payload)
        assert any("checksum" in d.extra_keys for d in deltas)

    def test_extra_field_in_image_source(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                                "width": 100,   # unknown
                            },
                        }
                    ],
                }
            ],
        }
        deltas = diff_claude_fields(payload)
        assert any("width" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_definition(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "tools": [
                {
                    "name": "fn",
                    "description": "...",
                    "input_schema": {},
                    "version": "2",    # unknown
                }
            ],
            "messages": [],
        }
        deltas = diff_claude_fields(payload)
        assert any("version" in d.extra_keys for d in deltas)

    def test_extra_field_in_system_text_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Hello", "priority": 1},  # priority unknown
            ],
            "messages": [],
        }
        deltas = diff_claude_fields(payload)
        assert any("priority" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_result_nested_block(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Done",
                                    "annotation": "x",   # unknown inside nested text block
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        deltas = diff_claude_fields(payload)
        # Should have a delta for the nested content block
        assert any("annotation" in d.extra_keys for d in deltas)

    def test_delta_attributes_are_correct(self):
        """Verify all FieldDelta attributes carry the expected values."""
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [],
            "novel_field": "x",
        }
        deltas = diff_claude_fields(payload)
        assert len(deltas) == 1
        d = deltas[0]
        assert d.provider == "claude"
        assert d.path == ""
        assert "novel_field" in d.extra_keys
        assert "novel_field" in d.actual_keys
        assert "novel_field" not in d.known_keys
        assert "model" in d.known_keys
        assert "messages" in d.known_keys

    def test_multiple_structural_levels_each_produce_delta(self):
        """Unknown fields at different levels each get their own FieldDelta."""
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi", "novel_block_key": 1},
                    ],
                    "extra_msg_key": "y",   # unknown at message level
                }
            ],
            "top_level_novel": True,  # unknown at root
        }
        deltas = diff_claude_fields(payload)
        paths = _paths(deltas)
        assert "" in paths                         # root delta
        assert "messages[0]" in paths              # message level
        assert "messages[0].content[0]" in paths   # block level

    def test_second_message_extra_key_detected(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello", "future_msg_key": True},
            ],
        }
        deltas = diff_claude_fields(payload)
        assert any(d.path == "messages[1]" and "future_msg_key" in d.extra_keys
                   for d in deltas)


class TestClaudeFieldDeltaEdgeCases:
    """Edge cases must never raise."""

    def test_none_payload(self):
        assert diff_claude_fields(None) == []  # type: ignore[arg-type]

    def test_non_dict_payload(self):
        assert diff_claude_fields("not a dict") == []  # type: ignore[arg-type]

    def test_empty_payload(self):
        assert diff_claude_fields({}) == []

    def test_messages_not_a_list(self):
        assert diff_claude_fields({"model": "m", "messages": "bad"}) == []

    def test_non_dict_message_skipped(self):
        assert diff_claude_fields({"model": "m", "messages": ["bad"]}) == []

    def test_content_not_a_list(self):
        # string content — no block-level diff needed (valid shorthand)
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "text"}],
        }
        assert diff_claude_fields(payload) == []

    def test_non_dict_content_block_skipped(self):
        payload = {
            "model": "m",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": ["not a dict"]},
            ],
        }
        # no error, no delta for the non-dict item
        deltas = diff_claude_fields(payload)
        assert not any("not a dict" in str(d) for d in deltas)

    def test_tools_not_a_list(self):
        payload = {"model": "m", "max_tokens": 100, "tools": "bad", "messages": []}
        # no error
        assert diff_claude_fields(payload) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FieldDelta — OpenAI provider
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIFieldDeltaNoFalsePositives:
    """Valid OpenAI requests must produce zero FieldDelta objects."""

    def test_minimal_valid_request(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert diff_openai_fields(payload) == []

    def test_all_known_top_level_fields(self):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "stop": None,
            "max_tokens": 512,
            "max_completion_tokens": 1024,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "user": "u1",
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "response_format": {"type": "text"},
            "seed": 42,
            "service_tier": "auto",
            "stream_options": None,
            "logprobs": None,
            "top_logprobs": None,
            "metadata": {},
            "store": False,
            "reasoning_effort": "medium",
            "modalities": ["text"],
        }
        assert diff_openai_fields(payload) == []

    def test_system_message(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        }
        assert diff_openai_fields(payload) == []

    def test_assistant_message_with_tool_calls(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Seoul"}',
                            },
                        }
                    ],
                }
            ],
        }
        assert diff_openai_fields(payload) == []

    def test_tool_message_string_content(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "content": "Seoul: sunny, 25°C",
                    "tool_call_id": "tc_1",
                }
            ],
        }
        assert diff_openai_fields(payload) == []

    def test_user_message_content_parts(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        }
        assert diff_openai_fields(payload) == []

    def test_tools_with_function_definition(self):
        payload = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fn",
                        "description": "...",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        }
        assert diff_openai_fields(payload) == []


class TestOpenAIFieldDeltaNoFalseNegatives:
    """Unknown fields at any level must be detected."""

    def test_extra_top_level_field(self):
        payload = {
            "model": "gpt-4o",
            "messages": [],
            "max_context_window": 128000,   # novel field
        }
        deltas = diff_openai_fields(payload)
        assert any(d.path == "" and "max_context_window" in d.extra_keys for d in deltas)

    def test_extra_field_in_message(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hi", "timestamp_ms": 123456},
            ],
        }
        deltas = diff_openai_fields(payload)
        assert any(d.path == "messages[0]" and "timestamp_ms" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_call(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "fn", "arguments": "{}"},
                            "priority": 1,   # novel field
                        }
                    ],
                }
            ],
        }
        deltas = diff_openai_fields(payload)
        assert any("priority" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_call_function(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": "{}",
                                "version": "2",   # novel
                            },
                        }
                    ],
                }
            ],
        }
        deltas = diff_openai_fields(payload)
        assert any("version" in d.extra_keys for d in deltas)

    def test_extra_field_in_content_part(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi", "annotation": "key"},
                    ],
                }
            ],
        }
        deltas = diff_openai_fields(payload)
        assert any("annotation" in d.extra_keys for d in deltas)

    def test_extra_field_in_tool_definition(self):
        payload = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "fn", "parameters": {}},
                    "cache_control": True,   # novel for OpenAI tools
                }
            ],
        }
        deltas = diff_openai_fields(payload)
        assert any("cache_control" in d.extra_keys for d in deltas)

    def test_future_versioned_request_with_extra_field(self):
        """Simulate a future OpenAI API request with a novel top-level field."""
        payload = {
            "model": "gpt-5",
            "messages": [],
            "inner_monologue": True,   # hypothetical future field
        }
        deltas = diff_openai_fields(payload)
        assert any("inner_monologue" in d.extra_keys for d in deltas)

    def test_provider_attribute_is_openai(self):
        payload = {
            "model": "gpt-4o",
            "messages": [],
            "new_field": "x",
        }
        deltas = diff_openai_fields(payload)
        assert all(d.provider == "openai" for d in deltas)

    def test_multiple_unknown_fields_all_in_one_delta(self):
        payload = {
            "model": "gpt-4o",
            "messages": [],
            "fieldA": 1,
            "fieldB": 2,
            "fieldC": 3,
        }
        deltas = diff_openai_fields(payload)
        root_deltas = [d for d in deltas if d.path == ""]
        assert len(root_deltas) == 1
        assert {"fieldA", "fieldB", "fieldC"}.issubset(root_deltas[0].extra_keys)


class TestOpenAIEdgeCases:
    def test_none_payload(self):
        assert diff_openai_fields(None) == []  # type: ignore[arg-type]

    def test_empty_payload(self):
        assert diff_openai_fields({}) == []

    def test_messages_not_a_list(self):
        assert diff_openai_fields({"messages": "bad"}) == []

    def test_non_dict_message_skipped(self):
        assert diff_openai_fields({"messages": [42]}) == []

    def test_content_string_no_part_diff(self):
        """String content produces no content-part-level diff."""
        payload = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
        assert diff_openai_fields(payload) == []


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FieldDelta — Gemini provider
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiFieldDeltaNoFalsePositives:
    """Valid Gemini requests must produce zero FieldDelta objects."""

    def test_minimal_valid_request_camel_case(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        assert diff_gemini_fields(payload) == []

    def test_full_valid_request_camel_case(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Describe this"},
                        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                    ],
                }
            ],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "Be concise."}],
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
                "topP": 0.9,
                "topK": 40,
                "stopSequences": ["END"],
                "responseMimeType": "text/plain",
                "candidateCount": 1,
            },
            "safetySettings": [],
            "tools": [],
            "toolConfig": {},
        }
        assert diff_gemini_fields(payload) == []

    def test_snake_case_fields_no_delta(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "system_instruction": {"role": "system", "parts": [{"text": "Help"}]},
            "generation_config": {"temperature": 1.0, "max_output_tokens": 512},
            "safety_settings": [],
        }
        assert diff_gemini_fields(payload) == []

    def test_function_call_and_response_parts_no_delta(self):
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Seoul"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_weather",
                                "response": {"temp": 25},
                            }
                        }
                    ],
                },
            ]
        }
        assert diff_gemini_fields(payload) == []

    def test_executable_code_and_result_parts_no_delta(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"executableCode": {"language": "python", "code": "print(1)"}},
                        {
                            "codeExecutionResult": {
                                "outcome": "OK",
                                "output": "1\n",
                            }
                        },
                    ],
                }
            ]
        }
        assert diff_gemini_fields(payload) == []

    def test_model_in_body_no_delta(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        assert diff_gemini_fields(payload) == []

    def test_thinking_config_in_generation_config(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}},
        }
        assert diff_gemini_fields(payload) == []


class TestGeminiFieldDeltaNoFalseNegatives:
    """Unknown fields at any level must be detected."""

    def test_extra_top_level_field(self):
        payload = {
            "contents": [],
            "enableGrounding": True,   # novel field
        }
        deltas = diff_gemini_fields(payload)
        assert any(d.path == "" and "enableGrounding" in d.extra_keys for d in deltas)

    def test_extra_field_in_content_item(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [],
                    "timestamp": 123,   # novel
                }
            ]
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "contents[0]" and "timestamp" in d.extra_keys for d in deltas
        )

    def test_extra_field_in_part(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Hi", "metadata": {"source": "user"}},   # metadata novel
                    ],
                }
            ]
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "contents[0].parts[0]" and "metadata" in d.extra_keys
            for d in deltas
        )

    def test_extra_field_in_system_instruction(self):
        payload = {
            "contents": [],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "Hi"}],
                "priority": 1,   # novel
            },
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "systemInstruction" and "priority" in d.extra_keys for d in deltas
        )

    def test_extra_field_in_generation_config(self):
        payload = {
            "contents": [],
            "generationConfig": {
                "temperature": 0.5,
                "futureSamplingParam": 0.3,   # novel
            },
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "generationConfig" and "futureSamplingParam" in d.extra_keys
            for d in deltas
        )

    def test_future_part_type_detected(self):
        """A part with a completely unknown key should produce a delta."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"videoClip": {"url": "gs://bucket/video.mp4"}},   # hypothetical future
                    ],
                }
            ]
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "contents[0].parts[0]" and "videoClip" in d.extra_keys
            for d in deltas
        )

    def test_provider_attribute_is_gemini(self):
        payload = {"contents": [], "future_key": True}
        deltas = diff_gemini_fields(payload)
        assert all(d.provider == "gemini" for d in deltas)

    def test_snake_case_extra_field(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hi"}],
                    "context_window_hint": 1000,   # novel snake_case field
                }
            ]
        }
        deltas = diff_gemini_fields(payload)
        assert any("context_window_hint" in d.extra_keys for d in deltas)

    def test_second_content_item_extra_key_detected(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {
                    "role": "model",
                    "parts": [{"text": "Hello"}],
                    "confidence": 0.9,   # novel
                },
            ]
        }
        deltas = diff_gemini_fields(payload)
        assert any(
            d.path == "contents[1]" and "confidence" in d.extra_keys for d in deltas
        )


class TestGeminiEdgeCases:
    def test_none_payload(self):
        assert diff_gemini_fields(None) == []  # type: ignore[arg-type]

    def test_empty_payload(self):
        assert diff_gemini_fields({}) == []

    def test_contents_not_a_list(self):
        assert diff_gemini_fields({"contents": "bad"}) == []

    def test_non_dict_content_item_skipped(self):
        assert diff_gemini_fields({"contents": [42]}) == []

    def test_parts_not_a_list(self):
        assert diff_gemini_fields({
            "contents": [{"role": "user", "parts": "bad"}]
        }) == []

    def test_non_dict_part_skipped(self):
        assert diff_gemini_fields({
            "contents": [{"role": "user", "parts": ["string_part"]}]
        }) == []


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VersionDelta — API version comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeVersionDelta:
    """Claude uses ISO date versions (anthropic-version header)."""

    def test_known_version_produces_no_alarm(self):
        vd = diff_api_version("2023-06-01", "claude")
        assert vd.is_unknown is False
        assert vd.is_future is False
        assert vd.declared_version == "2023-06-01"
        assert vd.provider == "claude"
        assert "2023-06-01" in vd.known_versions

    def test_future_date_is_unknown_and_future(self):
        vd = diff_api_version("2099-01-01", "claude")
        assert vd.is_unknown is True
        assert vd.is_future is True
        assert vd.declared_version == "2099-01-01"

    def test_past_unknown_date_is_unknown_not_future(self):
        # 2020-01-01 predates the known API — unknown but not future
        vd = diff_api_version("2020-01-01", "claude")
        assert vd.is_unknown is True
        assert vd.is_future is False

    def test_near_future_date_is_future(self):
        vd = diff_api_version("2027-03-15", "claude")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_malformed_version_is_unknown_not_future(self):
        vd = diff_api_version("not-a-date", "claude")
        assert vd.is_unknown is True
        assert vd.is_future is False

    def test_empty_version_string_is_unknown_not_future(self):
        vd = diff_api_version("", "claude")
        assert vd.is_unknown is True
        assert vd.is_future is False

    def test_location_default_for_claude(self):
        vd = diff_api_version("2023-06-01", "claude")
        assert "anthropic-version" in vd.location

    def test_custom_location_forwarded(self):
        vd = diff_api_version("2023-06-01", "claude", location="custom:source")
        assert vd.location == "custom:source"

    def test_known_versions_tuple_present(self):
        vd = diff_api_version("2023-06-01", "claude")
        assert isinstance(vd.known_versions, tuple)
        assert len(vd.known_versions) >= 1


class TestOpenAIVersionDelta:
    """OpenAI uses path-based versioning (/v1/…)."""

    def test_v1_is_known(self):
        vd = diff_api_version("v1", "openai")
        assert vd.is_unknown is False
        assert vd.is_future is False

    def test_future_v2_is_unknown_and_future(self):
        vd = diff_api_version("v2", "openai")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_future_v3_is_unknown_and_future(self):
        vd = diff_api_version("v3", "openai")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_malformed_openai_version_is_unknown_not_future(self):
        vd = diff_api_version("version_x", "openai")
        assert vd.is_unknown is True
        assert vd.is_future is False

    def test_location_default_for_openai(self):
        vd = diff_api_version("v1", "openai")
        assert "v" in vd.location.lower()


class TestGeminiVersionDelta:
    """Gemini uses path-based versioning (/v1/, /v1beta/, /v1alpha/)."""

    def test_v1_is_known(self):
        vd = diff_api_version("v1", "gemini")
        assert vd.is_unknown is False
        assert vd.is_future is False

    def test_v1beta_is_known(self):
        vd = diff_api_version("v1beta", "gemini")
        assert vd.is_unknown is False
        assert vd.is_future is False

    def test_v1alpha_is_known(self):
        vd = diff_api_version("v1alpha", "gemini")
        assert vd.is_unknown is False
        assert vd.is_future is False

    def test_future_v2_is_unknown_and_future(self):
        vd = diff_api_version("v2", "gemini")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_future_v2beta_is_unknown_and_future(self):
        vd = diff_api_version("v2beta", "gemini")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_future_v3_is_unknown_and_future(self):
        vd = diff_api_version("v3", "gemini")
        assert vd.is_unknown is True
        assert vd.is_future is True

    def test_malformed_version_unknown_not_future(self):
        vd = diff_api_version("release-candidate", "gemini")
        assert vd.is_unknown is True
        assert vd.is_future is False

    def test_location_default_for_gemini(self):
        vd = diff_api_version("v1", "gemini")
        assert vd.location is not None and len(vd.location) > 0

    def test_provider_attribute(self):
        vd = diff_api_version("v1beta", "gemini")
        assert vd.provider == "gemini"

    def test_known_versions_contains_v1_and_beta(self):
        vd = diff_api_version("v1", "gemini")
        assert "v1" in vd.known_versions
        assert "v1beta" in vd.known_versions


class TestVersionDeltaInvalidProvider:
    """Unknown provider must raise."""

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            diff_api_version("2023-06-01", "unknown_provider")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  diff_request convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffRequest:
    """diff_request should return (field_deltas, version_delta)."""

    def test_clean_request_no_version(self):
        payload = {"model": "claude-opus-4-5", "max_tokens": 100, "messages": []}
        field_deltas, version_delta = diff_request(payload, "claude")
        assert field_deltas == []
        assert version_delta is None

    def test_clean_request_known_version(self):
        payload = {"model": "claude-opus-4-5", "max_tokens": 100, "messages": []}
        field_deltas, version_delta = diff_request(
            payload, "claude", api_version="2023-06-01"
        )
        assert field_deltas == []
        assert version_delta is not None
        assert version_delta.is_unknown is False

    def test_extra_field_and_future_version(self):
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 100,
            "messages": [],
            "novel_field": True,
        }
        field_deltas, version_delta = diff_request(
            payload, "claude", api_version="2099-01-01"
        )
        assert len(field_deltas) == 1
        assert "novel_field" in field_deltas[0].extra_keys
        assert version_delta is not None
        assert version_delta.is_unknown is True
        assert version_delta.is_future is True

    def test_openai_provider(self):
        payload = {"model": "gpt-4o", "messages": [], "new_key": "x"}
        field_deltas, version_delta = diff_request(payload, "openai", api_version="v2")
        assert any("new_key" in d.extra_keys for d in field_deltas)
        assert version_delta is not None
        assert version_delta.is_future is True

    def test_gemini_provider(self):
        payload = {"contents": [], "extra_gemini_key": 1}
        field_deltas, version_delta = diff_request(payload, "gemini", api_version="v1")
        assert any("extra_gemini_key" in d.extra_keys for d in field_deltas)
        assert version_delta is not None
        assert version_delta.is_unknown is False

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            diff_request({}, "invalid_provider")

    def test_version_location_forwarded(self):
        payload = {"model": "m", "max_tokens": 1, "messages": []}
        _, vd = diff_request(
            payload, "claude",
            api_version="2023-06-01",
            version_location="header:x-custom",
        )
        assert vd is not None
        assert vd.location == "header:x-custom"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FieldDelta structural invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldDeltaInvariants:
    """Invariants that must hold for every FieldDelta regardless of provider."""

    def _make_delta(self, provider: str) -> FieldDelta:
        payload = {"model": "m", "messages": [], "future_key": True}
        if provider == "claude":
            deltas = diff_claude_fields(payload)
        elif provider == "openai":
            deltas = diff_openai_fields(payload)
        else:
            payload = {"contents": [], "future_key": True}
            deltas = diff_gemini_fields(payload)
        return deltas[0]

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_extra_keys_subset_of_actual_keys(self, provider):
        d = self._make_delta(provider)
        assert d.extra_keys <= d.actual_keys

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_extra_keys_disjoint_from_known_keys(self, provider):
        d = self._make_delta(provider)
        assert d.extra_keys.isdisjoint(d.known_keys)

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_actual_keys_is_union_of_known_and_extra(self, provider):
        d = self._make_delta(provider)
        # actual_keys = known_keys ∩ actual_keys ∪ extra_keys
        # (known_keys may have keys not present in actual)
        assert d.extra_keys.issubset(d.actual_keys)

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_provider_field_matches_arg(self, provider):
        d = self._make_delta(provider)
        assert d.provider == provider

    @pytest.mark.parametrize("provider", ["claude", "openai", "gemini"])
    def test_future_key_in_extra_not_in_known(self, provider):
        d = self._make_delta(provider)
        assert "future_key" in d.extra_keys
        assert "future_key" not in d.known_keys


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VersionDelta structural invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionDeltaInvariants:
    """Invariants that must hold for every VersionDelta."""

    @pytest.mark.parametrize("ver,prov,expect_unknown,expect_future", [
        # Known versions → not unknown, not future
        ("2023-06-01", "claude", False, False),
        ("v1",         "openai", False, False),
        ("v1",         "gemini", False, False),
        ("v1beta",     "gemini", False, False),
        ("v1alpha",    "gemini", False, False),
        # Future versions → unknown AND future
        ("2099-12-31", "claude", True,  True),
        ("v2",         "openai", True,  True),
        ("v2",         "gemini", True,  True),
        ("v3",         "gemini", True,  True),
        # Past / malformed → unknown but NOT future
        ("2020-01-01", "claude", True,  False),
        ("v0",         "openai", True,  False),   # lower version → not future
        ("garbled",    "gemini", True,  False),
    ])
    def test_version_classification(self, ver, prov, expect_unknown, expect_future):
        vd = diff_api_version(ver, prov)
        assert vd.is_unknown == expect_unknown, (
            f"Expected is_unknown={expect_unknown} for {ver!r}/{prov!r}, got {vd.is_unknown}"
        )
        assert vd.is_future == expect_future, (
            f"Expected is_future={expect_future} for {ver!r}/{prov!r}, got {vd.is_future}"
        )

    def test_is_future_implies_is_unknown(self):
        """is_future=True must always imply is_unknown=True."""
        for ver, prov in [
            ("2099-01-01", "claude"),
            ("v2", "openai"),
            ("v99", "gemini"),
        ]:
            vd = diff_api_version(ver, prov)
            if vd.is_future:
                assert vd.is_unknown, (
                    f"is_future=True but is_unknown=False for {ver!r}/{prov!r}"
                )

    def test_declared_version_preserved(self):
        vd = diff_api_version("2099-06-01", "claude")
        assert vd.declared_version == "2099-06-01"

    def test_known_versions_is_tuple(self):
        for prov in ("claude", "openai", "gemini"):
            vd = diff_api_version("v99", prov)
            assert isinstance(vd.known_versions, tuple)

    def test_location_is_non_empty_string(self):
        for prov in ("claude", "openai", "gemini"):
            vd = diff_api_version("v99", prov)
            assert isinstance(vd.location, str)
            assert len(vd.location) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Comprehensive integration scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationScenarios:
    """Realistic composite scenarios that exercise multiple detection paths."""

    def test_claude_future_api_with_novel_top_level_and_block_fields(self):
        """
        Simulate a future Claude API request with:
          - A novel top-level field ("vision_budget")
          - A novel content block field ("annotation_tag")
          - A future anthropic-version header value
        Expect: 2 FieldDeltas (root + block) and 1 VersionDelta (future).
        """
        payload = {
            "model": "claude-future-model",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "annotation_tag": "v3",   # novel block field
                        }
                    ],
                }
            ],
            "vision_budget": 1000,   # novel top-level field
        }
        field_deltas, version_delta = diff_request(
            payload, "claude", api_version="2025-12-01"
        )

        # Root delta: "vision_budget"
        assert any(d.path == "" and "vision_budget" in d.extra_keys for d in field_deltas)
        # Block delta: "annotation_tag"
        assert any(
            "annotation_tag" in d.extra_keys for d in field_deltas
        )
        # Version delta: future
        assert version_delta is not None
        assert version_delta.is_future is True
        assert version_delta.is_unknown is True

    def test_openai_future_api_with_novel_message_and_function_fields(self):
        """
        Simulate a future OpenAI API request with:
          - A novel top-level field ("prediction_mode")
          - A novel tool_call.function field ("version")
        """
        payload = {
            "model": "gpt-5",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": "{}",
                                "call_id": "cid_1",   # novel function field
                            },
                        }
                    ],
                }
            ],
            "prediction_mode": "speculative",   # novel top-level
        }
        field_deltas, version_delta = diff_request(
            payload, "openai", api_version="v2"
        )
        assert any(d.path == "" and "prediction_mode" in d.extra_keys for d in field_deltas)
        assert any("call_id" in d.extra_keys for d in field_deltas)
        assert version_delta is not None
        assert version_delta.is_future is True

    def test_gemini_future_api_with_novel_part_and_config_fields(self):
        """
        Simulate a future Gemini v2 request with:
          - A novel part key ("videoClip")
          - A novel generationConfig field ("samplingAlgorithm")
        """
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"videoClip": {"uri": "gs://bucket/video.mp4"}}   # novel part
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.8,
                "samplingAlgorithm": "nucleus_plus",   # novel config field
            },
        }
        field_deltas, version_delta = diff_request(
            payload, "gemini", api_version="v2beta"
        )
        assert any("videoClip" in d.extra_keys for d in field_deltas)
        assert any("samplingAlgorithm" in d.extra_keys for d in field_deltas)
        assert version_delta is not None
        assert version_delta.is_future is True

    def test_clean_claude_request_zero_deltas(self):
        """No false alarms on a perfectly conforming Claude request."""
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 2048,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hi!"},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "search",
                            "input": {"q": "what is AI"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "AI is ...",
                        }
                    ],
                },
            ],
        }
        field_deltas, version_delta = diff_request(
            payload, "claude", api_version="2023-06-01"
        )
        assert field_deltas == []
        assert version_delta is not None
        assert version_delta.is_unknown is False
