"""
Integration tests for Gemini wire format traversal and payload reconstruction.
Sub-AC 2c — every Gemini-specific field type scrubbed with no user configuration.

Tests assert:
  1.  systemInstruction (dict form with parts) — text parts are scrubbed.
  2.  systemInstruction (string shorthand) — scanned directly.
  3.  system_instruction (snake_case alias) — treated identically.
  4.  contents[*].parts[*].text — regular user/model text parts are scrubbed.
  5.  contents[*].parts[*].functionCall.args — all string leaf values recursively scanned.
  6.  contents[*].parts[*].functionResponse.response — all string leaf values recursively scanned.
  7.  contents[*].parts[*].executableCode.code — source code text is scanned.
  8.  contents[*].parts[*].codeExecutionResult.output — code output is scanned.
  9.  inlineData / inline_data parts — unscannable → coverage gap + block by default.
  10. fileData / file_data parts — unscannable → coverage gap + block by default.
  11. snake_case field aliases (function_call, function_response, inline_data,
      file_data, executable_code, code_execution_result) work identically.
  12. Unknown / unrecognized part keys raise a coverage alarm (unknown_fields).
  13. The sanitized payload is structurally valid (roles, names, IDs preserved).
  14. Cross-field placeholder consistency: same real value → same placeholder.
  15. No original PII/secret text survives in the sanitized payload.
  16. No user configuration required — plain Engine() provides full protection.
  17. Multi-turn / multi-content-item payloads are all scrubbed.
  18. Numeric / boolean / null values in functionCall.args pass through unchanged.
  19. Deeply nested JSON objects in args/response are fully walked.
  20. The original payload dict is never mutated.
  21. Edge cases: empty payload, missing contents, empty parts list, empty text.
  22. unscannable_action='warn_allow' does not block but still records coverage gap.
  23. unknown_field_action='warn_allow' does not block but still records unknown field.
  24. Mixed payload: PII in every Gemini-specific field type simultaneously sanitized.
"""
from __future__ import annotations

import copy
import re

import pytest

from pii_guard import Engine
from pii_guard.providers.gemini import (
    FieldScanEvent,
    GeminiRequestScrubResult,
    ScanField,
    scrub_gemini_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fresh_engine() -> Engine:
    """Return a new Engine with no pre-existing session state."""
    return Engine()


def _email_placeholders(text: str) -> list[str]:
    return re.findall(r"\[EMAIL_\d+\]", text)


def _phone_placeholders(text: str) -> list[str]:
    return re.findall(r"\[PHONE_\d+\]", text)


# ─────────────────────────────────────────────────────────────────────────────
# 1. systemInstruction — dict form with parts array
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemInstructionDict:
    """systemInstruction is a Content object (dict) with a parts list."""

    def test_email_in_system_instruction_text_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": "Always notify admin@corp.io of all issues."}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Go"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["systemInstruction"]["parts"][0]["text"]
        assert "admin@corp.io" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_api_key_in_system_instruction_blocks(self):
        key = "sk-ant-api03-" + "A" * 50
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": f"Use key={key} for authentication."}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Run"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["systemInstruction"]["parts"][0]["text"]
        assert key not in text
        assert result.should_block

    def test_clean_system_instruction_passes_through(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": "You are a helpful assistant."}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["systemInstruction"]["parts"][0]["text"]
        assert text == "You are a helpful assistant."
        assert not result.should_block

    def test_multiple_parts_in_system_instruction_all_scrubbed(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [
                    {"text": "Contact 이영희 at manager@example.io for help."},
                    {"text": "Phone: 010-9999-1234"},
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": "Ok"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        parts = result.sanitized_payload["systemInstruction"]["parts"]
        assert "manager@example.io" not in parts[0]["text"]
        assert "010-9999-1234" not in parts[1]["text"]
        assert not result.should_block

    def test_system_instruction_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": "Contact contact@corp.io"}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        si_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.SYSTEM_INSTRUCTION]
        assert si_evts, "Expected at least one SYSTEM_INSTRUCTION scan event"

    def test_system_instruction_role_field_preserved(self):
        """Role field inside systemInstruction dict is not touched."""
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "Be helpful. Contact admin@x.com"}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        si = result.sanitized_payload["systemInstruction"]
        assert si["role"] == "system"
        assert "admin@x.com" not in si["parts"][0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. systemInstruction — plain string shorthand
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemInstructionString:
    """systemInstruction is a plain string (shorthand, some SDK versions)."""

    def test_email_in_system_instruction_string_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": "Notify ops@example.com of all failures.",
            "contents": [{"role": "user", "parts": [{"text": "Go"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        si = result.sanitized_payload["systemInstruction"]
        assert "ops@example.com" not in si
        assert "[EMAIL_" in si
        assert not result.should_block

    def test_secret_in_system_instruction_string_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": f"Use AWS key: {key}",
            "contents": [{"role": "user", "parts": [{"text": "Go"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert key not in result.sanitized_payload["systemInstruction"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 3. system_instruction — snake_case alias
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemInstructionSnakeCase:
    """system_instruction (snake_case) is treated identically to systemInstruction."""

    def test_snake_case_system_instruction_dict_scrubbed(self):
        payload = {
            "model": "gemini-2.0-flash",
            "system_instruction": {
                "parts": [{"text": "Contact admin@snake.io for issues."}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["system_instruction"]["parts"][0]["text"]
        assert "admin@snake.io" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_snake_case_system_instruction_string_scrubbed(self):
        payload = {
            "model": "gemini-2.0-flash",
            "system_instruction": "Alert dev@snake.io always.",
            "contents": [{"role": "user", "parts": [{"text": "Run"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert "dev@snake.io" not in result.sanitized_payload["system_instruction"]
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 4. contents[*].parts[*].text — user and model text parts
# ─────────────────────────────────────────────────────────────────────────────

class TestContentsTextParts:
    """Regular text parts in contents (user and model roles) are scrubbed."""

    def test_email_in_user_text_part_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Send invoice to billing@company.com please."}],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert "billing@company.com" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_secret_in_user_text_part_blocks(self):
        key = "sk-ant-api03-" + "B" * 50
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Key is {key}"}],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert key not in text
        assert result.should_block

    def test_email_in_model_text_part_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Who do I contact?"}]},
                {
                    "role": "model",
                    "parts": [{"text": "Reach out to contact@org.com."}],
                },
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        text = result.sanitized_payload["contents"][1]["parts"][0]["text"]
        assert "contact@org.com" not in text
        assert "[EMAIL_" in text
        assert not result.should_block

    def test_multiple_text_parts_all_scrubbed(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "My email: user@example.com"},
                        {"text": "My phone: 010-1234-5678"},
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        parts = result.sanitized_payload["contents"][0]["parts"]
        assert "user@example.com" not in parts[0]["text"]
        assert "010-1234-5678" not in parts[1]["text"]
        assert not result.should_block

    def test_message_text_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello user@test.com"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        txt_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.MESSAGE_TEXT]
        assert txt_evts

    def test_rrn_in_user_text_blocks(self):
        """Korean RRN in user text → blocked."""
        rrn = "900505-1234564"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": f"My RRN: {rrn}"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert rrn not in result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert result.should_block

    def test_card_in_user_text_blocks(self):
        """Luhn-valid card number in user text → blocked."""
        card = "4532015112830366"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": f"Card: {card}"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert card not in result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 5. functionCall.args — function call arguments (JSON object)
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionCallArgs:
    """functionCall.args is a JSON object; all string leaf values are scanned."""

    def test_email_in_function_call_args_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_email",
                                "args": {
                                    "to": "recipient@domain.com",
                                    "subject": "Hello",
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]
        assert "recipient@domain.com" not in fc["args"]["to"]
        assert "[EMAIL_" in fc["args"]["to"]
        assert fc["args"]["subject"] == "Hello"  # non-PII unchanged
        assert not result.should_block

    def test_api_key_in_function_call_args_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "call_aws",
                                "args": {"access_key": key, "action": "list"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]
        assert key not in fc["args"]["access_key"]
        assert result.should_block

    def test_nested_dict_in_function_call_args_scanned(self):
        """Nested dicts inside functionCall.args are recursively scanned."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "store_contact",
                                "args": {
                                    "contact": {
                                        "email": "deep@nested.com",
                                        "phone": "010-1111-2222",
                                        "metadata": {"source": "CRM"},
                                    }
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        contact = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]["contact"]
        )
        assert "deep@nested.com" not in contact["email"]
        assert "010-1111-2222" not in contact["phone"]
        assert contact["metadata"]["source"] == "CRM"  # non-PII unchanged
        assert not result.should_block

    def test_array_values_in_function_call_args_scanned(self):
        """Array string values inside functionCall.args are recursively scanned."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "notify_users",
                                "args": {"emails": ["a@b.com", "c@d.io", "safe_value"]},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        emails = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]["emails"]
        )
        assert "a@b.com" not in emails
        assert "c@d.io" not in emails
        assert "safe_value" in emails  # non-PII unchanged
        assert not result.should_block

    def test_numeric_values_in_function_call_args_unchanged(self):
        """Numbers, booleans, and null in args pass through untouched."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "set_params",
                                "args": {
                                    "count": 42,
                                    "enabled": True,
                                    "ratio": 3.14,
                                    "nothing": None,
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        args = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]
        )
        assert args["count"] == 42
        assert args["enabled"] is True
        assert args["ratio"] == 3.14
        assert args["nothing"] is None
        assert not result.should_block

    def test_deeply_nested_function_call_args_scanned(self):
        """3-level nesting inside functionCall.args is fully walked."""
        key = "ghp_" + "G" * 40
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "configure",
                                "args": {
                                    "level1": {"level2": {"level3": {"token": key}}}
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        token = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]["level1"]["level2"]["level3"]["token"]
        )
        assert key not in token
        assert result.should_block

    def test_function_call_args_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "greet",
                                "args": {"message": "Hello contact@domain.com"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.FUNCTION_CALL_ARGS]
        assert fc_evts

    def test_function_call_name_preserved(self):
        """functionCall.name is a structural field and must not be modified."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "my_function_name",
                                "args": {"greeting": "hello world@test.com"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]
        assert fc["name"] == "my_function_name"


# ─────────────────────────────────────────────────────────────────────────────
# 6. functionResponse.response — function response values
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionResponseValues:
    """functionResponse.response is a JSON object; all string leaf values are scanned."""

    def test_email_in_function_response_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup_user",
                                "response": {
                                    "email": "user@response.com",
                                    "status": "active",
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fr = result.sanitized_payload["contents"][0]["parts"][0]["functionResponse"]
        assert "user@response.com" not in fr["response"]["email"]
        assert "[EMAIL_" in fr["response"]["email"]
        assert fr["response"]["status"] == "active"
        assert not result.should_block

    def test_secret_in_function_response_blocks(self):
        key = "hf_" + "Z" * 40
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_credentials",
                                "response": {"token": key},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fr = result.sanitized_payload["contents"][0]["parts"][0]["functionResponse"]
        assert key not in fr["response"]["token"]
        assert result.should_block

    def test_nested_function_response_recursively_scanned(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_contact",
                                "response": {
                                    "data": {
                                        "email": "resp@nested.com",
                                        "phone": "010-7777-8888",
                                    }
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        data = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionResponse"]["response"]["data"]
        )
        assert "resp@nested.com" not in data["email"]
        assert "010-7777-8888" not in data["phone"]
        assert not result.should_block

    def test_function_response_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "check",
                                "response": {"result": "Contact admin@corp.io"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fr_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.FUNCTION_RESPONSE]
        assert fr_evts

    def test_function_response_name_preserved(self):
        """functionResponse.name is a structural field and must not be modified."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_user_info",
                                "response": {"info": "user@test.com"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fr = result.sanitized_payload["contents"][0]["parts"][0]["functionResponse"]
        assert fr["name"] == "get_user_info"


# ─────────────────────────────────────────────────────────────────────────────
# 7. executableCode.code — source code text
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutableCode:
    """executableCode.code is text that should be scanned."""

    def test_email_in_executable_code_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": "send_email('admin@code.com', 'Hello')",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ec = result.sanitized_payload["contents"][0]["parts"][0]["executableCode"]
        assert "admin@code.com" not in ec["code"]
        assert "[EMAIL_" in ec["code"]
        assert ec["language"] == "PYTHON"
        assert not result.should_block

    def test_api_key_in_executable_code_blocks(self):
        key = "sk-ant-api03-" + "C" * 50
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": f"api_key = '{key}'",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ec = result.sanitized_payload["contents"][0]["parts"][0]["executableCode"]
        assert key not in ec["code"]
        assert result.should_block

    def test_executable_code_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": "print(dev@corp.io)",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ec_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.EXECUTABLE_CODE]
        assert ec_evts

    def test_executable_code_language_field_preserved(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executableCode": {
                                "language": "JAVASCRIPT",
                                "code": "const x = 1;",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ec = result.sanitized_payload["contents"][0]["parts"][0]["executableCode"]
        assert ec["language"] == "JAVASCRIPT"


# ─────────────────────────────────────────────────────────────────────────────
# 8. codeExecutionResult.output — code execution output
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExecutionResult:
    """codeExecutionResult.output is text that should be scanned."""

    def test_email_in_code_execution_result_masked(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "Result: user@output.com logged in",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        cer = result.sanitized_payload["contents"][0]["parts"][0]["codeExecutionResult"]
        assert "user@output.com" not in cer["output"]
        assert "[EMAIL_" in cer["output"]
        assert cer["outcome"] == "OUTCOME_OK"
        assert not result.should_block

    def test_api_key_in_code_execution_result_blocks(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": f"AWS Key: {key}",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        cer = result.sanitized_payload["contents"][0]["parts"][0]["codeExecutionResult"]
        assert key not in cer["output"]
        assert result.should_block

    def test_code_execution_result_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "Contacted admin@corp.io",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        cer_evts = [e for e in result.field_events
                    if e.scan_field == ScanField.CODE_EXECUTION_RESULT]
        assert cer_evts

    def test_code_execution_result_outcome_preserved(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_FAILED",
                                "output": "Error at user@test.com",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        cer = result.sanitized_payload["contents"][0]["parts"][0]["codeExecutionResult"]
        assert cer["outcome"] == "OUTCOME_FAILED"


# ─────────────────────────────────────────────────────────────────────────────
# 9. inlineData / inline_data — unscannable binary parts
# ─────────────────────────────────────────────────────────────────────────────

class TestInlineData:
    """inlineData parts are binary — unscannable → coverage gap + block by default."""

    def test_inline_data_coverage_gap_default_blocks(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgAB...",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.coverage_gaps, "inlineData part should record a coverage gap"
        assert result.should_block, "default unscannable_action=block should block"

    def test_inline_data_warn_allow_mode_does_not_block(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "iVBORw0KGgo=",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps  # gap still recorded
        assert not result.should_block  # but no block

    def test_inline_data_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/mp3",
                                "data": "abc123",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        id_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.INLINE_DATA]
        assert id_evts

    def test_inline_data_passed_through_unchanged(self):
        """The inlineData dict itself is not modified, only a gap is recorded."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "ORIGINAL_DATA",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        part = result.sanitized_payload["contents"][0]["parts"][0]
        assert part["inlineData"]["data"] == "ORIGINAL_DATA"  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 10. fileData / file_data — unscannable external file references
# ─────────────────────────────────────────────────────────────────────────────

class TestFileData:
    """fileData parts reference external files — unscannable → coverage gap + block."""

    def test_file_data_coverage_gap_default_blocks(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "https://storage.googleapis.com/bucket/doc.pdf",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.coverage_gaps, "fileData part should record a coverage gap"
        assert result.should_block, "default unscannable_action=block should block"

    def test_file_data_warn_allow_does_not_block(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "video/mp4",
                                "fileUri": "gs://bucket/video.mp4",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps
        assert not result.should_block

    def test_file_data_scan_field_tagged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "text/plain",
                                "fileUri": "gs://bucket/file.txt",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        fd_evts = [e for e in result.field_events
                   if e.scan_field == ScanField.FILE_DATA]
        assert fd_evts

    def test_file_data_passed_through_unchanged(self):
        """The fileData dict itself is not modified."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "gs://my-bucket/report.pdf",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        part = result.sanitized_payload["contents"][0]["parts"][0]
        assert part["fileData"]["fileUri"] == "gs://my-bucket/report.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# 11. snake_case field aliases
# ─────────────────────────────────────────────────────────────────────────────

class TestSnakeCaseFieldAliases:
    """
    snake_case part keys (function_call, function_response, inline_data,
    file_data, executable_code, code_execution_result) work identically
    to their camelCase counterparts.
    """

    def test_function_call_snake_case_args_scanned(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "send_email",
                                "args": {"to": "snake@domain.com", "subject": "Hi"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc = result.sanitized_payload["contents"][0]["parts"][0]["function_call"]
        assert "snake@domain.com" not in fc["args"]["to"]
        assert "[EMAIL_" in fc["args"]["to"]
        assert not result.should_block

    def test_function_response_snake_case_scanned(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "get_user",
                                "response": {"email": "resp@snake.io"},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fr = result.sanitized_payload["contents"][0]["parts"][0]["function_response"]
        assert "resp@snake.io" not in fr["response"]["email"]
        assert "[EMAIL_" in fr["response"]["email"]
        assert not result.should_block

    def test_inline_data_snake_case_coverage_gap(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": "base64encodeddata",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.coverage_gaps
        assert result.should_block

    def test_file_data_snake_case_coverage_gap(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "file_uri": "gs://bucket/doc.pdf",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.coverage_gaps
        assert result.should_block

    def test_executable_code_snake_case_scanned(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "executable_code": {
                                "language": "PYTHON",
                                "code": "contact('user@snake.com')",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ec = result.sanitized_payload["contents"][0]["parts"][0]["executable_code"]
        assert "user@snake.com" not in ec["code"]
        assert "[EMAIL_" in ec["code"]
        assert not result.should_block

    def test_code_execution_result_snake_case_scanned(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "code_execution_result": {
                                "outcome": "OUTCOME_OK",
                                "output": "Logged in as admin@snake.io",
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        cer = result.sanitized_payload["contents"][0]["parts"][0]["code_execution_result"]
        assert "admin@snake.io" not in cer["output"]
        assert "[EMAIL_" in cer["output"]
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 12. Unknown / unrecognized part keys → coverage alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownPartTypes:
    """Unknown parts raise a coverage alarm; blocking depends on unknown_field_action."""

    def test_unknown_part_keys_alarm_and_block_by_default(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"future_part_type": {"data": "something"}},
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.unknown_fields, "Expected unknown field alarm"
        assert any("future_part_type" in u for u in result.unknown_fields)
        assert result.should_block  # default unknown_field_action=block

    def test_unknown_part_warn_allow_does_not_block(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"new_future_type": {"value": "xyz"}},
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unknown_field_action="warn_allow"
        )
        assert result.unknown_fields
        assert not result.should_block

    def test_unknown_part_in_system_instruction_alarm(self):
        """Unknown part type in systemInstruction also triggers alarm."""
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [
                    {"text": "Be helpful"},
                    {"future_block": {"content": "xyz"}},
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert result.unknown_fields
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 13. Structural validity of sanitized payload
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizedPayloadStructure:
    """The sanitized payload must be structurally valid and preserve metadata."""

    def test_top_level_keys_preserved(self):
        payload = {
            "model": "gemini-2.0-flash",
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        sp = result.sanitized_payload
        assert sp["model"] == "gemini-2.0-flash"
        assert sp["generationConfig"]["temperature"] == 0.7
        assert sp["generationConfig"]["maxOutputTokens"] == 1024

    def test_content_item_roles_preserved(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        contents = result.sanitized_payload["contents"]
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_mixed_parts_structural_integrity(self):
        """A parts list with text + functionCall keeps structural fields."""
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"text": "I'll call the function."},
                        {
                            "functionCall": {
                                "name": "do_thing",
                                "args": {"param": "value@corp.com"},
                            }
                        },
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        parts = result.sanitized_payload["contents"][0]["parts"]
        assert "text" in parts[0]
        assert "functionCall" in parts[1]
        assert parts[1]["functionCall"]["name"] == "do_thing"

    def test_safety_settings_and_generation_config_untouched(self):
        """Non-content top-level fields are passed through unchanged."""
        payload = {
            "model": "gemini-2.0-flash",
            "safetySettings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            "generationConfig": {"stopSequences": ["STOP"], "temperature": 1.0},
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        sp = result.sanitized_payload
        assert sp["safetySettings"] == payload["safetySettings"]
        assert sp["generationConfig"]["stopSequences"] == ["STOP"]


# ─────────────────────────────────────────────────────────────────────────────
# 14. Cross-field placeholder consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFieldConsistency:
    """Same real value → same placeholder across all fields in one request."""

    def test_same_email_same_placeholder_across_fields(self):
        """Email in systemInstruction + user text → same placeholder."""
        email = "shared@example.com"
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": f"Contact {email} for support."}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Please email {email} now."}],
                }
            ],
        }
        engine = fresh_engine()
        result = scrub_gemini_request(payload, engine)
        si_text = result.sanitized_payload["systemInstruction"]["parts"][0]["text"]
        msg_text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        si_ph = _email_placeholders(si_text)
        msg_ph = _email_placeholders(msg_text)
        assert si_ph, "Expected placeholder in systemInstruction"
        assert msg_ph, "Expected placeholder in message text"
        assert si_ph[0] == msg_ph[0], "Same email must produce same placeholder"

    def test_same_email_in_text_and_function_call_args(self):
        """Email in text part and functionCall.args → same placeholder."""
        email = "shared_fc@corp.com"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": f"Send to {email}"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "notify",
                                "args": {"recipient": email},
                            }
                        }
                    ],
                },
            ],
        }
        engine = fresh_engine()
        result = scrub_gemini_request(payload, engine)
        usr_text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        fc_args = result.sanitized_payload["contents"][1]["parts"][0]["functionCall"]["args"]
        usr_ph = _email_placeholders(usr_text)
        fc_ph = _email_placeholders(fc_args["recipient"])
        assert usr_ph and fc_ph
        assert usr_ph[0] == fc_ph[0], "Same email in text and functionCall.args must match"

    def test_same_email_in_function_call_and_response(self):
        """Email in functionCall.args and functionResponse.response → same placeholder."""
        email = "roundtrip@corp.com"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "args": {"query": email},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"result": email},
                            }
                        }
                    ],
                },
            ],
        }
        engine = fresh_engine()
        result = scrub_gemini_request(payload, engine)
        fc_args = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]["args"]
        fr_resp = result.sanitized_payload["contents"][1]["parts"][0]["functionResponse"]["response"]
        fc_ph = _email_placeholders(fc_args["query"])
        fr_ph = _email_placeholders(fr_resp["result"])
        assert fc_ph and fr_ph
        assert fc_ph[0] == fr_ph[0]

    def test_different_emails_different_placeholders(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "From a@first.com"}]},
                {"role": "user", "parts": [{"text": "To b@second.io"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        ph_a = _email_placeholders(
            result.sanitized_payload["contents"][0]["parts"][0]["text"]
        )
        ph_b = _email_placeholders(
            result.sanitized_payload["contents"][1]["parts"][0]["text"]
        )
        assert ph_a and ph_b
        assert ph_a[0] != ph_b[0], "Different emails must get different placeholders"


# ─────────────────────────────────────────────────────────────────────────────
# 15 & 16. No raw PII/secret survives + no config required
# ─────────────────────────────────────────────────────────────────────────────

class TestNoPIISurvivesAndNoConfig:
    """Integration: no raw PII/secret in sanitized payload; plain Engine() protects all."""

    def test_full_mixed_gemini_payload_all_sanitized(self):
        """Full payload with PII in every Gemini field type is fully sanitized."""
        email = "victim@corp.io"
        phone = "010-8888-9999"
        api_key = "sk-ant-api03-" + "X" * 50

        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": f"Notify {email} of all issues."}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"User phone: {phone}"},
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"contact": email, "key": api_key},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "save",
                                "args": {"email": email, "backup": phone},
                            }
                        },
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": f"alert('{email}')",
                            }
                        },
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": f"Processed {phone}",
                            }
                        },
                    ],
                },
            ],
        }

        result = scrub_gemini_request(payload, fresh_engine())
        sp = result.sanitized_payload

        # systemInstruction
        assert email not in sp["systemInstruction"]["parts"][0]["text"]

        # user text
        assert phone not in sp["contents"][0]["parts"][0]["text"]

        # functionResponse
        fr_resp = sp["contents"][0]["parts"][1]["functionResponse"]["response"]
        assert email not in fr_resp["contact"]
        assert api_key not in fr_resp["key"]

        # functionCall args
        fc_args = sp["contents"][1]["parts"][0]["functionCall"]["args"]
        assert email not in fc_args["email"]
        assert phone not in fc_args["backup"]

        # executableCode
        assert email not in sp["contents"][1]["parts"][1]["executableCode"]["code"]

        # codeExecutionResult
        assert phone not in sp["contents"][1]["parts"][2]["codeExecutionResult"]["output"]

        # api_key is secret → should_block
        assert result.should_block

    def test_no_config_required(self):
        """
        Verify that a plain Engine() with no arguments provides protection
        for all category classes (secret, pii, korean_pii) out of the box.
        """
        rrn = "900505-1234564"           # valid Korean RRN checksum
        card = "4532015112830366"         # Luhn-valid Visa test number
        email = "test@example.com"
        aws_key = "AKIAIOSFODNN7EXAMPLE"

        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": f"RRN: {rrn}"}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Card: {card}"},
                        {
                            "functionResponse": {
                                "name": "check",
                                "response": {"email": email, "key": aws_key},
                            }
                        },
                    ],
                }
            ],
        }

        # No policy file, no custom config — plain Engine()
        result = scrub_gemini_request(payload, Engine())
        sp = result.sanitized_payload

        assert rrn not in sp["systemInstruction"]["parts"][0]["text"]
        assert card not in sp["contents"][0]["parts"][0]["text"]
        fr_resp = sp["contents"][0]["parts"][1]["functionResponse"]["response"]
        assert email not in fr_resp["email"]
        assert aws_key not in fr_resp["key"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 17. Multi-turn / multi-content-item payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnPayload:
    """Multi-turn conversations are fully scrubbed across all content items."""

    def test_pii_across_multiple_content_items(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "My email is first@a.com"}]},
                {"role": "model", "parts": [{"text": "Got it, first@a.com noted."}]},
                {"role": "user", "parts": [{"text": "Also reach second@b.com"}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        contents = result.sanitized_payload["contents"]
        assert "first@a.com" not in contents[0]["parts"][0]["text"]
        assert "first@a.com" not in contents[1]["parts"][0]["text"]
        assert "second@b.com" not in contents[2]["parts"][0]["text"]
        assert not result.should_block

    def test_same_value_consistent_across_turns(self):
        """Single engine session → same email always gets same placeholder."""
        email = "consistent@x.com"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": f"Email: {email}"}]},
                {"role": "model", "parts": [{"text": f"Sure, {email} it is."}]},
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        contents = result.sanitized_payload["contents"]
        ph_u = _email_placeholders(contents[0]["parts"][0]["text"])
        ph_m = _email_placeholders(contents[1]["parts"][0]["text"])
        assert ph_u and ph_m
        assert ph_u[0] == ph_m[0], "Same email must produce same placeholder across turns"

    def test_realistic_multi_turn_with_function_calls(self):
        """Realistic conversation: user → model functionCall → user functionResponse."""
        key = "AKIAIOSFODNN7EXAMPLE"
        email = "user@example.com"
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [
                {"role": "user", "parts": [{"text": f"Send to {email}"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send_notification",
                                "args": {"to": email, "subject": "Report"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "send_notification",
                                "response": {"status": "sent", "aws_key": key},
                            }
                        }
                    ],
                },
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        sp = result.sanitized_payload
        fc_args = sp["contents"][1]["parts"][0]["functionCall"]["args"]
        fr_resp = sp["contents"][2]["parts"][0]["functionResponse"]["response"]
        assert email not in fc_args["to"]
        assert key not in fr_resp["aws_key"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 18. Numeric / boolean / null in args pass through unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestNonStringLeafValues:
    """Non-string leaf values in functionCall.args and functionResponse.response pass through."""

    def test_numeric_args_unchanged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "compute",
                                "args": {
                                    "count": 42,
                                    "enabled": True,
                                    "ratio": 3.14,
                                    "nothing": None,
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        args = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]["args"]
        assert args["count"] == 42
        assert args["enabled"] is True
        assert args["ratio"] == 3.14
        assert args["nothing"] is None
        assert not result.should_block

    def test_numeric_response_values_unchanged(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "calculate",
                                "response": {
                                    "result": 42,
                                    "success": True,
                                    "error": None,
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        resp = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionResponse"]["response"]
        )
        assert resp["result"] == 42
        assert resp["success"] is True
        assert resp["error"] is None
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 19. Deeply nested JSON objects
# ─────────────────────────────────────────────────────────────────────────────

class TestDeeplyNestedJSON:
    """3+ level nesting in functionCall.args and functionResponse.response is fully walked."""

    def test_deeply_nested_function_call_args(self):
        key = "ghp_" + "G" * 40
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "configure",
                                "args": {
                                    "a": {"b": {"c": {"d": {"token": key}}}}
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        token = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionCall"]["args"]["a"]["b"]["c"]["d"]["token"]
        )
        assert key not in token
        assert result.should_block

    def test_deeply_nested_function_response(self):
        email = "deep@nested.io"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {
                                    "level1": {"level2": {"level3": {"contact": email}}}
                                },
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        contact = (
            result.sanitized_payload["contents"][0]["parts"][0]
            ["functionResponse"]["response"]["level1"]["level2"]["level3"]["contact"]
        )
        assert email not in contact
        assert "[EMAIL_" in contact
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 20. Original payload not mutated
# ─────────────────────────────────────────────────────────────────────────────

class TestOriginalPayloadNotMutated:
    def test_scrub_does_not_mutate_original(self):
        email = "original@check.com"
        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {"parts": [{"text": f"Email: {email}"}]},
            "contents": [
                {"role": "user", "parts": [{"text": email}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "fn",
                                "args": {"contact": email},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "fn",
                                "response": {"result": email},
                            }
                        }
                    ],
                },
            ],
        }
        original_copy = copy.deepcopy(payload)
        scrub_gemini_request(payload, fresh_engine())
        assert payload == original_copy, "scrub_gemini_request must not mutate the input"


# ─────────────────────────────────────────────────────────────────────────────
# 21. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_payload_does_not_raise(self):
        result = scrub_gemini_request({}, fresh_engine())
        assert result.sanitized_payload == {}
        assert not result.should_block
        assert not result.coverage_gaps

    def test_no_contents_key(self):
        result = scrub_gemini_request({"model": "gemini-2.0-flash"}, fresh_engine())
        assert result.sanitized_payload["model"] == "gemini-2.0-flash"
        assert not result.should_block

    def test_empty_contents_list(self):
        result = scrub_gemini_request(
            {"model": "gemini-2.0-flash", "contents": []}, fresh_engine()
        )
        assert result.sanitized_payload["contents"] == []
        assert not result.should_block

    def test_empty_parts_list(self):
        result = scrub_gemini_request(
            {
                "model": "gemini-2.0-flash",
                "contents": [{"role": "user", "parts": []}],
            },
            fresh_engine(),
        )
        assert result.sanitized_payload["contents"][0]["parts"] == []
        assert not result.should_block

    def test_empty_text_part(self):
        result = scrub_gemini_request(
            {
                "model": "gemini-2.0-flash",
                "contents": [{"role": "user", "parts": [{"text": ""}]}],
            },
            fresh_engine(),
        )
        assert result.sanitized_payload["contents"][0]["parts"][0]["text"] == ""
        assert not result.should_block

    def test_content_item_without_parts_key_skipped(self):
        """Content items without a 'parts' key are skipped without error."""
        result = scrub_gemini_request(
            {
                "model": "gemini-2.0-flash",
                "contents": [
                    {"role": "user"},  # no parts
                    {"role": "model", "parts": [{"text": "Hello user@skip.com"}]},
                ],
            },
            fresh_engine(),
        )
        # First item skipped, second scrubbed
        assert "user@skip.com" not in (
            result.sanitized_payload["contents"][1]["parts"][0]["text"]
        )
        assert not result.should_block

    def test_non_dict_content_item_skipped(self):
        """Non-dict entries in contents are skipped without error."""
        result = scrub_gemini_request(
            {
                "model": "gemini-2.0-flash",
                "contents": [
                    "not a dict",
                    {"role": "user", "parts": [{"text": "Hi user@test.com"}]},
                ],
            },
            fresh_engine(),
        )
        assert result.sanitized_payload["contents"][0] == "not a dict"
        assert "user@test.com" not in (
            result.sanitized_payload["contents"][1]["parts"][0]["text"]
        )
        assert not result.should_block

    def test_non_dict_part_skipped(self):
        """Non-dict entries in parts are passed through without error."""
        result = scrub_gemini_request(
            {
                "model": "gemini-2.0-flash",
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            "not a dict part",
                            {"text": "Contact user@test.com"},
                        ],
                    }
                ],
            },
            fresh_engine(),
        )
        assert result.sanitized_payload["contents"][0]["parts"][0] == "not a dict part"
        assert "user@test.com" not in (
            result.sanitized_payload["contents"][0]["parts"][1]["text"]
        )
        assert not result.should_block

    def test_private_key_pem_in_text_blocks(self):
        """PEM private key header in text content is blocked."""
        pem = "-----BEGIN RSA PRIVATE KEY-----"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Here is my key:\n{pem}\nMIIEowIBAAK..."}],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        assert pem not in result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert result.should_block

    def test_jwt_token_in_function_call_args_blocks(self):
        """JWT-style token in functionCall.args is detected and blocked."""
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "auth",
                                "args": {"token": jwt},
                            }
                        }
                    ],
                }
            ],
        }
        result = scrub_gemini_request(payload, fresh_engine())
        fc_args = result.sanitized_payload["contents"][0]["parts"][0]["functionCall"]["args"]
        assert jwt not in fc_args["token"]
        assert result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 22. unscannable_action='warn_allow' still records gap, no block
# ─────────────────────────────────────────────────────────────────────────────

class TestUnscannableWarnAllow:
    """warn_allow mode records coverage gaps without blocking."""

    def test_inline_data_warn_allow(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": "abc123"}},
                        {"text": "Describe this image"},
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps, "Coverage gap should still be recorded"
        assert not result.should_block, "warn_allow must not block"

    def test_file_data_warn_allow(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "gs://bucket/doc.pdf",
                            }
                        },
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        assert result.coverage_gaps
        assert not result.should_block

    def test_mixed_text_and_inline_data_text_scanned_image_gaped(self):
        """Text parts are scanned; inlineData parts get a coverage gap."""
        email = "mixed@example.com"
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"See {email} for context."},
                        {"inlineData": {"mimeType": "image/jpeg", "data": "abc123"}},
                    ],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        text = result.sanitized_payload["contents"][0]["parts"][0]["text"]
        assert email not in text
        assert "[EMAIL_" in text
        assert result.coverage_gaps  # image gap present
        assert not result.should_block  # text-only PII doesn't block


# ─────────────────────────────────────────────────────────────────────────────
# 23. unknown_field_action='warn_allow'
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownFieldWarnAllow:
    """warn_allow for unknown fields logs but does not block."""

    def test_unknown_part_warn_allow_does_not_block(self):
        payload = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [{"future_gemini_part": {"data": "xyz"}}],
                }
            ],
        }
        result = scrub_gemini_request(
            payload, fresh_engine(), unknown_field_action="warn_allow"
        )
        assert result.unknown_fields
        assert not result.should_block


# ─────────────────────────────────────────────────────────────────────────────
# 24. All Gemini field types covered in one request (comprehensive integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestComprehensiveIntegration:
    """All Gemini-specific field types in one request (camelCase and snake_case)."""

    def test_all_camel_case_field_types_scrubbed(self):
        """Verify every supported camelCase part type is correctly handled."""
        email = "test@allfields.io"
        phone = "010-5678-9012"
        key = "sk-ant-api03-" + "D" * 50

        payload = {
            "model": "gemini-2.0-flash",
            "systemInstruction": {
                "parts": [{"text": f"System: {email}"}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Text: {phone}"},
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"data": f"resp: {email}"},
                            }
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "imgdata",
                            }
                        },
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "gs://bucket/doc.pdf",
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "send",
                                "args": {"to": email, "key": key},
                            }
                        },
                        {
                            "executableCode": {
                                "language": "PYTHON",
                                "code": f"notify('{email}')",
                            }
                        },
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": f"Done: {phone}",
                            }
                        },
                    ],
                },
            ],
        }

        result = scrub_gemini_request(
            payload, fresh_engine(), unscannable_action="warn_allow"
        )
        sp = result.sanitized_payload

        # systemInstruction
        assert email not in sp["systemInstruction"]["parts"][0]["text"]

        # text part
        assert phone not in sp["contents"][0]["parts"][0]["text"]

        # functionResponse.response
        fr_resp = sp["contents"][0]["parts"][1]["functionResponse"]["response"]
        assert email not in fr_resp["data"]

        # inlineData — coverage gap, data unchanged
        id_part = sp["contents"][0]["parts"][2]
        assert "inlineData" in id_part
        assert id_part["inlineData"]["data"] == "imgdata"  # unchanged

        # fileData — coverage gap, uri unchanged
        fd_part = sp["contents"][0]["parts"][3]
        assert "fileData" in fd_part

        # functionCall args
        fc_args = sp["contents"][1]["parts"][0]["functionCall"]["args"]
        assert email not in fc_args["to"]
        assert key not in fc_args["key"]

        # executableCode
        assert email not in sp["contents"][1]["parts"][1]["executableCode"]["code"]

        # codeExecutionResult
        assert phone not in sp["contents"][1]["parts"][2]["codeExecutionResult"]["output"]

        # Coverage gaps recorded for inlineData and fileData
        assert len(result.coverage_gaps) >= 2

        # API key → should_block
        assert result.should_block
