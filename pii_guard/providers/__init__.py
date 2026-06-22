"""
Provider-specific wire format parsers for PII-Guard.

Sub-AC 1:   Claude provider request parser (claude_parser)
Sub-AC 2a:  Claude wire-format scrubber (claude)
Sub-AC 2:   OpenAI provider request parser (openai_parser)
Sub-AC 2b:  OpenAI wire-format scrubber (openai)
Sub-AC 2c:  Gemini (Google Generative AI API) wire-format scrubber
Sub-AC 3:   Gemini provider request parser (gemini_parser)
Sub-AC 3a:  Schema coverage detection — field-set diff + version diff (schema_coverage)
Sub-AC 3b:  Coverage alarm emission and action enforcement (coverage_alarm)
"""
# Sub-AC 1 — Claude pure parser (no scanning, no masking)
from .claude_parser import (
    ClaudeFieldMap,
    ParsedField as ClaudeParsedField,
    ScanField as ClaudeParserScanField,
    parse_claude_request,
)
from .claude import (
    ClaudeRequestScrubResult,
    FieldScanEvent as ClaudeFieldScanEvent,
    ScanField as ClaudeScanField,
    scrub_claude_request,
)
# Sub-AC 2 — OpenAI pure parser (no scanning, no masking)
from .openai_parser import (
    OpenAIFieldMap,
    ParsedField as OpenAIParsedField,
    ScanField as OpenAIParserScanField,
    parse_openai_request,
)
from .openai import (
    FieldScanEvent as OpenAIFieldScanEvent,
    OpenAIRequestScrubResult,
    ScanField as OpenAIScanField,
    scrub_openai_request,
)
# Sub-AC 2c — Gemini wire-format scrubber
from .gemini import (
    FieldScanEvent as GeminiFieldScanEvent,
    GeminiRequestScrubResult,
    ScanField as GeminiScanField,
    scrub_gemini_request,
)
# Sub-AC 3 — Gemini pure parser (no scanning, no masking)
from .gemini_parser import (
    GeminiFieldMap,
    ParsedField as GeminiParsedField,
    ScanField as GeminiParserScanField,
    parse_gemini_request,
)

# Sub-AC 3a — Schema coverage detection (pure field-set diff + version diff)
from .schema_coverage import (
    FieldDelta,
    VersionDelta,
    diff_api_version,
    diff_claude_fields,
    diff_gemini_fields,
    diff_openai_fields,
    diff_request,
)

# Sub-AC 3b — Coverage alarm emission and action enforcement
from .coverage_alarm import (
    AnyDelta,
    CoverageAlarmEvent,
    CoverageAlarmResult,
    apply_coverage_alarm_policy,
    emit_coverage_alarms,
)

__all__ = [
    # Claude — Sub-AC 1 parser
    "ClaudeFieldMap",
    "ClaudeParsedField",
    "ClaudeParserScanField",
    "parse_claude_request",
    # Claude — Sub-AC 2a scrubber
    "ClaudeRequestScrubResult",
    "ClaudeFieldScanEvent",
    "ClaudeScanField",
    "scrub_claude_request",
    # OpenAI — Sub-AC 2 parser
    "OpenAIFieldMap",
    "OpenAIParsedField",
    "OpenAIParserScanField",
    "parse_openai_request",
    # OpenAI — Sub-AC 2b scrubber
    "OpenAIFieldScanEvent",
    "OpenAIRequestScrubResult",
    "OpenAIScanField",
    "scrub_openai_request",
    # Gemini — Sub-AC 2c scrubber
    "GeminiFieldScanEvent",
    "GeminiRequestScrubResult",
    "GeminiScanField",
    "scrub_gemini_request",
    # Gemini — Sub-AC 3 parser
    "GeminiFieldMap",
    "GeminiParsedField",
    "GeminiParserScanField",
    "parse_gemini_request",
    # Schema coverage detection — Sub-AC 3a
    "FieldDelta",
    "VersionDelta",
    "diff_api_version",
    "diff_claude_fields",
    "diff_gemini_fields",
    "diff_openai_fields",
    "diff_request",
    # Coverage alarm emission and policy — Sub-AC 3b
    "AnyDelta",
    "CoverageAlarmEvent",
    "CoverageAlarmResult",
    "apply_coverage_alarm_policy",
    "emit_coverage_alarms",
]
