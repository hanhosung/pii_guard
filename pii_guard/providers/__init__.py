"""
Provider-specific wire format parsers for PII-Guard.

Sub-AC 2a: Claude (Anthropic Messages API)
Sub-AC 2b: OpenAI (chat-completions API)
Sub-AC 2c: Gemini (Google Generative AI API)
"""
from .claude import (
    ClaudeRequestScrubResult,
    FieldScanEvent as ClaudeFieldScanEvent,
    ScanField as ClaudeScanField,
    scrub_claude_request,
)
from .openai import (
    FieldScanEvent as OpenAIFieldScanEvent,
    OpenAIRequestScrubResult,
    ScanField as OpenAIScanField,
    scrub_openai_request,
)
from .gemini import (
    FieldScanEvent as GeminiFieldScanEvent,
    GeminiRequestScrubResult,
    ScanField as GeminiScanField,
    scrub_gemini_request,
)

__all__ = [
    # Claude
    "ClaudeRequestScrubResult",
    "ClaudeFieldScanEvent",
    "ClaudeScanField",
    "scrub_claude_request",
    # OpenAI
    "OpenAIFieldScanEvent",
    "OpenAIRequestScrubResult",
    "OpenAIScanField",
    "scrub_openai_request",
    # Gemini
    "GeminiFieldScanEvent",
    "GeminiRequestScrubResult",
    "GeminiScanField",
    "scrub_gemini_request",
]
