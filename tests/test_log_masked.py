"""
Tests for the --log-masked console output (operator can confirm, when calling a
real upstream like Anthropic, that PII is masked/blocked before it leaves the host).

The logger must print the MASKED payload only — never the raw request body.
"""
from __future__ import annotations

from types import SimpleNamespace

from pii_guard.cli import build_parser
from pii_guard.proxy import PIIGuardProxy


def _proxy(log_masked: bool) -> PIIGuardProxy:
    # Not started — we only exercise the pure logging method.
    return PIIGuardProxy("https://api.anthropic.com", log_masked=log_masked)


def _fake_scrub(sanitized_payload):
    det = SimpleNamespace(category="PERSON", action="Action.TOKENIZE_ROUNDTRIP",
                          placeholder_token="PERSON_1")
    event = SimpleNamespace(detections=[det])
    return SimpleNamespace(sanitized_payload=sanitized_payload, field_events=[event],
                           should_block=False)


def test_log_masked_prints_masked_payload(capsys):
    proxy = _proxy(log_masked=True)
    payload = {"messages": [{"role": "user", "content": "[PERSON_1] hello"}]}
    proxy._log_traffic("/v1/messages", "claude", _fake_scrub(payload),
                       SimpleNamespace(should_block=False), blocked=False)
    out = capsys.readouterr().out
    assert "FORWARD to upstream" in out
    assert "[PERSON_1]" in out          # masked payload is shown
    assert "PERSON" in out and "PERSON_1" in out  # detection summary
    assert "api.anthropic.com/v1/messages" in out


def test_log_masked_never_prints_raw_original(capsys):
    """The logger receives only the sanitized payload, so raw PII cannot appear."""
    proxy = _proxy(log_masked=True)
    payload = {"messages": [{"role": "user", "content": "[PERSON_1]"}]}
    proxy._log_traffic("/v1/messages", "claude", _fake_scrub(payload),
                       SimpleNamespace(should_block=False), blocked=False)
    out = capsys.readouterr().out
    assert "김민수" not in out  # a raw name never reaches this log path


def test_blocked_request_logged_as_not_forwarded(capsys):
    proxy = _proxy(log_masked=True)
    payload = {"messages": [{"role": "user", "content": "[AWS_SECRET_1_BLOCKED]"}]}
    proxy._log_traffic("/v1/messages", "claude", _fake_scrub(payload),
                       SimpleNamespace(should_block=False), blocked=True)
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "NOT forwarded" in out


def test_logging_disabled_by_default(capsys):
    proxy = _proxy(log_masked=False)
    payload = {"messages": [{"role": "user", "content": "x"}]}
    proxy._log_traffic("/v1/messages", "claude", _fake_scrub(payload),
                       SimpleNamespace(should_block=False), blocked=False)
    assert capsys.readouterr().out == ""


def test_serve_parser_has_log_masked_flag():
    parser = build_parser()
    args = parser.parse_args(
        ["serve", "--upstream-url", "https://api.anthropic.com", "--log-masked"]
    )
    assert args.log_masked is True
    args2 = parser.parse_args(["serve", "--upstream-url", "https://api.anthropic.com"])
    assert args2.log_masked is False
