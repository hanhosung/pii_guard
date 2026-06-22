"""
Regression test for the Stage-2 NER ↔ proxy wiring (found via E2E smoke test).

The NER engine (AC 10) is exercised directly by unit tests, but a real
end-to-end run through ``piiguard serve`` revealed that ``cmd_serve`` built the
proxy with a default ``Engine()`` that had **no** Stage-2 runner — so in
production the proxy only applied Stage-1 regex and silently forwarded
unstructured Korean PII (person names, addresses, organizations) to the upstream.

These tests lock in the fix: ``serve`` wires a Stage-2 NER runner into the
engine by default (secure-by-default), and ``--no-ner`` opts out.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from pii_guard.cli import build_parser, cmd_serve


class _StopServe(Exception):
    """Sentinel raised in place of signal.pause() to unblock cmd_serve in tests."""


def _serve_args(no_ner: bool) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_url="http://127.0.0.1:9",
        host="127.0.0.1",
        port=0,
        no_ner=no_ner,
    )


def _run_cmd_serve(args: argparse.Namespace) -> MagicMock:
    """
    Invoke cmd_serve with the proxy + blocking call mocked out, and return the
    patched PIIGuardProxy mock so the caller can inspect constructor kwargs.
    """
    with patch("pii_guard.cli.PIIGuardProxy") as proxy_cls, \
            patch("pii_guard.cli.signal.pause", side_effect=_StopServe):
        proxy_cls.return_value.port = 4444
        with pytest.raises(_StopServe):
            cmd_serve(args)
    return proxy_cls


class TestServeNerWiring:
    def test_ner_enabled_by_default(self):
        """serve (no flags) must pass an Engine with a Stage-2 runner attached."""
        proxy_cls = _run_cmd_serve(_serve_args(no_ner=False))
        engine = proxy_cls.call_args.kwargs["engine"]
        assert engine is not None, "serve must build an engine (not rely on proxy default)"
        # The default Engine() has no Stage-2 runner; a NER-wired engine does.
        assert getattr(engine, "_stage2_runner", None) is not None, (
            "serve must wire a Stage-2 NER runner into the engine by default "
            "(otherwise the proxy silently forwards unstructured Korean PII)"
        )
        runner = engine._stage2_runner
        if hasattr(runner, "close"):
            runner.close()

    def test_no_ner_flag_disables_stage2(self):
        """serve --no-ner must NOT attach a Stage-2 runner."""
        proxy_cls = _run_cmd_serve(_serve_args(no_ner=True))
        engine = proxy_cls.call_args.kwargs["engine"]
        # Either no engine override, or an engine without a Stage-2 runner.
        if engine is not None:
            assert getattr(engine, "_stage2_runner", None) is None

    def test_serve_parser_has_no_ner_flag(self):
        """The serve sub-command exposes the --no-ner escape hatch."""
        parser = build_parser()
        args = parser.parse_args(
            ["serve", "--upstream-url", "https://api.anthropic.com", "--no-ner"]
        )
        assert args.no_ner is True

    def test_serve_parser_ner_on_by_default(self):
        parser = build_parser()
        args = parser.parse_args(
            ["serve", "--upstream-url", "https://api.anthropic.com"]
        )
        assert args.no_ner is False
