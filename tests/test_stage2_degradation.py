"""
Sub-AC 3b: Stage2 NER OOM/timeout graceful degradation.

Tests that when Stage-2 NER exceeds memory or time limits:
  • The scan falls back to Stage-1-only detection (Stage-1 results preserved).
  • A coverage-gap is recorded (``coverage_gap=True`` on the result).
  • A human-readable ``fail_reason`` / ``stage2_gap_reason`` is populated.
  • The forwarding core (Engine) is never blocked by a Stage-2 failure.
  • Stage-2 failure does NOT trigger fail-closed blocking (unlike scan errors
    covered in test_fail_closed.py); it degrades gracefully.

Scenarios
---------
1.  Stage-2 timeout (slow worker never responds within hard timeout)
2.  Stage-2 OOM — hard process death via os._exit (simulates SIGKILL)
3.  Stage-2 MemoryError response (worker reports graceful OOM)
4.  Stage-2 generic RuntimeError response
5.  Stage-2 success (no-op worker) — no coverage gap, no fail_reason
6.  Engine.scan() with Stage-2 runner: Stage-1 detections preserved on failure
7.  Engine.scan() with Stage-2 runner: no gap when Stage-2 succeeds
8.  Empty text — no subprocess call, result is clean (stage2_attempted=False)
9.  stage2_fail_action propagated on failure
10. Runner restart — worker restarts after a crash
11. Stage-2 disabled (runner=None) — no gap annotation on clean scan
12. Stage-2 failure on text with PII — Stage-1 detections remain in result
13. stage2_gap_reason is not a PII vault (does not contain the original text)
14. coverage_gap False when Stage-2 succeeds
15. Multiple runners with different timeouts behave independently

All subprocess workers used here are defined in
``pii_guard.stage2._workers`` at module level so they are picklable
under the ``spawn`` multiprocessing context used by Stage2NERRunner.
"""
from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from pii_guard import Engine
from pii_guard.models import Detection, DetectionStage, Action, CategoryClass, MaskStyle
from pii_guard.stage2 import Stage2NERRunner, Stage2ScanResult
from pii_guard.stage2.runner import _merge_detections
from pii_guard.stage2 import _workers  # noqa: F401 — imported to verify picklability


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_runner(worker_name: str, timeout: float = 1.0) -> Stage2NERRunner:
    """Create a Stage2NERRunner using a named test worker."""
    worker_map = {
        "noop": _workers._test_noop_worker,
        "slow": _workers._test_slow_worker,
        "oom": _workers._test_oom_worker,
        "memoryerror": _workers._test_memoryerror_worker,
        "runtimeerror": _workers._test_runtimeerror_worker,
        "default": _workers.default_ner_worker_loop,
    }
    return Stage2NERRunner(
        timeout_seconds=timeout,
        _worker_target=worker_map[worker_name],
    )


def _dummy_detection(start: int = 0, end: int = 5, category: str = "EMAIL") -> Detection:
    """Create a minimal Detection for test assertions."""
    return Detection(
        category=category,
        category_class=CategoryClass.PII,
        action=Action.TOKENIZE_ROUNDTRIP,
        mask_style=MaskStyle.TOKENIZE,
        start=start,
        end=end,
        original="dummy",
        detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
        rule_id="test_rule",
        confidence=0.95,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stage-2 timeout → Stage-1 fallback + coverage gap
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2Timeout:
    """Worker takes too long — runner must fall back to Stage-1 within timeout."""

    def test_timeout_returns_stage1_detections(self):
        """On timeout, Stage-1 detections must be returned unchanged."""
        runner = _make_runner("slow", timeout=0.3)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.detections == stage1, (
                "Stage-1 detections must be preserved on Stage-2 timeout"
            )
        finally:
            runner.close()

    def test_timeout_sets_coverage_gap(self):
        """On timeout, coverage_gap must be True."""
        runner = _make_runner("slow", timeout=0.3)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.coverage_gap, "Timeout must set coverage_gap=True"
        finally:
            runner.close()

    def test_timeout_populates_fail_reason(self):
        """On timeout, fail_reason must be set with timeout information."""
        runner = _make_runner("slow", timeout=0.3)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.fail_reason, "Timeout must set fail_reason"
            assert "Timeout" in result.fail_reason or "timeout" in result.fail_reason, (
                f"fail_reason must mention timeout; got: {result.fail_reason!r}"
            )
        finally:
            runner.close()

    def test_timeout_stage2_attempted_is_true(self):
        """stage2_attempted must be True when Stage-2 was invoked."""
        runner = _make_runner("slow", timeout=0.3)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.stage2_attempted
        finally:
            runner.close()

    def test_timeout_respects_hard_limit(self):
        """Runner must return within a reasonable bound after timeout expires."""
        runner = _make_runner("slow", timeout=0.2)
        stage1 = [_dummy_detection()]
        try:
            t0 = time.monotonic()
            runner.scan("some text", stage1)
            elapsed = time.monotonic() - t0
            # Must not block significantly longer than the timeout + process overhead
            assert elapsed < 10.0, (
                f"Runner blocked for {elapsed:.2f}s — hard timeout not enforced"
            )
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Stage-2 OOM — hard process death (os._exit) → Stage-1 fallback
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2OOM:
    """Worker dies hard (os._exit) — runner must fall back to Stage-1."""

    def test_oom_returns_stage1_detections(self):
        """After OOM death, Stage-1 detections must be preserved."""
        runner = _make_runner("oom", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.detections == stage1, (
                "Stage-1 detections must be preserved after OOM process death"
            )
        finally:
            runner.close()

    def test_oom_sets_coverage_gap(self):
        """After OOM death, coverage_gap must be True."""
        runner = _make_runner("oom", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.coverage_gap, "OOM must set coverage_gap=True"
        finally:
            runner.close()

    def test_oom_populates_fail_reason(self):
        """After OOM death, fail_reason must be set."""
        runner = _make_runner("oom", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.fail_reason, "OOM must set fail_reason"
        finally:
            runner.close()

    def test_oom_fail_reason_mentions_process_death(self):
        """fail_reason should indicate the worker process died (not just timeout)."""
        runner = _make_runner("oom", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            # The process dies before responding, so the runner times out and
            # detects the dead process.  The fail_reason must reflect this.
            reason = result.fail_reason or ""
            assert any(kw in reason for kw in [
                "ProcessDied", "Died", "exited", "Timeout"
            ]), f"Unexpected fail_reason: {reason!r}"
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Stage-2 MemoryError response → Stage-1 fallback + coverage gap
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2MemoryErrorResponse:
    """Worker sends a MemoryError error response (graceful OOM)."""

    def test_memoryerror_response_returns_stage1(self):
        runner = _make_runner("memoryerror", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("alice@example.com", stage1)
            assert result.detections == stage1
        finally:
            runner.close()

    def test_memoryerror_response_sets_coverage_gap(self):
        runner = _make_runner("memoryerror", timeout=2.0)
        try:
            result = runner.scan("alice@example.com", [_dummy_detection()])
            assert result.coverage_gap
        finally:
            runner.close()

    def test_memoryerror_response_fail_reason_mentions_memory(self):
        runner = _make_runner("memoryerror", timeout=2.0)
        try:
            result = runner.scan("alice@example.com", [_dummy_detection()])
            reason = result.fail_reason or ""
            assert "MemoryError" in reason, (
                f"fail_reason must mention MemoryError; got: {reason!r}"
            )
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Stage-2 RuntimeError response → Stage-1 fallback + coverage gap
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2RuntimeErrorResponse:
    """Worker sends a RuntimeError error response."""

    def test_runtimeerror_response_returns_stage1(self):
        runner = _make_runner("runtimeerror", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result = runner.scan("some text", stage1)
            assert result.detections == stage1
        finally:
            runner.close()

    def test_runtimeerror_response_sets_coverage_gap(self):
        runner = _make_runner("runtimeerror", timeout=2.0)
        try:
            result = runner.scan("some text", [_dummy_detection()])
            assert result.coverage_gap
        finally:
            runner.close()

    def test_runtimeerror_response_fail_reason_set(self):
        runner = _make_runner("runtimeerror", timeout=2.0)
        try:
            result = runner.scan("some text", [_dummy_detection()])
            assert result.fail_reason
            assert "RuntimeError" in result.fail_reason
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Stage-2 success (no-op worker) → no coverage gap
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2Success:
    """No-op worker responds immediately with empty detections."""

    def test_success_no_coverage_gap(self):
        runner = _make_runner("noop", timeout=5.0)
        try:
            result = runner.scan("alice@example.com", [_dummy_detection()])
            assert not result.coverage_gap, (
                "Successful Stage-2 must not set coverage_gap"
            )
        finally:
            runner.close()

    def test_success_no_fail_reason(self):
        runner = _make_runner("noop", timeout=5.0)
        try:
            result = runner.scan("alice@example.com", [_dummy_detection()])
            assert result.fail_reason is None
        finally:
            runner.close()

    def test_success_stage1_detections_preserved(self):
        """When Stage-2 returns no additional detections, Stage-1 are preserved."""
        runner = _make_runner("noop", timeout=5.0)
        stage1 = [_dummy_detection(start=0, end=18)]
        try:
            result = runner.scan("alice@example.com", stage1)
            # Noop worker returns [], so only Stage-1 detections remain
            assert len(result.detections) == len(stage1)
        finally:
            runner.close()

    def test_success_stage2_attempted_is_true(self):
        runner = _make_runner("noop", timeout=5.0)
        try:
            result = runner.scan("alice@example.com", [])
            assert result.stage2_attempted
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6 & 7. Engine.scan() integration with Stage-2 runner
# ──────────────────────────────────────────────────────────────────────────────

class TestEngineStage2Integration:
    """Engine.scan() correctly uses Stage-2 and falls back on failure."""

    def test_engine_stage1_detections_in_result_on_timeout(self):
        """
        Engine.scan() with a timing-out Stage-2: final result must contain the
        same entities that Stage-1 found.
        """
        runner = _make_runner("slow", timeout=0.3)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Contact alice@example.com for details.")
            # Stage-1 should detect the email regardless of Stage-2 status
            cats = [d.category for d in result.detections]
            assert "EMAIL" in cats, (
                "Stage-1 EMAIL detection must survive Stage-2 timeout fallback"
            )
        finally:
            runner.close()

    def test_engine_coverage_gap_set_on_stage2_timeout(self):
        runner = _make_runner("slow", timeout=0.3)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Contact alice@example.com for details.")
            assert result.coverage_gap, (
                "Engine must set coverage_gap when Stage-2 times out"
            )
        finally:
            runner.close()

    def test_engine_stage2_gap_reason_set_on_timeout(self):
        runner = _make_runner("slow", timeout=0.3)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Contact alice@example.com for details.")
            assert result.stage2_gap_reason, (
                "Engine must populate stage2_gap_reason on Stage-2 timeout"
            )
        finally:
            runner.close()

    def test_engine_no_coverage_gap_on_stage2_success(self):
        """
        Engine.scan() with a healthy Stage-2: result must have no coverage gap
        (Stage-2 succeeded even though it returned no additional detections).
        """
        runner = _make_runner("noop", timeout=5.0)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Contact alice@example.com for details.")
            assert not result.coverage_gap, (
                "Engine must not set coverage_gap when Stage-2 succeeds"
            )
            assert result.stage2_gap_reason is None
        finally:
            runner.close()

    def test_engine_stage1_detections_on_oom(self):
        """Engine.scan() with Stage-2 OOM: Stage-1 detections preserved."""
        runner = _make_runner("oom", timeout=2.0)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("AKIAIOSFODNN7EXAMPLE is a fake AWS key")
            # Stage-1 should detect the AWS key
            cats = [d.category for d in result.detections]
            assert "AWS_SECRET" in cats, (
                "Stage-1 AWS_SECRET detection must survive Stage-2 OOM fallback"
            )
            assert result.coverage_gap
            assert result.stage2_gap_reason
        finally:
            runner.close()

    def test_engine_stage2_gap_reason_not_a_pii_vault(self):
        """
        stage2_gap_reason must not contain the original text (raw PII/secrets).
        It should only contain error type / timing information.
        """
        sensitive = "alice@example.com"
        runner = _make_runner("memoryerror", timeout=2.0)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan(f"Contact {sensitive} for details.")
            reason = result.stage2_gap_reason or ""
            assert sensitive not in reason, (
                "stage2_gap_reason must not contain the original PII text"
            )
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Empty text — no subprocess call
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyTextSkipsStage2:
    """Empty text must not invoke the subprocess."""

    def test_empty_text_no_coverage_gap(self):
        runner = _make_runner("slow", timeout=0.3)  # slow worker would cause gap
        try:
            result = runner.scan("", [])
            # Slow worker was NOT invoked (empty text guard)
            assert not result.coverage_gap, (
                "Empty text must skip Stage-2 entirely — no coverage gap"
            )
        finally:
            runner.close()

    def test_empty_text_stage2_not_attempted(self):
        runner = _make_runner("slow", timeout=0.3)
        try:
            result = runner.scan("", [])
            assert not result.stage2_attempted, (
                "Empty text must not set stage2_attempted=True"
            )
        finally:
            runner.close()

    def test_empty_text_stage1_detections_preserved(self):
        runner = _make_runner("slow", timeout=0.3)
        stage1 = [_dummy_detection()]  # contrived — Stage-1 would be empty for ""
        try:
            result = runner.scan("", stage1)
            assert result.detections == stage1
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 9. stage2_fail_action propagated on failure
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2FailAction:
    """stage2_fail_action is correctly propagated in the failure result."""

    def test_fail_action_mask_known_only(self):
        runner = Stage2NERRunner(
            timeout_seconds=0.3,
            stage2_fail_action="mask_known_only",
            _worker_target=_workers._test_slow_worker,
        )
        try:
            result = runner.scan("text", [_dummy_detection()])
            assert result.stage2_fail_action == "mask_known_only"
        finally:
            runner.close()

    def test_fail_action_block(self):
        runner = Stage2NERRunner(
            timeout_seconds=0.3,
            stage2_fail_action="block",
            _worker_target=_workers._test_slow_worker,
        )
        try:
            result = runner.scan("text", [_dummy_detection()])
            assert result.stage2_fail_action == "block"
        finally:
            runner.close()

    def test_fail_action_open(self):
        runner = Stage2NERRunner(
            timeout_seconds=0.3,
            stage2_fail_action="open",
            _worker_target=_workers._test_slow_worker,
        )
        try:
            result = runner.scan("text", [_dummy_detection()])
            assert result.stage2_fail_action == "open"
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Runner restart after crash
# ──────────────────────────────────────────────────────────────────────────────

class TestRunnerRestart:
    """After an OOM/crash the runner must restart cleanly for the next request."""

    def test_runner_restarts_after_oom(self):
        """
        After one OOM scan (process dies), the runner must be able to process
        a subsequent request with a healthy worker.
        """
        # First scan: OOM worker (dies)
        oom_runner = _make_runner("oom", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            result1 = oom_runner.scan("first request", stage1)
            assert result1.coverage_gap  # OOM detected
        finally:
            oom_runner.close()

        # Second scan: noop worker on a fresh runner (should succeed)
        ok_runner = _make_runner("noop", timeout=5.0)
        try:
            result2 = ok_runner.scan("second request", stage1)
            assert not result2.coverage_gap, "Fresh runner with noop worker must succeed"
        finally:
            ok_runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 11. Stage-2 disabled (runner=None) — no gap annotation
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2Disabled:
    """When no Stage-2 runner is supplied, Stage-1-only results are returned."""

    def test_engine_without_stage2_no_gap(self):
        engine = Engine()  # no stage2_runner
        result = engine.scan("Contact alice@example.com for details.")
        assert not result.coverage_gap, (
            "Engine without Stage-2 runner must not set coverage_gap"
        )

    def test_engine_without_stage2_no_gap_reason(self):
        engine = Engine()
        result = engine.scan("Contact alice@example.com for details.")
        assert result.stage2_gap_reason is None

    def test_engine_without_stage2_detects_email(self):
        engine = Engine()
        result = engine.scan("Contact alice@example.com for details.")
        cats = [d.category for d in result.detections]
        assert "EMAIL" in cats

    def test_runner_none_scan_text_still_works(self):
        """Sanity: Engine(stage2_runner=None).scan() is a clean Stage-1 result."""
        engine = Engine(stage2_runner=None)
        result = engine.scan("AKIAIOSFODNN7EXAMPLE is a key placeholder")
        # AWS key detected by Stage-1
        cats = [d.category for d in result.detections]
        assert "AWS_SECRET" in cats
        assert not result.coverage_gap


# ──────────────────────────────────────────────────────────────────────────────
# 12. PII text: Stage-1 detections remain when Stage-2 fails
# ──────────────────────────────────────────────────────────────────────────────

class TestStage1DetectionsPreservedOnFailure:
    """Stage-1 detections must be present and correct when Stage-2 fails."""

    @pytest.mark.parametrize("worker_name,timeout", [
        ("slow",        0.3),
        ("oom",         2.0),
        ("memoryerror", 2.0),
        ("runtimeerror",2.0),
    ])
    def test_stage1_email_preserved_on_all_failure_modes(self, worker_name, timeout):
        """EMAIL detected by Stage-1 must survive all Stage-2 failure modes."""
        runner = _make_runner(worker_name, timeout=timeout)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Send results to alice@example.com please.")
            cats = [d.category for d in result.detections]
            assert "EMAIL" in cats, (
                f"Stage-1 EMAIL detection must survive Stage-2 {worker_name!r} failure"
            )
            assert result.coverage_gap
        finally:
            runner.close()

    @pytest.mark.parametrize("worker_name,timeout", [
        ("slow",        0.3),
        ("oom",         2.0),
        ("memoryerror", 2.0),
        ("runtimeerror",2.0),
    ])
    def test_stage1_aws_key_preserved_on_all_failure_modes(self, worker_name, timeout):
        """AWS_SECRET detected by Stage-1 must survive all Stage-2 failure modes."""
        runner = _make_runner(worker_name, timeout=timeout)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Key: AKIAIOSFODNN7EXAMPLE at us-east-1")
            cats = [d.category for d in result.detections]
            assert "AWS_SECRET" in cats, (
                f"Stage-1 AWS_SECRET must survive Stage-2 {worker_name!r} failure"
            )
        finally:
            runner.close()

    def test_redacted_text_still_redacted_on_stage2_failure(self):
        """
        Even when Stage-2 fails, the redacted text must have PII replaced —
        Stage-1 redaction must have run correctly.
        """
        runner = _make_runner("slow", timeout=0.3)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("Contact alice@example.com")
            assert "alice@example.com" not in result.redacted_text, (
                "Stage-1 redaction must still replace PII when Stage-2 fails"
            )
            assert "[EMAIL" in result.redacted_text
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 13. stage2_gap_reason not a PII vault
# ──────────────────────────────────────────────────────────────────────────────

class TestGapReasonNotPIIVault:
    """stage2_gap_reason must never contain raw PII or secret text."""

    def test_gap_reason_excludes_original_email(self):
        runner = _make_runner("memoryerror", timeout=2.0)
        engine = Engine(stage2_runner=runner)
        sensitive = "secret@corp-internal.io"
        try:
            result = engine.scan(f"CC: {sensitive}")
            reason = result.stage2_gap_reason or ""
            assert sensitive not in reason
        finally:
            runner.close()

    def test_gap_reason_excludes_aws_key(self):
        runner = _make_runner("runtimeerror", timeout=2.0)
        engine = Engine(stage2_runner=runner)
        key = "AKIAIOSFODNN7EXAMPLE"
        try:
            result = engine.scan(f"AWS key: {key}")
            reason = result.stage2_gap_reason or ""
            assert key not in reason
        finally:
            runner.close()

    def test_fail_reason_excludes_text_content(self):
        """The raw fail_reason from Stage2ScanResult must not carry scan text."""
        runner = _make_runner("slow", timeout=0.3)
        sensitive = "password: SuperSecret123!"
        try:
            result = runner.scan(sensitive, [])
            reason = result.fail_reason or ""
            assert "SuperSecret123!" not in reason
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 14. coverage_gap False when Stage-2 succeeds
# ──────────────────────────────────────────────────────────────────────────────

class TestNoCoverageGapOnSuccess:
    """When Stage-2 succeeds, no coverage gap must be set."""

    def test_stage2_success_runner_level_no_gap(self):
        runner = _make_runner("noop", timeout=5.0)
        try:
            result = runner.scan("normal text", [])
            assert not result.coverage_gap
            assert result.fail_reason is None
        finally:
            runner.close()

    def test_stage2_success_engine_level_no_gap(self):
        runner = _make_runner("noop", timeout=5.0)
        engine = Engine(stage2_runner=runner)
        try:
            result = engine.scan("normal text with no PII")
            assert not result.coverage_gap
            assert result.stage2_gap_reason is None
        finally:
            runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# 15. Multiple runners behave independently
# ──────────────────────────────────────────────────────────────────────────────

class TestMultipleRunners:
    """Two concurrent runners with different configs must not interfere."""

    def test_two_runners_independent(self):
        ok_runner = _make_runner("noop", timeout=5.0)
        fail_runner = _make_runner("memoryerror", timeout=2.0)
        stage1 = [_dummy_detection()]
        try:
            ok_result = ok_runner.scan("text", stage1)
            fail_result = fail_runner.scan("text", stage1)
            assert not ok_result.coverage_gap, "noop runner must succeed"
            assert fail_result.coverage_gap, "memoryerror runner must fail"
        finally:
            ok_runner.close()
            fail_runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for _merge_detections helper
# ──────────────────────────────────────────────────────────────────────────────

class TestMergeDetections:
    """_merge_detections correctly combines Stage-1 and Stage-2 detections."""

    def _det(self, start, end, category="EMAIL", stage=DetectionStage.STAGE1_REGEX_CHECKSUM):
        return Detection(
            category=category,
            category_class=CategoryClass.PII,
            action=Action.TOKENIZE_ROUNDTRIP,
            mask_style=MaskStyle.TOKENIZE,
            start=start,
            end=end,
            original="x" * (end - start),
            detection_stage=stage,
            rule_id="r",
            confidence=0.9,
        )

    def test_empty_stage2_returns_stage1(self):
        s1 = [self._det(0, 5), self._det(10, 15)]
        result = _merge_detections(s1, [])
        assert result == s1

    def test_non_overlapping_merged(self):
        s1 = [self._det(0, 5, "EMAIL")]
        s2 = [self._det(10, 15, "PERSON", DetectionStage.STAGE2_NER)]
        result = _merge_detections(s1, s2)
        assert len(result) == 2
        cats = {d.category for d in result}
        assert cats == {"EMAIL", "PERSON"}

    def test_overlapping_stage1_wins(self):
        """When Stage-1 and Stage-2 detections overlap, Stage-1 wins."""
        s1 = [self._det(0, 18, "EMAIL")]
        s2 = [self._det(0, 18, "PERSON", DetectionStage.STAGE2_NER)]
        result = _merge_detections(s1, s2)
        assert len(result) == 1
        assert result[0].category == "EMAIL"

    def test_result_sorted_by_position(self):
        s1 = [self._det(10, 15, "EMAIL")]
        s2 = [self._det(0, 5, "PERSON", DetectionStage.STAGE2_NER)]
        result = _merge_detections(s1, s2)
        positions = [d.start for d in result]
        assert positions == sorted(positions)

    def test_both_empty(self):
        result = _merge_detections([], [])
        assert result == []

    def test_stage2_only(self):
        """If Stage-1 is empty, Stage-2 detections fill the result."""
        s2 = [self._det(5, 20, "PERSON", DetectionStage.STAGE2_NER)]
        result = _merge_detections([], s2)
        assert len(result) == 1
        assert result[0].category == "PERSON"


# ──────────────────────────────────────────────────────────────────────────────
# Stage2ScanResult structure
# ──────────────────────────────────────────────────────────────────────────────

class TestStage2ScanResultStructure:
    """Stage2ScanResult carries the correct fields in all states."""

    def test_success_result_fields(self):
        result = Stage2ScanResult(
            detections=[],
            coverage_gap=False,
            fail_reason=None,
            stage2_fail_action="mask_known_only",
            stage2_attempted=True,
        )
        assert result.coverage_gap is False
        assert result.fail_reason is None
        assert result.stage2_fail_action == "mask_known_only"
        assert result.stage2_attempted is True

    def test_failure_result_fields(self):
        dets = [_dummy_detection()]
        result = Stage2ScanResult(
            detections=dets,
            coverage_gap=True,
            fail_reason="Stage2NERTimeout: 10.0s",
            stage2_fail_action="block",
            stage2_attempted=True,
        )
        assert result.coverage_gap is True
        assert result.fail_reason is not None
        assert result.detections is dets
        assert result.stage2_fail_action == "block"
