"""
pii_guard/stage2/runner.py

Stage2 NER subprocess runner with per-block hard timeout and OOM isolation.

Architecture
------------
``Stage2NERRunner`` manages a long-lived subprocess (``multiprocessing.Process``
with the ``spawn`` context for macOS safety) that runs NER on text blocks.  The
runner communicates with the worker via a pair of ``multiprocessing.Queue``
objects (request and response), and enforces a configurable per-block hard
timeout.

Failure modes and degradation
------------------------------
In all failure cases the runner falls back to the Stage-1 detections that were
passed in by the caller, marks the result as a coverage gap, and records a
human-readable ``fail_reason``.  The forwarding core is never blocked by a
Stage-2 failure.

+-------------------------------+--------------------------------------------------+
| Failure                       | Detection                                        |
+===============================+==================================================+
| Worker timeout (stuck model)  | ``resp_q.get(timeout=N)`` raises ``queue.Empty`` |
+-------------------------------+--------------------------------------------------+
| Worker OOM / SIGKILL          | Process exits; ``resp_q.get`` times out then     |
|                               | ``process.is_alive()`` is False                  |
+-------------------------------+--------------------------------------------------+
| Worker raises MemoryError     | Worker catches + sends ``("error", "MemoryError:…")`` |
+-------------------------------+--------------------------------------------------+
| Worker raises any Exception   | Worker catches + sends ``("error", "…")``        |
+-------------------------------+--------------------------------------------------+
| Worker process start failure  | Caught in ``_ensure_worker_alive()``             |
+-------------------------------+--------------------------------------------------+

Usage
-----
    from pii_guard.stage2 import Stage2NERRunner
    from pii_guard import Engine

    runner = Stage2NERRunner(timeout_seconds=10.0)
    engine = Engine(stage2_runner=runner)
    result = engine.scan("Contact alice@example.com for AWS key AKIAIOSFODNN7EXAMPLE")
    # result.coverage_gap is True if Stage2 failed; Stage1 detections are preserved.
    runner.close()
"""
from __future__ import annotations

import multiprocessing
import queue as _queue_module
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..models import Detection
from . import _workers


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage2ScanResult:
    """
    Result from a Stage2 NER scan attempt.

    Carries the authoritative detection list to be used downstream:

    - **Success**: Stage-1 + Stage-2 detections merged (overlap-resolved).
    - **Failure**: Stage-1 detections only (unchanged fallback).

    Parameters
    ----------
    detections:
        Authoritative list of :class:`~pii_guard.models.Detection` objects
        after merging (or Stage-1-only on failure).
    coverage_gap:
        ``True`` when Stage-2 was attempted but failed.  The caller should
        log this as a coverage gap in the audit ledger.
    fail_reason:
        Human-readable description of why Stage-2 failed.  ``None`` on
        success.  Maps to ``ledger_event.fail_reason`` in the ontology.
    stage2_fail_action:
        Policy action taken on Stage-2 failure (from ontology:
        ``mask_known_only``, ``block``, or ``open``).
    stage2_attempted:
        ``True`` when Stage-2 was actually invoked (even if it failed).
    """

    detections: List[Detection]
    coverage_gap: bool
    fail_reason: Optional[str] = None
    stage2_fail_action: str = "mask_known_only"
    stage2_attempted: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

class Stage2NERRunner:
    """
    Manages a Stage-2 NER subprocess with hard per-block timeout and OOM isolation.

    The subprocess is started lazily on the first ``scan()`` call and restarted
    automatically if it dies (OOM or crash).  A clean shutdown can be requested
    via ``close()``.

    Parameters
    ----------
    timeout_seconds:
        Per-block hard timeout in seconds.  When the subprocess does not
        respond within this limit the runner falls back to Stage-1.
    stage2_fail_action:
        Ontology ``stage2_fail_action``:
        ``"mask_known_only"`` (default) — use Stage-1 detections only;
        ``"block"`` — the caller must block the request on failure;
        ``"open"``  — allow the request through with just a gap annotation.
    _worker_target:
        Override the subprocess worker function.  **For testing only.**
        Must be a module-level callable with signature
        ``(req_q: Queue, resp_q: Queue) -> None``.
    """

    DEFAULT_TIMEOUT: float = 10.0

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT,
        stage2_fail_action: str = "mask_known_only",
        _worker_target: Optional[Callable] = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._stage2_fail_action = stage2_fail_action
        self._worker_target = _worker_target or _workers.default_ner_worker_loop

        # Use the "spawn" start method — required on macOS and safe on Linux.
        # "fork" can deadlock when the parent uses threads or complex library state.
        self._mp_ctx = multiprocessing.get_context("spawn")

        # Subprocess state — initialised lazily
        self._process: Optional[multiprocessing.Process] = None
        self._req_q: Optional[multiprocessing.Queue] = None
        self._resp_q: Optional[multiprocessing.Queue] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        text: str,
        stage1_detections: List[Detection],
    ) -> Stage2ScanResult:
        """
        Run Stage-2 NER on *text* with timeout/OOM protection.

        Always returns a :class:`Stage2ScanResult`.  On any failure the result
        carries the original *stage1_detections* as the authoritative list and
        ``coverage_gap=True`` with a populated ``fail_reason``.

        Parameters
        ----------
        text:
            Original (pre-redaction) text to scan.
        stage1_detections:
            Detections from Stage-1 regex/checksum.  Used as the fallback on
            Stage-2 failure and as the overlap baseline when merging.

        Returns
        -------
        Stage2ScanResult
        """
        if not text:
            # Empty text — Stage-1 result is already complete; no subprocess call.
            return Stage2ScanResult(
                detections=stage1_detections,
                coverage_gap=False,
                stage2_attempted=False,
            )

        # Ensure the worker process is alive (start/restart as needed)
        try:
            self._ensure_worker_alive()
        except Exception as exc:  # noqa: BLE001
            return self._make_failure(
                stage1_detections,
                f"Stage2WorkerStartError: {type(exc).__name__}: {exc}",
            )

        # ── Send request ──────────────────────────────────────────────────────
        try:
            self._req_q.put(text, block=True, timeout=2.0)
        except _queue_module.Full:
            # Worker is still processing a previous request (should not happen
            # with maxsize=1 and sequential scan calls, but guard anyway)
            return self._make_failure(
                stage1_detections,
                "Stage2QueueFull: worker busy — skipping Stage-2 for this block",
            )
        except Exception as exc:  # noqa: BLE001
            self._reset_worker()
            return self._make_failure(
                stage1_detections,
                f"Stage2SendError: {type(exc).__name__}: {exc}",
            )

        # ── Wait for response with hard timeout ───────────────────────────────
        try:
            response = self._resp_q.get(block=True, timeout=self._timeout)
        except _queue_module.Empty:
            # Timeout — determine if the worker process also died (OOM/SIGKILL)
            reason = self._diagnose_timeout_reason()
            self._reset_worker()
            return self._make_failure(stage1_detections, reason)
        except (EOFError, OSError, ConnectionResetError) as exc:
            # Pipe broken — worker process died unexpectedly
            self._reset_worker()
            return self._make_failure(
                stage1_detections,
                f"Stage2WorkerDied: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            self._reset_worker()
            return self._make_failure(
                stage1_detections,
                f"Stage2ReceiveError: {type(exc).__name__}: {exc}",
            )

        # ── Parse response ─────────────────────────────────────────────────────
        if not isinstance(response, tuple) or len(response) != 2:
            self._reset_worker()
            return self._make_failure(
                stage1_detections,
                "Stage2ProtocolError: malformed response from worker",
            )

        status, data = response

        if status == "ok":
            # Success — merge Stage-1 + Stage-2 detections
            stage2_detections: List[Detection] = data if isinstance(data, list) else []
            merged = _merge_detections(stage1_detections, stage2_detections)
            return Stage2ScanResult(
                detections=merged,
                coverage_gap=False,
                fail_reason=None,
                stage2_attempted=True,
            )

        if status == "error":
            # Worker reported an error (MemoryError, RuntimeError, etc.)
            error_msg: str = data if isinstance(data, str) else str(data)
            return self._make_failure(
                stage1_detections,
                f"Stage2WorkerError: {error_msg}",
            )

        return self._make_failure(
            stage1_detections,
            f"Stage2ProtocolError: unknown status {status!r}",
        )

    def close(self) -> None:
        """
        Cleanly shut down the worker subprocess.

        Sends a ``None`` shutdown signal, waits up to 2 s for a clean exit,
        then terminates/kills the process if it has not exited.
        """
        self._reset_worker(send_shutdown=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_worker_alive(self) -> None:
        """Start (or restart) the worker process if it is not running."""
        if self._process is not None and self._process.is_alive():
            return

        # Previous process is dead (OOM, crash) — clean up before restarting.
        if self._process is not None:
            try:
                self._process.kill()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._process.join(timeout=0.5)
            except Exception:  # noqa: BLE001
                pass

        self._req_q = self._mp_ctx.Queue(maxsize=1)
        self._resp_q = self._mp_ctx.Queue(maxsize=1)
        self._process = self._mp_ctx.Process(
            target=self._worker_target,
            args=(self._req_q, self._resp_q),
            daemon=True,
        )
        self._process.start()

    def _diagnose_timeout_reason(self) -> str:
        """
        Build a failure reason string after a response-queue timeout.

        If the worker process has also died, the reason reflects probable OOM
        (process killed) rather than a mere hang.
        """
        if self._process is not None and not self._process.is_alive():
            code = self._process.exitcode
            return (
                f"Stage2NERTimeout+ProcessDied: worker process exited with code "
                f"{code} after {self._timeout:.1f}s (likely OOM / SIGKILL)"
            )
        return (
            f"Stage2NERTimeout: worker did not respond within "
            f"{self._timeout:.1f}s hard timeout"
        )

    def _reset_worker(self, *, send_shutdown: bool = False) -> None:
        """Terminate the worker process and clear all subprocess state."""
        if self._process is not None:
            if send_shutdown and self._req_q is not None:
                try:
                    self._req_q.put(None, block=False)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._process.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._process.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            if self._process.is_alive():
                try:
                    self._process.kill()
                    self._process.join(timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
        self._process = None
        self._req_q = None
        self._resp_q = None

    def _make_failure(
        self,
        stage1_detections: List[Detection],
        reason: str,
    ) -> Stage2ScanResult:
        """Return a coverage-gap failure result with Stage-1 detections."""
        return Stage2ScanResult(
            detections=stage1_detections,
            coverage_gap=True,
            fail_reason=reason,
            stage2_fail_action=self._stage2_fail_action,
            stage2_attempted=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Detection merge helper
# ──────────────────────────────────────────────────────────────────────────────

def _merge_detections(
    stage1: List[Detection],
    stage2: List[Detection],
) -> List[Detection]:
    """
    Merge Stage-1 and Stage-2 detections, resolving overlaps.

    Stage-1 detections have priority — they are already validated by
    regex/checksum and appear first in the combined list.  Stage-2 NER
    detections fill gaps not covered by Stage-1.

    Parameters
    ----------
    stage1:
        Detections from Stage-1 regex/checksum scanning.
    stage2:
        Additional detections from Stage-2 NER (may be empty).

    Returns
    -------
    List[Detection] sorted by start position, with overlaps resolved.
    """
    if not stage2:
        return list(stage1)

    # Combine; Stage-1 items come first so they win position ties.
    all_dets = list(stage1) + list(stage2)

    # Sort: earlier position first, then longer span wins among ties.
    # Stage-1 items appear before Stage-2 for the same (start, length) because
    # Python's sort is stable and they were listed first.
    all_dets.sort(key=lambda d: (d.start, -(d.end - d.start)))

    kept: List[Detection] = []
    occupied: List[tuple] = []

    for det in all_dets:
        overlap = any(
            not (det.end <= s or det.start >= e)
            for s, e in occupied
        )
        if not overlap:
            kept.append(det)
            occupied.append((det.start, det.end))

    kept.sort(key=lambda d: d.start)
    return kept
