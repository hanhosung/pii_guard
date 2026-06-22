"""
pii_guard/stage2/_workers.py

Subprocess worker functions for Stage2 NER.

All functions here are defined at module level so they are picklable for use
as ``multiprocessing.Process`` targets with the ``spawn`` start method (which
is the macOS default and required for safe subprocess creation).

Protocol
--------
  Request:  ``req_q.put(text: str | None)``  — ``None`` is a shutdown signal.
  Response: ``resp_q.put(("ok", List[Detection]) | ("error", str))``

Worker types
------------
``default_ner_worker_loop``
    Production stub — returns empty detections (Stage-1 already covered the
    high-recall pass).  Replace the body of ``_run_ner()`` with a real spaCy
    or HuggingFace NER call when a model is available.

``_test_noop_worker``
    Test helper — responds immediately with an empty list.

``_test_slow_worker``
    Test helper — receives the request but never sends a response, simulating
    a worker that is stuck (e.g. model inference hung).  Used to exercise the
    hard per-block timeout.

``_test_oom_worker``
    Test helper — simulates a process killed by the OS (OOM / SIGKILL) by
    calling ``os._exit(1)`` without sending a response.  The runner detects
    the dead process via the response-queue timeout + liveness check.

``_test_memoryerror_worker``
    Test helper — sends an error response carrying "MemoryError" text.  This
    simulates a *graceful* MemoryError caught inside the worker (the process
    itself survives but reports it cannot handle the request).

``_test_runtimeerror_worker``
    Test helper — sends a generic RuntimeError response.
"""
from __future__ import annotations

import queue as _queue


# ──────────────────────────────────────────────────────────────────────────────
# Production worker
# ──────────────────────────────────────────────────────────────────────────────

def default_ner_worker_loop(req_q, resp_q) -> None:  # pragma: no cover
    """
    Production Stage2 NER worker loop.

    Stub implementation: no NER model loaded, returns empty detections.
    To integrate a real model, replace ``_run_ner()`` with your model call.
    The function runs in a long-lived subprocess; import the model once (lazily)
    before the loop begins to amortise start-up cost across requests.
    """

    def _run_ner(text: str):
        # ── Integration point ─────────────────────────────────────────────────
        # Example (spaCy):
        #   import spacy
        #   nlp = spacy.load("en_core_web_sm")
        #   doc = nlp(text)
        #   return [_ent_to_detection(ent) for ent in doc.ents]
        return []  # stub — Stage-1 regex/checksum is the only active stage

    while True:
        try:
            item = req_q.get(timeout=60)
        except _queue.Empty:
            # Idle for 60 s — exit cleanly so the process does not linger
            break
        if item is None:
            break  # explicit shutdown signal

        try:
            detections = _run_ner(item)
            resp_q.put(("ok", detections))
        except MemoryError as exc:
            resp_q.put(("error", f"MemoryError: {exc}"))
        except Exception as exc:  # noqa: BLE001
            resp_q.put(("error", f"{type(exc).__name__}: {exc}"))


# ──────────────────────────────────────────────────────────────────────────────
# Test workers  ── DO NOT USE IN PRODUCTION
# ──────────────────────────────────────────────────────────────────────────────

def _test_noop_worker(req_q, resp_q) -> None:
    """Test worker: returns an empty detection list immediately."""
    while True:
        try:
            item = req_q.get(timeout=60)
        except _queue.Empty:
            break
        if item is None:
            break
        resp_q.put(("ok", []))


def _test_slow_worker(req_q, resp_q) -> None:
    """
    Test worker: receives the request but never responds.

    Simulates a worker whose model inference has hung.  Used to exercise the
    runner's hard per-block timeout.  The runner will time out, kill this
    process, and fall back to Stage-1 detections.
    """
    import time as _time

    try:
        req_q.get(timeout=60)  # receive the request (so the runner's put() succeeds)
    except _queue.Empty:
        return
    # Sleep forever — the runner's resp_q.get(timeout=N) will raise queue.Empty
    _time.sleep(9_999)


def _test_oom_worker(req_q, resp_q) -> None:
    """
    Test worker: simulates OOM process death by calling os._exit().

    Models the OS killing the subprocess with SIGKILL when it exceeds its
    memory budget.  The process exits without sending any response; the runner
    detects the dead process after the response-queue timeout fires.
    """
    import os as _os

    try:
        req_q.get(timeout=60)
    except _queue.Empty:
        return
    # Hard exit — no cleanup, no response sent.  Exit code 1 (mimics SIGKILL).
    _os._exit(1)


def _test_memoryerror_worker(req_q, resp_q) -> None:
    """
    Test worker: sends a MemoryError error response.

    Models a graceful MemoryError caught inside the worker (the process itself
    survives but cannot handle the request).
    """
    try:
        req_q.get(timeout=60)
    except _queue.Empty:
        return
    resp_q.put(("error", "MemoryError: cannot allocate NER model weights"))


def _test_runtimeerror_worker(req_q, resp_q) -> None:
    """Test worker: sends a generic RuntimeError error response."""
    try:
        req_q.get(timeout=60)
    except _queue.Empty:
        return
    resp_q.put(("error", "RuntimeError: NER model inference failed"))
