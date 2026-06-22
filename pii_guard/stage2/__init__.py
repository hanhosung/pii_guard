"""
pii_guard.stage2 — Stage-2 NER subprocess runner.

Provides :class:`Stage2NERRunner` for subprocess-isolated NER detection with
per-block hard timeout and graceful fallback to Stage-1 on failure.

Exports
-------
Stage2NERRunner
    Manages the NER subprocess and exposes :meth:`~Stage2NERRunner.scan`.
Stage2ScanResult
    Result dataclass carrying the authoritative detection list and coverage-gap
    metadata.

Quick example::

    from pii_guard.stage2 import Stage2NERRunner
    from pii_guard import Engine

    runner = Stage2NERRunner(timeout_seconds=10.0)
    engine = Engine(stage2_runner=runner)
    result = engine.scan("Contact alice@example.com")
    # result.coverage_gap is True if Stage2 NER failed; Stage1 results preserved.
    runner.close()
"""
from .runner import Stage2NERRunner, Stage2ScanResult

__all__ = ["Stage2NERRunner", "Stage2ScanResult"]
