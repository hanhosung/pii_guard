"""
pii_guard.stage2 — Stage-2 NER subprocess runner and policy integration layer.

Provides :class:`Stage2NERRunner` for subprocess-isolated NER detection with
per-block hard timeout and graceful fallback to Stage-1 on failure, plus
:class:`Stage2PolicyLayer` for confidence threshold → policy decision mapping.

Exports
-------
Stage2NERRunner
    Manages the NER subprocess and exposes :meth:`~Stage2NERRunner.scan`.
Stage2ScanResult
    Result dataclass carrying the authoritative detection list and coverage-gap
    metadata.
Stage2PolicyLayer
    Maps raw NER detections through the policy decision engine; applies
    per-entity-type confidence thresholds to produce mask/block/pass decisions.
Stage2PolicyResult
    Result dataclass from ``Stage2PolicyLayer.apply()`` with enforced and
    suppressed detection lists plus per-detection policy decisions.

Quick example::

    from pii_guard.stage2 import Stage2NERRunner, Stage2PolicyLayer
    from pii_guard import Engine
    from pii_guard.policy import PolicyConfig, CategoryPolicy

    # Configure per-entity-type thresholds
    cfg = PolicyConfig(categories={
        "PERSON": CategoryPolicy(min_confidence=0.85),
        "ORGANIZATION": CategoryPolicy(min_confidence=0.70, action="tokenize_roundtrip"),
    })
    layer = Stage2PolicyLayer(config=cfg)
    result = layer.apply(ner_detections)  # inject synthetic or real NER results
    # result.enforced  → detections to mask/block
    # result.suppressed → below threshold, pass through

    runner = Stage2NERRunner(timeout_seconds=10.0)
    engine = Engine(stage2_runner=runner)
    scan_result = engine.scan("Contact alice@example.com")
    runner.close()
"""
from .policy_layer import Stage2PolicyLayer, Stage2PolicyResult
from .runner import Stage2NERRunner, Stage2ScanResult

__all__ = [
    "Stage2NERRunner",
    "Stage2ScanResult",
    "Stage2PolicyLayer",
    "Stage2PolicyResult",
]
