"""
pii_guard/stage2/policy_layer.py

Stage-2 policy integration layer (Sub-AC 10.2).

Confidence threshold → policy decision integration for Stage-2 NER results.

Architecture
------------
``Stage2PolicyLayer`` wraps :class:`~pii_guard.decision.PolicyDecisionEngine`
and provides a Stage-2-specific API::

    raw_ner_detections = ner_engine.detect(text)      # from KoreanNEREngine
    layer = Stage2PolicyLayer(config=policy.config)
    result = layer.apply(raw_ner_detections)
    # result.enforced   → detections to mask/block (above threshold)
    # result.suppressed → detections below threshold (confidence too low)

Per-entity-type thresholds
--------------------------
Thresholds are read from ``PolicyConfig.categories[entity_type].min_confidence``.
When no per-entity override exists, the built-in ``CategorySpec.min_confidence``
is used (from ``categories.py``)::

    - PERSON:       0.80  (built-in default; override with policy YAML)
    - ADDRESS:      0.75  (built-in default)
    - ORGANIZATION: 0.70  (built-in default)

Example policy YAML overrides::

    categories:
      PERSON:
        min_confidence: 0.75   # NER PERSON detections below 0.75 are ignored
      ADDRESS:
        min_confidence: 0.85   # stricter for address entities
      ORGANIZATION:
        min_confidence: 0.70
        action: tokenize_roundtrip

Decision mapping
----------------
For each NER detection the policy engine applies this precedence chain:

  1. confidence < effective_min_confidence           →  suppress (allow/pass)
  2. pin-list HMAC match (if hmac_key supplied)      →  pin-list action
  3. per-category policy override (action, mask_style)
  4. CategorySpec built-in default action
  5. Unknown category                                →  block (secure default)

This module has no subprocess or I/O dependencies — it can be called directly
in unit tests with injected (mocked) NER results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..decision import PolicyDecision, PolicyDecisionEngine
from ..models import Action, Detection, MaskStyle
from ..policy import SECURE_DEFAULTS, PolicyConfig


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage2PolicyResult:
    """
    Result of applying policy decisions to a batch of Stage-2 NER detections.

    Attributes
    ----------
    enforced:
        Detections whose confidence met the effective threshold and whose
        resolved action is non-allow (mask, block, or tokenize_roundtrip).
        The detection's ``action`` and ``mask_style`` fields are mutated
        in-place to reflect the policy decision.
    suppressed:
        Detections whose confidence was below the effective per-entity
        threshold — these are ignored (policy action = allow / pass).
    decisions:
        Full per-detection :class:`~pii_guard.decision.PolicyDecision` objects
        keyed by the detection's position index in the original input list.
        Covers ALL input detections (both enforced and suppressed) for audit
        and diagnostic purposes.

    Notes
    -----
    ``total`` is a convenience property returning ``len(enforced) + len(suppressed)``.
    """

    enforced: List[Detection] = field(default_factory=list)
    suppressed: List[Detection] = field(default_factory=list)
    decisions: Dict[int, PolicyDecision] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total number of input detections processed."""
        return len(self.enforced) + len(self.suppressed)

    @property
    def enforcement_rate(self) -> float:
        """Fraction of detections that passed threshold (enforced / total)."""
        t = self.total
        return len(self.enforced) / t if t > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Policy layer
# ──────────────────────────────────────────────────────────────────────────────

class Stage2PolicyLayer:
    """
    Maps raw Stage-2 NER detection results through the policy decision engine.

    Given raw NER detections with confidence scores (e.g., from
    :class:`~pii_guard.stage2.korean_ner.KoreanNEREngine`), applies
    per-entity-type confidence thresholds and policy actions to produce
    final mask / block / pass decisions.

    This class is designed for **pure policy evaluation**:

    * No subprocess communication.
    * No file I/O.
    * Fully injectable inputs — tests can construct synthetic
      :class:`~pii_guard.models.Detection` objects and verify the policy
      output without running the real NER model.

    Parameters
    ----------
    config:
        Live policy config from :class:`~pii_guard.policy.PolicyLoader`.
        Defaults to :data:`~pii_guard.policy.SECURE_DEFAULTS`.
    channel:
        Originating channel name (e.g. ``"cli"``, ``"ouroboros"``).  Used
        for channel-level policy overrides.  ``None`` / ``""`` disables
        channel overrides.
    hmac_key:
        HMAC-SHA256 key for pin-list matching.  When ``None``, pin-list
        comparison is skipped.  Passed through to
        :class:`~pii_guard.decision.PolicyDecisionEngine`.

    Usage
    -----
    ::

        from pii_guard.stage2.policy_layer import Stage2PolicyLayer
        from pii_guard.policy import PolicyConfig, CategoryPolicy

        # Configure stricter threshold for PERSON entities
        cfg = PolicyConfig(categories={
            "PERSON": CategoryPolicy(min_confidence=0.85),
        })
        layer = Stage2PolicyLayer(config=cfg)

        # Inject synthetic NER detections (or use real ones)
        result = layer.apply(ner_detections)
        for det in result.enforced:
            print(f"  Enforce: {det.category} → {det.action.value}")
        for det in result.suppressed:
            print(f"  Suppress (below threshold): {det.category}")
    """

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        channel: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self._config: PolicyConfig = config if config is not None else SECURE_DEFAULTS
        self._decision_engine = PolicyDecisionEngine(
            config=self._config,
            channel=channel,
            hmac_key=hmac_key,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, ner_detections: List[Detection]) -> Stage2PolicyResult:
        """
        Apply policy decisions to a batch of raw NER detections.

        Decision pipeline per detection:

        1. **Confidence threshold check**: compute the effective
           ``min_confidence`` for the entity's category (per-category policy
           override, falling back to :class:`~pii_guard.categories.CategorySpec`
           built-in default, then 0.0 for unknown categories).
           If ``detection.confidence < effective_min`` → suppress.

        2. **Pin-list match** (if ``hmac_key`` was provided): if the HMAC of
           the normalised original matches a pin-list entry for the category,
           use the pin-list's action (may be ``allow``, ``block``, ``mask``, or
           ``tokenize_roundtrip``).

        3. **Per-category policy override**: if ``PolicyConfig.categories[cat]``
           has an explicit ``action`` / ``mask_style``, use those.

        4. **CategorySpec built-in default**: action / mask_style from
           ``categories.py``.

        5. **Unknown category** → block (secure default for unrecognised
           categories that lack a ``CategorySpec`` entry).

        Detections whose resolved action is non-allow have their ``action``
        and ``mask_style`` fields mutated in-place to reflect the policy
        decision.

        Parameters
        ----------
        ner_detections:
            Raw NER detections from Stage-2.  May be an empty list — returns a
            clean empty :class:`Stage2PolicyResult` without raising.

        Returns
        -------
        Stage2PolicyResult
            Partitioned into:

            * ``enforced`` — detections to mask/block (above threshold).
            * ``suppressed`` — detections below threshold (ignored / allowed).
            * ``decisions`` — per-detection :class:`~pii_guard.decision.PolicyDecision`
              map for audit / diagnostics.
        """
        result = Stage2PolicyResult()

        for idx, det in enumerate(ner_detections):
            decision = self._decision_engine.decide(det)
            result.decisions[idx] = decision

            if decision.is_allow:
                # Below threshold or explicitly allowed → suppress
                result.suppressed.append(det)
            else:
                # Mutate detection in-place to reflect resolved policy action
                det.action = Action(decision.action)
                det.mask_style = MaskStyle(decision.mask_style)
                result.enforced.append(det)

        return result

    def decide_single(self, detection: Detection) -> PolicyDecision:
        """
        Return the raw :class:`~pii_guard.decision.PolicyDecision` for a single
        detection without mutating the detection or building a batch result.

        Useful for diagnostics, logging, and per-detection unit tests.

        Parameters
        ----------
        detection:
            A :class:`~pii_guard.models.Detection` from Stage-2 NER.

        Returns
        -------
        PolicyDecision
            Fully resolved decision — never raises.
        """
        return self._decision_engine.decide(detection)

    @property
    def config(self) -> PolicyConfig:
        """Return the :class:`~pii_guard.policy.PolicyConfig` in use."""
        return self._config
