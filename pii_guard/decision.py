"""
PII-Guard Policy Decision Engine (Sub-AC 5c-i).

Given a :class:`~pii_guard.models.Detection` and the live
:class:`~pii_guard.policy.PolicyConfig`, resolves the final per-token
decision: **allow / mask / block / tokenize_roundtrip** — including which
mask style to apply.

Decision precedence for :meth:`PolicyDecisionEngine.decide` (highest → lowest)
-------------------------------------------------------------------------------
1. **Confidence threshold** — if ``detection.confidence <
   effective_min_confidence`` → ``allow`` (ignore the detection).
   *effective_min_confidence* is the per-category policy override if set,
   otherwise the CategorySpec built-in default.
2. **Pin-list HMAC match** (when *hmac_key* is supplied) — if the HMAC-SHA256
   of the normalised original matches a pin-list entry for the same category,
   the pin-list's action is used (``allow``, ``block``, ``mask``, or
   ``tokenize_roundtrip``).
3. **Per-category policy override** — if
   ``PolicyConfig.categories[category].action`` is set, it overrides the
   built-in default.  Similarly for ``mask_style``.
4. **CategorySpec built-in default** — the action and mask_style from the
   original category definition in ``categories.py``.
5. **Unknown category** → ``block`` (secure default; unknown categories are
   never silently allowed).

Failure / degrade decision precedence (highest → lowest)
---------------------------------------------------------
For all ``decide_*`` failure methods:

  ``stage2_fail_action``  : channel_override > category_policy_override > global
  ``on_infra_failure``    : global only (no channel-level override defined)
  ``on_content_failure``  : channel_override > global
  ``unscannable_action``  : channel_override > global
  ``fail_mode``           : channel_override > global
"""
from __future__ import annotations

import hashlib
import hmac as _hmac_module
from dataclasses import dataclass
from typing import Optional

from .categories import CATEGORY_MAP, CategorySpec
from .models import Action, Detection, MaskStyle
from .policy import SECURE_DEFAULTS, PolicyConfig


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyDecision:
    """
    Resolved action for a single detection.

    Attributes
    ----------
    action : str
        Final action — ``"allow"``, ``"mask"``, ``"block"``, or
        ``"tokenize_roundtrip"``.
    mask_style : str
        Resolved mask style — ``"tokenize"``, ``"partial"``, or
        ``"format_preserving"``.  Meaningful when *action* is ``"mask"``
        or ``"tokenize_roundtrip"``; present but ignored when *action* is
        ``"allow"`` or ``"block"``.
    effective_min_confidence : float
        The minimum confidence threshold that was applied.
    reason : str
        Non-PII reason string explaining which rule path produced this
        decision (for audit / diagnostics).  Examples:
        ``"below_min_confidence"``, ``"pin_list:relay"``,
        ``"category_override:EMAIL"``, ``"category_default:API_KEY"``,
        ``"unknown_category:CUSTOM_X"``.
    detector_id : str
        ``rule_id`` from the triggering detection (pass-through for ledger
        correlation).

    Properties
    ----------
    is_allow, is_block, is_mask — convenience predicates.
    """
    action: str
    mask_style: str
    effective_min_confidence: float
    reason: str
    detector_id: str = ""

    @property
    def is_allow(self) -> bool:
        """True when the resolved action is ``allow``."""
        return self.action == Action.ALLOW.value

    @property
    def is_block(self) -> bool:
        """True when the resolved action is ``block``."""
        return self.action == Action.BLOCK.value

    @property
    def is_mask(self) -> bool:
        """True when the resolved action is ``mask`` or ``tokenize_roundtrip``."""
        return self.action in (Action.MASK.value, Action.TOKENIZE_ROUNDTRIP.value)


@dataclass
class FailureDecision:
    """
    Resolved decision for an infrastructure / content failure event.

    Attributes
    ----------
    action : str
        Failure action string appropriate for the event type — e.g.
        ``"block"``, ``"warn_allow"``, ``"degrade_to_stage1"``,
        ``"mask_known_only"``, ``"open"``, ``"closed"``, ``"ocr"``.
    reason : str
        Which policy layer produced this decision:
        ``"channel_override:<channel>.<knob>"``,
        ``"category_override:<cat>.stage2_fail_action"``,
        or ``"global.<knob>"``.
    """
    action: str
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
# Policy Decision Engine
# ─────────────────────────────────────────────────────────────────────────────

class PolicyDecisionEngine:
    """
    Stateless (per-request) policy evaluator.

    Instantiate once per request (or per session) with the live
    :class:`~pii_guard.policy.PolicyConfig` and call :meth:`decide` for
    each detection returned by the scanning stage.

    Parameters
    ----------
    config : PolicyConfig | None
        Live policy config from :class:`~pii_guard.policy.PolicyLoader`.
        Falls back to :data:`~pii_guard.policy.SECURE_DEFAULTS` when
        ``None``.
    channel : str | None
        Originating channel name (e.g. ``"cli"``, ``"ouroboros"``).  Used
        to look up :class:`~pii_guard.policy.ChannelOverride` entries.
        ``None`` / ``""`` means no channel override is applied.
    hmac_key : bytes | None
        Key for pin-list HMAC comparison.  When ``None``, pin-list matching
        is skipped entirely.  The key is never persisted; callers should
        supply the same session key used for Ledger correlation.
    """

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        channel: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self._config: PolicyConfig = config if config is not None else SECURE_DEFAULTS
        self._channel: str = channel or ""
        self._hmac_key: Optional[bytes] = hmac_key

    # ── Per-detection decision ────────────────────────────────────────────────

    def decide(self, detection: Detection) -> PolicyDecision:
        """
        Resolve the final action and mask_style for *detection*.

        See module docstring for full precedence rules.

        Parameters
        ----------
        detection : Detection
            A detection object from :func:`~pii_guard.detector.scan_text`
            or Stage-2 NER.

        Returns
        -------
        PolicyDecision
            Fully resolved decision — never raises.
        """
        cat: str = detection.category
        cat_spec: Optional[CategorySpec] = CATEGORY_MAP.get(cat)
        cat_policy = self._config.categories.get(cat)

        # ── Step 1: Effective minimum confidence ─────────────────────────────
        if cat_policy is not None and cat_policy.min_confidence is not None:
            effective_min: float = cat_policy.min_confidence
        elif cat_spec is not None:
            effective_min = cat_spec.min_confidence
        else:
            effective_min = 0.0  # unknown category: no confidence floor

        # ── Step 2: Below confidence threshold → allow (ignore) ──────────────
        if detection.confidence < effective_min:
            return PolicyDecision(
                action=Action.ALLOW.value,
                mask_style=MaskStyle.TOKENIZE.value,
                effective_min_confidence=effective_min,
                reason="below_min_confidence",
                detector_id=detection.rule_id,
            )

        # ── Step 3: Pin-list HMAC match ───────────────────────────────────────
        if self._hmac_key is not None:
            pin_decision = self._check_pin_list(detection, cat, effective_min)
            if pin_decision is not None:
                return pin_decision

        # ── Step 4: Per-category policy override ─────────────────────────────
        if cat_policy is not None and cat_policy.action is not None:
            # action is overridden; mask_style may also be overridden
            overridden_mask_style: str = (
                cat_policy.mask_style
                if cat_policy.mask_style is not None
                else (
                    cat_spec.mask_style.value
                    if cat_spec is not None
                    else MaskStyle.TOKENIZE.value
                )
            )
            return PolicyDecision(
                action=cat_policy.action,
                mask_style=overridden_mask_style,
                effective_min_confidence=effective_min,
                reason=f"category_override:{cat}",
                detector_id=detection.rule_id,
            )

        # ── Step 5: CategorySpec built-in default ────────────────────────────
        if cat_spec is not None:
            return PolicyDecision(
                action=cat_spec.action.value,
                mask_style=cat_spec.mask_style.value,
                effective_min_confidence=effective_min,
                reason=f"category_default:{cat}",
                detector_id=detection.rule_id,
            )

        # ── Step 6: Unknown category → block (secure default) ────────────────
        return PolicyDecision(
            action=Action.BLOCK.value,
            mask_style=MaskStyle.TOKENIZE.value,
            effective_min_confidence=effective_min,
            reason=f"unknown_category:{cat}",
            detector_id=detection.rule_id,
        )

    # ── Failure / degrade decisions ───────────────────────────────────────────

    def decide_stage2_failure(
        self,
        category: Optional[str] = None,
    ) -> FailureDecision:
        """
        Resolve what to do when Stage-2 NER fails.

        Precedence (highest → lowest):

        1. Channel override ``stage2_fail_action``
        2. Per-category policy ``stage2_fail_action`` (if *category* given)
        3. Global policy ``stage2_fail_action``

        Returns
        -------
        FailureDecision
            ``action`` is one of ``"block"``, ``"mask_known_only"``, ``"open"``.
        """
        # 1. Channel override
        ch_val = self._config.channel_setting(self._channel, "stage2_fail_action")
        if ch_val is not None:
            return FailureDecision(
                action=ch_val,
                reason=f"channel_override:{self._channel}.stage2_fail_action",
            )

        # 2. Per-category override
        if category is not None:
            cat_policy = self._config.categories.get(category)
            if cat_policy is not None and cat_policy.stage2_fail_action is not None:
                return FailureDecision(
                    action=cat_policy.stage2_fail_action,
                    reason=f"category_override:{category}.stage2_fail_action",
                )

        # 3. Global default
        return FailureDecision(
            action=self._config.stage2_fail_action,
            reason="global.stage2_fail_action",
        )

    def decide_infra_failure(self) -> FailureDecision:
        """
        Resolve ``on_infra_failure``: ``"degrade_to_stage1"`` or ``"block"``.

        :class:`~pii_guard.policy.ChannelOverride` does not expose an
        ``on_infra_failure`` knob, so this always falls through to the global
        policy.

        Returns
        -------
        FailureDecision
            ``action`` is one of ``"degrade_to_stage1"``, ``"block"``.
        """
        # ChannelOverride has no on_infra_failure field, so channel_setting
        # always returns None here — fall straight to global.
        return FailureDecision(
            action=self._config.on_infra_failure,
            reason="global.on_infra_failure",
        )

    def decide_content_failure(self) -> FailureDecision:
        """
        Resolve ``on_content_failure``: ``"block"`` or ``"warn_allow"``.

        Precedence: channel override → global policy.

        Returns
        -------
        FailureDecision
            ``action`` is one of ``"block"``, ``"warn_allow"``.
        """
        ch_val = self._config.channel_setting(self._channel, "on_content_failure")
        if ch_val is not None:
            return FailureDecision(
                action=ch_val,
                reason=f"channel_override:{self._channel}.on_content_failure",
            )
        return FailureDecision(
            action=self._config.on_content_failure,
            reason="global.on_content_failure",
        )

    def decide_unscannable(self) -> FailureDecision:
        """
        Resolve ``unscannable_action``: ``"block"``, ``"warn_allow"``, or ``"ocr"``.

        Precedence: channel override → global policy.

        Returns
        -------
        FailureDecision
            ``action`` is one of ``"block"``, ``"warn_allow"``, ``"ocr"``.
        """
        ch_val = self._config.channel_setting(self._channel, "unscannable_action")
        if ch_val is not None:
            return FailureDecision(
                action=ch_val,
                reason=f"channel_override:{self._channel}.unscannable_action",
            )
        return FailureDecision(
            action=self._config.unscannable_action,
            reason="global.unscannable_action",
        )

    def decide_fail_mode(self) -> FailureDecision:
        """
        Resolve ``fail_mode``: ``"closed"`` or ``"open"``.

        Precedence: channel override → global policy.

        Returns
        -------
        FailureDecision
            ``action`` is ``"closed"`` (fail-closed, default) or ``"open"``
            (fail-open, permit degraded operation).
        """
        ch_val = self._config.channel_setting(self._channel, "fail_mode")
        if ch_val is not None:
            return FailureDecision(
                action=ch_val,
                reason=f"channel_override:{self._channel}.fail_mode",
            )
        return FailureDecision(
            action=self._config.fail_mode,
            reason="global.fail_mode",
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_pin_list(
        self,
        detection: Detection,
        category: str,
        effective_min: float,
    ) -> Optional[PolicyDecision]:
        """
        Check whether *detection*'s original value matches a pin-list entry.

        The pin-list stores opaque HMAC-SHA256 hashes (with an optional
        ``sha256:`` prefix) of normalised original values.  We compute the
        HMAC of ``detection.original.strip().lower()`` with the session
        ``hmac_key`` and compare it against each pin-list entry whose
        category matches.

        Returns
        -------
        PolicyDecision
            The pin-list entry's action, if a match is found.
        None
            If no matching pin-list entry exists.
        """
        if not self._config.pin_list or self._hmac_key is None:
            return None

        # Compute HMAC-SHA256 of the normalised original
        normalised = detection.original.strip().lower()
        computed_hex = _hmac_module.new(
            self._hmac_key,
            normalised.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        for pin in self._config.pin_list:
            if pin.category != category:
                continue

            # Stored hash may include a "sha256:" scheme prefix
            stored = pin.hash
            if stored.startswith("sha256:"):
                stored = stored[len("sha256:"):]

            # Constant-time comparison
            try:
                match = _hmac_module.compare_digest(computed_hex, stored)
            except (TypeError, ValueError):
                # compare_digest requires same-type comparable values
                continue

            if match:
                return PolicyDecision(
                    action=pin.action,
                    mask_style=MaskStyle.TOKENIZE.value,
                    effective_min_confidence=effective_min,
                    reason=f"pin_list:{pin.label or pin.hash[:12]}",
                    detector_id=detection.rule_id,
                )

        return None

    # ── Convenience: apply decisions back to a detection list ─────────────────

    def apply_decisions(
        self,
        detections: list,
    ) -> list:
        """
        Run :meth:`decide` on every detection and update each
        :class:`~pii_guard.models.Detection` object's ``action`` and
        ``mask_style`` fields in-place to reflect the policy decision.

        Detections whose resolved action is ``allow`` are removed from the
        returned list (they should not appear in the redacted output).

        Parameters
        ----------
        detections : list[Detection]
            Detections from Stage-1/Stage-2 scanning.

        Returns
        -------
        list[Detection]
            Filtered and mutated list — only non-allow detections, with
            ``action`` and ``mask_style`` updated per policy.
        """
        result = []
        for det in detections:
            pd = self.decide(det)
            if pd.is_allow:
                # Below threshold or explicitly allowed — drop from output
                continue
            # Mutate the detection to reflect the resolved decision
            det.action = Action(pd.action)
            det.mask_style = MaskStyle(pd.mask_style)
            result.append(det)
        return result
