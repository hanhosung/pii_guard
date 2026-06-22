"""
Unit tests for the PII-Guard Policy Decision Engine (Sub-AC 5c-i).

Decision paths covered
----------------------
1.  allow  — detection.confidence below effective_min_confidence
2.  allow  — per-category policy override action=allow
3.  mask   — per-category policy override action=mask with mask_style
4.  block  — built-in CategorySpec default for secrets (API_KEY, AWS_SECRET…)
5.  tokenize_roundtrip — built-in CategorySpec default for contact PII (EMAIL)
6.  fail-open   — decide_fail_mode() with fail_mode="open"
7.  fail-closed — decide_fail_mode() with fail_mode="closed" (SECURE_DEFAULT)
8.  degrade     — decide_infra_failure() with on_infra_failure="degrade_to_stage1"
9.  infra block — decide_infra_failure() with on_infra_failure="block"
10. channel override precedence — channel override beats global for every knob
11. pin-list HMAC match — pin-list entry overrides category default action
12. stage2 fail action — category_policy override > global
13. unknown category   — always block (secure default)
14. category override: action only — mask_style falls through to CategorySpec
15. category override: mask_style only (action=None) — uses CategorySpec action
16. tokenize_roundtrip unchanged — policy with no category override stays default
17. decide_content_failure: channel override > global
18. decide_unscannable: channel override > global
19. decide_stage2_failure: channel > category > global precedence chain
20. is_allow / is_block / is_mask predicates
21. apply_decisions: filters allow, mutates action/mask_style in-place
22. min_confidence override from policy takes precedence over CategorySpec
23. confidence exactly AT the threshold is accepted (not below)
24. pin-list: no match when category differs
25. pin-list: skipped when hmac_key is None
26. pin-list: sha256:-prefixed stored hash resolves correctly
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
from dataclasses import dataclass
from typing import Optional

import pytest

from pii_guard.categories import CATEGORY_MAP, EMAIL, API_KEY
from pii_guard.decision import FailureDecision, PolicyDecision, PolicyDecisionEngine
from pii_guard.models import Action, CategoryClass, Detection, DetectionStage, MaskStyle
from pii_guard.policy import (
    SECURE_DEFAULTS,
    CategoryPolicy,
    ChannelOverride,
    PinListEntry,
    PolicyConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_detection(
    category: str = "EMAIL",
    confidence: float = 0.95,
    original: str = "alice@example.com",
    rule_id: str = "email_rfc5322",
    action: Action = Action.TOKENIZE_ROUNDTRIP,
    mask_style: MaskStyle = MaskStyle.TOKENIZE,
    category_class: CategoryClass = CategoryClass.PII,
) -> Detection:
    """Create a minimal Detection for testing."""
    return Detection(
        category=category,
        category_class=category_class,
        action=action,
        mask_style=mask_style,
        start=0,
        end=len(original),
        original=original,
        detection_stage=DetectionStage.STAGE1_REGEX_CHECKSUM,
        rule_id=rule_id,
        confidence=confidence,
    )


def _make_config(**kwargs) -> PolicyConfig:
    """Create a PolicyConfig with given keyword overrides."""
    cfg = PolicyConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _hmac_of(key: bytes, value: str) -> str:
    """Compute HMAC-SHA256 hex of normalised *value* with *key*."""
    normalised = value.strip().lower()
    return _hmac.new(key, normalised.encode("utf-8"), hashlib.sha256).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 1. allow — below confidence threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestBelowConfidenceAllow:
    def test_returns_allow_when_below_categoryspec_min(self):
        """EMAIL has min_confidence=0.90 by default; a 0.80 hit is below → allow."""
        det = _make_detection(category="EMAIL", confidence=0.80)
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        assert decision.is_allow
        assert decision.reason == "below_min_confidence"

    def test_returns_allow_when_below_category_policy_min(self):
        """Per-category min_confidence=0.95 overrides the CategorySpec 0.90."""
        cfg = _make_config(categories={
            "EMAIL": CategoryPolicy(min_confidence=0.95),
        })
        det = _make_detection(category="EMAIL", confidence=0.92)
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.is_allow
        assert decision.effective_min_confidence == pytest.approx(0.95)

    def test_effective_min_confidence_from_policy_in_decision(self):
        """effective_min_confidence in PolicyDecision reflects the policy value."""
        cfg = _make_config(categories={
            "PHONE": CategoryPolicy(min_confidence=0.99),
        })
        det = _make_detection(category="PHONE", confidence=0.95,
                              original="010-1234-5678",
                              action=Action.TOKENIZE_ROUNDTRIP)
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.effective_min_confidence == pytest.approx(0.99)

    def test_categoryspec_min_used_when_no_policy_override(self):
        """Without a category policy override, CategorySpec.min_confidence is used."""
        # EMAIL.min_confidence is 0.90
        det = _make_detection(category="EMAIL", confidence=0.91)
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        # 0.91 >= 0.90 → should NOT be below threshold
        assert not decision.is_allow
        assert decision.effective_min_confidence == pytest.approx(0.90)


# ─────────────────────────────────────────────────────────────────────────────
# 2. allow — category policy override action=allow
# ─────────────────────────────────────────────────────────────────────────────

class TestAllowViaOverride:
    def test_category_override_action_allow(self):
        """A per-category policy override of action=allow produces an allow decision."""
        cfg = _make_config(categories={
            "EMAIL": CategoryPolicy(action="allow"),
        })
        det = _make_detection(category="EMAIL", confidence=0.95)
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.action == "allow"
        assert decision.is_allow
        assert "category_override:EMAIL" in decision.reason

    def test_api_key_overridden_to_allow(self):
        """Even a secret category can be overridden to allow (e.g., CI allowlists)."""
        cfg = _make_config(categories={
            "API_KEY": CategoryPolicy(action="allow"),
        })
        det = _make_detection(
            category="API_KEY", confidence=0.98,
            original="sk-proj-AAAAAAAAAAAAAAAAAAAAAA",
            action=Action.BLOCK,
            category_class=CategoryClass.SECRET,
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.is_allow


# ─────────────────────────────────────────────────────────────────────────────
# 3. mask — category policy override action=mask with mask_style
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskViaOverride:
    def test_category_override_action_mask(self):
        cfg = _make_config(categories={
            "PHONE": CategoryPolicy(action="mask", mask_style="partial"),
        })
        det = _make_detection(
            category="PHONE", confidence=0.95,
            original="010-1234-5678",
            action=Action.TOKENIZE_ROUNDTRIP,
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.action == "mask"
        assert decision.mask_style == "partial"
        assert decision.is_mask

    def test_mask_with_format_preserving_style(self):
        cfg = _make_config(categories={
            "CARD": CategoryPolicy(action="mask", mask_style="format_preserving"),
        })
        det = _make_detection(
            category="CARD", confidence=0.92,
            original="4111111111111111",
            action=Action.BLOCK,
            category_class=CategoryClass.PII,
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.action == "mask"
        assert decision.mask_style == "format_preserving"

    def test_tokenize_roundtrip_via_override(self):
        """action=tokenize_roundtrip is valid and is_mask returns True."""
        cfg = _make_config(categories={
            "API_KEY": CategoryPolicy(action="tokenize_roundtrip"),
        })
        det = _make_detection(
            category="API_KEY", confidence=0.98,
            original="sk-ant-api03-XXXXX",
            action=Action.BLOCK,
            category_class=CategoryClass.SECRET,
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.action == "tokenize_roundtrip"
        assert decision.is_mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. block — built-in CategorySpec default for secrets
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockDefault:
    @pytest.mark.parametrize("category,original", [
        ("API_KEY", "sk-proj-AAAAAAAAAAAAAAAAAAAAAA"),
        ("AWS_SECRET", "AKIAIOSFODNN7EXAMPLE"),
        ("GCP_KEY", "AIzaSyD-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"),
        ("PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----"),
        ("TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxx.yyy"),
        ("PASSWORD", "password: s3cr3tP@ss"),
        ("RRN", "901212-1234567"),
        ("CARD", "4111111111111111"),
    ])
    def test_secret_and_high_risk_pii_block_by_default(self, category, original):
        """Secret and high-risk PII categories block with no policy override."""
        cat_spec = CATEGORY_MAP.get(category)
        if cat_spec is None:
            pytest.skip(f"Category {category} not in CATEGORY_MAP")

        det = _make_detection(
            category=category, confidence=cat_spec.min_confidence + 0.01,
            original=original, action=cat_spec.action,
            category_class=cat_spec.category_class,
        )
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        assert decision.is_block, (
            f"{category}: expected block, got {decision.action!r} "
            f"(reason={decision.reason!r})"
        )
        assert decision.reason == f"category_default:{category}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. tokenize_roundtrip — built-in default for contact PII (EMAIL, PHONE …)
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizeRoundtripDefault:
    @pytest.mark.parametrize("category,original", [
        ("EMAIL", "alice@example.com"),
        ("PHONE", "010-1234-5678"),
        ("PERSON", "Name: John Smith"),
        ("ADDRESS", "서울특별시 강남구 테헤란로 101"),
        ("BIZ_NO", "123-45-67890"),
        ("KR_ACCOUNT", "123456-12-123456"),
    ])
    def test_contact_pii_tokenize_roundtrip_by_default(self, category, original):
        cat_spec = CATEGORY_MAP.get(category)
        if cat_spec is None:
            pytest.skip(f"Category {category} not in CATEGORY_MAP")

        det = _make_detection(
            category=category, confidence=cat_spec.min_confidence + 0.01,
            original=original, action=cat_spec.action,
            category_class=cat_spec.category_class,
        )
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        assert decision.action == "tokenize_roundtrip", (
            f"{category}: expected tokenize_roundtrip, got {decision.action!r}"
        )
        assert decision.is_mask


# ─────────────────────────────────────────────────────────────────────────────
# 6+7. fail-open / fail-closed
# ─────────────────────────────────────────────────────────────────────────────

class TestFailMode:
    def test_fail_closed_is_secure_default(self):
        """SECURE_DEFAULTS has fail_mode='closed'."""
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide_fail_mode()
        assert decision.action == "closed"
        assert "global.fail_mode" in decision.reason

    def test_fail_open_configured(self):
        cfg = _make_config(fail_mode="open")
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide_fail_mode()
        assert decision.action == "open"

    def test_fail_mode_channel_override_beats_global(self):
        """Channel fail_mode=closed overrides global fail_mode=open."""
        cfg = PolicyConfig(
            fail_mode="open",
            channel_overrides={"ouroboros": ChannelOverride(fail_mode="closed")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros")
        decision = engine.decide_fail_mode()
        assert decision.action == "closed"
        assert "channel_override:ouroboros.fail_mode" in decision.reason

    def test_fail_mode_no_channel_override_uses_global(self):
        """No channel match → falls through to global fail_mode."""
        cfg = PolicyConfig(
            fail_mode="open",
            channel_overrides={"cli": ChannelOverride(fail_mode="closed")},
        )
        # Channel is "ouroboros" which has no override
        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros")
        decision = engine.decide_fail_mode()
        assert decision.action == "open"

    def test_fail_mode_no_channel_set_uses_global(self):
        cfg = _make_config(fail_mode="open")
        engine = PolicyDecisionEngine(config=cfg, channel=None)
        decision = engine.decide_fail_mode()
        assert decision.action == "open"
        assert "global" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 8+9. decide_infra_failure — degrade vs. block
# ─────────────────────────────────────────────────────────────────────────────

class TestInfraFailure:
    def test_default_degrade_to_stage1(self):
        """SECURE_DEFAULTS has on_infra_failure='degrade_to_stage1'."""
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide_infra_failure()
        assert decision.action == "degrade_to_stage1"
        assert "global.on_infra_failure" in decision.reason

    def test_block_when_configured(self):
        cfg = _make_config(on_infra_failure="block")
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide_infra_failure()
        assert decision.action == "block"

    def test_channel_has_no_infra_failure_override(self):
        """ChannelOverride has no on_infra_failure field — always uses global."""
        cfg = PolicyConfig(
            on_infra_failure="block",
            # cli override does NOT override on_infra_failure
            channel_overrides={"cli": ChannelOverride(fail_mode="closed")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="cli")
        decision = engine.decide_infra_failure()
        # global on_infra_failure is "block"; no channel override
        assert decision.action == "block"
        assert "global" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 10. Channel override precedence for all knobs
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelOverridePrecedence:
    def test_unscannable_channel_beats_global(self):
        cfg = PolicyConfig(
            unscannable_action="block",  # global
            channel_overrides={"cli": ChannelOverride(unscannable_action="warn_allow")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="cli")
        decision = engine.decide_unscannable()
        assert decision.action == "warn_allow"
        assert "channel_override:cli.unscannable_action" in decision.reason

    def test_on_content_failure_channel_beats_global(self):
        cfg = PolicyConfig(
            on_content_failure="block",
            channel_overrides={"ouroboros": ChannelOverride(on_content_failure="warn_allow")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros")
        decision = engine.decide_content_failure()
        assert decision.action == "warn_allow"
        assert "channel_override:ouroboros.on_content_failure" in decision.reason

    def test_stage2_fail_action_channel_beats_global(self):
        cfg = PolicyConfig(
            stage2_fail_action="open",  # global
            channel_overrides={"ouroboros": ChannelOverride(stage2_fail_action="block")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros")
        decision = engine.decide_stage2_failure()
        assert decision.action == "block"
        assert "channel_override:ouroboros.stage2_fail_action" in decision.reason

    def test_wrong_channel_falls_through_to_global(self):
        """A channel override on 'cli' does not apply to 'ouroboros'."""
        cfg = PolicyConfig(
            unscannable_action="block",
            channel_overrides={"cli": ChannelOverride(unscannable_action="ocr")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros")
        decision = engine.decide_unscannable()
        assert decision.action == "block"
        assert "global" in decision.reason

    def test_empty_channel_falls_through_to_global(self):
        cfg = PolicyConfig(
            stage2_fail_action="mask_known_only",
            channel_overrides={"cli": ChannelOverride(stage2_fail_action="block")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="")
        decision = engine.decide_stage2_failure()
        assert decision.action == "mask_known_only"

    def test_none_channel_falls_through_to_global(self):
        cfg = PolicyConfig(
            fail_mode="open",
            channel_overrides={"ouroboros": ChannelOverride(fail_mode="closed")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel=None)
        decision = engine.decide_fail_mode()
        assert decision.action == "open"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Pin-list HMAC match
# ─────────────────────────────────────────────────────────────────────────────

class TestPinListMatch:
    KEY = b"test-hmac-key-for-pin-list-tests"

    def _pin_hash(self, value: str) -> str:
        return _hmac_of(self.KEY, value)

    def test_pin_list_allow_overrides_block(self):
        """A pinned EMAIL that would normally be tokenize_roundtrip is allowed."""
        pinned_email = "internal-relay@corp.example.com"
        h = self._pin_hash(pinned_email)
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="allow",
                                  label="relay")],
            pin_list_approved=True,
        )
        det = _make_detection(
            category="EMAIL", confidence=0.95, original=pinned_email,
        )
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        assert decision.is_allow
        assert "pin_list" in decision.reason

    def test_pin_list_block_overrides_allow_override(self):
        """Pin-list block trumps a per-category allow override."""
        pinned_val = "blocked@example.com"
        h = self._pin_hash(pinned_val)
        cfg = PolicyConfig(
            categories={"EMAIL": CategoryPolicy(action="allow")},
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="block",
                                  label="banned")],
            pin_list_approved=True,
        )
        det = _make_detection(category="EMAIL", confidence=0.95, original=pinned_val)
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        assert decision.is_block

    def test_pin_list_sha256_prefix_resolves(self):
        """A stored hash with 'sha256:' prefix is compared correctly."""
        pinned = "test-value@example.com"
        h = self._pin_hash(pinned)
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=f"sha256:{h}", category="EMAIL", action="allow")],
            pin_list_approved=True,
        )
        det = _make_detection(category="EMAIL", confidence=0.95, original=pinned)
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        assert decision.is_allow

    def test_pin_list_no_match_when_category_differs(self):
        """Pin-list entry for PHONE doesn't match an EMAIL detection."""
        pinned = "010-1234-5678"
        h = self._pin_hash(pinned)
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=h, category="PHONE", action="allow")],
            pin_list_approved=True,
        )
        det = _make_detection(category="EMAIL", confidence=0.95, original=pinned)
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        # Falls through to CategorySpec default
        assert decision.action == "tokenize_roundtrip"

    def test_pin_list_skipped_when_no_hmac_key(self):
        """Without hmac_key, pin-list is never consulted."""
        pinned = "alice@example.com"
        h = self._pin_hash(pinned)
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="allow")],
            pin_list_approved=True,
        )
        det = _make_detection(category="EMAIL", confidence=0.95, original=pinned)
        engine = PolicyDecisionEngine(config=cfg, hmac_key=None)
        decision = engine.decide(det)
        # No hmac_key → pin-list not checked → category default
        assert decision.action == "tokenize_roundtrip"

    def test_pin_list_normalisation(self):
        """Normalisation (strip+lower) means case-insensitive match."""
        pinned = "  Alice@Example.COM  "
        h = self._pin_hash(pinned)
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="allow")],
            pin_list_approved=True,
        )
        # Detection original with different case/padding
        det = _make_detection(
            category="EMAIL", confidence=0.95, original="alice@example.com",
        )
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        assert decision.is_allow

    def test_pin_list_wrong_value_no_match(self):
        """A hash for a different value does not match."""
        h = self._pin_hash("other@example.com")
        cfg = PolicyConfig(
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="allow")],
            pin_list_approved=True,
        )
        det = _make_detection(category="EMAIL", confidence=0.95, original="alice@example.com")
        engine = PolicyDecisionEngine(config=cfg, hmac_key=self.KEY)
        decision = engine.decide(det)
        assert decision.action == "tokenize_roundtrip"  # category default


# ─────────────────────────────────────────────────────────────────────────────
# 12. Stage-2 fail action precedence
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2FailAction:
    def test_global_default_mask_known_only(self):
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide_stage2_failure()
        assert decision.action == "mask_known_only"
        assert "global.stage2_fail_action" in decision.reason

    def test_global_override(self):
        cfg = _make_config(stage2_fail_action="block")
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide_stage2_failure()
        assert decision.action == "block"

    def test_category_override_beats_global(self):
        cfg = PolicyConfig(
            stage2_fail_action="open",       # global
            categories={"PERSON": CategoryPolicy(stage2_fail_action="block")},
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide_stage2_failure(category="PERSON")
        assert decision.action == "block"
        assert "category_override:PERSON" in decision.reason

    def test_channel_override_beats_category_override(self):
        """Channel override is highest precedence, even above category override."""
        cfg = PolicyConfig(
            stage2_fail_action="open",           # global
            categories={"PERSON": CategoryPolicy(stage2_fail_action="mask_known_only")},
            channel_overrides={"cli": ChannelOverride(stage2_fail_action="block")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="cli")
        decision = engine.decide_stage2_failure(category="PERSON")
        assert decision.action == "block"
        assert "channel_override:cli" in decision.reason

    def test_no_category_arg_skips_category_check(self):
        cfg = PolicyConfig(
            stage2_fail_action="mask_known_only",
            categories={"PERSON": CategoryPolicy(stage2_fail_action="block")},
        )
        engine = PolicyDecisionEngine(config=cfg)
        # No category arg → category override not consulted
        decision = engine.decide_stage2_failure(category=None)
        assert decision.action == "mask_known_only"

    def test_category_with_no_stage2_override_falls_to_global(self):
        """Category override exists but has no stage2_fail_action → global."""
        cfg = PolicyConfig(
            stage2_fail_action="block",
            categories={"EMAIL": CategoryPolicy(action="allow")},
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide_stage2_failure(category="EMAIL")
        assert decision.action == "block"
        assert "global" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 13. Unknown category → block (secure default)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownCategory:
    def test_unknown_category_blocks_by_default(self):
        det = _make_detection(
            category="COMPLETELY_UNKNOWN_CUSTOM",
            confidence=0.99,
            original="some-mystery-value",
            action=Action.ALLOW,  # source claims allow, but engine secures it
        )
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        assert decision.is_block
        assert "unknown_category:COMPLETELY_UNKNOWN_CUSTOM" in decision.reason

    def test_unknown_category_effective_min_is_zero(self):
        det = _make_detection(
            category="NOVEL_CAT", confidence=0.01, original="x",
            action=Action.ALLOW,
        )
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        # effective_min is 0.0 for unknown categories, so confidence=0.01 passes
        assert decision.effective_min_confidence == pytest.approx(0.0)
        # Still blocks (secure default for unknown)
        assert decision.is_block

    def test_unknown_category_with_policy_override_action(self):
        """Even an unknown category can be overridden in policy."""
        cfg = PolicyConfig(categories={"CUSTOM_CAT": CategoryPolicy(action="allow")})
        det = _make_detection(
            category="CUSTOM_CAT", confidence=0.99, original="something",
            action=Action.BLOCK,
        )
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.is_allow


# ─────────────────────────────────────────────────────────────────────────────
# 14. Category override: action only (mask_style falls through to CategorySpec)
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoryOverrideFallThrough:
    def test_action_override_inherits_categoryspec_mask_style(self):
        """Override action only; mask_style falls through to CategorySpec.mask_style."""
        # EMAIL.mask_style = MaskStyle.TOKENIZE
        cfg = PolicyConfig(categories={"EMAIL": CategoryPolicy(action="mask")})
        det = _make_detection(category="EMAIL", confidence=0.95)
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        assert decision.action == "mask"
        # EMAIL's CategorySpec.mask_style is "tokenize"
        assert decision.mask_style == MaskStyle.TOKENIZE.value

    def test_override_mask_style_only_keeps_categoryspec_action(self):
        """Only mask_style is overridden; action still comes from CategorySpec."""
        # EMAIL.action = tokenize_roundtrip (default)
        cfg = PolicyConfig(categories={
            "EMAIL": CategoryPolicy(action=None, mask_style="partial"),
        })
        det = _make_detection(category="EMAIL", confidence=0.95)
        engine = PolicyDecisionEngine(config=cfg)
        decision = engine.decide(det)
        # mask_style=None means action override is absent → category default
        assert decision.action == "tokenize_roundtrip"
        # The decision reason should reflect the category default
        assert "category_default:EMAIL" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 15. decide_content_failure — channel override vs global
# ─────────────────────────────────────────────────────────────────────────────

class TestContentFailure:
    def test_default_is_block(self):
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        d = engine.decide_content_failure()
        assert d.action == "block"

    def test_global_warn_allow(self):
        cfg = _make_config(on_content_failure="warn_allow")
        engine = PolicyDecisionEngine(config=cfg)
        d = engine.decide_content_failure()
        assert d.action == "warn_allow"

    def test_channel_override_beats_global(self):
        cfg = PolicyConfig(
            on_content_failure="block",
            channel_overrides={"cli": ChannelOverride(on_content_failure="warn_allow")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="cli")
        d = engine.decide_content_failure()
        assert d.action == "warn_allow"


# ─────────────────────────────────────────────────────────────────────────────
# 16. decide_unscannable — channel override vs global
# ─────────────────────────────────────────────────────────────────────────────

class TestUnscannable:
    def test_default_is_block(self):
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        d = engine.decide_unscannable()
        assert d.action == "block"

    def test_global_warn_allow(self):
        cfg = _make_config(unscannable_action="warn_allow")
        engine = PolicyDecisionEngine(config=cfg)
        d = engine.decide_unscannable()
        assert d.action == "warn_allow"

    def test_global_ocr(self):
        cfg = _make_config(unscannable_action="ocr")
        engine = PolicyDecisionEngine(config=cfg)
        d = engine.decide_unscannable()
        assert d.action == "ocr"

    def test_channel_override_ocr(self):
        cfg = PolicyConfig(
            unscannable_action="block",
            channel_overrides={"cli": ChannelOverride(unscannable_action="ocr")},
        )
        engine = PolicyDecisionEngine(config=cfg, channel="cli")
        d = engine.decide_unscannable()
        assert d.action == "ocr"
        assert "channel_override:cli.unscannable_action" in d.reason


# ─────────────────────────────────────────────────────────────────────────────
# 17. PolicyDecision predicates
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyDecisionPredicates:
    def test_is_allow_true(self):
        d = PolicyDecision(action="allow", mask_style="tokenize",
                           effective_min_confidence=0.9, reason="test")
        assert d.is_allow
        assert not d.is_block
        assert not d.is_mask

    def test_is_block_true(self):
        d = PolicyDecision(action="block", mask_style="tokenize",
                           effective_min_confidence=0.9, reason="test")
        assert d.is_block
        assert not d.is_allow
        assert not d.is_mask

    def test_is_mask_for_mask(self):
        d = PolicyDecision(action="mask", mask_style="partial",
                           effective_min_confidence=0.9, reason="test")
        assert d.is_mask
        assert not d.is_allow
        assert not d.is_block

    def test_is_mask_for_tokenize_roundtrip(self):
        d = PolicyDecision(action="tokenize_roundtrip", mask_style="tokenize",
                           effective_min_confidence=0.9, reason="test")
        assert d.is_mask


# ─────────────────────────────────────────────────────────────────────────────
# 18. apply_decisions — filters allow and mutates in-place
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDecisions:
    def test_allow_detections_filtered_out(self):
        """Detections resolved to allow are removed from output."""
        # EMAIL with confidence below threshold → allow
        det = _make_detection(category="EMAIL", confidence=0.50)
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        result = engine.apply_decisions([det])
        assert result == [], "Below-threshold detection should be filtered"

    def test_block_detections_retained_with_mutated_action(self):
        """A block detection is retained and action is mutated to block."""
        cat_spec = CATEGORY_MAP["API_KEY"]
        det = _make_detection(
            category="API_KEY", confidence=cat_spec.min_confidence + 0.05,
            original="sk-ant-api03-XXXXXXXX",
            action=cat_spec.action,
            category_class=cat_spec.category_class,
        )
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        result = engine.apply_decisions([det])
        assert len(result) == 1
        assert result[0].action == Action.BLOCK

    def test_policy_override_action_mutated_in_place(self):
        """Category override action=mask is applied to Detection.action."""
        cfg = PolicyConfig(categories={
            "EMAIL": CategoryPolicy(action="mask", mask_style="partial"),
        })
        det = _make_detection(category="EMAIL", confidence=0.95)
        engine = PolicyDecisionEngine(config=cfg)
        result = engine.apply_decisions([det])
        assert len(result) == 1
        assert result[0].action == Action.MASK
        assert result[0].mask_style == MaskStyle.PARTIAL

    def test_mixed_allow_and_block_filtered_correctly(self):
        """Mix of allow/block detections: only non-allow are returned."""
        email_det = _make_detection(category="EMAIL", confidence=0.95)
        apikey_det = _make_detection(
            category="API_KEY", confidence=0.98,
            original="sk-ant-api03-XXXXX",
            action=Action.BLOCK, category_class=CategoryClass.SECRET,
        )
        # Force email to allow via override
        cfg = PolicyConfig(categories={"EMAIL": CategoryPolicy(action="allow")})
        engine = PolicyDecisionEngine(config=cfg)
        result = engine.apply_decisions([email_det, apikey_det])
        assert len(result) == 1
        assert result[0].category == "API_KEY"

    def test_empty_list_returns_empty(self):
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        assert engine.apply_decisions([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 19. Confidence AT the threshold — accepted (not filtered)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceAtThreshold:
    def test_confidence_exactly_at_threshold_is_accepted(self):
        """Confidence == min_confidence should NOT be treated as 'below' threshold."""
        # EMAIL.min_confidence = 0.90; set confidence exactly to 0.90
        det = _make_detection(category="EMAIL", confidence=0.90)
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        # 0.90 >= 0.90 → accepted
        assert not decision.is_allow
        assert decision.action == "tokenize_roundtrip"

    def test_confidence_just_below_threshold_is_rejected(self):
        """Confidence just below min_confidence is filtered to allow."""
        # Use float arithmetic: EMAIL.min_confidence=0.90, test at 0.8999...
        det = _make_detection(category="EMAIL", confidence=0.8999)
        engine = PolicyDecisionEngine(config=SECURE_DEFAULTS)
        decision = engine.decide(det)
        assert decision.is_allow
        assert decision.reason == "below_min_confidence"


# ─────────────────────────────────────────────────────────────────────────────
# 20. FailureDecision dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureDecision:
    def test_construction(self):
        fd = FailureDecision(action="block", reason="global.fail_mode")
        assert fd.action == "block"
        assert fd.reason == "global.fail_mode"


# ─────────────────────────────────────────────────────────────────────────────
# 21. Multiple overlapping policy layers (integration-style)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPolicyMatrix:
    """Simulate a real policy config with multiple layers active simultaneously."""

    def test_full_policy_matrix(self):
        KEY = b"integration-test-hmac-key"
        pinned_email = "relay@corp.internal"
        h = _hmac_of(KEY, pinned_email)

        cfg = PolicyConfig(
            fail_mode="closed",
            on_content_failure="block",
            on_infra_failure="degrade_to_stage1",
            stage2_fail_action="mask_known_only",
            unscannable_action="block",
            categories={
                "EMAIL": CategoryPolicy(action="tokenize_roundtrip", min_confidence=0.92),
                "PHONE": CategoryPolicy(action="allow"),           # CI: allow all phones
                "API_KEY": CategoryPolicy(action="block"),          # already default but explicit
                "PERSON": CategoryPolicy(stage2_fail_action="block"),
            },
            pin_list=[PinListEntry(hash=h, category="EMAIL", action="allow",
                                   label="corp-relay")],
            pin_list_approved=True,
            channel_overrides={
                "cli": ChannelOverride(unscannable_action="warn_allow"),
                "ouroboros": ChannelOverride(stage2_fail_action="block",
                                            fail_mode="closed"),
            },
        )

        engine = PolicyDecisionEngine(config=cfg, channel="ouroboros", hmac_key=KEY)

        # Pinned email → allow (pin-list overrides tokenize_roundtrip)
        d1 = engine.decide(_make_detection("EMAIL", 0.95, original=pinned_email))
        assert d1.is_allow and "pin_list" in d1.reason

        # Regular email → tokenize_roundtrip
        d2 = engine.decide(_make_detection("EMAIL", 0.95, original="other@example.com"))
        assert d2.action == "tokenize_roundtrip"

        # Phone → allow (policy override)
        d3 = engine.decide(_make_detection("PHONE", 0.95, original="010-1234-5678",
                                           action=Action.TOKENIZE_ROUNDTRIP))
        assert d3.is_allow

        # API_KEY → block (policy override; also CategorySpec default)
        d4 = engine.decide(_make_detection(
            "API_KEY", 0.98, original="sk-ant-api03-XXXXX",
            action=Action.BLOCK, category_class=CategoryClass.SECRET,
        ))
        assert d4.is_block

        # Email below policy min_confidence (0.92) → allow
        d5 = engine.decide(_make_detection("EMAIL", 0.91, original="test@example.com"))
        assert d5.is_allow and d5.reason == "below_min_confidence"

        # Failure decisions for ouroboros channel
        fm = engine.decide_fail_mode()
        assert fm.action == "closed"   # channel override

        s2f = engine.decide_stage2_failure(category="PERSON")
        assert s2f.action == "block"   # channel override beats category override

        uns = engine.decide_unscannable()
        assert uns.action == "block"   # ouroboros has no unscannable_action override → global
