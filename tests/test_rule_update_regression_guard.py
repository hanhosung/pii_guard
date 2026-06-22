"""
Regression Guard CI test — Sub-AC 7.3
======================================

Loads both the pre-update and post-update rule/model snapshots, runs the golden
Korean PII corpus through both, and asserts that recall and precision do not
degrade beyond defined thresholds after any update.

This is the CI gate that must pass before any signed rule/model update is
applied.  It models the full update lifecycle:

1. Maintainer produces a *pre-update snapshot* (signed manifest of current rules).
2. Maintainer produces a *post-update snapshot* (signed manifest of proposed rules).
3. CI loads both, verifies signatures (rejects unsigned/tampered manifests early),
   builds engines from each, runs the golden corpus through both, and checks the
   per-category recall/precision delta.
4. Any regression beyond the allowed threshold — or any category with zero
   tolerance (e.g. RRN) losing even a fraction of a percent recall — fails CI
   and blocks the update.

Coverage matrix
---------------
Snapshot integrity
  [S1] snapshot_from_categories → signed manifest with category_spec entries
  [S2] engine_from_snapshot verifies signature then reconstructs Engine correctly
  [S3] serialize/deserialize round-trip preserves identical detection behaviour
  [S4] validator functions (rrn_checksum, luhn) survive round-trip

Update channel integrity (must reject before metrics are even computed)
  [I1] unsigned manifest raises UpdateRejectedError
  [I2] post-signing tamper raises UpdateRejectedError
  [I3] manifest signed with wrong key raises UpdateRejectedError

Non-regression gate — corpus integration
  [C1] identical rule snapshot produces zero delta → PASS
  [C2] remove phone_intl rule → ~17% PHONE recall drop → regression caught → FAIL
  [C3] remove RRN rule → 100% RRN recall drop → zero-tolerance violation → FAIL

Non-regression gate — unit tests on check_regression()
  [U1] synthetic 4% recall drop (< 5% threshold) → no violation → PASS
  [U2] synthetic 8% recall drop (> 5% threshold) → violation raised → FAIL
  [U3] synthetic 0.5% RRN recall drop (zero-tolerance) → violation raised → FAIL
  [U4] synthetic 2% precision drop (< 3% threshold) → no violation → PASS
  [U5] synthetic 5% precision drop (> 3% threshold) → violation raised → FAIL
  [U6] zero-delta update → no violations for any measured category

Absolute golden floor guard (independent of pre/post delta)
  [G1] baseline PHONE, RRN, KR_ACCOUNT meet absolute floor thresholds
  [G2] RRN recall ≥ 0.90 (high-severity Stage-1 backstop floor)

Regression comparison report
  [R1] prints human-readable CI comparison table (always passes; for visibility)

How to update detection thresholds
------------------------------------
When a rule improvement intentionally raises precision or recall above these
thresholds, update ``test_korean_pii_regression.py`` first, then update the
_ABSOLUTE_RECALL_FLOORS below to reflect the new performance baseline.

  1. Run: pytest tests/test_rule_update_regression_guard.py -v -s
  2. Confirm the new metrics in the comparison table.
  3. Sign the rule-set change via ``pii_guard.updater.UpdateSigner.sign()``.
  4. Open a PR with the detection-rule changelog entry.
  5. Update GOLDEN_AGGREGATE in test_korean_pii_regression.py.

This file is part of the control plane — it must NOT be modified by an
automated agent.  Threshold edits require out-of-band user review.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import pytest

from pii_guard import Engine
from pii_guard.categories import ALL_CATEGORIES, CategorySpec, PatternRule
from pii_guard.corpus import KoreanPIICorpus
from pii_guard.corpus.korean_pii import compute_precision_recall
from pii_guard.models import Action, CategoryClass, DetectionStage, MaskStyle
from pii_guard.updater import (
    UpdateManifest,
    UpdateRejectedError,
    UpdateSigner,
    UpdateVerifier,
    make_manifest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CORPUS_SEED: int = 42
_SAMPLES_PER_FORMAT: int = 5

#: Categories measured against the Korean PII corpus.
_MEASURED_CATEGORIES: Set[str] = {"PHONE", "RRN", "KR_ACCOUNT", "PERSON", "ADDRESS"}

#: Maximum allowed recall drop between pre- and post-update snapshots.
_MAX_RECALL_REGRESSION: float = 0.05   # 5 percentage-point drop allowed

#: Maximum allowed precision drop between pre- and post-update snapshots.
_MAX_PRECISION_REGRESSION: float = 0.03  # 3 percentage-point drop allowed

#: Categories that must never lose any recall — even a 0.1% drop fails CI.
#: These are the highest-severity categories where Stage-1 is the fail-safe backstop.
_ZERO_TOLERANCE_RECALL: Set[str] = {"RRN", "CARD"}

#: Absolute recall floors independent of the pre/post delta.
#: These must hold even for the "updated" engine after any rule change.
_ABSOLUTE_RECALL_FLOORS: Dict[str, float] = {
    "PHONE":      0.80,
    "RRN":        0.90,
    "KR_ACCOUNT": 0.80,
    "PERSON":     0.40,
    "ADDRESS":    0.60,
}

#: Regex flags we care about preserving in snapshots.
#: Python always adds re.UNICODE (32) — we strip it to keep flags portable.
_PRESERVED_RE_FLAGS: int = (
    re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE
)


# ─────────────────────────────────────────────────────────────────────────────
# Validator registry — built from live ALL_CATEGORIES at import time
# ─────────────────────────────────────────────────────────────────────────────

def _build_validator_registry() -> Dict[str, Callable]:
    """
    Build a ``{rule_id: validator_fn}`` map from the live ``ALL_CATEGORIES``.

    This avoids importing private validator functions (``_rrn_checksum`` etc.)
    directly from ``pii_guard.categories``.  When a snapshot is deserialised,
    validators are looked up by their rule_id (which is stable across updates)
    so checksum logic survives serialisation without pickling.

    New validators are automatically included when rules are added to
    ``ALL_CATEGORIES`` — no changes to this file are needed.
    """
    registry: Dict[str, Callable] = {}
    for cat in ALL_CATEGORIES:
        for rule in cat.rules:
            if rule.validator is not None:
                registry[rule.rule_id] = rule.validator
    return registry


#: Live registry keyed by rule_id.  Built once at module load.
_VALIDATOR_REGISTRY: Dict[str, Callable] = _build_validator_registry()


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot serialization / deserialization
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_category(cat: CategorySpec) -> str:
    """
    Serialise *cat* to a JSON string suitable as a ``category_spec`` manifest
    entry content.

    The ``validator_id`` field stores the rule_id of the rule that has a
    validator, so the function can be looked up from ``_VALIDATOR_REGISTRY``
    on deserialisation without pickling.
    """
    data: dict = {
        "category": cat.category,
        "category_class": cat.category_class.value,
        "action": cat.action.value,
        "mask_style": cat.mask_style.value,
        "min_confidence": cat.min_confidence,
        "detection_stage": cat.detection_stage.value,
        "rules": [
            {
                "rule_id": rule.rule_id,
                "pattern": rule.pattern.pattern,
                # Strip always-present UNICODE flag to keep flags portable across
                # Python versions.  The target platform re-adds it on compile.
                "flags": rule.pattern.flags & _PRESERVED_RE_FLAGS,
                "confidence": rule.confidence,
                # If this rule has a validator, store its rule_id for registry lookup.
                "validator_id": rule.rule_id if rule.validator is not None else None,
            }
            for rule in cat.rules
        ],
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _deserialize_category(content: str) -> CategorySpec:
    """
    Reconstruct a ``CategorySpec`` from manifest entry content JSON.

    Validators are looked up from ``_VALIDATOR_REGISTRY`` by the stored
    ``validator_id``.  Rules whose ``validator_id`` is absent or not in the
    registry are reconstructed without a validator (safe default for rules
    that never had one).
    """
    data = json.loads(content)
    rules: List[PatternRule] = []
    for r in data["rules"]:
        pattern = re.compile(r["pattern"], r.get("flags", 0))
        validator_id = r.get("validator_id")
        validator = _VALIDATOR_REGISTRY.get(validator_id) if validator_id else None
        rules.append(PatternRule(
            rule_id=r["rule_id"],
            pattern=pattern,
            confidence=r["confidence"],
            validator=validator,
        ))
    return CategorySpec(
        category=data["category"],
        category_class=CategoryClass(data["category_class"]),
        action=Action(data["action"]),
        mask_style=MaskStyle(data["mask_style"]),
        min_confidence=data["min_confidence"],
        rules=rules,
        detection_stage=DetectionStage(
            data.get("detection_stage", DetectionStage.STAGE1_REGEX_CHECKSUM.value)
        ),
    )


def snapshot_from_categories(
    categories: List[CategorySpec],
    signer: UpdateSigner,
    *,
    version: str = "1.0.0",
    timestamp: str = "2026-01-01T00:00:00Z",
) -> UpdateManifest:
    """
    Serialise *categories* into a **signed** ``UpdateManifest`` (a rule snapshot).

    Each category becomes one manifest entry with ``kind="category_spec"``.
    The manifest is signed with *signer* using HMAC-SHA256, as required by the
    update channel's integrity contract.

    Parameters
    ----------
    categories :
        Ordered rule set to snapshot.  The ordering is preserved and used as
        the detection-priority order when the Engine is reconstructed.
    signer :
        ``UpdateSigner`` holding the local update key.
    version :
        Manifest version string (for audit; does not affect detection).
    timestamp :
        ISO-8601 UTC timestamp for the snapshot (use a fixed value in tests for
        deterministic signatures).

    Returns
    -------
    UpdateManifest
        A signed manifest ready for ``engine_from_snapshot()``.
    """
    entries = [
        (cat.category, "category_spec", _serialize_category(cat))
        for cat in categories
    ]
    manifest = make_manifest(
        version, "rule_update", entries,
        timestamp=timestamp, author="pii-guard-ci",
    )
    return signer.sign(manifest)


def engine_from_snapshot(
    manifest: UpdateManifest,
    verifier: UpdateVerifier,
) -> Engine:
    """
    Verify *manifest* and construct an ``Engine`` from its embedded rule set.

    Verification is the first step — the engine is only built if the manifest
    passes the full HMAC + per-entry hash verification pipeline.  This enforces
    the constraint that rule updates can only be applied via the signed channel.

    Parameters
    ----------
    manifest :
        A signed snapshot produced by ``snapshot_from_categories``.
    verifier :
        ``UpdateVerifier`` for the signing key.

    Raises
    ------
    UpdateRejectedError
        If the manifest fails any integrity check (no signature, tampered fields,
        wrong key, bad entry hash).

    Returns
    -------
    Engine
        Ready to scan text using the rule set encoded in the snapshot.
    """
    verifier.verify(manifest)  # raises UpdateRejectedError on any failure
    categories: List[CategorySpec] = []
    for entry in manifest.entries:
        if entry.kind == "category_spec":
            categories.append(_deserialize_category(entry.content))
    return Engine(categories=categories)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_snapshot_metrics(
    engine: Engine,
    corpus: KoreanPIICorpus,
    categories: Optional[Set[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Run *corpus* through *engine* and return per-category (precision, recall).

    Parameters
    ----------
    engine :
        The engine to evaluate.
    corpus :
        The golden corpus to run against.
    categories :
        Which categories to measure.  Defaults to ``_MEASURED_CATEGORIES``.

    Returns
    -------
    Dict mapping category_name → (precision, recall).
    """
    if categories is None:
        categories = _MEASURED_CATEGORIES

    def detect_fn(text: str) -> Set[str]:
        result = engine.scan(text)
        return {d.category for d in result.detections}

    metrics: Dict[str, Tuple[float, float]] = {}
    for cat in sorted(categories):
        precision, recall = compute_precision_recall(corpus, detect_fn, cat)
        metrics[cat] = (precision, recall)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Regression checker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionViolation:
    """One category/metric pair that exceeded its regression threshold."""
    category: str
    metric: str            # "recall" or "precision"
    baseline: float
    updated: float
    delta: float           # updated - baseline (negative = regression)
    threshold: float       # most negative delta allowed (e.g. -0.05)
    message: str

    def __str__(self) -> str:
        return self.message


def check_regression(
    baseline: Dict[str, Tuple[float, float]],
    updated: Dict[str, Tuple[float, float]],
    *,
    max_recall_regression: float = _MAX_RECALL_REGRESSION,
    max_precision_regression: float = _MAX_PRECISION_REGRESSION,
    zero_tolerance_recall: Optional[Set[str]] = None,
) -> List[RegressionViolation]:
    """
    Compare *updated* metrics against *baseline* and return any violations.

    A violation is generated when:
    - The updated recall/precision drops by more than the allowed threshold.
    - OR the category is in ``zero_tolerance_recall`` and recall drops at all
      (even by a fraction of a percentage point).

    Parameters
    ----------
    baseline :
        ``{category: (precision, recall)}`` from the pre-update snapshot.
    updated :
        ``{category: (precision, recall)}`` from the post-update snapshot.
    max_recall_regression :
        Maximum allowed absolute recall drop (e.g. 0.05 = 5 percentage points).
    max_precision_regression :
        Maximum allowed absolute precision drop.
    zero_tolerance_recall :
        Categories for which ANY recall drop is a violation.  Defaults to
        ``_ZERO_TOLERANCE_RECALL`` ({"RRN", "CARD"}).

    Returns
    -------
    List[RegressionViolation]
        Empty → no regressions, update is safe to apply.
        Non-empty → CI must fail; do not apply the update.
    """
    if zero_tolerance_recall is None:
        zero_tolerance_recall = _ZERO_TOLERANCE_RECALL

    violations: List[RegressionViolation] = []

    for cat in sorted(baseline):
        if cat not in updated:
            continue  # category removed — covered by a separate coverage test

        base_precision, base_recall = baseline[cat]
        upd_precision, upd_recall = updated[cat]

        recall_delta = upd_recall - base_recall
        precision_delta = upd_precision - base_precision

        # ── Recall regression check ───────────────────────────────────────────
        if cat in zero_tolerance_recall:
            # Any recall drop — no matter how tiny — is a violation.
            recall_threshold = 0.0
        else:
            recall_threshold = -max_recall_regression

        if recall_delta < recall_threshold:
            violations.append(RegressionViolation(
                category=cat,
                metric="recall",
                baseline=base_recall,
                updated=upd_recall,
                delta=recall_delta,
                threshold=recall_threshold,
                message=(
                    f"[RECALL REGRESSION] {cat}: "
                    f"{base_recall:.4f} → {upd_recall:.4f} "
                    f"(Δ={recall_delta:+.4f}, min allowed Δ={recall_threshold:+.4f})"
                ),
            ))

        # ── Precision regression check ────────────────────────────────────────
        precision_threshold = -max_precision_regression
        if precision_delta < precision_threshold:
            violations.append(RegressionViolation(
                category=cat,
                metric="precision",
                baseline=base_precision,
                updated=upd_precision,
                delta=precision_delta,
                threshold=precision_threshold,
                message=(
                    f"[PRECISION REGRESSION] {cat}: "
                    f"{base_precision:.4f} → {upd_precision:.4f} "
                    f"(Δ={precision_delta:+.4f}, min allowed Δ={precision_threshold:+.4f})"
                ),
            ))

    return violations


def format_regression_report(
    baseline: Dict[str, Tuple[float, float]],
    updated: Dict[str, Tuple[float, float]],
    violations: List[RegressionViolation],
    *,
    pre_label: str = "pre-update",
    post_label: str = "post-update",
) -> str:
    """
    Format a human-readable comparison table for CI output.

    Always safe to print — no raw PII, only metric deltas.
    """
    col_w = 68
    lines = [
        "",
        "┌" + "─" * col_w + "┐",
        "│  Regression Guard — Rule Update Comparison Report" + " " * (col_w - 50) + "│",
        "│" + " " * col_w + "│",
        f"│  Pre : {pre_label:<{col_w - 8}}│",
        f"│  Post: {post_label:<{col_w - 8}}│",
        "│" + " " * col_w + "│",
        "│  {:<15} {:>20} {:>25}  │".format(
            "Category", "Recall  pre→post (Δ)", "Precision pre→post (Δ)"
        ),
        "│" + "─" * col_w + "│",
    ]

    all_cats = sorted(set(baseline) | set(updated))
    for cat in all_cats:
        if cat in baseline and cat in updated:
            bp, br = baseline[cat]
            up, ur = updated[cat]
            rd = ur - br
            pd = up - bp
            r_ok = "✓" if rd >= -_MAX_RECALL_REGRESSION else "✗"
            p_ok = "✓" if pd >= -_MAX_PRECISION_REGRESSION else "✗"
            line = (
                f"│  {cat:<14}  "
                f"rec {br:.3f}→{ur:.3f} ({rd:+.3f}){r_ok}  "
                f"prec {bp:.3f}→{up:.3f} ({pd:+.3f}){p_ok}"
            )
            lines.append(line + " " * max(0, col_w + 1 - len(line)) + "│")
        elif cat in baseline:
            bp, br = baseline[cat]
            line = f"│  {cat:<14}  rec {br:.3f}→REMOVED  prec {bp:.3f}→REMOVED"
            lines.append(line + " " * max(0, col_w + 1 - len(line)) + "│")
        else:
            up, ur = updated[cat]
            line = f"│  {cat:<14}  rec NEW→{ur:.3f}        prec NEW→{up:.3f}"
            lines.append(line + " " * max(0, col_w + 1 - len(line)) + "│")

    lines.append("│" + "─" * col_w + "│")
    if violations:
        result_line = f"│  RESULT: FAIL — {len(violations)} regression violation(s)"
        lines.append(result_line + " " * max(0, col_w + 1 - len(result_line)) + "│")
        for v in violations:
            msg = v.message[:col_w - 4]
            msg_line = f"│    {msg}"
            lines.append(msg_line + " " * max(0, col_w + 1 - len(msg_line)) + "│")
    else:
        result_line = "│  RESULT: PASS — no regressions detected"
        lines.append(result_line + " " * max(0, col_w + 1 - len(result_line)) + "│")
    lines.append("└" + "─" * col_w + "┘")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Modified rule factories — deterministic constructions for test scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _get_category(name: str) -> CategorySpec:
    """Return the CategorySpec for *name* from ALL_CATEGORIES."""
    for cat in ALL_CATEGORIES:
        if cat.category == name:
            return cat
    raise KeyError(f"Category {name!r} not found in ALL_CATEGORIES")


def _phone_without_intl() -> CategorySpec:
    """
    PHONE with the ``phone_intl`` rule removed.

    On the Korean corpus, ``phone_international`` format (+82-10-XXXX-XXXX) is
    only detected by ``phone_intl``.  Removing it causes ≈1/6 of PHONE format
    groups to go undetected (5 out of 30 samples ≈ 16.7% recall drop).

    Used by ``test_c2_phone_recall_regression_detected`` to verify that the
    regression guard catches a beyond-threshold recall drop.
    """
    phone = _get_category("PHONE")
    return CategorySpec(
        category=phone.category,
        category_class=phone.category_class,
        action=phone.action,
        mask_style=phone.mask_style,
        min_confidence=phone.min_confidence,
        rules=[r for r in phone.rules if r.rule_id != "phone_intl"],
    )


def _rrn_no_rules() -> CategorySpec:
    """
    RRN with all rules removed (empty rules list).

    Every RRN positive sample will be missed → 100% recall drop.  Used by
    ``test_c3_rrn_zero_tolerance_regression_detected`` to verify that the
    zero-tolerance check catches ANY recall loss on high-severity categories.
    """
    rrn = _get_category("RRN")
    return CategorySpec(
        category=rrn.category,
        category_class=rrn.category_class,
        action=rrn.action,
        mask_style=rrn.mask_style,
        min_confidence=rrn.min_confidence,
        rules=[],
    )


def _all_categories_with(substitutions: Dict[str, CategorySpec]) -> List[CategorySpec]:
    """
    Return ALL_CATEGORIES with specified categories replaced.

    The priority ordering of ALL_CATEGORIES is preserved — only the named
    categories are swapped out.  This models a targeted rule update where a
    single category is changed and all others are unchanged.

    Parameters
    ----------
    substitutions :
        ``{category_name: new_CategorySpec}`` — categories to replace.
    """
    return [
        substitutions.get(cat.category, cat)
        for cat in ALL_CATEGORIES
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def corpus() -> KoreanPIICorpus:
    """Deterministic Korean PII corpus (seed=42, 5 samples/format)."""
    return KoreanPIICorpus(seed=_CORPUS_SEED, samples_per_format=_SAMPLES_PER_FORMAT)


# Session-scoped key + signer + verifier for the main corpus integration tests.
# These are separate from the adversarial fixtures (which use function scope)
# so that session-level baselines are consistent within a test run.

@pytest.fixture(scope="session")
def _session_key() -> bytes:
    """Fresh 256-bit key for the session-scoped snapshot fixtures."""
    return UpdateSigner.generate_key()


@pytest.fixture(scope="session")
def _session_signer(_session_key: bytes) -> UpdateSigner:
    return UpdateSigner(_session_key)


@pytest.fixture(scope="session")
def _session_verifier(_session_key: bytes) -> UpdateVerifier:
    return UpdateVerifier(_session_key)


@pytest.fixture(scope="session")
def baseline_snapshot(
    _session_signer: UpdateSigner,
) -> UpdateManifest:
    """
    Signed snapshot of the current production rule set (ALL_CATEGORIES).

    This is the *pre-update* snapshot that all post-update snapshots are
    compared against.  It is signed with the session key and computed once.
    """
    return snapshot_from_categories(
        ALL_CATEGORIES,
        _session_signer,
        version="baseline",
        timestamp="2026-01-01T00:00:00Z",
    )


@pytest.fixture(scope="session")
def baseline_engine(
    baseline_snapshot: UpdateManifest,
    _session_verifier: UpdateVerifier,
) -> Engine:
    """Engine reconstructed from the baseline snapshot."""
    return engine_from_snapshot(baseline_snapshot, _session_verifier)


@pytest.fixture(scope="session")
def baseline_metrics(
    baseline_engine: Engine,
    corpus: KoreanPIICorpus,
) -> Dict[str, Tuple[float, float]]:
    """
    Per-category (precision, recall) for the baseline engine on the golden corpus.

    Computed once per session and reused across all regression-gate tests.
    """
    return _compute_snapshot_metrics(baseline_engine, corpus)


# Function-scoped adversarial fixtures

@pytest.fixture()
def key() -> bytes:
    """A fresh 256-bit HMAC key for adversarial tests."""
    return UpdateSigner.generate_key()


@pytest.fixture()
def other_key() -> bytes:
    """A second, distinct key (used in wrong-key test)."""
    return UpdateSigner.generate_key()


@pytest.fixture()
def signer(key: bytes) -> UpdateSigner:
    return UpdateSigner(key)


@pytest.fixture()
def verifier(key: bytes) -> UpdateVerifier:
    return UpdateVerifier(key)


# ─────────────────────────────────────────────────────────────────────────────
# [S] Snapshot integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotIntegrity:
    """
    Verify that the snapshot serialization/deserialization pipeline is correct.

    A broken round-trip would mean the regression guard is comparing the corpus
    against different rules than the ones in the manifest — giving false results.
    """

    def test_s1_snapshot_is_signed(
        self, signer: UpdateSigner, verifier: UpdateVerifier,
    ) -> None:
        """[S1] snapshot_from_categories always produces a signed manifest."""
        manifest = snapshot_from_categories(
            ALL_CATEGORIES, signer, timestamp="2026-01-01T00:00:00Z"
        )
        assert manifest.signature is not None
        assert manifest.signature.startswith("hmac-sha256:"), (
            f"Expected 'hmac-sha256:' prefix, got {manifest.signature!r}"
        )
        # Must pass verification
        verifier.verify(manifest)  # raises if invalid

    def test_s2_snapshot_contains_all_categories(
        self, signer: UpdateSigner,
    ) -> None:
        """[S2] One entry per category in the snapshot manifest."""
        manifest = snapshot_from_categories(
            ALL_CATEGORIES, signer, timestamp="2026-01-01T00:00:00Z"
        )
        entry_names = [e.name for e in manifest.entries if e.kind == "category_spec"]
        expected = [cat.category for cat in ALL_CATEGORIES]
        assert entry_names == expected, (
            f"Snapshot entry names don't match ALL_CATEGORIES order.\n"
            f"Expected: {expected}\n"
            f"Got:      {entry_names}"
        )

    def test_s3_round_trip_preserves_detection_behaviour(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
        corpus: KoreanPIICorpus,
    ) -> None:
        """
        [S3] Engine built from snapshot produces identical metrics to Engine().

        This proves that the serialization/deserialization round-trip does not
        corrupt the detection logic — the same rules in, the same results out.
        """
        manifest = snapshot_from_categories(
            ALL_CATEGORIES, signer, timestamp="2026-01-01T00:00:00Z"
        )
        rt_engine = engine_from_snapshot(manifest, verifier)
        direct_engine = Engine()  # uses ALL_CATEGORIES directly

        # Spot-check on the first 20 positive samples of each measured category
        mismatches = 0
        for cat_name in sorted(_MEASURED_CATEGORIES):
            samples = corpus.samples_for_category(cat_name)[:20]
            for sample in samples:
                r_rt = rt_engine.scan(sample.text)
                r_direct = direct_engine.scan(sample.text)
                cats_rt = {d.category for d in r_rt.detections}
                cats_direct = {d.category for d in r_direct.detections}
                if cats_rt != cats_direct:
                    mismatches += 1

        assert mismatches == 0, (
            f"{mismatches} sample(s) produced different detection categories "
            f"between the round-trip engine and the direct engine. "
            f"The serialization is NOT lossless."
        )

    def test_s4_rrn_validator_survives_roundtrip(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """
        [S4] The RRN checksum validator is preserved after snapshot round-trip.

        If validators are not restored, the RRN detector would fire on
        invalid RRNs that fail the checksum — silently lowering precision.
        """
        manifest = snapshot_from_categories(
            ALL_CATEGORIES, signer, timestamp="2026-01-01T00:00:00Z"
        )
        rt_engine = engine_from_snapshot(manifest, verifier)

        # A valid RRN must be detected
        valid_rrn = "800101-1XXXXXX".replace("XXXXXX", "234568")
        # Compute a real checksum-valid RRN
        from pii_guard.corpus.korean_pii import _compute_rrn_check_digit, _make_rrn
        rrn_str = _make_rrn("800101", 1, "23456")
        rrn_display = rrn_str[:6] + "-" + rrn_str[6:]
        result = rt_engine.scan(f"주민등록번호: {rrn_display}")
        detected_cats = {d.category for d in result.detections}
        assert "RRN" in detected_cats, (
            f"RRN not detected in round-trip engine for known-valid RRN {rrn_display!r}. "
            f"Validator may have been lost during snapshot round-trip."
        )

        # An invalid RRN (wrong checksum) must NOT fire on RRN
        # Construct a 13-digit string with wrong check digit
        bad_rrn_digits = rrn_str[:12] + str((int(rrn_str[12]) + 1) % 10)
        bad_rrn_display = bad_rrn_digits[:6] + "-" + bad_rrn_digits[6:]
        result_bad = rt_engine.scan(f"주민등록번호: {bad_rrn_display}")
        rrn_detections = [d for d in result_bad.detections if d.category == "RRN"]
        assert len(rrn_detections) == 0, (
            f"RRN fired on checksum-invalid RRN {bad_rrn_display!r} in round-trip engine. "
            f"The checksum validator was NOT preserved in the snapshot."
        )

    def test_s5_modified_confidence_survives_roundtrip(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """[S5] A custom min_confidence survives the serialization round-trip."""
        phone = _get_category("PHONE")
        modified_phone = CategorySpec(
            category=phone.category,
            category_class=phone.category_class,
            action=phone.action,
            mask_style=phone.mask_style,
            min_confidence=0.95,  # raised from 0.80
            rules=phone.rules,
        )
        categories = _all_categories_with({"PHONE": modified_phone})
        manifest = snapshot_from_categories(
            categories, signer, timestamp="2026-01-01T00:00:00Z"
        )
        rt_engine = engine_from_snapshot(manifest, verifier)

        # Check that the round-tripped engine uses the modified categories
        rt_cats = rt_engine._categories
        for cat in rt_cats:
            if cat.category == "PHONE":
                assert cat.min_confidence == 0.95, (
                    f"PHONE min_confidence should be 0.95 after round-trip, "
                    f"got {cat.min_confidence}"
                )
                break
        else:
            pytest.fail("PHONE category not found in round-tripped engine")

    def test_s6_removed_rule_not_present_after_roundtrip(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """[S6] A rule removed from a category is absent after snapshot round-trip."""
        phone_no_intl = _phone_without_intl()
        categories = _all_categories_with({"PHONE": phone_no_intl})
        manifest = snapshot_from_categories(
            categories, signer, timestamp="2026-01-01T00:00:00Z"
        )
        rt_engine = engine_from_snapshot(manifest, verifier)

        # Verify that phone_intl is absent
        for cat in rt_engine._categories:
            if cat.category == "PHONE":
                rule_ids = [r.rule_id for r in cat.rules]
                assert "phone_intl" not in rule_ids, (
                    f"phone_intl rule should be absent after round-trip but found: {rule_ids}"
                )
                break
        else:
            pytest.fail("PHONE category not found in round-tripped engine")


# ─────────────────────────────────────────────────────────────────────────────
# [I] Update channel integrity — must reject before metrics are computed
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateChannelIntegrity:
    """
    Verify that the signed-channel gate blocks unsigned/tampered/wrong-key
    manifests before any corpus comparison runs.

    A regression guard that accepts unsigned or tampered updates would give a
    false 'pass' to a maliciously weakened or silently modified rule set.
    """

    def test_i1_unsigned_manifest_rejected(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """[I1] An unsigned manifest raises UpdateRejectedError immediately."""
        # Build but do NOT sign the manifest
        entries = [
            (cat.category, "category_spec", _serialize_category(cat))
            for cat in ALL_CATEGORIES[:3]  # partial snapshot for speed
        ]
        unsigned = make_manifest(
            "1.0.0", "rule_update", entries, timestamp="2026-01-01T00:00:00Z"
        )
        assert unsigned.signature is None

        with pytest.raises(UpdateRejectedError, match="[Ss]ignature|unsigned"):
            engine_from_snapshot(unsigned, verifier)

    def test_i2_tampered_manifest_rejected(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """[I2] Modifying any field after signing raises UpdateRejectedError."""
        manifest = snapshot_from_categories(
            ALL_CATEGORIES[:3],
            signer,
            timestamp="2026-01-01T00:00:00Z",
        )
        tampered = copy.deepcopy(manifest)
        # Tamper with one field — HMAC should catch this
        tampered.entries[0].name = "injected_evil_rule"

        with pytest.raises(UpdateRejectedError):
            engine_from_snapshot(tampered, verifier)

    def test_i3_tampered_entry_content_rejected(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """[I3] Changing entry content (rule definition) after signing is caught."""
        manifest = snapshot_from_categories(
            ALL_CATEGORIES[:3],
            signer,
            timestamp="2026-01-01T00:00:00Z",
        )
        tampered = copy.deepcopy(manifest)
        # Attacker tries to weaken the email rule by changing the content
        original_content = json.loads(tampered.entries[0].content)
        original_content["min_confidence"] = 0.0  # drop threshold to zero
        tampered.entries[0].content = json.dumps(original_content)
        # sha256 field still holds the original hash → content hash mismatch

        with pytest.raises(UpdateRejectedError):
            engine_from_snapshot(tampered, verifier)

    def test_i4_wrong_key_manifest_rejected(
        self,
        key: bytes,
        other_key: bytes,
    ) -> None:
        """[I4] Manifest signed with key A is rejected when verified with key B."""
        signer_a = UpdateSigner(key)
        verifier_b = UpdateVerifier(other_key)

        manifest = snapshot_from_categories(
            ALL_CATEGORIES[:3],
            signer_a,
            timestamp="2026-01-01T00:00:00Z",
        )
        with pytest.raises(UpdateRejectedError, match="[Ss]ignature|HMAC|rejected"):
            engine_from_snapshot(manifest, verifier_b)

    def test_i5_engine_not_built_from_unsigned_snapshot(
        self,
        signer: UpdateSigner,
        verifier: UpdateVerifier,
    ) -> None:
        """
        [I5] engine_from_snapshot raises BEFORE building the Engine.

        The rejection must be immediate on verification failure — the Engine
        constructor must never be called with unverified rule data.
        """
        entries = [("PHONE", "category_spec", _serialize_category(_get_category("PHONE")))]
        unsigned = make_manifest("1.0.0", "rule_update", entries, timestamp="2026-01-01T00:00:00Z")
        # No sign() call

        with pytest.raises(UpdateRejectedError):
            engine_from_snapshot(unsigned, verifier)


# ─────────────────────────────────────────────────────────────────────────────
# [C] Non-regression gate — corpus integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusRegressionGate:
    """
    Corpus-based integration tests for the regression gate.

    These tests load pre/post snapshots, run the full golden corpus through
    both, and check the metric delta.  They are slower than unit tests but
    provide the ground-truth evidence that detection efficacy is maintained.
    """

    def test_c1_identical_snapshot_no_regression(
        self,
        _session_signer: UpdateSigner,
        _session_verifier: UpdateVerifier,
        corpus: KoreanPIICorpus,
        baseline_metrics: Dict[str, Tuple[float, float]],
    ) -> None:
        """
        [C1] A snapshot identical to the baseline produces zero delta → PASS.

        This verifies that the regression guard does not flag a no-op update as
        a regression — a false alarm would block safe updates.
        """
        identical_manifest = snapshot_from_categories(
            ALL_CATEGORIES,
            _session_signer,
            version="identical",
            timestamp="2026-06-01T00:00:00Z",  # different timestamp, same rules
        )
        identical_engine = engine_from_snapshot(identical_manifest, _session_verifier)
        identical_metrics = _compute_snapshot_metrics(identical_engine, corpus)

        violations = check_regression(baseline_metrics, identical_metrics)

        assert not violations, (
            f"Identical snapshot produced regression violations (false alarm):\n"
            + "\n".join(str(v) for v in violations)
        )

    def test_c2_phone_recall_regression_detected(
        self,
        _session_signer: UpdateSigner,
        _session_verifier: UpdateVerifier,
        corpus: KoreanPIICorpus,
        baseline_metrics: Dict[str, Tuple[float, float]],
    ) -> None:
        """
        [C2] Removing phone_intl causes ~17% PHONE recall drop → regression caught.

        The Korean corpus has 5 ``phone_international`` samples (format: +82-10-XXXX-XXXX)
        out of 30 total PHONE samples.  Without the ``phone_intl`` rule, none of
        these are detected.  The resulting ≈16.7% recall drop exceeds the 5%
        threshold → check_regression must report a violation.
        """
        degraded_categories = _all_categories_with({"PHONE": _phone_without_intl()})
        degraded_manifest = snapshot_from_categories(
            degraded_categories,
            _session_signer,
            version="no-phone-intl",
            timestamp="2026-01-01T00:00:00Z",
        )
        degraded_engine = engine_from_snapshot(degraded_manifest, _session_verifier)
        degraded_metrics = _compute_snapshot_metrics(degraded_engine, corpus)

        violations = check_regression(baseline_metrics, degraded_metrics)

        phone_recall_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "recall"
        ]
        assert phone_recall_violations, (
            f"Expected a PHONE recall regression violation but got none.\n"
            f"Baseline PHONE recall: {baseline_metrics['PHONE'][1]:.4f}\n"
            f"Degraded PHONE recall: {degraded_metrics['PHONE'][1]:.4f}\n"
            f"All violations: {violations}"
        )
        # Verify the delta direction is correct (recall went DOWN)
        v = phone_recall_violations[0]
        assert v.delta < 0, (
            f"PHONE recall violation delta should be negative (recall dropped), "
            f"got {v.delta:+.4f}"
        )
        assert v.delta < -_MAX_RECALL_REGRESSION, (
            f"PHONE recall drop should exceed threshold {_MAX_RECALL_REGRESSION}, "
            f"got delta={v.delta:+.4f}"
        )

    def test_c3_rrn_zero_tolerance_regression_detected(
        self,
        _session_signer: UpdateSigner,
        _session_verifier: UpdateVerifier,
        corpus: KoreanPIICorpus,
        baseline_metrics: Dict[str, Tuple[float, float]],
    ) -> None:
        """
        [C3] Any RRN recall drop triggers a violation (zero-tolerance category).

        RRN is a high-severity blocking category.  Stage-1 must reliably detect
        it even when Stage-2 NER is unavailable (graceful degradation backstop).
        The zero-tolerance policy means even a 0.001% recall drop fails CI.
        """
        degraded_categories = _all_categories_with({"RRN": _rrn_no_rules()})
        degraded_manifest = snapshot_from_categories(
            degraded_categories,
            _session_signer,
            version="no-rrn-rules",
            timestamp="2026-01-01T00:00:00Z",
        )
        degraded_engine = engine_from_snapshot(degraded_manifest, _session_verifier)
        degraded_metrics = _compute_snapshot_metrics(degraded_engine, corpus)

        violations = check_regression(baseline_metrics, degraded_metrics)

        rrn_violations = [v for v in violations if v.category == "RRN"]
        assert rrn_violations, (
            f"Expected an RRN recall violation but got none.\n"
            f"Baseline RRN recall: {baseline_metrics['RRN'][1]:.4f}\n"
            f"Degraded RRN recall: {degraded_metrics['RRN'][1]:.4f}"
        )
        # RRN recall should be essentially 0 with no rules
        assert degraded_metrics["RRN"][1] < 0.01, (
            f"Expected near-zero RRN recall with no rules, "
            f"got {degraded_metrics['RRN'][1]:.4f}"
        )

    def test_c4_comparison_report_printed(
        self,
        baseline_metrics: Dict[str, Tuple[float, float]],
        capsys,
    ) -> None:
        """
        [C4] The comparison report is printable and non-empty.

        This ensures the CI output provides human-readable evidence of
        detection efficacy for audit purposes (VerifiedDetection principle).
        """
        # Simulate a no-op update
        identical_metrics = dict(baseline_metrics)
        violations = check_regression(baseline_metrics, identical_metrics)

        report = format_regression_report(
            baseline_metrics,
            identical_metrics,
            violations,
            pre_label="production-v1.0.0",
            post_label="proposed-v1.0.1",
        )
        print(report)

        captured = capsys.readouterr()
        assert "Regression Guard" in captured.out
        assert "RESULT" in captured.out
        assert "PASS" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# [U] Non-regression gate — unit tests on check_regression()
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckRegressionUnit:
    """
    Unit tests for ``check_regression()`` using synthetic metric data.

    These tests do not run the corpus — they directly exercise the comparison
    logic with known inputs.  They run fast and provide precise coverage of
    boundary conditions that are hard to hit with real corpus data.
    """

    # ── Synthetic baseline used by all unit tests ─────────────────────────────

    _SYNTHETIC_BASELINE: Dict[str, Tuple[float, float]] = {
        "PHONE":      (1.000, 0.950),   # (precision, recall)
        "RRN":        (1.000, 0.980),
        "KR_ACCOUNT": (0.984, 0.960),
        "PERSON":     (1.000, 0.560),
        "ADDRESS":    (1.000, 0.840),
    }

    def test_u1_within_tolerance_recall_passes(self) -> None:
        """[U1] A 4% recall drop (< 5% threshold) produces no violations."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        # Drop recall by 0.04 (4% absolute) — just within 5% tolerance
        updated["PHONE"] = (bp, br - 0.04)

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_recall_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "recall"
        ]
        assert not phone_recall_violations, (
            f"4% recall drop should be within tolerance but got violation: "
            f"{phone_recall_violations}"
        )

    def test_u2_beyond_tolerance_recall_fails(self) -> None:
        """[U2] An 8% recall drop (> 5% threshold) generates a violation."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        updated["PHONE"] = (bp, br - 0.08)

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_recall_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "recall"
        ]
        assert phone_recall_violations, (
            f"8% recall drop should produce a violation but got none"
        )
        v = phone_recall_violations[0]
        assert v.delta < -_MAX_RECALL_REGRESSION, (
            f"Violation delta {v.delta:+.4f} should be < -{_MAX_RECALL_REGRESSION}"
        )

    def test_u3_zero_tolerance_any_drop_fails(self) -> None:
        """[U3] Any RRN recall drop — even 0.5% — triggers a violation."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["RRN"]
        # Drop by only 0.005 (0.5% absolute) — tiny but non-zero
        updated["RRN"] = (bp, br - 0.005)

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        rrn_recall_violations = [
            v for v in violations
            if v.category == "RRN" and v.metric == "recall"
        ]
        assert rrn_recall_violations, (
            f"0.5% RRN recall drop should violate zero-tolerance policy but got none"
        )

    def test_u4_zero_tolerance_zero_drop_passes(self) -> None:
        """[U4] Identical RRN recall → no zero-tolerance violation."""
        updated = dict(self._SYNTHETIC_BASELINE)
        # No change to RRN
        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        rrn_violations = [v for v in violations if v.category == "RRN"]
        assert not rrn_violations, (
            f"Zero RRN recall change should not produce a violation: {rrn_violations}"
        )

    def test_u5_within_tolerance_precision_passes(self) -> None:
        """[U5] A 2% precision drop (< 3% threshold) produces no violations."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        updated["PHONE"] = (bp - 0.02, br)   # precision drops by 2%

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_precision_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "precision"
        ]
        assert not phone_precision_violations, (
            f"2% precision drop should be within tolerance but got: {phone_precision_violations}"
        )

    def test_u6_beyond_tolerance_precision_fails(self) -> None:
        """[U6] A 5% precision drop (> 3% threshold) generates a violation."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        updated["PHONE"] = (bp - 0.05, br)   # precision drops by 5%

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_precision_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "precision"
        ]
        assert phone_precision_violations, (
            f"5% precision drop should produce a violation but got none"
        )

    def test_u7_zero_delta_no_violations(self) -> None:
        """[U6] Identical baseline and updated metrics → zero violations."""
        violations = check_regression(
            self._SYNTHETIC_BASELINE,
            dict(self._SYNTHETIC_BASELINE),
        )
        assert not violations, (
            f"Identical metrics should produce no violations: {violations}"
        )

    def test_u8_improvement_no_violations(self) -> None:
        """[U8] Improved recall (positive delta) never triggers a violation."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PERSON"]
        # Improve PERSON recall by 15% (better NER)
        updated["PERSON"] = (bp, min(1.0, br + 0.15))

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        person_violations = [v for v in violations if v.category == "PERSON"]
        assert not person_violations, (
            f"Improved recall should never produce a violation: {person_violations}"
        )

    def test_u9_exact_threshold_boundary_passes(self) -> None:
        """[U9] A recall drop of exactly _MAX_RECALL_REGRESSION passes (boundary).

        The comparison is ``delta < threshold`` (strictly less than), so a drop
        that equals the threshold exactly should NOT trigger a violation.

        We use ``round()`` to avoid floating-point arithmetic giving a result
        infinitesimally beyond the threshold (e.g. 0.95 - 0.05 → 0.899999…).
        """
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        # Round to avoid floating-point representation errors (0.95-0.05=0.8999…)
        updated["PHONE"] = (bp, round(br - _MAX_RECALL_REGRESSION, 10))

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_recall_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "recall"
        ]
        assert not phone_recall_violations, (
            f"Recall drop of exactly {_MAX_RECALL_REGRESSION} (at threshold) "
            f"should not produce a violation: {phone_recall_violations}"
        )

    def test_u10_just_beyond_threshold_fails(self) -> None:
        """[U10] A recall drop of threshold + epsilon fails."""
        epsilon = 1e-6
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        updated["PHONE"] = (bp, br - _MAX_RECALL_REGRESSION - epsilon)

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        phone_recall_violations = [
            v for v in violations
            if v.category == "PHONE" and v.metric == "recall"
        ]
        assert phone_recall_violations, (
            f"Recall drop of threshold + epsilon should produce a violation"
        )

    def test_u11_violation_message_is_informative(self) -> None:
        """[U11] Violation messages contain category name, metric, values, and delta."""
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["PHONE"]
        updated["PHONE"] = (bp, br - 0.10)

        violations = check_regression(self._SYNTHETIC_BASELINE, updated)
        assert violations, "Expected at least one violation"

        v = next(v for v in violations if v.category == "PHONE" and v.metric == "recall")
        assert "PHONE" in v.message
        assert "recall" in v.message.lower() or "RECALL" in v.message
        assert v.delta < 0, "Regression delta should be negative"

    def test_u12_custom_thresholds_respected(self) -> None:
        """[U12] Custom max_recall_regression and max_precision_regression override defaults."""
        # Use a very tight threshold: 1% max recall drop
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["ADDRESS"]
        updated["ADDRESS"] = (bp, br - 0.03)  # 3% drop — within default 5% but beyond custom 1%

        violations_default = check_regression(
            self._SYNTHETIC_BASELINE, updated,
            max_recall_regression=0.05,
        )
        violations_tight = check_regression(
            self._SYNTHETIC_BASELINE, updated,
            max_recall_regression=0.01,
        )

        # Default threshold allows 3% drop
        address_violations_default = [
            v for v in violations_default
            if v.category == "ADDRESS" and v.metric == "recall"
        ]
        assert not address_violations_default, (
            f"3% ADDRESS recall drop should pass with default 5% threshold"
        )

        # Tight threshold rejects 3% drop
        address_violations_tight = [
            v for v in violations_tight
            if v.category == "ADDRESS" and v.metric == "recall"
        ]
        assert address_violations_tight, (
            f"3% ADDRESS recall drop should fail with custom 1% threshold"
        )

    def test_u13_custom_zero_tolerance_set(self) -> None:
        """[U13] Custom zero_tolerance_recall set is respected."""
        # Make ADDRESS zero-tolerance
        updated = dict(self._SYNTHETIC_BASELINE)
        bp, br = self._SYNTHETIC_BASELINE["ADDRESS"]
        updated["ADDRESS"] = (bp, br - 0.001)  # tiny 0.1% drop

        # Default: ADDRESS is NOT zero-tolerance → tiny drop is within 5% threshold
        violations_default = check_regression(
            self._SYNTHETIC_BASELINE, updated,
            zero_tolerance_recall=set(),  # no zero-tolerance categories
        )
        address_violations_default = [
            v for v in violations_default
            if v.category == "ADDRESS" and v.metric == "recall"
        ]
        assert not address_violations_default, (
            f"0.1% ADDRESS recall drop should pass with empty zero-tolerance set"
        )

        # Custom: ADDRESS IS zero-tolerance → any drop fails
        violations_strict = check_regression(
            self._SYNTHETIC_BASELINE, updated,
            zero_tolerance_recall={"ADDRESS"},
        )
        address_violations_strict = [
            v for v in violations_strict
            if v.category == "ADDRESS" and v.metric == "recall"
        ]
        assert address_violations_strict, (
            f"0.1% ADDRESS recall drop should fail with ADDRESS in zero-tolerance set"
        )


# ─────────────────────────────────────────────────────────────────────────────
# [G] Absolute golden floor guard
# ─────────────────────────────────────────────────────────────────────────────

class TestAbsoluteGoldenFloors:
    """
    Verify that the baseline rule set meets absolute recall floors regardless
    of any update.

    These tests are independent of the pre/post delta — they check that even
    the baseline production rules maintain the minimum detection capability
    required for the GracefulDegradation guarantee (Stage-1 as fail-safe backstop).
    """

    @pytest.mark.parametrize("category,floor", sorted(_ABSOLUTE_RECALL_FLOORS.items()))
    def test_g1_baseline_recall_meets_absolute_floor(
        self,
        category: str,
        floor: float,
        baseline_metrics: Dict[str, Tuple[float, float]],
    ) -> None:
        """[G1] Baseline engine meets absolute recall floor for every measured category."""
        _, recall = baseline_metrics[category]
        assert recall >= floor, (
            f"{category} baseline recall {recall:.4f} < absolute floor {floor:.4f}. "
            f"The production rule set does not meet the minimum backstop requirement."
        )

    def test_g2_rrn_high_severity_floor(
        self, baseline_metrics: Dict[str, Tuple[float, float]]
    ) -> None:
        """[G2] RRN recall ≥ 0.90 — Stage-1 must be a reliable backstop for this category."""
        _, rrn_recall = baseline_metrics["RRN"]
        assert rrn_recall >= 0.90, (
            f"RRN baseline recall {rrn_recall:.4f} < 0.90. "
            f"RRN is a high-severity blocking category; Stage-1 must reliably detect it "
            f"to preserve the GracefulDegradation guarantee when Stage-2 NER is unavailable."
        )

    def test_g3_all_measured_categories_have_metrics(
        self, baseline_metrics: Dict[str, Tuple[float, float]]
    ) -> None:
        """[G3] Every category in _MEASURED_CATEGORIES has a metric entry."""
        missing = _MEASURED_CATEGORIES - set(baseline_metrics.keys())
        assert not missing, (
            f"Missing baseline metrics for categories: {sorted(missing)}. "
            f"Ensure the corpus generates samples for these categories."
        )

    def test_g4_absolute_floors_cover_measured_categories(self) -> None:
        """[G4] Every measured category has an absolute floor defined."""
        missing = _MEASURED_CATEGORIES - set(_ABSOLUTE_RECALL_FLOORS.keys())
        assert not missing, (
            f"Categories in _MEASURED_CATEGORIES without absolute floors: {sorted(missing)}. "
            f"Add a floor to _ABSOLUTE_RECALL_FLOORS for each."
        )


# ─────────────────────────────────────────────────────────────────────────────
# [R] Regression comparison report
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionReport:
    """
    Report generation — always passes.  Provides human-readable CI evidence
    of the pre/post metric comparison for audit and debugging.
    """

    def test_r1_full_comparison_report_printed(
        self,
        _session_signer: UpdateSigner,
        _session_verifier: UpdateVerifier,
        corpus: KoreanPIICorpus,
        baseline_metrics: Dict[str, Tuple[float, float]],
        capsys,
    ) -> None:
        """
        [R1] Print the full pre/post comparison table.

        Always passes; read the output for the canonical evidence of detection
        efficacy.  This report is the human-visible record that the regression
        guard was run and the rule set was validated.
        """
        # Simulate a phone-intl removal update to produce a non-trivial comparison
        degraded_categories = _all_categories_with({"PHONE": _phone_without_intl()})
        degraded_manifest = snapshot_from_categories(
            degraded_categories,
            _session_signer,
            version="example-degraded",
            timestamp="2026-01-01T00:00:00Z",
        )
        degraded_engine = engine_from_snapshot(degraded_manifest, _session_verifier)
        degraded_metrics = _compute_snapshot_metrics(degraded_engine, corpus)

        violations = check_regression(baseline_metrics, degraded_metrics)
        report = format_regression_report(
            baseline_metrics,
            degraded_metrics,
            violations,
            pre_label="baseline (ALL_CATEGORIES)",
            post_label="example-degraded (phone_intl removed)",
        )
        print(report)

        captured = capsys.readouterr()
        assert "Regression Guard" in captured.out, (
            "Comparison report was not printed — check format_regression_report()"
        )
        # The degraded update should show a FAIL result in the report
        assert "FAIL" in captured.out, (
            "Expected FAIL in report for degraded snapshot but got PASS"
        )
        # The baseline metrics should all appear
        for cat in sorted(_MEASURED_CATEGORIES):
            assert cat in captured.out, (
                f"Category {cat!r} missing from comparison report"
            )

    def test_r2_passing_report_contains_pass(
        self,
        baseline_metrics: Dict[str, Tuple[float, float]],
        capsys,
    ) -> None:
        """[R2] A no-regression comparison reports PASS."""
        violations = check_regression(baseline_metrics, dict(baseline_metrics))
        report = format_regression_report(
            baseline_metrics,
            dict(baseline_metrics),
            violations,
            pre_label="v1.0.0",
            post_label="v1.0.1 (no changes)",
        )
        print(report)
        captured = capsys.readouterr()
        assert "PASS" in captured.out, "No-regression report should say PASS"
        assert "FAIL" not in captured.out, "No-regression report should not say FAIL"

    def test_r3_violation_details_in_report(self) -> None:
        """[R3] format_regression_report includes per-violation messages."""
        baseline = {"PHONE": (1.0, 0.95), "RRN": (1.0, 0.98)}
        updated  = {"PHONE": (1.0, 0.80), "RRN": (1.0, 0.98)}  # 15% PHONE recall drop

        violations = check_regression(baseline, updated)
        report = format_regression_report(baseline, updated, violations)

        assert violations, "Expected at least one violation"
        assert "FAIL" in report, "Report should say FAIL"
        # The violation message text should appear in the report
        for v in violations:
            # The report truncates messages; check that the key parts appear somewhere
            assert v.category in report, f"{v.category} should appear in report"
