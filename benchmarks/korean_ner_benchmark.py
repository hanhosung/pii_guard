#!/usr/bin/env python3
"""
benchmarks/korean_ner_benchmark.py

Korean NER Precision/Recall Benchmark
======================================
Evaluates the full NER pipeline (Stage-1 regex/checksum + Stage-2 Presidio/spaCy)
against the labeled Korean PII corpus and reports per-entity-type precision and recall
for the three entity types the NER engine is responsible for:

  * PERSON       — person names detected by NER (including unlabelled sentence context)
  * ADDRESS      — location/address strings detected by NER
  * ORGANIZATION — Korean organisation names (only detectable via NER)

The benchmark runs two detection modes for comparison:

  1. Stage-1 only  — regex + checksum baseline (no NER model loaded)
  2. Full pipeline — Stage-1 + Stage-2 NER merged

For each mode, per-entity precision and recall are computed against the
ground-truth annotations in NERBenchmarkCorpus and written to a parseable
JSON metrics report.

Usage
-----
    # Minimal (JSON to stdout):
    python benchmarks/korean_ner_benchmark.py

    # Write report to file and gate on minimum thresholds:
    python benchmarks/korean_ner_benchmark.py \\
        --output /tmp/ner_metrics.json \\
        --thresholds \\
        --samples-per-format 10

    # Run Stage-1 baseline only (no NER model required):
    python benchmarks/korean_ner_benchmark.py --no-ner

Output JSON schema
------------------
{
  "metadata": {
    "corpus_seed": 42,
    "samples_per_format": 5,
    "ner_available": true,
    "pipeline": "stage1+stage2_ner",
    "min_confidence": 0.5
  },
  "stage1_metrics": {
    "PERSON":       {"precision": 0.9, "recall": 0.56, "tp": 25, "fp": 0, "fn": 20},
    "ADDRESS":      {"precision": 1.0, "recall": 0.84, "tp": 21, "fp": 0, "fn":  4},
    "ORGANIZATION": {"precision": 1.0, "recall": 0.0,  "tp":  0, "fp": 0, "fn": 25}
  },
  "full_pipeline_metrics": {
    "PERSON":       {"precision": 0.85, "recall": 0.72, "tp": 36, "fp": 6, "fn": 14},
    "ADDRESS":      {"precision": 0.90, "recall": 0.88, "tp": 22, "fp": 2, "fn":  3},
    "ORGANIZATION": {"precision": 0.88, "recall": 0.60, "tp": 15, "fp": 2, "fn": 10}
  },
  "ner_contribution": {
    "PERSON":       {"recall_gain": 0.16, "precision_delta": -0.05},
    "ADDRESS":      {"recall_gain": 0.04, "precision_delta": -0.10},
    "ORGANIZATION": {"recall_gain": 0.60, "precision_delta": -0.12}
  },
  "thresholds_met": true,
  "threshold_failures": []
}

Threshold defaults (used when --thresholds is given)
-----------------------------------------------------
Full pipeline (Stage-1 + NER):
  PERSON:       precision >= 0.50, recall >= 0.40
  ADDRESS:      precision >= 0.55, recall >= 0.45
  ORGANIZATION: precision >= 0.50, recall >= 0.40

These are intentionally conservative to account for the ``ko_core_news_sm``
small model's moderate precision/recall on Korean NER tasks.  Larger or
fine-tuned models would justify tighter floors.

Exit codes
----------
  0  Report written (thresholds met, or --thresholds not given)
  1  One or more threshold failures (only when --thresholds is given)
  2  Import / corpus error (NER not installed, model not downloaded, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so the script works when run from the repo root
# as well as when the package is installed.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# NER categories owned by Stage-2 (the types this benchmark measures)
# ─────────────────────────────────────────────────────────────────────────────
NER_OWNED_CATEGORIES: List[str] = ["PERSON", "ADDRESS", "ORGANIZATION"]

# Default minimum thresholds for the full-pipeline mode.
# Conservative to accommodate the small ko_core_news_sm model.
DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "PERSON":       {"precision": 0.50, "recall": 0.40},
    "ADDRESS":      {"precision": 0.55, "recall": 0.45},
    "ORGANIZATION": {"precision": 0.50, "recall": 0.40},
}


# ─────────────────────────────────────────────────────────────────────────────
# Precision / recall helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    positive_samples,
    negative_samples,
    detector_fn: Callable[[str], Set[str]],
    category: str,
) -> Dict[str, object]:
    """
    Compute precision/recall for *category* given a binary category-level
    detector_fn.

    Parameters
    ----------
    positive_samples:
        Corpus samples that contain at least one span of *category*.
    negative_samples:
        Corpus samples that contain NO spans of *category* (used for FP count).
    detector_fn:
        Callable(text) → frozenset/set of detected category names.
    category:
        The category to evaluate.

    Returns
    -------
    dict with keys: precision, recall, tp, fp, fn
    """
    tp = fp = fn = 0

    for sample in positive_samples:
        detected = category in detector_fn(sample.text)
        if detected:
            tp += 1
        else:
            fn += 1

    for sample in negative_samples:
        detected = category in detector_fn(sample.text)
        if detected:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _build_detector_fn(
    engine,
    ner_engine=None,
) -> Callable[[str], Set[str]]:
    """
    Build a combined detector function.

    When *ner_engine* is provided the detector runs Stage-1 (via *engine*)
    then Stage-2 NER and returns the union of detected categories.
    Without *ner_engine* only Stage-1 is applied.

    Parameters
    ----------
    engine:
        :class:`~pii_guard.engine.Engine` instance (Stage-1).
    ner_engine:
        :class:`~pii_guard.stage2.korean_ner.KoreanNEREngine` instance
        (Stage-2), or ``None`` for Stage-1 only.

    Returns
    -------
    callable(text: str) -> set[str]
    """
    def detect(text: str) -> Set[str]:
        # Stage-1
        s1_result = engine.scan(text)
        cats: Set[str] = {d.category for d in s1_result.detections}

        # Stage-2 NER (optional)
        if ner_engine is not None:
            s2_detections = ner_engine.detect(text)
            cats.update(d.category for d in s2_detections)

        return cats

    return detect


def _run_evaluation(
    corpus,
    detector_fn: Callable[[str], Set[str]],
) -> Dict[str, Dict[str, object]]:
    """
    Evaluate *detector_fn* against *corpus* for all NER-owned categories.

    Uses NER-clean negative samples for precision measurement so that
    corpus negatives containing real org names do not inflate FP counts.

    Returns
    -------
    dict mapping category name → metrics dict
    """
    # NER-clean negatives are the ground-truth negatives for this benchmark.
    if hasattr(corpus, "ner_clean_negatives"):
        ner_negatives = corpus.ner_clean_negatives()
    else:
        ner_negatives = corpus.negative_samples()

    metrics: Dict[str, Dict[str, object]] = {}
    for category in NER_OWNED_CATEGORIES:
        positive = corpus.samples_for_category(category)
        metrics[category] = _compute_metrics(
            positive_samples=positive,
            negative_samples=ner_negatives,
            detector_fn=detector_fn,
            category=category,
        )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Threshold gate
# ─────────────────────────────────────────────────────────────────────────────

def _check_thresholds(
    metrics: Dict[str, Dict[str, object]],
    thresholds: Dict[str, Dict[str, float]] = DEFAULT_THRESHOLDS,
) -> Tuple[bool, List[str]]:
    """
    Check whether all per-category metrics meet the minimum thresholds.

    Returns
    -------
    (all_met: bool, failures: List[str])
        *failures* is a list of human-readable failure descriptions.
    """
    failures: List[str] = []
    for category, floors in thresholds.items():
        if category not in metrics:
            failures.append(f"{category}: not measured")
            continue
        m = metrics[category]
        for metric_name, floor in floors.items():
            actual = m.get(metric_name, 0.0)
            if actual < floor:
                failures.append(
                    f"{category} {metric_name}: {actual:.4f} < {floor:.4f} (floor)"
                )
    return (len(failures) == 0), failures


# ─────────────────────────────────────────────────────────────────────────────
# NER contribution delta
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ner_contribution(
    stage1: Dict[str, Dict[str, object]],
    full: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """Compute how much Stage-2 NER improved recall and changed precision."""
    contribution: Dict[str, Dict[str, float]] = {}
    for cat in NER_OWNED_CATEGORIES:
        if cat not in stage1 or cat not in full:
            continue
        r1 = float(stage1[cat].get("recall", 0.0))
        r2 = float(full[cat].get("recall", 0.0))
        p1 = float(stage1[cat].get("precision", 1.0))
        p2 = float(full[cat].get("precision", 1.0))
        contribution[cat] = {
            "recall_gain": round(r2 - r1, 4),
            "precision_delta": round(p2 - p1, 4),
        }
    return contribution


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Korean NER precision/recall benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for the NERBenchmarkCorpus (default: 42)",
    )
    parser.add_argument(
        "--samples-per-format",
        type=int,
        default=5,
        metavar="INT",
        help="Samples per format variant (default: 5)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        metavar="FLOAT",
        help="NER minimum confidence threshold (default: 0.50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Write JSON report to PATH (default: stdout)",
    )
    parser.add_argument(
        "--thresholds",
        action="store_true",
        default=False,
        help="Exit 1 if any metric falls below the minimum threshold floors",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        default=False,
        help="Run Stage-1 only (skip NER; useful if presidio/spaCy not installed)",
    )
    parser.add_argument(
        "--ner-backend",
        choices=("spacy", "gliner", "nunerzero"),
        default="spacy",
        help="Stage-2 NER backend to benchmark: spacy | gliner | nunerzero "
             "(nunerzero = NuNER Zero candidate, R21/ADR-14). Default: spacy",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress messages on stderr",
    )
    return parser.parse_args(argv)


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[ner-benchmark] {msg}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    corpus_seed: int = 42,
    samples_per_format: int = 5,
    min_confidence: float = 0.50,
    no_ner: bool = False,
    apply_thresholds: bool = False,
    quiet: bool = False,
    ner_backend: str = "spacy",
) -> Dict:
    """
    Run the Korean NER benchmark and return the metrics report dict.

    This function contains the full benchmark logic and is importable for use
    in pytest tests (so the test does not need to spawn a subprocess).

    Parameters
    ----------
    corpus_seed:
        Random seed for corpus generation.
    samples_per_format:
        Number of samples per format variant.
    min_confidence:
        NER confidence threshold below which detections are discarded.
    no_ner:
        When True, skip Stage-2 NER and run Stage-1 only.
    apply_thresholds:
        When True, check metrics against DEFAULT_THRESHOLDS.
    quiet:
        Suppress progress messages on stderr.

    Returns
    -------
    dict — the metrics report (same structure as the JSON output).
    """
    # ── Import corpus ─────────────────────────────────────────────────────────
    try:
        from pii_guard.corpus.ner_benchmark_corpus import NERBenchmarkCorpus
    except ImportError as exc:
        _log(f"ERROR: could not import NERBenchmarkCorpus: {exc}", quiet)
        sys.exit(2)

    _log(
        f"Building corpus (seed={corpus_seed}, samples_per_format={samples_per_format}) ...",
        quiet,
    )
    corpus = NERBenchmarkCorpus(
        seed=corpus_seed,
        samples_per_format=samples_per_format,
    )
    _log(
        f"  corpus: {len(corpus.positive_samples())} positive, "
        f"{len(corpus.negative_samples())} negative, "
        f"{len(corpus.ner_clean_negatives())} NER-clean negatives",
        quiet,
    )

    # ── Import Stage-1 engine ─────────────────────────────────────────────────
    try:
        from pii_guard import Engine
    except ImportError as exc:
        _log(f"ERROR: could not import pii_guard.Engine: {exc}", quiet)
        sys.exit(2)

    stage1_engine = Engine()

    # ── Try to load Stage-2 NER engine ───────────────────────────────────────
    ner_engine = None
    ner_available = False
    ner_load_error: Optional[str] = None

    if not no_ner:
        try:
            # R18/R21: select the NER backend engine
            # (spacy default / gliner / nunerzero candidate).
            if ner_backend == "gliner":
                from pii_guard.stage2.gliner_ner import GLiNERNEREngine
                _log("Loading GLiNERNEREngine (GLiNER multilingual PII model) ...", quiet)
                ner_engine = GLiNERNEREngine(min_confidence=min_confidence)
            elif ner_backend == "nunerzero":
                from pii_guard.stage2.nunerzero_ner import NuNERZeroNEREngine
                _log("Loading NuNERZeroNEREngine (NuNER Zero, R21/ADR-14 candidate) ...", quiet)
                ner_engine = NuNERZeroNEREngine(min_confidence=min_confidence)
            else:
                from pii_guard.stage2.korean_ner import KoreanNEREngine
                _log("Loading KoreanNEREngine (Presidio + spaCy ko) ...", quiet)
                ner_engine = KoreanNEREngine(min_confidence=min_confidence)
            # Warm up — triggers lazy model loading
            ner_engine.detect("테스트")
            ner_available = True
            _log("  NER engine ready.", quiet)
        except RuntimeError as exc:
            ner_load_error = str(exc)
            _log(
                f"  WARNING: NER engine not available — running Stage-1 only.\n"
                f"  Reason: {ner_load_error}\n"
                f"  To enable NER: install the backend ([ner] for spacy, "
                f"[ner-gliner] for gliner AND nunerzero) and let the model download.",
                quiet,
            )
        except ImportError as exc:
            ner_load_error = str(exc)
            _log(f"  WARNING: NER backend deps not installed: {ner_load_error}", quiet)

    # ── Build detector functions ──────────────────────────────────────────────
    stage1_detector = _build_detector_fn(stage1_engine, ner_engine=None)
    full_detector = _build_detector_fn(stage1_engine, ner_engine=ner_engine)

    # ── Evaluate Stage-1 baseline ─────────────────────────────────────────────
    _log("Evaluating Stage-1 baseline ...", quiet)
    stage1_metrics = _run_evaluation(corpus, stage1_detector)

    # ── Evaluate full pipeline ────────────────────────────────────────────────
    _log(
        "Evaluating full pipeline (Stage-1 + NER) ..." if ner_available
        else "Evaluating Stage-1 (NER unavailable) ...",
        quiet,
    )
    full_pipeline_metrics = _run_evaluation(corpus, full_detector)

    # ── NER contribution delta ────────────────────────────────────────────────
    ner_contribution = _compute_ner_contribution(stage1_metrics, full_pipeline_metrics)

    # ── Threshold check ───────────────────────────────────────────────────────
    if apply_thresholds and ner_available:
        thresholds_met, threshold_failures = _check_thresholds(full_pipeline_metrics)
    elif apply_thresholds and not ner_available:
        # Can only check thresholds against Stage-1
        thresholds_met, threshold_failures = _check_thresholds(stage1_metrics)
    else:
        thresholds_met = True
        threshold_failures = []

    # ── Log per-category summary ──────────────────────────────────────────────
    if not quiet:
        _log("", quiet=False)
        _log("─" * 70, quiet=False)
        _log(
            f"{'Category':<16} {'Stage1 P':>10} {'Stage1 R':>10} "
            f"{'Full P':>10} {'Full R':>10} {'R gain':>10}",
            quiet=False,
        )
        _log("─" * 70, quiet=False)
        for cat in NER_OWNED_CATEGORIES:
            s1 = stage1_metrics.get(cat, {})
            fp = full_pipeline_metrics.get(cat, {})
            cont = ner_contribution.get(cat, {})
            _log(
                f"{cat:<16} "
                f"{s1.get('precision', 0):.3f}{'':>4} "
                f"{s1.get('recall', 0):.3f}{'':>4} "
                f"{fp.get('precision', 0):.3f}{'':>4} "
                f"{fp.get('recall', 0):.3f}{'':>4} "
                f"{cont.get('recall_gain', 0):+.3f}",
                quiet=False,
            )
        _log("─" * 70, quiet=False)
        if threshold_failures:
            _log("THRESHOLD FAILURES:", quiet=False)
            for msg in threshold_failures:
                _log(f"  ✗ {msg}", quiet=False)
        else:
            _log("All threshold checks passed.", quiet=False)
        _log("", quiet=False)

    # ── Build report ─────────────────────────────────────────────────────────
    report: Dict = {
        "metadata": {
            "corpus_seed": corpus_seed,
            "samples_per_format": samples_per_format,
            "ner_available": ner_available,
            "ner_backend": ner_backend,
            "pipeline": "stage1+stage2_ner" if ner_available else "stage1_only",
            "min_confidence": min_confidence,
            "ner_load_error": ner_load_error,
            "corpus_sizes": {
                "positive": len(corpus.positive_samples()),
                "negative": len(corpus.negative_samples()),
                "ner_clean_negatives": len(corpus.ner_clean_negatives()),
                "org_samples": len(corpus.samples_for_category("ORGANIZATION")),
            },
        },
        "stage1_metrics": stage1_metrics,
        "full_pipeline_metrics": full_pipeline_metrics,
        "ner_contribution": ner_contribution,
        "thresholds_met": thresholds_met,
        "threshold_failures": threshold_failures,
    }

    return report


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    report = run_benchmark(
        corpus_seed=args.corpus_seed,
        samples_per_format=args.samples_per_format,
        min_confidence=args.min_confidence,
        no_ner=args.no_ner,
        apply_thresholds=args.thresholds,
        quiet=args.quiet,
        ner_backend=args.ner_backend,
    )

    json_output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(json_output)
            fh.write("\n")
        _log(f"Report written to {args.output}", quiet=args.quiet)
    else:
        print(json_output)

    # Exit 1 when --thresholds was given and some floors were missed
    if args.thresholds and not report["thresholds_met"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
