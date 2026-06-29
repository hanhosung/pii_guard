#!/usr/bin/env python3
"""
benchmarks/compare_ner_backends.py

NER Backend Comparison + Adoption Gate (R21 / DESIGN ADR-14)
=============================================================
Runs the Korean NER benchmark across multiple Stage-2 backends on the SAME
labeled corpus (and optionally across a confidence-threshold sweep), then emits
a side-by-side precision/recall/F1 comparison table in Markdown and evaluates
the **ADR-14 adoption gate** for the NuNER Zero candidate against the GLiNER
baseline.

This is the orchestration layer of the ADR-14 procedure:

    wiring (done) → BENCH COMPARE (this script) → adoption gate → promote

It does not download or fine-tune anything itself — it calls
``korean_ner_benchmark.run_benchmark`` once per (backend, threshold) cell and
tabulates the results. Backends whose deps/models are unavailable are reported
as "unavailable" rather than crashing the whole comparison.

Adoption gate (ADR-14), candidate vs. baseline (default baseline = gliner):
  (a) NO recall regression in ANY category (candidate recall >= baseline recall
      minus --recall-tolerance), AND
  (b) ORG precision improvement OR overall (macro) F1 win.
Runtime budget (memory/cold-load/p95) is NOT measured here — verify separately.

Usage
-----
    # Compare gliner vs nunerzero at the default 0.50 threshold:
    python benchmarks/compare_ner_backends.py

    # Include spacy, sweep two thresholds, write a Markdown report, gate exit code:
    python benchmarks/compare_ner_backends.py \\
        --backends gliner,nunerzero,spacy \\
        --min-confidence 0.50,0.35 \\
        --samples-per-format 10 \\
        --output validation/NER_BACKEND_COMPARISON_nunerzero.md \\
        --gate

Exit codes
----------
  0  Report written (gate passed, or --gate not given)
  1  --gate given and the candidate FAILED the adoption gate
  2  No backend produced usable (NER-available) metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.korean_ner_benchmark import (  # noqa: E402
    NER_OWNED_CATEGORIES,
    run_benchmark,
)

#: Default backends to compare (baseline first).
_DEFAULT_BACKENDS = ["gliner", "nunerzero"]
#: Adoption-gate baseline backend.
_BASELINE_BACKEND = "gliner"


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro_f1(metrics: Dict[str, Dict[str, object]]) -> float:
    """Unweighted mean F1 across the NER-owned categories."""
    f1s = [
        _f1(float(metrics[c]["precision"]), float(metrics[c]["recall"]))
        for c in NER_OWNED_CATEGORIES
        if c in metrics
    ]
    return sum(f1s) / len(f1s) if f1s else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Run the benchmark for each (backend, threshold) cell
# ─────────────────────────────────────────────────────────────────────────────

def _run_cells(
    backends: List[str],
    thresholds: List[float],
    corpus_seed: int,
    samples_per_format: int,
    quiet: bool,
) -> List[Dict]:
    """
    Run run_benchmark for every (backend, threshold) combination.

    Returns a list of cell dicts:
      {backend, min_confidence, ner_available, metrics, macro_f1, load_error}
    """
    cells: List[Dict] = []
    for backend in backends:
        for thr in thresholds:
            if not quiet:
                print(
                    f"[compare] running backend={backend} min_confidence={thr} ...",
                    file=sys.stderr,
                )
            report = run_benchmark(
                corpus_seed=corpus_seed,
                samples_per_format=samples_per_format,
                min_confidence=thr,
                no_ner=False,
                apply_thresholds=False,
                quiet=quiet,
                ner_backend=backend,
            )
            metrics = report["full_pipeline_metrics"]
            cells.append({
                "backend": backend,
                "min_confidence": thr,
                "ner_available": report["metadata"]["ner_available"],
                "load_error": report["metadata"].get("ner_load_error"),
                "metrics": metrics,
                "macro_f1": round(_macro_f1(metrics), 4),
            })
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Adoption gate (ADR-14)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_gate(
    cells: List[Dict],
    candidate: str,
    baseline: str,
    recall_tolerance: float,
) -> Dict:
    """
    Evaluate the ADR-14 adoption gate for *candidate* vs *baseline*, comparing
    at each shared threshold. Returns a verdict dict (passed + reasons).

    A candidate "passes at a threshold" when, at that threshold:
      (a) no category's recall drops below baseline recall - recall_tolerance, AND
      (b) ORG precision improves OR candidate macro-F1 >= baseline macro-F1.
    The overall gate passes if the candidate passes at ANY shared threshold.
    """
    def cell_for(backend: str, thr: float) -> Optional[Dict]:
        for c in cells:
            if c["backend"] == backend and c["min_confidence"] == thr and c["ner_available"]:
                return c
        return None

    thresholds = sorted({c["min_confidence"] for c in cells})
    per_threshold: List[Dict] = []
    any_pass = False

    for thr in thresholds:
        base = cell_for(baseline, thr)
        cand = cell_for(candidate, thr)
        if base is None or cand is None:
            per_threshold.append({
                "min_confidence": thr,
                "evaluable": False,
                "note": f"missing usable metrics for {baseline if base is None else candidate}",
            })
            continue

        bm, cm = base["metrics"], cand["metrics"]
        recall_regressions = []
        for cat in NER_OWNED_CATEGORIES:
            if cat in bm and cat in cm:
                drop = float(bm[cat]["recall"]) - float(cm[cat]["recall"])
                if drop > recall_tolerance:
                    recall_regressions.append(
                        f"{cat} recall {float(cm[cat]['recall']):.3f} < "
                        f"{float(bm[cat]['recall']):.3f} (baseline) by {drop:.3f}"
                    )
        no_recall_regression = not recall_regressions

        org_prec_gain = (
            "ORGANIZATION" in bm and "ORGANIZATION" in cm
            and float(cm["ORGANIZATION"]["precision"]) > float(bm["ORGANIZATION"]["precision"])
        )
        f1_win = cand["macro_f1"] >= base["macro_f1"]
        quality_improved = org_prec_gain or f1_win

        passed = no_recall_regression and quality_improved
        any_pass = any_pass or passed
        per_threshold.append({
            "min_confidence": thr,
            "evaluable": True,
            "passed": passed,
            "no_recall_regression": no_recall_regression,
            "recall_regressions": recall_regressions,
            "org_precision_gain": org_prec_gain,
            "macro_f1_win": f1_win,
            "candidate_macro_f1": cand["macro_f1"],
            "baseline_macro_f1": base["macro_f1"],
        })

    return {
        "candidate": candidate,
        "baseline": baseline,
        "recall_tolerance": recall_tolerance,
        "passed": any_pass,
        "per_threshold": per_threshold,
        "note": (
            "Runtime budget (memory/cold-load/p95) is NOT checked here — "
            "verify separately before promotion (ADR-14)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(
    cells: List[Dict],
    thresholds: List[float],
    backends: List[str],
    gate: Optional[Dict],
    corpus_seed: int,
    samples_per_format: int,
) -> str:
    """Render the comparison as a Markdown document (string)."""
    lines: List[str] = []
    lines.append("# NER Backend Comparison — GLiNER vs spaCy vs NuNER Zero (R21 / ADR-14)")
    lines.append("")
    lines.append(
        f"> Corpus seed={corpus_seed} · samples_per_format={samples_per_format} · "
        f"full-pipeline (Stage1 + Stage2 NER) · auto-generated by "
        f"`benchmarks/compare_ner_backends.py`."
    )
    lines.append("")

    def cell_for(backend: str, thr: float) -> Optional[Dict]:
        for c in cells:
            if c["backend"] == backend and c["min_confidence"] == thr:
                return c
        return None

    for thr in thresholds:
        lines.append(f"## min_confidence = {thr}")
        lines.append("")
        header = "| 카테고리 | " + " | ".join(
            f"{b} P / R / F1" for b in backends
        ) + " |"
        sep = "| :-- | " + " | ".join(":--" for _ in backends) + " |"
        lines.append(header)
        lines.append(sep)
        for cat in NER_OWNED_CATEGORIES:
            row = [cat]
            for b in backends:
                c = cell_for(b, thr)
                if c is None or not c["ner_available"] or cat not in c["metrics"]:
                    row.append("n/a")
                    continue
                m = c["metrics"][cat]
                p, r = float(m["precision"]), float(m["recall"])
                row.append(f"{p:.3f} / {r:.3f} / {_f1(p, r):.3f}")
            lines.append("| " + " | ".join(row) + " |")
        # macro-F1 summary row
        macro_row = ["**macro-F1**"]
        for b in backends:
            c = cell_for(b, thr)
            macro_row.append(
                f"**{c['macro_f1']:.3f}**" if c and c["ner_available"] else "n/a"
            )
        lines.append("| " + " | ".join(macro_row) + " |")
        lines.append("")

    # Unavailable backends note (honest, P3 — no silent gaps).
    unavailable = [
        f"`{c['backend']}`@{c['min_confidence']}: {c['load_error']}"
        for c in cells if not c["ner_available"]
    ]
    if unavailable:
        lines.append("## ⚠️ Unavailable cells (deps/model missing — ran Stage-1 only)")
        lines.append("")
        for u in unavailable:
            lines.append(f"- {u}")
        lines.append("")

    if gate is not None:
        lines.append(f"## Adoption gate (ADR-14) — `{gate['candidate']}` vs `{gate['baseline']}`")
        lines.append("")
        lines.append(f"**Verdict: {'PASS ✅' if gate['passed'] else 'FAIL ❌'}**")
        lines.append("")
        lines.append(
            f"Criteria per shared threshold: (a) no recall regression "
            f"(tolerance {gate['recall_tolerance']}) in any category, AND "
            f"(b) ORG precision gain OR macro-F1 win. Passes overall if it "
            f"passes at any threshold."
        )
        lines.append("")
        lines.append("| min_conf | recall no-regress | ORG prec gain | macro-F1 win | pass |")
        lines.append("| :-- | :-- | :-- | :-- | :-- |")
        for t in gate["per_threshold"]:
            if not t.get("evaluable"):
                lines.append(f"| {t['min_confidence']} | — | — | — | n/a ({t['note']}) |")
                continue
            lines.append(
                f"| {t['min_confidence']} "
                f"| {'✓' if t['no_recall_regression'] else '✗'} "
                f"| {'✓' if t['org_precision_gain'] else '✗'} "
                f"| {'✓' if t['macro_f1_win'] else '✗'} "
                f"({t['candidate_macro_f1']:.3f} vs {t['baseline_macro_f1']:.3f}) "
                f"| {'✅' if t['passed'] else '❌'} |"
            )
        # Spell out any recall regressions for diagnosis.
        regressions = [
            (t["min_confidence"], reg)
            for t in gate["per_threshold"] if t.get("evaluable")
            for reg in t.get("recall_regressions", [])
        ]
        if regressions:
            lines.append("")
            lines.append("Recall regressions:")
            for thr, reg in regressions:
                lines.append(f"- @{thr}: {reg}")
        lines.append("")
        lines.append(f"> {gate['note']}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage-2 NER backends + evaluate the ADR-14 adoption gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(_DEFAULT_BACKENDS),
        help="Comma-separated backends to compare (default: gliner,nunerzero). "
             "Baseline for the gate is 'gliner'.",
    )
    parser.add_argument(
        "--min-confidence",
        type=str,
        default="0.50",
        help="Comma-separated confidence threshold(s) to sweep (default: 0.50).",
    )
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--samples-per-format", type=int, default=5)
    parser.add_argument(
        "--candidate", type=str, default="nunerzero",
        help="Candidate backend for the adoption gate (default: nunerzero).",
    )
    parser.add_argument(
        "--recall-tolerance", type=float, default=0.0,
        help="Allowed recall drop vs baseline before counting it a regression "
             "(default: 0.0 = strict).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write the Markdown report to PATH (default: stdout).",
    )
    parser.add_argument(
        "--json-output", type=str, default=None,
        help="Also write the raw cells+gate as JSON to PATH.",
    )
    parser.add_argument(
        "--gate", action="store_true", default=False,
        help="Exit 1 if the candidate FAILS the adoption gate.",
    )
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    thresholds = [float(t) for t in args.min_confidence.split(",") if t.strip()]

    cells = _run_cells(
        backends=backends,
        thresholds=thresholds,
        corpus_seed=args.corpus_seed,
        samples_per_format=args.samples_per_format,
        quiet=args.quiet,
    )

    if not any(c["ner_available"] for c in cells):
        print(
            "[compare] ERROR: no backend produced usable NER metrics "
            "(install [ner]/[ner-gliner] and let models download).",
            file=sys.stderr,
        )
        sys.exit(2)

    gate = None
    if args.candidate in backends and _BASELINE_BACKEND in backends:
        gate = _evaluate_gate(
            cells=cells,
            candidate=args.candidate,
            baseline=_BASELINE_BACKEND,
            recall_tolerance=args.recall_tolerance,
        )

    markdown = _render_markdown(
        cells=cells,
        thresholds=thresholds,
        backends=backends,
        gate=gate,
        corpus_seed=args.corpus_seed,
        samples_per_format=args.samples_per_format,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(markdown)
            fh.write("\n")
        if not args.quiet:
            print(f"[compare] Markdown report written to {args.output}", file=sys.stderr)
    else:
        print(markdown)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as fh:
            json.dump({"cells": cells, "gate": gate}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        if not args.quiet:
            print(f"[compare] JSON written to {args.json_output}", file=sys.stderr)

    if args.gate and gate is not None and not gate["passed"]:
        print("[compare] adoption gate FAILED for candidate "
              f"'{gate['candidate']}'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
