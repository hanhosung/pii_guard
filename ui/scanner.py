"""
Pure scan/format logic for the PII-Guard UI — no Streamlit dependency, so it is
unit-testable and reusable (e.g. by a CLI or the Streamlit app in app.py).
"""
from __future__ import annotations

from typing import List, Tuple

from pii_guard.engine import Engine


def action_name(action) -> str:
    return str(action).split(".")[-1]


def scan_text(engine: Engine, text: str) -> dict:
    """Scan one text and return a structured, render-ready result."""
    result = engine.scan(text)
    rows = [
        {
            "category": d.category,
            "action": action_name(d.action),
            "stage": str(d.detection_stage).split(".")[-1],
            "original": d.original,
            "placeholder": d.placeholder_token,
            "confidence": round(float(d.confidence), 2),
        }
        for d in result.detections
    ]
    return {
        "original": text,
        "masked": result.redacted_text,
        "rows": rows,
        "has_blocks": result.has_blocks,
        "has_masks": result.has_masks,
        "coverage_gap": result.coverage_gap,
        "stage2_gap_reason": result.stage2_gap_reason,
    }


def verdict(res: dict) -> Tuple[str, str]:
    """(label, streamlit-color) for the overall decision of one input."""
    if res["has_blocks"]:
        return "🔴 BLOCKED — 차단됨 (업스트림 미전달, fail-closed)", "red"
    if res["has_masks"]:
        return "🟡 MASKED — 마스킹 후 전달 가능", "orange"
    return "🟢 CLEAN — PII 미검출", "green"


def render_console_block(label: str, res: dict) -> str:
    """Build a plain-text console-style report for one scanned input."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"INPUT: {label}")
    lines.append("-" * 70)
    lines.append(f"VERDICT: {verdict(res)[0]}")
    lines.append("")
    lines.append("ORIGINAL:")
    lines.append(f"  {res['original']}")
    lines.append("")
    lines.append("MASKED (forwarded to LLM):")
    lines.append(f"  {res['masked']}")
    lines.append("")
    if res["rows"]:
        lines.append(f"DETECTIONS ({len(res['rows'])}):")
        for r in res["rows"]:
            mark = "BLOCK" if r["action"] == "BLOCK" else "mask "
            lines.append(
                f"  [{mark}] {r['category']:<13} {r['original']!r}"
                f"  → {r['placeholder']}  ({r['stage']}, conf={r['confidence']})"
            )
    else:
        lines.append("DETECTIONS: none")
    if res["coverage_gap"]:
        lines.append("")
        lines.append(f"!!  COVERAGE GAP: {res['stage2_gap_reason']} "
                     f"(Stage-2 degraded to Stage-1)")
    lines.append("=" * 70)
    return "\n".join(lines)
