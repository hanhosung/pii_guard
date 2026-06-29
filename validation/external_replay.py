#!/usr/bin/env python3
"""
validation/external_replay.py

External-LLM test REPLAY harness (R21 / ADR-14, NuNER Zero external evaluation)
==============================================================================
The original external-LLM test inputs (`*_cases.json`) and harness
(`efficacy_test.py`) are not in the repo (one-off / gitignored). BUT every
`EXTERNAL_LLM_TEST_2026-06-23_{dataset}_GLiNER.md` report embeds, per case:
the full input **text** and the **ground-truth PII** (codex/gemini have an
explicit `심은(...)` line; claude has `검출(...)` ∪ `미검출(...)`).

This script:
  1. PARSES those three GLiNER reports → reconstructs the 50-case dataset
     (claude 30 + codex 10 + gemini 10) as {dataset, id, title, text, truth[]}.
  2. REPLAYS each text through `Engine` + `Stage2NERRunner` for a chosen NER
     backend, scoring TP / FN / FP against the reconstructed ground truth using
     the same matching rules the originals describe (category-compatible +
     span/value containment; SECRET-class interchange; LOCATION≈ADDRESS).
  3. WRITES per-dataset Markdown reports in the existing house style plus a JSON
     metrics dump.

Because the ground truth is reconstructed (not the original hand-authored JSON),
run BOTH `gliner` (re-scored baseline) and `nunerzero` through this SAME harness
so the comparison is apples-to-apples. Raw FP is reported (no manual triage),
exactly as the original aggregate did.

Usage
-----
    # Reconstruct + replay one backend, write 3 reports + JSON:
    python validation/external_replay.py --ner-backend nunerzero
    python validation/external_replay.py --ner-backend gliner     # baseline

    # Just dump the reconstructed dataset (no model load):
    python validation/external_replay.py --dump-cases-only

Exit codes: 0 ok · 2 reconstruction/import error
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

#: Source GLiNER reports (ground truth carrier) per dataset.
_SOURCES = {
    "claude": "EXTERNAL_LLM_TEST_2026-06-23_claude_GLiNER.md",
    "codex": "EXTERNAL_LLM_TEST_2026-06-23_codex_GLiNER.md",
    "gemini": "EXTERNAL_LLM_TEST_2026-06-23_gemini_GLiNER.md",
}

#: NER-owned categories (where backends differ). Stage1 categories are identical.
_NER_CATEGORIES = {"PERSON", "ADDRESS", "ORGANIZATION", "LOCATION"}

#: SECRET-class categories are interchangeable for matching (per original note).
_SECRET_CLASS = {"API_KEY", "AWS_SECRET", "GCP_KEY", "TOKEN", "PRIVATE_KEY", "PASSWORD"}

#: Item regex: `CAT`=value, value runs until next ", `CAT`=" or end (commas OK in value).
_ITEM_RE = re.compile(r"`(\w+)`\s*=\s*(.*?)(?=(?:,\s*`\w+`\s*=)|$)")
#: Case header: "### [NN] title ..."
_HEADER_RE = re.compile(r"^###\s*\[(\d+)\]\s*(.*)$")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parse a GLiNER report into reconstructed cases
# ─────────────────────────────────────────────────────────────────────────────

def _parse_items(line: str) -> List[Tuple[str, str]]:
    """Extract [(CATEGORY, value), ...] from a '검출/미검출/심은' line. '—' → []."""
    after = line.split(":", 1)[1] if ":" in line else line
    if "—" in after and not _ITEM_RE.search(after):
        return []
    out: List[Tuple[str, str]] = []
    for m in _ITEM_RE.finditer(after):
        cat, val = m.group(1), m.group(2).strip().rstrip(",").strip()
        if val and val != "—":
            out.append((cat, val))
    return out


def _norm(s: str) -> str:
    """Normalize a value for containment matching: drop spaces, lowercase."""
    return re.sub(r"\s+", "", s).lower()


def _dedup(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for cat, val in items:
        key = (cat, _norm(val))
        if key not in seen:
            seen.add(key)
            out.append((cat, val))
    return out


def parse_report(path: str) -> List[Dict]:
    """
    Parse one GLiNER report into cases:
      {id, title, text, truth: [(cat, val)]}
    Text is from a ``` fence ``` (codex/gemini) or '> ' blockquote (claude).
    Ground truth = '심은' line if present, else 검출 ∪ 미검출.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    cases: List[Dict] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _HEADER_RE.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        case_id, title = m.group(1), m.group(2).strip()
        i += 1

        # Collect text (fence or blockquote) and the planted/detected lines until
        # the next "### " header.
        text_parts: List[str] = []
        planted: List[Tuple[str, str]] = []
        detected: List[Tuple[str, str]] = []
        missed: List[Tuple[str, str]] = []
        in_fence = False
        while i < n and not lines[i].startswith("### "):
            raw = lines[i].rstrip("\n")
            stripped = raw.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                i += 1
                continue
            if in_fence:
                text_parts.append(raw)
            elif raw.startswith(">"):
                text_parts.append(raw.lstrip("> ").rstrip())
            elif "심은" in raw and raw.lstrip().startswith("-"):
                planted = _parse_items(raw)
            elif "검출" in raw and "미검출" not in raw and raw.lstrip().startswith("-"):
                detected = _parse_items(raw)
            elif "미검출" in raw and raw.lstrip().startswith("-"):
                missed = _parse_items(raw)
            i += 1

        text = " ".join(p for p in text_parts if p).strip()
        truth = planted if planted else _dedup(detected + missed)
        if text and truth:
            cases.append({
                "id": case_id, "title": title, "text": text,
                "truth": [{"category": c, "value": v} for c, v in truth],
            })
    return cases


def reconstruct_all() -> Dict[str, List[Dict]]:
    """Parse all three reports → {dataset: [cases]}."""
    out: Dict[str, List[Dict]] = {}
    for ds, fname in _SOURCES.items():
        path = os.path.join(_SCRIPT_DIR, fname)
        out[ds] = parse_report(path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _cat_compatible(truth_cat: str, det_cat: str) -> bool:
    if truth_cat == det_cat:
        return True
    if truth_cat in _SECRET_CLASS and det_cat in _SECRET_CLASS:
        return True
    addr = {"ADDRESS", "LOCATION"}
    if truth_cat in addr and det_cat in addr:
        return True
    return False


def _value_match(truth_val: str, det_val: str) -> bool:
    a, b = _norm(truth_val), _norm(det_val)
    if not a or not b:
        return False
    return a in b or b in a


def score_case(truth: List[Dict], detections: List[Tuple[str, str]]) -> Dict:
    """
    Score one case. Returns dict with tp/fn/fp counts and the matched/missed/
    extra item lists. Each detection consumed by at most one truth item.
    """
    remaining = list(detections)  # (cat, val)
    matched, missed = [], []
    for t in truth:
        hit_idx = None
        for idx, (dc, dv) in enumerate(remaining):
            if _cat_compatible(t["category"], dc) and _value_match(t["value"], dv):
                hit_idx = idx
                break
        if hit_idx is not None:
            matched.append(t)
            remaining.pop(hit_idx)
        else:
            missed.append(t)
    extra = [{"category": c, "value": v} for c, v in remaining]  # FP candidates
    return {
        "tp": len(matched), "fn": len(missed), "fp": len(extra),
        "matched": matched, "missed": missed, "extra": extra,
    }


def _metrics(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return round(prec, 4), round(rec, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Replay through the engine for a backend
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(backend: str, min_confidence: float):
    """Construct Engine + warmed Stage2NERRunner for *backend*."""
    os.environ["PIIGUARD_NER_BACKEND"] = backend
    os.environ["PIIGUARD_NER_MIN_CONFIDENCE"] = str(min_confidence)
    from pii_guard import Engine
    from pii_guard.stage2 import Stage2NERRunner
    runner = Stage2NERRunner(timeout_seconds=20.0)
    engine = Engine(stage2_runner=runner, ner_backend=backend)
    runner.warmup()  # cold-load the model outside the per-block timeout
    return engine, runner


def replay_dataset(engine, cases: List[Dict]) -> Dict:
    """Replay all cases of a dataset; return aggregate + per-case results."""
    agg = {"tp": 0, "fn": 0, "fp": 0}
    per_cat: Dict[str, Dict[str, int]] = {}
    case_results = []
    for case in cases:
        result = engine.scan(case["text"])
        dets = [(d.category, d.original) for d in result.detections]
        sc = score_case(case["truth"], dets)
        agg["tp"] += sc["tp"]; agg["fn"] += sc["fn"]; agg["fp"] += sc["fp"]
        # Per-category recall bookkeeping (truth side).
        for t in case["truth"]:
            c = t["category"]
            per_cat.setdefault(c, {"tp": 0, "fn": 0})
            hit = t in sc["matched"]
            per_cat[c]["tp" if hit else "fn"] += 1
        case_results.append({
            "id": case["id"], "title": case["title"],
            "text": case["text"], "truth": case["truth"],
            **{k: sc[k] for k in ("tp", "fn", "fp", "missed", "extra")},
        })
    p, r = _metrics(agg["tp"], agg["fp"], agg["fn"])
    return {"aggregate": {**agg, "precision": p, "recall": r},
            "per_category": per_cat, "cases": case_results}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_report(dataset: str, backend: str, model: str, res: Dict) -> str:
    a = res["aggregate"]
    L: List[str] = []
    L.append(f"# 외부 LLM({dataset}) 재생성 테스트 — PII-Guard 검출 · {backend} 엔진 (2026-06-29)")
    L.append("")
    L.append(f"> ⚠️ **재구성 리플레이**: 원본 입력 JSON이 없어, `EXTERNAL_LLM_TEST_2026-06-23_{dataset}_GLiNER.md`에")
    L.append(f"> 임베드된 텍스트·정답(ground truth)을 파싱해 동일 케이스를 **{backend}** 백엔드로 재채점한 결과다.")
    L.append(f"> 생성 하니스: `validation/external_replay.py` · 엔진: `Engine` + `Stage2NERRunner`({model}).")
    L.append(f"> 정답·매칭은 GLiNER 리포트 기준으로 복원했으므로, 공정 비교를 위해 GLiNER도 동일 하니스로 재채점한다.")
    L.append("> 채점: 카테고리 호환(SECRET 클래스 내 호환·LOCATION≈ADDRESS) + 값 포함 매칭 · **raw FP(트리아지 없음)**.")
    L.append("")
    L.append("## 1. 핵심 결과")
    L.append("")
    L.append(f"- 케이스 {len(res['cases'])}개 · 정답 PII 총 {a['tp'] + a['fn']}개")
    L.append(f"- **검출(TP) {a['tp']} · 미검출(FN) {a['fn']} · 오탐후보(FP) {a['fp']}**")
    L.append(f"- **재현율(recall) = {a['recall']:.3f}** · **정밀도(precision) = {a['precision']:.3f}**")
    L.append("")
    L.append("## 2. NER 카테고리별 재현율 (PERSON/ADDRESS/ORGANIZATION)")
    L.append("")
    L.append("| 카테고리 | TP | FN | recall |")
    L.append("| :-- | --: | --: | --: |")
    for c in ["PERSON", "ADDRESS", "ORGANIZATION", "LOCATION"]:
        if c in res["per_category"]:
            pc = res["per_category"][c]
            tot = pc["tp"] + pc["fn"]
            rec = pc["tp"] / tot if tot else 1.0
            L.append(f"| {c} | {pc['tp']} | {pc['fn']} | {rec:.3f} |")
    L.append("")
    L.append("## 3. 부록 — 케이스별 미검출/오탐후보")
    L.append("")
    for c in res["cases"]:
        flag = []
        if c["missed"]:
            flag.append("❌ 미검출: " + ", ".join(f"`{m['category']}`={m['value']}" for m in c["missed"]))
        if c["extra"]:
            flag.append("⚠️ 오탐후보: " + ", ".join(f"`{e['category']}`={e['value']}" for e in c["extra"]))
        status = " · ".join(flag) if flag else "✅ 미검출/오탐 없음"
        L.append(f"### [{c['id']}] {c['title']}")
        L.append(f"- TP {c['tp']} · FN {c['fn']} · FP {c['fp']}")
        L.append(f"- {status}")
        L.append("")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ner-backend", choices=("gliner", "nunerzero", "spacy"),
                   default="nunerzero")
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--model", default=None, help="Model label for the report header.")
    p.add_argument("--dump-cases-only", action="store_true", default=False,
                   help="Only reconstruct + write external_cases_reconstructed.json.")
    p.add_argument("--quiet", action="store_true", default=False)
    return p.parse_args(argv)


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"[external-replay] {msg}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    datasets = reconstruct_all()
    total = sum(len(v) for v in datasets.values())
    _log(f"reconstructed {total} cases: "
         + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()), args.quiet)

    dump_path = os.path.join(_SCRIPT_DIR, "external_cases_reconstructed.json")
    with open(dump_path, "w", encoding="utf-8") as fh:
        json.dump(datasets, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    _log(f"wrote {dump_path}", args.quiet)
    if args.dump_cases_only:
        return

    backend = args.ner_backend
    model = args.model or {
        "gliner": "urchade/gliner_multi_pii-v1",
        "nunerzero": "numind/NuNER_Zero",
        "spacy": "ko_core_news_lg",
    }[backend]
    # File label. NuNER Zero gets the clean house-style name (new file). The
    # gliner/spacy *baselines* get a "_replay" suffix so this harness can NEVER
    # overwrite the authoritative original `..._{GLiNER,spaCy}.md` source reports
    # (those carry the ground truth and must stay intact).
    label = {
        "gliner": "GLiNER_replay",
        "nunerzero": "NuNERZero",
        "spacy": "spaCy_replay",
    }[backend]
    # Hard guard against clobbering any pre-existing source report.
    _SOURCE_PATHS = {os.path.join(_SCRIPT_DIR, f) for f in _SOURCES.values()}

    _log(f"loading engine (backend={backend}, model={model}) ...", args.quiet)
    engine, runner = build_engine(backend, args.min_confidence)
    try:
        summary = {}
        for ds, cases in datasets.items():
            _log(f"replaying {ds} ({len(cases)} cases) ...", args.quiet)
            res = replay_dataset(engine, cases)
            summary[ds] = res["aggregate"]
            md = render_report(ds, backend, model, res)
            out_md = os.path.join(
                _SCRIPT_DIR, f"EXTERNAL_LLM_TEST_2026-06-23_{ds}_{label}.md")
            if out_md in _SOURCE_PATHS:  # belt-and-suspenders: never clobber a source
                raise RuntimeError(f"refusing to overwrite source report: {out_md}")
            with open(out_md, "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
            out_json = os.path.join(
                _SCRIPT_DIR, f"external_replay_{ds}_{backend}.json")
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            _log(f"  {ds}: R={res['aggregate']['recall']:.3f} "
                 f"P={res['aggregate']['precision']:.3f} → {out_md}", args.quiet)
    finally:
        runner.close()

    print(json.dumps({"backend": backend, "model": model, "datasets": summary},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
