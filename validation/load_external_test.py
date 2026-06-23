# -*- coding: utf-8 -*-
"""
외부 LLM(Codex 등)이 생성한 테스트 JSON을 PII-Guard에 입력해 채점한다.

입력 JSON 형식 (EXTERNAL_TEST_PROMPT.md 참고):
  [
    {"id":1, "kind":"VOC", "title":"...", "text":"...",
     "expected":[["PERSON","김민준"], ["PHONE","010-..."]],
     "traps":["ORD-2024-0613"]}
  ]

실행:
  PYTHONPATH=. .venv/bin/python validation/load_external_test.py validation/external_cases.json
산출물: EXTERNAL_REPORT.md + external_log.txt
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from pii_guard.engine import Engine
from pii_guard.stage2.runner import Stage2NERRunner

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "external_log.txt")
MD_PATH = os.path.join(HERE, "EXTERNAL_REPORT.md")

SECRET_CATS = {"API_KEY", "AWS_SECRET", "GCP_KEY", "TOKEN", "PRIVATE_KEY", "PASSWORD"}


def norm(s):
    return str(s).replace(" ", "").replace("-", "").replace("\n", "").lower()


def span_match(a, b):
    na, nb = norm(a), norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def cat_ok(d, e):
    return d == e or (d in SECRET_CATS and e in SECRET_CATS)


def load_cases(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("입력 JSON은 배열이어야 합니다.")
    cases = []
    for i, c in enumerate(data):
        if "text" not in c or "expected" not in c:
            raise SystemExit(f"항목 {i}: 'text'와 'expected'가 필요합니다.")
        cases.append({
            "id": c.get("id", i + 1),
            "kind": c.get("kind", "ITEM"),
            "title": c.get("title", f"item {i+1}"),
            "text": c["text"],
            "expected": [tuple(x) for x in c["expected"]],
            "traps": c.get("traps", []),
        })
    return cases


def run(path):
    cases = load_cases(path)
    eng = Engine(stage2_runner=Stage2NERRunner())
    log, rows = [], []
    per_cat = defaultdict(lambda: {"tp": 0, "fn": 0})
    tot_tp = tot_fn = tot_fp = 0

    def L(s=""):
        log.append(s)

    L("=" * 90)
    L(f"외부 생성 테스트 채점 · 입력: {os.path.basename(path)} · {len(cases)}항목")
    L("엔진: Stage1 + Stage2 NER(lg) + proximity · 20 카테고리")
    L("=" * 90)

    for c in cases:
        res = eng.scan(c["text"])
        dets = [(d.category, d.original) for d in res.detections]
        used, tp, fn = set(), [], []
        for ecat, eval_ in c["expected"]:
            hit = None
            for j, (dc, dv) in enumerate(dets):
                if j in used:
                    continue
                if cat_ok(dc, ecat) and span_match(dv, eval_):
                    hit = j
                    break
            if hit is not None:
                used.add(hit); tp.append((ecat, eval_)); per_cat[ecat]["tp"] += 1
            else:
                fn.append((ecat, eval_)); per_cat[ecat]["fn"] += 1
        fps = [dets[j] for j in range(len(dets)) if j not in used]
        tot_tp += len(tp); tot_fn += len(fn); tot_fp += len(fps)

        L("\n" + "─" * 90)
        L(f"[{c['id']}] {c['title']}  ({len(c['text'])}자)  block={res.has_blocks}")
        L(f"  ✅검출 {len(tp)} / ❌미검출 {len(fn)} / ⚠️오탐후보 {len(fps)}")
        if fn:
            L("  미검출: " + ", ".join(f"{a}={b}" for a, b in fn))
        if fps:
            L("  오탐후보: " + ", ".join(f"{a}={b}" for a, b in fps[:10]))
        rows.append({"id": c["id"], "title": c["title"], "kind": c["kind"],
                     "len": len(c["text"]), "n": len(c["expected"]),
                     "tp": len(tp), "fn": len(fn), "fp": len(fps),
                     "text": c["text"], "expected": c["expected"], "traps": c["traps"],
                     "tp_items": tp, "fn_items": fn, "fp_items": fps,
                     "block": res.has_blocks})

    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
    L("\n" + "=" * 90)
    L(f"재현율 = {tot_tp}/{tot_tp+tot_fn} = {recall:.3f}")
    L(f"정밀도 = {tot_tp}/{tot_tp+tot_fp} = {precision:.3f}")
    L("=" * 90)
    open(LOG_PATH, "w", encoding="utf-8").write("\n".join(log))

    generate_md(rows, per_cat, tot_tp, tot_fn, tot_fp, recall, precision, path)
    print(f"recall={recall:.3f} precision={precision:.3f} TP={tot_tp} FN={tot_fn} FP={tot_fp}")
    print("report →", MD_PATH)


def generate_md(rows, per_cat, tp, fn, fp, recall, precision, path):
    o = []
    W = o.append
    W("# 외부 생성 테스트 — PII-Guard 검출 리포트")
    W("")
    W(f"> 입력: `{os.path.basename(path)}` ({len(rows)}항목) · 외부 LLM(Codex 등) 생성 텍스트를 "
      "PII-Guard에 입력해 채점. 증거: [`external_log.txt`](./external_log.txt)")
    W("> 엔진: Stage1(정규식·체크섬) + Stage2 NER(ko_core_news_lg) + proximity · 20 카테고리.")
    W("")
    W("## 1. 핵심 결과")
    W("")
    W("| 지표 | 수치 |")
    W("| :-- | :-- |")
    W(f"| **재현율(Recall)** | **{recall:.3f}**  ({tp}/{tp+fn}) |")
    W(f"| **정밀도(Precision)** | **{precision:.3f}**  ({tp}/{tp+fp}) |")
    W(f"| 검출 TP / 미검출 FN / 오탐후보 FP | {tp} / {fn} / {fp} |")
    W("")
    W("> ※ 오탐후보(FP)에는 ground truth에 라벨 안 된 실제 PII(은행명 등)나 NER 변동분이 섞일 수 "
      "있으니, 아래 §4 부록의 항목별 오탐 목록을 검토해 진짜 over-masking과 구분하세요.")
    W("")
    W("## 2. 카테고리별 재현율")
    W("")
    W("| 카테고리 | TP | FN | recall |")
    W("| :-- | --: | --: | --: |")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        flag = " ⚠️" if r < 0.85 else ""
        W(f"| {cat} | {c['tp']} | {c['fn']} | {r:.2f}{flag} |")
    W("")
    W("## 3. 항목별 요약")
    W("")
    W("| # | 제목 | 길이 | 심은 | 검출 | 미검출 | 오탐 | block |")
    W("| --: | :-- | --: | --: | --: | --: | --: | :--: |")
    for r in rows:
        W(f"| {r['id']} | {r['title']} | {r['len']} | {r['n']} | {r['tp']} | {r['fn']} | "
          f"{r['fp']} | {'🔴' if r['block'] else '—'} |")
    W("")
    W("## 4. 부록 — 전체 항목(텍스트·검출/미검출)")
    W("")
    for r in rows:
        W(f"### [{r['id']}] {r['title']}  ({r['len']}자)" + (" · 🔴 block" if r['block'] else ""))
        W("")
        W("```")
        W(r["text"])
        W("```")
        W("")
        W("- **심은({0})**: {1}".format(r["n"],
          ", ".join(f"`{a}`={b}" for a, b in r["expected"]) or "—"))
        W("- ✅ **검출({0})**: {1}".format(r["tp"],
          ", ".join(f"`{a}`={b}" for a, b in r["tp_items"]) or "—"))
        W("- ❌ **미검출({0})**: {1}".format(r["fn"],
          ", ".join(f"`{a}`={b}" for a, b in r["fn_items"]) or "—"))
        W("- ⚠️ **오탐후보({0})**: {1}".format(r["fp"],
          ", ".join(f"`{a}`={b}" for a, b in r["fp_items"]) or "—"))
        W("")
    open(MD_PATH, "w", encoding="utf-8").write("\n".join(o))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용법: load_external_test.py <cases.json>")
    run(sys.argv[1])
