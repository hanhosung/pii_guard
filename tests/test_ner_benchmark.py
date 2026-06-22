"""
Sub-AC 4: Korean NER corpus precision/recall benchmark tests.
=============================================================

Verifies that the NER benchmark evaluation harness:

1. Runs the full NER pipeline (Stage-1 regex/checksum + Stage-2 Presidio/spaCy)
   against the labeled NERBenchmarkCorpus.
2. Produces a parseable JSON metrics report with the required structure.
3. Reports per-entity-type precision and recall for the NER-owned categories:
   PERSON, ADDRESS, and ORGANIZATION.
4. (Optional, NER-only) Metrics for NER-owned categories meet the minimum
   precision/recall thresholds.

These tests exercise the benchmark at two levels:

  * Direct import  — imports ``run_benchmark()`` in-process; fast, no subprocess.
  * Subprocess     — invokes the script via ``python benchmarks/korean_ner_benchmark.py``
                     to verify the CLI produces valid JSON on stdout.

Skip strategy
-------------
Tests that require the real NER model (Presidio + ko_core_news_sm) are gated
by a session-scoped ``ner_available`` fixture.  If the model is not installed,
those tests are skipped rather than failed, so CI passes on environments that
do not have the ~300 MB Korean spaCy model.

When NER IS installed, precision/recall threshold tests run and gate CI.

Run:
    # Quick (Stage-1 baseline only, no model needed):
    pytest tests/test_ner_benchmark.py -v -k "not ner_threshold"

    # Full (requires presidio + ko_core_news_sm):
    pytest tests/test_ner_benchmark.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Project root (benchmarks/ directory lives here)
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_BENCHMARK_SCRIPT = _PROJECT_ROOT / "benchmarks" / "korean_ner_benchmark.py"

# ─────────────────────────────────────────────────────────────────────────────
# Expected NER-owned categories in the report
# ─────────────────────────────────────────────────────────────────────────────
NER_CATEGORIES = ["PERSON", "ADDRESS", "ORGANIZATION"]

# Minimum thresholds for the full NER pipeline (conservative — small spaCy model)
MIN_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "PERSON":       {"precision": 0.50, "recall": 0.40},
    "ADDRESS":      {"precision": 0.55, "recall": 0.45},
    "ORGANIZATION": {"precision": 0.50, "recall": 0.40},
}

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ner_available() -> bool:
    """True if presidio-analyzer and ko_core_news_sm are installed."""
    try:
        from presidio_analyzer import AnalyzerEngine  # noqa: F401
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import SpacyRecognizer  # noqa: F401
        import spacy
        # Check model is downloaded (does not load full model — just checks presence)
        if not spacy.util.is_package("ko_core_news_sm"):
            return False
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture(scope="session")
def benchmark_report_no_ner():
    """
    Run benchmark in Stage-1-only mode (no NER required).

    Returns the raw dict from run_benchmark() for structural assertions.
    """
    # Add project root to path if needed
    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmarks.korean_ner_benchmark import run_benchmark
    return run_benchmark(
        corpus_seed=42,
        samples_per_format=5,
        no_ner=True,
        apply_thresholds=False,
        quiet=True,
    )


@pytest.fixture(scope="session")
def benchmark_report_with_ner(ner_available):
    """
    Run benchmark with the full NER pipeline.

    Skips the session if NER is not installed.  When NER is available the
    KoreanNEREngine is loaded once for the whole session.
    """
    if not ner_available:
        pytest.skip("NER not available (presidio or ko_core_news_sm not installed)")

    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmarks.korean_ner_benchmark import run_benchmark
    return run_benchmark(
        corpus_seed=42,
        samples_per_format=5,
        no_ner=False,
        apply_thresholds=False,
        quiet=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Benchmark script CLI — subprocess invocation
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkScript:
    """The benchmark script must be runnable and produce parseable JSON."""

    def test_benchmark_script_exists(self):
        """benchmarks/korean_ner_benchmark.py must be present."""
        assert _BENCHMARK_SCRIPT.exists(), (
            f"Benchmark script not found: {_BENCHMARK_SCRIPT}"
        )

    def test_benchmark_script_produces_parseable_json(self):
        """
        Running the script with --no-ner (no model required) must produce
        valid JSON on stdout without exiting with a non-zero code.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(_BENCHMARK_SCRIPT),
                "--no-ner",
                "--corpus-seed", "42",
                "--samples-per-format", "3",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Benchmark script exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[:2000]}"
        )
        assert result.stdout.strip(), "Benchmark script produced no stdout output"

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Benchmark script stdout is not valid JSON: {exc}\n"
                f"stdout (first 500 chars): {result.stdout[:500]}"
            )

        # Verify top-level structure
        assert "metadata" in report, "Report missing 'metadata' key"
        assert "stage1_metrics" in report, "Report missing 'stage1_metrics' key"
        assert "full_pipeline_metrics" in report, "Report missing 'full_pipeline_metrics' key"

    def test_benchmark_script_no_ner_outputs_stage1_only_pipeline(self):
        """With --no-ner the pipeline should be reported as 'stage1_only'."""
        result = subprocess.run(
            [
                sys.executable,
                str(_BENCHMARK_SCRIPT),
                "--no-ner",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        report = json.loads(result.stdout)
        assert report["metadata"]["pipeline"] == "stage1_only"
        assert report["metadata"]["ner_available"] is False

    def test_benchmark_script_output_to_file(self, tmp_path):
        """
        --output PATH must write the JSON report to the given file and
        produce no JSON on stdout.
        """
        out_file = tmp_path / "ner_report.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_BENCHMARK_SCRIPT),
                "--no-ner",
                "--output", str(out_file),
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Non-zero exit: {result.returncode}\nstderr: {result.stderr[:500]}"
        )
        assert out_file.exists(), f"Output file not created at {out_file}"
        content = out_file.read_text(encoding="utf-8")
        report = json.loads(content)
        assert "entity_metrics" in report or "stage1_metrics" in report


# ─────────────────────────────────────────────────────────────────────────────
# 2. Report structure — validate required fields
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkReportStructure:
    """The metrics report dict must have the required top-level structure."""

    def test_metadata_present(self, benchmark_report_no_ner):
        r = benchmark_report_no_ner
        assert "metadata" in r, "Report missing 'metadata'"

    def test_metadata_fields(self, benchmark_report_no_ner):
        meta = benchmark_report_no_ner["metadata"]
        required = [
            "corpus_seed", "samples_per_format", "ner_available", "pipeline",
            "min_confidence",
        ]
        for field in required:
            assert field in meta, f"metadata missing field '{field}'"

    def test_stage1_metrics_present(self, benchmark_report_no_ner):
        assert "stage1_metrics" in benchmark_report_no_ner, (
            "Report missing 'stage1_metrics'"
        )

    def test_full_pipeline_metrics_present(self, benchmark_report_no_ner):
        assert "full_pipeline_metrics" in benchmark_report_no_ner, (
            "Report missing 'full_pipeline_metrics'"
        )

    def test_ner_contribution_present(self, benchmark_report_no_ner):
        assert "ner_contribution" in benchmark_report_no_ner, (
            "Report missing 'ner_contribution'"
        )

    def test_thresholds_met_field_present(self, benchmark_report_no_ner):
        assert "thresholds_met" in benchmark_report_no_ner, (
            "Report missing 'thresholds_met'"
        )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_stage1_metrics_has_ner_categories(
        self, benchmark_report_no_ner, category
    ):
        """stage1_metrics must have an entry for each NER-owned category."""
        stage1 = benchmark_report_no_ner["stage1_metrics"]
        assert category in stage1, (
            f"stage1_metrics missing category '{category}'. "
            f"Found: {list(stage1.keys())}"
        )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_full_pipeline_metrics_has_ner_categories(
        self, benchmark_report_no_ner, category
    ):
        """full_pipeline_metrics must have an entry for each NER-owned category."""
        full = benchmark_report_no_ner["full_pipeline_metrics"]
        assert category in full, (
            f"full_pipeline_metrics missing category '{category}'. "
            f"Found: {list(full.keys())}"
        )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_category_metric_has_required_fields(
        self, benchmark_report_no_ner, category
    ):
        """Each category entry must contain precision, recall, tp, fp, fn."""
        full = benchmark_report_no_ner["full_pipeline_metrics"]
        entry = full[category]
        for field in ("precision", "recall", "tp", "fp", "fn"):
            assert field in entry, (
                f"full_pipeline_metrics[{category!r}] missing field '{field}': "
                f"found {list(entry.keys())}"
            )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_precision_in_unit_interval(self, benchmark_report_no_ner, category):
        """Precision must be in [0.0, 1.0]."""
        entry = benchmark_report_no_ner["full_pipeline_metrics"][category]
        assert 0.0 <= entry["precision"] <= 1.0, (
            f"{category} precision {entry['precision']} outside [0, 1]"
        )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_recall_in_unit_interval(self, benchmark_report_no_ner, category):
        """Recall must be in [0.0, 1.0]."""
        entry = benchmark_report_no_ner["full_pipeline_metrics"][category]
        assert 0.0 <= entry["recall"] <= 1.0, (
            f"{category} recall {entry['recall']} outside [0, 1]"
        )

    def test_tp_fp_fn_are_non_negative_integers(self, benchmark_report_no_ner):
        """tp, fp, fn counts must be non-negative integers."""
        for stage_key in ("stage1_metrics", "full_pipeline_metrics"):
            for cat, entry in benchmark_report_no_ner[stage_key].items():
                for count_field in ("tp", "fp", "fn"):
                    val = entry[count_field]
                    assert isinstance(val, int) and val >= 0, (
                        f"{stage_key}[{cat}][{count_field}] = {val!r} "
                        f"is not a non-negative int"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stage-1 baseline sanity checks (no NER required)
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1Baseline:
    """Stage-1 baseline metrics must satisfy known characteristics."""

    def test_organization_stage1_recall_is_zero(self, benchmark_report_no_ner):
        """
        Stage-1 regex/checksum has no ORGANIZATION detector.
        Recall for ORGANIZATION under Stage-1 must be 0.0 (complete gap).
        """
        entry = benchmark_report_no_ner["stage1_metrics"]["ORGANIZATION"]
        assert entry["recall"] == 0.0, (
            f"ORGANIZATION Stage-1 recall should be 0.0 (no regex detector), "
            f"got {entry['recall']}.  This is a corpus or pipeline configuration error."
        )

    def test_organization_stage1_tp_is_zero(self, benchmark_report_no_ner):
        """Stage-1 must produce zero true-positive ORGANIZATION detections."""
        entry = benchmark_report_no_ner["stage1_metrics"]["ORGANIZATION"]
        assert entry["tp"] == 0, (
            f"Stage-1 TP for ORGANIZATION should be 0, got {entry['tp']}"
        )

    def test_organization_stage1_precision_is_one_by_convention(
        self, benchmark_report_no_ner
    ):
        """
        When TP=FP=0 (no detections at all), precision is 1.0 by the 0/0=1
        convention used in the corpus utility.  This is the expected Stage-1
        ORGANIZATION precision.
        """
        entry = benchmark_report_no_ner["stage1_metrics"]["ORGANIZATION"]
        assert entry["precision"] == 1.0, (
            f"Stage-1 ORGANIZATION precision should be 1.0 (no detections), "
            f"got {entry['precision']}"
        )

    def test_corpus_has_organization_positive_samples(self, benchmark_report_no_ner):
        """
        The corpus must have ORGANIZATION positive samples for recall to be
        measurable.  fn > 0 confirms there are org positives the Stage-1 missed.
        """
        entry = benchmark_report_no_ner["stage1_metrics"]["ORGANIZATION"]
        assert entry["fn"] > 0, (
            f"ORGANIZATION fn={entry['fn']}: no ORGANIZATION positive samples found.  "
            f"NERBenchmarkCorpus._build_organization_samples() may not have run."
        )

    def test_person_stage1_recall_non_zero(self, benchmark_report_no_ner):
        """
        Stage-1 detects PERSON in labelled Korean formats (성명: 이름: 담당자: …).
        Recall must be > 0.
        """
        entry = benchmark_report_no_ner["stage1_metrics"]["PERSON"]
        assert entry["recall"] > 0.0, (
            f"Stage-1 PERSON recall is 0.0 — the labeled Korean person-name "
            f"formats should be detected by Stage-1 regex."
        )

    def test_address_stage1_recall_non_zero(self, benchmark_report_no_ner):
        """
        Stage-1 detects Korean addresses via province-prefix regex.
        Recall must be > 0.
        """
        entry = benchmark_report_no_ner["stage1_metrics"]["ADDRESS"]
        assert entry["recall"] > 0.0, (
            f"Stage-1 ADDRESS recall is 0.0 — Korean address patterns with "
            f"province prefixes should be caught by Stage-1 regex."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. NER pipeline metrics — require NER model
# ─────────────────────────────────────────────────────────────────────────────

class TestNERPipelineMetrics:
    """
    Full-pipeline (Stage-1 + Stage-2 NER) metrics.

    All tests in this class are skipped when NER is not installed.
    """

    def test_ner_available_flag(self, benchmark_report_with_ner):
        """When NER runs successfully, ner_available must be True."""
        assert benchmark_report_with_ner["metadata"]["ner_available"] is True

    def test_pipeline_is_stage1_plus_ner(self, benchmark_report_with_ner):
        assert benchmark_report_with_ner["metadata"]["pipeline"] == "stage1+stage2_ner"

    def test_organization_recall_improved_by_ner(self, benchmark_report_with_ner):
        """
        Stage-2 NER must improve ORGANIZATION recall above 0.0.
        (Stage-1 has zero ORGANIZATION recall; any NER detection is an improvement.)
        """
        full_recall = benchmark_report_with_ner["full_pipeline_metrics"]["ORGANIZATION"]["recall"]
        assert full_recall > 0.0, (
            f"ORGANIZATION recall is still 0.0 after Stage-2 NER.  "
            f"The NER model should detect at least some Korean organisation names "
            f"from the corpus.  "
            f"Check that ko_core_news_sm is loaded and the ORGANIZATION corpus "
            f"samples use organisations the model recognises."
        )

    def test_ner_contribution_org_recall_gain_positive(
        self, benchmark_report_with_ner
    ):
        """
        ner_contribution[ORGANIZATION][recall_gain] must be > 0.

        This is the key proof that Stage-2 NER adds detection coverage
        for organisation names that Stage-1 regex cannot catch.
        """
        gain = benchmark_report_with_ner["ner_contribution"]["ORGANIZATION"]["recall_gain"]
        assert gain > 0.0, (
            f"NER recall gain for ORGANIZATION is {gain:.4f} — expected > 0.  "
            f"Stage-2 NER is not improving ORGANIZATION detection.  "
            f"Verify that the NER engine and corpus are configured correctly."
        )

    @pytest.mark.parametrize("category", NER_CATEGORIES)
    def test_full_pipeline_precision_not_zero(
        self, benchmark_report_with_ner, category
    ):
        """Precision must be > 0.0 for all NER-owned categories."""
        entry = benchmark_report_with_ner["full_pipeline_metrics"][category]
        assert entry["precision"] > 0.0, (
            f"Full pipeline precision for {category!r} is 0.0 — "
            f"all detections are false positives (no true positives).  "
            f"Check corpus labelling and model configuration."
        )

    @pytest.mark.parametrize("category,floors", MIN_THRESHOLDS.items())
    def test_ner_threshold_precision(
        self, benchmark_report_with_ner, category, floors
    ):
        """Full-pipeline precision must meet the minimum threshold floor."""
        entry = benchmark_report_with_ner["full_pipeline_metrics"][category]
        floor = floors["precision"]
        actual = entry["precision"]
        assert actual >= floor, (
            f"{category} precision {actual:.4f} < minimum floor {floor:.4f}.  "
            f"tp={entry['tp']}, fp={entry['fp']}, fn={entry['fn']}.  "
            f"Improve precision by tightening the min_confidence threshold or "
            f"expanding the negative sample set."
        )

    @pytest.mark.parametrize("category,floors", MIN_THRESHOLDS.items())
    def test_ner_threshold_recall(
        self, benchmark_report_with_ner, category, floors
    ):
        """Full-pipeline recall must meet the minimum threshold floor."""
        entry = benchmark_report_with_ner["full_pipeline_metrics"][category]
        floor = floors["recall"]
        actual = entry["recall"]
        assert actual >= floor, (
            f"{category} recall {actual:.4f} < minimum floor {floor:.4f}.  "
            f"tp={entry['tp']}, fp={entry['fp']}, fn={entry['fn']}.  "
            f"The NER model may need better contexts for this category; "
            f"consider using ko_core_news_lg for higher recall."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. NERBenchmarkCorpus structure tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNERBenchmarkCorpus:
    """Verify the NERBenchmarkCorpus produces correct ORGANIZATION samples."""

    @pytest.fixture(scope="class")
    def ner_corpus(self):
        from pii_guard.corpus.ner_benchmark_corpus import NERBenchmarkCorpus
        return NERBenchmarkCorpus(seed=42, samples_per_format=5)

    def test_corpus_has_organization_category(self, ner_corpus):
        counts = ner_corpus.category_counts()
        assert "ORGANIZATION" in counts, (
            f"NERBenchmarkCorpus has no ORGANIZATION samples.  "
            f"Found categories: {list(counts.keys())}"
        )

    def test_organization_sample_count(self, ner_corpus):
        """5 formats × 5 samples = 25 ORGANIZATION positive samples."""
        org_count = len(ner_corpus.samples_for_category("ORGANIZATION"))
        assert org_count >= 25, (
            f"Expected ≥ 25 ORGANIZATION samples (5 formats × 5), got {org_count}"
        )

    def test_organization_span_integrity(self, ner_corpus):
        """Every ORGANIZATION span value must match text[start:end]."""
        for sample in ner_corpus.samples_for_category("ORGANIZATION"):
            for span in sample.spans:
                if span.category == "ORGANIZATION":
                    extracted = sample.text[span.start:span.end]
                    assert extracted == span.value, (
                        f"ORGANIZATION span mismatch: "
                        f"text[{span.start}:{span.end}]={extracted!r} "
                        f"!= span.value={span.value!r} in {sample.text!r}"
                    )

    def test_ner_clean_negatives_exist(self, ner_corpus):
        """ner_clean_negatives() must return at least 10 samples."""
        neg = ner_corpus.ner_clean_negatives()
        assert len(neg) >= 10, (
            f"Expected ≥ 10 NER-clean negatives, got {len(neg)}"
        )

    def test_ner_clean_negatives_have_no_spans(self, ner_corpus):
        """All NER-clean negatives must be flagged is_negative=True with no spans."""
        for s in ner_corpus.ner_clean_negatives():
            assert s.is_negative is True, f"NER-clean negative not flagged: {s.text!r}"
            assert s.spans == [], f"NER-clean negative has spans: {s.text!r}"

    def test_organization_source_tags(self, ner_corpus):
        """All five organisation format tags must appear in the corpus."""
        expected_tags = {
            "org_affiliation_label",
            "org_employee_sentence",
            "org_contact_from",
            "org_company_label",
            "org_representative",
        }
        found_tags = {s.source_tag for s in ner_corpus.samples_for_category("ORGANIZATION")}
        missing = expected_tags - found_tags
        assert not missing, (
            f"ORGANIZATION format tags missing from corpus: {sorted(missing)}"
        )

    def test_organization_spans_are_hangul_or_mixed(self, ner_corpus):
        """Organisation span values must start with a Hangul or ASCII character."""
        import re
        hangul_or_ascii = re.compile(r"^[가-힣A-Za-z]")
        for sample in ner_corpus.samples_for_category("ORGANIZATION"):
            for span in sample.spans:
                if span.category == "ORGANIZATION":
                    assert hangul_or_ascii.match(span.value), (
                        f"ORGANIZATION name has unexpected first character: "
                        f"{span.value!r}"
                    )

    def test_deterministic_across_seeds(self):
        """Two corpora with the same seed produce identical org sample texts."""
        from pii_guard.corpus.ner_benchmark_corpus import NERBenchmarkCorpus
        c1 = NERBenchmarkCorpus(seed=7, samples_per_format=3)
        c2 = NERBenchmarkCorpus(seed=7, samples_per_format=3)
        org1 = [s.text for s in c1.samples_for_category("ORGANIZATION")]
        org2 = [s.text for s in c2.samples_for_category("ORGANIZATION")]
        assert org1 == org2, "Same seed produced different ORGANIZATION sample texts"

    def test_ner_owned_categories_returned(self, ner_corpus):
        """ner_owned_categories() must return the three NER-responsible categories."""
        cats = ner_corpus.ner_owned_categories()
        assert set(cats) == {"PERSON", "ADDRESS", "ORGANIZATION"}, (
            f"Unexpected ner_owned_categories: {cats}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON serialisability (regression guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSerialisation:
    """The benchmark report must be JSON-serialisable end-to-end."""

    def test_report_is_json_serialisable(self, benchmark_report_no_ner):
        """round-trip: dict → JSON string → dict must be lossless."""
        json_str = json.dumps(benchmark_report_no_ner, ensure_ascii=False)
        reparsed = json.loads(json_str)
        assert reparsed["metadata"] == benchmark_report_no_ner["metadata"], (
            "Metadata changed after JSON round-trip"
        )

    def test_report_contains_corpus_sizes(self, benchmark_report_no_ner):
        """metadata.corpus_sizes must report non-zero counts for all keys."""
        sizes = benchmark_report_no_ner["metadata"].get("corpus_sizes", {})
        assert sizes, "metadata.corpus_sizes is missing or empty"
        for key in ("positive", "negative", "ner_clean_negatives", "org_samples"):
            assert key in sizes, f"corpus_sizes missing '{key}'"
        assert sizes["org_samples"] > 0, (
            "corpus_sizes.org_samples is 0 — ORGANIZATION samples not added to corpus"
        )
