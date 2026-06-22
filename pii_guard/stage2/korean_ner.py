"""
pii_guard/stage2/korean_ner.py

Korean NER engine using Microsoft Presidio orchestrating a spaCy Korean model
(ko_core_news_sm) to detect unstructured Korean PII — person names, addresses
(locations), and organizations — that Stage-1 regex cannot catch.

Architecture
------------
``KoreanNEREngine`` wraps Presidio's ``AnalyzerEngine`` configured with the
``ko_core_news_sm`` spaCy model.  The spaCy Korean model uses its own entity
label scheme (PS=Person, LC=Location, OG=Organization); we register a
``NerModelConfiguration`` that maps these to Presidio's canonical entity types
(PERSON, LOCATION, ORGANIZATION).  A custom ``SpacyRecognizer`` restricted to
``supported_language='ko'`` is the only recognizer in the registry, so no
English-default recognizers interfere.

Detected entities are converted to PII-Guard ``Detection`` objects:

  Presidio entity    →  PII-Guard category  class           action
  -----------------     ------------------  ----------      -----------------
  PERSON             →  PERSON              KOREAN_PII      TOKENIZE_ROUNDTRIP
  LOCATION           →  ADDRESS             KOREAN_PII      TOKENIZE_ROUNDTRIP
  ORGANIZATION       →  ORGANIZATION        KOREAN_PII      TOKENIZE_ROUNDTRIP

Confidence scores come from the spaCy NER scorer (forwarded by Presidio).
The default NER score is 0.70 when the model does not emit per-span probabilities;
entities above MIN_CONFIDENCE are included.

This module is designed to be imported inside the Stage-2 subprocess worker
(``default_ner_worker_loop``), so models are loaded lazily once per subprocess
and amortised across requests.  It must never be imported in the main process
paths to avoid polluting the parent process memory.

Usage::

    engine = KoreanNEREngine()           # lazy — no model loaded yet
    detections = engine.detect("김철수 씨께서 서울특별시에 방문했습니다.")
    for det in detections:
        print(det.category, det.confidence, det.original)

Notes on model limitations
---------------------------
- ``ko_core_news_sm`` is a small model (~14 MB).  Precision and recall are
  moderate; it works best on well-formed Korean text.
- Particles can be attached to entity spans (e.g. "홍길동은" instead of
  "홍길동").  We post-process to strip common Korean postpositional particles
  from entity text so placeholders match cleanly.
- When Presidio does not receive a per-token score from spaCy, it uses the
  ``default_score`` from NerModelConfiguration (0.70).  This is the minimum
  non-zero confidence value returned by this engine.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..models import (
    Action,
    CategoryClass,
    Detection,
    DetectionStage,
    MaskStyle,
)

logger = logging.getLogger(__name__)

# Minimum confidence threshold — detections below this are discarded.
MIN_CONFIDENCE: float = 0.50

# Mapping from Presidio entity type to PII-Guard category name.
_PRESIDIO_TO_CATEGORY: dict = {
    "PERSON":       "PERSON",
    "LOCATION":     "ADDRESS",
    "ORGANIZATION": "ORGANIZATION",
}

# Mapping from Presidio entity type to PII-Guard CategoryClass.
_ENTITY_CLASS: dict = {
    "PERSON":       CategoryClass.KOREAN_PII,
    "LOCATION":     CategoryClass.KOREAN_PII,
    "ORGANIZATION": CategoryClass.KOREAN_PII,
}

# All entities that get TOKENIZE_ROUNDTRIP (mask with placeholder, rehydrate
# on inbound responses so agent sees real values).
_ENTITY_ACTION: dict = {
    "PERSON":       Action.TOKENIZE_ROUNDTRIP,
    "LOCATION":     Action.TOKENIZE_ROUNDTRIP,
    "ORGANIZATION": Action.TOKENIZE_ROUNDTRIP,
}

# Korean postpositional particles (조사) commonly attached to entity spans
# by the spaCy tokeniser.  We strip these suffixes from detected text so
# placeholders are clean.
_KO_PARTICLES = (
    "이", "가", "을", "를", "은", "는", "의", "에", "에서", "으로", "로",
    "와", "과", "이나", "나", "도", "만", "까지", "부터", "이라", "라",
    "에게", "한테", "께", "에서는", "에게서", "씨", "님",
)


def _strip_ko_particle(text: str) -> str:
    """
    Strip a trailing Korean postpositional particle from *text*.

    Tries each particle from longest to shortest to avoid partial matches.
    Returns the stripped string, or the original if no particle is found.
    Only strips if the remaining text is at least 2 characters (to avoid
    stripping meaningful trailing characters from short names).
    """
    for particle in sorted(_KO_PARTICLES, key=len, reverse=True):
        if text.endswith(particle) and len(text) - len(particle) >= 2:
            return text[: len(text) - len(particle)]
    return text


class KoreanNEREngine:
    """
    Presidio-based Korean NER engine wrapping the ``ko_core_news_sm`` spaCy model.

    The engine is initialised lazily — the spaCy model and Presidio pipeline
    are loaded on the first call to :meth:`detect` and reused for subsequent
    calls within the same process (subprocess-lifetime singleton).

    Parameters
    ----------
    min_confidence:
        Detections with a score below this threshold are discarded.
        Defaults to :data:`MIN_CONFIDENCE` (0.50).
    strip_particles:
        When ``True`` (default), trailing Korean postpositional particles are
        stripped from entity spans so that "홍길동은" becomes "홍길동".
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        strip_particles: bool = True,
    ) -> None:
        self._min_confidence = min_confidence
        self._strip_particles = strip_particles
        self._analyzer: Optional[object] = None  # lazy — set on first use

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> List[Detection]:
        """
        Detect Korean PII entities in *text*.

        Runs Presidio with the Korean spaCy NER model and converts results
        to PII-Guard :class:`~pii_guard.models.Detection` objects.

        Parameters
        ----------
        text:
            Unstructured Korean (or mixed) text to scan.

        Returns
        -------
        List[Detection]
            Detected entities sorted by start position.  May be empty.
        """
        if not text or not text.strip():
            return []

        analyzer = self._get_analyzer()
        try:
            results = analyzer.analyze(text=text, language="ko")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Presidio analysis failed: %s", exc)
            return []

        detections: List[Detection] = []
        for result in results:
            if result.score < self._min_confidence:
                continue
            presidio_entity = result.entity_type
            category = _PRESIDIO_TO_CATEGORY.get(presidio_entity)
            if category is None:
                continue  # unknown entity type — skip

            # Extract original span text
            original = text[result.start: result.end]

            # Optionally strip trailing Korean particles
            clean_text = original
            adjusted_end = result.end
            if self._strip_particles:
                clean_text = _strip_ko_particle(original)
                if len(clean_text) < len(original):
                    adjusted_end = result.start + len(clean_text)

            det = Detection(
                category=category,
                category_class=_ENTITY_CLASS[presidio_entity],
                action=_ENTITY_ACTION[presidio_entity],
                mask_style=MaskStyle.TOKENIZE,
                start=result.start,
                end=adjusted_end,
                original=clean_text,
                detection_stage=DetectionStage.STAGE2_NER,
                rule_id=f"ner_ko_{presidio_entity.lower()}",
                confidence=result.score,
            )
            detections.append(det)

        # Sort by start position
        detections.sort(key=lambda d: d.start)
        return detections

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_analyzer(self):
        """
        Return the Presidio AnalyzerEngine, initialising it lazily on first call.

        Import of Presidio and spaCy is deferred until here so that importing
        this module in the main process (e.g. for type checking) does not load
        the ~400 MB spaCy model into the parent address space.
        """
        if self._analyzer is not None:
            return self._analyzer

        self._analyzer = _build_presidio_analyzer()
        return self._analyzer


def _build_presidio_analyzer():
    """
    Construct and return a Presidio ``AnalyzerEngine`` for Korean text.

    Loads the ``ko_core_news_sm`` spaCy model and configures:

    * ``NerModelConfiguration`` with Korean spaCy label → Presidio entity
      mapping (PS→PERSON, LC→LOCATION, OG→ORGANIZATION, DT→DATE_TIME).
    * A ``SpacyRecognizer`` restricted to ``supported_language='ko'`` and
      the three PII-relevant entity types.
    * An empty recognizer registry (no English-default recognizers).

    Raises
    ------
    RuntimeError
        When ``ko_core_news_sm`` is not installed or Presidio imports fail.
    """
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import SpacyRecognizer
    except ImportError as exc:
        raise RuntimeError(
            "presidio-analyzer is not installed. "
            "Run: pip install presidio-analyzer presidio-anonymizer"
        ) from exc

    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "ko", "model_name": "ko_core_news_sm"}
                ],
                "ner_model_configuration": {
                    # Map Korean spaCy NER labels → Presidio canonical entity names
                    "model_to_presidio_entity_mapping": {
                        "PS": "PERSON",       # 사람/인물 (Person)
                        "LC": "LOCATION",     # 장소/지역 (Location)
                        "OG": "ORGANIZATION", # 기관/조직 (Organization)
                        "DT": "DATE_TIME",    # 날짜 (Date) — not mapped to PII but included
                    },
                    # Default confidence when spaCy does not emit per-span scores
                    "default_score": 0.70,
                },
            }
        )
        nlp_engine = provider.create_engine()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load Korean spaCy model 'ko_core_news_sm'. "
            "Run: python -m spacy download ko_core_news_sm"
        ) from exc

    # Korean recognizer — maps spaCy NER entities to PERSON/LOCATION/ORGANIZATION
    korean_recognizer = SpacyRecognizer(
        supported_language="ko",
        supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
        ner_strength=0.70,  # base confidence when model does not score
    )

    registry = RecognizerRegistry()
    registry.add_recognizer(korean_recognizer)

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
    )
    return analyzer
