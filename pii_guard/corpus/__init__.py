"""
Korean PII red-team corpus — synthetic data generator for detection validation.

Public API::

    from pii_guard.corpus import KoreanPIICorpus
    corpus = KoreanPIICorpus()
    samples = corpus.all_samples()   # List[CorpusSample]
"""
from .korean_pii import KoreanPIICorpus, CorpusSample, PIISpan

__all__ = ["KoreanPIICorpus", "CorpusSample", "PIISpan"]
