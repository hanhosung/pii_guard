"""
Korean PII red-team corpus — synthetic data generator for detection validation.

Public API::

    from pii_guard.corpus import KoreanPIICorpus
    corpus = KoreanPIICorpus()
    samples = corpus.all_samples()   # List[CorpusSample]

    # NER benchmark corpus (adds ORGANIZATION samples):
    from pii_guard.corpus import NERBenchmarkCorpus
    ner_corpus = NERBenchmarkCorpus()
"""
from .korean_pii import KoreanPIICorpus, CorpusSample, PIISpan
from .ner_benchmark_corpus import NERBenchmarkCorpus

__all__ = ["KoreanPIICorpus", "CorpusSample", "PIISpan", "NERBenchmarkCorpus"]
