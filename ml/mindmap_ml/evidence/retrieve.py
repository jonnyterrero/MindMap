"""High-level retrieval: passages-with-citations + evidence for a factor/outcome.

This is the seam the narrative layer (Phase 6) calls. It guarantees every result
carries a citation, so the LLM can only reference evidence that actually exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .curate import EvidencePrior, find_prior, load_priors
from .index import EvidenceIndex, RetrievedPassage
from .ingest import SEED_CORPUS_DIR, ingest_corpus


@lru_cache(maxsize=4)
def _index_for(corpus_dir: Path) -> EvidenceIndex:
    return EvidenceIndex(ingest_corpus(corpus_dir))


def retrieve(query: str, k: int = 3, corpus_dir: Path = SEED_CORPUS_DIR) -> list[RetrievedPassage]:
    """Top-k passages for a free-text query, each with a citation."""
    return _index_for(corpus_dir).query(query, k=k, min_score=0.0)


@dataclass
class Evidence:
    """Everything grounding a single (factor, outcome) recommendation."""

    factor: str
    outcome: str
    prior: EvidencePrior | None
    passages: list[RetrievedPassage]

    @property
    def is_grounded(self) -> bool:
        return self.prior is not None or len(self.passages) > 0

    @property
    def citations(self) -> list[str]:
        cites: list[str] = []
        if self.prior is not None:
            cites.append(self.prior.citation)
        for r in self.passages:
            if r.passage.citation not in cites:
                cites.append(r.passage.citation)
        return cites


@lru_cache(maxsize=64)
def evidence_for(factor: str, outcome: str, k: int = 2) -> Evidence:
    """Curated prior (if any) + supporting retrieved passages for a factor/outcome.

    Used to gate recommendations: only surface advice when ``is_grounded`` is True.
    Cached: there are only a handful of (factor, outcome) pairs in practice.
    """
    prior = find_prior(factor, outcome, load_priors())
    query = f"{factor.replace('_', ' ')} {outcome}"
    passages = retrieve(query, k=k)
    return Evidence(factor=factor, outcome=outcome, prior=prior, passages=passages)
