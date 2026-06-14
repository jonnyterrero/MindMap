"""Local, dependency-free vector index over evidence passages.

Deliberately lightweight: a pure-numpy TF-IDF embedder + cosine similarity, so
retrieval is offline, deterministic, and testable with no model download. The
:class:`Embedder` protocol is the seam where a heavier backend
(sentence-transformers, etc.) can be substituted later without touching callers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ingest import Passage

_TOKEN_RE = re.compile(r"[a-z][a-z']+")
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "with", "this", "these", "their", "they", "them", "not", "no", "but", "can", "could", "may", "might", "more", "most", "than", "then", "there", "here", "over", "under"]
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class Embedder(Protocol):
    def fit(self, docs: list[str]) -> Embedder: ...
    def transform(self, docs: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Minimal TF-IDF with L2-normalized rows (cosine-ready)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, docs: list[str]) -> TfidfEmbedder:
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(_tokenize(doc)):
                df[term] = df.get(term, 0) + 1
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = max(len(docs), 1)
        self.idf = np.zeros(len(self.vocab))
        for term, i in self.vocab.items():
            # smoothed idf
            self.idf[i] = math.log((1 + n) / (1 + df[term])) + 1.0
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        mat = np.zeros((len(docs), len(self.vocab)))
        for r, doc in enumerate(docs):
            toks = _tokenize(doc)
            if not toks:
                continue
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    mat[r, j] += 1.0
            mat[r] /= len(toks)  # term frequency
        mat *= self.idf  # tf-idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float


class EvidenceIndex:
    def __init__(self, passages: list[Passage], embedder: Embedder | None = None):
        self.passages = passages
        self.embedder: Embedder = embedder or TfidfEmbedder()
        texts = [p.text for p in passages]
        self.matrix = self.embedder.fit(texts).transform(texts) if texts else np.zeros((0, 0))

    def query(self, text: str, k: int = 5, min_score: float = 0.0) -> list[RetrievedPassage]:
        if not self.passages:
            return []
        q = self.embedder.transform([text])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        out = [RetrievedPassage(self.passages[i], float(sims[i])) for i in order]
        return [r for r in out if r.score > min_score]
