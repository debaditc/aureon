"""Retrievers. Sparse = BM25 (exact lexical). Dense = LSA (semantic, offline).

NOTE ON THE DENSE STAND-IN: In production you swap DenseRetriever for a
transformer embedding model (sentence-transformers, etc.). We use TF-IDF+SVD
(Latent Semantic Analysis) here because it is a genuine *semantic* retriever
that runs fully offline with no model download. It is weaker than a modern
encoder, so absolute numbers below are directional, not authoritative. The
harness, baselines, and method plumbing are the reusable artifact.
"""
from __future__ import annotations
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _tok(s: str) -> list[str]:
    return s.lower().replace("-", " ").split()


class BM25Retriever:
    """Sparse lexical retriever. Strong on exact terms / codes / entities."""
    def __init__(self, docs: list[str]):
        self.docs = docs
        self._bm25 = BM25Okapi([_tok(d) for d in docs])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(_tok(query)), dtype=float)


class DenseRetriever:
    """Semantic retriever. Strong on paraphrase / conceptual queries.

    encoder: optional callable(list[str]) -> np.ndarray of shape (n, dim).
        If given, it is used to embed both docs and queries (this is how you
        plug in a sentence-transformer for the real BEIR test). If None, we
        fall back to an offline LSA (TF-IDF + SVD) stand-in.
    """
    def __init__(self, docs: list[str], n_components: int = 16, seed: int = 0,
                 encoder=None):
        self.docs = docs
        self._encoder = encoder
        if encoder is not None:
            self._emb = normalize(np.asarray(encoder(docs), dtype=float))
            self._vec = self._svd = None
        else:
            self._vec = TfidfVectorizer()
            X = self._vec.fit_transform(docs)
            k = min(n_components, min(X.shape) - 1)
            self._svd = TruncatedSVD(n_components=k, random_state=seed)
            self._emb = normalize(self._svd.fit_transform(X))

    def scores(self, query: str) -> np.ndarray:
        if self._encoder is not None:
            qv = normalize(np.asarray(self._encoder([query]), dtype=float))
        else:
            q = self._vec.transform([query])
            qv = normalize(self._svd.transform(q))
        return (self._emb @ qv.T).ravel()  # cosine in [-1, 1]
