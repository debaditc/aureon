"""Public API. One class, one method.

    from aureon import HybridSearch
    hs = HybridSearch(docs)                      # LSA dense by default
    hs.search("gateway GW-09 timeout")           # -> ranked [{doc, score, rank}]
    hs.search("lower our cloud costs", explain=True)   # -> Explanation (for the 3D demo)

Swap in a real encoder for the BEIR test:
    hs = HybridSearch(docs, encoder=my_sentence_transformer_encode)
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from .retrievers import BM25Retriever, DenseRetriever, _tok
from . import fusion
from .fusion import Explanation

_METHODS = ("bm25", "dense", "fixed", "rrf", "adaptive",
            "combsum", "combmnz", "zscore", "softmax", "dbsf",
            "wrrf", "isr", "borda", "adaptive_rrf")


class HybridSearch:
    def __init__(self, docs: list[str], encoder=None, n_components: int = 16):
        if not docs:
            raise ValueError("docs must be non-empty")
        self.docs = list(docs)
        self.n = len(self.docs)
        self.sparse = BM25Retriever(self.docs)
        self.dense = DenseRetriever(self.docs, n_components=n_components,
                                    encoder=encoder)
        self._df = self._corpus_df(self.docs)

    @staticmethod
    def _corpus_df(docs) -> dict[str, int]:
        df = Counter()
        for d in docs:
            for t in set(_tok(d)):
                df[t] += 1
        return dict(df)

    def explain(self, query: str, method: str = "adaptive", **kw) -> Explanation:
        """Return the full Explanation (dense/sparse/alpha/fused/order)."""
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        ds = self.dense.scores(query)
        ss = self.sparse.scores(query)
        if method == "bm25":
            order = np.argsort(-ss)
            return Explanation(query, float("nan"), "bm25", ds, ss, ss, order)
        if method == "dense":
            order = np.argsort(-ds)
            return Explanation(query, 1.0, "dense", ds, ss, ds, order)
        if method == "fixed":
            return fusion.fixed_alpha(ds, ss, query, kw.get("alpha", 0.5))
        if method == "rrf":
            return fusion.rrf(ds, ss, query, kw.get("k", 60))
        if method == "combsum":
            return fusion.comb_sum(ds, ss, query, kw.get("norm", "minmax"))
        if method == "combmnz":
            return fusion.comb_mnz(ds, ss, query, kw.get("norm", "minmax"))
        if method == "zscore":
            return fusion.comb_sum(ds, ss, query, "zscore")
        if method == "softmax":
            return fusion.comb_sum(ds, ss, query, "softmax")
        if method == "dbsf":
            return fusion.dbsf(ds, ss, query)
        if method == "wrrf":
            return fusion.weighted_rrf(ds, ss, query, kw.get("k", 60),
                                       kw.get("weights", (1.0, 1.0)))
        if method == "isr":
            return fusion.isr(ds, ss, query)
        if method == "borda":
            return fusion.borda(ds, ss, query)
        if method == "adaptive_rrf":
            return fusion.adaptive_rrf(ds, ss, query, self._df, self.n,
                                       kw.get("lo", 0.25), kw.get("hi", 0.75),
                                       kw.get("k", 60))
        return fusion.adaptive(ds, ss, query, self._df, self.n,
                               kw.get("lo", 0.25), kw.get("hi", 0.75))

    def search(self, query: str, method: str = "adaptive", k: int = 10,
               explain: bool = False, **kw):
        """Ranked results. explain=True returns the Explanation instead."""
        exp = self.explain(query, method=method, **kw)
        if explain:
            return exp
        return [dict(doc=int(i), rank=r, score=float(exp.fused[i]),
                     text=self.docs[i])
                for r, i in enumerate(exp.order[:k], 1)]
