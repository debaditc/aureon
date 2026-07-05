"""Metrics for offline retrieval evaluation (binary relevance).

Two families live here:

  * Quality / relevance metrics a data scientist expects in an eval report:
    nDCG@k, MRR, MAP (per-query AP), R-Precision, Recall@k, Precision@k,
    Hit/Success@k -- plus an ``evaluate()`` aggregator that returns them all
    for one ranking.
  * A paired bootstrap significance test so we don't celebrate noise-level
    deltas between methods.

`order` is an iterable of doc ids, best-first. `rel` is the set of relevant
doc ids for the query. Everything is label-driven and cutoff-explicit.
"""
from __future__ import annotations
import numpy as np


def _topk(order, k: int) -> list[int]:
    return [int(d) for d in list(order)[:k]]


def ndcg_at_k(order, rel: set[int], k: int = 10) -> float:
    """Normalized discounted cumulative gain. Rewards putting relevant docs
    high; the workhorse ranking metric."""
    top = _topk(order, k)
    dcg = sum((1.0 / np.log2(i + 2)) for i, d in enumerate(top) if d in rel)
    n_rel = min(len(rel), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(order, rel: set[int], k: int = 10) -> float:
    """Fraction of the relevant set retrieved within the top-k."""
    if not rel:
        return 0.0
    return len(set(_topk(order, k)) & rel) / len(rel)


def precision_at_k(order, rel: set[int], k: int = 10) -> float:
    """Fraction of the top-k that is relevant."""
    if k <= 0:
        return 0.0
    return len(set(_topk(order, k)) & rel) / k


def hit_at_k(order, rel: set[int], k: int = 10) -> float:
    """Success@k: 1.0 if at least one relevant doc is in the top-k, else 0.0."""
    return 1.0 if set(_topk(order, k)) & rel else 0.0


def mrr(order, rel: set[int], k: int | None = None) -> float:
    """Reciprocal rank of the FIRST relevant doc (0 if none). Optionally
    capped at rank k. Averaged across queries this is the MRR."""
    for i, d in enumerate(order, 1):
        if k is not None and i > k:
            break
        if int(d) in rel:
            return 1.0 / i
    return 0.0


def average_precision(order, rel: set[int], k: int | None = None) -> float:
    """Average precision for one query. Mean over relevant hits of the
    precision at each hit's rank. Averaged across queries this is the MAP."""
    if not rel:
        return 0.0
    hits, ap = 0, 0.0
    for i, d in enumerate(order, 1):
        if k is not None and i > k:
            break
        if int(d) in rel:
            hits += 1
            ap += hits / i
    return ap / min(len(rel), k) if k else ap / len(rel)


def r_precision(order, rel: set[int]) -> float:
    """Precision at cutoff R = |rel|. Cutoff-free, self-calibrating to how
    many relevant docs the query actually has."""
    return precision_at_k(order, rel, len(rel)) if rel else 0.0


def evaluate(order, rel: set[int], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """All quality metrics for a single ranking, keyed for a results table.

    Returns ndcg@K, mrr, map (this query's AP), r_prec, and recall@k /
    precision@k / hit@k for each k in ``ks`` (K = max(ks) for the rank metrics).
    """
    kmax = max(ks)
    out: dict[str, float] = {
        f"ndcg@{kmax}": ndcg_at_k(order, rel, kmax),
        "mrr": mrr(order, rel),
        "map": average_precision(order, rel),
        "r_prec": r_precision(order, rel),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(order, rel, k)
        out[f"p@{k}"] = precision_at_k(order, rel, k)
        out[f"hit@{k}"] = hit_at_k(order, rel, k)
    return out


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0):
    """P(mean(a) > mean(b)) style test. Returns (delta, p_value) for a-b>0."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    delta = float(d.mean())
    # two-sided-ish: fraction of resamples on the wrong side of 0
    p = float(np.mean(means <= 0)) if delta > 0 else float(np.mean(means >= 0))
    return delta, p
