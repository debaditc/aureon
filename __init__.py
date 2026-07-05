"""aureon (Adaptive Unified Retrieval Engine): dense + sparse fusion with a query router.

    from aureon import HybridSearch
    hs = HybridSearch(docs)
    hs.search("gateway GW-09 timeout", explain=True)

Fusion methods, quality metrics, and the efficiency harness are re-exported here
for evaluation scripts (see aureon.benchmark for the full report).
"""
from __future__ import annotations

from .core import HybridSearch
from .fusion import (
    Explanation,
    fixed_alpha, comb_sum, comb_mnz, dbsf,       # score-normalization family
    rrf, weighted_rrf, isr, borda,               # rank-based family
    adaptive, adaptive_rrf, oracle,              # routed / diagnostic
)
from .eval import (
    ndcg_at_k, recall_at_k, precision_at_k, hit_at_k,
    mrr, average_precision, r_precision, evaluate, paired_bootstrap,
)
from .timing import LatencyStats, measure

__all__ = [
    "HybridSearch", "Explanation",
    # fusion
    "fixed_alpha", "comb_sum", "comb_mnz", "dbsf",
    "rrf", "weighted_rrf", "isr", "borda",
    "adaptive", "adaptive_rrf", "oracle",
    # quality metrics
    "ndcg_at_k", "recall_at_k", "precision_at_k", "hit_at_k",
    "mrr", "average_precision", "r_precision", "evaluate", "paired_bootstrap",
    # efficiency
    "LatencyStats", "measure",
]
__version__ = "0.2.0"
