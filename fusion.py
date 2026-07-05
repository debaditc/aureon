"""Fusion strategies + the explain-mode data contract.

The contract (Explanation): per query we expose, for every candidate, the raw
dense score, raw sparse score, the chosen weight alpha, the fused score, and the
final rank. This is what repo 2's 3D demo will consume. It is designed in from
day one so the API never has to warp to add it later.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Explanation:
    query: str
    alpha: float                      # weight on DENSE (1-alpha on sparse); NaN for RRF
    method: str
    dense_raw: np.ndarray
    sparse_raw: np.ndarray
    fused: np.ndarray
    order: np.ndarray                 # doc indices, best-first
    meta: dict = field(default_factory=dict)

    def top(self, k: int = 5) -> list[dict]:
        out = []
        for rank, i in enumerate(self.order[:k], 1):
            out.append(dict(doc=int(i), rank=rank,
                            dense=float(self.dense_raw[i]),
                            sparse=float(self.sparse_raw[i]),
                            fused=float(self.fused[i])))
        return out


# --------------------------------------------------------------------------- #
# Score normalizers. Each maps a raw score vector to a comparable range so two
# retrievers with different score distributions can be summed.
# --------------------------------------------------------------------------- #
def _minmax(x: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]. Aligns ranges, not distributions (its weakness)."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize to mean 0 / std 1. Distribution-aware; handles outliers and
    range mismatch better than min-max."""
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Turn scores into a probability distribution. Emphasizes the top; useful
    when only the head of each list should carry weight."""
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum() if e.sum() > 0 else np.zeros_like(x)


def _dbsf_norm(x: np.ndarray) -> np.ndarray:
    """Distribution-Based Score Fusion normalizer: min-max between mean-3sigma
    and mean+3sigma, clipped to [0,1]. Robust to per-retriever scale/offset;
    this is what modern hybrid stores reach for."""
    m, s = x.mean(), x.std()
    if s == 0:
        return np.zeros_like(x)
    lo, hi = m - 3 * s, m + 3 * s
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


_NORMS = {"minmax": _minmax, "zscore": _zscore,
          "softmax": _softmax, "dbsf": _dbsf_norm}


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based competition-free ranks (best score -> rank 1)."""
    n = len(x)
    r = np.empty(n)
    r[np.argsort(-x)] = np.arange(1, n + 1)
    return r


def _alpha_of(wd: float, ws: float) -> float:
    tot = wd + ws
    return wd / tot if tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Score-normalization fusion family
# --------------------------------------------------------------------------- #
def fixed_alpha(dense: np.ndarray, sparse: np.ndarray, query: str,
                alpha: float = 0.5) -> Explanation:
    """Naive weighted score fusion on min-max normalized scores.
    Known-fragile: normalization aligns ranges, not distributions."""
    fused = alpha * _minmax(dense) + (1 - alpha) * _minmax(sparse)
    order = np.argsort(-fused)
    return Explanation(query, alpha, f"fixed@{alpha:.2f}", dense, sparse, fused, order)


def comb_sum(dense: np.ndarray, sparse: np.ndarray, query: str,
             norm: str = "minmax", weights: tuple[float, float] = (0.5, 0.5)) -> Explanation:
    """CombSUM: weighted sum of normalized scores. With norm='zscore' or
    'softmax' this is z-score / softmax fusion; the classic Fox & Shaw combiner."""
    f = _NORMS[norm]
    wd, ws = weights
    fused = wd * f(dense) + ws * f(sparse)
    order = np.argsort(-fused)
    return Explanation(query, _alpha_of(wd, ws), f"combsum/{norm}",
                       dense, sparse, fused, order, meta={"norm": norm})


def comb_mnz(dense: np.ndarray, sparse: np.ndarray, query: str,
             norm: str = "minmax", weights: tuple[float, float] = (0.5, 0.5)) -> Explanation:
    """CombMNZ: CombSUM scaled by how many retrievers 'matched' the doc.
    We count a retriever as matching when its RAW score > 0 (BM25 gives exact
    zeros on no lexical overlap; dense cosine > 0 means positive similarity), so
    docs both retrievers agree on get boosted."""
    f = _NORMS[norm]
    wd, ws = weights
    hits = (dense > 0).astype(float) + (sparse > 0).astype(float)
    fused = hits * (wd * f(dense) + ws * f(sparse))
    order = np.argsort(-fused)
    return Explanation(query, _alpha_of(wd, ws), f"combmnz/{norm}",
                       dense, sparse, fused, order, meta={"norm": norm})


def dbsf(dense: np.ndarray, sparse: np.ndarray, query: str,
         weights: tuple[float, float] = (0.5, 0.5)) -> Explanation:
    """Distribution-Based Score Fusion: CombSUM over the 3-sigma normalizer."""
    exp = comb_sum(dense, sparse, query, norm="dbsf", weights=weights)
    exp.method = "dbsf"
    return exp


# --------------------------------------------------------------------------- #
# Rank-based fusion family (score-distribution-agnostic)
# --------------------------------------------------------------------------- #
def rrf(dense: np.ndarray, sparse: np.ndarray, query: str,
        k: int = 60) -> Explanation:
    """Reciprocal Rank Fusion. Rank-based, score-distribution-agnostic.
    The baseline that actually matters."""
    dr, sr = _ranks(dense), _ranks(sparse)
    fused = 1.0 / (k + dr) + 1.0 / (k + sr)
    order = np.argsort(-fused)
    return Explanation(query, float("nan"), f"rrf@{k}", dense, sparse, fused, order)


def weighted_rrf(dense: np.ndarray, sparse: np.ndarray, query: str,
                 k: int = 60, weights: tuple[float, float] = (1.0, 1.0)) -> Explanation:
    """RRF with per-retriever weights. Rank-based like RRF but lets a router
    tilt toward dense or sparse without touching score distributions."""
    dr, sr = _ranks(dense), _ranks(sparse)
    wd, ws = weights
    fused = wd / (k + dr) + ws / (k + sr)
    order = np.argsort(-fused)
    return Explanation(query, _alpha_of(wd, ws), f"wrrf@{k}", dense, sparse, fused, order)


def isr(dense: np.ndarray, sparse: np.ndarray, query: str) -> Explanation:
    """Inverse Square Rank fusion: sum of 1/rank^2. Sharper head emphasis than
    RRF -- rewards a doc that either retriever ranks very near the top."""
    dr, sr = _ranks(dense), _ranks(sparse)
    fused = 1.0 / dr**2 + 1.0 / sr**2
    order = np.argsort(-fused)
    return Explanation(query, float("nan"), "isr", dense, sparse, fused, order)


def borda(dense: np.ndarray, sparse: np.ndarray, query: str) -> Explanation:
    """Borda count: each retriever awards (n - rank) points; sum them. Linear
    in rank, so it values agreement across the whole list, not just the head."""
    n = len(dense)
    dr, sr = _ranks(dense), _ranks(sparse)
    fused = (n - dr) + (n - sr)
    order = np.argsort(-fused)
    return Explanation(query, float("nan"), "borda", dense, sparse, fused, order)


def _lexicality(query: str, corpus_df: dict[str, int], n_docs: int) -> float:
    """Heuristic router signal in [0,1]. High => query looks lexical/exact
    (rare tokens, codes, digits) => lean sparse. Low => conceptual => lean dense.
    Deliberately simple and label-free; this is the honest weak router."""
    toks = query.lower().replace("-", " ").split()
    if not toks:
        return 0.5
    signals = []
    for t in toks:
        df = corpus_df.get(t, 0)
        rare = 1.0 if df <= 2 else 0.0            # rare in corpus
        has_digit = 1.0 if any(c.isdigit() for c in t) else 0.0
        signals.append(max(rare, has_digit))
    return float(np.mean(signals))


def adaptive(dense: np.ndarray, sparse: np.ndarray, query: str,
             corpus_df: dict[str, int], n_docs: int,
             lo: float = 0.25, hi: float = 0.75) -> Explanation:
    """Query-adaptive weighting. Router maps lexicality -> alpha (weight on dense).
    Lexical query -> low alpha (favor sparse). Conceptual -> high alpha."""
    lex = _lexicality(query, corpus_df, n_docs)
    alpha = hi - (hi - lo) * lex          # lex=0 -> hi (dense); lex=1 -> lo (sparse)
    fused = alpha * _minmax(dense) + (1 - alpha) * _minmax(sparse)
    order = np.argsort(-fused)
    return Explanation(query, alpha, "adaptive", dense, sparse, fused, order,
                       meta={"lexicality": lex})


def adaptive_rrf(dense: np.ndarray, sparse: np.ndarray, query: str,
                 corpus_df: dict[str, int], n_docs: int,
                 lo: float = 0.25, hi: float = 0.75, k: int = 60) -> Explanation:
    """The router (lexicality -> dense weight) feeding weighted RRF instead of
    score fusion. Marries per-query weighting to a rank-based fuser, so the
    weight decision is decoupled from the two retrievers' score scales."""
    lex = _lexicality(query, corpus_df, n_docs)
    wd = hi - (hi - lo) * lex              # lex=0 -> dense-heavy; lex=1 -> sparse-heavy
    exp = weighted_rrf(dense, sparse, query, k, (wd, 1.0 - wd))
    return Explanation(query, wd, "adaptive_rrf", dense, sparse, exp.fused,
                       exp.order, meta={"lexicality": lex})


def oracle(dense: np.ndarray, sparse: np.ndarray, query: str,
           rel: set[int], grid=None) -> Explanation:
    """Cheating upper bound: pick the alpha that maximizes this query's nDCG,
    with hindsight. Diagnostic only. If oracle can't beat RRF, the *concept*
    of per-query weighting is dead. If oracle beats RRF but adaptive doesn't,
    the concept is fine and the *router* is the problem."""
    from .eval import ndcg_at_k
    grid = grid if grid is not None else np.linspace(0, 1, 21)
    best_a, best_order, best_s = 0.5, None, -1.0
    for a in grid:
        fused = a * _minmax(dense) + (1 - a) * _minmax(sparse)
        order = np.argsort(-fused)
        s = ndcg_at_k(order, rel, 10)
        if s > best_s:
            best_s, best_a, best_order = s, a, order
    fused = best_a * _minmax(dense) + (1 - best_a) * _minmax(sparse)
    return Explanation(query, float(best_a), "oracle", dense, sparse, fused,
                       best_order, meta={"oracle_ndcg": best_s})
