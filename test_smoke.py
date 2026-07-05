import numpy as np
import pytest
from aureon import (
    HybridSearch, rrf, ndcg_at_k,
    comb_sum, comb_mnz, dbsf, weighted_rrf, isr, borda,
    recall_at_k, precision_at_k, hit_at_k, mrr, average_precision,
    r_precision, evaluate, measure, LatencyStats,
)
from aureon.core import _METHODS

DOCS = [
    "The XR-2000 terminal returned error code E-417 during settlement.",
    "Reducing cloud spend requires rightsizing and reserved capacity.",
    "We cut our AWS bill by moving batch jobs to spot instances.",
    "Portfolio drawdown breached the risk limit and unwound positions.",
]


def test_search_returns_ranked_results():
    hs = HybridSearch(DOCS)
    out = hs.search("XR-2000 error E-417", k=3)
    assert len(out) == 3
    assert out[0]["doc"] == 0            # exact-match query should surface doc 0
    assert out[0]["rank"] == 1
    assert {"doc", "rank", "score", "text"} <= out[0].keys()


def test_explain_contract():
    hs = HybridSearch(DOCS)
    exp = hs.search("lower cloud costs", method="adaptive", explain=True)
    assert exp.dense_raw.shape == (len(DOCS),)
    assert exp.sparse_raw.shape == (len(DOCS),)
    assert 0.0 <= exp.alpha <= 1.0
    assert len(exp.order) == len(DOCS)
    assert sorted(exp.order.tolist()) == list(range(len(DOCS)))  # a permutation


def test_rrf_is_a_valid_permutation():
    ds = np.array([0.9, 0.1, 0.5, 0.2])
    ss = np.array([1.0, 8.0, 0.0, 3.0])
    exp = rrf(ds, ss, "q")
    assert sorted(exp.order.tolist()) == [0, 1, 2, 3]


def test_ndcg_bounds_and_perfect():
    assert ndcg_at_k([0, 1, 2], {0, 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([2, 3, 0], set(), 10) == 0.0


def test_bad_method_raises():
    hs = HybridSearch(DOCS)
    with pytest.raises(ValueError):
        hs.search("q", method="nonsense")


def test_pluggable_encoder():
    # trivial deterministic "encoder" to prove the hook works end-to-end
    def enc(texts):
        return np.array([[len(t), t.count("a")] for t in texts], dtype=float)
    hs = HybridSearch(DOCS, encoder=enc)
    out = hs.search("anything", k=2)
    assert len(out) == 2


# --- new fusion methods --------------------------------------------------- #

@pytest.mark.parametrize("method", _METHODS)
def test_every_method_returns_a_permutation(method):
    hs = HybridSearch(DOCS)
    exp = hs.search("XR-2000 error E-417", method=method, explain=True)
    assert sorted(exp.order.tolist()) == list(range(len(DOCS)))
    assert exp.fused.shape == (len(DOCS),)


@pytest.mark.parametrize("fn", [comb_sum, comb_mnz, dbsf, weighted_rrf, isr, borda])
def test_fusion_fns_are_valid_permutations(fn):
    ds = np.array([0.9, 0.1, 0.5, 0.2])
    ss = np.array([1.0, 8.0, 0.0, 3.0])
    exp = fn(ds, ss, "q")
    assert sorted(exp.order.tolist()) == [0, 1, 2, 3]


def test_score_norm_variants():
    ds = np.array([0.9, 0.1, 0.5, 0.2])
    ss = np.array([1.0, 8.0, 0.0, 3.0])
    for norm in ("minmax", "zscore", "softmax", "dbsf"):
        exp = comb_sum(ds, ss, "q", norm=norm)
        assert sorted(exp.order.tolist()) == [0, 1, 2, 3]


def test_weighted_rrf_collapses_to_rrf_at_equal_weights():
    ds = np.array([0.9, 0.1, 0.5, 0.2])
    ss = np.array([1.0, 8.0, 0.0, 3.0])
    a = weighted_rrf(ds, ss, "q", weights=(1.0, 1.0))
    b = rrf(ds, ss, "q")
    assert np.allclose(a.fused, b.fused)


# --- new quality metrics -------------------------------------------------- #

def test_set_metrics_bounds_and_perfect():
    order, rel = [0, 1, 2, 3], {0, 1}
    assert recall_at_k(order, rel, 2) == pytest.approx(1.0)
    assert precision_at_k(order, rel, 2) == pytest.approx(1.0)
    assert hit_at_k(order, rel, 2) == 1.0
    assert hit_at_k([2, 3], rel, 2) == 0.0
    assert recall_at_k(order, set(), 2) == 0.0     # no relevant -> 0, no crash


def test_mrr_and_ap_and_rprec():
    # first relevant at rank 2 -> reciprocal rank 1/2
    assert mrr([5, 0, 1], {0, 1}) == pytest.approx(0.5)
    # perfect ranking of both relevant docs -> AP = 1.0, R-Prec = 1.0
    assert average_precision([0, 1, 9], {0, 1}) == pytest.approx(1.0)
    assert r_precision([0, 1, 9], {0, 1}) == pytest.approx(1.0)


def test_evaluate_bundle_keys():
    m = evaluate([0, 1, 2, 3], {0, 1}, ks=(5, 10))
    assert {"ndcg@10", "mrr", "map", "r_prec",
            "recall@5", "p@5", "hit@5", "recall@10", "p@10", "hit@10"} <= m.keys()


# --- efficiency harness --------------------------------------------------- #

def test_measure_returns_latency_stats():
    stats = measure(lambda: sum(range(100)), repeat=20, warmup=2)
    assert isinstance(stats, LatencyStats)
    assert stats.n == 20
    assert stats.mean_ms >= 0.0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.qps > 0.0
