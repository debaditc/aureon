"""Evaluation harness: quality AND efficiency for every fusion method.

    python -m aureon.benchmark    (or the installed console script: aureon-bench)

Three reports, so a data scientist can compare methods on the axes that matter:

  1. QUALITY  -- nDCG@10, MRR, MAP, R-Precision, Recall@10, Precision@10 per
     method, plus the lexical-vs-semantic nDCG split (the router thesis).
  2. EFFICIENCY -- per-method fusion latency (mean / p50 / p95 / p99) and
     throughput (QPS), timed on precomputed retriever scores so the numbers
     isolate the *fusion* cost. Shared retrieval + index-build cost is reported
     once alongside.
  3. SIGNIFICANCE -- paired bootstrap of adaptive / adaptive_rrf / oracle vs the
     RRF baseline on nDCG@10, so we don't over-read noise-level deltas.

The oracle is a hindsight-optimal upper bound: if it can't beat RRF the concept
is dead; if it beats RRF but the routers don't, the router is the bottleneck.
"""
from collections import Counter
import numpy as np

from .retrievers import BM25Retriever, DenseRetriever, _tok
from . import fusion
from .eval import evaluate, ndcg_at_k, paired_bootstrap
from .timing import LatencyStats, time_once
from .data import DOCS, QUERIES

K = 10
KS = (5, 10)
# Metrics shown in the quality table (keys come from eval.evaluate).
QUALITY_COLS = [f"ndcg@{K}", "mrr", "map", "r_prec", f"recall@{K}", f"p@{K}"]
QUALITY_HDR = {f"ndcg@{K}": "nDCG@10", "mrr": "MRR", "map": "MAP",
               "r_prec": "R-Prec", f"recall@{K}": "R@10", f"p@{K}": "P@10"}


def _corpus_df(docs):
    df = Counter()
    for d in docs:
        for t in set(_tok(d)):
            df[t] += 1
    return dict(df)


def _method_table(df, n):
    """name -> fn(ds, ss, q, rel) -> order (best-first doc ids).

    One registry drives both scoring and timing, so the two reports never drift.
    """
    return {
        "bm25":         lambda ds, ss, q, rel: np.argsort(-ss),
        "dense":        lambda ds, ss, q, rel: np.argsort(-ds),
        "fixed@0.50":   lambda ds, ss, q, rel: fusion.fixed_alpha(ds, ss, q, 0.5).order,
        "combsum":      lambda ds, ss, q, rel: fusion.comb_sum(ds, ss, q, "minmax").order,
        "combmnz":      lambda ds, ss, q, rel: fusion.comb_mnz(ds, ss, q, "minmax").order,
        "zscore":       lambda ds, ss, q, rel: fusion.comb_sum(ds, ss, q, "zscore").order,
        "softmax":      lambda ds, ss, q, rel: fusion.comb_sum(ds, ss, q, "softmax").order,
        "dbsf":         lambda ds, ss, q, rel: fusion.dbsf(ds, ss, q).order,
        "rrf":          lambda ds, ss, q, rel: fusion.rrf(ds, ss, q).order,
        "wrrf":         lambda ds, ss, q, rel: fusion.weighted_rrf(ds, ss, q).order,
        "isr":          lambda ds, ss, q, rel: fusion.isr(ds, ss, q).order,
        "borda":        lambda ds, ss, q, rel: fusion.borda(ds, ss, q).order,
        "adaptive":     lambda ds, ss, q, rel: fusion.adaptive(ds, ss, q, df, n).order,
        "adaptive_rrf": lambda ds, ss, q, rel: fusion.adaptive_rrf(ds, ss, q, df, n).order,
        "oracle*":      lambda ds, ss, q, rel: fusion.oracle(ds, ss, q, rel).order,
    }


def _latency(fn, cases, repeat, warmup):
    """Time ``fn`` over every case, ``repeat`` times, after ``warmup`` passes."""
    from time import perf_counter
    for _ in range(warmup):
        for ds, ss, q, rel in cases:
            fn(ds, ss, q, rel)
    samples = []
    for _ in range(repeat):
        for ds, ss, q, rel in cases:
            t0 = perf_counter()
            fn(ds, ss, q, rel)
            samples.append(perf_counter() - t0)
    return LatencyStats.from_samples(samples)


def run(docs=DOCS, queries=QUERIES, encoder=None, repeat=100, warmup=5):
    # --- build indexes (measure the one-off cost) ---
    t_bm25 = t_dense = 0.0
    bm25 = dense = None

    def _bm():
        nonlocal bm25
        bm25 = BM25Retriever(docs)

    def _dn():
        nonlocal dense
        dense = DenseRetriever(docs, encoder=encoder)

    t_bm25 = time_once(_bm)
    t_dense = time_once(_dn)
    df = _corpus_df(docs)
    n = len(docs)

    methods = _method_table(df, n)

    # Precompute retriever scores once per query (shared across all methods).
    cases = [(dense.scores(q), bm25.scores(q), q, rel) for q, rel, _ in queries]
    qtypes = [t for _, _, t in queries]

    # --- QUALITY ---
    # per method: {metric_key: [per-query values]} and per-type nDCG lists
    qual = {m: {c: [] for c in QUALITY_COLS} for m in methods}
    by_type = {m: {"lexical": [], "semantic": []} for m in methods}
    ndcg_all = {m: [] for m in methods}
    for (ds, ss, q, rel), qtype in zip(cases, qtypes):
        for m, fn in methods.items():
            order = fn(ds, ss, q, rel)
            mets = evaluate(order, rel, ks=KS)
            for c in QUALITY_COLS:
                qual[m][c].append(mets[c])
            nd = ndcg_at_k(order, rel, K)
            ndcg_all[m].append(nd)
            by_type[m][qtype].append(nd)

    # --- EFFICIENCY ---
    eff = {m: _latency(fn, cases, repeat, warmup) for m, fn in methods.items()}

    _print_reports(docs, queries, methods, qual, by_type, ndcg_all, eff,
                   t_bm25, t_dense, repeat)
    return {"quality": qual, "by_type": by_type, "efficiency": eff}


def _print_reports(docs, queries, methods, qual, by_type, ndcg_all, eff,
                   t_bm25, t_dense, repeat):
    n = len(docs)
    n_lex = sum(t == "lexical" for _, _, t in queries)
    print(f"\nCorpus: {n} docs | Queries: {len(queries)} "
          f"({n_lex} lexical, {len(queries) - n_lex} semantic)")
    print(f"Index build: bm25 {t_bm25 * 1e3:.1f} ms | dense {t_dense * 1e3:.1f} ms "
          f"(one-off, shared by all methods)\n")

    # 1) QUALITY
    print("QUALITY (higher is better)")
    hdr = f"{'method':<13}" + "".join(f"{QUALITY_HDR[c]:>9}" for c in QUALITY_COLS)
    print(hdr); print("-" * len(hdr))
    for m in methods:
        row = f"{m:<13}" + "".join(f"{np.mean(qual[m][c]):>9.3f}" for c in QUALITY_COLS)
        print(row)

    # 2) nDCG split by query type (the router thesis)
    print("\nnDCG@10 BY QUERY TYPE")
    hdr = f"{'method':<13}{'all':>9}{'lexical':>9}{'semantic':>9}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        print(f"{m:<13}{np.mean(ndcg_all[m]):>9.3f}"
              f"{np.mean(by_type[m]['lexical']):>9.3f}"
              f"{np.mean(by_type[m]['semantic']):>9.3f}")
    print("\n* oracle = hindsight-optimal alpha per query (cheating upper bound)")

    # 3) EFFICIENCY
    print(f"\nEFFICIENCY (fusion cost only; {repeat} reps x {len(queries)} queries)")
    hdr = (f"{'method':<13}{'mean_ms':>9}{'p50_ms':>9}{'p95_ms':>9}"
           f"{'p99_ms':>9}{'QPS':>10}")
    print(hdr); print("-" * len(hdr))
    for m in methods:
        s = eff[m]
        print(f"{m:<13}{s.mean_ms:>9.4f}{s.p50_ms:>9.4f}{s.p95_ms:>9.4f}"
              f"{s.p99_ms:>9.4f}{s.qps:>10.0f}")

    # 4) SIGNIFICANCE vs RRF baseline
    print("\nSIGNIFICANCE vs RRF (nDCG@10, paired bootstrap)")
    for m in ("adaptive", "adaptive_rrf", "oracle*"):
        delta, p = paired_bootstrap(ndcg_all[m], ndcg_all["rrf"])
        print(f"  {m:<13} - rrf:  delta = {delta:+.3f}   p = {p:.3f}")
    print()


def main():
    """Console entry point (returns None so setuptools doesn't echo the dict)."""
    run()


if __name__ == "__main__":
    main()
