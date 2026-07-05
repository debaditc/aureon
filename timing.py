"""Efficiency metrics: a tiny, dependency-light latency harness.

Quality tells you *which* fusion method ranks best; efficiency tells you what it
costs. Data scientists comparing methods want both, so we report per-call
latency percentiles (mean / p50 / p95 / p99) and throughput (QPS) with warmup
and repeats to damp jitter. Kept separate from `eval.py` (relevance) on purpose.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np


def _percentile(samples_s: np.ndarray, p: float) -> float:
    return float(np.percentile(samples_s, p)) * 1e3  # -> ms


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    qps: float                        # calls/sec = 1000 / mean_ms

    @classmethod
    def from_samples(cls, samples_s) -> "LatencyStats":
        """Build stats from a list of per-call durations in SECONDS."""
        s = np.asarray(list(samples_s), dtype=float)
        if s.size == 0:
            return cls(0, *([float("nan")] * 6), float("nan"))
        mean_ms = float(s.mean()) * 1e3
        return cls(
            n=int(s.size),
            mean_ms=mean_ms,
            p50_ms=_percentile(s, 50),
            p95_ms=_percentile(s, 95),
            p99_ms=_percentile(s, 99),
            min_ms=float(s.min()) * 1e3,
            max_ms=float(s.max()) * 1e3,
            qps=(1e3 / mean_ms) if mean_ms > 0 else float("inf"),
        )

    def as_dict(self) -> dict[str, float]:
        return {"mean_ms": self.mean_ms, "p50_ms": self.p50_ms,
                "p95_ms": self.p95_ms, "p99_ms": self.p99_ms, "qps": self.qps}


def measure(fn, *, repeat: int = 200, warmup: int = 20) -> LatencyStats:
    """Time a zero-argument callable ``fn`` ``repeat`` times (after ``warmup``
    untimed calls) and return LatencyStats over the timed samples."""
    for _ in range(max(0, warmup)):
        fn()
    samples = np.empty(repeat, dtype=float)
    for i in range(repeat):
        t0 = perf_counter()
        fn()
        samples[i] = perf_counter() - t0
    return LatencyStats.from_samples(samples)


def time_once(fn) -> float:
    """Wall-clock seconds for a single call. Handy for one-off costs like
    index build time."""
    t0 = perf_counter()
    fn()
    return perf_counter() - t0
