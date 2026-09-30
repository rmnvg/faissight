"""Recall@k and latency across nprobe / efSearch values, for picking a search setting."""

from __future__ import annotations

import hashlib
import platform
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

from faissight.core import ivf
from faissight.core._faiss import import_faiss
from faissight.core.search import GroundTruth, resolve_search_params
from faissight.core.types import LoadedIndex
from faissight.core.vectors import VectorSource

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
ProgressFn = Callable[[float, str], None]

DEFAULT_N_QUERIES = 200
DEFAULT_EF_VALUES = (16, 32, 64, 128, 256, 512)

# omp_set_num_threads is process-wide: serialise timed sweeps so one can't restore the
# thread count while another is still measuring.
_TIMING_LOCK = threading.Lock()


class SweepParam(str, Enum):
    NPROBE = "nprobe"
    EF_SEARCH = "efSearch"


def param_for(li: LoadedIndex) -> SweepParam:
    """The search parameter that trades speed for recall on this index."""
    if li.kind.is_ivf:
        return SweepParam.NPROBE
    if li.kind.is_hnsw:
        return SweepParam.EF_SEARCH
    raise ValueError(f"{li.kind.value} indexes have no search parameter to sweep.")


def default_values(li: LoadedIndex, param: SweepParam | str | None = None) -> list[int]:
    """nprobe: powers of two up to nlist (plus nlist). efSearch: 16..512."""
    param = SweepParam(param) if param is not None else param_for(li)
    _check_param(li, param)
    if param is SweepParam.NPROBE:
        nlist = int(li.ivf.nlist)
        values = [1 << i for i in range(nlist.bit_length()) if (1 << i) <= nlist]
        if values[-1] != nlist:
            values.append(nlist)
        return values
    return list(DEFAULT_EF_VALUES)


def check_values(
    li: LoadedIndex, param: SweepParam | str | None, values: Sequence[int] | None
) -> tuple[SweepParam, list[int]]:
    """Resolve the parameter and values (defaults if ``None``), validating every value.

    Raises ``ValueError`` for values the index rejects (e.g. nprobe > nlist).
    """
    p = SweepParam(param) if param is not None else param_for(li)
    _check_param(li, p)
    vals = sorted({int(v) for v in (values if values is not None else default_values(li, p))})
    if not vals:
        raise ValueError("Give at least one value to sweep.")
    for v in vals:
        if p is SweepParam.NPROBE:
            resolve_search_params(li, nprobe=v)
        else:
            resolve_search_params(li, ef_search=v)
    return p, vals


def _check_param(li: LoadedIndex, param: SweepParam) -> None:
    if param is SweepParam.NPROBE and not li.kind.is_ivf:
        raise ValueError("nprobe sweeps need an IVF index.")
    if param is SweepParam.EF_SEARCH and not li.kind.is_hnsw:
        raise ValueError("efSearch sweeps need an HNSW index.")


# --- query sets --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySet:
    """Query vectors for a sweep. ``exclude_ids[i]`` is query i's own id if it is stored."""

    vectors: FloatArray
    exclude_ids: IntArray | None
    origin: Literal["given", "sampled"]
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.vectors)


def given_queries(li: LoadedIndex, vectors: npt.ArrayLike) -> QuerySet:
    q = np.ascontiguousarray(vectors, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != li.d or len(q) == 0:
        raise ValueError(f"Queries must have shape (n, {li.d}) with n >= 1, got {q.shape}.")
    if not np.isfinite(q).all():
        raise ValueError("Queries contain NaN or infinite values.")
    return QuerySet(q, None, "given")


def sample_queries(source: VectorSource, n: int = DEFAULT_N_QUERIES, seed: int = 0) -> QuerySet:
    """``n`` stored vectors as queries (seeded); each excludes its own id from results."""
    if len(source) == 0:
        raise ValueError("Cannot sample queries from an empty index.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if n < 1:
        raise ValueError(f"n_queries must be >= 1, got {n}.")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(source), size=min(n, len(source)), replace=False))
    return QuerySet(source.vectors[rows], source.ids[rows].astype(np.int64), "sampled", seed)


# --- sweep -------------------------------------------------------------------------------


N_WORST_QUERIES = 10


@dataclass(frozen=True)
class WorstQuery:
    """One of the lowest-recall queries at a setting."""

    query_no: int
    """Row in the query set."""
    id: int | None
    """The query's own stored id for sampled queries; ``None`` for given queries."""
    recall: float
    probe_coverage: float | None = None
    """IVF: share of its true neighbours in the lists probed at this setting (caps recall)."""


@dataclass(frozen=True)
class SweepPoint:
    value: int
    recall: float
    """Mean recall@k over the query set."""
    latency_mean_ms: float
    latency_p95_ms: float
    """Per-query latency, single-threaded."""
    recall_ci_low: float | None = None
    recall_ci_high: float | None = None
    """Approximate 95% interval for the mean recall (normal approximation over queries)."""
    recall_distribution: list[tuple[float, int]] = field(default_factory=list)
    """Exact per-query recall distribution as ``(recall, n_queries)``, lowest recall first.

    Recall@k takes few distinct values (at most k + 1), so this stays small.
    """
    worst_queries: list[WorstQuery] = field(default_factory=list)
    """Up to ``N_WORST_QUERIES`` lowest-recall queries, worst first (ties by query order)."""
    probe_coverage: float | None = None
    """IVF: mean share of true neighbours in probed lists, the most recall probing allows.

    ``probe_coverage - recall`` is the share lost inside probed lists (compression,
    transforms, ties); ``1 - probe_coverage`` is the share in lists that weren't probed.
    """

    def fraction_below(self, target_recall: float) -> float | None:
        """Share of queries whose own recall is below ``target_recall`` (None if unknown)."""
        total = sum(n for _, n in self.recall_distribution)
        if total == 0:
            return None
        below = sum(n for r, n in self.recall_distribution if r < target_recall - 1e-9)
        return below / total


def summarize_recalls(
    recalls: npt.NDArray[np.float64],
    exclude_ids: IntArray | None,
    coverage: npt.NDArray[np.float64] | None = None,
) -> tuple[float, float, list[tuple[float, int]], list[WorstQuery]]:
    """95% interval of the mean, exact distribution and worst queries for per-query recalls.

    ``coverage`` (IVF) is each query's probe coverage, attached to its worst queries.
    """
    n = len(recalls)
    mean = float(recalls.mean())
    if n > 1:
        half = 1.96 * float(recalls.std(ddof=1)) / np.sqrt(n)
        low, high = max(0.0, mean - half), min(1.0, mean + half)
    else:
        low, high = 0.0, 1.0
    values, counts = np.unique(np.round(recalls, 6), return_counts=True)
    distribution = [(float(v), int(c)) for v, c in zip(values, counts, strict=True)]
    order = np.argsort(recalls, kind="stable")[:N_WORST_QUERIES]
    worst = [
        WorstQuery(
            int(i),
            int(exclude_ids[i]) if exclude_ids is not None else None,
            float(recalls[i]),
            float(coverage[i]) if coverage is not None else None,
        )
        for i in order
    ]
    return low, high, distribution, worst


@dataclass(frozen=True)
class SweepResult:
    param: SweepParam
    k: int
    n_queries: int
    query_origin: Literal["given", "sampled"]
    truth_reconstructed: bool
    points: list[SweepPoint]
    repeats: int = 3
    seed: int = 0
    query_sha256: str = ""
    environment: dict[str, str | int] = field(default_factory=dict)
    query_seed: int | None = None
    coverage_curve: npt.NDArray[np.float64] | None = field(default=None, repr=False, compare=False)
    """IVF: probe coverage at every nprobe ``0..nlist`` (see :func:`ivf.probe_coverage`)."""

    def coverage_at(self, nprobe: int) -> float | None:
        """Probe coverage at any nprobe, measured or not; ``None`` without IVF data."""
        if self.coverage_curve is None:
            return None
        return float(self.coverage_curve[min(max(nprobe, 0), len(self.coverage_curve) - 1)])

    def nprobe_for_coverage(self, coverage: float) -> int | None:
        """Smallest nprobe whose probe coverage reaches ``coverage``; ``None`` if none does."""
        if self.coverage_curve is None:
            return None
        hit = np.nonzero(self.coverage_curve >= coverage - 1e-9)[0]
        return int(hit[0]) if len(hit) else None

    def recommend(self, target_recall: float, *, confident: bool = False) -> SweepPoint | None:
        """Smallest parameter value whose mean recall meets ``target_recall``.

        Picks by parameter value, not measured latency, because latency is noisy and
        cost grows with the parameter. With ``confident``, the lower end of the recall
        interval must meet the target instead of the mean.
        """
        ok = [p for p in self.points if _meets(p, target_recall, confident)]
        return min(ok, key=lambda p: p.value) if ok else None

    def fastest(self, target_recall: float, *, confident: bool = False) -> SweepPoint | None:
        """The measured-fastest setting (mean latency) that meets ``target_recall``.

        Often the same as :meth:`recommend`; when it isn't, the gap is usually timing noise
        unless it is large or repeats across runs.
        """
        ok = [p for p in self.points if _meets(p, target_recall, confident)]
        return min(ok, key=lambda p: (p.latency_mean_ms, p.value)) if ok else None

    def pareto(self) -> list[SweepPoint]:
        """Points not beaten on both recall and mean latency, fastest first."""
        front = [
            p
            for p in self.points
            if not any(
                (o.recall >= p.recall and o.latency_mean_ms < p.latency_mean_ms)
                or (o.recall > p.recall and o.latency_mean_ms <= p.latency_mean_ms)
                for o in self.points
            )
        ]
        return sorted(front, key=lambda p: p.latency_mean_ms)


def _meets(p: SweepPoint, target_recall: float, confident: bool) -> bool:
    recall = p.recall_ci_low if confident and p.recall_ci_low is not None else p.recall
    return recall >= target_recall - 1e-9


def ground_truth_ids(gt: GroundTruth, queries: QuerySet, k: int) -> IntArray:
    """Exact top-k ids for every query (self excluded for sampled queries)."""
    return gt.search_batch(queries.vectors, k, queries.exclude_ids)


def sweep(
    li: LoadedIndex,
    queries: QuerySet,
    truth: IntArray,
    *,
    param: SweepParam | str | None = None,
    values: Sequence[int] | None = None,
    k: int = 10,
    truth_reconstructed: bool = False,
    progress: ProgressFn | None = None,
    repeats: int = 3,
    seed: int = 0,
    assignments: ivf.Assignments | None = None,
) -> SweepResult:
    """Measure recall@k and per-query latency at each parameter value.

    Queries run one at a time on a single FAISS thread (restored afterwards), after an
    untimed warm-up pass, so latencies are stable and comparable across values. For an
    nprobe sweep, ``assignments`` (from :func:`ivf.assignments`) adds probe coverage: how
    many true neighbours sit in probed lists, which separates probing from ranking losses.
    """
    faiss = import_faiss()
    param, vals = check_values(li, param, values)
    if repeats < 1 or seed < 0:
        raise ValueError("repeats must be >= 1 and seed must be non-negative.")
    if len(queries) == 0:
        raise ValueError("Give at least one query.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if truth.shape != (len(queries), k):
        raise ValueError(f"truth must have shape ({len(queries)}, {k}), got {truth.shape}.")
    report = progress or (lambda frac, msg: None)

    def params_for(v: int) -> object:
        if param is SweepParam.NPROBE:
            return resolve_search_params(li, nprobe=v)[0]
        return resolve_search_params(li, ef_search=v)[0]

    all_params = [params_for(v) for v in vals]
    excl = queries.exclude_ids
    k_search = min(k + (excl is not None), max(li.ntotal, 1))
    n = len(queries)
    points: list[SweepPoint] = []
    ranks, curve = None, None
    if param is SweepParam.NPROBE and assignments is not None:
        report(0.0, "Locating true neighbours' lists")
        ranks = ivf.truth_probe_ranks(li, queries.vectors, truth, assignments)
        curve = ivf.probe_coverage(ranks, int(li.ivf.nlist))

    with _TIMING_LOCK:
        prev_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            for step, (v, sp) in enumerate(zip(vals, all_params, strict=True)):
                report(step / len(vals), f"{param.value}={v}: warming up")
                li.index.search(queries.vectors, k_search, params=sp)
                latencies = np.empty(n * repeats, dtype=np.float64)
                recalls = np.empty(n, dtype=np.float64)
                rng = np.random.default_rng(seed)
                for repeat in range(repeats):
                    for j, i in enumerate(rng.permutation(n)):
                        if j % 32 == 0:
                            report(
                                (step + (repeat + j / n) / repeats) / len(vals),
                                f"{param.value}={v}: repeat {repeat + 1}/{repeats}",
                            )
                        q = queries.vectors[i : i + 1]
                        t0 = time.perf_counter()
                        _, found = li.index.search(q, k_search, params=sp)
                        latencies[repeat * n + j] = time.perf_counter() - t0
                        if repeat == 0:
                            row = found[0]
                            if excl is not None:
                                row = row[row != excl[i]]
                            true_row = truth[i][truth[i] >= 0]
                            recalls[i] = np.isin(true_row, row[:k]).mean() if len(true_row) else 1.0
                coverage = ivf.query_probe_coverage(ranks, v) if ranks is not None else None
                low, high, distribution, worst = summarize_recalls(recalls, excl, coverage)
                points.append(
                    SweepPoint(
                        value=v,
                        recall=float(recalls.mean()),
                        latency_mean_ms=float(latencies.mean() * 1000),
                        latency_p95_ms=float(np.percentile(latencies, 95) * 1000),
                        recall_ci_low=low,
                        recall_ci_high=high,
                        recall_distribution=distribution,
                        worst_queries=worst,
                        probe_coverage=float(coverage.mean()) if coverage is not None else None,
                    )
                )
        finally:
            faiss.omp_set_num_threads(prev_threads)
    report(1.0, "Done")
    return SweepResult(
        param,
        k,
        n,
        queries.origin,
        truth_reconstructed,
        points,
        repeats=repeats,
        seed=seed,
        query_sha256=query_fingerprint(queries),
        environment=benchmark_environment(),
        query_seed=queries.seed,
        coverage_curve=curve,
    )


def query_fingerprint(queries: QuerySet) -> str:
    """Fingerprint query values, shape and self-exclusions using canonical byte order."""
    digest = hashlib.sha256(str(queries.vectors.shape).encode())
    digest.update(np.ascontiguousarray(queries.vectors, dtype="<f4").tobytes())
    if queries.exclude_ids is not None:
        digest.update(np.ascontiguousarray(queries.exclude_ids, dtype="<i8").tobytes())
    return digest.hexdigest()


def benchmark_environment() -> dict[str, str | int]:
    """Versions and hardware context for a single-threaded latency measurement."""
    faiss = import_faiss()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": str(faiss.__version__),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "threads": 1,
    }
