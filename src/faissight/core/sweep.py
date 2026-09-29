"""Recall@k and latency across nprobe / efSearch values, for picking a search setting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

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

    def __len__(self) -> int:
        return len(self.vectors)


def given_queries(li: LoadedIndex, vectors: npt.ArrayLike) -> QuerySet:
    q = np.ascontiguousarray(vectors, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != li.d or len(q) == 0:
        raise ValueError(f"Queries must have shape (n, {li.d}) with n >= 1, got {q.shape}.")
    return QuerySet(q, None, "given")


def sample_queries(source: VectorSource, n: int = DEFAULT_N_QUERIES, seed: int = 0) -> QuerySet:
    """``n`` stored vectors as queries (seeded); each excludes its own id from results."""
    if n < 1:
        raise ValueError(f"n_queries must be >= 1, got {n}.")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(source), size=min(n, len(source)), replace=False))
    return QuerySet(source.vectors[rows], source.ids[rows].astype(np.int64), "sampled")


# --- sweep -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    value: int
    recall: float
    """Mean recall@k over the query set."""
    latency_mean_ms: float
    latency_p95_ms: float
    """Per-query latency, single-threaded."""


@dataclass(frozen=True)
class SweepResult:
    param: SweepParam
    k: int
    n_queries: int
    query_origin: Literal["given", "sampled"]
    truth_reconstructed: bool
    points: list[SweepPoint]

    def recommend(self, target_recall: float) -> SweepPoint | None:
        """Smallest parameter value whose mean recall meets ``target_recall``.

        Picks by parameter value, not measured latency, because latency is noisy and
        cost grows with the parameter.
        """
        ok = [p for p in self.points if p.recall >= target_recall - 1e-9]
        return min(ok, key=lambda p: p.value) if ok else None

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
) -> SweepResult:
    """Measure recall@k and per-query latency at each parameter value.

    Queries run one at a time on a single FAISS thread (restored afterwards), after an
    untimed warm-up pass, so latencies are stable and comparable across values.
    """
    faiss = import_faiss()
    param = SweepParam(param) if param is not None else param_for(li)
    _check_param(li, param)
    vals = sorted({int(v) for v in (values if values is not None else default_values(li, param))})
    if not vals:
        raise ValueError("Give at least one value to sweep.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if truth.shape != (len(queries), k):
        raise ValueError(f"truth must have shape ({len(queries)}, {k}), got {truth.shape}.")
    report = progress or (lambda frac, msg: None)

    def params_for(v: int) -> object:
        if param is SweepParam.NPROBE:
            return resolve_search_params(li, nprobe=v)[0]
        return resolve_search_params(li, ef_search=v)[0]

    all_params = [params_for(v) for v in vals]  # validates every value before timing
    excl = queries.exclude_ids
    k_search = min(k + (excl is not None), max(li.ntotal, 1))
    n = len(queries)
    points: list[SweepPoint] = []

    with _TIMING_LOCK:
        prev_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            report(0.0, "Warming up")
            li.index.search(queries.vectors, k_search, params=all_params[0])
            for step, (v, sp) in enumerate(zip(vals, all_params, strict=True)):
                report(step / len(vals), f"{param.value}={v}")
                latencies = np.empty(n, dtype=np.float64)
                recalls = np.empty(n, dtype=np.float64)
                for i in range(n):
                    q = queries.vectors[i : i + 1]
                    t0 = time.perf_counter()
                    _, found = li.index.search(q, k_search, params=sp)
                    latencies[i] = time.perf_counter() - t0
                    row = found[0]
                    if excl is not None:
                        row = row[row != excl[i]]
                    true_row = truth[i][truth[i] >= 0]
                    recalls[i] = np.isin(true_row, row[:k]).mean() if len(true_row) else 1.0
                points.append(
                    SweepPoint(
                        value=v,
                        recall=float(recalls.mean()),
                        latency_mean_ms=float(latencies.mean() * 1000),
                        latency_p95_ms=float(np.percentile(latencies, 95) * 1000),
                    )
                )
        finally:
            faiss.omp_set_num_threads(prev_threads)
    report(1.0, "Done")
    return SweepResult(param, k, n, queries.origin, truth_reconstructed, points)
