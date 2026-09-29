"""Compare two indexes against the same raw vectors and query set."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from faissight.core._faiss import import_faiss
from faissight.core.search import GroundTruth, recall_at_k, resolve_search_params
from faissight.core.sweep import (
    _TIMING_LOCK,
    QuerySet,
    benchmark_environment,
    ground_truth_ids,
    query_fingerprint,
)
from faissight.core.types import LoadedIndex
from faissight.core.vectors import VectorSource, validate_ids


@dataclass(frozen=True)
class IndexMeasurement:
    """Recall and repeated single-query latency; size is serialized bytes, not process RAM."""

    kind: str
    params: dict[str, int]
    serialized_bytes: int
    recall: float
    latency_mean_ms: float
    latency_p95_ms: float


@dataclass(frozen=True)
class QueryDifference:
    """Neighbour membership differences for one query; lists retain result rank order."""

    query_no: int
    left_recall: float
    right_recall: float
    left_only: list[int]
    right_only: list[int]
    overlap: int


@dataclass(frozen=True)
class ComparisonResult:
    """A paired experiment with explicit provenance and per-query changes."""

    metric: str
    k: int
    n_queries: int
    query_origin: str
    query_sha256: str
    repeats: int
    seed: int
    environment: dict[str, str | int]
    left: IndexMeasurement
    right: IndexMeasurement
    differences: list[QueryDifference]
    query_seed: int | None = None


def compare_indexes(
    left: LoadedIndex,
    right: LoadedIndex,
    source: VectorSource,
    queries: QuerySet,
    *,
    k: int = 10,
    repeats: int = 3,
    seed: int = 0,
    left_nprobe: int | None = None,
    right_nprobe: int | None = None,
    left_ef_search: int | None = None,
    right_ef_search: int | None = None,
) -> ComparisonResult:
    """Evaluate compatible indexes using shared raw ground truth and paired query order.

    Raw vectors and ids must describe both indexes. Timing alternates which index runs
    first each repetition, uses one FAISS thread, and never mutates search parameters.
    """
    if not left.is_supported or not right.is_supported:
        raise ValueError("Comparison requires supported indexes.")
    if left.d != right.d or left.metric != right.metric:
        raise ValueError("Indexes must have the same input dimension and distance metric.")
    if source.reconstructed:
        raise ValueError("Comparison requires raw vectors for independent ground truth.")
    if source.vectors.shape != (left.ntotal, left.d):
        raise ValueError("Raw vectors must match the index shape.")
    validate_ids(left, source.ids)
    validate_ids(right, source.ids)
    if k < 1 or repeats < 1 or seed < 0 or len(queries) == 0:
        raise ValueError("k, repeats and query count must be positive; seed must be non-negative.")
    if queries.vectors.ndim != 2 or queries.vectors.shape[1] != left.d:
        raise ValueError("Query dimension must match the indexes.")
    if not np.isfinite(queries.vectors).all():
        raise ValueError("Queries must be finite.")
    if queries.exclude_ids is not None and queries.exclude_ids.shape != (len(queries),):
        raise ValueError("Query exclusions must have one id per query.")
    faiss = import_faiss()
    truth = ground_truth_ids(GroundTruth(source, left.metric), queries, k)
    settings = [
        resolve_search_params(left, left_nprobe, left_ef_search),
        resolve_search_params(right, right_nprobe, right_ef_search),
    ]
    n = len(queries)
    count = min(k + (queries.exclude_ids is not None), max(left.ntotal, 1))
    latencies = [np.empty(n * repeats), np.empty(n * repeats)]
    results: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(2)]
    indexes = [left, right]
    with _TIMING_LOCK:
        previous = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            for li, (params, _) in zip(indexes, settings, strict=True):
                li.index.search(queries.vectors, count, params=params)
            rng = np.random.default_rng(seed)
            for repeat in range(repeats):
                for j, i in enumerate(rng.permutation(n)):
                    for side in (0, 1) if repeat % 2 == 0 else (1, 0):
                        q = queries.vectors[i : i + 1]
                        start = time.perf_counter()
                        _, found = indexes[side].index.search(q, count, params=settings[side][0])
                        latencies[side][repeat * n + j] = (time.perf_counter() - start) * 1000
                        if repeat == 0:
                            excluded = (
                                int(queries.exclude_ids[i])
                                if queries.exclude_ids is not None
                                else -1
                            )
                            results[side][i] = [
                                int(x) for x in found[0] if x >= 0 and x != excluded
                            ][:k]
        finally:
            faiss.omp_set_num_threads(previous)
    recalls = [[recall_at_k(row, truth[i]) for i, row in enumerate(rows)] for rows in results]
    measurements = [
        IndexMeasurement(
            li.kind.value,
            settings[side][1],
            int(faiss.serialize_index(li.index).nbytes),
            float(np.mean(recalls[side])),
            float(latencies[side].mean()),
            float(np.percentile(latencies[side], 95)),
        )
        for side, li in enumerate(indexes)
    ]
    differences = []
    for i, (a, b) in enumerate(zip(*results, strict=True)):
        a_set, b_set = set(a), set(b)
        differences.append(
            QueryDifference(
                i,
                recalls[0][i],
                recalls[1][i],
                [x for x in a if x not in b_set],
                [x for x in b if x not in a_set],
                len(a_set & b_set),
            )
        )
    return ComparisonResult(
        left.metric.value,
        k,
        n,
        queries.origin,
        query_fingerprint(queries),
        repeats,
        seed,
        benchmark_environment(),
        measurements[0],
        measurements[1],
        differences,
        query_seed=queries.seed,
    )
