"""Search with per-call parameters, exact ground truth, recall and IVF miss explanations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.types import LoadedIndex, Metric
from faissight.core.vectors import VectorSource

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class SearchResult:
    """Top-k results for one query, best first. ``ids`` are user-facing; ``-1`` = empty slot."""

    ids: IntArray
    distances: FloatArray
    metric: Metric
    latency_ms: float = 0.0
    params: dict[str, int] = field(default_factory=dict)
    """Search parameters actually used (e.g. ``{"nprobe": 8}``)."""

    @property
    def valid_ids(self) -> IntArray:
        return self.ids[self.ids >= 0]


def _as_query(query: npt.ArrayLike, d: int) -> FloatArray:
    q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != d:
        raise ValueError(f"Query has dimension {q.shape[1]}, index expects {d}.")
    return q


def resolve_search_params(
    li: LoadedIndex, nprobe: int | None = None, ef_search: int | None = None
) -> tuple[Any, dict[str, int]]:
    """Build a FAISS ``SearchParameters`` object without mutating the shared index.

    Unset values fall back to what the index currently has. Returns ``(params, used)``
    where ``used`` records the effective values for display.
    """
    faiss = import_faiss()
    used: dict[str, int] = {}
    params: Any = None
    if li.kind.is_ivf:
        if ef_search is not None:
            raise ValueError("efSearch only applies to HNSW indexes.")
        nlist = int(li.ivf.nlist)
        nprobe = int(li.ivf.nprobe) if nprobe is None else int(nprobe)
        if not 1 <= nprobe <= nlist:
            raise ValueError(f"nprobe must be between 1 and nlist={nlist}, got {nprobe}.")
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        used["nprobe"] = nprobe
    elif li.kind.is_hnsw:
        if nprobe is not None:
            raise ValueError("nprobe only applies to IVF indexes.")
        ef_search = int(li.core.hnsw.efSearch) if ef_search is None else int(ef_search)
        if ef_search < 1:
            raise ValueError(f"efSearch must be >= 1, got {ef_search}.")
        params = faiss.SearchParametersHNSW(efSearch=ef_search)
        used["efSearch"] = ef_search
    elif nprobe is not None or ef_search is not None:
        raise ValueError(f"{li.kind.value} indexes take no search parameters.")

    if params is not None and li.has_refine:
        # IndexRefine rejects plain params, and its own params default k_factor to 1,
        # so pass the index's k_factor through explicitly to keep results unchanged.
        params = faiss.IndexRefineSearchParameters(
            k_factor=li.refine_k_factor, base_index_params=params
        )
    return params, used


def search(
    li: LoadedIndex,
    query: npt.ArrayLike,
    k: int,
    *,
    nprobe: int | None = None,
    ef_search: int | None = None,
    exclude_id: int | None = None,
) -> SearchResult:
    """Search the index for one query vector (input space, dimension ``li.d``).

    ``nprobe``/``ef_search`` apply to this call only. ``exclude_id`` drops that id from the
    results (used when the query is itself a stored vector) while still returning ``k``.
    """
    if not li.is_supported:
        raise ValueError(f"Cannot search: {li.unsupported_reason}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    q = _as_query(query, li.d)
    params, used = resolve_search_params(li, nprobe, ef_search)
    k_search = min(k + (exclude_id is not None), max(li.ntotal, 1))

    t0 = time.perf_counter()
    distances, ids = li.index.search(q, k_search, params=params)
    latency_ms = (time.perf_counter() - t0) * 1000

    ids, distances = _drop_excluded(ids[0], distances[0], exclude_id, k)
    return SearchResult(ids, distances, li.metric, latency_ms, used)


def _drop_excluded(
    ids: IntArray, distances: FloatArray, exclude_id: int | None, k: int
) -> tuple[IntArray, FloatArray]:
    if exclude_id is not None:
        keep = ids != exclude_id
        ids, distances = ids[keep], distances[keep]
    return ids[:k].astype(np.int64), distances[:k].astype(np.float32)


class GroundTruth:
    """Exact k-NN over a :class:`VectorSource`, using the same metric as the index.

    Builds a brute-force ``IndexFlat`` (costs ``n * d * 4`` bytes on top of the source).
    """

    def __init__(self, source: VectorSource, metric: Metric) -> None:
        faiss = import_faiss()
        if metric is Metric.OTHER:
            raise ValueError("Ground truth needs an L2 or inner-product metric.")
        self.source = source
        self.metric = metric
        d = source.vectors.shape[1]
        faiss_metric = faiss.METRIC_L2 if metric is Metric.L2 else faiss.METRIC_INNER_PRODUCT
        self._index = faiss.IndexFlat(d, faiss_metric)
        self._index.add(source.vectors)

    @property
    def reconstructed(self) -> bool:
        """True when computed on vectors decoded from the index, not the raw vectors."""
        return self.source.reconstructed

    def search(self, query: npt.ArrayLike, k: int, exclude_id: int | None = None) -> SearchResult:
        """Exact top-k (user-facing ids), optionally excluding one id."""
        q = _as_query(query, self._index.d)
        k_search = min(k + (exclude_id is not None), max(len(self.source), 1))
        t0 = time.perf_counter()
        distances, rows = self._index.search(q, k_search)
        latency_ms = (time.perf_counter() - t0) * 1000
        rows = rows[0]
        ids = np.where(rows >= 0, self.source.ids[np.maximum(rows, 0)], -1)
        ids, dist = _drop_excluded(ids, distances[0], exclude_id, k)
        return SearchResult(ids, dist, self.metric, latency_ms)


def recall_at_k(found: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """Fraction of the true neighbours (ignoring ``-1`` slots) present in ``found``.

    Returns 1.0 when there are no true neighbours to find.
    """
    truth_ids = np.asarray(truth, dtype=np.int64)
    truth_ids = truth_ids[truth_ids >= 0]
    if len(truth_ids) == 0:
        return 1.0
    return float(np.isin(truth_ids, np.asarray(found, dtype=np.int64)).mean())
