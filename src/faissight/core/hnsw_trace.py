"""A step-by-step, *reconstructed* trace of HNSW search.

FAISS doesn't expose its search internals, so this re-implements the algorithm over the
extracted graph and stored vectors:

- upper levels: greedy descent from the entry point (FAISS ``greedy_update_nearest``);
- level 0: FAISS's ``search_from_candidates`` with a bounded candidate queue of capacity
  ``efSearch``, a top-k heap over every evaluated node, and the relative-distance stopping
  rule.

Distance ties and FAISS's batched distance evaluation can still make the trace diverge in
rare cases; tests hold overlap with FAISS's own results at >= 95%.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.hnsw import HnswGraph
from faissight.core.types import LoadedIndex, Metric

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Visit:
    node: int
    distance: float
    """Search distance: squared L2, or negated inner product (lower is closer either way)."""
    accepted: bool
    """Upper levels: became the new nearest. Level 0: entered the candidate queue."""


@dataclass(frozen=True)
class TraceStep:
    """Scanning the links of one node."""

    level: int
    expanded: int
    expanded_distance: float
    visits: list[Visit]


@dataclass
class HnswTrace:
    entry_point: int
    max_level: int
    ef: int
    k: int
    steps: list[TraceStep] = field(default_factory=list)
    results: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """Internal node ids of the top-k, best first."""
    result_distances: FloatArray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    level_entries: dict[int, int] = field(default_factory=dict)
    """Node where the search entered each level."""

    def visited(self, level: int | None = None) -> set[int]:
        """Nodes whose distance was computed (optionally on one level), plus entry points."""
        out = {v.node for s in self.steps if level is None or s.level == level for v in s.visits}
        out.update(n for lv, n in self.level_entries.items() if level is None or lv == level)
        return out

    def expansions(self, level: int) -> list[int]:
        return [s.expanded for s in self.steps if s.level == level]


class NeighbourOutcome(str, Enum):
    FOUND = "FOUND"
    VISITED_NOT_KEPT = "VISITED_NOT_KEPT"
    """Its distance was computed on level 0 but it still missed the top-k. The top-k is kept
    over every evaluated node, so with exact (Flat) storage this only happens on ties; with
    SQ/PQ storage, approximate distances can rank it out."""
    NOT_REACHED = "NOT_REACHED"
    """Never evaluated: no expanded node on the visited frontier linked to it."""


def storage_vectors(li: LoadedIndex) -> FloatArray:
    """Core-space vectors of every HNSW node, by internal id (decoded for SQ/PQ storage)."""
    faiss = import_faiss()
    storage = faiss.downcast_index(li.core.storage)
    out: FloatArray = storage.reconstruct_n(0, storage.ntotal).astype(np.float32, copy=False)
    return out


class _MinimaxHeap:
    """faiss::HNSW::MinimaxHeap: a capacity-bounded max-heap of (distance, id).

    ``pop_min`` marks the minimum invalid in place instead of removing it, so popped
    entries still take capacity until pushed out, exactly as in FAISS.
    """

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.heap: list[list[Any]] = []  # [-distance, id]; id == -1 means popped
        self.nvalid = 0

    def push(self, node: int, d: float) -> bool:
        """Insert; returns False if rejected (full and not better than the worst)."""
        if len(self.heap) == self.cap:
            if d >= -self.heap[0][0]:
                return False
            if self.heap[0][1] != -1:
                self.nvalid -= 1
            heapq.heappop(self.heap)
        heapq.heappush(self.heap, [-d, node])
        self.nvalid += 1
        return True

    def size(self) -> int:
        return self.nvalid

    def pop_min(self) -> tuple[int, float]:
        best = None
        for e in self.heap:
            if e[1] != -1 and (best is None or -e[0] < -best[0]):
                best = e
        assert best is not None
        node, d = int(best[1]), float(-best[0])
        best[1] = -1  # same key, so the heap invariant still holds
        self.nvalid -= 1
        return node, d

    def count_below(self, thresh: float) -> int:
        return sum(1 for e in self.heap if e[1] != -1 and -e[0] < thresh)


class _Distances:
    def __init__(self, vectors: FloatArray, query: FloatArray, metric: Metric) -> None:
        self.x = vectors
        self.q = query.astype(np.float32).ravel()
        self.ip = metric is Metric.IP

    def __call__(self, nodes: npt.ArrayLike) -> npt.NDArray[np.float64]:
        v = self.x[np.asarray(nodes, dtype=np.int64)]
        if self.ip:
            out: npt.NDArray[np.float64] = -(v @ self.q).astype(np.float64)
            return out
        diff = v - self.q
        sq: npt.NDArray[np.float64] = np.einsum("ij,ij->i", diff, diff).astype(np.float64)
        return sq


def trace_search(
    g: HnswGraph,
    vectors: FloatArray,
    query_core: npt.ArrayLike,
    k: int,
    ef_search: int,
    metric: Metric = Metric.L2,
) -> HnswTrace:
    """Replay HNSW search for one core-space query and record every step."""
    if k < 1 or ef_search < 1:
        raise ValueError("k and ef_search must be >= 1.")
    if g.entry_point < 0 or len(vectors) == 0:
        raise ValueError("The HNSW graph is empty: there is no search to trace.")
    dist = _Distances(vectors, np.asarray(query_core, dtype=np.float32), metric)
    # faiss 1.15 sizes the level-0 candidate queue with efSearch itself, not max(efSearch, k)
    # as older sources suggest; measured: matching it gives 100% overlap at efSearch < k too.
    ef = ef_search
    trace = HnswTrace(entry_point=g.entry_point, max_level=g.max_level, ef=ef, k=k)

    nearest = g.entry_point
    d_nearest = float(dist([nearest])[0])
    for level in range(g.max_level, 0, -1):
        trace.level_entries[level] = nearest
        while True:
            prev, d_prev = nearest, d_nearest
            nbrs = g.neighbours(prev, level)
            visits = []
            if len(nbrs):
                ds = dist(nbrs)
                for nb, d in zip(nbrs.tolist(), ds.tolist(), strict=True):
                    better = d < d_nearest
                    if better:
                        nearest, d_nearest = nb, d
                    visits.append(Visit(nb, d, better))
            trace.steps.append(TraceStep(level, prev, d_prev, visits))
            if nearest == prev:
                break

    trace.level_entries[0] = nearest
    # Level 0 mirrors faiss::search_from_candidates: a bounded candidate queue of capacity
    # efSearch (popped entries keep occupying slots), a separate top-k heap over every evaluated
    # node, and the relative-distance stop (break once >= efSearch candidates beat d0).
    queue = _MinimaxHeap(ef)
    queue.push(nearest, d_nearest)
    visited = {nearest}
    topk: list[tuple[float, int]] = [(-d_nearest, nearest)]  # max-heap via negation
    while queue.size() > 0:
        c, d_c = queue.pop_min()
        if queue.count_below(d_c) >= ef_search:
            break
        fresh = [int(n) for n in g.neighbours(c, 0) if int(n) not in visited]
        visits = []
        if fresh:
            visited.update(fresh)
            for nb, d in zip(fresh, dist(fresh).tolist(), strict=True):
                if len(topk) < k:
                    heapq.heappush(topk, (-d, nb))
                elif d < -topk[0][0]:
                    heapq.heapreplace(topk, (-d, nb))
                visits.append(Visit(nb, d, queue.push(nb, d)))
        trace.steps.append(TraceStep(0, c, d_c, visits))

    best = sorted((-nd, n) for nd, n in topk)
    trace.results = np.array([n for _, n in best], dtype=np.int64)
    trace.result_distances = np.array([d for d, _ in best], dtype=np.float32)
    return trace


def neighbour_outcomes(trace: HnswTrace, truth_internal: npt.ArrayLike) -> list[NeighbourOutcome]:
    """What happened to each true neighbour (internal ids) during the traced search."""
    found = set(trace.results.tolist())
    seen0 = trace.visited(0)
    out = []
    for t in np.asarray(truth_internal, dtype=np.int64).tolist():
        if t in found:
            out.append(NeighbourOutcome.FOUND)
        elif t in seen0:
            out.append(NeighbourOutcome.VISITED_NOT_KEPT)
        else:
            out.append(NeighbourOutcome.NOT_REACHED)
    return out


def trace_for_index(
    li: LoadedIndex,
    g: HnswGraph,
    vectors: FloatArray,
    query: npt.ArrayLike,
    k: int,
    ef_search: int | None = None,
) -> HnswTrace:
    """Trace an input-space query: applies any PreTransform, defaults efSearch to the index's."""
    ef = int(li.core.hnsw.efSearch) if ef_search is None else int(ef_search)
    q = li.to_core_space(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    return trace_search(g, vectors, q, k, ef, li.metric)


def internal_ids(li: LoadedIndex, user_ids: npt.ArrayLike) -> IntArray:
    """Inverse of ``LoadedIndex.user_ids`` for HNSW (``-1`` for unknown ids)."""
    uid = np.asarray(user_ids, dtype=np.int64)
    if li.ids is None:
        return np.where((uid >= 0) & (uid < li.ntotal), uid, -1)
    order = np.argsort(li.ids)
    pos = np.searchsorted(li.ids, uid, sorter=order)
    pos = np.minimum(pos, len(order) - 1)
    hit = li.ids[order[pos]] == uid
    out: IntArray = np.where(hit, order[pos], -1).astype(np.int64)
    return out


def summarize(trace: HnswTrace) -> dict[str, Any]:
    """Counts for display: expansions and distance computations per level."""
    per_level: dict[int, dict[str, int]] = {}
    for s in trace.steps:
        d = per_level.setdefault(s.level, {"expansions": 0, "distance_computations": 0})
        d["expansions"] += 1
        d["distance_computations"] += len(s.visits)
    return {"ef": trace.ef, "k": trace.k, "levels": per_level}
