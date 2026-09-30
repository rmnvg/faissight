"""HNSW graph extraction: node levels, neighbours per level, entry point and degree stats.

Node ids here are internal HNSW offsets (0..ntotal-1); map them with
``LoadedIndex.user_ids`` before showing them to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.types import LoadedIndex

IntArray = npt.NDArray[np.int64]


# Link-slot cells gathered per batch when scanning a whole level (x 8 bytes per index).
_BATCH_CELLS = 1 << 21


class NotAnHnswIndexError(ValueError):
    """The loaded index has no HNSW core."""


@dataclass(frozen=True)
class HnswGraph:
    """The link structure of an ``IndexHNSW``, copied out of FAISS."""

    node_levels: IntArray
    """Top level of each node (0-based). FAISS stores ``level + 1``; this is converted."""
    offsets: IntArray
    """Node ``i``'s links live in ``neighbors[offsets[i]:offsets[i + 1]]``."""
    neighbors: npt.NDArray[np.int32]
    cum_per_level: IntArray
    """Within a node's slice, level ``l`` spans ``[cum[l], cum[l + 1])``; ``-1`` pads."""
    entry_point: int
    max_level: int
    _owner: Any = field(default=None, repr=False, compare=False)
    """The FAISS index whose memory ``offsets``/``neighbors`` borrow, kept alive with them."""

    @property
    def ntotal(self) -> int:
        return len(self.node_levels)

    def neighbours(self, node: int, level: int) -> IntArray:
        """Valid links of ``node`` at ``level`` (empty if the node isn't on that level)."""
        if level > self.node_levels[node]:
            return np.empty(0, dtype=np.int64)
        base = int(self.offsets[node])
        row = self.neighbors[
            base + int(self.cum_per_level[level]) : base + int(self.cum_per_level[level + 1])
        ]
        return row[row >= 0].astype(np.int64)

    def nodes_at_level(self, level: int) -> IntArray:
        """Nodes present on ``level`` (every node is on level 0)."""
        return np.flatnonzero(self.node_levels >= level).astype(np.int64)

    def degrees(self, level: int) -> IntArray:
        """Out-degree of every node present on ``level``.

        Works through the nodes in batches, so the temporary (nodes x link slots) index
        stays bounded (about 16 MB) however large the graph is.
        """
        nodes = self.nodes_at_level(level)
        lo, hi = int(self.cum_per_level[level]), int(self.cum_per_level[level + 1])
        slots = np.arange(lo, hi)
        deg = np.empty(len(nodes), dtype=np.int64)
        step = max(1, _BATCH_CELLS // max(hi - lo, 1))
        for start in range(0, len(nodes), step):
            idx = self.offsets[nodes[start : start + step]][:, None] + slots[None, :]
            deg[start : start + step] = (self.neighbors[idx] >= 0).sum(axis=1)
        return deg

    def edges(self, level: int, nodes: npt.ArrayLike | None = None) -> tuple[IntArray, IntArray]:
        """Directed links ``(src, dst)`` at ``level``; restricted to ``nodes`` if given."""
        src_nodes = self.nodes_at_level(level) if nodes is None else np.asarray(nodes, np.int64)
        src_nodes = src_nodes[self.node_levels[src_nodes] >= level]
        lo, hi = int(self.cum_per_level[level]), int(self.cum_per_level[level + 1])
        idx = self.offsets[src_nodes][:, None] + np.arange(lo, hi)[None, :]
        dst = self.neighbors[idx].astype(np.int64)
        src = np.repeat(src_nodes, hi - lo).reshape(dst.shape)
        keep = dst >= 0
        if nodes is not None:
            keep &= np.isin(dst, src_nodes)
        return src[keep], dst[keep]


def extract_graph(li: LoadedIndex) -> HnswGraph:
    """Copy the HNSW link arrays out of the index."""
    if not li.kind.is_hnsw:
        raise NotAnHnswIndexError(f"{' → '.join(li.class_chain)} is not an HNSW index.")
    faiss = import_faiss()
    h = li.core.hnsw
    return HnswGraph(
        node_levels=faiss.vector_to_array(h.levels).astype(np.int64) - 1,
        # The link arrays are as large as the graph: borrow them read-only, don't copy.
        offsets=_borrow(faiss, h.offsets, np.int64),
        neighbors=_borrow(faiss, h.neighbors, np.int32),
        cum_per_level=faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64),
        entry_point=int(h.entry_point),
        max_level=int(h.max_level),
        _owner=li.index,
    )


def _borrow(faiss: Any, vec: Any, dtype: type[np.generic]) -> npt.NDArray[Any]:
    """A read-only numpy view of a FAISS ``std::vector`` (same item size as ``dtype``)."""
    if vec.size() == 0:
        return np.empty(0, dtype=dtype)
    arr: npt.NDArray[Any] = faiss.rev_swig_ptr(vec.data(), vec.size()).view(dtype)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class LevelStats:
    level: int
    n_nodes: int
    max_links: int
    """Link slots per node on this level (2M on level 0, M above)."""
    degree_mean: float
    degree_min: int
    degree_max: int
    degree_hist: list[int]
    """``degree_hist[d]`` = nodes with out-degree ``d`` (0..max_links)."""


@dataclass(frozen=True)
class HnswStats:
    entry_point: int
    max_level: int
    levels: list[LevelStats]
    """Top level first."""


def graph_stats(g: HnswGraph) -> HnswStats:
    """Nodes and out-degree distribution per level."""
    out = []
    for level in range(g.max_level, -1, -1):
        deg = g.degrees(level)
        max_links = int(g.cum_per_level[level + 1] - g.cum_per_level[level])
        hist = np.bincount(deg, minlength=max_links + 1)[: max_links + 1]
        out.append(
            LevelStats(
                level=level,
                n_nodes=len(deg),
                max_links=max_links,
                degree_mean=float(deg.mean()) if len(deg) else 0.0,
                degree_min=int(deg.min()) if len(deg) else 0,
                degree_max=int(deg.max()) if len(deg) else 0,
                degree_hist=hist.astype(int).tolist(),
            )
        )
    return HnswStats(entry_point=g.entry_point, max_level=g.max_level, levels=out)


def neighbourhood(g: HnswGraph, seeds: npt.ArrayLike, level: int, limit: int) -> IntArray:
    """Up to ``limit`` nodes reached by breadth-first search from ``seeds`` on ``level``."""
    seen: dict[int, None] = {}
    frontier = [int(s) for s in np.asarray(seeds, dtype=np.int64) if g.node_levels[int(s)] >= level]
    for s in frontier:
        seen.setdefault(s)
    while frontier and len(seen) < limit:
        nxt = []
        for node in frontier:
            for nb in g.neighbours(node, level):
                if len(seen) >= limit:
                    break
                if int(nb) not in seen:
                    seen[int(nb)] = None
                    nxt.append(int(nb))
        frontier = nxt
    return np.fromiter(seen, dtype=np.int64, count=len(seen))[:limit]
