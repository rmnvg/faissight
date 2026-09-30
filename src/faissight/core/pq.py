"""Quantization error: how far the vectors an index stores are from the raw vectors.

Needs the raw vectors. The stored side is decoded with ``reconstruct_all`` (in input space),
so for PreTransform indexes the error also includes the transform's loss (e.g. PCA).

For IVF-PQ/SQ the index's asymmetric distance between an exact query and a code equals
the distance to that code's reconstruction, so "approximate distance" below is what the
index actually ranks by.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.ivf import Assignments
from faissight.core.types import LoadedIndex, Metric
from faissight.core.vectors import VectorSource

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


class RawVectorsRequiredError(ValueError):
    """Quantization error can't be measured without the raw vectors."""

    def __init__(self) -> None:
        super().__init__("Provide --vectors to measure quantization error.")
        self.hint = "Pass the raw vectors that were added to the index (vectors=... or --vectors)."


def reconstruction_errors(raw: VectorSource, stored: VectorSource) -> tuple[F64, F64]:
    """Per-vector squared error ``||x - x_hat||^2`` and relative error ``/ ||x||^2``.

    Both sources are sorted by id and must hold the same ids.
    """
    if not np.array_equal(raw.ids, stored.ids):
        raise ValueError("Raw and stored vectors don't cover the same ids.")
    x = raw.vectors.astype(np.float64)
    diff = x - stored.vectors.astype(np.float64)
    err = np.einsum("ij,ij->i", diff, diff)
    norm = np.einsum("ij,ij->i", x, x)
    rel = np.divide(err, norm, out=np.zeros_like(err), where=norm > 0)
    return err, rel


@dataclass(frozen=True)
class Distortion:
    """True vs approximate query-to-vector distances for sampled pairs."""

    true: FloatArray
    approx: FloatArray
    near: npt.NDArray[np.bool_]
    """True where the target is one of the query's exact nearest neighbours."""
    correlation: float
    """Pearson correlation of true vs approximate distances over all sampled pairs."""
    near_correlation: float
    """The same over near pairs only: what decides nearest-neighbour ranking. Random far
    pairs span a wide range and inflate the overall correlation."""


def distance_distortion(
    raw: VectorSource,
    stored: VectorSource,
    metric: Metric,
    *,
    n_queries: int = 100,
    n_near: int = 20,
    n_random: int = 20,
    seed: int = 0,
) -> Distortion:
    """Sample queries (stored vectors), each paired with its nearest and some random vectors.

    For L2 the distances are squared L2; for IP they are inner products.
    """
    faiss = import_faiss()
    rng = np.random.default_rng(seed)
    n = len(raw)
    if n == 0:
        raise ValueError("Distance distortion needs at least one stored vector.")
    q_rows = rng.choice(n, size=min(n_queries, n), replace=False)
    queries = raw.vectors[q_rows]
    flat = faiss.IndexFlat(
        raw.vectors.shape[1], faiss.METRIC_INNER_PRODUCT if metric is Metric.IP else faiss.METRIC_L2
    )
    flat.add(raw.vectors)
    _, near_rows = flat.search(queries, min(n_near + 1, n))
    pairs_q, pairs_t, near = [], [], []
    for qi, found in enumerate(near_rows):
        nn = found[(found >= 0) & (found != q_rows[qi])][:n_near]
        rand = rng.choice(n, size=min(n_random, n), replace=False)
        rand = rand[rand != q_rows[qi]]
        targets = np.concatenate([nn, rand])
        pairs_q.append(np.full(len(targets), qi))
        pairs_t.append(targets)
        near.append(np.r_[np.ones(len(nn), bool), np.zeros(len(rand), bool)])
    q_idx = np.concatenate(pairs_q)
    ti = np.concatenate(pairs_t)
    q = queries[q_idx].astype(np.float64)
    x = raw.vectors[ti].astype(np.float64)
    xh = stored.vectors[ti].astype(np.float64)
    if metric is Metric.IP:
        true, approx = np.einsum("ij,ij->i", q, x), np.einsum("ij,ij->i", q, xh)
    else:
        true = np.einsum("ij,ij->i", q - x, q - x)
        approx = np.einsum("ij,ij->i", q - xh, q - xh)
    near_mask = np.concatenate(near)
    return Distortion(
        true.astype(np.float32),
        approx.astype(np.float32),
        near_mask,
        _corr(true, approx),
        _corr(true[near_mask], approx[near_mask]),
    )


def _corr(a: F64, b: F64) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class QuantizationReport:
    ids: IntArray
    errors: F64
    """Squared reconstruction error per vector, aligned with ``ids`` (sorted)."""
    relative: F64
    list_nos: IntArray | None
    code_size: int | None
    """Bytes per stored code, when the index reports it."""
    raw_bytes: int
    """Bytes per raw float32 vector."""
    distortion: Distortion

    @property
    def mean(self) -> float:
        return float(self.errors.mean()) if len(self.errors) else 0.0

    def quantiles(self) -> dict[str, float]:
        q = np.quantile(self.errors, [0.5, 0.95, 1.0]) if len(self.errors) else np.zeros(3)
        return {"median": float(q[0]), "p95": float(q[1]), "max": float(q[2])}

    def histogram(self, bins: int = 40) -> tuple[F64, IntArray]:
        """``(edges, counts)`` of the squared error."""
        hi = float(self.errors.max()) if len(self.errors) else 0.0
        counts, edges = np.histogram(self.errors, bins=bins, range=(0.0, hi if hi > 0 else 1.0))
        return edges, counts.astype(np.int64)

    def per_list(self) -> list[tuple[int, int, float]]:
        """``(list_no, size, mean error)`` for every non-empty IVF list."""
        if self.list_nos is None:
            return []
        valid = self.list_nos >= 0
        lists = self.list_nos[valid]
        sums = np.bincount(lists, weights=self.errors[valid])
        sizes = np.bincount(lists)
        return [(int(i), int(sizes[i]), float(sums[i] / sizes[i])) for i in np.flatnonzero(sizes)]

    def worst(self, n: int = 50) -> list[tuple[int, float, float]]:
        """``(id, error, relative error)`` of the ``n`` worst-reconstructed vectors."""
        order = np.argsort(-self.errors, kind="stable")[:n]
        return [(int(self.ids[i]), float(self.errors[i]), float(self.relative[i])) for i in order]


def code_size(li: LoadedIndex) -> int | None:
    """Bytes per stored code, if the core index exposes it."""
    size = getattr(li.core, "code_size", None)
    if size is None and li.kind.is_hnsw:
        faiss = import_faiss()
        size = getattr(faiss.downcast_index(li.core.storage), "code_size", None)
    return int(size) if size is not None else None


def analyze(
    li: LoadedIndex,
    raw: VectorSource | None,
    stored: VectorSource,
    assignments: Assignments | None = None,
    *,
    seed: int = 0,
) -> QuantizationReport:
    """Everything the quantization view shows. ``raw`` must be the user's raw vectors."""
    if raw is None or raw.reconstructed:
        raise RawVectorsRequiredError()
    err, rel = reconstruction_errors(raw, stored)
    return QuantizationReport(
        ids=raw.ids,
        errors=err,
        relative=rel,
        list_nos=assignments.lookup(raw.ids) if assignments is not None else None,
        code_size=code_size(li),
        raw_bytes=4 * li.d,
        distortion=distance_distortion(raw, stored, li.metric, seed=seed),
    )
