"""2-D/3-D projections of stored vectors, centroids and queries, with sampling and a disk cache.

Everything is projected in the index's *core* space (after any PreTransform), because that
is where IVF centroids live and where the index actually compares vectors.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.ivf import Assignments, centroids
from faissight.core.types import LoadedIndex
from faissight.core.vectors import VectorSource

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
ProgressFn = Callable[[float, str], None]

DEFAULT_MAX_POINTS = 50_000
DEFAULT_SEED = 42
UMAP_NEIGHBORS = 15
UMAP_INSTALL_HINT = 'Install UMAP with `pip install "faissight[umap]"`, or use method="pca".'
_CACHE_VERSION = 1


class ProjectionMethod(str, Enum):
    PCA = "pca"
    UMAP = "umap"


class ProjectionUnavailableError(ImportError):
    """The requested projection method needs an optional dependency."""

    def __init__(self) -> None:
        super().__init__(f"UMAP is not installed. {UMAP_INSTALL_HINT}")
        self.hint = UMAP_INSTALL_HINT


# --- PCA ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PcaModel:
    """A fitted PCA: ``transform(x) = (x - mean) @ components.T``."""

    mean: FloatArray
    components: FloatArray
    """Shape ``(dims, d)``, orthonormal rows, largest variance first."""
    explained_variance_ratio: FloatArray

    def transform(self, x: npt.ArrayLike) -> FloatArray:
        xa = np.asarray(x, dtype=np.float32)
        return ((xa - self.mean) @ self.components.T).astype(np.float32)


def fit_pca(x: npt.ArrayLike, dims: int) -> PcaModel:
    """PCA via eigendecomposition of the covariance matrix (numpy only).

    ``O(n d^2)`` like an SVD but several times faster for ``n >> d``. Component signs are
    fixed (largest-magnitude loading positive) so results are deterministic.
    """
    xa = np.asarray(x, dtype=np.float64)
    n, d = xa.shape
    if not 1 <= dims <= d:
        raise ValueError(f"dims must be between 1 and d={d}, got {dims}.")
    mean = xa.mean(axis=0)
    xc = xa - mean
    cov = xc.T @ xc / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:dims]
    comps = eigvecs[:, order].T
    signs = np.sign(comps[np.arange(dims), np.abs(comps).argmax(axis=1)])
    comps *= np.where(signs == 0, 1, signs)[:, None]
    total = eigvals.clip(min=0).sum()
    ratio = eigvals[order].clip(min=0) / total if total > 0 else np.zeros(dims)
    return PcaModel(mean.astype(np.float32), comps.astype(np.float32), ratio.astype(np.float32))


# --- sampling ----------------------------------------------------------------------------


def stratified_sample(
    n: int, max_points: int, groups: npt.ArrayLike | None = None, seed: int = DEFAULT_SEED
) -> IntArray:
    """Deterministically choose at most ``max_points`` of ``n`` rows, sorted.

    With ``groups`` (e.g. IVF list per row), every non-empty group keeps at least one row
    and the rest is allocated proportionally, so small lists still show up.
    """
    if max_points < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points}.")
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if groups is None:
        return np.sort(rng.choice(n, size=max_points, replace=False)).astype(np.int64)

    g = np.asarray(groups, dtype=np.int64)
    labels, inverse, sizes = np.unique(g, return_inverse=True, return_counts=True)
    if len(labels) >= max_points:
        # More groups than points: one row from each of max_points random groups.
        chosen_groups = rng.choice(len(labels), size=max_points, replace=False)
        quota = np.zeros(len(labels), dtype=np.int64)
        quota[chosen_groups] = 1
    else:
        # One per group, then split the remainder proportionally (largest remainder).
        quota = np.ones(len(labels), dtype=np.int64)
        remaining = max_points - len(labels)
        extra = (sizes - 1) * remaining / max(int((sizes - 1).sum()), 1)
        quota += np.floor(extra).astype(np.int64)
        short = max_points - int(quota.sum())
        if short > 0:
            frac_order = np.argsort(-(extra - np.floor(extra)), kind="stable")
            room = sizes - quota
            for gi in frac_order:
                if short == 0:
                    break
                if room[gi] > 0:
                    quota[gi] += 1
                    short -= 1
        np.minimum(quota, sizes, out=quota)

    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    picks = [
        rng.choice(order[s : s + size], size=q, replace=False)
        for s, size, q in zip(starts, sizes, quota, strict=True)
        if q > 0
    ]
    return np.sort(np.concatenate(picks)).astype(np.int64)


# --- projection --------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """Low-dimensional coordinates for a (sampled) set of stored vectors."""

    method: ProjectionMethod
    dims: int
    ids: IntArray
    """User-facing ids of the projected points."""
    coords: FloatArray
    """Shape ``(len(ids), dims)``."""
    list_nos: IntArray | None
    """IVF list of each point (IVF indexes only)."""
    centroid_coords: FloatArray | None
    """Shape ``(nlist, dims)`` (IVF indexes only)."""
    n_total: int
    """Number of stored vectors before sampling."""
    pca: PcaModel | None = None

    @property
    def sampled(self) -> bool:
        return len(self.ids) < self.n_total


def core_vectors(li: LoadedIndex, source: VectorSource, ids: npt.ArrayLike) -> FloatArray:
    """Vectors for ``ids`` in the index's core space (after any PreTransform)."""
    x = source.get(ids)
    return li.to_core_space(x) if li.vector_transforms else x


def _sample(
    li: LoadedIndex,
    source: VectorSource,
    max_points: int,
    seed: int,
    assignments: Assignments | None,
) -> tuple[IntArray | None, IntArray, IntArray, FloatArray]:
    list_nos_all = assignments.lookup(source.ids) if assignments is not None else None
    rows = stratified_sample(len(source), max_points, list_nos_all, seed)
    ids = source.ids[rows]
    return list_nos_all, rows, ids, core_vectors(li, source, ids)


def cached_projection(
    li: LoadedIndex,
    source: VectorSource,
    cache: ProjectionCache,
    *,
    method: ProjectionMethod | str,
    dims: int,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = DEFAULT_SEED,
    assignments: Assignments | None = None,
) -> Projection | None:
    """The projection ``compute_projection`` would load from ``cache``, without computing."""
    *_, x = _sample(li, source, max_points, seed, assignments)
    return cache.load(cache.key(ProjectionMethod(method), dims, max_points, seed, x))


def compute_projection(
    li: LoadedIndex,
    source: VectorSource,
    *,
    method: ProjectionMethod | str = ProjectionMethod.PCA,
    dims: int = 2,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = DEFAULT_SEED,
    assignments: Assignments | None = None,
    cache: ProjectionCache | None = None,
    progress: ProgressFn | None = None,
) -> Projection:
    """Project a sample of stored vectors (plus IVF centroids) to ``dims`` dimensions.

    ``assignments`` enables stratified sampling by IVF list and per-point list numbers.
    With ``cache``, results are read from / written to disk.
    """
    method = ProjectionMethod(method)
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}.")
    report = progress or (lambda frac, msg: None)

    report(0.0, "Sampling points")
    list_nos_all, rows, ids, x = _sample(li, source, max_points, seed, assignments)

    key = None
    if cache is not None:
        key = cache.key(method, dims, max_points, seed, x)
        hit = cache.load(key)
        if hit is not None:
            report(1.0, "Loaded from cache")
            return hit

    cents = centroids(li) if li.ivf is not None else None
    report(0.1, f"Fitting {method.value.upper()} on {len(ids):,} points")
    pca = None
    if method is ProjectionMethod.PCA:
        pca = fit_pca(x, dims)
        coords = pca.transform(x)
        cent_coords = pca.transform(cents) if cents is not None else None
    else:
        model = _fit_umap(x, dims, seed)
        coords = np.asarray(model.embedding_, dtype=np.float32)
        report(0.9, "Placing centroids")
        cent_coords = (
            np.asarray(model.transform(cents), dtype=np.float32) if cents is not None else None
        )

    proj = Projection(
        method=method,
        dims=dims,
        ids=ids,
        coords=coords,
        list_nos=list_nos_all[rows] if list_nos_all is not None else None,
        centroid_coords=cent_coords,
        n_total=len(source),
        pca=pca,
    )
    if cache is not None and key is not None:
        cache.save(key, proj)
    report(1.0, "Done")
    return proj


def _fit_umap(x: FloatArray, dims: int, seed: int) -> Any:
    try:
        import umap
    except ImportError as e:
        raise ProjectionUnavailableError() from e
    n_neighbors = min(UMAP_NEIGHBORS, max(2, len(x) - 1))
    reducer = umap.UMAP(n_components=dims, n_neighbors=n_neighbors, min_dist=0.1, random_state=seed)
    with warnings.catch_warnings():
        # "n_jobs overridden by random_state" and "graph not fully connected" are expected.
        warnings.simplefilter("ignore", UserWarning)
        return reducer.fit(x)


def place_points(
    proj: Projection, x_core: npt.ArrayLike, reference_core: npt.ArrayLike | None, k: int = 15
) -> FloatArray:
    """Coordinates for new vectors (e.g. queries) in core space.

    PCA is linear, so it's exact. For UMAP, ``umap.transform`` takes seconds per call, so a
    query is placed at the inverse-distance-weighted mean of the coordinates of its ``k``
    nearest projected points. ``reference_core`` are the core-space vectors of
    ``proj.ids``, in the same order.
    """
    x = np.atleast_2d(np.asarray(x_core, dtype=np.float32))
    if proj.pca is not None:
        return proj.pca.transform(x)
    if reference_core is None:
        raise ValueError("UMAP placement needs the core-space vectors of proj.ids.")
    faiss = import_faiss()
    ref = np.ascontiguousarray(reference_core, dtype=np.float32)
    k = min(k, len(ref))
    nn = faiss.IndexFlatL2(ref.shape[1])
    nn.add(ref)
    dist, idx = nn.search(x, k)
    w = 1.0 / (np.sqrt(np.maximum(dist, 0)) + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    placed: FloatArray = np.einsum("qk,qkd->qd", w, proj.coords[idx]).astype(np.float32)
    return placed


# --- disk cache --------------------------------------------------------------------------


def default_cache_root() -> Path:
    """``$FAISSIGHT_CACHE_DIR``, else ``$XDG_CACHE_HOME/faissight``, else ``~/.cache/faissight``."""
    if env := os.environ.get("FAISSIGHT_CACHE_DIR"):
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "faissight"


def index_fingerprint(li: LoadedIndex) -> str:
    """sha1 of the index bytes: the file on disk, or the serialized in-memory index."""
    h = hashlib.sha1()
    if li.path is not None:
        with li.path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        faiss = import_faiss()
        h.update(faiss.serialize_index(li.index).tobytes())
    return h.hexdigest()


class ProjectionCache:
    """Projections stored as ``.npz`` under ``<root>/<index_sha1>/``."""

    def __init__(self, index_sha1: str, root: Path | None = None) -> None:
        self.dir = (root or default_cache_root()) / index_sha1

    def key(
        self,
        method: ProjectionMethod,
        dims: int,
        max_points: int,
        seed: int,
        sampled_vectors: FloatArray,
    ) -> str:
        """Cache key; includes a hash of the sampled vectors (raw vs reconstructed differ)."""
        vec_hash = hashlib.sha1(np.ascontiguousarray(sampled_vectors).tobytes()).hexdigest()
        return f"proj-v{_CACHE_VERSION}-{method.value}-{dims}d-{max_points}-s{seed}-{vec_hash[:12]}"

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.npz"

    def load(self, key: str) -> Projection | None:
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                pca = None
                if z["has_pca"]:
                    pca = PcaModel(z["pca_mean"], z["pca_components"], z["pca_ratio"])
                return Projection(
                    method=ProjectionMethod(str(z["method"])),
                    dims=int(z["dims"]),
                    ids=z["ids"],
                    coords=z["coords"],
                    list_nos=z["list_nos"] if z["has_lists"] else None,
                    centroid_coords=z["centroid_coords"] if z["has_centroids"] else None,
                    n_total=int(z["n_total"]),
                    pca=pca,
                )
        except (OSError, KeyError, ValueError):
            return None  # corrupt or stale entry: recompute

    def save(self, key: str, proj: Projection) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        empty = np.empty(0, dtype=np.float32)
        arrays: dict[str, Any] = {
            "method": np.array(proj.method.value),
            "dims": np.array(proj.dims),
            "ids": proj.ids,
            "coords": proj.coords,
            "n_total": np.array(proj.n_total),
            "has_lists": np.array(proj.list_nos is not None),
            "list_nos": proj.list_nos if proj.list_nos is not None else empty,
            "has_centroids": np.array(proj.centroid_coords is not None),
            "centroid_coords": proj.centroid_coords if proj.centroid_coords is not None else empty,
            "has_pca": np.array(proj.pca is not None),
            "pca_mean": proj.pca.mean if proj.pca else empty,
            "pca_components": proj.pca.components if proj.pca else empty,
            "pca_ratio": proj.pca.explained_variance_ratio if proj.pca else empty,
        }
        # Write then rename, so a concurrent reader never sees a half-written file.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, self.path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
