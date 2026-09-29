"""In-memory state for one inspected index: inputs, lazily computed caches, background jobs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from faissight.core import hnsw as hnsw_mod
from faissight.core import hnsw_trace, ivf
from faissight.core import pq as pq_mod
from faissight.core.embed import (
    Embedder,
    EmbedderUnavailableError,
    looks_normalized,
    sentence_transformer_embedder,
)
from faissight.core.jobs import Job, JobRunner
from faissight.core.loader import load_index
from faissight.core.metadata import Metadata, load_metadata
from faissight.core.projection import (
    DEFAULT_MAX_POINTS,
    Projection,
    ProjectionCache,
    ProjectionMethod,
    ProjectionUnavailableError,
    cached_projection,
    compute_projection,
    core_vectors,
    default_cache_root,
    index_fingerprint,
    place_points,
)
from faissight.core.search import GroundTruth, QueryReport, explain_query, resolve_query
from faissight.core.sweep import (
    DEFAULT_N_QUERIES,
    QuerySet,
    SweepParam,
    SweepResult,
    check_values,
    given_queries,
    ground_truth_ids,
    sample_queries,
    sweep,
)
from faissight.core.types import LoadedIndex, Metric
from faissight.core.vectors import (
    VectorMismatchError,
    VectorSource,
    from_arrays,
    reconstruct_all,
)

T = TypeVar("T")
ArrayInput = str | os.PathLike[str] | npt.ArrayLike


class InputError(ValueError):
    """An input doesn't fit the index; ``hint`` says how to fix it."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


def _load_array(value: ArrayInput, name: str) -> np.ndarray[Any, Any]:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_file():
            raise InputError("FILE_NOT_FOUND", f"{name} file not found: {path}", "Check the path.")
        try:
            return np.asarray(np.load(path, allow_pickle=False))
        except (ValueError, OSError) as e:
            raise InputError(
                "BAD_NPY", f"Could not read {name} from {path}: {e}", f"Save {name} with np.save."
            ) from e
    return np.asarray(value)


class DemoLimitError(ValueError):
    """A request exceeds what a public demo allows."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.hint = "This is a read-only demo with limits. Run faissight locally to lift them."


@dataclass(frozen=True)
class DemoLimits:
    """Caps for a shared, read-only deployment (e.g. a Hugging Face Space)."""

    max_k: int = 100
    max_ef_search: int = 1024
    max_sweep_queries: int = 200
    max_sweep_values: int = 12
    max_sweep_k: int = 50
    umap_from_cache_only: bool = True


class Session:
    """Everything faissight knows about one index.

    Expensive state (reconstructed vectors, ground truth, IVF assignments, projections) is
    computed on first use and cached; projections and embedder loading run as background
    jobs. Safe to share across request threads.
    """

    def __init__(
        self,
        index: Any,
        *,
        vectors: ArrayInput | None = None,
        ids: ArrayInput | None = None,
        metadata: Any = None,
        embedder: str | Embedder | None = None,
        queries: ArrayInput | None = None,
        max_points: int = DEFAULT_MAX_POINTS,
        cache_root: Path | None = None,
        disk_cache: bool = True,
        normalize_text: bool | None = None,
        demo_mode: bool = False,
    ) -> None:
        self.li: LoadedIndex = load_index(index)
        self.jobs = JobRunner()
        self.max_points = int(max_points)
        if self.max_points < 1:
            raise InputError("BAD_MAX_POINTS", "max_points must be >= 1.", "Use e.g. 50000.")
        # Resolve now, not in the background thread, so later env changes can't redirect it.
        self._cache_root = cache_root or default_cache_root()
        self._disk_cache = disk_cache
        self._lock = threading.RLock()
        self._lazy: dict[str, Any] = {}
        self._key_locks: dict[str, threading.RLock] = {}

        self._raw: VectorSource | None = None
        if vectors is not None:
            x = _load_array(vectors, "vectors")
            id_arr = _load_array(ids, "ids") if ids is not None else None
            try:
                self._raw = from_arrays(self.li, x, id_arr)
            except VectorMismatchError as e:
                raise InputError("VECTOR_MISMATCH", str(e), e.hint) from e
        elif ids is not None:
            raise InputError("IDS_WITHOUT_VECTORS", "--ids needs --vectors.", "Pass both.")

        self.queries: np.ndarray[Any, Any] | None = None
        if queries is not None:
            q = _load_array(queries, "queries")
            if q.ndim != 2 or q.shape[1] != self.li.d:
                raise InputError(
                    "QUERY_MISMATCH",
                    f"Queries have shape {q.shape}, the index expects (n, {self.li.d}).",
                    "Pass query vectors with the index's input dimension.",
                )
            self.queries = np.ascontiguousarray(q, dtype=np.float32)

        try:
            self.metadata: Metadata | None = (
                load_metadata(metadata) if metadata is not None else None
            )
        except ValueError as e:
            raise InputError("BAD_METADATA", str(e), getattr(e, "hint", "")) from e

        self.embedder_name: str | None = None
        self._embedder: Embedder | None = None
        self._embedder_model: str | None = None
        if isinstance(embedder, str):
            self.embedder_name = embedder
            self._embedder_model = embedder
        elif embedder is not None:
            self.embedder_name = getattr(embedder, "__name__", type(embedder).__name__)
            self._embedder = embedder
        self.normalize_text = normalize_text
        self.demo_limits: DemoLimits | None = DemoLimits() if demo_mode else None

    # --- lazily computed state ------------------------------------------------------------

    def _once(self, name: str, fn: Callable[[], T]) -> T:
        """Compute ``fn()`` once per ``name``, thread-safely.

        Each name has its own lock, so a computation that waits on a background job (which
        itself needs other lazy values) can't deadlock the session.
        """
        with self._lock:
            key_lock = self._key_locks.setdefault(name, threading.RLock())
        with key_lock:
            if name not in self._lazy:
                self._lazy[name] = fn()
            value: T = self._lazy[name]
            return value

    @property
    def has_raw_vectors(self) -> bool:
        return self._raw is not None

    @property
    def source(self) -> VectorSource:
        """Raw vectors if given, else vectors reconstructed from the index (computed once)."""
        if self._raw is not None:
            return self._raw
        return self._once("reconstructed", lambda: reconstruct_all(self.li))

    @property
    def ground_truth(self) -> GroundTruth:
        return self._once("ground_truth", lambda: GroundTruth(self.source, self.li.metric))

    @property
    def assignments(self) -> ivf.Assignments | None:
        if not self.li.kind.is_ivf:
            return None
        return self._once("assignments", lambda: ivf.assignments(self.li))

    def list_stats(self) -> ivf.ListStats:
        return self._once("list_stats", lambda: ivf.list_stats(self.li))

    @property
    def index_sha1(self) -> str:
        return self._once("index_sha1", lambda: index_fingerprint(self.li))

    def known_ids(self) -> npt.NDArray[np.int64]:
        """User-facing ids of the stored vectors, found as cheaply as possible."""
        if self._raw is not None:
            return self._raw.ids
        if self.li.ids is not None:
            return self.li.ids
        if self.assignments is not None:
            return self.assignments.ids
        return np.arange(self.li.ntotal, dtype=np.int64)

    def metadata_coverage(self) -> float | None:
        metadata = self.metadata
        if metadata is None:
            return None
        return self._once("metadata_coverage", lambda: metadata.coverage(self.known_ids()))

    # --- embedder -------------------------------------------------------------------------

    def _should_normalize_text(self) -> bool:
        if self.normalize_text is not None:
            return self.normalize_text
        if self._raw is not None:
            return looks_normalized(self._raw.vectors)
        return self.li.metric is Metric.IP

    def embedder_job(self) -> Job[Embedder] | None:
        """Background job loading the sentence-transformers model (None if not applicable)."""
        if self._embedder_model is None:
            return None
        model = self._embedder_model
        return self.jobs.get_or_start(
            ("embedder", model),
            lambda progress: sentence_transformer_embedder(
                model, normalize=self._should_normalize_text()
            ),
        )

    def get_embedder(self) -> Embedder | None:
        """The embedder, waiting for the model to load if needed. Raises if loading failed."""
        if self._embedder is not None:
            return self._embedder
        if self._embedder_model is None:
            return None
        return self.jobs.run_sync(
            ("embedder", self._embedder_model),
            lambda progress: sentence_transformer_embedder(
                self._embedder_model or "", normalize=self._should_normalize_text()
            ),
        )

    # --- queries --------------------------------------------------------------------------

    def query(
        self,
        *,
        id: int | None = None,
        vector: npt.ArrayLike | None = None,
        text: str | None = None,
        k: int = 10,
        nprobe: int | None = None,
        ef_search: int | None = None,
        compare: bool = True,
    ) -> QueryReport:
        """Resolve, search, and (with ``compare``) explain against exact ground truth."""
        if self.demo_limits is not None:
            lim = self.demo_limits
            if k > lim.max_k:
                raise DemoLimitError(f"k is capped at {lim.max_k} in this demo.")
            if ef_search is not None and ef_search > lim.max_ef_search:
                raise DemoLimitError(f"efSearch is capped at {lim.max_ef_search} in this demo.")
        rq = resolve_query(
            self.li,
            id=id,
            vector=vector,
            text=text,
            source=self.source if id is not None else None,
            embedder=self.get_embedder() if text is not None else None,
        )
        return explain_query(
            self.li,
            rq,
            k,
            nprobe=nprobe,
            ef_search=ef_search,
            ground_truth=self.ground_truth if compare else None,
            assignments=self.assignments if compare else None,
        )

    # --- projections ----------------------------------------------------------------------

    def projection_job(
        self, method: ProjectionMethod | str = ProjectionMethod.PCA, dims: int = 2
    ) -> Job[Projection]:
        """Start (or return) the background job computing this projection."""
        method = ProjectionMethod(method)
        if dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {dims}.")
        if method is ProjectionMethod.UMAP and importlib.util.find_spec("umap") is None:
            raise ProjectionUnavailableError()
        if not self.li.is_supported:
            raise ValueError(f"Cannot project: {self.li.unsupported_reason}")
        key = ("projection", method.value, dims)
        if (
            self.demo_limits is not None
            and self.demo_limits.umap_from_cache_only
            and method is ProjectionMethod.UMAP
            and self.jobs.get(key) is None
            and not self._umap_cached(dims)
        ):
            raise DemoLimitError("UMAP isn't precomputed for this view in the demo; use PCA.")

        def work(progress: Callable[[float, str], None]) -> Projection:
            cache = ProjectionCache(self.index_sha1, self._cache_root) if self._disk_cache else None
            return compute_projection(
                self.li,
                self.source,
                method=method,
                dims=dims,
                max_points=self.max_points,
                assignments=self.assignments,
                cache=cache,
                progress=progress,
            )

        return self.jobs.get_or_start(key, work)

    def _umap_cached(self, dims: int) -> bool:
        if not self._disk_cache:
            return False
        cache = ProjectionCache(self.index_sha1, self._cache_root)
        hit = cached_projection(
            self.li,
            self.source,
            cache,
            method=ProjectionMethod.UMAP,
            dims=dims,
            max_points=self.max_points,
            assignments=self.assignments,
        )
        return hit is not None

    def place(self, proj: Projection, vector: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Coordinates of an input-space vector in a finished projection."""
        x_core = self.li.to_core_space(np.atleast_2d(np.asarray(vector, dtype=np.float32)))
        ref = None
        if proj.pca is None:
            key = f"proj_ref:{proj.method.value}:{proj.dims}"
            ref = self._once(key, lambda: core_vectors(self.li, self.source, proj.ids))
        placed: npt.NDArray[np.float32] = place_points(proj, x_core, ref)[0]
        return placed

    # --- quantization ---------------------------------------------------------------------

    def pq_job(self) -> Job[pq_mod.QuantizationReport]:
        """Background job measuring reconstruction error. Needs raw vectors."""
        if not self.li.is_supported:
            raise ValueError(f"Cannot analyse: {self.li.unsupported_reason}")
        if self._raw is None:
            raise pq_mod.RawVectorsRequiredError()
        raw = self._raw

        def work(progress: Callable[[float, str], None]) -> pq_mod.QuantizationReport:
            progress(0.1, "Decoding stored vectors")
            stored = self._once("reconstructed", lambda: reconstruct_all(self.li))
            progress(0.6, "Measuring error and distortion")
            return pq_mod.analyze(self.li, raw, stored, self.assignments)

        return self.jobs.get_or_start(("pq",), work)

    # --- HNSW -----------------------------------------------------------------------------

    @property
    def hnsw_graph(self) -> hnsw_mod.HnswGraph:
        return self._once("hnsw_graph", lambda: hnsw_mod.extract_graph(self.li))

    @property
    def hnsw_vectors(self) -> npt.NDArray[np.float32]:
        """Core-space vector of every HNSW node, by internal id."""
        return self._once("hnsw_vectors", lambda: hnsw_trace.storage_vectors(self.li))

    def hnsw_layout(self) -> npt.NDArray[np.float32]:
        """2-D position of every HNSW node, from the session's PCA projection."""

        def compute() -> npt.NDArray[np.float32]:
            job = self.projection_job(ProjectionMethod.PCA, 2)
            job.wait()
            if job.error is not None:
                raise job.error
            assert job.result is not None
            assert job.result.pca is not None
            return job.result.pca.transform(self.hnsw_vectors)

        return self._once("hnsw_layout", compute)

    def hnsw_trace(
        self, vector: npt.ArrayLike, k: int, ef_search: int | None = None
    ) -> hnsw_trace.HnswTrace:
        if not self.li.kind.is_hnsw:
            raise ValueError("HNSW traces need an HNSW index.")
        return hnsw_trace.trace_for_index(
            self.li, self.hnsw_graph, self.hnsw_vectors, vector, k, ef_search
        )

    # --- sweeps ---------------------------------------------------------------------------

    def sweep_queries(self, n_queries: int = DEFAULT_N_QUERIES) -> QuerySet:
        """The ``--queries`` set (first ``n_queries``) or ``n_queries`` sampled stored vectors."""
        if self.queries is not None:
            return given_queries(self.li, self.queries[:n_queries])
        return self._once(
            f"sweep_queries:{n_queries}", lambda: sample_queries(self.source, n_queries)
        )

    def sweep_job(
        self,
        param: SweepParam | str | None = None,
        values: list[int] | None = None,
        k: int = 10,
        n_queries: int = DEFAULT_N_QUERIES,
    ) -> tuple[str, Job[SweepResult]]:
        """Validate, then start (or return) the background sweep. Returns ``(job_id, job)``."""
        if not self.li.is_supported:
            raise ValueError(f"Cannot sweep: {self.li.unsupported_reason}")
        if k < 1 or n_queries < 1:
            raise ValueError("k and n_queries must be >= 1.")
        p, vals = check_values(self.li, param, values)
        if self.demo_limits is not None:
            lim = self.demo_limits
            if n_queries > lim.max_sweep_queries:
                raise DemoLimitError(f"Sweeps use at most {lim.max_sweep_queries} queries here.")
            if len(vals) > lim.max_sweep_values:
                raise DemoLimitError(f"Sweeps try at most {lim.max_sweep_values} values here.")
            if k > lim.max_sweep_k:
                raise DemoLimitError(f"Sweeps use k <= {lim.max_sweep_k} here.")
            if p is SweepParam.EF_SEARCH and max(vals) > lim.max_ef_search:
                raise DemoLimitError(f"efSearch is capped at {lim.max_ef_search} in this demo.")
        key = ("sweep", p.value, tuple(vals), k, n_queries)
        job_id = hashlib.sha1(repr(key).encode()).hexdigest()[:12]

        def work(progress: Callable[[float, str], None]) -> SweepResult:
            progress(0.0, "Computing exact ground truth")
            qs = self.sweep_queries(n_queries)
            truth = self._once(
                f"sweep_truth:{qs.origin}:{len(qs)}:{k}",
                lambda: ground_truth_ids(self.ground_truth, qs, k),
            )
            return sweep(
                self.li,
                qs,
                truth,
                param=p,
                values=vals,
                k=k,
                truth_reconstructed=not self.has_raw_vectors,
                progress=progress,
            )

        with self._lock:
            self._lazy.setdefault("sweep_ids", {})[job_id] = key
        return job_id, self.jobs.get_or_start(key, work)

    def get_sweep_job(self, job_id: str) -> Job[SweepResult] | None:
        with self._lock:
            key = self._lazy.get("sweep_ids", {}).get(job_id)
        return self.jobs.get(key) if key is not None else None

    def start_background(self) -> None:
        """Kick off work the UI will want soon: the default projection and the embedder."""
        if self.li.is_supported:
            self.projection_job(ProjectionMethod.PCA, 2)
        with contextlib.suppress(EmbedderUnavailableError):  # surfaced on first text query
            self.embedder_job()
