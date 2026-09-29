"""In-memory state for one inspected index: inputs, lazily computed caches, background jobs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from faissight.core import ivf
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

    # --- lazily computed state ------------------------------------------------------------

    def _once(self, name: str, fn: Callable[[], T]) -> T:
        with self._lock:
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

        return self.jobs.get_or_start(("projection", method.value, dims), work)

    def place(self, proj: Projection, vector: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Coordinates of an input-space vector in a finished projection."""
        x_core = self.li.to_core_space(np.atleast_2d(np.asarray(vector, dtype=np.float32)))
        ref = None
        if proj.pca is None:
            key = f"proj_ref:{proj.method.value}:{proj.dims}"
            ref = self._once(key, lambda: core_vectors(self.li, self.source, proj.ids))
        placed: npt.NDArray[np.float32] = place_points(proj, x_core, ref)[0]
        return placed

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
