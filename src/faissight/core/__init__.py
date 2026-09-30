"""Core library: FAISS inspection with no web dependencies.

Typical flow::

    from faissight import core

    li = core.load_index("my.index")                   # detect kind, unwrap wrappers
    source = core.from_arrays(li, raw_vectors)          # or core.reconstruct_all(li)
    gt = core.GroundTruth(source, li.metric)
    q = core.resolve_query(li, id=123, source=source)   # or vector=..., text=... + embedder
    report = core.explain_query(
        li, q, k=10, nprobe=8, ground_truth=gt, assignments=core.assignments(li)
    )
    report.recall, report.ivf_trace.min_nprobe_for_all

A bare search is ``core.search.search(li, vector, k, nprobe=...)`` and a sweep is
``core.sweep.sweep(li, queries, truth, ...)``. Neither function is re-exported here because
it would shadow its module.

FAISS is imported lazily, so importing this package works without FAISS installed.
"""

from faissight.core._faiss import FaissNotInstalledError
from faissight.core.comparison import ComparisonResult, compare_indexes
from faissight.core.embed import Embedder, EmbedderUnavailableError
from faissight.core.ivf import (
    Assignments,
    ListStats,
    NotAnIVFIndexError,
    assignments,
    centroids,
    imbalance_factor,
    list_members,
    list_sizes,
    list_stats,
)
from faissight.core.jobs import Job, JobRunner, JobStatus
from faissight.core.loader import IndexLoadError, load_index
from faissight.core.metadata import Metadata, MetadataError, load_metadata
from faissight.core.projection import (
    PcaModel,
    Projection,
    ProjectionCache,
    ProjectionMethod,
    ProjectionUnavailableError,
    compute_projection,
    fit_pca,
    index_fingerprint,
    place_points,
    stratified_sample,
)
from faissight.core.search import (
    GroundTruth,
    IvfTrace,
    MissReason,
    NeighbourTrace,
    QueryError,
    QueryKind,
    QueryReport,
    ResolvedQuery,
    SearchResult,
    explain_query,
    recall_at_k,
    resolve_query,
    trace_ivf,
)
from faissight.core.sweep import (
    QuerySet,
    SweepParam,
    SweepPoint,
    SweepResult,
    WorstQuery,
    default_values,
    given_queries,
    ground_truth_ids,
    sample_queries,
)
from faissight.core.types import IndexKind, IndexParams, LoadedIndex, Metric, TransformInfo
from faissight.core.vectors import VectorMismatchError, VectorSource, from_arrays, reconstruct_all

__all__ = [
    "Assignments",
    "ComparisonResult",
    "Embedder",
    "EmbedderUnavailableError",
    "FaissNotInstalledError",
    "GroundTruth",
    "IndexKind",
    "IndexLoadError",
    "IndexParams",
    "IvfTrace",
    "Job",
    "JobRunner",
    "JobStatus",
    "ListStats",
    "LoadedIndex",
    "Metadata",
    "MetadataError",
    "Metric",
    "MissReason",
    "NeighbourTrace",
    "NotAnIVFIndexError",
    "PcaModel",
    "Projection",
    "ProjectionCache",
    "ProjectionMethod",
    "ProjectionUnavailableError",
    "QueryError",
    "QueryKind",
    "QueryReport",
    "QuerySet",
    "ResolvedQuery",
    "SearchResult",
    "SweepParam",
    "SweepPoint",
    "SweepResult",
    "TransformInfo",
    "VectorMismatchError",
    "VectorSource",
    "WorstQuery",
    "assignments",
    "centroids",
    "compare_indexes",
    "compute_projection",
    "default_values",
    "explain_query",
    "fit_pca",
    "from_arrays",
    "given_queries",
    "ground_truth_ids",
    "imbalance_factor",
    "index_fingerprint",
    "list_members",
    "list_sizes",
    "list_stats",
    "load_index",
    "load_metadata",
    "place_points",
    "recall_at_k",
    "reconstruct_all",
    "resolve_query",
    "sample_queries",
    "stratified_sample",
    "trace_ivf",
]
