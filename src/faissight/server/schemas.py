"""Pydantic request/response models for the JSON API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, PlainSerializer, WithJsonSchema, model_validator

from faissight.core.search import MissReason


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


Scalar = int | float | bool | str

# Preserve existing numeric ids while encoding values outside JavaScript's exact range.
# Python continues to use int internally; clients can send decimal strings as well.
UserId = Annotated[
    int,
    Field(ge=-1, le=2**63 - 1),
    WithJsonSchema(
        {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": r"^-?[0-9]+$"}]},
        mode="validation",
    ),
    PlainSerializer(
        lambda value: str(value) if value > 2**53 - 1 else value,
        return_type=int | str,
        when_used="json",
    ),
]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    hint: str | None = None


# --- info --------------------------------------------------------------------------------


class TransformOut(BaseModel):
    name: str
    d_in: int
    d_out: int


class InputsOut(BaseModel):
    raw_vectors: bool
    metadata_rows: int | None
    metadata_columns: list[str]
    metadata_coverage: float | None
    embedder: str | None
    embedder_status: Literal["loading", "ready", "failed"] | None
    queries: int | None


class MemoryOut(BaseModel):
    vectors_bytes: int | None
    """Raw vectors (n x d x 4); None without --vectors."""
    vectors_mapped: bool
    """Raw vectors are read from a memory-mapped file (--mmap), not held in RAM."""
    mmap_note: str | None
    """Why --mmap could not keep the vectors mapped, if so."""
    reconstruct_bytes: int
    """Memory needed to decode every stored vector (quantization analysis; ground truth and
    maps without --vectors)."""
    reconstructed: bool
    """The decoded vectors are already in memory."""
    pq_workspace_bytes: int


class SweepDefaults(BaseModel):
    param: Literal["nprobe", "efSearch"]
    values: list[int]
    max_value: int | None
    """Largest accepted value (nlist for nprobe; None when unbounded)."""


class CompareCandidateOut(BaseModel):
    index: int
    name: str
    kind: str
    ntotal: int
    params: dict[str, Scalar]
    search_param: Literal["nprobe", "efSearch"] | None
    """The speed/recall knob of this index, if any."""
    max_value: int | None
    """Largest accepted value (nlist for nprobe; None when unbounded)."""


class DemoLimitsOut(BaseModel):
    max_k: int
    max_ef_search: int
    max_sweep_queries: int
    max_sweep_values: int
    max_sweep_k: int
    umap_from_cache_only: bool


class InfoResponse(BaseModel):
    version: str
    name: str
    kind: str
    class_chain: list[str]
    transforms: list[TransformOut]
    d: int
    core_d: int
    ntotal: int
    metric: str
    higher_is_closer: bool
    is_trained: bool
    params: dict[str, Scalar]
    has_id_map: bool
    has_refine: bool
    supported: bool
    unsupported_reason: str | None
    ground_truth_source: Literal["raw", "reconstructed"] | None
    max_points: int
    sweep: SweepDefaults | None
    demo_limits: DemoLimitsOut | None = None
    """Set when the server runs in read-only demo mode."""
    compare: list[CompareCandidateOut] = []
    """Indexes given with --compare (index 0 is the first); empty when there are none."""
    inputs: InputsOut
    memory: MemoryOut


# --- IVF ---------------------------------------------------------------------------------


class TopList(BaseModel):
    list_no: int
    size: int


class ListSizesResponse(BaseModel):
    nlist: int
    sizes: list[int]
    n_empty: int
    imbalance_factor: float
    min: int
    median: float
    max: int
    top_lists: list[TopList]
    top_5pct_share: float


class MemberOut(BaseModel):
    id: UserId
    snippet: dict[str, str] | None = None


class ListMembersResponse(BaseModel):
    list_no: int
    size: int
    offset: int
    limit: int
    members: list[MemberOut]


# --- jobs & projection -------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    status: Literal["running", "done", "failed", "cancelled"]
    progress: float
    message: str
    error: str | None = None


class ProjectionResponse(BaseModel):
    status: Literal["done"] = "done"
    kind: Literal["points", "centroids"]
    method: Literal["pca", "umap"]
    dims: Literal[2, 3]
    ids: list[UserId]
    """Point ids (``kind=points``) or list numbers (``kind=centroids``)."""
    x: list[float]
    y: list[float]
    z: list[float] | None = None
    list_nos: list[int] | None = None
    n_total: int
    sampled: bool
    explained_variance: list[float] | None = None
    cache_warning: str | None = None
    """Set when the projection couldn't be saved to the disk cache (it is still valid)."""


class ProjectionRef(BaseModel):
    method: Literal["pca", "umap"] = "pca"
    dims: Literal[2, 3] = 2


# --- search & trace ----------------------------------------------------------------------


class QueryIn(BaseModel):
    id: UserId | None = None
    vector: list[float] | None = None
    text: str | None = None
    row: int | None = Field(None, ge=0)
    """Row of the evaluation query set given with ``--queries`` (as sweeps number them)."""

    @model_validator(mode="after")
    def _exactly_one(self) -> QueryIn:
        given = [k for k in ("id", "vector", "text", "row") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(
                f"Give exactly one of id, vector, text or row (got {given or 'none'})."
            )
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: QueryIn
    k: int = Field(10, ge=1, le=1000)
    nprobe: int | None = Field(None, ge=1)
    ef_search: int | None = Field(None, ge=1, alias="efSearch")
    compare: bool = True
    candidates: int | None = Field(None, ge=1, le=10000)
    trace: bool = False
    """On an IVF index, also return the probe trace (saves a second /trace/ivf round trip)."""
    projection: ProjectionRef | None = None
    """If this projection is ready, the response includes the query's coordinates."""

    @model_validator(mode="after")
    def _trace_needs_compare(self) -> SearchRequest:
        if self.candidates is not None and self.candidates < self.k:
            raise ValueError("candidates must be >= k.")
        if self.trace and not self.compare:
            raise ValueError("trace explains misses against exact ground truth; set compare.")
        return self


class ResultRow(BaseModel):
    rank: int
    id: UserId
    distance: float
    list_no: int | None = None
    in_truth: bool | None = None
    snippet: dict[str, str] | None = None


class TruthRow(BaseModel):
    rank: int
    id: UserId
    distance: float
    list_no: int | None = None
    probe_rank: int | None = None
    reason: MissReason | None = None
    found_rank: int | None = None
    snippet: dict[str, str] | None = None


class SearchResponse(BaseModel):
    query_kind: Literal["id", "vector", "text", "row"]
    metric: str
    higher_is_closer: bool
    k: int
    params: dict[str, int]
    latency_ms: float
    results: list[ResultRow]
    reranked: list[ResultRow] | None = None
    reranked_recall: float | None = None
    reranked_latency_ms: float | None = None
    candidate_count: int | None = None
    truth: list[TruthRow] | None = None
    recall: float | None = None
    truth_source: Literal["raw", "reconstructed"] | None = None
    min_nprobe: int | None = None
    reason_counts: dict[str, int] | None = None
    query_coords: list[float] | None = None
    ivf_trace: IvfTraceResponse | None = None
    """Present when ``trace`` was requested on an IVF index."""


class ProbeRow(BaseModel):
    probe_rank: int
    list_no: int
    centroid_distance: float
    probed: bool
    size: int
    n_true_neighbours: int


class IvfTraceResponse(BaseModel):
    nprobe: int
    nlist: int
    min_nprobe: int
    recall: float | None
    metric: str
    higher_is_closer: bool
    probes: list[ProbeRow]
    neighbours: list[TruthRow]


class MetadataResponse(BaseModel):
    id: UserId
    row: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


# --- sweep -------------------------------------------------------------------------------


class SweepRequest(BaseModel):
    param: Literal["nprobe", "efSearch"] | None = None
    values: list[int] | None = Field(None, min_length=1, max_length=64)
    k: int = Field(10, ge=1, le=1000)
    n_queries: int = Field(200, ge=1, le=10_000)
    repeats: int = Field(3, ge=1, le=20)
    seed: int = Field(0, ge=0, le=2**32 - 1)


class WorstQueryOut(BaseModel):
    query_no: int
    id: UserId | None
    """The query's own stored id (sampled queries); None for given queries."""
    recall: float
    probe_coverage: float | None = None
    """IVF: share of its true neighbours in the probed lists (caps its recall)."""


class SweepPointOut(BaseModel):
    value: int
    recall: float
    latency_mean_ms: float
    latency_p95_ms: float
    recall_ci_low: float | None
    recall_ci_high: float | None
    """Approximate 95% interval for the mean recall."""
    recall_distribution: list[tuple[float, int]]
    """``(recall, n_queries)`` pairs, lowest recall first."""
    worst_queries: list[WorstQueryOut]
    probe_coverage: float | None = None
    """IVF: mean share of true neighbours in probed lists, the most recall probing allows."""


class SweepResultOut(BaseModel):
    param: Literal["nprobe", "efSearch"]
    k: int
    n_queries: int
    query_origin: Literal["given", "sampled"]
    truth_source: Literal["raw", "reconstructed"]
    points: list[SweepPointOut]
    pareto_values: list[int]
    repeats: int
    seed: int
    query_sha256: str
    environment: dict[str, str | int]
    query_seed: int | None = None


class SweepJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed", "cancelled"]
    progress: float
    message: str
    error: str | None = None
    result: SweepResultOut | None = None


class EvidenceOut(BaseModel):
    label: str
    value: str


class SuggestionOut(BaseModel):
    kind: Literal[
        "PROBE_MORE",
        "RANKING_LIMIT",
        "SEARCH_WIDER",
        "RECALL_PLATEAU",
        "FAILING_QUERIES",
        "LIST_IMBALANCE",
        "LATENCY_BUDGET",
    ]
    title: str
    detail: str
    evidence: list[EvidenceOut]
    sweep_values: list[int] | None
    """Values for a follow-up sweep of the same parameter."""
    view: Literal["overview", "quantization", "compare", "query"] | None
    query_id: UserId | None
    """With ``view == "query"``: the stored id to open."""
    at_value: int | None
    """The parameter value the evidence was measured at."""


class SweepAdviceResponse(BaseModel):
    target_recall: float
    confident: bool
    max_p95_ms: float | None = None
    suggestions: list[SuggestionOut]
    """Most important first; empty when nothing needs changing."""


class RunIndexOut(BaseModel):
    sha1: str
    name: str | None
    kind: str
    class_chain: list[str]
    ntotal: int
    d: int
    metric: str
    params: dict[str, Scalar]


class RunQueriesOut(BaseModel):
    origin: Literal["given", "sampled"]
    n: int
    sha256: str
    seed: int | None


class RunSettingsOut(BaseModel):
    param: Literal["nprobe", "efSearch"]
    values: list[int]
    k: int
    repeats: int
    seed: int
    truth_source: Literal["raw", "reconstructed"]
    ground_truth_fingerprint: str | None = None
    """Identifies the raw --vectors corpus ground truth was computed on; None when the
    index itself is the ground-truth identity (reconstructed) or the run predates this."""


class RunDecisionOut(BaseModel):
    target_recall: float
    confident: bool
    max_p95_ms: float | None
    status: Literal["ok", "recall", "latency"]
    recommended: int | None


class RunWorstQueryOut(BaseModel):
    query_no: int
    id: UserId | None
    """The query's own stored id (sampled queries); None for given queries."""
    recall: float


class RunPointOut(BaseModel):
    value: int
    recall: float
    recall_ci_low: float | None
    recall_ci_high: float | None
    latency_mean_ms: float
    latency_p95_ms: float
    probe_coverage: float | None
    recall_distribution: list[tuple[float, int]]
    worst_queries: list[RunWorstQueryOut]


class SweepRunOut(BaseModel):
    """A saveable record of a finished sweep (``faissight.sweep-run``); see ``core.runs``."""

    format: str
    version: int
    label: str | None
    created_at: str
    faissight: str
    index: RunIndexOut
    queries: RunQueriesOut
    settings: RunSettingsOut
    environment: dict[str, str | int]
    decision: RunDecisionOut
    points: list[RunPointOut]


class BaselineRequest(BaseModel):
    target: float = Field(0.95, ge=0.0, le=1.0)
    confident: bool = False
    max_p95_ms: float | None = Field(None, gt=0)
    """The decision the current run is judged by: target recall and optional p95 budget."""
    baseline: dict[str, Any]
    """A run saved with ``GET /api/sweep/{job_id}/run`` or ``faissight sweep --save``."""
    max_recall_drop: float = Field(0.01, ge=0, le=1)
    """A setting regresses when its recall falls by more than this (absolute)."""
    max_p95_increase: float = Field(0.2, ge=0)
    """... or its p95 latency grows by more than this (relative, 0.2 = 20%)..."""
    min_p95_increase_ms: float = Field(0.05, ge=0)
    """... and by more than this many milliseconds (sub-ms timings are noisy)."""


class PointDeltaOut(BaseModel):
    value: int
    baseline_recall: float
    recall: float
    recall_change: float
    baseline_p95_ms: float
    p95_ms: float
    p95_change: float
    recall_regressed: bool
    latency_regressed: bool


class BaselineResponse(BaseModel):
    baseline_label: str | None
    baseline_created_at: str | None
    baseline_index: str | None
    """File name of the baseline's index, when it had one."""
    points: list[PointDeltaOut]
    comparable: bool
    """False when no setting could be judged (see ``notes``): not a pass, nothing ran."""
    regressed: bool
    recall_comparable: bool
    latency_comparable: bool
    notes: list[str]
    baseline_recommended: int | None
    recommended: int | None
    max_recall_drop: float
    max_p95_increase: float
    min_p95_increase_ms: float


# --- comparison --------------------------------------------------------------------------


class CompareRequest(BaseModel):
    candidate: int = Field(0, ge=0)
    k: int = Field(10, ge=1, le=1000)
    n_queries: int = Field(200, ge=1, le=10_000)
    repeats: int = Field(3, ge=1, le=20)
    seed: int = Field(0, ge=0, le=2**32 - 1)
    left_nprobe: int | None = Field(None, ge=1)
    right_nprobe: int | None = Field(None, ge=1)
    left_ef_search: int | None = Field(None, ge=1)
    right_ef_search: int | None = Field(None, ge=1)


class CompareMeasurementOut(BaseModel):
    name: str
    kind: str
    params: dict[str, int]
    serialized_bytes: int
    """Size of ``faiss.serialize_index``; not the process's resident memory."""
    recall: float
    recall_ci_low: float | None
    recall_ci_high: float | None
    latency_mean_ms: float
    latency_p95_ms: float


class QueryChangeOut(BaseModel):
    query_no: int
    id: UserId | None
    """The query's own stored id (sampled queries); None for given queries."""
    left_recall: float
    right_recall: float
    left_only: list[UserId]
    right_only: list[UserId]
    overlap: int


class CompareResultOut(BaseModel):
    metric: str
    k: int
    n_queries: int
    query_origin: Literal["given", "sampled"]
    query_sha256: str
    query_seed: int | None
    repeats: int
    seed: int
    environment: dict[str, str | int]
    left: CompareMeasurementOut
    right: CompareMeasurementOut
    n_changed: int
    """Queries whose result sets differ between the two indexes."""
    n_improved: int
    """Queries with higher recall on the right index."""
    n_worsened: int
    changes: list[QueryChangeOut]
    """Changed queries, largest recall change first (capped)."""
    changes_truncated: bool


class CompareJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed", "cancelled"]
    progress: float
    message: str
    error: str | None = None
    result: CompareResultOut | None = None


# --- HNSW --------------------------------------------------------------------------------


class HnswLevelStats(BaseModel):
    level: int
    n_nodes: int
    max_links: int
    degree_mean: float
    degree_min: int
    degree_max: int
    degree_hist: list[int]


class HnswStatsResponse(BaseModel):
    entry_point: UserId
    max_level: int
    m: int
    ef_search: int
    ef_construction: int
    levels: list[HnswLevelStats]
    """Top level first."""


class HnswGraphResponse(BaseModel):
    level: int
    n_level_nodes: int
    sampled: bool
    ids: list[UserId]
    x: list[float]
    y: list[float]
    top_levels: list[int]
    edges_src: list[int]
    """Indices into ``ids``; each undirected pair appears once."""
    edges_dst: list[int]


class HnswVisitOut(BaseModel):
    node: UserId
    distance: float
    accepted: bool


class HnswStepOut(BaseModel):
    expanded: UserId
    expanded_distance: float
    visits: list[HnswVisitOut]


class HnswLevelTrace(BaseModel):
    level: int
    entry: UserId
    steps: list[HnswStepOut]


class HnswTraceNeighbour(BaseModel):
    rank: int
    id: UserId
    distance: float
    outcome: Literal["FOUND", "VISITED_NOT_KEPT", "NOT_REACHED"]
    top_level: int
    snippet: dict[str, str] | None = None


class HnswTraceNodes(BaseModel):
    ids: list[UserId]
    x: list[float]
    y: list[float]
    top_levels: list[int]


class HnswTraceResponse(BaseModel):
    metric: str
    higher_is_closer: bool
    ef_search: int
    k: int
    entry_point: UserId
    max_level: int
    levels: list[HnswLevelTrace]
    """Top level first; level 0 last."""
    results: list[ResultRow]
    faiss_ids: list[UserId]
    """What ``index.search`` itself returned, for checking the reconstruction."""
    overlap_with_faiss: float
    truth: list[HnswTraceNeighbour] | None = None
    recall: float | None = None
    nodes: HnswTraceNodes
    """Layout for every node in the trace: ``ids``, ``x``, ``y``, ``top_levels``."""
    query_xy: list[float] | None = None


# --- quantization ------------------------------------------------------------------------


class PqHistogram(BaseModel):
    edges: list[float]
    counts: list[int]


class PqListError(BaseModel):
    list_no: int
    size: int
    mean_error: float


class PqWorst(BaseModel):
    id: UserId
    error: float
    relative: float
    list_no: int | None = None
    snippet: dict[str, str] | None = None


class PqDistortion(BaseModel):
    true: list[float]
    approx: list[float]
    near: list[bool]
    correlation: float
    near_correlation: float


class PqErrorResponse(BaseModel):
    available: bool
    reason: str | None = None
    hint: str | None = None
    kind: str | None = None
    metric: str | None = None
    n: int | None = None
    code_size: int | None = None
    raw_bytes: int | None = None
    has_transform: bool = False
    mean: float | None = None
    median: float | None = None
    p95: float | None = None
    max: float | None = None
    relative_mean: float | None = None
    histogram: PqHistogram | None = None
    per_list: list[PqListError] | None = None
    worst: list[PqWorst] | None = None
    distortion: PqDistortion | None = None


class EvaluationRequest(BaseModel):
    judgements: str = Field(min_length=1, max_length=5_000_000)
    k: int = Field(10, ge=1, le=1000)
    candidates: int | None = Field(None, ge=1, le=10_000)
    nprobe: int | None = Field(None, ge=1)
    ef_search: int | None = Field(None, ge=1)


class RelevanceMetricsOut(BaseModel):
    recall: float
    mrr: float
    ndcg: float


class EvaluationRowOut(BaseModel):
    row: int
    metrics: RelevanceMetricsOut
    latency_ms: float
    ids: list[str]
    reranked_metrics: RelevanceMetricsOut | None = None
    reranked_latency_ms: float | None = None
    reranked_ids: list[str] | None = None


class EvaluationResultOut(BaseModel):
    k: int
    metric: str
    params: dict[str, int]
    n_queries: int
    ndcg_gain: str
    unjudged: str
    metrics: RelevanceMetricsOut
    reranked_metrics: RelevanceMetricsOut | None
    candidates: int | None
    queries: list[EvaluationRowOut]


class EvaluationJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed", "cancelled"]
    progress: float
    message: str
    error: str | None = None
    result: EvaluationResultOut | None = None


class HistorySummary(BaseModel):
    id: str
    kind: str
    label: str
    index: str
    created_at: str


class HistoryResponse(BaseModel):
    enabled: bool
    warning: str | None = None
    current_index: str | None = None
    """sha1 of the open index, to tell its runs from other indexes' (None when disabled)."""
    runs: list[HistorySummary]


class HistoryRecord(HistorySummary):
    data: dict[str, Any]


class HistoryRename(BaseModel):
    label: str = Field(min_length=1, max_length=200)


class HistoryCompareRequest(BaseModel):
    baseline: str
    current: str


class HistoryComparison(BaseModel):
    comparable: bool
    regressed: bool
    notes: list[str]
    points: list[PointDeltaOut]
