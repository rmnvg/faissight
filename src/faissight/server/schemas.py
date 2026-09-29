"""Pydantic request/response models for the JSON API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = int | float | bool | str


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


class SweepDefaults(BaseModel):
    param: Literal["nprobe", "efSearch"]
    values: list[int]
    max_value: int | None
    """Largest accepted value (nlist for nprobe; None when unbounded)."""


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
    inputs: InputsOut


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
    id: int
    snippet: dict[str, str] | None = None


class ListMembersResponse(BaseModel):
    list_no: int
    size: int
    offset: int
    limit: int
    members: list[MemberOut]


# --- jobs & projection -------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    status: Literal["running", "done", "failed"]
    progress: float
    message: str
    error: str | None = None


class ProjectionResponse(BaseModel):
    status: Literal["done"] = "done"
    kind: Literal["points", "centroids"]
    method: Literal["pca", "umap"]
    dims: Literal[2, 3]
    ids: list[int]
    """Point ids (``kind=points``) or list numbers (``kind=centroids``)."""
    x: list[float]
    y: list[float]
    z: list[float] | None = None
    list_nos: list[int] | None = None
    n_total: int
    sampled: bool
    explained_variance: list[float] | None = None


class ProjectionRef(BaseModel):
    method: Literal["pca", "umap"] = "pca"
    dims: Literal[2, 3] = 2


# --- search & trace ----------------------------------------------------------------------


class QueryIn(BaseModel):
    id: int | None = None
    vector: list[float] | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> QueryIn:
        given = [k for k in ("id", "vector", "text") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(f"Give exactly one of id, vector or text (got {given or 'none'}).")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: QueryIn
    k: int = Field(10, ge=1, le=1000)
    nprobe: int | None = Field(None, ge=1)
    ef_search: int | None = Field(None, ge=1, alias="efSearch")
    compare: bool = True
    projection: ProjectionRef | None = None
    """If this projection is ready, the response includes the query's coordinates."""


class ResultRow(BaseModel):
    rank: int
    id: int
    distance: float
    list_no: int | None = None
    in_truth: bool | None = None
    snippet: dict[str, str] | None = None


class TruthRow(BaseModel):
    rank: int
    id: int
    distance: float
    list_no: int | None = None
    probe_rank: int | None = None
    reason: str | None = None
    found_rank: int | None = None
    snippet: dict[str, str] | None = None


class SearchResponse(BaseModel):
    query_kind: Literal["id", "vector", "text"]
    metric: str
    higher_is_closer: bool
    k: int
    params: dict[str, int]
    latency_ms: float
    results: list[ResultRow]
    truth: list[TruthRow] | None = None
    recall: float | None = None
    truth_source: Literal["raw", "reconstructed"] | None = None
    min_nprobe: int | None = None
    reason_counts: dict[str, int] | None = None
    query_coords: list[float] | None = None


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
    id: int
    row: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


# --- sweep -------------------------------------------------------------------------------


class SweepRequest(BaseModel):
    param: Literal["nprobe", "efSearch"] | None = None
    values: list[int] | None = Field(None, min_length=1, max_length=64)
    k: int = Field(10, ge=1, le=1000)
    n_queries: int = Field(200, ge=1, le=10_000)


class SweepPointOut(BaseModel):
    value: int
    recall: float
    latency_mean_ms: float
    latency_p95_ms: float


class SweepResultOut(BaseModel):
    param: Literal["nprobe", "efSearch"]
    k: int
    n_queries: int
    query_origin: Literal["given", "sampled"]
    truth_source: Literal["raw", "reconstructed"]
    points: list[SweepPointOut]
    pareto_values: list[int]


class SweepJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed"]
    progress: float
    message: str
    error: str | None = None
    result: SweepResultOut | None = None


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
    entry_point: int
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
    ids: list[int]
    x: list[float]
    y: list[float]
    top_levels: list[int]
    edges_src: list[int]
    """Indices into ``ids``; each undirected pair appears once."""
    edges_dst: list[int]


class HnswVisitOut(BaseModel):
    node: int
    distance: float
    accepted: bool


class HnswStepOut(BaseModel):
    expanded: int
    expanded_distance: float
    visits: list[HnswVisitOut]


class HnswLevelTrace(BaseModel):
    level: int
    entry: int
    steps: list[HnswStepOut]


class HnswTraceNeighbour(BaseModel):
    rank: int
    id: int
    distance: float
    outcome: Literal["FOUND", "VISITED_NOT_KEPT", "NOT_REACHED"]
    top_level: int
    snippet: dict[str, str] | None = None


class HnswTraceResponse(BaseModel):
    metric: str
    higher_is_closer: bool
    ef_search: int
    k: int
    entry_point: int
    max_level: int
    levels: list[HnswLevelTrace]
    """Top level first; level 0 last."""
    results: list[ResultRow]
    faiss_ids: list[int]
    """What ``index.search`` itself returned, for checking the reconstruction."""
    overlap_with_faiss: float
    truth: list[HnswTraceNeighbour] | None = None
    recall: float | None = None
    nodes: dict[str, list[float]]
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
    id: int
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
