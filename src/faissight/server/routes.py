"""JSON API endpoints (mounted under ``/api``)."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from faissight import __version__
from faissight.core import ivf
from faissight.core.jobs import JobStatus
from faissight.core.projection import Projection
from faissight.core.search import MissReason, NeighbourTrace, QueryReport
from faissight.server import schemas as S
from faissight.session import Session

router = APIRouter()
COORD_DECIMALS = 4


class ApiError(Exception):
    """Raised by routes; rendered as ``{error_code, message, hint}``."""

    def __init__(self, status: int, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint


def get_session(request: Request) -> Session:
    session: Session = request.app.state.session
    return session


SessionDep = Annotated[Session, Depends(get_session)]


def supported_session(session: SessionDep) -> Session:
    if not session.li.is_supported:
        raise ApiError(
            400,
            "UNSUPPORTED_INDEX",
            session.li.unsupported_reason or "Unsupported index.",
            "faissight can only show basic stats (see /api/info) for this index.",
        )
    return session


SupportedDep = Annotated[Session, Depends(supported_session)]


def ivf_session(session: SupportedDep) -> Session:
    if not session.li.kind.is_ivf:
        raise ApiError(
            400,
            "NOT_IVF",
            f"{session.li.kind.value} is not an IVF index.",
            "This view needs an IVF index.",
        )
    return session


IvfDep = Annotated[Session, Depends(ivf_session)]


def _snippet(session: Session, id: int) -> dict[str, str] | None:
    return session.metadata.snippet(id) if session.metadata is not None else None


def _truth_source(session: Session) -> Literal["raw", "reconstructed"]:
    return "raw" if session.has_raw_vectors else "reconstructed"


# --- basics ------------------------------------------------------------------------------


@router.get("/health", response_model=S.HealthResponse)
def health() -> S.HealthResponse:
    return S.HealthResponse()


@router.get("/info", response_model=S.InfoResponse)
def info(session: SessionDep) -> S.InfoResponse:
    li = session.li
    embedder_status: Literal["loading", "ready", "failed"] | None = None
    if session.embedder_name is not None:
        job = session.embedder_job()
        if job is None or job.status is JobStatus.DONE:
            embedder_status = "ready"
        elif job.status is JobStatus.FAILED:
            embedder_status = "failed"
        else:
            embedder_status = "loading"
    md = session.metadata
    return S.InfoResponse(
        version=__version__,
        name=li.path.name if li.path else "in-memory index",
        kind=li.kind.value,
        class_chain=li.class_chain,
        transforms=[S.TransformOut(name=t.name, d_in=t.d_in, d_out=t.d_out) for t in li.transforms],
        d=li.d,
        core_d=li.core_d,
        ntotal=li.ntotal,
        metric=li.metric.value,
        higher_is_closer=li.metric.higher_is_closer,
        is_trained=li.is_trained,
        params=li.params.as_dict(),
        has_id_map=li.has_id_map,
        has_refine=li.has_refine,
        supported=li.is_supported,
        unsupported_reason=li.unsupported_reason,
        ground_truth_source=_truth_source(session) if li.is_supported else None,
        max_points=session.max_points,
        inputs=S.InputsOut(
            raw_vectors=session.has_raw_vectors,
            metadata_rows=len(md) if md is not None else None,
            metadata_columns=md.columns if md is not None else [],
            metadata_coverage=session.metadata_coverage(),
            embedder=session.embedder_name,
            embedder_status=embedder_status,
            queries=len(session.queries) if session.queries is not None else None,
        ),
    )


@router.get("/metadata/{id}", response_model=S.MetadataResponse)
def metadata(id: int, session: SessionDep) -> S.MetadataResponse:
    if session.metadata is None:
        raise ApiError(404, "NO_METADATA", "No metadata was loaded.", "Start with --meta.")
    row = session.metadata.get(id)
    if row is None:
        raise ApiError(404, "NOT_FOUND", f"No metadata for id {id}.")
    return S.MetadataResponse(id=id, row=row)


# --- IVF ---------------------------------------------------------------------------------


@router.get("/ivf/lists", response_model=S.ListSizesResponse)
def ivf_lists(
    session: IvfDep, top: Annotated[int, Query(ge=1, le=1000)] = 20
) -> S.ListSizesResponse:
    st = session.list_stats()
    top_lists = [S.TopList(list_no=i, size=s) for i, s in st.top_lists[:top]]
    if top > len(st.top_lists):  # list_stats caches the default top-20
        order = np.argsort(-st.sizes, kind="stable")[:top]
        top_lists = [S.TopList(list_no=int(i), size=int(st.sizes[i])) for i in order]
    return S.ListSizesResponse(
        nlist=len(st.sizes),
        sizes=st.sizes.tolist(),
        n_empty=st.n_empty,
        imbalance_factor=st.imbalance_factor,
        min=st.min,
        median=st.median,
        max=st.max,
        top_lists=top_lists,
        top_5pct_share=st.top_5pct_share,
    )


@router.get("/ivf/list/{list_no}", response_model=S.ListMembersResponse)
def ivf_list(
    list_no: int,
    session: IvfDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
) -> S.ListMembersResponse:
    try:
        members = ivf.list_members(session.li, list_no)
    except IndexError as e:
        raise ApiError(404, "NOT_FOUND", str(e)) from e
    page = members[offset : offset + limit]
    return S.ListMembersResponse(
        list_no=list_no,
        size=len(members),
        offset=offset,
        limit=limit,
        members=[S.MemberOut(id=int(i), snippet=_snippet(session, int(i))) for i in page],
    )


# --- projection --------------------------------------------------------------------------


def _round(a: npt.NDArray[np.float32]) -> list[float]:
    rounded: list[float] = np.round(a.astype(np.float64), COORD_DECIMALS).tolist()
    return rounded


def _ready_projection(session: Session, method: str, dims: int) -> Projection | None:
    job = session.jobs.get(("projection", method, dims))
    return job.result if job is not None and job.is_done else None


@router.get(
    "/projection",
    response_model=S.ProjectionResponse,
    responses={202: {"model": S.JobStatusResponse}},
)
def projection(
    session: SupportedDep,
    kind: Literal["points", "centroids"] = "points",
    method: Literal["pca", "umap"] = "pca",
    dims: Annotated[int, Query(ge=2, le=3)] = 2,
) -> S.ProjectionResponse | JSONResponse:
    if kind == "centroids" and not session.li.kind.is_ivf:
        raise ApiError(400, "NOT_IVF", "Only IVF indexes have centroids.")
    job = session.projection_job(method, dims)
    if job.status is JobStatus.FAILED:
        raise ApiError(
            500, "PROJECTION_FAILED", f"Projection failed: {job.error}", "See the server log."
        )
    if not job.is_done:
        body = S.JobStatusResponse(**job.as_dict())
        return JSONResponse(status_code=202, content=body.model_dump())
    proj = job.result
    assert proj is not None
    ev = proj.pca.explained_variance_ratio.tolist() if proj.pca is not None else None
    if kind == "centroids":
        assert proj.centroid_coords is not None
        coords = proj.centroid_coords
        ids: list[int] = list(range(len(coords)))
        list_nos: list[int] | None = ids
    else:
        coords = proj.coords
        ids = proj.ids.tolist()
        list_nos = proj.list_nos.tolist() if proj.list_nos is not None else None
    return S.ProjectionResponse(
        kind=kind,
        method=proj.method.value,
        dims=proj.dims,  # type: ignore[arg-type]
        ids=ids,
        x=_round(coords[:, 0]),
        y=_round(coords[:, 1]),
        z=_round(coords[:, 2]) if proj.dims == 3 else None,
        list_nos=list_nos,
        n_total=proj.n_total,
        sampled=proj.sampled,
        explained_variance=ev,
    )


# --- search & trace ----------------------------------------------------------------------


def _run_query(session: Session, req: S.SearchRequest, force_compare: bool = False) -> QueryReport:
    q = req.query
    return session.query(
        id=q.id,
        vector=q.vector,
        text=q.text,
        k=req.k,
        nprobe=req.nprobe,
        ef_search=req.ef_search,
        compare=req.compare or force_compare,
    )


def _truth_rows(
    session: Session, report: QueryReport, traces: list[NeighbourTrace] | None
) -> list[S.TruthRow] | None:
    if report.truth is None:
        return None
    by_id = {t.id: t for t in traces or []}
    found = {int(i): r for r, i in enumerate(report.result.ids) if i >= 0}
    list_nos = session.assignments.lookup(report.truth.ids) if session.assignments else None
    rows = []
    for rank, (tid, dist) in enumerate(zip(report.truth.ids, report.truth.distances, strict=True)):
        if tid < 0:
            continue
        t = by_id.get(int(tid))
        reason = t.reason.value if t else (MissReason.FOUND.value if int(tid) in found else None)
        rows.append(
            S.TruthRow(
                rank=rank,
                id=int(tid),
                distance=float(dist),
                list_no=int(list_nos[rank]) if list_nos is not None else None,
                probe_rank=t.probe_rank if t else None,
                reason=reason,
                found_rank=found.get(int(tid)),
                snippet=_snippet(session, int(tid)),
            )
        )
    return rows


@router.post("/search", response_model=S.SearchResponse)
def search(req: S.SearchRequest, session: SupportedDep) -> S.SearchResponse:
    report = _run_query(session, req)
    r = report.result
    truth_ids = set(report.truth.valid_ids.tolist()) if report.truth is not None else None
    list_nos = session.assignments.lookup(r.ids) if session.assignments is not None else None
    results = [
        S.ResultRow(
            rank=rank,
            id=int(i),
            distance=float(d),
            list_no=int(list_nos[rank]) if list_nos is not None else None,
            in_truth=(int(i) in truth_ids) if truth_ids is not None else None,
            snippet=_snippet(session, int(i)),
        )
        for rank, (i, d) in enumerate(zip(r.ids, r.distances, strict=True))
        if i >= 0
    ]
    trace = report.ivf_trace
    query_coords = None
    if req.projection is not None:
        proj = _ready_projection(session, req.projection.method, req.projection.dims)
        if proj is not None:
            query_coords = _round(session.place(proj, report.query.vector))
    return S.SearchResponse(
        query_kind=report.query.kind.value,
        metric=r.metric.value,
        higher_is_closer=r.metric.higher_is_closer,
        k=req.k,
        params=r.params,
        latency_ms=r.latency_ms,
        results=results,
        truth=_truth_rows(session, report, trace.neighbours if trace else None),
        recall=report.recall,
        truth_source=_truth_source(session) if report.truth is not None else None,
        min_nprobe=trace.min_nprobe_for_all if trace else None,
        reason_counts={k.value: v for k, v in trace.reason_counts().items()} if trace else None,
        query_coords=query_coords,
    )


@router.post("/trace/ivf", response_model=S.IvfTraceResponse)
def trace_ivf(req: S.SearchRequest, session: IvfDep) -> S.IvfTraceResponse:
    report = _run_query(session, req, force_compare=True)
    trace = report.ivf_trace
    assert trace is not None
    sizes = session.list_stats().sizes
    per_list: dict[int, int] = {}
    for n in trace.neighbours:
        per_list[n.list_no] = per_list.get(n.list_no, 0) + 1
    probes = [
        S.ProbeRow(
            probe_rank=rank,
            list_no=int(list_no),
            centroid_distance=float(dist),
            probed=rank < trace.nprobe,
            size=int(sizes[list_no]),
            n_true_neighbours=per_list.get(int(list_no), 0),
        )
        for rank, (list_no, dist) in enumerate(
            zip(trace.probe_order, trace.centroid_distances, strict=True)
        )
        if list_no >= 0
    ]
    return S.IvfTraceResponse(
        nprobe=trace.nprobe,
        nlist=len(sizes),
        min_nprobe=trace.min_nprobe_for_all,
        recall=report.recall,
        metric=report.result.metric.value,
        higher_is_closer=report.result.metric.higher_is_closer,
        probes=probes,
        neighbours=_truth_rows(session, report, trace.neighbours) or [],
    )
