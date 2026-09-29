"""JSON API endpoints (mounted under ``/api``)."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from faissight import __version__
from faissight.core import hnsw as H
from faissight.core import hnsw_trace as HT
from faissight.core import ivf
from faissight.core import pq as PQ
from faissight.core.jobs import Job, JobStatus
from faissight.core.projection import Projection
from faissight.core.search import MissReason, NeighbourTrace, QueryReport
from faissight.core.sweep import SweepResult, default_values, param_for
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
        sweep=_sweep_defaults(session),
        demo_limits=S.DemoLimitsOut(**session.demo_limits.__dict__)
        if session.demo_limits is not None
        else None,
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
        ivf_trace=_ivf_trace_response(session, report) if req.trace and trace else None,
    )


@router.post("/trace/ivf", response_model=S.IvfTraceResponse)
def trace_ivf(req: S.SearchRequest, session: IvfDep) -> S.IvfTraceResponse:
    return _ivf_trace_response(session, _run_query(session, req, force_compare=True))


def _ivf_trace_response(session: Session, report: QueryReport) -> S.IvfTraceResponse:
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


# --- sweep -------------------------------------------------------------------------------


def _sweep_defaults(session: Session) -> S.SweepDefaults | None:
    li = session.li
    if not li.is_supported or not (li.kind.is_ivf or li.kind.is_hnsw):
        return None
    param = param_for(li)
    return S.SweepDefaults(
        param=param.value,
        values=default_values(li, param),
        max_value=int(li.ivf.nlist) if li.kind.is_ivf else None,
    )


def _sweep_response(job_id: str, job: Job[SweepResult], session: Session) -> JSONResponse:
    snapshot = job.as_dict()
    result = None
    if snapshot["status"] == "done" and job.result is not None:
        r = job.result
        result = S.SweepResultOut(
            param=r.param.value,
            k=r.k,
            n_queries=r.n_queries,
            query_origin=r.query_origin,
            truth_source="reconstructed" if r.truth_reconstructed else "raw",
            points=[
                S.SweepPointOut(
                    value=p.value,
                    recall=p.recall,
                    latency_mean_ms=p.latency_mean_ms,
                    latency_p95_ms=p.latency_p95_ms,
                )
                for p in r.points
            ],
            pareto_values=[p.value for p in r.pareto()],
            repeats=r.repeats,
            seed=r.seed,
            query_sha256=r.query_sha256,
            environment=r.environment,
            query_seed=r.query_seed,
        )
    body = S.SweepJobResponse(job_id=job_id, result=result, **snapshot)
    status = 202 if snapshot["status"] == "running" else 200
    return JSONResponse(status_code=status, content=body.model_dump())


@router.post(
    "/sweep",
    response_model=S.SweepJobResponse,
    responses={202: {"model": S.SweepJobResponse}},
)
def start_sweep(req: S.SweepRequest, session: SupportedDep) -> JSONResponse:
    """Start (or reuse) a background sweep; poll ``GET /sweep/{job_id}`` until done."""
    job_id, job = session.sweep_job(
        req.param, req.values, req.k, req.n_queries, req.repeats, req.seed
    )
    return _sweep_response(job_id, job, session)


@router.get(
    "/sweep/{job_id}",
    response_model=S.SweepJobResponse,
    responses={202: {"model": S.SweepJobResponse}},
)
def get_sweep(job_id: str, session: SessionDep) -> JSONResponse:
    job = session.get_sweep_job(job_id)
    if job is None:
        raise ApiError(
            404, "NOT_FOUND", f"No sweep with id {job_id}.", "Start one with POST /api/sweep."
        )
    return _sweep_response(job_id, job, session)


@router.delete("/sweep/{job_id}", response_model=S.SweepJobResponse)
def cancel_sweep(job_id: str, session: SessionDep) -> JSONResponse:
    """Cancel a queued sweep or stop a running sweep at its next progress checkpoint."""
    job = session.get_sweep_job(job_id)
    if job is None:
        raise ApiError(404, "NOT_FOUND", f"No sweep with id {job_id}.")
    session.jobs.cancel(job.key)
    return _sweep_response(job_id, job, session)


# --- HNSW --------------------------------------------------------------------------------


def hnsw_session(session: SupportedDep) -> Session:
    if not session.li.kind.is_hnsw:
        raise ApiError(
            400,
            "NOT_HNSW",
            f"{session.li.kind.value} is not an HNSW index.",
            "This view needs an HNSW index.",
        )
    return session


HnswDep = Annotated[Session, Depends(hnsw_session)]


def _user_id(session: Session, internal: int) -> int:
    return int(session.li.user_ids(np.array([internal], dtype=np.int64))[0])


@router.get("/hnsw/stats", response_model=S.HnswStatsResponse)
def hnsw_stats(session: HnswDep) -> S.HnswStatsResponse:
    g = session.hnsw_graph
    st = H.graph_stats(g)
    p = session.li.params
    return S.HnswStatsResponse(
        entry_point=_user_id(session, st.entry_point),
        max_level=st.max_level,
        m=int(p.hnsw_m or 0),
        ef_search=int(p.ef_search or 0),
        ef_construction=int(p.ef_construction or 0),
        levels=[S.HnswLevelStats(**lv.__dict__) for lv in st.levels],
    )


@router.get("/hnsw/graph", response_model=S.HnswGraphResponse)
def hnsw_graph(
    session: HnswDep,
    level: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=20_000)] = 2_000,
    around: int | None = None,
) -> S.HnswGraphResponse:
    """Nodes and links on one level. Large levels are sampled by BFS from ``around``
    (a user-facing id) or from the entry point."""
    g = session.hnsw_graph
    if level > g.max_level:
        raise ApiError(404, "NOT_FOUND", f"Level {level} doesn't exist (max level {g.max_level}).")
    level_nodes = g.nodes_at_level(level)
    if len(level_nodes) <= limit and around is None:
        nodes = level_nodes
    else:
        seed = g.entry_point
        if around is not None:
            internal = int(HT.internal_ids(session.li, [around])[0])
            if internal < 0:
                raise ApiError(404, "NOT_FOUND", f"Id {around} is not in the index.")
            if g.node_levels[internal] < level:
                raise ApiError(400, "BAD_REQUEST", f"Id {around} is not on level {level}.")
            seed = internal
        nodes = H.neighbourhood(g, [seed], level, limit)
    src, dst = g.edges(level, nodes)
    pairs = {(min(a, b), max(a, b)) for a, b in zip(src.tolist(), dst.tolist(), strict=True)}
    pos = {int(n): i for i, n in enumerate(nodes.tolist())}
    xy = session.hnsw_layout()[nodes]
    return S.HnswGraphResponse(
        level=level,
        n_level_nodes=len(level_nodes),
        sampled=len(nodes) < len(level_nodes),
        ids=session.li.user_ids(nodes).tolist(),
        x=_round(xy[:, 0]),
        y=_round(xy[:, 1]),
        top_levels=g.node_levels[nodes].tolist(),
        edges_src=[pos[a] for a, _ in sorted(pairs)],
        edges_dst=[pos[b] for _, b in sorted(pairs)],
    )


@router.post("/trace/hnsw", response_model=S.HnswTraceResponse)
def trace_hnsw(req: S.SearchRequest, session: HnswDep) -> S.HnswTraceResponse:
    li = session.li
    report = session.query(
        id=req.query.id,
        vector=req.query.vector,
        text=req.query.text,
        k=req.k,
        ef_search=req.ef_search,
        compare=req.compare,
    )
    ef = int(report.result.params["efSearch"])
    # Trace k+1 when the query is a stored vector, then drop it, like the search does.
    extra = 1 if report.query.exclude_id is not None else 0
    tr = session.hnsw_trace(report.query.vector, req.k + extra, ef)
    g = session.hnsw_graph
    ip = li.metric.higher_is_closer

    def shown(d: float) -> float:
        return -d if ip else d  # the trace searches on -IP; show similarities for IP

    uid = li.user_ids
    levels = []
    for level in range(g.max_level, -1, -1):
        steps = [s for s in tr.steps if s.level == level]
        levels.append(
            S.HnswLevelTrace(
                level=level,
                entry=int(uid(np.array([tr.level_entries[level]]))[0]),
                steps=[
                    S.HnswStepOut(
                        expanded=int(uid(np.array([s.expanded]))[0]),
                        expanded_distance=shown(s.expanded_distance),
                        visits=[
                            S.HnswVisitOut(
                                node=int(v), distance=shown(vis.distance), accepted=vis.accepted
                            )
                            for v, vis in zip(
                                uid(np.array([x.node for x in s.visits], dtype=np.int64)).tolist(),
                                s.visits,
                                strict=True,
                            )
                        ],
                    )
                    for s in steps
                ],
            )
        )

    result_ids = uid(tr.results)
    result_d = tr.result_distances
    if extra:
        keep = result_ids != report.query.exclude_id
        result_ids, result_d = result_ids[keep], result_d[keep]
    results = [
        S.ResultRow(rank=r, id=int(i), distance=shown(float(d)), snippet=_snippet(session, int(i)))
        for r, (i, d) in enumerate(zip(result_ids[: req.k], result_d[: req.k], strict=True))
    ]
    faiss_ids = report.result.valid_ids.tolist()
    overlap = len({r.id for r in results} & set(faiss_ids)) / max(len(faiss_ids), 1)

    truth_rows = None
    if report.truth is not None:
        t_ids = report.truth.valid_ids
        outcomes = HT.neighbour_outcomes(tr, HT.internal_ids(li, t_ids))
        internal = HT.internal_ids(li, t_ids)
        truth_rows = [
            S.HnswTraceNeighbour(
                rank=r,
                id=int(t),
                distance=float(d),
                outcome=o.value,
                top_level=int(g.node_levels[n]),
                snippet=_snippet(session, int(t)),
            )
            for r, (t, d, o, n) in enumerate(
                zip(t_ids, report.truth.distances, outcomes, internal, strict=True)
            )
        ]

    # Layout for every node the UI will draw: visited, results, truth, level entries.
    involved = set(tr.visited()) | set(tr.results.tolist()) | {s.expanded for s in tr.steps}
    if report.truth is not None:
        involved |= {int(n) for n in HT.internal_ids(li, report.truth.valid_ids) if n >= 0}
    nodes = np.array(sorted(involved), dtype=np.int64)
    xy = session.hnsw_layout()[nodes]
    q_core = li.to_core_space(np.asarray(report.query.vector, dtype=np.float32)[None])
    job = session.jobs.get(("projection", "pca", 2))
    pca = job.result.pca if job is not None and job.is_done and job.result is not None else None
    return S.HnswTraceResponse(
        metric=li.metric.value,
        higher_is_closer=ip,
        ef_search=ef,
        k=req.k,
        entry_point=int(uid(np.array([g.entry_point]))[0]),
        max_level=g.max_level,
        levels=levels,
        results=results,
        faiss_ids=faiss_ids,
        overlap_with_faiss=overlap,
        truth=truth_rows,
        recall=report.recall,
        nodes=S.HnswTraceNodes(
            ids=uid(nodes).tolist(),
            x=_round(xy[:, 0]),
            y=_round(xy[:, 1]),
            top_levels=g.node_levels[nodes].tolist(),
        ),
        query_xy=_round(pca.transform(q_core)[0]) if pca is not None else None,
    )


# --- quantization ------------------------------------------------------------------------


def _r(values: npt.NDArray[np.float32] | npt.NDArray[np.float64], digits: int = 6) -> list[float]:
    """Round to significant digits for compact JSON."""
    out: list[float] = [float(f"{v:.{digits}g}") for v in np.asarray(values, dtype=np.float64)]
    return out


@router.get(
    "/pq/error",
    response_model=S.PqErrorResponse,
    responses={202: {"model": S.JobStatusResponse}},
)
def pq_error(session: SupportedDep) -> S.PqErrorResponse | JSONResponse:
    """Reconstruction error of the stored vectors (needs raw vectors)."""
    try:
        job = session.pq_job()
    except PQ.RawVectorsRequiredError as e:
        return S.PqErrorResponse(available=False, reason=str(e), hint=e.hint)
    if job.status is JobStatus.FAILED:
        raise ApiError(500, "PQ_FAILED", f"Quantization analysis failed: {job.error}")
    if not job.is_done:
        return JSONResponse(
            status_code=202, content=S.JobStatusResponse(**job.as_dict()).model_dump()
        )
    rep = job.result
    assert rep is not None
    q = rep.quantiles()
    edges, counts = rep.histogram()
    per_list = rep.per_list()
    list_of = (
        dict(zip(rep.ids.tolist(), rep.list_nos.tolist(), strict=True))
        if rep.list_nos is not None
        else {}
    )
    d = rep.distortion
    return S.PqErrorResponse(
        available=True,
        kind=session.li.kind.value,
        metric=session.li.metric.value,
        n=len(rep.ids),
        code_size=rep.code_size,
        raw_bytes=rep.raw_bytes,
        has_transform=bool(session.li.transforms),
        mean=rep.mean,
        median=q["median"],
        p95=q["p95"],
        max=q["max"],
        relative_mean=float(rep.relative.mean()),
        histogram=S.PqHistogram(edges=_r(edges), counts=counts.tolist()),
        per_list=[S.PqListError(list_no=i, size=n, mean_error=m) for i, n, m in per_list]
        if rep.list_nos is not None
        else None,
        worst=[
            S.PqWorst(
                id=i, error=e, relative=r, list_no=list_of.get(i), snippet=_snippet(session, i)
            )
            for i, e, r in rep.worst(50)
        ],
        distortion=S.PqDistortion(
            true=_r(d.true, 5),
            approx=_r(d.approx, 5),
            near=d.near.tolist(),
            correlation=d.correlation,
            near_correlation=d.near_correlation,
        ),
    )
