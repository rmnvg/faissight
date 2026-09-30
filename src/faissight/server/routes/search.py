"""Search with ground-truth comparison, and the IVF probe trace."""

from __future__ import annotations

from fastapi import APIRouter

from faissight.core.search import MissReason, NeighbourTrace, QueryReport
from faissight.server import schemas as S
from faissight.server.routes.common import (
    IvfDep,
    SupportedDep,
    round_coords,
    snippet,
    truth_source,
)
from faissight.server.routes.projection import ready_projection
from faissight.session import Session

router = APIRouter()


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
                snippet=snippet(session, int(tid)),
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
            snippet=snippet(session, int(i)),
        )
        for rank, (i, d) in enumerate(zip(r.ids, r.distances, strict=True))
        if i >= 0
    ]
    trace = report.ivf_trace
    query_coords = None
    if req.projection is not None:
        proj = ready_projection(session, req.projection.method, req.projection.dims)
        if proj is not None:
            query_coords = round_coords(session.place(proj, report.query.vector))
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
        truth_source=truth_source(session) if report.truth is not None else None,
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
