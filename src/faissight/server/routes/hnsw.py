"""HNSW graph structure and search traces."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, Query

from faissight.core import hnsw as H
from faissight.core import hnsw_trace as HT
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SupportedDep,
    round_coords,
    snippet,
)
from faissight.session import Session

router = APIRouter()


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
        x=round_coords(xy[:, 0]),
        y=round_coords(xy[:, 1]),
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
        S.ResultRow(rank=r, id=int(i), distance=shown(float(d)), snippet=snippet(session, int(i)))
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
                snippet=snippet(session, int(t)),
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
            x=round_coords(xy[:, 0]),
            y=round_coords(xy[:, 1]),
            top_levels=g.node_levels[nodes].tolist(),
        ),
        query_xy=round_coords(pca.transform(q_core)[0]) if pca is not None else None,
    )
