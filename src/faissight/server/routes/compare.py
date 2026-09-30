"""Comparisons with other indexes given at startup (background jobs)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from faissight.core.comparison import ComparisonResult, IndexMeasurement, changed_queries
from faissight.core.jobs import Job
from faissight.core.sweep import param_for
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SessionDep,
    SupportedDep,
    job_response,
)
from faissight.session import Candidate, Session

router = APIRouter()


MAX_COMPARE_CHANGES = 200


def candidate_out(i: int, c: Candidate) -> S.CompareCandidateOut:
    li = c.li
    search_param: Literal["nprobe", "efSearch"] | None = None
    if li.kind.is_ivf or li.kind.is_hnsw:
        search_param = param_for(li).value
    return S.CompareCandidateOut(
        index=i,
        name=c.name,
        kind=li.kind.value,
        ntotal=li.ntotal,
        params=li.params.as_dict(),
        search_param=search_param,
        max_value=int(li.ivf.nlist) if li.kind.is_ivf else None,
    )


def _measurement_out(name: str, m: IndexMeasurement) -> S.CompareMeasurementOut:
    return S.CompareMeasurementOut(
        name=name,
        kind=m.kind,
        params=m.params,
        serialized_bytes=m.serialized_bytes,
        recall=m.recall,
        recall_ci_low=m.recall_ci_low,
        recall_ci_high=m.recall_ci_high,
        latency_mean_ms=m.latency_mean_ms,
        latency_p95_ms=m.latency_p95_ms,
    )


def _compare_response(
    job_id: str, job: Job[ComparisonResult], session: Session, candidate: int
) -> JSONResponse:
    snapshot = job.as_dict()
    result = None
    if snapshot["status"] == "done" and job.result is not None:
        r = job.result
        changed = changed_queries(r)
        li = session.li
        result = S.CompareResultOut(
            metric=r.metric,
            k=r.k,
            n_queries=r.n_queries,
            query_origin="given" if r.query_origin == "given" else "sampled",
            query_sha256=r.query_sha256,
            query_seed=r.query_seed,
            repeats=r.repeats,
            seed=r.seed,
            environment=r.environment,
            left=_measurement_out(li.path.name if li.path else "in-memory index", r.left),
            right=_measurement_out(session.candidates[candidate].name, r.right),
            n_changed=len(changed),
            n_improved=sum(d.right_recall > d.left_recall for d in r.differences),
            n_worsened=sum(d.right_recall < d.left_recall for d in r.differences),
            changes=[
                S.QueryChangeOut(
                    query_no=d.query_no,
                    id=d.query_id,
                    left_recall=d.left_recall,
                    right_recall=d.right_recall,
                    left_only=d.left_only,
                    right_only=d.right_only,
                    overlap=d.overlap,
                )
                for d in changed[:MAX_COMPARE_CHANGES]
            ],
            changes_truncated=len(changed) > MAX_COMPARE_CHANGES,
        )
    return job_response(S.CompareJobResponse(job_id=job_id, result=result, **snapshot))


def _compare_job_or_404(session: Session, job_id: str) -> tuple[Job[ComparisonResult], int]:
    found = session.get_compare_job(job_id)
    if found is None:
        raise ApiError(
            404,
            "NOT_FOUND",
            f"No comparison with id {job_id}.",
            "Start one with POST /api/compare.",
        )
    return found


@router.post(
    "/compare",
    response_model=S.CompareJobResponse,
    responses={202: {"model": S.CompareJobResponse}},
)
def start_compare(req: S.CompareRequest, session: SupportedDep) -> JSONResponse:
    """Start (or reuse) a background comparison; poll ``GET /compare/{job_id}``."""
    if not session.candidates:
        raise ApiError(
            400,
            "NO_CANDIDATES",
            "There is no other index to compare with.",
            "Start faissight with --compare OTHER.index (and --vectors).",
        )
    job_id, job = session.compare_job(
        req.candidate,
        req.k,
        req.n_queries,
        req.repeats,
        req.seed,
        req.left_nprobe,
        req.right_nprobe,
        req.left_ef_search,
        req.right_ef_search,
    )
    return _compare_response(job_id, job, session, req.candidate)


@router.get(
    "/compare/{job_id}",
    response_model=S.CompareJobResponse,
    responses={202: {"model": S.CompareJobResponse}},
)
def get_compare(job_id: str, session: SessionDep) -> JSONResponse:
    job, candidate = _compare_job_or_404(session, job_id)
    return _compare_response(job_id, job, session, candidate)


@router.delete("/compare/{job_id}", response_model=S.CompareJobResponse)
def cancel_compare(job_id: str, session: SessionDep) -> JSONResponse:
    """Cancel a queued comparison or stop a running one at its next checkpoint."""
    job, candidate = _compare_job_or_404(session, job_id)
    session.jobs.cancel(job.key)
    return _compare_response(job_id, job, session, candidate)
