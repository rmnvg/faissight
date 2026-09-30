"""nprobe/efSearch sweeps (background jobs)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from faissight.core.jobs import Job
from faissight.core.sweep import SweepResult, default_values, param_for
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SessionDep,
    SupportedDep,
    job_response,
)
from faissight.session import Session

router = APIRouter()


def sweep_defaults(session: Session) -> S.SweepDefaults | None:
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
                    recall_ci_low=p.recall_ci_low,
                    recall_ci_high=p.recall_ci_high,
                    recall_distribution=p.recall_distribution,
                    worst_queries=[
                        S.WorstQueryOut(
                            query_no=w.query_no,
                            id=w.id,
                            recall=w.recall,
                            probe_coverage=w.probe_coverage,
                        )
                        for w in p.worst_queries
                    ],
                    probe_coverage=p.probe_coverage,
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
    return job_response(S.SweepJobResponse(job_id=job_id, result=result, **snapshot))


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


@router.get("/sweep/{job_id}/advice", response_model=S.SweepAdviceResponse)
def sweep_advice(
    job_id: str,
    session: SessionDep,
    target: Annotated[float, Query(ge=0.0, le=1.0, description="Target recall@k.")] = 0.95,
    confident: Annotated[
        bool, Query(description="Judge by the 95% lower bound of recall, not the mean.")
    ] = False,
    max_p95_ms: Annotated[
        float | None, Query(gt=0, description="p95 latency budget per query (ms).")
    ] = None,
) -> S.SweepAdviceResponse:
    """Suggested next steps for a finished sweep, with the measurements behind each."""
    job = session.get_sweep_job(job_id)
    if job is None:
        raise ApiError(
            404, "NOT_FOUND", f"No sweep with id {job_id}.", "Start one with POST /api/sweep."
        )
    if not job.is_done or job.result is None:
        raise ApiError(
            409, "NOT_READY", "This sweep hasn't finished.", "Poll GET /api/sweep/{job_id} first."
        )
    suggestions = session.sweep_advice(
        job.result, target, confident=confident, max_p95_ms=max_p95_ms
    )
    return S.SweepAdviceResponse(
        target_recall=target,
        confident=confident,
        max_p95_ms=max_p95_ms,
        suggestions=[
            S.SuggestionOut.model_validate({**asdict(s), "kind": s.kind.value}) for s in suggestions
        ],
    )


@router.delete("/sweep/{job_id}", response_model=S.SweepJobResponse)
def cancel_sweep(job_id: str, session: SessionDep) -> JSONResponse:
    """Cancel a queued sweep or stop a running sweep at its next progress checkpoint."""
    job = session.get_sweep_job(job_id)
    if job is None:
        raise ApiError(404, "NOT_FOUND", f"No sweep with id {job_id}.")
    session.jobs.cancel(job.key)
    return _sweep_response(job_id, job, session)
