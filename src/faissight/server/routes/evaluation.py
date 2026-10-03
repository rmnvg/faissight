"""Labelled relevance evaluation jobs."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from faissight.server import schemas as S
from faissight.server.routes.common import ApiError, SessionDep, SupportedDep, job_response

router = APIRouter()


def _response(job_id: str, session: SessionDep) -> JSONResponse:
    job = session.jobs.get(("evaluation", job_id))
    if job is None:
        raise ApiError(404, "NOT_FOUND", "Evaluation is no longer on the server.")
    snapshot = job.as_dict()
    return job_response(
        S.EvaluationJobResponse(
            job_id=job_id,
            **snapshot,
            result=S.EvaluationResultOut.model_validate(job.result)
            if snapshot["status"] == "done"
            else None,
        )
    )


@router.post(
    "/evaluation",
    response_model=S.EvaluationJobResponse,
    responses={202: {"model": S.EvaluationJobResponse}},
)
def start(req: S.EvaluationRequest, session: SupportedDep) -> JSONResponse:
    job_id, _ = session.evaluation_job(**req.model_dump())
    return _response(job_id, session)


@router.get(
    "/evaluation/{job_id}",
    response_model=S.EvaluationJobResponse,
    responses={202: {"model": S.EvaluationJobResponse}},
)
def get(job_id: str, session: SessionDep) -> JSONResponse:
    return _response(job_id, session)


@router.delete("/evaluation/{job_id}", response_model=S.EvaluationJobResponse)
def cancel(job_id: str, session: SessionDep) -> JSONResponse:
    session.jobs.cancel(("evaluation", job_id))
    return _response(job_id, session)
