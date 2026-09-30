"""PQ/SQ reconstruction error and distance distortion (background job)."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from faissight.core import pq as PQ
from faissight.core.jobs import JobStatus
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SupportedDep,
    snippet,
)

router = APIRouter()


def _r(values: npt.NDArray[np.float32] | npt.NDArray[np.float64], digits: int = 6) -> list[float]:
    """Round to significant digits for compact JSON."""
    out: list[float] = [float(f"{v:.{digits}g}") for v in np.asarray(values, dtype=np.float64)]
    return out


@router.get(
    "/pq/error",
    response_model=S.PqErrorResponse,
    responses={202: {"model": S.JobStatusResponse}},
)
def pq_error(
    session: SupportedDep,
    retry: Annotated[bool, Query(description="Restart a failed analysis.")] = False,
) -> S.PqErrorResponse | JSONResponse:
    """Reconstruction error of the stored vectors (needs raw vectors)."""
    try:
        job = session.pq_job(retry=retry)
    except PQ.RawVectorsRequiredError as e:
        return S.PqErrorResponse(available=False, reason=str(e), hint=e.hint)
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise ApiError(
            500,
            "PQ_FAILED",
            f"Quantization analysis failed: {job.error}",
            "See the server log, then retry with retry=true.",
        )
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
                id=i, error=e, relative=r, list_no=list_of.get(i), snippet=snippet(session, i)
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
