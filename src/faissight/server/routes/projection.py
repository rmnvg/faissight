"""2-D/3-D projections of the stored vectors (background jobs)."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from faissight.core.jobs import JobStatus
from faissight.core.projection import Projection
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SupportedDep,
    round_coords,
)
from faissight.session import Session

router = APIRouter()


def ready_projection(session: Session, method: str, dims: int) -> Projection | None:
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
    retry: Annotated[bool, Query(description="Restart a failed projection.")] = False,
) -> S.ProjectionResponse | JSONResponse:
    if kind == "centroids" and not session.li.kind.is_ivf:
        raise ApiError(400, "NOT_IVF", "Only IVF indexes have centroids.")
    job = session.projection_job(method, dims, retry=retry)
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise ApiError(
            500,
            "PROJECTION_FAILED",
            f"Projection failed: {job.error}",
            "See the server log, then retry with retry=true.",
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
        x=round_coords(coords[:, 0]),
        y=round_coords(coords[:, 1]),
        z=round_coords(coords[:, 2]) if proj.dims == 3 else None,
        list_nos=list_nos,
        n_total=proj.n_total,
        sampled=proj.sampled,
        explained_variance=ev,
        cache_warning=proj.cache_warning,
    )
