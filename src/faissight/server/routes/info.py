"""Health, index info and metadata rows."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter

from faissight import __version__
from faissight.core.jobs import JobStatus
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    SessionDep,
    truth_source,
)
from faissight.server.routes.compare import candidate_out
from faissight.server.routes.sweep import sweep_defaults

router = APIRouter()


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
        ground_truth_source=truth_source(session) if li.is_supported else None,
        max_points=session.max_points,
        sweep=sweep_defaults(session),
        compare=[candidate_out(i, c) for i, c in enumerate(session.candidates)],
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
        memory=S.MemoryOut(**asdict(session.memory_estimate())),
    )


@router.get("/metadata/{id}", response_model=S.MetadataResponse)
def metadata(id: int, session: SessionDep) -> S.MetadataResponse:
    if session.metadata is None:
        raise ApiError(404, "NO_METADATA", "No metadata was loaded.", "Start with --meta.")
    row = session.metadata.get(id)
    if row is None:
        raise ApiError(404, "NOT_FOUND", f"No metadata for id {id}.")
    return S.MetadataResponse(id=id, row=row)
