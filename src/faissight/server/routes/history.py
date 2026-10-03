"""Persistent local run history and comparison of saved sweeps."""

from dataclasses import asdict

from fastapi import APIRouter

from faissight.core.runs import compare_runs
from faissight.server import schemas as S
from faissight.server.routes.common import ApiError, SessionDep

router = APIRouter()


@router.get("/history", response_model=S.HistoryResponse)
def listing(session: SessionDep) -> S.HistoryResponse:
    return S.HistoryResponse(
        enabled=session.history.enabled,
        warning=session.history_warning,
        current_index=session.index_sha1 if session.history.enabled else None,
        runs=[S.HistorySummary.model_validate(r) for r in session.history.list()],
    )


@router.get("/history/{run_id}", response_model=S.HistoryRecord)
def get(run_id: str, session: SessionDep) -> S.HistoryRecord:
    try:
        return S.HistoryRecord.model_validate(session.history.get(run_id))
    except FileNotFoundError as e:
        raise ApiError(404, "NOT_FOUND", "This saved run is no longer available.") from e


@router.patch("/history/{run_id}", response_model=S.HistoryRecord)
def rename(run_id: str, req: S.HistoryRename, session: SessionDep) -> S.HistoryRecord:
    get(run_id, session)
    session.history.rename(run_id, req.label)
    return get(run_id, session)


@router.delete("/history/{run_id}")
def delete(run_id: str, session: SessionDep) -> dict[str, bool]:
    get(run_id, session)
    session.history.delete(run_id)
    return {"deleted": True}


@router.post("/history/compare", response_model=S.HistoryComparison)
def compare(req: S.HistoryCompareRequest, session: SessionDep) -> S.HistoryComparison:
    baseline, current = get(req.baseline, session), get(req.current, session)
    if baseline.kind != "sweep" or current.kind != "sweep":
        raise ValueError("Choose two sweep runs to compare.")
    result = compare_runs(baseline.data, current.data)
    return S.HistoryComparison(
        comparable=result.comparable,
        regressed=result.regressed,
        notes=result.notes,
        points=[
            S.PointDeltaOut(**asdict(p), recall_change=p.recall_change, p95_change=p.p95_change)
            for p in result.points
        ],
    )
