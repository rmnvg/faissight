"""Shared route dependencies and helpers."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from faissight.server import schemas as S
from faissight.session import Session

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


def snippet(session: Session, id: int) -> dict[str, str] | None:
    return session.metadata.snippet(id) if session.metadata is not None else None


def truth_source(session: Session) -> Literal["raw", "reconstructed"]:
    return "raw" if session.has_raw_vectors else "reconstructed"


def round_coords(a: npt.NDArray[np.float32]) -> list[float]:
    rounded: list[float] = np.round(a.astype(np.float64), COORD_DECIMALS).tolist()
    return rounded


def job_response(body: S.SweepJobResponse | S.CompareJobResponse) -> JSONResponse:
    """A background job's JSON: 202 while running, 200 once finished.

    ``mode="json"`` applies the large-id-as-string serializer to any ids in the result.
    """
    return JSONResponse(
        status_code=202 if body.status == "running" else 200, content=body.model_dump(mode="json")
    )
