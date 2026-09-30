"""JSON API endpoints (mounted under ``/api``), one module per feature."""

from __future__ import annotations

from fastapi import APIRouter

from faissight.server.routes import compare, hnsw, info, ivf, pq, projection, search, sweep
from faissight.server.routes.common import ApiError

router = APIRouter()
for _module in (info, ivf, projection, search, sweep, compare, hnsw, pq):
    router.include_router(_module.router)

__all__ = ["ApiError", "router"]
