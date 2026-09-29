"""FastAPI app factory: JSON API under ``/api`` plus the built single-page frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from faissight import __version__
from faissight.core.embed import EmbedderUnavailableError
from faissight.core.ivf import NotAnIVFIndexError
from faissight.core.projection import ProjectionUnavailableError
from faissight.core.search import QueryError
from faissight.server.routes import ApiError, router
from faissight.session import InputError, Session

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
log = logging.getLogger("faissight")

_NOT_BUILT = """<!doctype html><html><head><meta charset="utf-8"><title>faissight</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>faissight</h1><p>The API is running at <a href="api/info">api/info</a>, but the web UI
has not been built.</p><p>From a source checkout run
<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>.</p></body></html>"""


def _error(status: int, code: str, message: str, hint: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error_code": code, "message": message, "hint": hint}
    )


def create_app(session: Session, static_dir: Path | None = None) -> FastAPI:
    """Build the app for one :class:`Session`. ``static_dir`` defaults to the bundled UI."""
    app = FastAPI(
        title="faissight",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.session = session
    app.include_router(router, prefix="/api")
    _install_error_handlers(app)
    _mount_frontend(app, (static_dir or STATIC_DIR).resolve())
    return app


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, e: ApiError) -> JSONResponse:
        return _error(e.status, e.code, e.message, e.hint)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError) -> JSONResponse:
        parts = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return _error(422, "VALIDATION_ERROR", "; ".join(parts), "Check the request fields.")

    simple = {
        QueryError: (400, "QUERY_ERROR", None),
        NotAnIVFIndexError: (400, "NOT_IVF", "This view needs an IVF index."),
    }
    for exc_type, (status, code, hint) in simple.items():

        async def handler(
            _: Request,
            e: Exception,
            status: int = status,
            code: str = code,
            hint: str | None = hint,
        ) -> JSONResponse:
            return _error(status, code, str(e), hint)

        app.add_exception_handler(exc_type, handler)

    @app.exception_handler(InputError)
    async def input_error(_: Request, e: InputError) -> JSONResponse:
        return _error(400, e.code, str(e), e.hint)

    @app.exception_handler(ProjectionUnavailableError)
    async def umap_missing(_: Request, e: ProjectionUnavailableError) -> JSONResponse:
        return _error(400, "UMAP_UNAVAILABLE", str(e), e.hint)

    @app.exception_handler(EmbedderUnavailableError)
    async def embedder_missing(_: Request, e: EmbedderUnavailableError) -> JSONResponse:
        return _error(400, "EMBEDDER_UNAVAILABLE", str(e), e.hint)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, e: ValueError) -> JSONResponse:
        return _error(400, "BAD_REQUEST", str(e))

    @app.exception_handler(Exception)
    async def internal_error(_: Request, e: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return _error(500, "INTERNAL_ERROR", f"{type(e).__name__}: {e}", "See the server log.")


def _mount_frontend(app: FastAPI, static_dir: Path) -> None:
    index_html = static_dir / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "NOT_FOUND", f"No API endpoint /{path}.")
        if not index_html.is_file():
            return HTMLResponse(_NOT_BUILT)
        target = (static_dir / path).resolve()
        # Serve real files (assets, favicon); everything else is a client-side route.
        if path and target.is_file() and target.is_relative_to(static_dir):
            return FileResponse(target)
        return FileResponse(index_html)
