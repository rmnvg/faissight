# faissight

A Python-first retrieval debugger for FAISS: a local web UI (FastAPI backend + React frontend) that shows index internals, explains missed neighbours against exact ground truth, and tunes recall vs latency. The full build plan is in `PLAN.md`; work through it phase by phase and stop after each phase for review.

## Repo layout

- `src/faissight/`: Python package
  - `cli.py`: Typer app (`faissight serve | info | sweep`)
  - `session.py`: in-memory state (index, vectors, metadata, caches)
  - `core/`: pure library (loader, ivf, hnsw, pq, search, hnsw_trace, sweep, projection, metadata, embed). **Must not import FastAPI or anything web-related.**
  - `server/`: FastAPI app factory, routes, pydantic schemas
  - `static/`: built frontend (gitignored; produced by `npm run build`, shipped in the wheel)
- `frontend/`: React + TypeScript + Vite + Tailwind; builds into `src/faissight/static`
- `examples/`: synthetic and RAG demo generators, notebook
- `tests/`: pytest; fixtures build small indexes in tmp dirs
- `docs/DECISIONS.md`: one line per non-obvious technical decision (keep it updated)

## Commands

```bash
uv sync --all-extras              # install everything (Python 3.12 pinned)
uv run pytest                     # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                       # strict on faissight.core
uv run faissight --version

# CI also runs Python 3.10 with older numpy (2.2), whose stubs are stricter; check it in a side env:
UV_PROJECT_ENVIRONMENT=/tmp/faissight-py310 UV_PYTHON=3.10 uv sync --all-extras --locked
UV_PROJECT_ENVIRONMENT=/tmp/faissight-py310 UV_PYTHON=3.10 uv run mypy

cd frontend && npm install && npm run build   # builds into src/faissight/static
uv build                          # wheel + sdist (build the frontend first)
```

## Conventions

- `src/` layout; type hints everywhere; docstrings on public functions.
- `faissight.core` is a plain library usable without the server.
- IDs exposed to the UI are always **user-facing ids** (after IDMap), never internal offsets.
- Always report the metric (`L2` or `IP`) alongside distances. For IP, larger = closer.
- Never return more than `max_points` (default 50,000) points to the frontend; sample deterministically (seeded), stratified by IVF list when applicable.
- Cache expensive work (projection, ground truth, sweep) on the Session; projections also on disk at `~/.cache/faissight/<index_sha1>/`.
- API errors are structured JSON: `{error_code, message, hint}`.
- FAISS is an optional extra (`faissight[faiss-cpu]`); import it lazily and raise a clear install hint on failure.
- If the installed FAISS API differs from the plan, trust FAISS, adapt, and leave a short comment.
- Don't add dependencies outside PLAN.md section 4 without asking.
- One git commit per completed step, conventional messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Never push; the maintainer pushes.
- Write tests alongside code; every public function in `faissight.core` gets a test.
