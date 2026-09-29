# Decisions log

One line per non-obvious technical decision.

- Dev environment pinned to Python 3.12 (`.python-version`); `requires-python>=3.10`. System Python 3.14 may lack faiss-cpu wheels.
- `httpx` added to the `dev` extra: FastAPI's `TestClient` requires it.
- `__version__` is read from installed package metadata so `pyproject.toml` is the single source of truth.
- Frontend pinned to React 18 (per plan) although the Vite template defaults to 19.
- Vite `base: './'` (relative asset URLs) so the built UI works under path-prefixing proxies (Jupyter/Colab/HF Spaces).
- Tailwind v4 via `@tailwindcss/vite` plugin (no PostCSS/tailwind.config needed).
- `static/` is gitignored but force-included in wheel and sdist via hatch `artifacts`; the frontend must be built before `uv build`.
