# Decisions log

One line per non-obvious technical decision.

- Dev environment pinned to Python 3.12 (`.python-version`); `requires-python>=3.10`. System Python 3.14 may lack faiss-cpu wheels.
- `httpx` added to the `dev` extra: FastAPI's `TestClient` requires it.
- `__version__` is read from installed package metadata so `pyproject.toml` is the single source of truth.
